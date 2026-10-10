"""Bounded local CPU integration pilot; not a multi-node/GPU launcher."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_cpu_fixtures import CPUFixtureFactory
from r2v_data_v2.h3.ra2va_group_export import _write_rows
from r2v_data_v2.h3.ra2va_group_fake import _budget_finished, pilot_summary
from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
from r2v_data_v2.h3.ra2va_group_production import (
    STAGES,
    GroupCoordinator,
    StageDraining,
    StageSnapshot,
    StaleGeneration,
    read_json,
)
from r2v_data_v2.h3.t2va_production import atomic_json
from r2v_data_v2.h3.training_manifest_export import (
    RA2VA_TASK_ORDER,
    write_training_manifests,
)


def _upstream(coordinator, snapshot, ordinal):
    index = STAGES.index(snapshot.stage)
    rows = {stage: read_json(coordinator.result_path(StageSnapshot(snapshot.group_id, stage,
        snapshot.generation - index + i, "complete"), ordinal)) for i, stage in enumerate(STAGES[:index])}
    return rows


def _worker(coordinator, worker_id, stop, fixtures_path, ffmpeg):
    factory = CPUFixtureFactory(fixtures_path, ffmpeg)
    while not stop.is_set():
        snapshot = coordinator.select_current()
        if snapshot is None:
            if _budget_finished(coordinator):
                return
            stop.wait(.05)
            continue
        try:
            if snapshot.phase == "draining":
                coordinator.advance(snapshot)
                stop.wait(.05)
                continue
            with coordinator.join(snapshot, worker_id) as session:
                with GroupStageWorker(snapshot.stage, coordinator.group_root(snapshot.group_id) / "artifacts",
                                      backend_factory=factory, ffmpeg=ffmpeg) as backend:
                    while not stop.is_set():
                        claim = coordinator.claim(snapshot, worker_id)
                        if claim is None:
                            stats = coordinator.progress(snapshot)
                            if stats["pending"] == stats["active"] == 0:
                                coordinator.advance(snapshot)
                                break
                            stop.wait(.05)
                            continue
                        with claim:
                            started = time.monotonic()
                            rows = _upstream(coordinator, snapshot, claim.task.ordinal)
                            if any(r["status"] != "ready" for r in rows.values()):
                                status, payload = "skipped", {"reason": "upstream_terminal"}
                            else:
                                try:
                                    payload = backend.process(claim.task, {k: r["payload"] for k, r in rows.items()})
                                    status = payload.get("status", "ready")
                                    if payload.get("product_failed_count", 0):
                                        status = "failed"
                                except (ValueError, OSError) as exc:
                                    status, payload = "failed", {"failure_reason": f"{type(exc).__name__}: {exc}"}
                            coordinator.publish(claim, status, {**payload, "execution": "cpu_fixture"},
                                                time.monotonic() - started)
                coordinator.acknowledge_release(session)
                coordinator.advance(snapshot)
        except (StaleGeneration, StageDraining):
            continue


def _collect_export(group_root, snapshot=None):
    scope = None
    if snapshot is not None:
        scope = read_json(group_root / "stages" / f"{snapshot.generation:06d}-export/dispatch.json")
    partitioned = scope is not None and "part_index" in scope
    output_root = (group_root / "parts" / f"part-{scope['part_index']:03d}"
                   if partitioned else group_root)
    summary_path = output_root / "bundle/summary.json"
    if summary_path.exists():
        return read_json(summary_path)
    output = summary_path.parent

    def results(stage):
        if partitioned:
            generation = snapshot.generation - (len(STAGES) - 1) + STAGES.index(stage)
            return sorted((group_root / "stages" / f"{generation:06d}-{stage}/results").glob("*.json"))
        return sorted((group_root / "stages").glob(f"*-{stage}/results/*.json"))

    roots = results("export")
    tasks, donors, summaries = {key: [] for key in RA2VA_TASK_ORDER}, [], []
    for path in roots:
        result = read_json(path)
        if "bundle_root" not in result["payload"]:
            continue
        summary = result["payload"]
        summaries.append(summary)
        clip_root = Path(summary["bundle_root"])
        for key in RA2VA_TASK_ORDER:
            tasks[key].extend(read_json_lines(clip_root / "training" / f"{key}.jsonl"))
        for donor in read_json(clip_root / "voice_donor_reserve.json")["donors"]:
            for segment in donor["segments"]:
                segment["relative_path"] = os.path.relpath(segment["audio_path"], output)
            donors.append(donor)
    output.mkdir(parents=True, exist_ok=True)
    jobs, records = [], []
    for path in results("asr"):
        result = read_json(path)
        if "job" in result["payload"]:
            jobs.append(result["payload"]["job"])
    for path in results("mimo"):
        records.append(read_json(path)["payload"])
    atomic_json(output / "source_contract.json", {"schema_version": "r2v.h3.group_source_contract.1", "jobs": jobs})
    _write_rows(output / "records.jsonl", records)
    if not (output / "training").exists():
        with tempfile.TemporaryDirectory(prefix=".training-", dir=output) as temporary:
            stage = Path(temporary) / "training"
            write_training_manifests(output_root=stage, rows=tasks, task_order=RA2VA_TASK_ORDER)
            stage.rename(output / "training")
    if not (output / "voice_donor_reserve.json").exists():
        atomic_json(output / "voice_donor_reserve.json", {"schema_version": "r2v.h3.voice_donor_reserve.1", "donors": donors})
    summary = {"group": group_root.name, "bundle_root": str(output),
        "product_ready_count": sum(s["product_ready_count"] for s in summaries),
        "product_failed_count": sum(s["product_failed_count"] for s in summaries),
        "task_counts": {key: len(value) for key, value in tasks.items()}, "donor_count": len(donors),
        "donor_segment_count": sum(d["segment_count"] for d in donors)}
    if partitioned:
        summary.update({k: scope[k] for k in (
            "part_index", "part_count", "start_ordinal", "end_ordinal", "selected_count")})
    atomic_json(summary_path, summary)
    return summary


def read_json_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_cpu_pipeline(*, source_root, run_root, fixtures_path, workers=2, limit=20, max_groups=1,
                     ffmpeg="ffmpeg"):
    if workers < 1:
        raise ValueError("workers must be positive")
    coordinator = GroupCoordinator(Path(run_root), Path(source_root), limit, max_groups=max_groups,
                                   mode="cpu_pipeline_pilot")
    context = mp.get_context("spawn")
    stop = context.Event()
    children = [context.Process(target=_worker, args=(coordinator, f"cpu-{os.getpid()}-{i}", stop,
                                                     fixtures_path, ffmpeg)) for i in range(workers)]
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *args: stop.set())
    started = time.monotonic()
    try:
        for child in children:
            child.start()
        while any(child.is_alive() for child in children) and not stop.is_set():
            for child in children:
                child.join(.05)
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("CPU integration Worker interrupted; resume the same run-id")
    finally:
        stop.set()
        for child in children:
            if child.pid is None:
                continue
            child.join(2)
            if child.is_alive():
                child.terminate()
                child.join(2)
            if child.is_alive():
                child.kill()
                child.join()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if any(child.exitcode not in (None, 0) for child in children):
        raise RuntimeError("CPU integration Worker interrupted; resume the same run-id")
    summary = pilot_summary(Path(run_root))
    summary["mode"] = "cpu_pipeline_pilot"
    summary["simulated_mimo_request_count"] = sum(
        read_json(path)["payload"].get("model_call_count", 0)
        for path in Path(run_root).glob("groups/*/stages/*-mimo/results/*.json"))
    summary["exports"] = [_collect_export(coordinator.group_root(group)) for group in summary["started_groups"]
                          if (coordinator.group_root(group) / "PILOT_COMPLETE").exists()]
    summary["invocation_elapsed_seconds"] = time.monotonic() - started
    atomic_json(Path(run_root) / "pipeline_summary.json", summary)
    return summary
