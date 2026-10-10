"""Explicitly fake CPU workers; no Audio, model, or training products are made."""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import signal
import threading
import time
import uuid
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_production import (
    STAGES,
    GroupCoordinator,
    StageDraining,
    StaleGeneration,
    read_json,
)

OUTPUT_ROOT = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA")
SOURCE_ROOT = OUTPUT_ROOT.parent / "in_pair_reference/post_mask"


def pilot_run_root(output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("run-id must be a simple directory name")
    root = (output_root / run_id).resolve()
    public = Path("/mnt/workspace/public")
    if root.is_relative_to(public) and not root.is_relative_to(OUTPUT_ROOT):
        raise ValueError("public output is authorized only below R2VA")
    return root


class FakeWorker:
    def __init__(self, stage: str):
        self.stage = stage

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def process(self, task):
        return {"fake": True, "stage": self.stage, "sample_id": task.sample_id,
                "model_call_count": 0, "picture_count": len(task.pictures)}


def _budget_finished(coordinator):
    control = coordinator.status()
    return coordinator.budget_finished(control)


def run_fake_worker(coordinator, worker_id, stop_event, *, poll_seconds=0.05):
    while not stop_event.is_set():
        snapshot = coordinator.select_current()
        if snapshot is None:
            if _budget_finished(coordinator):
                return
            stop_event.wait(poll_seconds)
            continue
        try:
            if snapshot.phase == "draining":
                coordinator.advance(snapshot)
                stop_event.wait(poll_seconds)
                continue
            with coordinator.join(snapshot, worker_id) as session:
                with FakeWorker(snapshot.stage) as backend:
                    while not stop_event.is_set():
                        claim = coordinator.claim(snapshot, worker_id)
                        if claim is None:
                            stats = coordinator.progress(snapshot)
                            if stats["pending"] == stats["active"] == 0:
                                coordinator.advance(snapshot)
                                break
                            stop_event.wait(poll_seconds)
                            continue
                        with claim:
                            started = time.monotonic()
                            index = STAGES.index(snapshot.stage)
                            upstream = None
                            if index:
                                previous = snapshot.__class__(snapshot.group_id, STAGES[index - 1],
                                                              snapshot.generation - 1, "complete")
                                upstream = read_json(coordinator.result_path(previous, claim.task.ordinal))
                            if upstream and upstream["status"] != "ready":
                                status, payload = "skipped", {"fake": True, "reason": "upstream_terminal",
                                                              "upstream_status": upstream["status"]}
                            else:
                                status, payload = "ready", backend.process(claim.task)
                            coordinator.publish(claim, status, payload, time.monotonic() - started)
                coordinator.acknowledge_release(session)
                coordinator.advance(snapshot)
        except (StaleGeneration, StageDraining):
            continue


def pilot_summary(run_root: Path) -> dict:
    control = read_json(run_root / "control.json")
    stages = []
    completed = 0
    for group in control["started_groups"]:
        group_root = run_root / "groups" / group
        completed += (group_root / "PILOT_COMPLETE").is_file() or (group_root / "COMPLETE").is_file()
        metadata = read_json(group_root / "inventory/inventory.json")
        for path in sorted((group_root / "stages").glob("*/dispatch.json")):
            dispatch = read_json(path)
            done = sum(dispatch[k] for k in ("ready", "failed", "skipped"))
            elapsed = max(dispatch.get("finished_at", time.time()) - dispatch["started_at"], 0.000001)
            stages.append({"group": group, "stage": path.parent.name.split("-", 1)[1],
                           "generation": dispatch["generation"],
                           "pending": metadata["selected_count"] - done - len(dispatch["active"]),
                           "active": len(dispatch["active"]),
                           **{k: dispatch[k] for k in ("ready", "failed", "skipped")},
                           "elapsed_seconds": elapsed, "tasks_per_second": done / elapsed})
    result = {"mode": control["mode"], "settings": control["settings"],
            "completed_groups": completed, "started_groups": control["started_groups"],
            "model_call_count": 0, "stages": stages}
    if control["mode"].startswith("http_native"):
        calls = sum(read_json(path)["payload"].get("model_call_count", 0)
                    for path in run_root.glob("groups/*/stages/*-mimo/results/*.json"))
        result["model_call_count"] = calls if control["mode"] in ("http_native_pilot", "http_native_production") else 0
        result["simulated_mimo_request_count"] = calls if control["mode"] == "http_native_cpu_pilot" else 0
        result["exports"] = [read_json(path) for path in run_root.glob("groups/*/bundle/summary.json")]
    return result


def run_fake_pilot(*, source_root: Path, run_root: Path, workers: int, limit: int,
                   max_groups: int, poll_seconds: float, start_row=0) -> dict:
    if workers < 1 or poll_seconds <= 0:
        raise ValueError("workers and poll_seconds must be positive")
    coordinator = GroupCoordinator(run_root, source_root, limit, max_groups=max_groups, start_row=start_row)
    context = mp.get_context("spawn")
    stop = context.Event()
    identifier = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    children = [context.Process(target=run_fake_worker,
                args=(coordinator, f"{identifier}-{i}", stop),
                kwargs={"poll_seconds": poll_seconds}) for i in range(workers)]
    handlers = {}

    def request_stop(signum, frame):
        stop.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
    started = time.monotonic()
    try:
        for child in children:
            child.start()
        while any(child.is_alive() for child in children) and not stop.is_set():
            for child in children:
                child.join(0.05)
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("Fake Worker interrupted; resume the same run-id")
        if any(child.exitcode not in (None, 0) for child in children):
            raise RuntimeError("Fake Worker interrupted; resume the same run-id")
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
    result = pilot_summary(run_root)
    result["invocation_elapsed_seconds"] = time.monotonic() - started
    return result
