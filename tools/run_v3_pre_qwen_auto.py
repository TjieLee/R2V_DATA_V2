#!/usr/bin/env python3
"""Launch one elastic pre-Qwen Stage2 worker per visible local GPU."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.pre_qwen_production import (
    DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS,
    check_stage2_candidate_judge_health,
    ensure_sam3_session_reuse_identity,
    enumerate_annotation_shards,
    initialize_frame_prefetch_worker,
    inventory,
    iter_frame_prefetch_tasks,
    load_config_identity,
    run_frame_prefetch_task,
    validate_stage2_preflight,
)
from tools.run_v3_pre_qwen_batch import DEFAULT_STAGE2_IDLE_EXIT_SECONDS


def _empty_prefetch_diagnostics(workers: int) -> dict[str, object]:
    return {
        "frame_prefetch_workers": workers,
        "frame_prefetch_submitted": 0,
        "frame_prefetch_completed": 0,
        "frame_prefetch_skipped_existing": 0,
        "frame_prefetch_failed": 0,
        "frame_prefetch_wall_seconds": 0.0,
    }


def _last_worker_result(path: Path, *, offset: int) -> dict[str, object] | None:
    with path.open("rb") as handle:
        handle.seek(offset)
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and (
            "sam3_request_timing" in value or "sam3_session_reuse_mode" in value
        ):
            return value
    return None


def _aggregate_sam3_timing(
    worker_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    categories = {
        "start_session": "handle_request.start_session",
        "reset_session": "handle_request.reset_session",
        "add_prompt": "handle_request.add_prompt",
        "propagate": "handle_stream_request.propagate_in_video",
        "close_session": "handle_request.close_session",
    }
    category_totals = {
        name: {"calls": 0, "total_seconds": 0.0}
        for name in categories
    }
    request_seconds = 0.0
    track_seconds = 0.0
    per_gpu: dict[str, object] = {}
    performance_per_gpu: dict[str, object] = {}
    anchor_totals: dict[str, int] = {}
    rescue_totals: dict[str, int] = {}

    for worker, value in sorted(worker_results.items()):
        request_timing = value.get("sam3_request_timing")
        performance = value.get("sam3_backend_performance_counters")
        anchor = value.get("sam3_anchor_search_counters")
        rescue = value.get("sam3_recall_rescue_counters")
        request_timing = request_timing if isinstance(request_timing, dict) else {}
        performance = performance if isinstance(performance, dict) else {}
        anchor = anchor if isinstance(anchor, dict) else {}
        rescue = rescue if isinstance(rescue, dict) else {}
        per_gpu[worker] = request_timing
        performance_per_gpu[worker] = performance

        for timing in request_timing.values():
            if not isinstance(timing, dict):
                continue
            seconds = timing.get("total_seconds")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                request_seconds += max(0.0, float(seconds))
        for name, category in categories.items():
            timing = request_timing.get(category)
            if not isinstance(timing, dict):
                continue
            calls = timing.get("calls")
            seconds = timing.get("total_seconds")
            if isinstance(calls, int) and not isinstance(calls, bool):
                category_totals[name]["calls"] += max(0, calls)
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                category_totals[name]["total_seconds"] += max(0.0, float(seconds))

        track = performance.get("sam3_segment_model_call_time_seconds")
        if isinstance(track, (int, float)) and not isinstance(track, bool):
            track_seconds += max(0.0, float(track))
        for source, target in ((anchor, anchor_totals), (rescue, rescue_totals)):
            for name, counter in source.items():
                if isinstance(counter, int) and not isinstance(counter, bool):
                    target[name] = target.get(name, 0) + max(0, counter)

    result: dict[str, object] = {
        "sam3_timing_per_gpu": per_gpu,
        "sam3_backend_performance_per_gpu": performance_per_gpu,
        "sam3_anchor_search_totals": anchor_totals,
        "sam3_recall_rescue_totals": rescue_totals,
        "sam3_timing_total_request_seconds": request_seconds,
        "sam3_timing_total_track_seconds": track_seconds,
        "sam3_timing_request_seconds_fraction_of_track": (
            request_seconds / track_seconds if track_seconds else 0.0
        ),
        "sam3_timing_unattributed_seconds": max(
            0.0, track_seconds - request_seconds
        ),
    }
    for name, totals in category_totals.items():
        result[f"sam3_timing_total_{name}_calls"] = totals["calls"]
        result[f"sam3_timing_total_{name}_seconds"] = totals["total_seconds"]
    return result


def _aggregate_sam3_session_reuse(
    worker_results: dict[str, dict[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    counter_names = (
        "sam3_logical_start_session_calls",
        "sam3_logical_close_session_calls",
        "sam3_physical_start_session_calls",
        "sam3_physical_reset_session_calls",
        "sam3_physical_close_session_calls",
        "sam3_reused_start_session_calls",
        "sam3_session_resource_switches",
    )
    totals = {name: 0 for name in counter_names}
    per_gpu: dict[str, dict[str, object]] = {}
    for worker, value in sorted(worker_results.items()):
        worker_values: dict[str, object] = {"sam3_session_reuse_mode": mode}
        for name in counter_names:
            counter = value.get(name)
            if isinstance(counter, int) and not isinstance(counter, bool):
                normalized = max(0, counter)
                totals[name] += normalized
                worker_values[name] = normalized
        per_gpu[worker] = worker_values
    return {
        "sam3_session_reuse_mode": mode,
        "sam3_session_reuse_per_gpu": per_gpu,
        **totals,
    }


@dataclass
class _FramePrefetchController:
    workers: int
    shard_paths: list[Path]
    base_config: Path
    output_root: Path
    diagnostics: dict[str, object] = field(init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.diagnostics = _empty_prefetch_diagnostics(self.workers)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("frame prefetch controller already started")
        self._thread = threading.Thread(
            target=self._run,
            name="stage2-frame-prefetch-controller",
            daemon=False,
        )
        self._thread.start()

    def stop_and_join(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return dict(self.diagnostics)

    def _append_diagnostic(self, value: dict[str, object]) -> None:
        path = self.output_root / "logs" / "frame-prefetch.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": time.time(), **value}
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record_future(self, future: Future[dict[str, object]]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - prefetch cannot fail Stage2
            result = {"status": "failed", "reason": str(exc)}
        status = result.get("status")
        if status == "completed":
            key = "frame_prefetch_completed"
        elif status == "skipped_existing":
            key = "frame_prefetch_skipped_existing"
        else:
            key = "frame_prefetch_failed"
        self.diagnostics[key] = int(self.diagnostics[key]) + 1
        self._append_diagnostic(result)

    def _run(self) -> None:
        started = time.perf_counter()
        pending: set[Future[dict[str, object]]] = set()
        executor: ProcessPoolExecutor | None = None
        try:
            executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_frame_prefetch_worker,
                initargs=(self.base_config, self.output_root),
            )
            tasks = iter_frame_prefetch_tasks(self.shard_paths)
            exhausted = False
            maximum_pending = max(1, self.workers * 2)
            while not self._stop.is_set():
                while (
                    not exhausted
                    and not self._stop.is_set()
                    and len(pending) < maximum_pending
                ):
                    try:
                        task = next(tasks)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(executor.submit(run_frame_prefetch_task, task))
                    self.diagnostics["frame_prefetch_submitted"] = (
                        int(self.diagnostics["frame_prefetch_submitted"]) + 1
                    )
                if not pending:
                    break
                done, _ = wait(
                    pending,
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    pending.remove(future)
                    self._record_future(future)
        except Exception as exc:  # noqa: BLE001 - prefetch cannot fail Stage2
            self.diagnostics["frame_prefetch_failed"] = (
                int(self.diagnostics["frame_prefetch_failed"]) + 1
            )
            self._append_diagnostic(
                {"status": "failed", "reason": f"prefetch controller: {exc}"}
            )
        finally:
            for future in pending:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            for future in pending:
                if future.done() and not future.cancelled():
                    self._record_future(future)
            self.diagnostics["frame_prefetch_wall_seconds"] = max(
                0.0, time.perf_counter() - started
            )


def parse_gpus(value: str | None) -> list[str]:
    if value is not None:
        gpus = [item.strip() for item in value.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip():
            gpus = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("detected GPU list must be non-empty and unique")
    return gpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus")
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS
    )
    parser.add_argument(
        "--idle-exit-seconds", type=float, default=DEFAULT_STAGE2_IDLE_EXIT_SECONDS
    )
    parser.add_argument("--frame-prefetch-workers", type=int, default=0)
    parser.add_argument("--sam3-request-timing", action="store_true")
    parser.add_argument(
        "--sam3-session-reuse-mode",
        choices=("off", "clip_reset_v1"),
        default="off",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    worker_log_root: Path | None = None,
) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    if args.idle_exit_seconds < 0:
        raise ValueError("--idle-exit-seconds must be non-negative")
    if args.frame_prefetch_workers < 0:
        raise ValueError("--frame-prefetch-workers must be non-negative")
    prepare_started = time.perf_counter()
    gpus = parse_gpus(args.gpus)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config = load_config_identity(args.base_config).config
    preflight = validate_stage2_preflight(config)
    shard_paths = enumerate_annotation_shards(args.input_root)
    report: dict[str, object] = {
        **preflight,
        "stage1_completed_shards": len(shard_paths),
        "execution_chunk_rows": args.chunk_rows,
        "idle_exit_seconds": args.idle_exit_seconds,
        **_empty_prefetch_diagnostics(args.frame_prefetch_workers),
        "output_root": str(args.output_root.resolve(strict=False)),
        "detected_local_gpus": gpus,
        "rank": rank,
        "world_size": world_size,
    }
    if args.dry_run:
        report.update(inventory(args.input_root, args.output_root))
        report["prepare_seconds"] = max(0.0, time.perf_counter() - prepare_started)
        if args.sam3_request_timing:
            report.update(_aggregate_sam3_timing({}))
        if args.sam3_session_reuse_mode != "off":
            report.update(
                _aggregate_sam3_session_reuse(
                    {},
                    mode=args.sam3_session_reuse_mode,
                )
            )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return report
    ensure_sam3_session_reuse_identity(
        args.output_root,
        mode=args.sam3_session_reuse_mode,
    )
    report.update(check_stage2_candidate_judge_health(config))
    prefetch: _FramePrefetchController | None = None
    prefetch_diagnostics = _empty_prefetch_diagnostics(
        args.frame_prefetch_workers
    )
    if args.frame_prefetch_workers:
        prefetch = _FramePrefetchController(
            workers=args.frame_prefetch_workers,
            shard_paths=shard_paths,
            base_config=args.base_config,
            output_root=args.output_root,
        )
        prefetch.start()
    tool = Path(__file__).with_name("run_v3_pre_qwen_batch.py")
    processes: list[subprocess.Popen[bytes]] = []
    codes: list[int] = []
    logs = []
    worker_logs: list[tuple[str, Path, int]] = []
    log_root = worker_log_root or args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    worker_startup_started = time.perf_counter()
    try:
        for local_slot, gpu in enumerate(gpus):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log_path = log_root / f"rank-{rank}-gpu-{gpu}.log"
            log = log_path.open("ab")
            log.seek(0, os.SEEK_END)
            worker_logs.append((f"rank{rank}_gpu{gpu}", log_path, log.tell()))
            logs.append(log)
            command = [
                sys.executable,
                str(tool),
                "--input-root",
                str(args.input_root),
                "--base-config",
                str(args.base_config),
                "--output-root",
                str(args.output_root),
                "--gpu",
                gpu,
                "--claim-loop",
                "--scan-offset",
                str(rank * len(gpus) + local_slot),
                "--chunk-rows",
                str(args.chunk_rows),
                "--idle-exit-seconds",
                str(args.idle_exit_seconds),
            ]
            if args.sam3_request_timing:
                command.append("--sam3-request-timing")
            if args.sam3_session_reuse_mode != "off":
                command.extend(
                    [
                        "--sam3-session-reuse-mode",
                        args.sam3_session_reuse_mode,
                    ]
                )
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        worker_startup_seconds = max(
            0.0, time.perf_counter() - worker_startup_started
        )
        codes = [process.wait() for process in processes]
    finally:
        for log in logs:
            log.close()
        if prefetch is not None:
            prefetch_diagnostics = prefetch.stop_and_join()
    runtime_diagnostics: dict[str, object] = {}
    if args.sam3_request_timing or args.sam3_session_reuse_mode != "off":
        worker_results = {
            worker: value
            for worker, path, offset in worker_logs
            if (value := _last_worker_result(path, offset=offset)) is not None
        }
        if args.sam3_request_timing:
            runtime_diagnostics.update(_aggregate_sam3_timing(worker_results))
        if args.sam3_session_reuse_mode != "off":
            runtime_diagnostics.update(
                _aggregate_sam3_session_reuse(
                    worker_results,
                    mode=args.sam3_session_reuse_mode,
                )
            )
    result = {
        **report,
        "prepare_seconds": max(0.0, worker_startup_started - prepare_started),
        "worker_startup_seconds": worker_startup_seconds,
        "worker_exit_codes": codes,
        **prefetch_diagnostics,
        **runtime_diagnostics,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if any(codes):
        raise SystemExit(max(codes))
    return result


if __name__ == "__main__":
    main()
