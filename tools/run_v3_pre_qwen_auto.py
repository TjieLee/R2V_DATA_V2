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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
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
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return report
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
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    worker_startup_started = time.perf_counter()
    try:
        for local_slot, gpu in enumerate(gpus):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log = (log_root / f"rank-{rank}-gpu-{gpu}.log").open("ab")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
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
                    ],
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
    result = {
        **report,
        "prepare_seconds": max(0.0, worker_startup_started - prepare_started),
        "worker_startup_seconds": worker_startup_seconds,
        "worker_exit_codes": codes,
        **prefetch_diagnostics,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if any(codes):
        raise SystemExit(max(codes))
    return result


if __name__ == "__main__":
    main()
