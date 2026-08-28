#!/usr/bin/env python3
"""Run or inspect restart-safe standalone pre-Qwen Visual Stage2 shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.pre_qwen_production import (
    DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS,
    AnnotationShard,
    ShardBusyError,
    build_execution_chunks,
    check_stage2_candidate_judge_health,
    close_backend,
    compact_execution_chunks,
    default_backend_factory,
    enable_sam3_request_timing,
    enable_sam3_session_reuse,
    ensure_execution_identity,
    ensure_sam3_session_reuse_identity,
    enumerate_annotation_shards,
    execution_chunk_path,
    inspect_stage2_shard,
    load_annotation_shard,
    load_config_identity,
    process_execution_chunk,
    process_shard,
    sam3_worker_session_reuse_diagnostics,
    sam3_worker_timing_diagnostics,
    validate_stage2_preflight,
)

DEFAULT_STAGE2_IDLE_EXIT_SECONDS = 60.0
STAGE2_IDLE_BACKOFF_SECONDS = 1.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shard", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--claim-loop", action="store_true")
    parser.add_argument("--scan-offset", type=int, default=0)
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS
    )
    parser.add_argument(
        "--idle-exit-seconds", type=float, default=DEFAULT_STAGE2_IDLE_EXIT_SECONDS
    )
    parser.add_argument("--sam3-request-timing", action="store_true")
    parser.add_argument(
        "--sam3-session-reuse-mode",
        choices=("off", "clip_reset_v1"),
        default="off",
    )
    parser.add_argument("--inspect", type=Path)
    return parser


def _rotated(paths: list[Path], offset: int) -> list[Path]:
    if not paths:
        return []
    start = offset % len(paths)
    return [*paths[start:], *paths[:start]]


def run_claim_loop(
    *,
    input_root: Path,
    base_config: Path,
    output_root: Path,
    scan_offset: int,
    chunk_rows: int = DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS,
    idle_exit_seconds: float = DEFAULT_STAGE2_IDLE_EXIT_SECONDS,
    sam3_request_timing: bool = False,
    sam3_session_reuse_mode: str = "off",
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    if idle_exit_seconds < 0:
        raise ValueError("--idle-exit-seconds must be non-negative")
    identity = load_config_identity(base_config)
    preflight = validate_stage2_preflight(identity.config)
    paths = enumerate_annotation_shards(input_root)
    ensure_sam3_session_reuse_identity(
        output_root,
        mode=sam3_session_reuse_mode,  # type: ignore[arg-type]
    )
    ensure_execution_identity(output_root, chunk_rows=chunk_rows)
    if all((output_root / "parts" / path.name).is_file() for path in paths):
        result = {
            **preflight,
            "results": [],
            "outstanding_or_busy": False,
            "chunks_claimed": 0,
            "chunks_completed": 0,
            "chunks_resumed": 0,
            "annotation_shards_loaded": 0,
            "worker_idle_scan_seconds": 0.0,
            "idle_rescans": 0,
            "idle_exit_seconds": idle_exit_seconds,
        }
        if sam3_request_timing:
            result.update(sam3_worker_timing_diagnostics(None, None))
        if sam3_session_reuse_mode != "off":
            result.update(
                sam3_worker_session_reuse_diagnostics(
                    None,
                    mode=sam3_session_reuse_mode,  # type: ignore[arg-type]
                )
            )
        return result
    service_health = check_stage2_candidate_judge_health(identity.config)
    backend_started = time.perf_counter()
    backend = default_backend_factory(identity.config)
    timing_collector = (
        enable_sam3_request_timing(backend) if sam3_request_timing else None
    )
    reuse_diagnostics = enable_sam3_session_reuse(
        backend,
        mode=sam3_session_reuse_mode,  # type: ignore[arg-type]
    )
    backend_closed = False

    def finalize_runtime_diagnostics(result: dict[str, object]) -> dict[str, object]:
        nonlocal backend_closed
        if sam3_session_reuse_mode != "off":
            backend_closed = True
            close_backend(backend)
        if sam3_request_timing:
            result.update(sam3_worker_timing_diagnostics(backend, timing_collector))
        if sam3_session_reuse_mode != "off":
            result.update(
                sam3_worker_session_reuse_diagnostics(reuse_diagnostics)
            )
        return result
    backend_startup_seconds = max(0.0, time.perf_counter() - backend_started)
    results: list[dict[str, object]] = []
    retryable_this_run: set[tuple[Path, str]] = set()
    shard_cache: dict[Path, AnnotationShard] = {}
    annotation_shard_load_seconds = 0.0
    chunks_claimed = chunks_completed = chunks_resumed = 0
    worker_idle_scan_seconds = 0.0
    idle_rescans = 0
    idle_started_at: float | None = None
    scan_cursor = scan_offset
    worker_started = time.perf_counter()
    try:
        while True:
            scan_started = time.perf_counter()
            claimed = False
            outstanding = False
            for path in _rotated(paths, scan_cursor):
                completed = output_root / "parts" / path.name
                if completed.is_file():
                    continue
                outstanding = True
                resolved = path.resolve(strict=True)
                shard = shard_cache.get(resolved)
                if shard is None:
                    load_started = time.perf_counter()
                    shard = load_annotation_shard(path)
                    annotation_shard_load_seconds += max(
                        0.0, time.perf_counter() - load_started
                    )
                    shard_cache[resolved] = shard
                chunks = build_execution_chunks(shard, chunk_rows=chunk_rows)
                for chunk in _rotated(list(chunks), scan_cursor):
                    chunk_final = execution_chunk_path(output_root, shard, chunk)
                    if chunk_final.is_file():
                        continue
                    retry_key = (resolved, chunk.stem)
                    if retry_key in retryable_this_run:
                        continue
                    try:
                        result = process_execution_chunk(
                            shard,
                            chunk,
                            output_root=output_root,
                            config_identity=identity,
                            backend=backend,
                            chunk_rows=chunk_rows,
                        )
                    except ShardBusyError:
                        continue
                    claimed = True
                    idle_started_at = None
                    chunks_claimed += 1
                    results.append(result)
                    if result.get("retryable") is True:
                        retryable_this_run.add(retry_key)
                    else:
                        chunks_completed += 1
                        chunks_resumed += int(result.get("resumed") is True)
                        try:
                            compacted = compact_execution_chunks(
                                shard,
                                output_root=output_root,
                                config_identity=identity,
                                chunk_rows=chunk_rows,
                            )
                        except ShardBusyError:
                            compacted = None
                        if compacted is not None:
                            results.append(compacted)
                    scan_cursor += 1
                    break
                if claimed:
                    break
                try:
                    compacted = compact_execution_chunks(
                        shard,
                        output_root=output_root,
                        config_identity=identity,
                        chunk_rows=chunk_rows,
                    )
                except ShardBusyError:
                    compacted = None
                if compacted is not None:
                    results.append(compacted)
                    claimed = True
                    idle_started_at = None
                    scan_cursor += 1
                    break
            if not claimed:
                worker_idle_scan_seconds += max(
                    0.0, time.perf_counter() - scan_started
                )
                if outstanding:
                    now = monotonic()
                    if idle_started_at is None:
                        idle_started_at = now
                    idle_elapsed = max(0.0, now - idle_started_at)
                    if idle_elapsed < idle_exit_seconds:
                        sleep_seconds = min(
                            STAGE2_IDLE_BACKOFF_SECONDS,
                            idle_exit_seconds - idle_elapsed,
                        )
                        before_sleep = monotonic()
                        sleep(sleep_seconds)
                        worker_idle_scan_seconds += max(
                            0.0, monotonic() - before_sleep
                        )
                        idle_rescans += 1
                        scan_cursor += 1
                        continue
                result = {
                    **preflight,
                    **service_health,
                    "results": results,
                    "outstanding_or_busy": outstanding,
                    "sam3_backend_startup_seconds": backend_startup_seconds,
                    "worker_active_seconds": max(
                        0.0, time.perf_counter() - worker_started
                    ),
                    "worker_idle_scan_seconds": worker_idle_scan_seconds,
                    "annotation_shard_load_seconds": annotation_shard_load_seconds,
                    "annotation_shards_loaded": len(shard_cache),
                    "chunks_claimed": chunks_claimed,
                    "chunks_completed": chunks_completed,
                    "chunks_resumed": chunks_resumed,
                    "idle_rescans": idle_rescans,
                    "idle_exit_seconds": idle_exit_seconds,
                }
                return finalize_runtime_diagnostics(result)
    finally:
        if not backend_closed:
            close_backend(backend)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.inspect is not None:
        result = inspect_stage2_shard(args.inspect)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if args.base_config is None or args.output_root is None:
        raise ValueError("--base-config and --output-root are required")
    if (args.input_shard is None) == (args.input_root is None):
        raise ValueError("provide exactly one of --input-shard or --input-root")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.claim_loop:
        if args.input_root is None:
            raise ValueError("--claim-loop requires --input-root")
        result = run_claim_loop(
            input_root=args.input_root,
            base_config=args.base_config,
            output_root=args.output_root,
            scan_offset=args.scan_offset,
            chunk_rows=args.chunk_rows,
            idle_exit_seconds=args.idle_exit_seconds,
            sam3_request_timing=args.sam3_request_timing,
            sam3_session_reuse_mode=args.sam3_session_reuse_mode,
        )
    else:
        if args.input_shard is None:
            raise ValueError("single batch mode requires --input-shard")
        identity = load_config_identity(args.base_config)
        preflight = validate_stage2_preflight(identity.config)
        ensure_sam3_session_reuse_identity(
            args.output_root,
            mode=args.sam3_session_reuse_mode,
        )
        completed = args.output_root / "parts" / args.input_shard.name
        service_health = (
            {"candidate_judge_health": "not_checked_completed"}
            if completed.is_file()
            else check_stage2_candidate_judge_health(identity.config)
        )
        backend = None if completed.is_file() else default_backend_factory(identity.config)
        timing_collector = (
            enable_sam3_request_timing(backend)
            if args.sam3_request_timing and backend is not None
            else None
        )
        reuse_diagnostics = (
            enable_sam3_session_reuse(
                backend,
                mode=args.sam3_session_reuse_mode,
            )
            if backend is not None
            else None
        )
        backend_closed = False
        try:
            result = {
                **preflight,
                **service_health,
                **process_shard(
                    args.input_shard,
                    output_root=args.output_root,
                    config_identity=identity,
                    backend=backend,
                    chunk_rows=args.chunk_rows,
                ),
            }
            if args.sam3_session_reuse_mode != "off" and backend is not None:
                backend_closed = True
                close_backend(backend)
            if args.sam3_request_timing:
                result.update(
                    sam3_worker_timing_diagnostics(backend, timing_collector)
                )
            if args.sam3_session_reuse_mode != "off":
                result.update(
                    sam3_worker_session_reuse_diagnostics(
                        reuse_diagnostics,
                        mode=args.sam3_session_reuse_mode,
                    )
                )
        finally:
            if backend is not None and not backend_closed:
                close_backend(backend)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
