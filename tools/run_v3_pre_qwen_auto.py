#!/usr/bin/env python3
"""Launch one elastic pre-Qwen Stage2 worker per visible local GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.pre_qwen_production import (
    DEFAULT_STAGE2_EXECUTION_CHUNK_ROWS,
    check_stage2_candidate_judge_health,
    enumerate_annotation_shards,
    inventory,
    load_config_identity,
    validate_stage2_preflight,
)
from tools.run_v3_pre_qwen_batch import DEFAULT_STAGE2_IDLE_EXIT_SECONDS


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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be positive")
    if args.idle_exit_seconds < 0:
        raise ValueError("--idle-exit-seconds must be non-negative")
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
    result = {
        **report,
        "prepare_seconds": max(0.0, worker_startup_started - prepare_started),
        "worker_startup_seconds": worker_startup_seconds,
        "worker_exit_codes": codes,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if any(codes):
        raise SystemExit(max(codes))
    return result


if __name__ == "__main__":
    main()
