#!/usr/bin/env python3
"""Launch one elastic pre-Qwen Stage2 worker per visible local GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.pre_qwen_production import (
    inventory,
    load_config_identity,
    validate_qwen_free_preflight,
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    gpus = parse_gpus(args.gpus)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    preflight = validate_qwen_free_preflight(
        load_config_identity(args.base_config).config
    )
    report = {
        **inventory(args.input_root, args.output_root),
        **preflight,
        "detected_local_gpus": gpus,
        "rank": rank,
        "world_size": world_size,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return report
    tool = Path(__file__).with_name("run_v3_pre_qwen_batch.py")
    processes: list[subprocess.Popen[bytes]] = []
    codes: list[int] = []
    logs = []
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
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
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [process.wait() for process in processes]
    finally:
        for log in logs:
            log.close()
    result = {**report, "worker_exit_codes": codes}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if any(codes):
        raise SystemExit(max(codes))
    return result


if __name__ == "__main__":
    main()
