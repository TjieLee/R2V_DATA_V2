#!/usr/bin/env python3
"""Prepare or run an isolated fixed-quota pre-Qwen Visual Stage2 canary."""

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

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.pre_qwen_production import (
    CanaryManifest,
    CanarySelectionRow,
    canary_summary,
    close_backend,
    default_backend_factory,
    load_config_identity,
    prepare_canary,
    run_canary_worker,
    validate_qwen_free_preflight,
)


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU identifiers")
    return gpus


def _load_identity_files(
    output_root: Path,
) -> tuple[CanaryManifest, list[CanarySelectionRow]]:
    manifest = CanaryManifest.model_validate_json(
        (output_root / "canary_manifest.json").read_text(encoding="utf-8")
    )
    selection = [
        CanarySelectionRow.model_validate(json.loads(line))
        for line in (output_root / "selection.jsonl").read_bytes().splitlines()
        if line
    ]
    return manifest, selection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus")
    parser.add_argument("--samples-per-gpu", type=int, default=10)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker-slot", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.worker_slot is not None:
        manifest, selection = _load_identity_files(args.output_root)
        if not 0 <= args.worker_slot < len(manifest.gpus):
            raise ValueError("canary worker slot is outside configured GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = manifest.gpus[args.worker_slot]
        identity = load_config_identity(manifest.base_config_path)
        validate_qwen_free_preflight(identity.config)
        completed_path = args.output_root / "workers" / f"gpu-{args.worker_slot}.jsonl"
        backend = None if completed_path.is_file() else default_backend_factory(identity.config)
        try:
            result = run_canary_worker(
                output_root=args.output_root,
                manifest=manifest,
                selection=selection,
                gpu_slot=args.worker_slot,
                backend=backend,
            )
        finally:
            if backend is not None:
                close_backend(backend)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if args.input_root is None or args.base_config is None or args.gpus is None:
        raise ValueError("--input-root, --base-config, and --gpus are required")
    if args.samples_per_gpu < 1:
        raise ValueError("--samples-per-gpu must be positive")
    gpus = _parse_gpus(args.gpus)
    manifest, selection = prepare_canary(
        input_root=args.input_root,
        base_config=args.base_config,
        output_root=args.output_root,
        gpus=gpus,
        samples_per_gpu=args.samples_per_gpu,
    )
    prepared = {
        "status": "prepared",
        "selected_samples": manifest.selected_samples,
        "per_gpu": {
            f"gpu{slot}": sum(item.gpu_slot == slot for item in selection)
            for slot in range(len(gpus))
        },
        "qwen_required": False,
        "qwen_calls": 0,
        "output_root": str(args.output_root.resolve(strict=False)),
    }
    if args.prepare_only:
        print(json.dumps(prepared, ensure_ascii=False, sort_keys=True))
        return prepared
    started = time.perf_counter()
    processes: list[subprocess.Popen[bytes]] = []
    codes: list[int] = []
    logs = []
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        for slot, gpu in enumerate(gpus):
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            log = (log_root / f"gpu-{slot}.log").open("ab")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output-root",
                        str(args.output_root),
                        "--worker-slot",
                        str(slot),
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
    summary = canary_summary(args.output_root, manifest)
    summary["elapsed_seconds"] = max(0.0, time.perf_counter() - started)
    summary["worker_exit_codes"] = codes
    write_json_atomic(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if any(codes):
        raise SystemExit(max(codes))
    return summary


if __name__ == "__main__":
    main()
