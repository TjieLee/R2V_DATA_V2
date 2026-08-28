#!/usr/bin/env python3
"""Run or inspect restart-safe standalone pre-Qwen Visual Stage2 shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.pre_qwen_production import (
    ShardBusyError,
    close_backend,
    default_backend_factory,
    enumerate_annotation_shards,
    inspect_stage2_shard,
    load_config_identity,
    process_shard,
    validate_qwen_free_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-shard", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--claim-loop", action="store_true")
    parser.add_argument("--scan-offset", type=int, default=0)
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
) -> dict[str, object]:
    identity = load_config_identity(base_config)
    preflight = validate_qwen_free_preflight(identity.config)
    paths = _rotated(enumerate_annotation_shards(input_root), scan_offset)
    if all((output_root / "parts" / path.name).is_file() for path in paths):
        return {**preflight, "results": [], "outstanding_or_busy": False}
    backend = default_backend_factory(identity.config)
    results: list[dict[str, object]] = []
    retryable_this_run: set[Path] = set()
    try:
        while True:
            claimed = False
            outstanding = False
            for path in paths:
                completed = output_root / "parts" / path.name
                if completed.is_file():
                    continue
                outstanding = True
                if path in retryable_this_run:
                    continue
                try:
                    result = process_shard(
                        path,
                        output_root=output_root,
                        config_identity=identity,
                        backend=backend,
                    )
                except ShardBusyError:
                    continue
                claimed = True
                results.append(result)
                if result.get("retryable") is True:
                    retryable_this_run.add(path)
                paths = _rotated(paths, 1)
                break
            if not claimed:
                return {
                    **preflight,
                    "results": results,
                    "outstanding_or_busy": outstanding,
                }
    finally:
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
        )
    else:
        if args.input_shard is None:
            raise ValueError("single batch mode requires --input-shard")
        identity = load_config_identity(args.base_config)
        preflight = validate_qwen_free_preflight(identity.config)
        completed = args.output_root / "parts" / args.input_shard.name
        backend = None if completed.is_file() else default_backend_factory(identity.config)
        try:
            result = {
                **preflight,
                **process_shard(
                    args.input_shard,
                    output_root=args.output_root,
                    config_identity=identity,
                    backend=backend,
                ),
            }
        finally:
            if backend is not None:
                close_backend(backend)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
