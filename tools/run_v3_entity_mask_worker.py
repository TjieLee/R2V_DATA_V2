"""Run one persistent SAM3 worker over an assigned whole-shard range."""

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
    close_backend,
    default_backend_factory,
    enable_sam3_session_reuse,
    enumerate_annotation_shards,
    load_config_identity,
    process_shard,
    validate_execution_identity,
    validate_sam3_session_reuse_identity,
)
from tools.run_v3_entity_mask_auto import (
    PRODUCTION_CHUNK_ROWS,
    PRODUCTION_SESSION_REUSE_MODE,
    balanced_shard_bounds,
    expected_static_topology,
    validate_entity_mask_production_config,
    validate_static_topology,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--local-gpu-count", type=int, required=True)
    parser.add_argument("--local-slot", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--global-worker-id", type=int, required=True)
    parser.add_argument("--shard-start-position", type=int, required=True)
    parser.add_argument("--shard-end-position", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.world_size < 1 or args.local_gpu_count < 1:
        raise ValueError("world size and local GPU count must be positive")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("worker rank is outside WORLD_SIZE")
    if not 0 <= args.local_slot < args.local_gpu_count:
        raise ValueError("worker local slot is outside local GPU count")
    expected_worker_id = args.rank * args.local_gpu_count + args.local_slot
    if args.global_worker_id != expected_worker_id:
        raise ValueError("global worker id does not match rank and local slot")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.gpu:
        raise ValueError("worker CUDA_VISIBLE_DEVICES does not match assigned GPU")

    identity = load_config_identity(args.base_config)
    validate_entity_mask_production_config(identity.config)
    shard_paths = enumerate_annotation_shards(args.input_root)
    global_worker_count = args.world_size * args.local_gpu_count
    expected_start, expected_end = balanced_shard_bounds(
        len(shard_paths),
        global_worker_count,
        args.global_worker_id,
    )
    if (
        args.shard_start_position != expected_start
        or args.shard_end_position != expected_end
    ):
        raise ValueError("worker shard range does not match static assignment")

    topology = expected_static_topology(
        input_root=args.input_root,
        shard_paths=shard_paths,
        world_size=args.world_size,
        local_gpu_count=args.local_gpu_count,
    )
    validate_static_topology(args.output_root, topology)
    validate_execution_identity(
        args.output_root,
        chunk_rows=PRODUCTION_CHUNK_ROWS,
    )
    validate_sam3_session_reuse_identity(
        args.output_root,
        mode=PRODUCTION_SESSION_REUSE_MODE,
    )

    assigned = shard_paths[expected_start:expected_end]
    result: dict[str, object] = {
        "rank": args.rank,
        "world_size": args.world_size,
        "local_slot": args.local_slot,
        "gpu": args.gpu,
        "global_worker_id": args.global_worker_id,
        "global_worker_count": global_worker_count,
        "shard_start_position": expected_start,
        "shard_end_position": expected_end,
        "shard_count": len(assigned),
        "processed_shards": 0,
        "skipped_shards": 0,
        "retryable": False,
    }
    if not assigned:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result

    backend = default_backend_factory(identity.config)
    enable_sam3_session_reuse(
        backend,
        mode=PRODUCTION_SESSION_REUSE_MODE,
    )
    try:
        for shard_path in assigned:
            shard_result = process_shard(
                shard_path,
                output_root=args.output_root,
                config_identity=identity,
                backend=backend,
                acquire_lock=False,
                execution_identity_prevalidated=True,
                static_owner=True,
                chunk_rows=PRODUCTION_CHUNK_ROWS,
            )
            if shard_result.get("retryable") is True:
                result.update(
                    {
                        "retryable": True,
                        "retryable_shard": shard_path.name,
                        "retryable_stage": shard_result.get("stage"),
                    }
                )
                break
            result["processed_shards"] = int(result["processed_shards"]) + 1
            result["skipped_shards"] = int(result["skipped_shards"]) + int(
                shard_result.get("skipped") is True
            )
    finally:
        close_backend(backend)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    worker_result = main()
    if worker_result.get("retryable") is True:
        raise SystemExit(2)
