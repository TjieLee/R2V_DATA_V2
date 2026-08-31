"""Launch fixed whole-shard Entity Mask Stage2 production workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pre_qwen_production import (
    _safe_output_root,
    check_stage2_candidate_judge_health,
    ensure_execution_identity,
    ensure_sam3_session_reuse_identity,
    enumerate_annotation_shards,
    load_config_identity,
    validate_execution_identity,
    validate_sam3_session_reuse_identity,
    validate_stage2_preflight,
)
from tools.run_v3_pre_qwen_auto import parse_gpus

PRODUCTION_INPUT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_annotations"
)
PRODUCTION_OUTPUT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_mask"
)
PRODUCTION_BASE_CONFIG = Path(
    "/mnt/workspace/litengjie/data/entity_mask_configs/production.yaml"
)
PRODUCTION_LOG_ROOT = Path("/mnt/workspace/litengjie/data/entity_mask_logs")
PRODUCTION_CANDIDATE_JUDGE_BASE_URL = "http://6.167.57.88:8000/v1"
PRODUCTION_CANDIDATE_JUDGE_MODEL = (
    "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
)
PRODUCTION_CHUNK_ROWS = 100
PRODUCTION_SESSION_REUSE_MODE = "clip_reset_v1"
STATIC_ASSIGNMENT_SCHEMA_VERSION = 1
STATIC_ASSIGNMENT_STRATEGY = "static_whole_shard_v1"
STARTUP_IDENTITY_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class StaticShardAssignment:
    local_slot: int
    gpu: str
    global_worker_id: int
    shard_start_position: int
    shard_end_position: int

    @property
    def shard_count(self) -> int:
        return self.shard_end_position - self.shard_start_position


def balanced_shard_bounds(
    num_shards: int,
    num_workers: int,
    worker_id: int,
) -> tuple[int, int]:
    if num_shards < 0:
        raise ValueError("num_shards must be non-negative")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must be within global worker count")
    base, remainder = divmod(num_shards, num_workers)
    start = worker_id * base + min(worker_id, remainder)
    count = base + int(worker_id < remainder)
    return start, start + count


def build_local_assignments(
    shard_paths: Sequence[Path],
    *,
    rank: int,
    world_size: int,
    gpus: Sequence[str],
) -> tuple[StaticShardAssignment, ...]:
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("local GPU list must be non-empty and unique")
    local_gpu_count = len(gpus)
    global_worker_count = world_size * local_gpu_count
    assignments: list[StaticShardAssignment] = []
    for local_slot, gpu in enumerate(gpus):
        global_worker_id = rank * local_gpu_count + local_slot
        start, end = balanced_shard_bounds(
            len(shard_paths),
            global_worker_count,
            global_worker_id,
        )
        assignments.append(
            StaticShardAssignment(
                local_slot=local_slot,
                gpu=gpu,
                global_worker_id=global_worker_id,
                shard_start_position=start,
                shard_end_position=end,
            )
        )
    return tuple(assignments)


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"entity-mask production requires {name}={expected!r}; got {actual!r}"
        )


def validate_entity_mask_production_config(config: V3Config) -> None:
    validate_stage2_preflight(config)
    service = config.qwen.candidate_judge
    _require_equal("sam3.save_debug_overlays", config.sam3.save_debug_overlays, False)
    _require_equal("debug.save_diagnostics", config.debug.save_diagnostics, False)
    _require_equal(
        "sam3.object_rescue_mode",
        config.sam3.object_rescue_mode,
        "phrase_retry_v1",
    )
    _require_equal(
        "sam3.not_found_rescue_mode",
        config.sam3.not_found_rescue_mode,
        "entity_phrase_retry_v1",
    )
    _require_equal(
        "sam3.multi_instance_rescue_mode",
        config.sam3.multi_instance_rescue_mode,
        "qwen_anchor_select_v1",
    )
    _require_equal(
        "sam3.anchor_search_mode",
        config.sam3.anchor_search_mode,
        "progressive_v1",
    )
    _require_equal(
        "qwen.candidate_judge.base_url",
        service.base_url,
        PRODUCTION_CANDIDATE_JUDGE_BASE_URL,
    )
    _require_equal(
        "qwen.candidate_judge.model",
        service.model,
        PRODUCTION_CANDIDATE_JUDGE_MODEL,
    )
    _require_equal("qwen.candidate_judge.temperature", service.temperature, 0.0)
    _require_equal("qwen.candidate_judge.max_tokens", service.max_tokens, 1024)
    _require_equal(
        "qwen.candidate_judge.timeout_seconds",
        service.timeout_seconds,
        3600,
    )


def _shard_names_sha256(shard_paths: Sequence[Path]) -> str:
    payload = "\n".join(path.name for path in shard_paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_static_topology(
    *,
    input_root: Path,
    shard_paths: Sequence[Path],
    world_size: int,
    local_gpu_count: int,
) -> dict[str, object]:
    return {
        "schema_version": STATIC_ASSIGNMENT_SCHEMA_VERSION,
        "strategy": STATIC_ASSIGNMENT_STRATEGY,
        "input_root": str(input_root.expanduser().resolve(strict=False)),
        "world_size": world_size,
        "local_gpu_count": local_gpu_count,
        "global_worker_count": world_size * local_gpu_count,
        "stage1_completed_shards": len(shard_paths),
        "shard_names_sha256": _shard_names_sha256(shard_paths),
        "chunk_rows": PRODUCTION_CHUNK_ROWS,
        "sam3_session_reuse_mode": PRODUCTION_SESSION_REUSE_MODE,
    }


def _static_topology_path(output_root: Path) -> Path:
    return output_root / "_internal" / "entity-mask-static-assignment.json"


def validate_static_topology(
    output_root: Path,
    expected: dict[str, object],
) -> Path:
    path = _static_topology_path(output_root)
    if not path.is_file():
        raise FileNotFoundError(f"missing Entity Mask static topology: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Entity Mask static topology must contain an object")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Entity Mask static topology mismatch: {key}")
    return path


def coordinate_startup_identity(
    *,
    output_root: Path,
    rank: int,
    expected_topology: dict[str, object],
    timeout_seconds: float = STARTUP_IDENTITY_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    root = _safe_output_root(output_root)
    topology_path = _static_topology_path(root)
    if rank == 0:
        ensure_execution_identity(root, chunk_rows=PRODUCTION_CHUNK_ROWS)
        ensure_sam3_session_reuse_identity(
            root,
            mode=PRODUCTION_SESSION_REUSE_MODE,
        )
        if topology_path.is_file():
            validate_static_topology(root, expected_topology)
        else:
            write_json_atomic(topology_path, expected_topology)
            validate_static_topology(root, expected_topology)
        return
    if timeout_seconds < 0:
        raise ValueError("startup identity timeout must be non-negative")
    deadline = monotonic() + timeout_seconds
    while True:
        try:
            validate_static_topology(root, expected_topology)
            validate_execution_identity(root, chunk_rows=PRODUCTION_CHUNK_ROWS)
            validate_sam3_session_reuse_identity(
                root,
                mode=PRODUCTION_SESSION_REUSE_MODE,
            )
        except FileNotFoundError as exc:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "timed out waiting for complete Entity Mask startup identity"
                ) from exc
            sleep(min(1.0, remaining))
        else:
            return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _environment_topology() -> tuple[int, int]:
    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("RANK and WORLD_SIZE must be integers") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("RANK must satisfy 0 <= RANK < WORLD_SIZE")
    return rank, world_size


def _plan(
    *,
    rank: int,
    world_size: int,
    gpus: Sequence[str],
    shard_paths: Sequence[Path],
    assignments: Sequence[StaticShardAssignment],
) -> dict[str, object]:
    return {
        "event": "startup_plan",
        "scheduling_strategy": STATIC_ASSIGNMENT_STRATEGY,
        "rank": rank,
        "world_size": world_size,
        "local_gpu_count": len(gpus),
        "global_worker_count": world_size * len(gpus),
        "stage1_completed_shards": len(shard_paths),
        "nothing_to_do": not shard_paths,
        "execution_chunk_rows": PRODUCTION_CHUNK_ROWS,
        "frame_prefetch_workers": 0,
        "sam3_request_timing": False,
        "sam3_session_reuse_mode": PRODUCTION_SESSION_REUSE_MODE,
        "workers": [
            {
                "local_slot": assignment.local_slot,
                "gpu": assignment.gpu,
                "global_worker_id": assignment.global_worker_id,
                "shard_start_position": assignment.shard_start_position,
                "shard_end_position": assignment.shard_end_position,
                "shard_count": assignment.shard_count,
            }
            for assignment in assignments
        ],
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if not PRODUCTION_BASE_CONFIG.is_file():
        raise FileNotFoundError(
            "missing entity-mask production config: "
            f"{PRODUCTION_BASE_CONFIG}; copy the validated Stage2 config here"
        )
    root = _safe_output_root(PRODUCTION_OUTPUT_ROOT)
    identity = load_config_identity(PRODUCTION_BASE_CONFIG)
    validate_entity_mask_production_config(identity.config)
    rank, world_size = _environment_topology()
    gpus = parse_gpus(None)
    shard_paths = enumerate_annotation_shards(PRODUCTION_INPUT_ROOT)
    assignments = build_local_assignments(
        shard_paths,
        rank=rank,
        world_size=world_size,
        gpus=gpus,
    )
    report = _plan(
        rank=rank,
        world_size=world_size,
        gpus=gpus,
        shard_paths=shard_paths,
        assignments=assignments,
    )
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return report
    if not shard_paths:
        result = {**report, "event": "completed", "worker_exit_codes": []}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return result

    report.update(check_stage2_candidate_judge_health(identity.config))
    topology = expected_static_topology(
        input_root=PRODUCTION_INPUT_ROOT,
        shard_paths=shard_paths,
        world_size=world_size,
        local_gpu_count=len(gpus),
    )
    coordinate_startup_identity(
        output_root=root,
        rank=rank,
        expected_topology=topology,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)

    worker_tool = Path(__file__).with_name("run_v3_entity_mask_worker.py")
    PRODUCTION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for assignment in assignments:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = assignment.gpu
            log_path = PRODUCTION_LOG_ROOT / f"rank-{rank}-gpu-{assignment.gpu}.log"
            log = log_path.open("ab")
            logs.append(log)
            command = [
                sys.executable,
                str(worker_tool),
                "--input-root",
                str(PRODUCTION_INPUT_ROOT),
                "--base-config",
                str(PRODUCTION_BASE_CONFIG),
                "--output-root",
                str(root),
                "--rank",
                str(rank),
                "--world-size",
                str(world_size),
                "--local-gpu-count",
                str(len(gpus)),
                "--local-slot",
                str(assignment.local_slot),
                "--gpu",
                assignment.gpu,
                "--global-worker-id",
                str(assignment.global_worker_id),
                "--shard-start-position",
                str(assignment.shard_start_position),
                "--shard-end-position",
                str(assignment.shard_end_position),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [process.wait() for process in processes]
    finally:
        for log in logs:
            log.close()
    result = {**report, "event": "completed", "worker_exit_codes": codes}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if any(codes):
        raise SystemExit(max(codes))
    return result


if __name__ == "__main__":
    main()
