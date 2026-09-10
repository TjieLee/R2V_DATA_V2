"""Immutable, whole-shard recovery planning; no model or artifact mutation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3 import pre_qwen_production as stage2
from tools import run_v3_entity_mask_auto as production

PLAN_VERSION = "r2v.v3.entity_mask_recovery_plan.1"
ASSIGNMENT_POLICY = "remaining_rows_lpt_v1"
DEFAULT_PLAN = Path("/mnt/workspace/litengjie/data/entity_mask_recovery/plan-v1.json")


class RecoveryShard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    position: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    estimated_durable_rows: int = Field(ge=0)
    estimated_remaining_rows: int = Field(ge=0)
    original_worker_id: int = Field(ge=0)
    original_rank: int = Field(ge=0)
    original_local_slot: int = Field(ge=0)
    original_shard_start: int = Field(ge=0)
    original_shard_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_rows(self) -> RecoveryShard:
        if (
            self.estimated_durable_rows + self.estimated_remaining_rows
            != self.row_count
        ):
            raise ValueError("recovery row estimates must sum to source row count")
        return self


class RecoveryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = PLAN_VERSION
    assignment_policy: str = ASSIGNMENT_POLICY
    created_at: str
    input_root: str
    output_root: str
    base_config_path: str
    base_config_sha256: str
    base_config_fingerprint: str
    frozen_topology: dict[str, object]
    source_shard_sha256s: dict[str, str]
    canonical_shard_sha256s: dict[str, str]
    shards: list[RecoveryShard]
    plan_sha256: str = "0" * 64


def plan_digest(plan: RecoveryPlan) -> str:
    payload = json.dumps(
        plan.model_dump(exclude={"plan_sha256"}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_topology(
    input_root: Path,
    output_root: Path,
    identity: stage2.ConfigIdentity,
) -> tuple[dict[str, object], list[Path]]:
    stage2._safe_output_root(output_root)
    stage2._validate_live_config_identity(identity)
    production.validate_entity_mask_production_config(identity.config)
    paths = stage2.enumerate_annotation_shards(input_root)
    # Original ownership comes ONLY from this marker, never recovery RANK/WORLD_SIZE.
    marker = production._static_topology_path(output_root)
    topology = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(topology, dict) or (
        topology.get("world_size"),
        topology.get("local_gpu_count"),
        topology.get("global_worker_count"),
    ) != (5, 8, 40):
        raise ValueError("recovery requires the original frozen 5x8/40 topology")
    expected = production.expected_static_topology(
        input_root=input_root,
        shard_paths=paths,
        world_size=topology["world_size"],
        local_gpu_count=topology["local_gpu_count"],
    )
    production.validate_static_topology(output_root, expected)
    stage2.validate_execution_identity(
        output_root, chunk_rows=production.PRODUCTION_CHUNK_ROWS
    )
    stage2.validate_sam3_session_reuse_identity(
        output_root, mode=production.PRODUCTION_SESSION_REUSE_MODE
    )
    return topology, paths


def validate_shard_metadata(
    shard: stage2.AnnotationShard,
    output_root: Path,
    identity: stage2.ConfigIdentity,
) -> None:
    stage2._require_recovery_shard_lineage(output_root, shard)
    path = stage2._meta_path(output_root, shard)
    if path.is_file():
        meta = json.loads(path.read_text(encoding="utf-8"))
        for key, expected in stage2._expected_meta(shard, identity).items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"recovery shard metadata mismatch: {shard.path.name}: {key}"
                )


def canonical_complete(
    shard: stage2.AnnotationShard,
    output_root: Path,
    identity: stage2.ConfigIdentity,
) -> bool:
    validate_shard_metadata(shard, output_root, identity)
    path = output_root / "parts" / shard.path.name
    if not path.is_file():
        return False
    stage2._validate_completed_shard(
        path,
        shard=shard,
        output_root=output_root,
        config_identity=identity,
        validate_artifacts=False,
    )
    return True


def estimate_durable_rows(shard: stage2.AnnotationShard, output_root: Path) -> int:
    """Cheap upper estimate: complete JSONL prefixes with existing clip metadata.

    Hash/semantic validation and exact first-invalid-artifact truncation stay in
    the existing recovery processor. A torn partial EOF is ignored, never edited.
    """
    durable = 0
    for chunk in stage2.build_execution_chunks(
        shard, chunk_rows=production.PRODUCTION_CHUNK_ROWS
    ):
        final = stage2.execution_chunk_path(output_root, shard, chunk)
        partial = final.with_name(final.name + ".partial")
        if final.exists() and partial.exists():
            raise ValueError("recovery found both completed and partial chunk")
        path = final if final.is_file() else partial
        if not path.is_file():
            continue
        expected = shard.rows[chunk.row_offset_start : chunk.row_offset_end]
        prefix_valid = True
        count = 0
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    if path == final:
                        raise ValueError(
                            "completed recovery chunk has an incomplete tail"
                        )
                    break
                value = stage2.Stage2Row.model_validate(json.loads(line))
                if count >= len(expected):
                    raise ValueError("recovery chunk has too many rows")
                source = expected[count]
                if (
                    value.source_index != source["source_index"]
                    or value.clip_uid != source.get("clip_uid")
                    or value.input_annotation_shard != str(shard.path)
                    or value.input_annotation_shard_sha256 != shard.sha256
                    or value.input_row_sha256 != stage2._row_sha256(source)
                ):
                    raise ValueError("recovery estimate row provenance mismatch")
                if value.artifact_root is not None:
                    workspace = (
                        output_root
                        / "artifacts"
                        / shard.path.stem
                        / str(value.clip_uid)
                    )
                    if (output_root / value.artifact_root).resolve(
                        strict=False
                    ) != workspace:
                        raise ValueError("recovery estimate artifact root mismatch")
                    clip_dir = workspace / "run" / "clips" / str(value.clip_uid)
                    files = [
                        workspace / "state.json",
                        workspace / "run" / "run.json",
                        clip_dir / "clip.json",
                    ]
                    if value.status != "failed_frames":
                        files += [
                            clip_dir / "frames" / "frames.json",
                            clip_dir / "masks.rle.json",
                        ]
                    prefix_valid = prefix_valid and all(
                        file.is_file() for file in files
                    )
                elif not value.status.startswith(("skipped_", "failed_")):
                    raise ValueError("recovery estimate row has no artifact provenance")
                durable += int(prefix_valid)
                count += 1
        if path == final and count != chunk.row_count:
            raise ValueError("completed recovery chunk is missing rows")
    return durable


def build_plan(
    input_root: Path,
    output_root: Path,
    identity: stage2.ConfigIdentity,
    *,
    emit: Callable[..., None] = lambda **kwargs: None,
) -> RecoveryPlan:
    topology, paths = frozen_topology(input_root, output_root, identity)
    owners = {
        position: (worker, start, end)
        for worker in range(topology["global_worker_count"])
        for start, end in [
            production.balanced_shard_bounds(
                len(paths), topology["global_worker_count"], worker
            )
        ]
        for position in range(start, end)
    }
    source_hashes, canonical_hashes, pending = {}, {}, []
    for position, path in enumerate(paths):
        emit(
            event="recovery_inventory_shard",
            shard=path.name,
            position=position,
            total_shards=len(paths),
        )
        shard = stage2.load_annotation_shard(path)
        source_hashes[path.name] = shard.sha256
        if canonical_complete(shard, output_root, identity):
            canonical_hashes[path.name] = stage2._sha256_file(
                output_root / "parts" / path.name
            )
            continue
        worker, start, end = owners[position]
        durable = estimate_durable_rows(shard, output_root)
        pending.append(
            RecoveryShard(
                name=path.name,
                position=position,
                input_sha256=shard.sha256,
                row_count=len(shard.rows),
                estimated_durable_rows=durable,
                estimated_remaining_rows=len(shard.rows) - durable,
                original_worker_id=worker,
                original_rank=worker // topology["local_gpu_count"],
                original_local_slot=worker % topology["local_gpu_count"],
                original_shard_start=start,
                original_shard_end=end,
            )
        )
    plan = RecoveryPlan(
        created_at=stage2._utc_now(),
        input_root=str(input_root),
        output_root=str(output_root),
        base_config_path=str(identity.path),
        base_config_sha256=identity.sha256,
        base_config_fingerprint=identity.fingerprint,
        frozen_topology=topology,
        source_shard_sha256s=source_hashes,
        canonical_shard_sha256s=canonical_hashes,
        shards=pending,
    )
    return plan.model_copy(update={"plan_sha256": plan_digest(plan)})


def validate_plan(
    plan: RecoveryPlan,
    input_root: Path,
    output_root: Path,
    identity: stage2.ConfigIdentity,
    *,
    check_source_hashes: bool = True,
) -> None:
    topology, paths = frozen_topology(input_root, output_root, identity)
    if (
        plan.schema_version != PLAN_VERSION
        or plan.assignment_policy != ASSIGNMENT_POLICY
        or plan.plan_sha256 != plan_digest(plan)
    ):
        raise ValueError("recovery plan version/checksum mismatch")
    if (
        plan.input_root,
        plan.output_root,
        plan.base_config_path,
        plan.base_config_sha256,
        plan.base_config_fingerprint,
        plan.frozen_topology,
    ) != (
        str(input_root),
        str(output_root),
        str(identity.path),
        identity.sha256,
        identity.fingerprint,
        topology,
    ):
        raise ValueError("recovery plan identity mismatch")
    names = [path.name for path in paths]
    if list(plan.source_shard_sha256s) != names:
        raise ValueError("recovery plan Stage1 shard list mismatch")
    pending_names = [shard.name for shard in plan.shards]
    if (
        len(pending_names) != len(set(pending_names))
        or set(pending_names) & set(plan.canonical_shard_sha256s)
        or set(pending_names) | set(plan.canonical_shard_sha256s) != set(names)
    ):
        raise ValueError("recovery plan shard inventory mismatch")
    if [shard.position for shard in plan.shards] != sorted(
        shard.position for shard in plan.shards
    ):
        raise ValueError("recovery plan shard order mismatch")
    for shard in plan.shards:
        start, end = production.balanced_shard_bounds(
            len(paths), topology["global_worker_count"], shard.original_worker_id
        )
        if (
            not start <= shard.position < end
            or names[shard.position] != shard.name
            or shard.input_sha256 != plan.source_shard_sha256s[shard.name]
            or (
                shard.original_rank,
                shard.original_local_slot,
                shard.original_shard_start,
                shard.original_shard_end,
            )
            != (
                shard.original_worker_id // topology["local_gpu_count"],
                shard.original_worker_id % topology["local_gpu_count"],
                start,
                end,
            )
        ):
            raise ValueError("recovery plan original shard ownership mismatch")
    if check_source_hashes:
        for path in paths:
            if stage2._sha256_file(path) != plan.source_shard_sha256s[path.name]:
                raise ValueError(f"recovery plan source hash mismatch: {path.name}")
        for name, digest in plan.canonical_shard_sha256s.items():
            if stage2._sha256_file(output_root / "parts" / name) != digest:
                raise ValueError(f"recovery plan canonical hash mismatch: {name}")


def load_plan(
    path: Path,
    input_root: Path,
    output_root: Path,
    identity: stage2.ConfigIdentity,
    *,
    check_source_hashes: bool = True,
) -> RecoveryPlan:
    plan = RecoveryPlan.model_validate_json(path.read_text(encoding="utf-8"))
    validate_plan(
        plan, input_root, output_root, identity, check_source_hashes=check_source_hashes
    )
    return plan


def obtain_plan(
    path: Path,
    input_root: Path,
    output_root: Path,
    identity: stage2.ConfigIdentity,
    *,
    rank: int,
    timeout_seconds: float = 1800,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    emit: Callable[..., None] = lambda **kwargs: None,
) -> RecoveryPlan:
    if path.is_file():
        return load_plan(path, input_root, output_root, identity)
    if rank == 0:
        # Publication lock only; it is not a distributed work queue or ownership system.
        with stage2.shard_lock(path.with_suffix(".lock")):
            if path.is_file():
                return load_plan(path, input_root, output_root, identity)
            plan = build_plan(input_root, output_root, identity, emit=emit)
            validate_plan(plan, input_root, output_root, identity)
            write_json_atomic(path, plan.model_dump(mode="json"))
            return plan
    deadline = monotonic() + timeout_seconds
    while not path.is_file():
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for rank 0 recovery plan")
        emit(event="recovery_plan_waiting", recovery_rank=rank, plan_path=str(path))
        sleep(min(5.0, remaining))
    return load_plan(path, input_root, output_root, identity)


def weighted_assignment(
    shards: Sequence[RecoveryShard],
    *,
    world_size: int,
    local_gpu_count: int,
) -> list[list[RecoveryShard]]:
    if world_size < 1 or local_gpu_count < 1:
        raise ValueError("physical recovery topology must be positive")
    slots: list[list[RecoveryShard]] = [[] for _ in range(world_size * local_gpu_count)]
    loads = [0] * len(slots)
    for shard in sorted(
        shards, key=lambda item: (-item.estimated_remaining_rows, item.position)
    ):
        slot = min(
            range(len(slots)),
            key=lambda index: (loads[index], len(slots[index]), index),
        )
        slots[slot].append(shard)
        loads[slot] += max(
            1, shard.estimated_remaining_rows
        )  # compaction-only still costs work
    return slots
