from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import (
    build_background_candidates,
    validate_background_reference,
)
from r2v_data_v2.v3.config import (
    ALLOWED_DATASET_ROOT,
    ALLOWED_WRITABLE_ROOT,
    V3Config,
    load_config,
)
from r2v_data_v2.v3.frames import (
    FrameDecoder,
    OpenCvFrameDecoder,
    _build_clip_frames,
    validate_sampled_frames,
)
from r2v_data_v2.v3.rank import build_coverage_state
from r2v_data_v2.v3.schemas import (
    AnnotationState,
    ClipSource,
)
from r2v_data_v2.v3.segment import (
    SegmentationBackend,
    _validate_existing_masks,
    build_sam3_segment_backend,
    segment_clips,
)
from r2v_data_v2.v3.storage import RunStorage

VISUAL_ALGORITHM_FREEZE = "d056c32b76db4b3d7c0358b38e996e7a91a288d1"
STAGE2_SCHEMA_VERSION = "r2v.v3.pre_qwen_stage2.1"
STAGE2_META_SCHEMA_VERSION = "r2v.v3.pre_qwen_shard_meta.1"
CLIP_CHECKPOINT_SCHEMA_VERSION = "r2v.v3.pre_qwen_clip_checkpoint.1"
CANARY_SCHEMA_VERSION = "r2v.v3.pre_qwen_canary.1"
DEFAULT_PRODUCTION_OUTPUT_ROOT = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1"
)
DEFAULT_CANARY_ROOT = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_pre_qwen_canary"
)

_SHARD_NAME = re.compile(r"shard-(\d{9})-(\d{9})\.jsonl")
_STAGES = {
    "annotation_ready": 0,
    "frames_ready": 1,
    "masks_ready": 2,
    "coverage_ready": 3,
    "background_ready": 4,
    "row_committed": 5,
}

TerminalStatus = Literal[
    "skipped_annotation_failed",
    "skipped_no_entities",
    "coverage_rejected",
    "ready_no_background",
    "ready_background_rejected",
    "ready_background_pending_remove",
    "failed_input",
    "failed_frames",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Stage2Row(StrictModel):
    schema_version: Literal["r2v.v3.pre_qwen_stage2.1"] = STAGE2_SCHEMA_VERSION
    source_index: int = Field(ge=0)
    clip_uid: str | None
    input_annotation_shard: str
    input_annotation_shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: TerminalStatus
    artifact_root: str | None = None
    annotation_entity_count: int = Field(default=0, ge=0)
    sam3_entity_ready: int = Field(default=0, ge=0)
    sam3_entity_not_found: int = Field(default=0, ge=0)
    sam3_entity_failed: int = Field(default=0, ge=0)
    coverage_passed: bool | None = None
    background_status: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_terminal(self) -> Stage2Row:
        ready = self.status.startswith("ready_")
        if ready and not self.coverage_passed:
            raise ValueError("ready Stage2 rows require passed coverage")
        if self.status == "coverage_rejected" and self.coverage_passed is not False:
            raise ValueError("coverage_rejected requires coverage_passed=false")
        if self.status.startswith("skipped_") and self.artifact_root is not None:
            raise ValueError("skipped Stage2 rows cannot publish artifacts")
        if self.status in {"failed_input", "failed_frames"} and not self.reason:
            raise ValueError("failed Stage2 rows require a reason")
        return self


class ClipCheckpoint(StrictModel):
    schema_version: Literal[
        "r2v.v3.pre_qwen_clip_checkpoint.1"
    ] = CLIP_CHECKPOINT_SCHEMA_VERSION
    stage: Literal[
        "annotation_ready",
        "frames_ready",
        "masks_ready",
        "coverage_ready",
        "background_ready",
        "row_committed",
    ]
    source_index: int = Field(ge=0)
    clip_uid: str
    input_annotation_shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_config_fingerprint: str
    visual_algorithm_freeze: Literal[
        "d056c32b76db4b3d7c0358b38e996e7a91a288d1"
    ] = VISUAL_ALGORITHM_FREEZE
    updated_at: str


class CanarySelectionRow(StrictModel):
    gpu_slot: int = Field(ge=0)
    input_annotation_shard: str
    input_annotation_shard_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_index: int = Field(ge=0)
    clip_uid: str
    video_path: str
    entity_count: int = Field(gt=0)
    input_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CanaryManifest(StrictModel):
    schema_version: Literal["r2v.v3.pre_qwen_canary.1"] = CANARY_SCHEMA_VERSION
    input_root: str
    base_config_path: str
    base_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_config_fingerprint: str
    visual_algorithm_freeze: Literal[
        "d056c32b76db4b3d7c0358b38e996e7a91a288d1"
    ] = VISUAL_ALGORITHM_FREEZE
    selection_policy: Literal[
        "first_n_shards_first_k_eligible_v1"
    ] = "first_n_shards_first_k_eligible_v1"
    canary_shards: int = Field(gt=0)
    samples_per_shard: int = Field(gt=0)
    gpus: list[str]
    gpu_count: int = Field(gt=0)
    samples_per_gpu: int = Field(gt=0)
    selected_samples: int = Field(gt=0)
    selected_shard_names: list[str]
    duplicate_clip_uid_skipped: int = Field(default=0, ge=0)
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qwen_required: Literal[False] = False
    qwen_calls: Literal[0] = 0
    created_at: str

    @model_validator(mode="after")
    def validate_selection_identity(self) -> CanaryManifest:
        if self.gpu_count != len(self.gpus):
            raise ValueError("canary gpu_count must match gpus")
        if self.selected_samples != self.canary_shards * self.samples_per_shard:
            raise ValueError("canary selected_samples does not match shard policy")
        if self.selected_samples != self.gpu_count * self.samples_per_gpu:
            raise ValueError("canary selected_samples does not match GPU quota")
        if len(self.selected_shard_names) != self.canary_shards:
            raise ValueError("canary selected_shard_names does not match shard count")
        if len(self.selected_shard_names) != len(set(self.selected_shard_names)):
            raise ValueError("canary selected_shard_names must be unique")
        return self


@dataclass(frozen=True)
class AnnotationShard:
    path: Path
    sha256: str
    nominal_start: int
    nominal_end: int
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ConfigIdentity:
    path: Path
    sha256: str
    fingerprint: str
    config: V3Config


class RetryableStageError(RuntimeError):
    def __init__(self, stage: str, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


class ShardBusyError(RuntimeError):
    pass


class BackendFactory(Protocol):
    def __call__(self, config: V3Config) -> SegmentationBackend: ...


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _row_sha256(row: dict[str, object]) -> str:
    return _sha256_bytes(_json_line(row))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_line(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_atomic(path: Path, values: Sequence[object]) -> None:
    content = b"".join(_json_line(value) for value in values)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _safe_output_root(path: str | Path, *, canary: bool = False) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    writable = ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    public = ALLOWED_DATASET_ROOT.resolve(strict=False)
    if writable not in resolved.parents:
        raise ValueError("Stage2 output_root must be below the writable root")
    if resolved == public or public in resolved.parents:
        raise ValueError("Stage2 must not write below the public dataset root")
    production_root = DEFAULT_PRODUCTION_OUTPUT_ROOT.resolve(strict=False)
    if canary and (
        resolved == production_root or production_root in resolved.parents
        or resolved in production_root.parents
    ):
        raise ValueError(
            "canary output_root and production Stage2 root must be fully disjoint"
        )
    return resolved


def load_config_identity(path: str | Path) -> ConfigIdentity:
    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.read_bytes()
    config = load_config(resolved)
    after = resolved.read_bytes()
    if before != after:
        raise ValueError("base config changed while its identity was computed")
    return ConfigIdentity(
        path=resolved,
        sha256=_sha256_bytes(before),
        fingerprint=config.fingerprint(),
        config=config,
    )


def _validate_live_config_identity(identity: ConfigIdentity) -> None:
    if _sha256_file(identity.path) != identity.sha256:
        raise ValueError("base config changed after Stage2 startup")


def validate_qwen_free_preflight(config: V3Config) -> dict[str, object]:
    policy = {
        "object_rescue_mode": config.sam3.object_rescue_mode,
        "not_found_rescue_mode": config.sam3.not_found_rescue_mode,
        "multi_instance_rescue_mode": config.sam3.multi_instance_rescue_mode,
        "anchor_search_mode": config.sam3.anchor_search_mode,
    }
    if config.sam3.multi_instance_rescue_mode == "qwen_anchor_select_v1":
        raise ValueError(
            "Pre-Qwen Stage2 is not Qwen-free under current frozen SAM3 config."
        )
    return {**policy, "qwen_required": False, "qwen_calls": 0}


def _shard_bounds(path: Path) -> tuple[int, int]:
    match = _SHARD_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("annotation shard filename must use canonical Stage1 bounds")
    start, end = (int(value) for value in match.groups())
    if end < start:
        raise ValueError("annotation shard filename has reversed bounds")
    return start, end


def _annotation_state(row: dict[str, object]) -> AnnotationState:
    if row.get("status") == "ready":
        return AnnotationState.model_validate(
            {
                "status": "ready",
                "instruction_template": row.get("instruction_template"),
                "entities": row.get("entities"),
                "background": row.get("background"),
                "reason": row.get("reason"),
            }
        )
    return AnnotationState.model_validate(
        {"status": "failed", "reason": row.get("reason")}
    )


def _validate_video_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ready annotation row requires video_path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError("annotation video_path must be absolute")
    resolved = raw.resolve(strict=True)
    dataset = ALLOWED_DATASET_ROOT.resolve(strict=False)
    if dataset not in resolved.parents or not resolved.is_file():
        raise ValueError("annotation video_path must be a file below dataset root")
    if resolved.suffix.lower() != ".mp4":
        raise ValueError("processed Stage2 video input must be an MP4")
    return resolved


def load_annotation_shard(path: str | Path) -> AnnotationShard:
    resolved = Path(path).expanduser().resolve(strict=True)
    start, end = _shard_bounds(resolved)
    rows: list[dict[str, object]] = []
    ready_clip_uids: dict[str, int] = {}
    with resolved.open("rb") as handle:
        for offset, line in enumerate(handle):
            if not line.endswith(b"\n"):
                raise ValueError("completed annotation shard has an incomplete tail")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("annotation shard row must contain an object")
            expected = start + offset
            if value.get("source_index") != expected:
                raise ValueError("annotation shard source_index is not contiguous")
            if expected > end:
                raise ValueError("annotation shard exceeds filename bounds")
            status = value.get("status")
            if status not in {"ready", "failed"}:
                raise ValueError("annotation shard status is invalid")
            annotation = _annotation_state(value)
            if status == "ready":
                clip_uid = value.get("clip_uid")
                if not isinstance(clip_uid, str) or not clip_uid.strip():
                    raise ValueError("ready annotation row requires clip_uid")
                prior_source_index = ready_clip_uids.get(clip_uid)
                if prior_source_index is not None:
                    raise ValueError(
                        "duplicate ready clip_uid within annotation shard: "
                        f"clip_uid={clip_uid} first_source_index={prior_source_index} "
                        f"second_source_index={expected}"
                    )
                ready_clip_uids[clip_uid] = expected
                if annotation.entities:
                    _validate_video_path(value.get("video_path"))
            elif (
                value.get("entities")
                or value.get("background") is not None
                or value.get("instruction_template")
            ):
                raise ValueError("failed annotation row published semantic content")
            rows.append(value)
    if not rows:
        raise ValueError("completed annotation shard must not be empty")
    return AnnotationShard(
        path=resolved,
        sha256=_sha256_file(resolved),
        nominal_start=start,
        nominal_end=end,
        rows=tuple(rows),
    )


def enumerate_annotation_shards(input_root: str | Path) -> list[Path]:
    root = Path(input_root).expanduser().resolve(strict=True)
    parts = root / "parts"
    paths = [path for path in parts.glob("shard-*.jsonl") if path.is_file()]
    for path in paths:
        _shard_bounds(path)
    return sorted(paths, key=lambda path: _shard_bounds(path)[0])


def _clip_source(row: dict[str, object], video_path: Path) -> ClipSource:
    source_index = int(row["source_index"])
    source_video_id = row.get("source_video_id")
    shot_index = row.get("shot_index")
    return ClipSource(
        video_path=str(video_path),
        parent_video_id=(
            str(source_video_id) if source_video_id is not None else str(row["clip_uid"])
        ),
        clip_suffix=str(shot_index if shot_index is not None else source_index),
        source_index=source_index,
        caption_raw="",
        metadata={
            "stage1_annotation_shard": str(row.get("_input_shard", "")),
            "source_video_id": source_video_id,
            "source_video_path": row.get("source_video_path"),
            "shot_index": shot_index,
        },
    )


def _workspace_config(base: V3Config, workspace: Path) -> V3Config:
    return replace(
        base,
        run_root=workspace / "run",
        export_root=workspace / "unused-export",
        sam3=replace(base.sam3, device="cuda:0"),
    )


def _checkpoint_path(workspace: Path) -> Path:
    return workspace / "state.json"


def _read_checkpoint(workspace: Path) -> ClipCheckpoint | None:
    path = _checkpoint_path(workspace)
    if not path.is_file():
        return None
    return ClipCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


def _write_checkpoint(
    workspace: Path,
    *,
    stage: str,
    row: dict[str, object],
    shard_sha256: str,
    config_identity: ConfigIdentity,
) -> ClipCheckpoint:
    value = ClipCheckpoint(
        stage=stage,
        source_index=int(row["source_index"]),
        clip_uid=str(row["clip_uid"]),
        input_annotation_shard_sha256=shard_sha256,
        input_row_sha256=_row_sha256(row),
        base_config_sha256=config_identity.sha256,
        base_config_fingerprint=config_identity.fingerprint,
        updated_at=_utc_now(),
    )
    write_json_atomic(_checkpoint_path(workspace), value.model_dump(mode="json"))
    return value


def _validate_checkpoint_identity(
    checkpoint: ClipCheckpoint,
    *,
    row: dict[str, object],
    shard_sha256: str,
    config_identity: ConfigIdentity,
) -> None:
    expected = (
        checkpoint.source_index == row.get("source_index"),
        checkpoint.clip_uid == row.get("clip_uid"),
        checkpoint.input_annotation_shard_sha256 == shard_sha256,
        checkpoint.input_row_sha256 == _row_sha256(row),
        checkpoint.base_config_sha256 == config_identity.sha256,
        checkpoint.base_config_fingerprint == config_identity.fingerprint,
    )
    if not all(expected):
        raise ValueError("existing clip checkpoint provenance does not match input")


class _RecordingBackend:
    def __init__(self, backend: SegmentationBackend) -> None:
        self.backend = backend
        self.exceptions: list[BaseException] = []

    def track(self, *args: object, **kwargs: object) -> object:
        try:
            return self.backend.track(*args, **kwargs)  # type: ignore[attr-defined]
        except BaseException as exc:
            self.exceptions.append(exc)
            raise

    def __getattr__(self, name: str) -> object:
        return getattr(self.backend, name)


def _artifact_relative(output_root: Path, workspace: Path) -> str:
    return workspace.relative_to(output_root).as_posix()


def _terminal_from_storage(
    *,
    storage: RunStorage,
    row: dict[str, object],
    shard: AnnotationShard,
    output_root: Path,
    workspace: Path,
) -> Stage2Row:
    clip_uid = str(row["clip_uid"])
    annotation = _annotation_state(row)
    frames = validate_sampled_frames(storage, clip_uid)
    masks = _validate_existing_masks(
        storage,
        clip_uid=clip_uid,
        entities=annotation.entities,
    )
    clip = storage.read_clip(clip_uid)
    expected_coverage = build_coverage_state(
        artifact=masks,
        entities=annotation.entities,
        required_visible_frames=storage.config.coverage.required_visible_frames,
    )
    if clip.coverage != expected_coverage:
        raise ValueError("stored coverage does not match frozen rank result")
    background = clip.references.background
    if expected_coverage.passed:
        if background is None:
            raise ValueError("passed coverage is missing background state")
        validate_background_reference(storage, clip_uid, background, frames=frames)
    elif background is not None:
        raise ValueError("coverage-rejected clip must not run background")
    counts = Counter(entity.status for entity in masks.entities.values())
    if not expected_coverage.passed:
        status: TerminalStatus = "coverage_rejected"
        background_status = None
    else:
        assert background is not None
        background_status = background.status
        if background.status == "none":
            status = "ready_no_background"
        elif background.status == "rejected":
            status = "ready_background_rejected"
        elif background.status == "pending_remove":
            status = "ready_background_pending_remove"
        else:
            raise ValueError(
                f"pre-Qwen Stage2 cannot publish background status {background.status}"
            )
    return Stage2Row(
        source_index=int(row["source_index"]),
        clip_uid=clip_uid,
        input_annotation_shard=str(shard.path),
        input_annotation_shard_sha256=shard.sha256,
        input_row_sha256=_row_sha256(row),
        status=status,
        artifact_root=_artifact_relative(output_root, workspace),
        annotation_entity_count=len(annotation.entities),
        sam3_entity_ready=counts["ready"],
        sam3_entity_not_found=counts["not_found"],
        sam3_entity_failed=counts["failed"],
        coverage_passed=expected_coverage.passed,
        background_status=background_status,
    )


def process_ready_clip(
    *,
    row: dict[str, object],
    shard: AnnotationShard,
    output_root: Path,
    workspace: Path,
    config_identity: ConfigIdentity,
    backend: SegmentationBackend,
    decoder: FrameDecoder | None = None,
) -> Stage2Row:
    annotation = _annotation_state(row)
    if annotation.status != "ready" or not annotation.entities:
        raise ValueError("process_ready_clip requires ready entities")
    video_path = _validate_video_path(row.get("video_path"))
    checkpoint = _read_checkpoint(workspace)
    if checkpoint is not None:
        _validate_checkpoint_identity(
            checkpoint,
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    elif workspace.exists() and any(workspace.iterdir()):
        raise ValueError("clip workspace is non-empty without a checkpoint")

    config = _workspace_config(config_identity.config, workspace)
    storage = RunStorage(config)
    if checkpoint is None:
        storage.initialize(git_commit=VISUAL_ALGORITHM_FREEZE)
        storage.create_clip(
            clip_uid=str(row["clip_uid"]),
            source=_clip_source(row, video_path),
        )
        storage.write_annotation(str(row["clip_uid"]), annotation)
        checkpoint = _write_checkpoint(
            workspace,
            stage="annotation_ready",
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    clip_uid = str(row["clip_uid"])

    if _STAGES[checkpoint.stage] < _STAGES["frames_ready"]:
        manifest = storage.frames_manifest_path(clip_uid)
        if manifest.is_file():
            validate_sampled_frames(storage, clip_uid)
        else:
            try:
                _build_clip_frames(
                    config,
                    storage,
                    clip_uid=clip_uid,
                    video_path=video_path,
                    decoder=decoder or OpenCvFrameDecoder(),
                )
            except ValueError as exc:
                return Stage2Row(
                    source_index=int(row["source_index"]),
                    clip_uid=clip_uid,
                    input_annotation_shard=str(shard.path),
                    input_annotation_shard_sha256=shard.sha256,
                    input_row_sha256=_row_sha256(row),
                    status="failed_frames",
                    artifact_root=_artifact_relative(output_root, workspace),
                    annotation_entity_count=len(annotation.entities),
                    reason=str(exc),
                )
            except Exception as exc:
                raise RetryableStageError("frames", str(exc)) from exc
        checkpoint = _write_checkpoint(
            workspace,
            stage="frames_ready",
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    else:
        validate_sampled_frames(storage, clip_uid)

    if _STAGES[checkpoint.stage] < _STAGES["masks_ready"]:
        if storage.masks_path(clip_uid).is_file():
            _validate_existing_masks(storage, clip_uid=clip_uid, entities=annotation.entities)
        else:
            recording = _RecordingBackend(backend)
            stats = segment_clips(config, storage, backend=recording)  # type: ignore[arg-type]
            if recording.exceptions:
                storage.masks_path(clip_uid).unlink(missing_ok=True)
                raise RetryableStageError("sam3", str(recording.exceptions[0]))
            if stats.failed:
                storage.masks_path(clip_uid).unlink(missing_ok=True)
                raise RetryableStageError("sam3", "SAM3 stage failed without artifact")
            _validate_existing_masks(storage, clip_uid=clip_uid, entities=annotation.entities)
        checkpoint = _write_checkpoint(
            workspace,
            stage="masks_ready",
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    else:
        _validate_existing_masks(storage, clip_uid=clip_uid, entities=annotation.entities)

    if _STAGES[checkpoint.stage] < _STAGES["coverage_ready"]:
        masks = storage.read_masks(clip_uid)
        coverage = build_coverage_state(
            artifact=masks,
            entities=annotation.entities,
            required_visible_frames=config.coverage.required_visible_frames,
        )
        storage.write_coverage(clip_uid, coverage)
        checkpoint = _write_checkpoint(
            workspace,
            stage="coverage_ready",
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    else:
        clip = storage.read_clip(clip_uid)
        if clip.coverage is None:
            raise ValueError("coverage_ready checkpoint is missing coverage")
        expected = build_coverage_state(
            artifact=storage.read_masks(clip_uid),
            entities=annotation.entities,
            required_visible_frames=config.coverage.required_visible_frames,
        )
        if clip.coverage != expected:
            raise ValueError("stored coverage is corrupt")

    clip = storage.read_clip(clip_uid)
    assert clip.coverage is not None
    if not clip.coverage.passed:
        return _terminal_from_storage(
            storage=storage,
            row=row,
            shard=shard,
            output_root=output_root,
            workspace=workspace,
        )

    if _STAGES[checkpoint.stage] < _STAGES["background_ready"]:
        if clip.references.background is not None:
            validate_background_reference(
                storage,
                clip_uid,
                clip.references.background,
            )
        else:
            stats = build_background_candidates(config, storage)
            if stats.failed:
                raise RetryableStageError(
                    "background", "background stage failed without valid state"
                )
        _write_checkpoint(
            workspace,
            stage="background_ready",
            row=row,
            shard_sha256=shard.sha256,
            config_identity=config_identity,
        )
    return _terminal_from_storage(
        storage=storage,
        row=row,
        shard=shard,
        output_root=output_root,
        workspace=workspace,
    )


def _skipped_row(row: dict[str, object], shard: AnnotationShard) -> Stage2Row:
    status: TerminalStatus = (
        "skipped_annotation_failed"
        if row.get("status") == "failed"
        else "skipped_no_entities"
    )
    entities = row.get("entities")
    return Stage2Row(
        source_index=int(row["source_index"]),
        clip_uid=row.get("clip_uid") if isinstance(row.get("clip_uid"), str) else None,
        input_annotation_shard=str(shard.path),
        input_annotation_shard_sha256=shard.sha256,
        input_row_sha256=_row_sha256(row),
        status=status,
        annotation_entity_count=len(entities) if isinstance(entities, list) else 0,
        reason=str(row.get("reason")) if row.get("reason") else None,
    )


def _read_output_rows(
    path: Path,
    *,
    shard: AnnotationShard,
    recover_tail: bool,
) -> list[Stage2Row]:
    if not path.exists():
        return []
    mode = "r+b" if recover_tail else "rb"
    rows: list[Stage2Row] = []
    with path.open(mode) as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not recover_tail:
                    raise ValueError("completed Stage2 shard has an incomplete tail")
                handle.seek(line_start)
                handle.truncate(line_start)
                handle.flush()
                os.fsync(handle.fileno())
                break
            value = Stage2Row.model_validate(json.loads(line))
            if len(rows) >= len(shard.rows):
                raise ValueError("Stage2 shard contains too many rows")
            expected = shard.rows[len(rows)]
            if (
                value.source_index != expected.get("source_index")
                or value.clip_uid
                != (expected.get("clip_uid") if isinstance(expected.get("clip_uid"), str) else None)
                or value.input_annotation_shard_sha256 != shard.sha256
                or value.input_row_sha256 != _row_sha256(expected)
            ):
                raise ValueError("Stage2 output is not an exact annotation prefix")
            rows.append(value)
    return rows


def _meta_path(output_root: Path, shard: AnnotationShard) -> Path:
    return output_root / "parts" / f"{shard.path.stem}.meta.json"


def _expected_meta(
    shard: AnnotationShard,
    config_identity: ConfigIdentity,
) -> dict[str, object]:
    return {
        "schema_version": STAGE2_META_SCHEMA_VERSION,
        "input_annotation_shard": str(shard.path),
        "input_annotation_shard_sha256": shard.sha256,
        "source_index_start": shard.rows[0]["source_index"],
        "source_index_end": shard.rows[-1]["source_index"],
        "input_row_count": len(shard.rows),
        "visual_algorithm_freeze": VISUAL_ALGORITHM_FREEZE,
        "sam3_checkpoint": (
            str(config_identity.config.sam3.model_path)
            if config_identity.config.sam3.model_path is not None
            else None
        ),
        "frame_count": 10,
        "base_config_path": str(config_identity.path),
        "base_config_sha256": config_identity.sha256,
        "base_config_fingerprint": config_identity.fingerprint,
        "qwen_required": False,
        "qwen_calls": 0,
    }


def _ensure_meta(
    shard: AnnotationShard,
    *,
    output_root: Path,
    config_identity: ConfigIdentity,
) -> Path:
    path = _meta_path(output_root, shard)
    expected = _expected_meta(shard, config_identity)
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(f"Stage2 shard metadata mismatch: {key}")
    else:
        write_json_atomic(path, {**expected, "created_at": _utc_now(), "completed_at": None})
    return path


@contextmanager
def shard_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError as exc:
            raise ShardBusyError(str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _failure_attempt(
    output_root: Path,
    shard: AnnotationShard,
    row: dict[str, object],
    exc: RetryableStageError,
) -> None:
    _append_jsonl(
        output_root / "failures" / f"{shard.path.stem}.jsonl",
        {
            "timestamp": _utc_now(),
            "source_index": row["source_index"],
            "clip_uid": row.get("clip_uid"),
            "stage": exc.stage,
            "reason": exc.reason,
            "retryable": True,
        },
    )


def _validate_materialized_row(
    value: Stage2Row,
    *,
    input_row: dict[str, object],
    shard: AnnotationShard,
    output_root: Path,
    config_identity: ConfigIdentity,
) -> None:
    if value.artifact_root is None:
        return
    workspace = (output_root / value.artifact_root).resolve(strict=True)
    if output_root.resolve(strict=False) not in workspace.parents:
        raise ValueError("Stage2 artifact root escaped output_root")
    checkpoint = _read_checkpoint(workspace)
    if checkpoint is None:
        raise ValueError("published Stage2 artifact is missing its checkpoint")
    _validate_checkpoint_identity(
        checkpoint,
        row=input_row,
        shard_sha256=shard.sha256,
        config_identity=config_identity,
    )
    if value.status == "failed_frames":
        return
    storage = RunStorage(_workspace_config(config_identity.config, workspace))
    expected = _terminal_from_storage(
        storage=storage,
        row=input_row,
        shard=shard,
        output_root=output_root,
        workspace=workspace,
    )
    comparable = (
        "source_index",
        "clip_uid",
        "input_annotation_shard_sha256",
        "input_row_sha256",
        "status",
        "artifact_root",
        "annotation_entity_count",
        "sam3_entity_ready",
        "sam3_entity_not_found",
        "sam3_entity_failed",
        "coverage_passed",
        "background_status",
    )
    if any(getattr(value, name) != getattr(expected, name) for name in comparable):
        raise ValueError("Stage2 row does not match its durable clip artifacts")


def process_shard(
    input_shard: str | Path,
    *,
    output_root: str | Path,
    config_identity: ConfigIdentity,
    backend: SegmentationBackend | None,
    decoder: FrameDecoder | None = None,
    acquire_lock: bool = True,
) -> dict[str, object]:
    validate_qwen_free_preflight(config_identity.config)
    _validate_live_config_identity(config_identity)
    root = _safe_output_root(output_root)
    shard = load_annotation_shard(input_shard)
    part = root / "parts" / shard.path.name
    partial = part.with_name(f"{part.name}.partial")
    lock_path = root / "locks" / f"{shard.path.stem}.lock"

    @contextmanager
    def maybe_lock() -> Iterator[None]:
        if acquire_lock:
            with shard_lock(lock_path):
                yield
        else:
            yield

    with maybe_lock():
        _ensure_meta(shard, output_root=root, config_identity=config_identity)
        if part.is_file():
            completed = _read_output_rows(part, shard=shard, recover_tail=False)
            if len(completed) != len(shard.rows):
                raise ValueError("completed Stage2 shard is missing rows")
            for value, input_row in zip(completed, shard.rows):
                _validate_materialized_row(
                    value,
                    input_row=input_row,
                    shard=shard,
                    output_root=root,
                    config_identity=config_identity,
                )
            return {"path": str(part), "rows": len(completed), "skipped": True}
        completed = _read_output_rows(partial, shard=shard, recover_tail=True)
        for row in shard.rows[len(completed) :]:
            annotation = _annotation_state(row)
            if annotation.status == "failed" or not annotation.entities:
                result = _skipped_row(row, shard)
            else:
                if backend is None:
                    raise RuntimeError("unfinished Stage2 shard requires SAM3 backend")
                workspace = root / "artifacts" / shard.path.stem / str(row["clip_uid"])
                try:
                    result = process_ready_clip(
                        row=row,
                        shard=shard,
                        output_root=root,
                        workspace=workspace,
                        config_identity=config_identity,
                        backend=backend,
                        decoder=decoder,
                    )
                except RetryableStageError as exc:
                    _failure_attempt(root, shard, row, exc)
                    return {
                        "path": str(partial),
                        "rows": len(completed),
                        "skipped": False,
                        "retryable": True,
                        "source_index": row["source_index"],
                        "stage": exc.stage,
                    }
            _append_jsonl(partial, result.model_dump(mode="json"))
            completed.append(result)
            if result.artifact_root is not None:
                workspace = root / result.artifact_root
                checkpoint = _read_checkpoint(workspace)
                if checkpoint is not None:
                    _write_checkpoint(
                        workspace,
                        stage="row_committed",
                        row=row,
                        shard_sha256=shard.sha256,
                        config_identity=config_identity,
                    )
        validated = _read_output_rows(partial, shard=shard, recover_tail=False)
        if len(validated) != len(shard.rows):
            raise RuntimeError("Stage2 did not publish every annotation row")
        for value, input_row in zip(validated, shard.rows):
            _validate_materialized_row(
                value,
                input_row=input_row,
                shard=shard,
                output_root=root,
                config_identity=config_identity,
            )
        if _sha256_file(shard.path) != shard.sha256:
            raise ValueError("input annotation shard changed during Stage2 processing")
        meta_path = _meta_path(root, shard)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "completed_at": _utc_now(),
                "counters": dict(Counter(row.status for row in validated)),
            }
        )
        write_json_atomic(meta_path, meta)
        os.replace(partial, part)
        _fsync_directory(part.parent)
        return {"path": str(part), "rows": len(validated), "skipped": False}


def inspect_stage2_shard(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve(strict=True)
    meta_path = resolved.with_name(f"{resolved.stem}.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows: list[Stage2Row] = []
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise ValueError("Stage2 shard has an incomplete tail")
            rows.append(Stage2Row.model_validate(json.loads(line)))
    statuses = Counter(row.status for row in rows)
    result: dict[str, object] = {
        "path": str(resolved),
        "rows": len(rows),
        "source_start_index": rows[0].source_index if rows else None,
        "source_end_index": rows[-1].source_index if rows else None,
        "input_annotation_sha256": meta.get("input_annotation_shard_sha256"),
        "artifact_validation": "not_requested",
        "total_annotation_entities": sum(row.annotation_entity_count for row in rows),
        "sam3_entity_ready": sum(row.sam3_entity_ready for row in rows),
        "sam3_entity_not_found": sum(row.sam3_entity_not_found for row in rows),
        "sam3_entity_failed": sum(row.sam3_entity_failed for row in rows),
        "coverage_passed": sum(row.coverage_passed is True for row in rows),
        "coverage_rejected": statuses["coverage_rejected"],
        "background_none": statuses["ready_no_background"],
        "background_rejected": statuses["ready_background_rejected"],
        "background_pending_remove": statuses["ready_background_pending_remove"],
        "failed_frames": statuses["failed_frames"],
        "failed_sam3": 0,
        "failed_coverage": 0,
        "failed_background": 0,
        "annotation_failed": statuses["skipped_annotation_failed"],
        "no_entities": statuses["skipped_no_entities"],
    }
    result.update(statuses)
    return result


def _lock_busy(path: Path) -> bool:
    try:
        with shard_lock(path):
            return False
    except ShardBusyError:
        return True


def inventory(input_root: str | Path, output_root: str | Path) -> dict[str, object]:
    root = _safe_output_root(output_root)
    shards = enumerate_annotation_shards(input_root)
    stage1 = Counter[str]()
    total_rows = entities = 0
    for path in shards:
        shard = load_annotation_shard(path)
        total_rows += len(shard.rows)
        for row in shard.rows:
            annotation = _annotation_state(row)
            if annotation.status == "failed":
                stage1["annotation_failed"] += 1
            elif annotation.entities:
                stage1["annotation_ready_with_entities"] += 1
                entities += len(annotation.entities)
            else:
                stage1["annotation_ready_zero_entities"] += 1
    completed = sum((root / "parts" / path.name).is_file() for path in shards)
    partial = sum(
        (root / "parts" / f"{path.name}.partial").is_file() for path in shards
    )
    busy = sum(
        _lock_busy(root / "locks" / f"{path.stem}.lock") for path in shards
    )
    substages = Counter[str]()
    for state_path in (root / "artifacts").glob("*/*/state.json"):
        state = ClipCheckpoint.model_validate_json(state_path.read_text(encoding="utf-8"))
        substages[state.stage] += 1
    committed_rows: list[Stage2Row] = []
    for path in sorted((root / "parts").glob("shard-*.jsonl*")):
        if path.name.endswith(".meta.json"):
            continue
        with path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    break
                committed_rows.append(Stage2Row.model_validate(json.loads(line)))
    committed_statuses = Counter(row.status for row in committed_rows)
    retryable_failures = sum(
        len(path.read_bytes().splitlines())
        for path in (root / "failures").glob("shard-*.jsonl")
    )
    return {
        "stage1_completed_shards": len(shards),
        "stage1_total_rows": total_rows,
        "annotation_ready": stage1["annotation_ready_with_entities"]
        + stage1["annotation_ready_zero_entities"],
        "annotation_failed": stage1["annotation_failed"],
        "annotation_ready_zero_entities": stage1["annotation_ready_zero_entities"],
        "annotation_ready_with_entities": stage1["annotation_ready_with_entities"],
        "annotation_entities_total": entities,
        "stage2_completed_shards": completed,
        "stage2_partial_shards": partial,
        "outstanding_shards": len(shards) - completed,
        "busy_shard_locks": busy,
        "substage_checkpoints": dict(substages),
        "stage2_rows_committed": len(committed_rows),
        "frames_ready_not_committed": substages["frames_ready"],
        "masks_ready_not_committed": substages["masks_ready"],
        "coverage_ready_not_committed": substages["coverage_ready"],
        "background_ready_not_committed": substages["background_ready"],
        "coverage_passed": sum(row.coverage_passed is True for row in committed_rows),
        "coverage_rejected": committed_statuses["coverage_rejected"],
        "background_none": committed_statuses["ready_no_background"],
        "background_rejected": committed_statuses["ready_background_rejected"],
        "background_pending_remove": committed_statuses[
            "ready_background_pending_remove"
        ],
        "retryable_infrastructure_failures": retryable_failures,
        "output_root": str(root),
    }


def select_canary_rows(
    input_root: str | Path,
    *,
    gpu_count: int,
    canary_shards: int,
    samples_per_shard: int,
) -> list[CanarySelectionRow]:
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    if canary_shards < 1:
        raise ValueError("canary_shards must be positive")
    if samples_per_shard < 1:
        raise ValueError("samples_per_shard must be positive")
    needed = canary_shards * samples_per_shard
    if needed % gpu_count:
        raise ValueError(
            "canary_shards * samples_per_shard must be divisible by gpu_count; "
            "adjust --canary-shards or --samples-per-shard"
        )
    samples_per_gpu = needed // gpu_count
    shard_paths = enumerate_annotation_shards(input_root)
    if len(shard_paths) < canary_shards:
        raise ValueError(
            f"only {len(shard_paths)} completed Stage1 shards; "
            f"{canary_shards} required"
        )
    selected: list[CanarySelectionRow] = []
    seen_clip_uids: dict[str, tuple[str, int]] = {}
    for path in shard_paths[:canary_shards]:
        shard = load_annotation_shard(path)
        shard_selected = 0
        for row in shard.rows:
            annotation = _annotation_state(row)
            if annotation.status != "ready" or not annotation.entities:
                continue
            video_path = _validate_video_path(row.get("video_path"))
            clip_uid = str(row["clip_uid"])
            previous = seen_clip_uids.get(clip_uid)
            if previous is not None:
                first_shard, first_source_index = previous
                raise ValueError(
                    "duplicate clip_uid across Stage1 shards: "
                    f"clip_uid={clip_uid} "
                    f"first_shard={first_shard} "
                    f"first_source_index={first_source_index} "
                    f"second_shard={shard.path.name} "
                    f"second_source_index={row['source_index']}"
                )
            seen_clip_uids[clip_uid] = (shard.path.name, int(row["source_index"]))
            position = len(selected)
            selected.append(
                CanarySelectionRow(
                    gpu_slot=position // samples_per_gpu,
                    input_annotation_shard=str(shard.path),
                    input_annotation_shard_sha256=shard.sha256,
                    source_index=int(row["source_index"]),
                    clip_uid=clip_uid,
                    video_path=str(video_path),
                    entity_count=len(annotation.entities),
                    input_row_sha256=_row_sha256(row),
                )
            )
            shard_selected += 1
            if shard_selected == samples_per_shard:
                break
        if shard_selected != samples_per_shard:
            raise ValueError(
                f"{shard.path.name} has only {shard_selected} eligible rows; "
                f"{samples_per_shard} required"
            )
    if len(selected) != needed:
        raise AssertionError("canary selection did not produce the required row count")
    return selected


def prepare_canary(
    *,
    input_root: str | Path,
    base_config: str | Path,
    output_root: str | Path,
    gpus: Sequence[str],
    canary_shards: int,
    samples_per_shard: int,
) -> tuple[CanaryManifest, list[CanarySelectionRow]]:
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("canary GPUs must be a non-empty unique list")
    root = _safe_output_root(output_root, canary=True)
    config_identity = load_config_identity(base_config)
    validate_qwen_free_preflight(config_identity.config)
    selected_samples = canary_shards * samples_per_shard
    if selected_samples % len(gpus):
        raise ValueError(
            "canary_shards * samples_per_shard must be divisible by GPU count; "
            "adjust the canary selection parameters"
        )
    samples_per_gpu = selected_samples // len(gpus)
    selection_path = root / "selection.jsonl"
    manifest_path = root / "canary_manifest.json"
    if manifest_path.is_file() or selection_path.is_file():
        if not (manifest_path.is_file() and selection_path.is_file()):
            raise ValueError("canary identity files are incomplete")
        manifest = CanaryManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        raw = selection_path.read_bytes()
        selection = [
            CanarySelectionRow.model_validate(json.loads(line))
            for line in raw.splitlines()
            if line
        ]
        expected = (
            manifest.input_root == str(Path(input_root).expanduser().resolve(strict=True)),
            manifest.base_config_path == str(config_identity.path),
            manifest.base_config_sha256 == config_identity.sha256,
            manifest.base_config_fingerprint == config_identity.fingerprint,
            manifest.gpus == list(gpus),
            manifest.gpu_count == len(gpus),
            manifest.canary_shards == canary_shards,
            manifest.samples_per_shard == samples_per_shard,
            manifest.samples_per_gpu == samples_per_gpu,
            manifest.selection_sha256 == _sha256_bytes(raw),
            manifest.selected_samples == len(selection),
            manifest.selected_shard_names
            == list(
                dict.fromkeys(
                    Path(item.input_annotation_shard).name for item in selection
                )
            ),
            all(
                sum(item.gpu_slot == slot for item in selection)
                == manifest.samples_per_gpu
                for slot in range(len(gpus))
            ),
        )
        if not all(expected):
            raise ValueError("existing canary identity does not match request")
        for item in selection:
            shard = load_annotation_shard(item.input_annotation_shard)
            if shard.sha256 != item.input_annotation_shard_sha256:
                raise ValueError("canary input annotation shard SHA changed")
            source_row = shard.rows[item.source_index - shard.nominal_start]
            if _row_sha256(source_row) != item.input_row_sha256:
                raise ValueError("canary selected input row changed")
        return manifest, selection
    if root.exists() and any(root.iterdir()):
        raise ValueError("canary output root is non-empty without identity")
    root.mkdir(parents=True, exist_ok=True)
    selection = select_canary_rows(
        input_root,
        gpu_count=len(gpus),
        canary_shards=canary_shards,
        samples_per_shard=samples_per_shard,
    )
    selected_shard_names = list(
        dict.fromkeys(Path(item.input_annotation_shard).name for item in selection)
    )
    raw = b"".join(_json_line(item.model_dump(mode="json")) for item in selection)
    _write_jsonl_atomic(
        selection_path,
        [item.model_dump(mode="json") for item in selection],
    )
    manifest = CanaryManifest(
        input_root=str(Path(input_root).expanduser().resolve(strict=True)),
        base_config_path=str(config_identity.path),
        base_config_sha256=config_identity.sha256,
        base_config_fingerprint=config_identity.fingerprint,
        canary_shards=canary_shards,
        samples_per_shard=samples_per_shard,
        gpus=list(gpus),
        gpu_count=len(gpus),
        samples_per_gpu=samples_per_gpu,
        selected_samples=len(selection),
        selected_shard_names=selected_shard_names,
        duplicate_clip_uid_skipped=0,
        selection_sha256=_sha256_bytes(raw),
        created_at=_utc_now(),
    )
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return manifest, selection


def run_canary_worker(
    *,
    output_root: str | Path,
    manifest: CanaryManifest,
    selection: Sequence[CanarySelectionRow],
    gpu_slot: int,
    backend: SegmentationBackend | None,
    decoder: FrameDecoder | None = None,
) -> dict[str, object]:
    root = _safe_output_root(output_root, canary=True)
    config_identity = load_config_identity(manifest.base_config_path)
    validate_qwen_free_preflight(config_identity.config)
    if (
        config_identity.sha256 != manifest.base_config_sha256
        or config_identity.fingerprint != manifest.base_config_fingerprint
    ):
        raise ValueError("canary base config identity changed")
    selected = [item for item in selection if item.gpu_slot == gpu_slot]
    worker_path = root / "workers" / f"gpu-{gpu_slot}.jsonl"
    partial = worker_path.with_name(f"{worker_path.name}.partial")
    expected_rows: list[dict[str, object]] = []
    for item in selected:
        shard = load_annotation_shard(item.input_annotation_shard)
        if shard.sha256 != item.input_annotation_shard_sha256:
            raise ValueError("canary selected annotation shard SHA changed")
        row = shard.rows[item.source_index - shard.nominal_start]
        if _row_sha256(row) != item.input_row_sha256:
            raise ValueError("canary selected annotation row changed")
        expected_rows.append(row)
    lock_path = root / "locks" / f"canary-gpu-{gpu_slot}.lock"
    with shard_lock(lock_path):
        completed: list[Stage2Row] = []
        if worker_path.is_file():
            lines = worker_path.read_bytes().splitlines()
            completed = [Stage2Row.model_validate(json.loads(line)) for line in lines]
            if len(completed) != len(selected):
                raise ValueError("completed canary worker is missing selected rows")
            for value, item, row in zip(completed, selected, expected_rows):
                _validate_materialized_row(
                    value,
                    input_row=row,
                    shard=load_annotation_shard(item.input_annotation_shard),
                    output_root=root,
                    config_identity=config_identity,
                )
            return {"gpu_slot": gpu_slot, "selected": len(selected), "completed": len(completed), "skipped": True}
        if partial.is_file():
            with partial.open("r+b") as handle:
                while True:
                    start = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        handle.seek(start)
                        handle.truncate(start)
                        handle.flush()
                        os.fsync(handle.fileno())
                        break
                    completed.append(Stage2Row.model_validate(json.loads(line)))
        for index, item in enumerate(selected[: len(completed)]):
            value = completed[index]
            if (
                value.source_index != item.source_index
                or value.clip_uid != item.clip_uid
                or value.input_row_sha256 != item.input_row_sha256
            ):
                raise ValueError("canary worker output is not an exact selection prefix")
            _validate_materialized_row(
                value,
                input_row=expected_rows[index],
                shard=load_annotation_shard(item.input_annotation_shard),
                output_root=root,
                config_identity=config_identity,
            )
        for item, row in zip(selected[len(completed) :], expected_rows[len(completed) :]):
            if backend is None:
                raise RuntimeError("unfinished canary worker requires SAM3 backend")
            shard = load_annotation_shard(item.input_annotation_shard)
            workspace = root / "artifacts" / item.clip_uid
            try:
                result = process_ready_clip(
                    row=row,
                    shard=shard,
                    output_root=root,
                    workspace=workspace,
                    config_identity=config_identity,
                    backend=backend,
                    decoder=decoder,
                )
            except RetryableStageError as exc:
                _append_jsonl(
                    root / "logs" / f"gpu-{gpu_slot}.failures.jsonl",
                    {
                        "timestamp": _utc_now(),
                        "source_index": item.source_index,
                        "clip_uid": item.clip_uid,
                        "stage": exc.stage,
                        "reason": exc.reason,
                        "retryable": True,
                    },
                )
                return {
                    "gpu_slot": gpu_slot,
                    "selected": len(selected),
                    "completed": len(completed),
                    "retryable": True,
                }
            _append_jsonl(partial, result.model_dump(mode="json"))
            completed.append(result)
            _write_checkpoint(
                workspace,
                stage="row_committed",
                row=row,
                shard_sha256=shard.sha256,
                config_identity=config_identity,
            )
        worker_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, worker_path)
        _fsync_directory(worker_path.parent)
        return {"gpu_slot": gpu_slot, "selected": len(selected), "completed": len(completed), "skipped": False}


def canary_summary(output_root: str | Path, manifest: CanaryManifest) -> dict[str, object]:
    root = _safe_output_root(output_root, canary=True)
    all_rows: list[Stage2Row] = []
    per_gpu: dict[str, dict[str, int]] = {}
    complete = True
    outstanding_retryable_work = 0
    for slot in range(len(manifest.gpus)):
        path = root / "workers" / f"gpu-{slot}.jsonl"
        partial = path.with_name(f"{path.name}.partial")
        row_source = path if path.is_file() else partial
        rows: list[Stage2Row] = []
        if row_source.is_file():
            for line in row_source.read_bytes().splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                rows.append(Stage2Row.model_validate(json.loads(line)))
        all_rows.extend(rows)
        per_gpu[f"gpu{slot}"] = {
            "selected": manifest.samples_per_gpu,
            "completed": len(rows),
        }
        complete = complete and len(rows) == manifest.samples_per_gpu
        failure_log = root / "logs" / f"gpu-{slot}.failures.jsonl"
        if len(rows) != manifest.samples_per_gpu and failure_log.is_file():
            outstanding_retryable_work += 1
    artifacts = [path for path in (root / "artifacts").rglob("*") if path.is_file()]
    frames_bytes = sum(path.stat().st_size for path in artifacts if "/frames/" in path.as_posix())
    masks_bytes = sum(path.stat().st_size for path in artifacts if path.name == "masks.rle.json")
    total_bytes = sum(path.stat().st_size for path in artifacts)
    statuses = Counter(row.status for row in all_rows)
    failed_input = statuses["failed_input"]
    failed_frames = statuses["failed_frames"]
    terminal_failures = failed_input + failed_frames
    selection_complete = complete and len(all_rows) == manifest.selected_samples
    functional_pass = (
        selection_complete
        and outstanding_retryable_work == 0
        and failed_input == 0
        and failed_frames == 0
    )
    summary = {
        "status": "complete" if selection_complete else "partial",
        "selection_complete": selection_complete,
        "functional_status": "pass" if functional_pass else "fail",
        "selected_samples": manifest.selected_samples,
        "duplicate_clip_uid_skipped": manifest.duplicate_clip_uid_skipped,
        "per_gpu": per_gpu,
        "annotation_entities_total": sum(row.annotation_entity_count for row in all_rows),
        "frames_completed": sum(
            row.artifact_root is not None and row.status != "failed_frames"
            for row in all_rows
        ),
        "sam3_clips": sum(row.coverage_passed is not None for row in all_rows),
        "sam3_entities": sum(row.annotation_entity_count for row in all_rows),
        "sam3_ready": sum(row.sam3_entity_ready for row in all_rows),
        "sam3_not_found": sum(row.sam3_entity_not_found for row in all_rows),
        "sam3_failed": sum(row.sam3_entity_failed for row in all_rows),
        "coverage_passed": sum(row.coverage_passed is True for row in all_rows),
        "coverage_rejected": statuses["coverage_rejected"],
        "background_none": statuses["ready_no_background"],
        "background_rejected": statuses["ready_background_rejected"],
        "background_pending_remove": statuses["ready_background_pending_remove"],
        "failed_input": failed_input,
        "failed_frames": failed_frames,
        "terminal_failures": terminal_failures,
        "outstanding_retryable_work": outstanding_retryable_work,
        "retryable_failures": sum(
            len(path.read_bytes().splitlines())
            for path in (root / "logs").glob("*.failures.jsonl")
        ),
        "qwen_required": False,
        "qwen_calls": 0,
        "frames_bytes": frames_bytes,
        "masks_bytes": masks_bytes,
        "total_artifact_bytes": total_bytes,
        "mean_artifact_bytes_per_processed_clip": total_bytes / len(all_rows) if all_rows else 0.0,
    }
    write_json_atomic(root / "summary.json", summary)
    return summary


def close_backend(backend: object) -> None:
    closer = getattr(backend, "close", None)
    if callable(closer):
        closer()


def default_backend_factory(config: V3Config) -> SegmentationBackend:
    return build_sam3_segment_backend(
        replace(config, sam3=replace(config.sam3, device="cuda"))
    )


def remove_orphan_temporary_artifacts(output_root: str | Path) -> list[str]:
    root = _safe_output_root(output_root)
    removed: list[str] = []
    for path in (root / "artifacts").glob("**/.tmp-*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed.append(path.relative_to(root).as_posix())
    return sorted(removed)
