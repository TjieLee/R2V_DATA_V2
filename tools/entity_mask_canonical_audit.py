"""Read-only FAST closure checks; no image reads, content hashes, or model calls."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from r2v_data_v2.v3 import pre_qwen_production as stage2
from r2v_data_v2.v3.background import _resolve_run_artifact, _validate_source_frame


def _require_file(path: Path, *, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"required artifact is not a file inside {root}: {path}")
    return resolved


def validate_fast_materialized_row(
    value: stage2.Stage2Row,
    *,
    input_row: dict[str, object],
    shard: stage2.AnnotationShard,
    output_root: Path,
    config_identity: stage2.ConfigIdentity,
) -> None:
    """Validate metadata/paths, leaving image-byte integrity to the FULL audit."""
    if value.artifact_root is None:
        if value.status not in {
            "skipped_annotation_failed",
            "skipped_no_entities",
            "failed_input",
        }:
            raise ValueError("materialized Stage2 row has no artifact_root")
        return
    if Path(value.artifact_root).is_absolute():
        raise ValueError("Stage2 artifact_root must be output-root relative")
    workspace = (output_root / value.artifact_root).resolve(strict=True)
    expected = output_root / "artifacts" / shard.path.stem / str(value.clip_uid)
    if workspace != expected or not workspace.is_dir():
        raise ValueError("Stage2 workspace path identity mismatch")
    _require_file(workspace / "state.json", root=workspace)
    checkpoint = stage2._read_checkpoint(workspace)
    if checkpoint is None:
        raise ValueError("published Stage2 artifact is missing its checkpoint")
    stage2._validate_checkpoint_identity(
        checkpoint,
        row=input_row,
        shard_sha256=shard.sha256,
        config_identity=config_identity,
    )
    config, storage = stage2._stage2_workspace_storage(
        config_identity.config,
        workspace,
        output_root=output_root,
    )
    _require_file(storage.run_path, root=workspace)
    run = storage.read_run()
    if (
        run.run_id != storage.root.name
        or run.git_commit != stage2.VISUAL_ALGORITHM_FREEZE
        or run.config_hash != config.fingerprint()
        or run.model_identifiers != config.model_identifiers()
        or run.source_manifest_path != str(config.dataset_json.resolve(strict=False))
    ):
        raise ValueError("Stage2 run.json identity mismatch")
    clip_uid = str(value.clip_uid)
    clip_dir = storage.clip_dir(clip_uid)
    _require_file(storage.clip_path(clip_uid), root=clip_dir)
    clip = storage.read_clip(clip_uid)
    annotation = stage2._annotation_state(input_row)
    source = stage2._clip_source(
        input_row, Path(str(input_row["video_path"])).resolve(strict=False)
    )
    if (clip.clip_uid, clip.source, clip.annotation) != (clip_uid, source, annotation):
        raise ValueError("Stage2 clip source/annotation identity mismatch")
    if value.annotation_entity_count != len(annotation.entities):
        raise ValueError("Stage2 row annotation entity count mismatch")
    # A terminal frame-build failure legitimately has no completed frames/masks.
    if value.status == "failed_frames":
        return

    _require_file(storage.frames_manifest_path(clip_uid), root=clip_dir)
    frames = storage.read_frames(clip_uid)
    _require_file(storage.masks_path(clip_uid), root=clip_dir)
    masks = storage.read_masks(clip_uid)
    if frames.clip_uid != clip_uid or masks.clip_uid != clip_uid:
        raise ValueError("Stage2 frame/mask manifest clip identity mismatch")
    if (frames.width, frames.height) != (masks.width, masks.height):
        raise ValueError("Stage2 frame/mask manifest dimensions mismatch")
    for frame in frames.frames:
        # Stat/path checks only: never read/hash/decode the sampled JPEG bytes.
        _require_file(clip_dir / frame.image_path, root=clip_dir)
    coverage = stage2.build_coverage_state(
        artifact=masks,
        entities=annotation.entities,
        required_visible_frames=config.coverage.required_visible_frames,
    )
    if clip.coverage != coverage or value.coverage_passed != coverage.passed:
        raise ValueError("Stage2 row/stored coverage does not match mask metadata")
    counts = Counter(entity.status for entity in masks.entities.values())
    if (
        value.sam3_entity_ready,
        value.sam3_entity_not_found,
        value.sam3_entity_failed,
    ) != (counts["ready"], counts["not_found"], counts["failed"]):
        raise ValueError("Stage2 row does not match mask entity counts")
    background = clip.references.background
    if not coverage.passed:
        if (
            background is not None
            or value.background_status is not None
            or value.status != "coverage_rejected"
        ):
            raise ValueError("coverage-rejected Stage2 row has inconsistent background")
        return
    if background is None:
        raise ValueError("passed coverage is missing background state")
    expected_status = {
        "none": "ready_no_background",
        "rejected": "ready_background_rejected",
        "pending_remove": "ready_background_pending_remove",
    }.get(background.status)
    if value.status != expected_status or value.background_status != background.status:
        raise ValueError("Stage2 terminal row does not match background state")
    if background.source_image_path is not None:
        # This helper only compares frame slot/index/path metadata; no image I/O.
        _validate_source_frame(storage, clip_uid, background, frames)
    if background.output_image_path is not None:
        _require_file(
            _resolve_run_artifact(storage, background.output_image_path), root=clip_dir
        )
    for mask_path in (background.source_mask_path, background.generation_mask_path):
        if mask_path is not None:
            _require_file(
                _resolve_run_artifact(storage, mask_path), root=clip_dir / "background"
            )
