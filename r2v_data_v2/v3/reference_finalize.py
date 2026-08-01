from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from r2v_data_v2.v3.config import ReferenceFinalizeConfig, V3Config
from r2v_data_v2.v3.schemas import (
    BackgroundReferenceState,
    ClipRecord,
    EntityReferenceState,
    FinalizedBackgroundReference,
    FinalizedEntityReference,
    ReferenceFinalizationState,
    ReferenceQualityFlag,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class EntityNormalizationResult:
    image: Image.Image
    source_width: int
    source_height: int
    source_content_bbox_xyxy: tuple[int, int, int, int]
    normalized_content_bbox_xyxy: tuple[int, int, int, int]
    source_content_width: int
    source_content_height: int
    normalized_content_width: int
    normalized_content_height: int
    source_foreground_area_pixels: int
    normalized_foreground_area_pixels: int
    source_foreground_ratio: float
    normalized_foreground_ratio: float
    scale_factor: float
    source_border_contact_count: int
    low_resolution: bool


@dataclass(frozen=True)
class ReferenceFinalizeStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    entities_finalized: int = 0
    entities_tier_a: int = 0
    entities_tier_b: int = 0
    entities_rejected: int = 0
    backgrounds_validated: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("entity reference alpha must not be empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _validate_normalization_arguments(
    *,
    canvas_size: int,
    content_max_side: int,
    max_upscale: float,
) -> None:
    if (
        not isinstance(canvas_size, int)
        or isinstance(canvas_size, bool)
        or canvas_size < 1
    ):
        raise ValueError("canvas_size must be a positive integer")
    if (
        not isinstance(content_max_side, int)
        or isinstance(content_max_side, bool)
        or not 1 <= content_max_side <= canvas_size
    ):
        raise ValueError(
            "content_max_side must be a positive integer no greater than canvas_size"
        )
    if (
        not isinstance(max_upscale, (int, float))
        or isinstance(max_upscale, bool)
        or not math.isfinite(max_upscale)
        or max_upscale <= 0
    ):
        raise ValueError("max_upscale must be finite and positive")


def normalize_entity_reference_image(
    image: Image.Image,
    *,
    canvas_size: int,
    content_max_side: int,
    max_upscale: float,
) -> EntityNormalizationResult:
    if not isinstance(image, Image.Image):
        raise TypeError("entity reference must be a PIL image")
    _validate_normalization_arguments(
        canvas_size=canvas_size,
        content_max_side=content_max_side,
        max_upscale=max_upscale,
    )
    if image.mode != "RGBA":
        raise ValueError("entity reference must be RGBA")
    if image.width < 1 or image.height < 1:
        raise ValueError("entity reference dimensions must be positive")
    image.load()
    source = np.asarray(image, dtype=np.uint8)
    alpha = source[..., 3]
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("entity reference alpha must be binary")
    foreground = alpha == 255
    bbox = _content_bbox(foreground)
    if not np.all(source[..., :3][~foreground] == 255):
        raise ValueError("entity reference transparent RGB must be white")

    x1, y1, x2, y2 = bbox
    content_width = x2 - x1
    content_height = y2 - y1
    content_long_side = max(content_width, content_height)
    required_scale = content_max_side / content_long_side
    low_resolution = required_scale > max_upscale
    target_long_side = min(
        content_max_side,
        max(1, math.floor(content_long_side * max_upscale + 1e-12)),
    )
    scale_factor = target_long_side / content_long_side
    if content_width >= content_height:
        target_width = target_long_side
        target_height = max(
            1,
            math.floor(content_height * scale_factor + 0.5),
        )
    else:
        target_height = target_long_side
        target_width = max(
            1,
            math.floor(content_width * scale_factor + 0.5),
        )

    cropped = image.crop(bbox)
    if cropped.size == (target_width, target_height):
        resized = cropped.copy()
    else:
        resized = cropped.resize(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS,
        )
    resized_pixels = np.asarray(resized, dtype=np.uint8).copy()
    resized_foreground = resized_pixels[..., 3] >= 128
    if not np.any(resized_foreground):
        raise ValueError("normalized entity alpha became empty")
    resized_pixels[..., 3] = np.where(
        resized_foreground,
        255,
        0,
    ).astype(np.uint8)
    resized_pixels[..., :3][~resized_foreground] = 255

    left = (canvas_size - target_width) // 2
    top = (canvas_size - target_height) // 2
    canvas = np.full((canvas_size, canvas_size, 4), 255, dtype=np.uint8)
    canvas[..., 3] = 0
    canvas[
        top : top + target_height,
        left : left + target_width,
    ] = resized_pixels
    normalized_foreground = canvas[..., 3] == 255
    normalized_bbox = _content_bbox(normalized_foreground)
    nx1, ny1, nx2, ny2 = normalized_bbox
    source_area = int(np.count_nonzero(foreground))
    normalized_area = int(np.count_nonzero(normalized_foreground))
    border_contact_count = sum(
        (
            bool(np.any(foreground[0, :])),
            bool(np.any(foreground[-1, :])),
            bool(np.any(foreground[:, 0])),
            bool(np.any(foreground[:, -1])),
        )
    )
    return EntityNormalizationResult(
        image=Image.fromarray(canvas, mode="RGBA"),
        source_width=image.width,
        source_height=image.height,
        source_content_bbox_xyxy=bbox,
        normalized_content_bbox_xyxy=normalized_bbox,
        source_content_width=content_width,
        source_content_height=content_height,
        normalized_content_width=nx2 - nx1,
        normalized_content_height=ny2 - ny1,
        source_foreground_area_pixels=source_area,
        normalized_foreground_area_pixels=normalized_area,
        source_foreground_ratio=source_area / (image.width * image.height),
        normalized_foreground_ratio=(
            normalized_area / (canvas_size * canvas_size)
        ),
        scale_factor=scale_factor,
        source_border_contact_count=border_contact_count,
        low_resolution=low_resolution,
    )


def _quality_flags(
    reference: EntityReferenceState,
    result: EntityNormalizationResult | None,
) -> list[ReferenceQualityFlag]:
    flags: list[ReferenceQualityFlag] = []
    if reference.reference_scope == "local":
        flags.append("local_reference")
    if reference.visible_region != "whole":
        flags.append("non_whole_visible_region")
    if result is not None and result.low_resolution:
        flags.append("low_resolution")
    if result is not None and result.source_border_contact_count:
        flags.append("border_contact")
    return flags


def _resolve_entity_source(
    storage: RunStorage,
    clip_uid: str,
    reference: EntityReferenceState,
) -> Path:
    if reference.image_path is None:
        raise ValueError("ready entity reference is missing image_path")
    value = Path(reference.image_path).expanduser()
    candidate = value if value.is_absolute() else storage.root / value
    resolved = candidate.resolve(strict=False)
    root = storage.root.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("entity reference image must remain inside run_root") from exc
    expected = storage.selected_entity_path(
        clip_uid,
        reference.entity_id,
    ).resolve(strict=False)
    if resolved != expected:
        raise ValueError("entity reference image must be selected/eN.png")
    if not resolved.is_file():
        raise FileNotFoundError(f"entity reference image is missing: {resolved}")
    return resolved


def _resolve_background_source(
    storage: RunStorage,
    clip_uid: str,
    background: BackgroundReferenceState,
) -> Path:
    if background.output_image_path is None:
        raise ValueError("ready background is missing output_image_path")
    value = Path(background.output_image_path).expanduser()
    if not value.is_absolute() and value.parts[:1] == ("frames",):
        candidate = storage.clip_dir(clip_uid) / value
    else:
        candidate = value if value.is_absolute() else storage.root / value
    resolved = candidate.resolve(strict=False)
    root = storage.root.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("background image must remain inside run_root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"background image is missing: {resolved}")
    return resolved


def _entity_metadata(
    *,
    config: ReferenceFinalizeConfig,
    storage: RunStorage,
    reference: EntityReferenceState,
    token: str,
    source_path: Path,
    normalized_path: Path,
    source_sha256: str,
    normalized_sha256: str,
    result: EntityNormalizationResult,
) -> FinalizedEntityReference:
    flags = _quality_flags(reference, result)
    quality_tier = "A" if not flags else "B"
    return FinalizedEntityReference(
        entity_id=reference.entity_id,
        token=token,
        status="ready",
        source_image_path=storage.relative_artifact_path(source_path),
        normalized_image_path=storage.relative_artifact_path(normalized_path),
        source_width=result.source_width,
        source_height=result.source_height,
        normalized_width=config.entity_canvas_size,
        normalized_height=config.entity_canvas_size,
        source_content_bbox_xyxy=result.source_content_bbox_xyxy,
        normalized_content_bbox_xyxy=result.normalized_content_bbox_xyxy,
        source_content_width=result.source_content_width,
        source_content_height=result.source_content_height,
        normalized_content_width=result.normalized_content_width,
        normalized_content_height=result.normalized_content_height,
        source_foreground_area_pixels=result.source_foreground_area_pixels,
        normalized_foreground_area_pixels=(
            result.normalized_foreground_area_pixels
        ),
        source_foreground_ratio=result.source_foreground_ratio,
        normalized_foreground_ratio=result.normalized_foreground_ratio,
        scale_factor=result.scale_factor,
        source_border_contact_count=result.source_border_contact_count,
        reference_scope=reference.reference_scope,
        visible_region=reference.visible_region,
        whole_entity_recognizable=reference.whole_entity_recognizable,
        identity_features_visible=reference.identity_features_visible,
        quality_tier=quality_tier,
        quality_flags=flags,
        normalization_profile=config.normalization_profile,
        source_sha256=source_sha256,
        normalized_sha256=normalized_sha256,
    )




def _validate_normalized_artifact(
    path: Path,
    metadata: FinalizedEntityReference,
    config: ReferenceFinalizeConfig,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"normalized entity image is missing: {path}")
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("normalized entity reference must be PNG")
        if opened.mode != "RGBA":
            raise ValueError("normalized entity reference must be RGBA")
        if opened.size != (
            config.entity_canvas_size,
            config.entity_canvas_size,
        ):
            raise ValueError("normalized entity reference has incorrect dimensions")
        pixels = np.asarray(opened, dtype=np.uint8)
    alpha = pixels[..., 3]
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("normalized entity alpha must be binary")
    foreground = alpha == 255
    if not np.any(foreground):
        raise ValueError("normalized entity alpha must not be empty")
    if not np.all(pixels[..., :3][~foreground] == 255):
        raise ValueError("normalized entity transparent RGB must be white")
    bbox = _content_bbox(foreground)
    area = int(np.count_nonzero(foreground))
    if metadata.normalized_content_bbox_xyxy != bbox:
        raise ValueError("normalized entity bbox does not match metadata")
    if metadata.normalized_foreground_area_pixels != area:
        raise ValueError("normalized entity area does not match metadata")
    expected_ratio = area / (pixels.shape[0] * pixels.shape[1])
    if not math.isclose(
        metadata.normalized_foreground_ratio or -1,
        expected_ratio,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("normalized entity ratio does not match metadata")
    if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > config.entity_content_max_side:
        raise ValueError("normalized entity content exceeds configured maximum")
    if metadata.scale_factor is None or (
        metadata.scale_factor > config.entity_max_upscale + 1e-12
    ):
        raise ValueError("normalized entity scale exceeds configured maximum")
    if metadata.normalization_profile != config.normalization_profile:
        raise ValueError("normalized entity profile does not match configuration")
    if metadata.normalized_sha256 != _sha256(path):
        raise ValueError("normalized entity SHA256 does not match metadata")


def _finalize_entity(
    config: ReferenceFinalizeConfig,
    storage: RunStorage,
    *,
    clip_uid: str,
    reference: EntityReferenceState,
    token: str,
) -> tuple[FinalizedEntityReference, Path]:
    source_path = _resolve_entity_source(storage, clip_uid, reference)
    source_sha256 = _sha256(source_path)
    with Image.open(source_path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("entity reference must be a PNG")
        if opened.mode != "RGBA":
            raise ValueError("entity reference must be RGBA")
        source_image = opened.copy()
    result = normalize_entity_reference_image(
        source_image,
        canvas_size=config.entity_canvas_size,
        content_max_side=config.entity_content_max_side,
        max_upscale=config.entity_max_upscale,
    )
    normalized_path = storage.finalized_entity_path(
        clip_uid,
        reference.entity_id,
    )
    temporary = storage.finalize_output_temporary_path(
        clip_uid,
        reference.entity_id,
    )
    try:
        result.image.save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        metadata = _entity_metadata(
            config=config,
            storage=storage,
            reference=reference,
            token=token,
            source_path=source_path,
            normalized_path=normalized_path,
            source_sha256=source_sha256,
            normalized_sha256=_sha256(temporary),
            result=result,
        )
        _validate_normalized_artifact(temporary, metadata, config)
        if _sha256(source_path) != source_sha256:
            raise RuntimeError("source entity reference changed during finalization")
        return metadata, temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _rejected_entity(
    config: ReferenceFinalizeConfig,
    storage: RunStorage,
    *,
    clip_uid: str,
    reference: EntityReferenceState,
    token: str,
    reason: str,
) -> FinalizedEntityReference:
    source_path = storage.selected_entity_path(clip_uid, reference.entity_id)
    source_image_path = reference.image_path or storage.relative_artifact_path(
        source_path
    )
    return FinalizedEntityReference(
        entity_id=reference.entity_id,
        token=token,
        status="rejected",
        source_image_path=source_image_path,
        reference_scope=reference.reference_scope,
        visible_region=reference.visible_region,
        whole_entity_recognizable=reference.whole_entity_recognizable,
        identity_features_visible=reference.identity_features_visible,
        quality_tier="reject",
        quality_flags=_quality_flags(reference, None),
        normalization_profile=config.normalization_profile,
        reason=reason,
    )


def _background_metadata(
    storage: RunStorage,
    *,
    clip_uid: str,
    background: BackgroundReferenceState,
    token: str,
) -> FinalizedBackgroundReference:
    if background.status not in {"clean_raw", "ready_removed"}:
        raise ValueError("paired background is not ready")
    path = _resolve_background_source(storage, clip_uid, background)
    before = _sha256(path)
    with Image.open(path) as opened:
        opened.load()
        if opened.mode != "RGB":
            raise ValueError("ready background must remain RGB")
        width, height = opened.size
    if width < 1 or height < 1:
        raise ValueError("ready background dimensions must be positive")
    if _sha256(path) != before:
        raise RuntimeError("background reference changed during validation")
    if background.source_frame_index is None:
        raise ValueError("ready background is missing source_frame_index")
    return FinalizedBackgroundReference(
        token=token,
        image_path=storage.relative_artifact_path(path),
        width=width,
        height=height,
        mode="RGB",
        sha256=before,
        source_frame_index=background.source_frame_index,
        synthetic=background.status == "ready_removed",
    )


def _validate_existing_finalization(
    config: ReferenceFinalizeConfig,
    storage: RunStorage,
    clip: ClipRecord,
) -> None:
    state = clip.reference_finalization
    if state is None or state.status != "ready":
        raise ValueError("existing reference finalization is not ready")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise ValueError("existing reference finalization requires ready pairing")
    references = {
        reference.entity_id: reference
        for reference in clip.references.entities
        if reference.status == "ready"
    }
    for metadata in state.entities:
        reference = references[metadata.entity_id]
        source_path = _resolve_entity_source(
            storage,
            clip.clip_uid,
            reference,
        )
        source_sha256 = _sha256(source_path)
        if source_sha256 != metadata.source_sha256:
            raise ValueError("source entity SHA256 does not match finalization")
        with Image.open(source_path) as opened:
            opened.load()
            if opened.format != "PNG" or opened.mode != "RGBA":
                raise ValueError("source entity artifact contract changed")
            source_image = opened.copy()
        result = normalize_entity_reference_image(
            source_image,
            canvas_size=config.entity_canvas_size,
            content_max_side=config.entity_content_max_side,
            max_upscale=config.entity_max_upscale,
        )
        normalized_path = storage.finalized_entity_path(
            clip.clip_uid,
            metadata.entity_id,
        )
        expected = _entity_metadata(
            config=config,
            storage=storage,
            reference=reference,
            token=clip.pairing.tokens[metadata.entity_id],
            source_path=source_path,
            normalized_path=normalized_path,
            source_sha256=source_sha256,
            normalized_sha256=_sha256(normalized_path),
            result=result,
        )
        if metadata != expected:
            raise ValueError("existing entity finalization metadata is stale")
        _validate_normalized_artifact(normalized_path, metadata, config)
    expected_files = {
        f"{entity.entity_id}.png" for entity in state.entities
    }
    actual_files = {
        path.name
        for path in storage.finalized_dir(clip.clip_uid).glob("e*.png")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("finalized entity files do not match metadata")
    if clip.pairing.background_token is not None:
        background = clip.references.background
        if background is None or state.background is None:
            raise ValueError("existing background finalization is missing")
        expected_background = _background_metadata(
            storage,
            clip_uid=clip.clip_uid,
            background=background,
            token=clip.pairing.background_token,
        )
        if state.background != expected_background:
            raise ValueError("existing background finalization metadata is stale")
    elif state.background is not None:
        raise ValueError("unpaired background finalization is not allowed")


def _publish_finalization(
    storage: RunStorage,
    *,
    clip_uid: str,
    state: ReferenceFinalizationState,
    temporary_images: dict[str, Path],
) -> None:
    directory = storage.finalized_dir(clip_uid)
    directory.mkdir(parents=True, exist_ok=True)
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for existing in directory.glob("e*.png"):
            if not existing.is_file():
                continue
            backup = storage.finalize_output_backup_path(
                clip_uid,
                existing.stem,
            )
            existing.replace(backup)
            backups[existing] = backup
        for entity_id, temporary in temporary_images.items():
            destination = storage.finalized_entity_path(clip_uid, entity_id)
            temporary.replace(destination)
            published.append(destination)
        storage.write_reference_finalization(clip_uid, state)
    except Exception:
        for destination in published:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for temporary in temporary_images.values():
            temporary.unlink(missing_ok=True)
        storage.cleanup_reference_finalize_artifacts(clip_uid)


def _failure_details(exc: Exception) -> dict[str, object]:
    return {"exception_type": type(exc).__name__}


def finalize_references(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
) -> ReferenceFinalizeStats:
    config.validate()
    if storage.root != config.resolved_run_root:
        raise ValueError(
            "storage run_root does not match reference_finalize configuration"
        )
    if not config.reference_finalize.enabled:
        raise ValueError("V3 reference_finalize stage is disabled")
    clips = list(storage.iter_clips())
    if not clips:
        raise FileNotFoundError(
            "reference_finalize requires manifest stage to create clip.json records first"
        )
    counters = {
        field: 0 for field in ReferenceFinalizeStats.__dataclass_fields__
    }
    for initial_clip in clips:
        clip = storage.read_clip(initial_clip.clip_uid)
        if clip.pairing is None or clip.pairing.status != "ready":
            counters["skipped_not_ready"] += 1
            continue
        if clip.reference_finalization is not None and not overwrite:
            try:
                _validate_existing_finalization(
                    config.reference_finalize,
                    storage,
                    clip,
                )
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clips
                storage.append_failure(
                    stage="reference_finalize",
                    clip_uid=clip.clip_uid,
                    reason=str(exc),
                    details=_failure_details(exc),
                )
                counters["failed"] += 1
            else:
                counters["skipped_existing"] += 1
            continue

        counters["processed"] += 1
        temporary_images: dict[str, Path] = {}
        try:
            ready_references = {
                reference.entity_id: reference
                for reference in clip.references.entities
                if reference.status == "ready"
            }
            entity_metadata: list[FinalizedEntityReference] = []
            rejected_ids: list[str] = []
            for entity_id in clip.pairing.retained_entity_ids:
                reference = ready_references[entity_id]
                token = clip.pairing.tokens[entity_id]
                try:
                    metadata, temporary = _finalize_entity(
                        config.reference_finalize,
                        storage,
                        clip_uid=clip.clip_uid,
                        reference=reference,
                        token=token,
                    )
                except (OSError, ValueError) as exc:
                    metadata = _rejected_entity(
                        config.reference_finalize,
                        storage,
                        clip_uid=clip.clip_uid,
                        reference=reference,
                        token=token,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    rejected_ids.append(entity_id)
                else:
                    temporary_images[entity_id] = temporary
                entity_metadata.append(metadata)

            background_metadata = None
            if clip.pairing.background_token is not None:
                background = clip.references.background
                if background is None:
                    raise ValueError("paired background metadata is missing")
                background_metadata = _background_metadata(
                    storage,
                    clip_uid=clip.clip_uid,
                    background=background,
                    token=clip.pairing.background_token,
                )
            state = ReferenceFinalizationState(
                status="failed" if rejected_ids else "ready",
                entities=entity_metadata,
                background=background_metadata,
                reason=(
                    "entity_reference_rejected"
                    if rejected_ids
                    else None
                ),
            )
            _publish_finalization(
                storage,
                clip_uid=clip.clip_uid,
                state=state,
                temporary_images=temporary_images,
            )
            for entity in entity_metadata:
                if entity.status == "rejected":
                    counters["entities_rejected"] += 1
                else:
                    counters["entities_finalized"] += 1
                    counters[
                        "entities_tier_a"
                        if entity.quality_tier == "A"
                        else "entities_tier_b"
                    ] += 1
            if background_metadata is not None:
                counters["backgrounds_validated"] += 1
            if rejected_ids:
                counters["failed"] += 1
                storage.append_failure(
                    stage="reference_finalize",
                    clip_uid=clip.clip_uid,
                    reason="entity_reference_rejected",
                    details={"rejected_entity_ids": rejected_ids},
                )
        except Exception as exc:  # noqa: BLE001 - continue with later clips
            for temporary in temporary_images.values():
                temporary.unlink(missing_ok=True)
            failed_state = ReferenceFinalizationState(
                status="failed",
                reason=str(exc),
            )
            failure_details = _failure_details(exc)
            try:
                storage.write_reference_finalization(
                    clip.clip_uid,
                    failed_state,
                )
                storage.cleanup_reference_finalize_artifacts(
                    clip.clip_uid,
                    remove_published=True,
                )
            except Exception as persistence_exc:  # noqa: BLE001
                failure_details["state_persistence_error"] = (
                    f"{type(persistence_exc).__name__}: {persistence_exc}"
                )
            storage.append_failure(
                stage="reference_finalize",
                clip_uid=clip.clip_uid,
                reason=str(exc),
                details=failure_details,
            )
            counters["failed"] += 1
    stats = ReferenceFinalizeStats(**counters)
    storage.update_stage_counts("reference_finalize", stats.to_dict())
    return stats