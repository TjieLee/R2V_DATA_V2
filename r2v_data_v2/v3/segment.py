from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.mask_codec import (
    decode_binary_mask,
    encode_binary_mask,
)
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
    Sam3SegmentationBackend,
    SegmentationBackend,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    MaskRle,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class SegmentStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    entities_ready: int = 0
    entities_not_found: int = 0
    entities_failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CrossEntityDuplicateDecision:
    first_entity_id: str
    second_entity_id: str
    common_valid_frame_count: int
    high_overlap_frame_count: int
    high_overlap_frame_ratio: float
    median_mask_iou: float
    is_duplicate: bool
    winner_entity_id: str | None = None
    loser_entity_id: str | None = None


def _empty_rle(height: int, width: int) -> MaskRle:
    return encode_binary_mask(np.zeros((height, width), dtype=bool))


def _empty_frame(
    slot: int,
    height: int,
    width: int,
    *,
    track_valid: bool = True,
) -> TrackedMaskFrame:
    return TrackedMaskFrame(
        slot=slot,
        present=False,
        track_valid=track_valid,
        area_pixels=0,
        area_ratio=0.0,
        rle=_empty_rle(height, width),
    )


def _empty_frames(
    height: int,
    width: int,
    *,
    invalid_slots: set[int] | None = None,
) -> list[TrackedMaskFrame]:
    invalid = invalid_slots or set()
    return [
        _empty_frame(
            slot,
            height,
            width,
            track_valid=slot not in invalid,
        )
        for slot in range(10)
    ]


def _failed_entity(
    entity: AnnotationEntity,
    *,
    height: int,
    width: int,
    reason: str,
) -> TrackedEntityMasks:
    return TrackedEntityMasks(
        status="failed",
        reference_type=entity.reference_type,
        grounding_prompt=entity.grounding_prompt,
        frames=_empty_frames(height, width),
        reason=reason,
    )


def _valid_track_frames(track: TrackedEntityMasks) -> list[TrackedMaskFrame]:
    return [
        frame
        for frame in track.frames
        if frame.present and frame.track_valid and frame.area_pixels > 0
    ]


def _track_quality(
    track: TrackedEntityMasks,
    *,
    annotation_index: int,
) -> tuple[int, float, int]:
    frames = _valid_track_frames(track)
    confidences = [
        float(frame.confidence)
        for frame in frames
        if frame.confidence is not None
    ]
    median_confidence = float(np.median(confidences)) if confidences else -math.inf
    return len(frames), median_confidence, -annotation_index


def compare_cross_entity_tracks(
    first_entity_id: str,
    first: TrackedEntityMasks,
    second_entity_id: str,
    second: TrackedEntityMasks,
    *,
    first_annotation_index: int,
    second_annotation_index: int,
    minimum_common_frames: int = 3,
    high_overlap_iou: float = 0.80,
    median_duplicate_iou: float = 0.85,
    minimum_high_overlap_ratio: float = 0.75,
) -> CrossEntityDuplicateDecision:
    """Compare two published tracks without changing either artifact."""
    if first.reference_type == "group" or second.reference_type == "group":
        return CrossEntityDuplicateDecision(
            first_entity_id=first_entity_id,
            second_entity_id=second_entity_id,
            common_valid_frame_count=0,
            high_overlap_frame_count=0,
            high_overlap_frame_ratio=0.0,
            median_mask_iou=0.0,
            is_duplicate=False,
        )
    if first.status != "ready" or second.status != "ready":
        return CrossEntityDuplicateDecision(
            first_entity_id=first_entity_id,
            second_entity_id=second_entity_id,
            common_valid_frame_count=0,
            high_overlap_frame_count=0,
            high_overlap_frame_ratio=0.0,
            median_mask_iou=0.0,
            is_duplicate=False,
        )

    first_by_slot = {frame.slot: frame for frame in _valid_track_frames(first)}
    second_by_slot = {frame.slot: frame for frame in _valid_track_frames(second)}
    ious: list[float] = []
    for slot in sorted(first_by_slot.keys() & second_by_slot.keys()):
        first_mask = decode_binary_mask(first_by_slot[slot].rle)
        second_mask = decode_binary_mask(second_by_slot[slot].rle)
        intersection = int(np.count_nonzero(first_mask & second_mask))
        union = int(np.count_nonzero(first_mask | second_mask))
        if union > 0:
            ious.append(intersection / union)

    common_count = len(ious)
    high_count = sum(value >= high_overlap_iou for value in ious)
    high_ratio = high_count / common_count if common_count else 0.0
    median_iou = float(np.median(ious)) if ious else 0.0
    is_duplicate = (
        common_count >= minimum_common_frames
        and median_iou >= median_duplicate_iou
        and high_ratio >= minimum_high_overlap_ratio
    )
    winner: str | None = None
    loser: str | None = None
    if is_duplicate:
        first_quality = _track_quality(
            first,
            annotation_index=first_annotation_index,
        )
        second_quality = _track_quality(
            second,
            annotation_index=second_annotation_index,
        )
        if first_quality >= second_quality:
            winner, loser = first_entity_id, second_entity_id
        else:
            winner, loser = second_entity_id, first_entity_id
    return CrossEntityDuplicateDecision(
        first_entity_id=first_entity_id,
        second_entity_id=second_entity_id,
        common_valid_frame_count=common_count,
        high_overlap_frame_count=high_count,
        high_overlap_frame_ratio=high_ratio,
        median_mask_iou=median_iou,
        is_duplicate=is_duplicate,
        winner_entity_id=winner,
        loser_entity_id=loser,
    )


def _deduplicate_cross_entity_tracks(
    entities: list[AnnotationEntity],
    tracks: dict[str, TrackedEntityMasks],
    *,
    height: int,
    width: int,
) -> dict[str, TrackedEntityMasks]:
    annotation_index = {
        entity.entity_id: index for index, entity in enumerate(entities)
    }
    ranked_ids = sorted(
        (
            entity.entity_id
            for entity in entities
            if tracks[entity.entity_id].status == "ready"
            and entity.reference_type != "group"
        ),
        key=lambda entity_id: _track_quality(
            tracks[entity_id],
            annotation_index=annotation_index[entity_id],
        ),
        reverse=True,
    )
    entities_by_id = {entity.entity_id: entity for entity in entities}
    deduplicated = dict(tracks)
    for winner_position, winner_id in enumerate(ranked_ids):
        if deduplicated[winner_id].status != "ready":
            continue
        for loser_id in ranked_ids[winner_position + 1 :]:
            if deduplicated[loser_id].status != "ready":
                continue
            decision = compare_cross_entity_tracks(
                winner_id,
                deduplicated[winner_id],
                loser_id,
                deduplicated[loser_id],
                first_annotation_index=annotation_index[winner_id],
                second_annotation_index=annotation_index[loser_id],
            )
            if not decision.is_duplicate:
                continue
            assert decision.winner_entity_id == winner_id
            deduplicated[loser_id] = _failed_entity(
                entities_by_id[loser_id],
                height=height,
                width=width,
                reason=f"duplicate_cross_entity_track:{winner_id}",
            )
    return deduplicated


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    y_values, x_values = np.nonzero(mask)
    if len(x_values) == 0:
        raise ValueError("cannot calculate a bbox for an empty mask")
    return (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max()) + 1,
        int(y_values.max()) + 1,
    )


def _validate_observation(
    observation: BackendMaskObservation,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if not 0 <= observation.slot < 10:
        raise ValueError("backend observation slot is outside 0 through 9")
    if not observation.object_id:
        raise ValueError("backend observation object ID must not be empty")
    if not math.isfinite(observation.confidence):
        raise ValueError("backend observation confidence must be finite")
    mask = np.asarray(observation.mask)
    if mask.shape != (height, width):
        raise ValueError("backend mask dimensions do not match sampled frames")
    encode_binary_mask(mask)
    return mask.astype(bool, copy=False)


def _entity_masks_from_result(
    entity: AnnotationEntity,
    result: EntityTrackResult,
    *,
    height: int,
    width: int,
) -> TrackedEntityMasks:
    if result.status != "ready":
        return TrackedEntityMasks(
            status=result.status,
            reference_type=entity.reference_type,
            grounding_prompt=entity.grounding_prompt,
            frames=_empty_frames(height, width),
            reason=result.reason,
        )

    observations_by_slot: dict[int, list[tuple[BackendMaskObservation, np.ndarray]]] = {
        slot: [] for slot in range(10)
    }
    all_object_ids: set[str] = set()
    for observation in result.observations:
        mask = _validate_observation(
            observation,
            height=height,
            width=width,
        )
        observations_by_slot[observation.slot].append(
            (observation, mask)
        )
        all_object_ids.add(observation.object_id)

    if entity.reference_type != "group" and len(all_object_ids) > 1:
        return _failed_entity(
            entity,
            height=height,
            width=width,
            reason=(
                "multiple backend identities cannot be unioned for a "
                "subject or object"
            ),
        )
    if entity.reference_type == "group" and len(all_object_ids) > 1:
        return _failed_entity(
            entity,
            height=height,
            width=width,
            reason="unverified_multi_object_group",
        )

    frames: list[TrackedMaskFrame] = []
    for slot in range(10):
        observed = observations_by_slot[slot]
        if len({item.object_id for item, _ in observed}) != len(observed):
            return _failed_entity(
                entity,
                height=height,
                width=width,
                reason="backend returned duplicate object IDs in one frame",
            )
        if any(not item.valid for item, _ in observed):
            frames.append(
                _empty_frame(
                    slot,
                    height,
                    width,
                    track_valid=False,
                )
            )
            continue
        nonempty = [
            (item, mask) for item, mask in observed if mask.any()
        ]
        if not nonempty:
            frames.append(_empty_frame(slot, height, width))
            continue
        if entity.reference_type != "group" and len(nonempty) > 1:
            return _failed_entity(
                entity,
                height=height,
                width=width,
                reason=(
                    "multiple masks cannot be unioned for a subject or object"
                ),
            )
        ordered = sorted(nonempty, key=lambda value: value[0].object_id)
        combined = np.logical_or.reduce([mask for _, mask in ordered])
        object_ids = [item.object_id for item, _ in ordered]
        confidences = [item.confidence for item, _ in ordered]
        area_pixels = int(combined.sum())
        frames.append(
            TrackedMaskFrame(
                slot=slot,
                present=True,
                track_valid=True,
                confidence=min(confidences),
                backend_confidences=confidences,
                backend_object_ids=object_ids,
                area_pixels=area_pixels,
                area_ratio=area_pixels / (height * width),
                bbox_xyxy=_bbox_from_mask(combined),
                rle=encode_binary_mask(combined),
            )
        )

    present_frames = [frame for frame in frames if frame.present]
    if not present_frames:
        return TrackedEntityMasks(
            status="not_found",
            reference_type=entity.reference_type,
            grounding_prompt=entity.grounding_prompt,
            frames=frames,
            reason="backend returned no valid non-empty masks",
        )
    published_ids = sorted(
        {
            object_id
            for frame in present_frames
            for object_id in frame.backend_object_ids
        }
    )
    return TrackedEntityMasks(
        status="ready",
        reference_type=entity.reference_type,
        grounding_prompt=entity.grounding_prompt,
        backend_object_ids=published_ids,
        frames=frames,
    )


def _validate_existing_masks(
    storage: RunStorage,
    *,
    clip_uid: str,
    entities: list[AnnotationEntity],
) -> TrackedMasksArtifact:
    artifact = storage.read_masks(clip_uid)
    frames = validate_sampled_frames(storage, clip_uid)
    if artifact.clip_uid != clip_uid:
        raise ValueError("mask artifact clip_uid does not match its clip")
    if (artifact.width, artifact.height) != (frames.width, frames.height):
        raise ValueError("mask artifact dimensions do not match sampled frames")
    if list(artifact.entities) != [entity.entity_id for entity in entities]:
        raise ValueError("mask artifact entity IDs do not match annotation order")
    for entity in entities:
        masks = artifact.entities[entity.entity_id]
        if (
            masks.reference_type != entity.reference_type
            or masks.grounding_prompt != entity.grounding_prompt
        ):
            raise ValueError(
                "mask artifact entity semantics do not match annotation"
            )
    return artifact


def _overlay_text(
    entity_id: str,
    grounding_prompt: str,
    frame: TrackedMaskFrame,
) -> str:
    confidence = (
        "none" if frame.confidence is None else f"{frame.confidence:.4f}"
    )
    object_ids = ",".join(frame.backend_object_ids) or "none"
    return (
        f"{entity_id} | slot={frame.slot:02d} | present={frame.present} | "
        f"confidence={confidence} | area_ratio={frame.area_ratio:.6f} | "
        f"object_ids={object_ids}\n{grounding_prompt}"
    )


def _write_debug_overlays(
    storage: RunStorage,
    *,
    clip_uid: str,
    frame_paths: list[Path],
    artifact: TrackedMasksArtifact,
) -> None:
    destination = storage.segment_debug_dir(clip_uid)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for entity_id, entity in artifact.entities.items():
        rendered: list[Image.Image] = []
        for frame_path, frame_result in zip(frame_paths, entity.frames):
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
            mask = decode_binary_mask(frame_result.rle)
            if frame_result.present:
                color = Image.new("RGBA", image.size, (255, 40, 40, 0))
                alpha = Image.fromarray(mask.astype(np.uint8) * 90)
                color.putalpha(alpha)
                image = Image.alpha_composite(
                    image.convert("RGBA"),
                    color,
                ).convert("RGB")
            draw = ImageDraw.Draw(image)
            if frame_result.bbox_xyxy is not None:
                draw.rectangle(
                    frame_result.bbox_xyxy,
                    outline=(255, 255, 0),
                    width=2,
                )
            text = _overlay_text(
                entity_id,
                entity.grounding_prompt,
                frame_result,
            )
            draw.multiline_text(
                (5, 5),
                text,
                fill=(255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            image.save(
                destination / f"{entity_id}_{frame_result.slot:02d}.jpg",
                format="JPEG",
                quality=92,
            )
            rendered.append(image)

        thumb_width = 320
        thumb_height = max(
            1,
            round(thumb_width * artifact.height / artifact.width),
        )
        sheet = Image.new(
            "RGB",
            (thumb_width * 5, thumb_height * 2),
            (0, 0, 0),
        )
        for slot, image in enumerate(rendered):
            thumbnail = image.copy()
            thumbnail.thumbnail(
                (thumb_width, thumb_height),
                Image.Resampling.LANCZOS,
            )
            sheet.paste(
                thumbnail,
                (
                    (slot % 5) * thumb_width,
                    (slot // 5) * thumb_height,
                ),
            )
        sheet.save(
            destination / f"contact_sheet_{entity_id}.jpg",
            format="JPEG",
            quality=92,
        )


def _segment_clips_with_backend(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool,
    backend: SegmentationBackend,
) -> SegmentStats:
    processed = skipped_existing = skipped_not_ready = failed = 0
    ready_count = not_found_count = entity_failed_count = 0
    for clip in storage.iter_clips():
        annotation = clip.annotation
        if annotation is None or annotation.status != "ready":
            skipped_not_ready += 1
            continue
        destination = storage.masks_path(clip.clip_uid)
        if destination.is_file() and not overwrite:
            try:
                _validate_existing_masks(
                    storage,
                    clip_uid=clip.clip_uid,
                    entities=annotation.entities,
                )
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clip artifacts
                storage.append_failure(
                    clip_uid=clip.clip_uid,
                    stage="segment",
                    reason=str(exc),
                )
                failed += 1
            else:
                skipped_existing += 1
            continue

        try:
            frames = validate_sampled_frames(storage, clip.clip_uid)
            frame_paths = [
                storage.clip_dir(clip.clip_uid) / frame.image_path
                for frame in frames.frames
            ]
            storage.prepare_masks_publication(clip.clip_uid)
            tracked_entities: dict[str, TrackedEntityMasks] = {}
            for entity in annotation.entities:
                try:
                    result = backend.track(
                        frame_paths=frame_paths,
                        entity_id=entity.entity_id,
                        reference_type=entity.reference_type,
                        grounding_prompt=entity.grounding_prompt,
                    )
                    entity_masks = _entity_masks_from_result(
                        entity,
                        result,
                        height=frames.height,
                        width=frames.width,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate entity tracking
                    entity_masks = _failed_entity(
                        entity,
                        height=frames.height,
                        width=frames.width,
                        reason=str(exc),
                    )
                    storage.append_failure(
                        clip_uid=clip.clip_uid,
                        stage="segment",
                        reason=str(exc),
                        details={"entity_id": entity.entity_id},
                    )
                tracked_entities[entity.entity_id] = entity_masks

            tracked_entities = _deduplicate_cross_entity_tracks(
                annotation.entities,
                tracked_entities,
                height=frames.height,
                width=frames.width,
            )
            for entity_masks in tracked_entities.values():
                if entity_masks.status == "ready":
                    ready_count += 1
                elif entity_masks.status == "not_found":
                    not_found_count += 1
                else:
                    entity_failed_count += 1

            artifact = TrackedMasksArtifact(
                clip_uid=clip.clip_uid,
                height=frames.height,
                width=frames.width,
                entities=tracked_entities,
            )
            if (
                config.debug.save_diagnostics
                or config.sam3.save_debug_overlays
            ):
                _write_debug_overlays(
                    storage,
                    clip_uid=clip.clip_uid,
                    frame_paths=frame_paths,
                    artifact=artifact,
                )
            storage.write_masks(clip.clip_uid, artifact)
            processed += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-clip stage failures
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="segment",
                reason=str(exc),
            )
            failed += 1
    stats = SegmentStats(
        processed=processed,
        skipped_existing=skipped_existing,
        skipped_not_ready=skipped_not_ready,
        failed=failed,
        entities_ready=ready_count,
        entities_not_found=not_found_count,
        entities_failed=entity_failed_count,
    )
    storage.update_stage_counts("segment", stats.to_dict())
    return stats


def segment_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    backend: SegmentationBackend | None = None,
) -> SegmentStats:
    if backend is not None:
        return _segment_clips_with_backend(
            config,
            storage,
            overwrite=overwrite,
            backend=backend,
        )

    if config.sam3.model_path is None:
        raise ValueError(
            "sam3.model_path must be configured before segment runs"
        )
    owned_backend = Sam3SegmentationBackend(config.sam3)
    try:
        return _segment_clips_with_backend(
            config,
            storage,
            overwrite=overwrite,
            backend=owned_backend,
        )
    finally:
        owned_backend.close()
