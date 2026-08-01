from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.schemas import (
    AnnotationState,
    BackgroundReferenceState,
    EntityReferenceState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage

_BACKGROUND_SLOT_PRIORITY = (5, 4, 6, 3, 7, 2, 8, 1, 9, 0)
_PRIORITY_INDEX = {slot: index for index, slot in enumerate(_BACKGROUND_SLOT_PRIORITY)}


@dataclass(frozen=True)
class BackgroundStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    none: int = 0
    clean_raw: int = 0
    pending_remove: int = 0
    rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BackgroundCandidate:
    frame: SampledFrame
    union_mask: np.ndarray
    area_pixels: int
    area_ratio: float


def validate_background_inputs(
    *,
    clip_uid: str,
    annotation: AnnotationState,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> None:
    if frames.clip_uid != clip_uid:
        raise ValueError("frames artifact clip_uid does not match its clip")
    if masks.clip_uid != clip_uid:
        raise ValueError("mask artifact clip_uid does not match its clip")
    if frames.sampled_frame_count != 10 or masks.sampled_frame_count != 10:
        raise ValueError("background inputs must contain exactly ten frame slots")
    if (frames.width, frames.height) != (masks.width, masks.height):
        raise ValueError("frames and masks dimensions do not match")
    if [frame.slot for frame in frames.frames] != list(range(10)):
        raise ValueError("frame slots must be unique and ordered from 0 through 9")
    expected_entities = [entity.entity_id for entity in annotation.entities]
    if list(masks.entities) != expected_entities:
        raise ValueError("mask artifact entity IDs do not match annotation order")
    expected_size = (frames.height, frames.width)
    for entity in annotation.entities:
        tracked = masks.entities[entity.entity_id]
        if (
            tracked.reference_type != entity.reference_type
            or tracked.grounding_prompt != entity.grounding_prompt
        ):
            raise ValueError("mask artifact entity semantics do not match annotation")
        if [frame.slot for frame in tracked.frames] != list(range(10)):
            raise ValueError(
                "tracked entity slots must be unique and ordered from 0 through 9"
            )
        if any(frame.rle.size != expected_size for frame in tracked.frames):
            raise ValueError("tracked mask dimensions do not match sampled frames")


def _decode_tracked_mask(
    *,
    encoded: object,
    expected_shape: tuple[int, int],
    area_pixels: int,
    area_ratio: float,
    present: bool,
) -> np.ndarray:
    decoded = np.asarray(decode_binary_mask(encoded))
    if decoded.ndim != 2 or decoded.shape != expected_shape:
        raise ValueError("decoded tracked mask dimensions do not match frames")
    if not np.isin(decoded, (False, True)).all():
        raise ValueError("decoded tracked mask must be binary")
    binary = decoded.astype(bool, copy=False)
    actual_area = int(np.count_nonzero(binary))
    actual_ratio = actual_area / (expected_shape[0] * expected_shape[1])
    if actual_area != area_pixels or not math.isclose(
        actual_ratio,
        area_ratio,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("tracked mask area diagnostics do not match decoded mask")
    if present != (actual_area > 0):
        raise ValueError("tracked mask presence does not match decoded mask")
    return binary


def build_union_foreground_mask(
    masks: TrackedMasksArtifact,
    slot: int,
) -> np.ndarray | None:
    if not 0 <= slot < 10:
        raise ValueError("background source slot must be between 0 and 9")
    expected_shape = (masks.height, masks.width)
    union_mask = np.zeros(expected_shape, dtype=bool)
    for tracked in masks.entities.values():
        if tracked.status != "ready":
            raise ValueError("union masks require complete ready entity tracking")
        frame = tracked.frames[slot]
        if frame.slot != slot:
            raise ValueError("tracked mask slot does not match its position")
        decoded = _decode_tracked_mask(
            encoded=frame.rle,
            expected_shape=expected_shape,
            area_pixels=frame.area_pixels,
            area_ratio=frame.area_ratio,
            present=frame.present,
        )
        if not frame.track_valid:
            return None
        if frame.present:
            union_mask = np.logical_or(union_mask, decoded)
    return union_mask


def select_background_source_frame(
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> BackgroundCandidate | None:
    candidates: list[BackgroundCandidate] = []
    total_pixels = frames.height * frames.width
    for frame in frames.frames:
        union_mask = build_union_foreground_mask(masks, frame.slot)
        if union_mask is None:
            continue
        area_pixels = int(np.count_nonzero(union_mask))
        candidates.append(
            BackgroundCandidate(
                frame=frame,
                union_mask=union_mask,
                area_pixels=area_pixels,
                area_ratio=area_pixels / total_pixels,
            )
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate.area_pixels,
            _PRIORITY_INDEX[candidate.frame.slot],
        ),
    )


def build_background_reference_state(
    *,
    candidate: BackgroundCandidate,
    max_pending_remove_area_ratio: float,
    source_mask_path: str | None = None,
) -> BackgroundReferenceState:
    common: dict[str, object] = {
        "source_image_path": candidate.frame.image_path,
        "source_frame_slot": candidate.frame.slot,
        "source_frame_index": candidate.frame.source_frame_index,
        "source_foreground_area_pixels": candidate.area_pixels,
        "source_foreground_area_ratio": candidate.area_ratio,
    }
    if candidate.area_pixels == 0:
        return BackgroundReferenceState(
            status="clean_raw",
            output_image_path=candidate.frame.image_path,
            **common,
        )
    if source_mask_path is None:
        raise ValueError("non-empty background candidates require a source mask")
    if candidate.area_ratio <= max_pending_remove_area_ratio:
        return BackgroundReferenceState(
            status="pending_remove",
            source_mask_path=source_mask_path,
            **common,
        )
    return BackgroundReferenceState(
        status="rejected",
        source_mask_path=source_mask_path,
        reason="foreground_mask_too_large",
        **common,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_mask_png(
    path: Path,
    *,
    width: int,
    height: int,
    expected_area: int,
    expected_mask: np.ndarray | None = None,
) -> np.ndarray:
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("background source mask must be a PNG")
        if opened.size != (width, height):
            raise ValueError("background source mask dimensions do not match frame")
        pixels = np.asarray(opened.convert("L"))
    if pixels.ndim != 2 or not np.isin(pixels, (0, 255)).all():
        raise ValueError("background source mask must contain only 0 and 255")
    binary = pixels == 255
    if int(np.count_nonzero(binary)) != expected_area:
        raise ValueError("background source mask area does not match state")
    if expected_mask is not None and not np.array_equal(binary, expected_mask):
        raise ValueError("background source mask does not match exact union mask")
    return binary


def _write_source_mask_atomic(
    storage: RunStorage,
    clip_uid: str,
    mask: np.ndarray,
) -> tuple[Path, bool]:
    directory = storage.prepare_background_publication(clip_uid)
    temporary = directory / f".tmp-{uuid.uuid4().hex}.png"
    area_pixels = int(np.count_nonzero(mask))
    try:
        with temporary.open("wb") as handle:
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                handle,
                format="PNG",
            )
            handle.flush()
            os.fsync(handle.fileno())
        _validate_mask_png(
            temporary,
            width=mask.shape[1],
            height=mask.shape[0],
            expected_area=area_pixels,
            expected_mask=mask,
        )
        digest = _sha256(temporary)
        destination = storage.background_source_mask_path(clip_uid, digest)
        existed = destination.is_file()
        if existed:
            if temporary.read_bytes() != destination.read_bytes():
                raise ValueError("background source mask hash collision")
            temporary.unlink()
        else:
            temporary.replace(destination)
        _validate_mask_png(
            destination,
            width=mask.shape[1],
            height=mask.shape[0],
            expected_area=area_pixels,
            expected_mask=mask,
        )
        return destination, not existed
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_run_artifact(storage: RunStorage, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        raise ValueError("background artifact paths must be relative to run_root")
    resolved = (storage.root / path).resolve(strict=False)
    try:
        resolved.relative_to(storage.root.resolve())
    except ValueError as exc:
        raise ValueError("background artifact must remain inside run_root") from exc
    return resolved


def _validate_source_frame(
    storage: RunStorage,
    clip_uid: str,
    state: BackgroundReferenceState,
    frames: SampledFramesArtifact,
) -> SampledFrame:
    if state.source_frame_slot is None:
        raise ValueError("background source frame slot is missing")
    frame = frames.frames[state.source_frame_slot]
    if (
        frame.slot != state.source_frame_slot
        or frame.source_frame_index != state.source_frame_index
    ):
        raise ValueError("background source frame provenance does not match frames")
    run_relative = storage.relative_artifact_path(
        storage.clip_dir(clip_uid) / frame.image_path
    )
    if state.source_image_path not in {frame.image_path, run_relative}:
        raise ValueError("background source image path does not match frames manifest")
    return frame


def _validate_ready_removed(
    storage: RunStorage,
    clip_uid: str,
    state: BackgroundReferenceState,
    *,
    frames: SampledFramesArtifact,
    frame: SampledFrame,
    source_mask: np.ndarray,
) -> None:
    from r2v_data_v2.v3.remove import build_generation_mask

    if (
        state.output_image_path is None
        or state.generation_mask_path is None
        or state.generation_mask_area_pixels is None
        or state.generation_mask_area_ratio is None
        or state.generation_mask_dilation_pixels is None
        or state.output_sha256 is None
    ):
        raise ValueError("ready removed background metadata is incomplete")

    output_path = _resolve_run_artifact(storage, state.output_image_path)
    expected_output = (
        storage.clip_dir(clip_uid) / "selected" / "bg_removed.png"
    ).resolve(strict=False)
    if output_path != expected_output:
        raise ValueError("ready removed output path must be selected/bg_removed.png")
    if not output_path.is_file():
        raise FileNotFoundError("ready removed background output is missing")
    if _sha256(output_path) != state.output_sha256:
        raise ValueError("ready removed background output hash is invalid")

    generation_path = _resolve_run_artifact(storage, state.generation_mask_path)
    background_dir = storage.background_dir(clip_uid).resolve(strict=False)
    if generation_path.parent != background_dir:
        raise ValueError("background generation mask is outside background_dir")
    prefix = "generation_mask_"
    if (
        not generation_path.name.startswith(prefix)
        or not generation_path.name.endswith(".png")
        or _sha256(generation_path)
        != generation_path.name[len(prefix) : -len(".png")]
    ):
        raise ValueError("background generation mask content hash is invalid")
    generation_mask = _validate_mask_png(
        generation_path,
        width=frames.width,
        height=frames.height,
        expected_area=state.generation_mask_area_pixels,
    )
    expected_ratio = state.generation_mask_area_pixels / (
        frames.width * frames.height
    )
    if not math.isclose(
        state.generation_mask_area_ratio,
        expected_ratio,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("background generation mask ratio does not match state")
    if not np.all(generation_mask[source_mask]):
        raise ValueError("background generation mask does not contain source mask")
    expected_generation = build_generation_mask(
        source_mask,
        dilation_pixels=state.generation_mask_dilation_pixels,
    )
    if not np.array_equal(generation_mask, expected_generation):
        raise ValueError(
            "background generation mask does not match recorded dilation"
        )

    source_path = (storage.clip_dir(clip_uid) / frame.image_path).resolve(
        strict=False
    )
    if not source_path.is_file():
        raise FileNotFoundError("ready removed source image is missing")
    with Image.open(source_path) as opened:
        opened.load()
        source = np.asarray(opened.convert("RGB"))
    with Image.open(output_path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("ready removed background output must be a PNG")
        if opened.size != (frames.width, frames.height):
            raise ValueError("ready removed background output dimensions are invalid")
        output = np.asarray(opened.convert("RGB"))
    if not np.array_equal(output[~generation_mask], source[~generation_mask]):
        raise ValueError(
            "ready removed background changed pixels outside generation mask"
        )
    if not np.any(output[generation_mask] != source[generation_mask]):
        raise ValueError(
            "ready removed background did not change the generation mask"
        )


def validate_background_reference(
    storage: RunStorage,
    clip_uid: str,
    state: BackgroundReferenceState,
    *,
    frames: SampledFramesArtifact | None = None,
) -> None:
    if state.status == "none":
        return
    validated_frames = frames or validate_sampled_frames(storage, clip_uid)
    if state.status == "rejected" and state.source_image_path is None:
        return
    frame = _validate_source_frame(storage, clip_uid, state, validated_frames)
    source_mask: np.ndarray | None = None
    if state.status == "clean_raw":
        return
    if state.source_mask_path is not None:
        mask_path = _resolve_run_artifact(storage, state.source_mask_path)
        background_dir = storage.background_dir(clip_uid).resolve(strict=False)
        if mask_path.parent != background_dir:
            raise ValueError("background source mask is outside background_dir")
        prefix = "source_mask_"
        if (
            not mask_path.name.startswith(prefix)
            or not mask_path.name.endswith(".png")
            or _sha256(mask_path) != mask_path.name[len(prefix) : -len(".png")]
        ):
            raise ValueError("background source mask content hash is invalid")
        if state.source_foreground_area_pixels is None:
            raise ValueError("background source mask is missing area diagnostics")
        source_mask = _validate_mask_png(
            mask_path,
            width=validated_frames.width,
            height=validated_frames.height,
            expected_area=state.source_foreground_area_pixels,
        )
        expected_ratio = state.source_foreground_area_pixels / (
            validated_frames.width * validated_frames.height
        )
        if state.source_foreground_area_ratio is None or not math.isclose(
            state.source_foreground_area_ratio,
            expected_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("background source mask ratio does not match state")
    if state.status == "ready_removed":
        if source_mask is None:
            raise ValueError("ready removed background source mask is missing")
        _validate_ready_removed(
            storage,
            clip_uid,
            state,
            frames=validated_frames,
            frame=frame,
            source_mask=source_mask,
        )


def _publish_state(
    storage: RunStorage,
    clip_uid: str,
    entities: list[EntityReferenceState],
    state: BackgroundReferenceState,
    *,
    keep_mask: Path | None = None,
) -> None:
    references = ReferencesState(
        entities=entities,
        background=state,
    )
    storage.write_references(clip_uid, references)
    storage.cleanup_background_artifacts(clip_uid, keep=keep_mask)


def build_background_candidates(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
) -> BackgroundStats:
    counters = {
        "processed": 0,
        "skipped_existing": 0,
        "skipped_not_ready": 0,
        "failed": 0,
        "none": 0,
        "clean_raw": 0,
        "pending_remove": 0,
        "rejected": 0,
    }
    for clip in storage.iter_clips():
        annotation = clip.annotation
        if (
            annotation is None
            or annotation.status != "ready"
            or clip.coverage is None
            or not clip.coverage.passed
        ):
            counters["skipped_not_ready"] += 1
            continue
        existing = clip.references.background
        if existing is not None and not overwrite:
            try:
                validate_background_reference(storage, clip.clip_uid, existing)
            except Exception as exc:  # noqa: BLE001 - isolate corrupt clip artifacts
                storage.append_failure(
                    clip_uid=clip.clip_uid,
                    stage="background",
                    reason=str(exc),
                )
                counters["failed"] += 1
            else:
                counters["skipped_existing"] += 1
            continue
        created_mask: Path | None = None
        try:
            if not config.background.enabled:
                state = BackgroundReferenceState(
                    status="none",
                    reason="background_stage_disabled",
                )
                _publish_state(
                    storage,
                    clip.clip_uid,
                    list(clip.references.entities),
                    state,
                )
            elif annotation.background is None:
                state = BackgroundReferenceState(
                    status="none",
                    reason="annotation_has_no_background",
                )
                _publish_state(
                    storage,
                    clip.clip_uid,
                    list(clip.references.entities),
                    state,
                )
            else:
                frames = validate_sampled_frames(storage, clip.clip_uid)
                masks = storage.read_masks(clip.clip_uid)
                validate_background_inputs(
                    clip_uid=clip.clip_uid,
                    annotation=annotation,
                    frames=frames,
                    masks=masks,
                )
                if any(
                    tracked.status in {"not_found", "failed"}
                    for tracked in masks.entities.values()
                ):
                    state = BackgroundReferenceState(
                        status="rejected",
                        reason="incomplete_foreground_tracking",
                    )
                    _publish_state(
                        storage,
                        clip.clip_uid,
                        list(clip.references.entities),
                        state,
                    )
                else:
                    candidate = select_background_source_frame(frames, masks)
                    if candidate is None:
                        state = BackgroundReferenceState(
                            status="rejected",
                            reason="no_valid_background_source_frame",
                        )
                        _publish_state(
                            storage,
                            clip.clip_uid,
                            list(clip.references.entities),
                            state,
                        )
                    elif candidate.area_pixels == 0:
                        state = build_background_reference_state(
                            candidate=candidate,
                            max_pending_remove_area_ratio=(
                                config.background.max_pending_remove_area_ratio
                            ),
                        )
                        _publish_state(
                            storage,
                            clip.clip_uid,
                            list(clip.references.entities),
                            state,
                        )
                    else:
                        mask_path, created = _write_source_mask_atomic(
                            storage,
                            clip.clip_uid,
                            candidate.union_mask,
                        )
                        created_mask = mask_path if created else None
                        state = build_background_reference_state(
                            candidate=candidate,
                            max_pending_remove_area_ratio=(
                                config.background.max_pending_remove_area_ratio
                            ),
                            source_mask_path=storage.relative_artifact_path(mask_path),
                        )
                        _publish_state(
                            storage,
                            clip.clip_uid,
                            list(clip.references.entities),
                            state,
                            keep_mask=mask_path,
                        )
            counters["processed"] += 1
            counters[state.status] += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-clip failures
            if created_mask is not None:
                created_relative = storage.relative_artifact_path(created_mask)
                try:
                    published = storage.read_clip(
                        clip.clip_uid
                    ).references.background
                except Exception:  # noqa: BLE001 - preserve uncertain artifacts
                    published = None
                    created_relative = None
                if (
                    created_relative is not None
                    and (
                        published is None
                        or published.source_mask_path != created_relative
                    )
                ):
                    created_mask.unlink(missing_ok=True)
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="background",
                reason=str(exc),
            )
            counters["failed"] += 1
    stats = BackgroundStats(**counters)
    storage.update_stage_counts("background", stats.to_dict())
    return stats
