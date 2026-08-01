from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import validate_background_reference
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
    EntityReferenceJudge,
    EntityReferenceJudgeFailure,
    QwenEntityReferenceJudge,
    validate_entity_reference_decision,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    ClipRecord,
    EntityReferenceState,
    PairingState,
    ReferencesState,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage

_SLOT_PRIORITY = (5, 4, 6, 3, 7, 2, 8, 1, 9, 0)
_SLOT_PRIORITY_INDEX = {
    slot: index for index, slot in enumerate(_SLOT_PRIORITY)
}
_ENTITY_PNG = re.compile(r"e[1-9]\d*\.png")


@dataclass(frozen=True)
class EntityReferenceCandidate:
    candidate_id: str
    entity_id: str
    frame_slot: int
    source_frame_index: int
    image_path: str
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area_pixels: int
    area_ratio: float
    bbox_fill_ratio: float
    border_contact_count: int
    normalized_center_distance: float


@dataclass(frozen=True)
class PairStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    ready: int = 0
    rejected: int = 0
    entities_ready: int = 0
    entities_rejected: int = 0
    backgrounds_bound: int = 0
    repaired: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_source_and_mask(
    source_image: Image.Image,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(source_image, Image.Image):
        raise TypeError("source_image must be a PIL image")
    source = np.asarray(source_image.convert("RGB"))
    binary = np.asarray(mask)
    if binary.ndim != 2 or binary.shape != source.shape[:2]:
        raise ValueError("mask dimensions must match source_image")
    if not np.isin(binary, (False, True, 0, 1)).all():
        raise ValueError("mask must be binary")
    binary = binary.astype(bool, copy=False)
    if not np.any(binary):
        raise ValueError("mask must not be empty")
    return source, binary


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError("cannot calculate a bbox for an empty mask")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def build_candidate_context_image(
    source_image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    source, binary = _validate_source_and_mask(source_image, mask)
    context = source.copy()
    context[~binary] = (context[~binary].astype(np.uint16) * 35 // 100).astype(
        np.uint8
    )
    bbox = _bbox_from_mask(binary)
    result = Image.fromarray(context, mode="RGB")
    draw = ImageDraw.Draw(result)
    x1, y1, x2, y2 = bbox
    border = (
        max(0, x1 - 1),
        max(0, y1 - 1),
        min(result.width - 1, x2),
        min(result.height - 1, y2),
    )
    draw.rectangle(border, outline=(255, 48, 48), width=2)
    restored = np.asarray(result).copy()
    restored[binary] = source[binary]
    return Image.fromarray(restored, mode="RGB")


def build_reference_crop(
    source_image: Image.Image,
    mask: np.ndarray,
    *,
    crop_padding_ratio: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if (
        not isinstance(crop_padding_ratio, float)
        or not math.isfinite(crop_padding_ratio)
        or not 0 <= crop_padding_ratio <= 0.5
    ):
        raise ValueError(
            "crop_padding_ratio must be a finite float between 0 and 0.5"
        )
    source, binary = _validate_source_and_mask(source_image, mask)
    x1, y1, x2, y2 = _bbox_from_mask(binary)
    padding = math.ceil(max(x2 - x1, y2 - y1) * crop_padding_ratio)
    crop_box = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(source_image.width, x2 + padding),
        min(source_image.height, y2 + padding),
    )
    left, top, right, bottom = crop_box
    source_crop = source[top:bottom, left:right]
    mask_crop = binary[top:bottom, left:right]
    rgba = np.full((*mask_crop.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = np.where(mask_crop, 255, 0).astype(np.uint8)
    rgba[..., :3][mask_crop] = source_crop[mask_crop]
    return Image.fromarray(rgba, mode="RGBA"), crop_box


def _decode_candidate_mask(
    tracked_frame: object,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    encoded = tracked_frame.rle
    if encoded.size != expected_shape:
        raise ValueError("tracked mask RLE size does not match sampled frames")
    decoded = np.asarray(decode_binary_mask(encoded))
    if decoded.shape != expected_shape:
        raise ValueError("decoded tracked mask dimensions do not match frames")
    binary = decoded.astype(bool, copy=False)
    area = int(np.count_nonzero(binary))
    area_ratio = area / (expected_shape[0] * expected_shape[1])
    if area != tracked_frame.area_pixels or not math.isclose(
        area_ratio,
        tracked_frame.area_ratio,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("tracked mask area diagnostics do not match decoded mask")
    if tracked_frame.present != (area > 0):
        raise ValueError("tracked mask presence does not match decoded mask")
    return binary


def _candidate_from_frame(
    *,
    storage: RunStorage,
    clip_uid: str,
    entity: AnnotationEntity,
    frame: SampledFrame,
    tracked_frame: object,
    frames: SampledFramesArtifact,
) -> EntityReferenceCandidate:
    if tracked_frame.slot != frame.slot:
        raise ValueError("tracked mask slot does not match sampled frame")
    binary = _decode_candidate_mask(
        tracked_frame,
        expected_shape=(frames.height, frames.width),
    )
    bbox = _bbox_from_mask(binary)
    if tracked_frame.bbox_xyxy != bbox:
        raise ValueError("tracked mask bbox does not match decoded mask")
    image_path = storage.frame_path(clip_uid, frame.slot)
    with Image.open(image_path) as opened:
        if opened.size != (frames.width, frames.height):
            raise ValueError("sampled frame image dimensions are invalid")
    area = int(np.count_nonzero(binary))
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_distance = math.hypot(
        center_x - frames.width / 2,
        center_y - frames.height / 2,
    ) / math.hypot(frames.width, frames.height)
    border_contact = sum(
        (
            bool(np.any(binary[0, :])),
            bool(np.any(binary[-1, :])),
            bool(np.any(binary[:, 0])),
            bool(np.any(binary[:, -1])),
        )
    )
    return EntityReferenceCandidate(
        candidate_id="candidate_0",
        entity_id=entity.entity_id,
        frame_slot=frame.slot,
        source_frame_index=frame.source_frame_index,
        image_path=storage.relative_artifact_path(image_path),
        mask=binary.copy(),
        bbox_xyxy=bbox,
        area_pixels=area,
        area_ratio=area / (frames.width * frames.height),
        bbox_fill_ratio=area / bbox_area,
        border_contact_count=border_contact,
        normalized_center_distance=center_distance,
    )


def build_entity_reference_candidates(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> list[EntityReferenceCandidate]:
    if frames.clip_uid != clip_uid or masks.clip_uid != clip_uid:
        raise ValueError("pair input clip_uid does not match its clip")
    if (frames.width, frames.height) != (masks.width, masks.height):
        raise ValueError("frames and masks dimensions do not match")
    tracked = masks.entities.get(entity.entity_id)
    if tracked is None:
        raise ValueError("mask artifact is missing an annotation entity")
    if (
        tracked.reference_type != entity.reference_type
        or tracked.grounding_prompt != entity.grounding_prompt
    ):
        raise ValueError("mask artifact entity semantics do not match annotation")
    if tracked.status != "ready":
        return []
    if [item.slot for item in tracked.frames] != list(range(10)):
        raise ValueError("tracked entity slots must be ordered from 0 through 9")
    candidates: list[EntityReferenceCandidate] = []
    for frame, tracked_frame in zip(frames.frames, tracked.frames):
        if (
            not tracked_frame.present
            or not tracked_frame.track_valid
            or tracked_frame.area_pixels <= 0
        ):
            continue
        candidates.append(
            _candidate_from_frame(
                storage=storage,
                clip_uid=clip_uid,
                entity=entity,
                frame=frame,
                tracked_frame=tracked_frame,
                frames=frames,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.border_contact_count,
            -candidate.area_pixels,
            -candidate.bbox_fill_ratio,
            candidate.normalized_center_distance,
            _SLOT_PRIORITY_INDEX[candidate.frame_slot],
        )
    )
    shortlisted = candidates[: config.pair.max_candidates_per_entity]
    return [
        replace(candidate, candidate_id=f"candidate_{index}")
        for index, candidate in enumerate(shortlisted, start=1)
    ]


def _resolve_run_artifact(storage: RunStorage, relative_path: str) -> Path:
    path = (storage.root / relative_path).resolve(strict=False)
    root = storage.root.resolve(strict=False)
    if root not in path.parents:
        raise ValueError("reference artifact is outside run_root")
    return path


def _validate_reference_png(
    path: Path,
    *,
    expected: Image.Image,
) -> None:
    if not path.is_file():
        raise ValueError("ready entity reference artifact is missing")
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("ready entity reference must be a PNG")
        if opened.mode != "RGBA":
            raise ValueError("ready entity reference must be RGBA")
        actual = np.asarray(opened)
    expected_pixels = np.asarray(expected)
    if actual.shape != expected_pixels.shape:
        raise ValueError("ready entity reference dimensions are invalid")
    alpha = actual[..., 3]
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("ready entity reference alpha must be binary")
    if not np.array_equal(actual, expected_pixels):
        expected_alpha = expected_pixels[..., 3]
        if not np.array_equal(alpha, expected_alpha):
            raise ValueError("ready entity reference alpha does not match mask")
        if not np.array_equal(
            actual[..., :3][alpha == 255],
            expected_pixels[..., :3][alpha == 255],
        ):
            raise ValueError("ready entity reference source pixels were changed")
        raise ValueError("ready entity reference transparent RGB must be white")


def _selected_candidate_for_state(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    state: EntityReferenceState,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> EntityReferenceCandidate:
    matches = [
        candidate
        for candidate in build_entity_reference_candidates(
            config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            frames=frames,
            masks=masks,
        )
        if candidate.source_frame_index == state.source_frame_index
    ]
    if len(matches) != 1:
        raise ValueError("reference source_frame_index is not a unique candidate")
    return matches[0]


def validate_entity_reference_artifact(
    config: V3Config,
    storage: RunStorage,
    clip_uid: str,
    annotation_entity: AnnotationEntity,
    reference_state: EntityReferenceState,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> None:
    final_path = storage.selected_entity_path(
        clip_uid,
        annotation_entity.entity_id,
    ).resolve(strict=False)
    if reference_state.status == "rejected":
        if final_path.exists():
            raise ValueError("rejected entity reference has a final artifact")
        return
    if reference_state.synthetic:
        raise ValueError("synthetic entity references are not allowed")
    if reference_state.image_path is None:
        raise ValueError("ready entity reference is missing image_path")
    state_path = _resolve_run_artifact(storage, reference_state.image_path)
    if state_path != final_path:
        raise ValueError("ready entity reference path must be selected/eN.png")
    candidate = _selected_candidate_for_state(
        config,
        storage,
        clip_uid=clip_uid,
        entity=annotation_entity,
        state=reference_state,
        frames=frames,
        masks=masks,
    )
    source_path = _resolve_run_artifact(storage, candidate.image_path)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
        source.load()
    expected, _ = build_reference_crop(
        source,
        candidate.mask,
        crop_padding_ratio=config.pair.crop_padding_ratio,
    )
    _validate_reference_png(state_path, expected=expected)


def _rejected_reference(entity_id: str, reason: str) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=entity_id,
        status="rejected",
        reference_scope="reject",
        visible_region="custom",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason=reason,
        synthetic=False,
    )


def _tokens_for_retained(
    retained: list[str],
    entities: dict[str, AnnotationEntity],
) -> dict[str, str]:
    counters = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity_id in retained:
        reference_type = entities[entity_id].reference_type
        counters[reference_type] += 1
        tokens[entity_id] = (
            f"<ref_{reference_type}_{counters[reference_type]}>"
        )
    return tokens


def _load_source_images(
    storage: RunStorage,
    candidates: list[EntityReferenceCandidate],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for candidate in candidates:
        if candidate.image_path in images:
            continue
        with Image.open(
            _resolve_run_artifact(storage, candidate.image_path)
        ) as opened:
            image = opened.convert("RGB")
            image.load()
        images[candidate.image_path] = image
    return images


def _write_debug_attempt(
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    candidates: list[EntityReferenceCandidate],
    attempt: EntityReferenceDecisionAttempt,
) -> None:
    if not storage.config.debug.save_diagnostics:
        return
    directory = storage.pair_debug_dir(clip_uid, entity.entity_id)
    write_json_atomic(
        directory / "request.json",
        {
            "entity_id": entity.entity_id,
            "candidate_ids": [item.candidate_id for item in candidates],
        },
    )
    write_json_atomic(
        directory / "raw_responses.json",
        {"responses": list(attempt.raw_responses)},
    )


def _validate_pair_inputs(
    clip: ClipRecord,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> None:
    if clip.annotation is None or clip.annotation.status != "ready":
        raise ValueError("pair requires ready annotation")
    if frames.clip_uid != clip.clip_uid or masks.clip_uid != clip.clip_uid:
        raise ValueError("pair artifact clip_uid does not match clip")
    if (frames.width, frames.height) != (masks.width, masks.height):
        raise ValueError("frames and masks dimensions do not match")
    annotation_ids = [entity.entity_id for entity in clip.annotation.entities]
    if list(masks.entities) != annotation_ids:
        raise ValueError("mask entity IDs must match annotation order")
    for entity in clip.annotation.entities:
        tracked = masks.entities[entity.entity_id]
        if (
            tracked.reference_type != entity.reference_type
            or tracked.grounding_prompt != entity.grounding_prompt
        ):
            raise ValueError("mask entity semantics do not match annotation")


def _validate_existing_pairing(
    config: V3Config,
    storage: RunStorage,
    clip: ClipRecord,
    *,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> None:
    if clip.pairing is None or clip.annotation is None:
        raise ValueError("existing pairing is missing")
    expected_ids = [entity.entity_id for entity in clip.annotation.entities]
    actual_ids = [state.entity_id for state in clip.references.entities]
    if actual_ids != expected_ids:
        raise ValueError("existing references do not match annotation order")
    entity_by_id = {
        entity.entity_id: entity for entity in clip.annotation.entities
    }
    for state in clip.references.entities:
        validate_entity_reference_artifact(
            config,
            storage,
            clip.clip_uid,
            entity_by_id[state.entity_id],
            state,
            frames,
            masks,
        )
    background = clip.references.background
    if clip.pairing.background_token is not None:
        if background is None or background.status not in {
            "clean_raw",
            "ready_removed",
        }:
            raise ValueError("ready background token has no ready artifact")
        validate_background_reference(
            storage,
            clip.clip_uid,
            background,
            frames=frames,
        )
    elif (
        clip.pairing.status == "ready"
        and background is not None
        and background.status in {"clean_raw", "ready_removed"}
    ):
        raise ValueError("ready background was not bound")


def _publish_pair_result(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    references: ReferencesState,
    pairing: PairingState,
    temporary_images: dict[str, tuple[Path, Image.Image]],
) -> None:
    selected = storage.selected_path(clip_uid, "placeholder").parent
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for entity_id, (temporary, expected) in temporary_images.items():
            _validate_reference_png(temporary, expected=expected)
            final = storage.selected_entity_path(clip_uid, entity_id)
            if final in published:
                raise ValueError("duplicate entity reference destination")
        for existing in selected.glob("*.png"):
            if _ENTITY_PNG.fullmatch(existing.name) is None:
                continue
            entity_id = existing.stem
            backup = storage.pair_output_backup_path(clip_uid, entity_id)
            existing.replace(backup)
            backups[existing] = backup
        for entity_id, (temporary, _) in temporary_images.items():
            final = storage.selected_entity_path(clip_uid, entity_id)
            temporary.replace(final)
            published.append(final)
        storage.write_references_and_pairing(clip_uid, references, pairing)
    except Exception:
        for final in published:
            final.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists():
                backup.replace(final)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in temporary_images.values():
            temporary.unlink(missing_ok=True)
        storage.cleanup_pair_artifacts(clip_uid)


def _failure_details(exc: Exception) -> dict[str, object]:
    if isinstance(exc, EntityReferenceJudgeFailure):
        return exc.to_dict()
    return {"exception_type": type(exc).__name__}


def pair_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    judge: EntityReferenceJudge | None = None,
) -> PairStats:
    config.validate()
    if storage.root != config.resolved_run_root:
        raise ValueError("storage run_root does not match pair configuration")
    if not config.pair.enabled:
        raise ValueError("V3 pair stage is disabled")
    counters = {field: 0 for field in PairStats.__dataclass_fields__}
    active_judge = judge
    owned_judge: QwenEntityReferenceJudge | None = None
    try:
        for initial_clip in storage.iter_clips():
            clip = storage.read_clip(initial_clip.clip_uid)
            if (
                clip.annotation is None
                or clip.annotation.status != "ready"
                or clip.coverage is None
                or not clip.coverage.passed
            ):
                counters["skipped_not_ready"] += 1
                continue
            if clip.pairing is not None and not overwrite:
                try:
                    frames = validate_sampled_frames(storage, clip.clip_uid)
                    masks = storage.read_masks(clip.clip_uid)
                    _validate_pair_inputs(clip, frames, masks)
                    _validate_existing_pairing(
                        config,
                        storage,
                        clip,
                        frames=frames,
                        masks=masks,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate corrupt clips
                    storage.append_failure(
                        stage="pair",
                        clip_uid=clip.clip_uid,
                        reason=str(exc),
                        details=_failure_details(exc),
                    )
                    counters["failed"] += 1
                else:
                    counters["skipped_existing"] += 1
                continue
            try:
                frames = validate_sampled_frames(storage, clip.clip_uid)
                masks = storage.read_masks(clip.clip_uid)
                _validate_pair_inputs(clip, frames, masks)
            except Exception:  # noqa: BLE001 - incomplete inputs are not eligible
                counters["skipped_not_ready"] += 1
                continue
            counters["processed"] += 1
            temporary_images: dict[str, tuple[Path, Image.Image]] = {}
            try:
                entity_states: list[EntityReferenceState] = []
                assert clip.annotation is not None
                for entity in clip.annotation.entities:
                    tracked: TrackedEntityMasks = masks.entities[entity.entity_id]
                    if tracked.status != "ready":
                        entity_states.append(
                            _rejected_reference(
                                entity.entity_id,
                                f"tracking_not_ready:{tracked.status}",
                            )
                        )
                        continue
                    candidates = build_entity_reference_candidates(
                        config,
                        storage,
                        clip_uid=clip.clip_uid,
                        entity=entity,
                        frames=frames,
                        masks=masks,
                    )
                    if not candidates:
                        entity_states.append(
                            _rejected_reference(
                                entity.entity_id,
                                "no_valid_reference_candidate",
                            )
                        )
                        continue
                    if active_judge is None:
                        judge_config = config.qwen.candidate_judge
                        if judge_config is None:
                            raise RuntimeError("candidate judge is not configured")
                        owned_judge = QwenEntityReferenceJudge(
                            judge_config,
                            repair_retries=config.pair.repair_retries,
                            crop_padding_ratio=config.pair.crop_padding_ratio,
                        )
                        active_judge = owned_judge
                    source_images = _load_source_images(storage, candidates)
                    attempt = active_judge.decide(
                        entity=entity,
                        candidates=candidates,
                        source_images=source_images,
                    )
                    counters["repaired"] += int(attempt.repair_attempts > 0)
                    _write_debug_attempt(
                        storage,
                        clip_uid=clip.clip_uid,
                        entity=entity,
                        candidates=candidates,
                        attempt=attempt,
                    )
                    decision = attempt.decision
                    decision_issues = validate_entity_reference_decision(
                        decision,
                        candidate_ids={item.candidate_id for item in candidates},
                    )
                    if decision_issues:
                        messages = "; ".join(
                            issue.message for issue in decision_issues
                        )
                        raise ValueError(f"invalid judge decision: {messages}")
                    if decision.reference_scope == "reject":
                        entity_states.append(
                            _rejected_reference(
                                entity.entity_id,
                                decision.scope_reason,
                            )
                        )
                        continue
                    if (
                        decision.reference_scope == "local"
                        and not config.reference_scope.allow_local
                    ):
                        entity_states.append(
                            _rejected_reference(
                                entity.entity_id,
                                "local_reference_disabled",
                            )
                        )
                        continue
                    selected = next(
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id
                        == decision.selected_candidate_id
                    )
                    source = source_images[selected.image_path]
                    reference_image, _ = build_reference_crop(
                        source,
                        selected.mask,
                        crop_padding_ratio=config.pair.crop_padding_ratio,
                    )
                    temporary = storage.pair_output_temporary_path(
                        clip.clip_uid,
                        entity.entity_id,
                    )
                    reference_image.save(temporary, format="PNG")
                    temporary_images[entity.entity_id] = (
                        temporary,
                        reference_image,
                    )
                    entity_states.append(
                        EntityReferenceState(
                            entity_id=entity.entity_id,
                            status="ready",
                            reference_scope=decision.reference_scope,
                            visible_region=decision.visible_region,
                            whole_entity_recognizable=(
                                decision.whole_entity_recognizable
                            ),
                            identity_features_visible=(
                                decision.identity_features_visible
                            ),
                            scope_reason=decision.scope_reason,
                            image_path=storage.relative_artifact_path(
                                storage.selected_entity_path(
                                    clip.clip_uid,
                                    entity.entity_id,
                                )
                            ),
                            source_frame_index=selected.source_frame_index,
                            synthetic=False,
                        )
                    )
                retained = [
                    state.entity_id
                    for state in entity_states
                    if state.status == "ready"
                ]
                background = clip.references.background
                background_token: str | None = None
                if (
                    background is not None
                    and background.status in {"clean_raw", "ready_removed"}
                ):
                    validate_background_reference(
                        storage,
                        clip.clip_uid,
                        background,
                        frames=frames,
                    )
                    background_token = "<ref_bg_1>"
                if not set(retained).intersection(
                    clip.coverage.qualifying_entity_ids
                ):
                    pairing = PairingState(
                        status="rejected",
                        reason="no_qualifying_ready_reference",
                    )
                else:
                    entity_by_id = {
                        entity.entity_id: entity
                        for entity in clip.annotation.entities
                    }
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(retained, entity_by_id),
                        background_token=background_token,
                    )
                references = ReferencesState(
                    entities=entity_states,
                    background=background,
                )
                _publish_pair_result(
                    config,
                    storage,
                    clip_uid=clip.clip_uid,
                    references=references,
                    pairing=pairing,
                    temporary_images=temporary_images,
                )
                ready_count = sum(
                    state.status == "ready" for state in entity_states
                )
                counters["entities_ready"] += ready_count
                counters["entities_rejected"] += (
                    len(entity_states) - ready_count
                )
                counters[pairing.status] += 1
                counters["backgrounds_bound"] += int(
                    pairing.background_token is not None
                )
            except Exception as exc:  # noqa: BLE001 - continue with later clips
                for temporary, _ in temporary_images.values():
                    temporary.unlink(missing_ok=True)
                storage.cleanup_pair_artifacts(clip.clip_uid)
                storage.append_failure(
                    stage="pair",
                    clip_uid=clip.clip_uid,
                    reason=str(exc),
                    details=_failure_details(exc),
                )
                counters["failed"] += 1
    finally:
        if owned_judge is not None:
            owned_judge.close()
    stats = PairStats(**counters)
    storage.update_stage_counts("pair", stats.to_dict())
    return stats
