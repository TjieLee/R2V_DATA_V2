from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import validate_background_reference
from r2v_data_v2.v3.background_final_guard import (
    FinalBackgroundJudge,
    FinalBackgroundJudgeFailure,
    QwenFinalBackgroundJudge,
    load_final_background_image,
)
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.cross_pair_judge import (
    CrossPairDecisionAttempt,
    CrossPairJudge,
    CrossPairJudgeFailure,
    CrossPairTargetEvidenceMode,
    QwenCrossPairJudge,
)
from r2v_data_v2.v3.frames import validate_sampled_frames
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.reference_completion import (
    run_reference_completion_fallbacks,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    QwenLocalizedCompletionJudge,
    QwenReferenceCompletionBackend,
)
from r2v_data_v2.v3.reference_geometry import (
    content_geometry_from_mask,
    tiny_content_reason,
)
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
    EntityReferenceJudge,
    EntityReferenceJudgeFailure,
    QwenEntityReferenceJudge,
    validate_entity_reference_decision,
)
from r2v_data_v2.v3.reference_prefilter import (
    NEAR_SILHOUETTE_RULE,
    RELATIVE_BLUR_V2_RULE,
    ReferencePrefilterResult,
    prefilter_entity_reference_candidates,
)
from r2v_data_v2.v3.sam3_backend import SegmentationBackend
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    BackgroundReferenceState,
    ClipRecord,
    EntityReferenceState,
    PairingState,
    RawEntityReferenceDecision,
    ReferencesState,
    ReferenceType,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage

_SLOT_PRIORITY = (5, 4, 6, 3, 7, 2, 8, 1, 9, 0)
_SLOT_PRIORITY_INDEX = {slot: index for index, slot in enumerate(_SLOT_PRIORITY)}
_ENTITY_PNG = re.compile(r"e[1-9]\d*\.png")
_CROSS_PAIR_CONTACT_SHEET_COLUMNS = 5
_CROSS_PAIR_CONTACT_SHEET_PANEL_MAX_SIDE = 384
_CROSS_PAIR_CONTACT_SHEET_LABEL_HEIGHT = 28
_MIN_ERODED_SHARPNESS_PIXELS = 9


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
    sharpness_score: float = 0.0
    significant_component_count: int = 1
    largest_component_ratio: float = 1.0
    second_largest_component_ratio: float = 0.0


@dataclass(frozen=True)
class MaskComponent:
    area_pixels: int
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class MaskComponentDiagnostics:
    significant_component_count: int
    largest_component_ratio: float
    second_largest_component_ratio: float

    @property
    def severely_fragmented(self) -> bool:
        return (
            self.largest_component_ratio < 0.70
            or self.second_largest_component_ratio > 0.20
            or self.significant_component_count > 3
        )


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
    cross_pair_attempted: int = 0
    cross_pair_ready: int = 0
    cross_pair_repaired: int = 0
    completion_attempted: int = 0
    completion_ready: int = 0
    completion_rejected: int = 0
    prefilter_candidates_examined: int = 0
    prefilter_candidates_filtered: int = 0
    prefilter_near_silhouette_filtered: int = 0
    prefilter_relative_blur_v2_filtered: int = 0
    prefilter_entities_3_to_2: int = 0
    prefilter_entities_3_to_1: int = 0
    prefilter_entities_3_to_0: int = 0
    prefilter_entities_2_to_1: int = 0
    prefilter_entities_2_to_0: int = 0
    prefilter_qwen_calls_skipped: int = 0
    prefilter_fail_open_entities: int = 0
    background_final_guard_attempted: int = 0
    background_final_guard_accepted: int = 0
    background_final_guard_rejected: int = 0
    background_final_guard_failed_closed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _BackgroundFinalGuardRuntime:
    def __init__(
        self,
        config: V3Config,
        storage: RunStorage,
        counters: dict[str, int],
        judge: FinalBackgroundJudge | None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.counters = counters
        self.active_judge = judge
        self.owned_judge: QwenFinalBackgroundJudge | None = None
        self.evaluated_clip_uids: set[str] = set()

    def _judge(self) -> FinalBackgroundJudge:
        if self.active_judge is None:
            service = self.config.qwen.background_final_judge
            if service is None:
                raise RuntimeError("final background judge is not configured")
            self.owned_judge = QwenFinalBackgroundJudge(service)
            self.active_judge = self.owned_judge
        return self.active_judge

    def _write_debug(
        self,
        *,
        clip: ClipRecord,
        background: BackgroundReferenceState,
        review: object | None,
        raw_response: str | None,
        error: Exception | None,
        bound_after_guard: bool,
    ) -> None:
        if not self.config.debug.save_diagnostics:
            return
        annotation_background = (
            clip.annotation.background if clip.annotation is not None else None
        )
        payload: dict[str, object] = {
            "clip_uid": clip.clip_uid,
            "background_status": background.status,
            "background_phrase": (
                annotation_background.phrase
                if annotation_background is not None
                else None
            ),
            "background_grounding_prompt": (
                annotation_background.grounding_prompt
                if annotation_background is not None
                else None
            ),
            "mode": self.config.pair.background_final_guard_mode,
            "verdict": None,
            "background_matches_description": None,
            "no_unexpected_foreground_subject": None,
            "usable_background_information": None,
            "no_obvious_artifacts": None,
            "reason": str(error) if error is not None else None,
            "raw_response": raw_response,
            "error": str(error) if error is not None else None,
            "bound_after_guard": bound_after_guard,
        }
        if review is not None:
            review_payload = review.model_dump(mode="json")
            payload.update(review_payload)
        destination = (
            self.storage.clip_dir(clip.clip_uid)
            / "debug"
            / "background_final_guard.json"
        )
        try:
            write_json_atomic(destination, payload)
        except Exception:  # noqa: BLE001,S110 - diagnostics cannot reject a sample
            pass

    def token_for_ready_pairing(
        self,
        *,
        clip: ClipRecord,
        frames: SampledFramesArtifact,
    ) -> str | None:
        background = clip.references.background
        if background is None or background.status not in {
            "clean_raw",
            "ready_removed",
        }:
            return None
        if self.config.pair.background_final_guard_mode == "off":
            validate_background_reference(
                self.storage,
                clip.clip_uid,
                background,
                frames=frames,
            )
            return "<ref_bg_1>"

        self.evaluated_clip_uids.add(clip.clip_uid)
        self.counters["background_final_guard_attempted"] += 1
        raw_response: str | None = None
        try:
            validate_background_reference(
                self.storage,
                clip.clip_uid,
                background,
                frames=frames,
            )
            annotation_background = (
                clip.annotation.background if clip.annotation is not None else None
            )
            if annotation_background is None:
                raise ValueError("ready background has no annotation semantics")
            image = load_final_background_image(
                self.storage,
                clip_uid=clip.clip_uid,
                background=background,
            )
            attempt = self._judge().review(
                image=image,
                background_phrase=annotation_background.phrase,
                background_grounding_prompt=(
                    annotation_background.grounding_prompt
                ),
                background_status=background.status,
            )
            raw_response = attempt.raw_response
        except Exception as exc:  # noqa: BLE001 - background-only fail closed
            if isinstance(exc, FinalBackgroundJudgeFailure):
                raw_response = exc.raw_response
            self.counters["background_final_guard_failed_closed"] += 1
            self._write_debug(
                clip=clip,
                background=background,
                review=None,
                raw_response=raw_response,
                error=exc,
                bound_after_guard=False,
            )
            return None

        accepted = attempt.review.verdict == "accept"
        self.counters[
            "background_final_guard_accepted"
            if accepted
            else "background_final_guard_rejected"
        ] += 1
        self._write_debug(
            clip=clip,
            background=background,
            review=attempt.review,
            raw_response=raw_response,
            error=None,
            bound_after_guard=accepted,
        )
        return "<ref_bg_1>" if accepted else None

    def close(self) -> None:
        if self.owned_judge is not None:
            try:
                self.owned_judge.close()
            except Exception:  # noqa: BLE001,S110 - cleanup cannot reject samples
                pass


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


def _foreground_components(mask: np.ndarray) -> tuple[MaskComponent, ...]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("component diagnostics require a non-empty 2D mask")

    parents: list[int] = []
    run_areas: list[int] = []
    run_bboxes: list[tuple[int, int, int, int]] = []

    def make_set(
        area: int,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> int:
        label = len(parents)
        parents.append(label)
        run_areas.append(area)
        run_bboxes.append(bbox_xyxy)
        return label

    def find(label: int) -> int:
        root = label
        while parents[root] != root:
            root = parents[root]
        while parents[label] != label:
            parent = parents[label]
            parents[label] = root
            label = parent
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    previous_runs: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(binary):
        padded = np.pad(row.astype(np.int8, copy=False), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        current_runs: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = make_set(
                end - start,
                (start, row_index, end, row_index + 1),
            )
            while (
                previous_index < len(previous_runs)
                and previous_runs[previous_index][1] < start
            ):
                previous_index += 1
            overlap_index = previous_index
            while (
                overlap_index < len(previous_runs)
                and previous_runs[overlap_index][0] <= end
            ):
                union(label, previous_runs[overlap_index][2])
                overlap_index += 1
            current_runs.append((start, end, label))
        previous_runs = current_runs

    component_areas: dict[int, int] = {}
    component_bboxes: dict[int, tuple[int, int, int, int]] = {}
    for label, area in enumerate(run_areas):
        root = find(label)
        component_areas[root] = component_areas.get(root, 0) + area
        x1, y1, x2, y2 = run_bboxes[label]
        previous = component_bboxes.get(root)
        if previous is not None:
            x1 = min(x1, previous[0])
            y1 = min(y1, previous[1])
            x2 = max(x2, previous[2])
            y2 = max(y2, previous[3])
        component_bboxes[root] = (x1, y1, x2, y2)
    return tuple(
        sorted(
            (
                MaskComponent(
                    area_pixels=area,
                    bbox_xyxy=component_bboxes[root],
                )
                for root, area in component_areas.items()
            ),
            key=lambda component: (
                -component.area_pixels,
                component.bbox_xyxy,
            ),
        )
    )


def mask_component_diagnostics(mask: np.ndarray) -> MaskComponentDiagnostics:
    components = _foreground_components(mask)
    component_areas = [component.area_pixels for component in components]
    total_area = sum(component_areas)
    significant_area = max(16.0, total_area * 0.02)
    return MaskComponentDiagnostics(
        significant_component_count=sum(
            area >= significant_area for area in component_areas
        ),
        largest_component_ratio=component_areas[0] / total_area,
        second_largest_component_ratio=(
            component_areas[1] / total_area if len(component_areas) > 1 else 0.0
        ),
    )


def build_candidate_context_image(
    source_image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    source, binary = _validate_source_and_mask(source_image, mask)
    context = source.copy()
    context[~binary] = (context[~binary].astype(np.uint16) * 35 // 100).astype(np.uint8)
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


def build_cross_pair_target_contact_sheet(
    frame_images: list[tuple[int, Image.Image]],
    *,
    panel_max_side: int = _CROSS_PAIR_CONTACT_SHEET_PANEL_MAX_SIDE,
) -> Image.Image:
    if (
        isinstance(panel_max_side, bool)
        or not isinstance(panel_max_side, int)
        or panel_max_side <= 0
    ):
        raise ValueError(
            "cross-pair contact-sheet panel_max_side must be a positive integer"
        )
    slots = [slot for slot, _ in frame_images]
    if slots != list(range(10)):
        raise ValueError(
            "cross-pair target frames must contain ordered slots 0 through 9"
        )
    for _, image in frame_images:
        if not isinstance(image, Image.Image):
            raise TypeError("cross-pair target frames must be PIL images")

    label_height = _CROSS_PAIR_CONTACT_SHEET_LABEL_HEIGHT
    panel_height = panel_max_side + label_height
    sheet = Image.new(
        "RGB",
        (
            panel_max_side * _CROSS_PAIR_CONTACT_SHEET_COLUMNS,
            panel_height * 2,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (slot, image) in enumerate(frame_images):
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail(
            (panel_max_side, panel_max_side),
            Image.Resampling.LANCZOS,
        )
        column = index % _CROSS_PAIR_CONTACT_SHEET_COLUMNS
        row = index // _CROSS_PAIR_CONTACT_SHEET_COLUMNS
        origin_x = column * panel_max_side
        origin_y = row * panel_height
        image_x = origin_x + (panel_max_side - thumbnail.width) // 2
        image_y = origin_y + label_height + (panel_max_side - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))
        draw.text(
            (origin_x + 8, origin_y + 6),
            f"slot {slot}",
            fill=(0, 0, 0),
        )
    return sheet


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
        raise ValueError("crop_padding_ratio must be a finite float between 0 and 0.5")
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
        source = np.asarray(opened.convert("RGB"), dtype=np.float32)
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
    sharpness_score = _masked_sharpness_score(source, binary)
    component_diagnostics = mask_component_diagnostics(binary)
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
        sharpness_score=sharpness_score,
        significant_component_count=(
            component_diagnostics.significant_component_count
        ),
        largest_component_ratio=component_diagnostics.largest_component_ratio,
        second_largest_component_ratio=(
            component_diagnostics.second_largest_component_ratio
        ),
    )


def _masked_sharpness_score(source: np.ndarray, mask: np.ndarray) -> float:
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("sharpness source must be an H-W-RGB array")
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != source.shape[:2] or not binary.any():
        raise ValueError("sharpness mask must match source and be non-empty")
    evaluation_mask = _eroded_sharpness_mask(binary)
    gray = 0.299 * source[..., 0] + 0.587 * source[..., 1] + 0.114 * source[..., 2]
    padded = np.pad(gray.astype(np.float64, copy=False), 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    values = laplacian[evaluation_mask]
    score = float(np.var(values)) if values.size > 1 else 0.0
    if not math.isfinite(score) or score < 0:
        raise ValueError("sharpness score must be finite and non-negative")
    return score


def _eroded_sharpness_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 2, mode="constant", constant_values=False)
    eroded = np.ones_like(binary, dtype=bool)
    for row_offset in range(5):
        for column_offset in range(5):
            eroded &= padded[
                row_offset : row_offset + binary.shape[0],
                column_offset : column_offset + binary.shape[1],
            ]
    return (
        eroded
        if np.count_nonzero(eroded) >= _MIN_ERODED_SHARPNESS_PIXELS
        else binary
    )


def entity_candidate_geometry_thresholds(
    config: V3Config,
    reference_type: ReferenceType,
) -> tuple[int, int]:
    if (
        config.pair.entity_geometry_mode == "type_aware_v1"
        and reference_type == "object"
    ):
        return 64 * 64, 96
    return (
        config.reference_edit.min_source_content_area_pixels,
        config.reference_edit.min_source_content_long_side_pixels,
    )


def _build_entity_reference_candidates(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> tuple[list[EntityReferenceCandidate], bool, bool]:
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
        return [], False, False
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
    minimum_area_pixels, minimum_long_side_pixels = (
        entity_candidate_geometry_thresholds(config, entity.reference_type)
    )
    non_tiny_candidates = [
        candidate
        for candidate in candidates
        if tiny_content_reason(
            content_geometry_from_mask(candidate.mask),
            minimum_area_pixels=minimum_area_pixels,
            minimum_long_side_pixels=minimum_long_side_pixels,
        )
        is None
    ]
    all_candidates_tiny = bool(candidates) and not non_tiny_candidates
    eligible_candidates = [
        candidate
        for candidate in non_tiny_candidates
        if entity.reference_type == "group"
        or not MaskComponentDiagnostics(
            significant_component_count=candidate.significant_component_count,
            largest_component_ratio=candidate.largest_component_ratio,
            second_largest_component_ratio=(
                candidate.second_largest_component_ratio
            ),
        ).severely_fragmented
    ]
    all_non_tiny_candidates_fragmented = (
        bool(non_tiny_candidates) and not eligible_candidates
    )
    eligible_candidates.sort(
        key=lambda candidate: (
            candidate.border_contact_count,
            -candidate.area_ratio,
            -candidate.sharpness_score,
            candidate.normalized_center_distance,
            _SLOT_PRIORITY_INDEX[candidate.frame_slot],
        )
    )
    shortlisted = eligible_candidates[: config.pair.max_candidates_per_entity]
    return [
        replace(candidate, candidate_id=f"candidate_{index}")
        for index, candidate in enumerate(shortlisted, start=1)
    ], all_candidates_tiny, all_non_tiny_candidates_fragmented


def build_entity_reference_candidates(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
) -> list[EntityReferenceCandidate]:
    candidates, _, _ = _build_entity_reference_candidates(
        config,
        storage,
        clip_uid=clip_uid,
        entity=entity,
        frames=frames,
        masks=masks,
    )
    return candidates


def _resolve_run_artifact(storage: RunStorage, relative_path: str) -> Path:
    path = (storage.root / relative_path).resolve(strict=False)
    root = storage.root.resolve(strict=False)
    if root not in path.parents:
        raise ValueError("reference artifact is outside run_root")
    return path


def _validate_reference_png(
    path: Path,
    *,
    expected: Image.Image | None,
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
    alpha = actual[..., 3]
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("ready entity reference alpha must be binary")
    if not np.any(alpha == 255):
        raise ValueError("ready entity reference alpha must not be empty")
    if np.any(actual[..., :3][alpha == 0] != 255):
        raise ValueError("ready entity reference transparent RGB must be white")
    if expected is None:
        return
    expected_pixels = np.asarray(expected)
    if actual.shape != expected_pixels.shape:
        raise ValueError("ready entity reference dimensions are invalid")
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


def _validate_boogu_reference_png(path: Path) -> None:
    if not path.is_file():
        raise ValueError("ready Boogu reference artifact is missing")
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("ready Boogu reference must be a PNG")
        if opened.mode != "RGB":
            raise ValueError("ready Boogu reference must be native RGB")
        if opened.width <= 0 or opened.height <= 0:
            raise ValueError("ready Boogu reference dimensions are invalid")


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
        clip = storage.read_clip(clip_uid)
        preserved_after_rejection = (
            clip.reference_edit is not None
            and clip.reference_edit.status == "ready"
            and any(
                item.entity_id == annotation_entity.entity_id
                and item.status == "rejected"
                for item in clip.reference_edit.entities
            )
        )
        if final_path.exists() and not preserved_after_rejection:
            raise ValueError("rejected entity reference has a final artifact")
        return
    if reference_state.image_path is None:
        raise ValueError("ready entity reference is missing image_path")
    state_path = _resolve_run_artifact(storage, reference_state.image_path)

    source_clip_uid = reference_state.source_clip_uid
    source_entity_id = reference_state.source_entity_id
    is_legacy = source_clip_uid is None and source_entity_id is None
    is_self = (
        source_clip_uid == clip_uid and source_entity_id == annotation_entity.entity_id
    )
    if reference_state.synthetic:
        if not is_self:
            raise ValueError("generated fallback must use self provenance")
        metadata_value = reference_state.generation_metadata_path
        if metadata_value is None:
            raise ValueError("generated fallback metadata path is missing")
        metadata_path = _resolve_run_artifact(storage, metadata_value)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("backend") == "boogu_image_0_1_edit_turbo":
            expected_path = (
                storage.reference_edit_dir(clip_uid)
                / annotation_entity.entity_id
                / "final_reference_1k.png"
            ).resolve(strict=False)
            if state_path != expected_path:
                raise ValueError(
                    "ready Boogu reference path must use final_reference_1k.png"
                )
            _validate_boogu_reference_png(state_path)
        else:
            if state_path != final_path:
                raise ValueError(
                    "legacy generated reference path must be selected/eN.png"
                )
            _validate_reference_png(state_path, expected=None)
        output_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
        if output_sha256 != reference_state.generation_output_sha256:
            raise ValueError("generated fallback output hash changed")
        if (
            metadata.get("status") != "accepted"
            or metadata.get("clip_uid") != clip_uid
            or metadata.get("entity_id") != annotation_entity.entity_id
            or metadata.get("generated_reference_sha256") != output_sha256
        ):
            raise ValueError("generated fallback metadata does not match state")
        source_path_value = metadata.get("source_image_path")
        if not isinstance(source_path_value, str):
            raise ValueError("generated fallback source path is missing")
        source_path = _resolve_run_artifact(storage, source_path_value)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            source_sha256 != reference_state.generation_source_sha256
            or metadata.get("source_image_sha256") != source_sha256
        ):
            raise ValueError("generated fallback source hash changed")
        return
    if state_path != final_path:
        raise ValueError("ready real reference path must be selected/eN.png")
    if is_legacy or is_self:
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
        return

    target_clip = storage.read_clip(clip_uid)
    if source_clip_uid == clip_uid:
        raise ValueError("cross-pair provenance must use a different clip_uid")
    donor_clip = storage.read_clip(source_clip_uid)
    if donor_clip.source.parent_video_id != target_clip.source.parent_video_id:
        raise ValueError("cross-pair donor must have the exact same parent_video_id")
    if donor_clip.annotation is None or donor_clip.annotation.status != "ready":
        raise ValueError("cross-pair donor annotation is not ready")
    if donor_clip.pairing is None or donor_clip.pairing.status != "ready":
        raise ValueError("cross-pair donor pairing is not ready")
    donor_entities = {
        entity.entity_id: entity for entity in donor_clip.annotation.entities
    }
    donor_entity = donor_entities.get(source_entity_id)
    if donor_entity is None:
        raise ValueError("cross-pair donor entity is missing")
    if donor_entity.reference_type != annotation_entity.reference_type:
        raise ValueError("cross-pair donor reference type does not match target")
    donor_states = {state.entity_id: state for state in donor_clip.references.entities}
    donor_state = donor_states.get(source_entity_id)
    if (
        donor_state is None
        or donor_state.status != "ready"
        or donor_state.reference_scope != "full"
        or not donor_state.identity_features_visible
        or donor_state.synthetic
        or source_entity_id not in donor_clip.pairing.retained_entity_ids
    ):
        raise ValueError("cross-pair donor reference is no longer eligible")
    if donor_state.source_clip_uid not in {None, source_clip_uid} or (
        donor_state.source_entity_id not in {None, source_entity_id}
    ):
        raise ValueError("cross-pair donor references cannot be chained")
    donor_frames = validate_sampled_frames(storage, source_clip_uid)
    donor_masks = storage.read_masks(source_clip_uid)
    _validate_pair_inputs(donor_clip, donor_frames, donor_masks)
    validate_entity_reference_artifact(
        config,
        storage,
        source_clip_uid,
        donor_entity,
        donor_state,
        donor_frames,
        donor_masks,
    )
    # A cross-paired target can be validated from either a masked candidate
    # or the complete sampled-frame artifact already validated by the caller.
    # Building candidates still validates masked evidence when it exists;
    # an empty shortlist is valid for context-only cross-pairing.
    build_entity_reference_candidates(
        config,
        storage,
        clip_uid=clip_uid,
        entity=annotation_entity,
        frames=frames,
        masks=masks,
    )
    inherited_fields = (
        "reference_scope",
        "visible_region",
        "whole_entity_recognizable",
        "identity_features_visible",
        "scope_reason",
        "viewpoint",
        "independent_reference_value",
        "requires_substantial_invention",
        "primary_identity_region_visible",
        "major_structure_visible",
        "truncation_severity",
        "discrete_foreground_instance",
        "mask_matches_target",
        "completion_needed_for_reference_use",
        "detached_target_fragments_present",
        "source_frame_index",
    )
    if any(
        getattr(reference_state, field) != getattr(donor_state, field)
        for field in inherited_fields
    ):
        raise ValueError("cross-pair reference did not inherit donor quality fields")
    donor_path = storage.selected_entity_path(
        source_clip_uid,
        source_entity_id,
    )
    if state_path.read_bytes() != donor_path.read_bytes():
        raise ValueError("cross-pair target artifact bytes do not match donor PNG")


def _rejected_reference(
    entity_id: str,
    reason: str,
    *,
    decision: RawEntityReferenceDecision | None = None,
) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=entity_id,
        status="rejected",
        reference_scope="reject",
        visible_region="custom",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason=reason,
        image_quality=(decision.image_quality if decision is not None else None),
        completeness=(decision.completeness if decision is not None else None),
        viewpoint=(decision.viewpoint if decision is not None else None),
        independent_reference_value=(
            decision.independent_reference_value if decision is not None else None
        ),
        requires_substantial_invention=(
            decision.requires_substantial_invention if decision is not None else None
        ),
        primary_identity_region_visible=(
            decision.primary_identity_region_visible
            if decision is not None
            else None
        ),
        major_structure_visible=(
            decision.major_structure_visible if decision is not None else None
        ),
        truncation_severity=(
            decision.truncation_severity if decision is not None else None
        ),
        discrete_foreground_instance=(
            decision.discrete_foreground_instance if decision is not None else None
        ),
        mask_matches_target=(
            decision.mask_matches_target if decision is not None else None
        ),
        completion_needed_for_reference_use=(
            decision.completion_needed_for_reference_use
            if decision is not None
            else None
        ),
        detached_target_fragments_present=(
            decision.detached_target_fragments_present
            if decision is not None
            else None
        ),
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
        tokens[entity_id] = f"<ref_{reference_type}_{counters[reference_type]}>"
    return tokens


def _load_source_images(
    storage: RunStorage,
    candidates: list[EntityReferenceCandidate],
) -> dict[str, Image.Image]:
    images: dict[str, Image.Image] = {}
    for candidate in candidates:
        if candidate.image_path in images:
            continue
        with Image.open(_resolve_run_artifact(storage, candidate.image_path)) as opened:
            image = opened.convert("RGB")
            image.load()
        images[candidate.image_path] = image
    return images


def _load_cross_pair_target_frame_images(
    storage: RunStorage,
    *,
    clip_uid: str,
    frames: SampledFramesArtifact,
) -> list[tuple[int, Image.Image]]:
    frame_images: list[tuple[int, Image.Image]] = []
    for frame in frames.frames:
        with Image.open(storage.frame_path(clip_uid, frame.slot)) as opened:
            image = opened.convert("RGB")
            image.load()
        frame_images.append((frame.slot, image))
    return frame_images


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


def _write_prefilter_debug(
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    original_candidates: list[EntityReferenceCandidate],
    result: ReferencePrefilterResult[EntityReferenceCandidate] | None,
    error: Exception | None = None,
) -> None:
    if not storage.config.debug.save_diagnostics:
        return
    if (result is None) == (error is None):
        raise ValueError("prefilter debug requires exactly one result or error")
    payload: dict[str, object] = {
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "original_candidate_ids": [
            candidate.candidate_id for candidate in original_candidates
        ],
        "prefilter_fail_open": error is not None,
    }
    if result is not None:
        payload.update(
            {
                "retained_candidate_ids": [
                    candidate.candidate_id
                    for candidate in result.retained_candidates
                ],
                "candidates": [
                    decision.to_dict() for decision in result.decisions
                ],
            }
        )
    else:
        assert error is not None
        payload.update(
            {
                "retained_candidate_ids": [
                    candidate.candidate_id for candidate in original_candidates
                ],
                "reason": "reference_prefilter_failed_open",
                "error": str(error),
            }
        )
    directory = storage.pair_debug_dir(clip_uid, entity.entity_id)
    write_json_atomic(directory / "prefilter.json", payload)


def _record_prefilter_stats(
    counters: dict[str, int],
    result: ReferencePrefilterResult[EntityReferenceCandidate],
) -> None:
    before = len(result.original_candidates)
    after = len(result.retained_candidates)
    counters["prefilter_candidates_examined"] += before
    counters["prefilter_candidates_filtered"] += result.filtered_count
    counters["prefilter_near_silhouette_filtered"] += sum(
        NEAR_SILHOUETTE_RULE in decision.flagged_by
        for decision in result.decisions
    )
    counters["prefilter_relative_blur_v2_filtered"] += sum(
        RELATIVE_BLUR_V2_RULE in decision.flagged_by
        for decision in result.decisions
    )
    transition_key = {
        (3, 2): "prefilter_entities_3_to_2",
        (3, 1): "prefilter_entities_3_to_1",
        (3, 0): "prefilter_entities_3_to_0",
        (2, 1): "prefilter_entities_2_to_1",
        (2, 0): "prefilter_entities_2_to_0",
    }.get((before, after))
    if transition_key is not None:
        counters[transition_key] += 1
    counters["prefilter_qwen_calls_skipped"] += int(after == 0)


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
    entity_by_id = {entity.entity_id: entity for entity in clip.annotation.entities}
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


@dataclass(frozen=True)
class _CrossPairDonor:
    clip: ClipRecord
    entity: AnnotationEntity
    reference: EntityReferenceState
    image_path: Path


def _natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _build_same_parent_donor_index(
    config: V3Config,
    storage: RunStorage,
) -> dict[tuple[str, str], tuple[_CrossPairDonor, ...]]:
    donors_by_key: dict[tuple[str, str], list[_CrossPairDonor]] = {}
    for initial_donor in storage.iter_clips():
        donor_clip = storage.read_clip(initial_donor.clip_uid)
        if (
            donor_clip.annotation is None
            or donor_clip.annotation.status != "ready"
            or donor_clip.pairing is None
            or donor_clip.pairing.status != "ready"
        ):
            continue
        donor_states = {
            state.entity_id: state for state in donor_clip.references.entities
        }
        eligible_references: list[tuple[AnnotationEntity, EntityReferenceState]] = []
        for donor_entity in donor_clip.annotation.entities:
            donor_state = donor_states.get(donor_entity.entity_id)
            is_legacy = (
                donor_state is not None
                and donor_state.source_clip_uid is None
                and donor_state.source_entity_id is None
            )
            is_self = (
                donor_state is not None
                and donor_state.source_clip_uid == donor_clip.clip_uid
                and donor_state.source_entity_id == donor_entity.entity_id
            )
            if (
                donor_state is None
                or donor_state.status != "ready"
                or donor_state.reference_scope != "full"
                or not donor_state.identity_features_visible
                or donor_state.synthetic
                or donor_entity.entity_id not in donor_clip.pairing.retained_entity_ids
                or not (is_legacy or is_self)
            ):
                continue
            eligible_references.append((donor_entity, donor_state))
        if not eligible_references:
            continue
        try:
            donor_frames = validate_sampled_frames(
                storage,
                donor_clip.clip_uid,
            )
            donor_masks = storage.read_masks(donor_clip.clip_uid)
            _validate_pair_inputs(donor_clip, donor_frames, donor_masks)
        except Exception:  # noqa: BLE001, S112 - skip invalid donor clip
            continue
        for donor_entity, donor_state in eligible_references:
            try:
                validate_entity_reference_artifact(
                    config,
                    storage,
                    donor_clip.clip_uid,
                    donor_entity,
                    donor_state,
                    donor_frames,
                    donor_masks,
                )
            except Exception:  # noqa: BLE001, S112 - skip invalid donors
                continue
            key = (
                donor_clip.source.parent_video_id,
                donor_entity.reference_type,
            )
            donors_by_key.setdefault(key, []).append(
                _CrossPairDonor(
                    clip=donor_clip,
                    entity=donor_entity,
                    reference=donor_state,
                    image_path=storage.selected_entity_path(
                        donor_clip.clip_uid,
                        donor_entity.entity_id,
                    ),
                )
            )
    return {
        key: tuple(
            sorted(
                donors,
                key=lambda donor: (
                    _natural_sort_key(donor.clip.source.clip_suffix),
                    donor.clip.clip_uid,
                    donor.entity.entity_id,
                ),
            )
        )
        for key, donors in donors_by_key.items()
    }


def _donors_for_target(
    config: V3Config,
    donor_index: dict[tuple[str, str], tuple[_CrossPairDonor, ...]],
    *,
    target_clip: ClipRecord,
    target_entity: AnnotationEntity,
) -> tuple[_CrossPairDonor, ...]:
    key = (
        target_clip.source.parent_video_id,
        target_entity.reference_type,
    )
    selected: list[_CrossPairDonor] = []
    for donor in donor_index.get(key, ()):
        if donor.clip.clip_uid == target_clip.clip_uid:
            continue
        selected.append(donor)
        if len(selected) == config.pair.same_parent_max_donor_references:
            break
    return tuple(selected)


def _write_cross_pair_debug(
    storage: RunStorage,
    *,
    target_clip: ClipRecord,
    target_entity: AnnotationEntity,
    target_evidence_mode: CrossPairTargetEvidenceMode,
    target_frame_slots: tuple[int, ...],
    target_context_image: Image.Image,
    donor: _CrossPairDonor,
    attempt: CrossPairDecisionAttempt | None = None,
    failure: CrossPairJudgeFailure | None = None,
) -> None:
    if not storage.config.debug.save_diagnostics:
        return
    root = storage.pair_debug_dir(
        target_clip.clip_uid,
        target_entity.entity_id,
    )
    safe_donor = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{donor.clip.clip_uid}-{donor.entity.entity_id}",
    )
    directory = root / "cross_pair" / safe_donor
    directory.mkdir(parents=True, exist_ok=True)
    if target_evidence_mode == "sampled_frames":
        target_context_image.save(
            directory / "target_contact_sheet.png",
            format="PNG",
        )
    raw_responses = (
        list(attempt.raw_responses)
        if attempt is not None
        else list(failure.raw_responses if failure is not None else [])
    )
    issues = [] if failure is None else [issue.to_dict() for issue in failure.issues]
    write_json_atomic(
        directory / "decision.json",
        {
            "target_evidence_mode": target_evidence_mode,
            "target_frame_slots": list(target_frame_slots),
            "target": {
                "clip_uid": target_clip.clip_uid,
                "entity_id": target_entity.entity_id,
                "reference_type": target_entity.reference_type,
                "phrase": target_entity.phrase,
                "grounding_prompt": target_entity.grounding_prompt,
            },
            "donor": {
                "clip_uid": donor.clip.clip_uid,
                "entity_id": donor.entity.entity_id,
                "reference_type": donor.entity.reference_type,
                "phrase": donor.entity.phrase,
                "grounding_prompt": donor.entity.grounding_prompt,
                "image_path": storage.relative_artifact_path(donor.image_path),
            },
            "raw_responses": raw_responses,
            "issues": issues,
            "repair_attempts": (
                attempt.repair_attempts if attempt is not None else None
            ),
            "decision": (
                attempt.decision.model_dump(mode="json")
                if attempt is not None
                else None
            ),
        },
    )


def _pairing_from_references(
    clip: ClipRecord,
    entity_states: list[EntityReferenceState],
    *,
    bind_ready_background: bool,
) -> PairingState:
    if clip.annotation is None or clip.coverage is None:
        raise ValueError("cross-pair target inputs are incomplete")
    retained = [state.entity_id for state in entity_states if state.status == "ready"]
    if not set(retained).intersection(clip.coverage.qualifying_entity_ids):
        return PairingState(
            status="rejected",
            reason="no_qualifying_ready_reference",
        )
    entity_by_id = {entity.entity_id: entity for entity in clip.annotation.entities}
    background = clip.references.background
    background_token = (
        "<ref_bg_1>"
        if bind_ready_background
        and background is not None
        and background.status in {"clean_raw", "ready_removed"}
        else None
    )
    return PairingState(
        status="ready",
        retained_entity_ids=retained,
        tokens=_tokens_for_retained(retained, entity_by_id),
        background_token=background_token,
    )


def _publish_cross_pair_result(
    storage: RunStorage,
    *,
    clip_uid: str,
    references: ReferencesState,
    pairing: PairingState,
    donor_paths: dict[str, Path],
) -> None:
    temporary_paths: dict[str, Path] = {}
    expected_bytes: dict[str, bytes] = {}
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for entity_id, donor_path in donor_paths.items():
            donor_bytes = donor_path.read_bytes()
            temporary = storage.pair_output_temporary_path(
                clip_uid,
                entity_id,
            )
            shutil.copyfile(donor_path, temporary)
            if temporary.read_bytes() != donor_bytes:
                raise ValueError("cross-pair temporary PNG bytes changed during copy")
            temporary_paths[entity_id] = temporary
            expected_bytes[entity_id] = donor_bytes
        for entity_id, temporary in temporary_paths.items():
            final = storage.selected_entity_path(clip_uid, entity_id)
            if final.exists():
                backup = storage.pair_output_backup_path(clip_uid, entity_id)
                final.replace(backup)
                backups[final] = backup
            temporary.replace(final)
            published.append(final)
            if final.read_bytes() != expected_bytes[entity_id]:
                raise ValueError("cross-pair target PNG bytes changed during publish")
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
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        storage.cleanup_pair_artifacts(clip_uid)


def _run_same_parent_cross_pair_fallback(
    config: V3Config,
    storage: RunStorage,
    *,
    target_clip_uids: set[str],
    counters: dict[str, int],
    judge: CrossPairJudge | None,
    background_guard: _BackgroundFinalGuardRuntime,
) -> None:
    if not config.pair.same_parent_fallback_enabled or not target_clip_uids:
        return
    donor_index = _build_same_parent_donor_index(config, storage)
    active_judge = judge
    owned_judge: QwenCrossPairJudge | None = None
    try:
        for target_clip_uid in sorted(target_clip_uids):
            target_clip = storage.read_clip(target_clip_uid)
            temporary_donors: dict[str, Path] = {}
            try:
                if (
                    target_clip.annotation is None
                    or target_clip.annotation.status != "ready"
                    or target_clip.coverage is None
                    or not target_clip.coverage.passed
                    or target_clip.pairing is None
                ):
                    continue
                target_frames = validate_sampled_frames(
                    storage,
                    target_clip.clip_uid,
                )
                target_masks = storage.read_masks(target_clip.clip_uid)
                _validate_pair_inputs(target_clip, target_frames, target_masks)
                states_by_id = {
                    state.entity_id: state for state in target_clip.references.entities
                }
                original_statuses = {
                    entity_id: state.status for entity_id, state in states_by_id.items()
                }
                for target_entity in target_clip.annotation.entities:
                    current_state = states_by_id.get(target_entity.entity_id)
                    if (
                        current_state is not None
                        and current_state.status == "ready"
                        and current_state.reference_scope == "full"
                    ):
                        continue
                    target_candidates = build_entity_reference_candidates(
                        config,
                        storage,
                        clip_uid=target_clip.clip_uid,
                        entity=target_entity,
                        frames=target_frames,
                        masks=target_masks,
                    )
                    donors = _donors_for_target(
                        config,
                        donor_index,
                        target_clip=target_clip,
                        target_entity=target_entity,
                    )
                    if not donors:
                        continue
                    if target_candidates:
                        target_evidence_mode: CrossPairTargetEvidenceMode = (
                            "masked_candidate"
                        )
                        target_candidate = target_candidates[0]
                        target_frame_slots = (target_candidate.frame_slot,)
                        target_sources = _load_source_images(
                            storage,
                            [target_candidate],
                        )
                        target_source = target_sources[target_candidate.image_path]
                        target_context = build_candidate_context_image(
                            target_source,
                            target_candidate.mask,
                        )
                        target_crop, _ = build_reference_crop(
                            target_source,
                            target_candidate.mask,
                            crop_padding_ratio=config.pair.crop_padding_ratio,
                        )
                    else:
                        target_evidence_mode = "sampled_frames"
                        target_frame_images = _load_cross_pair_target_frame_images(
                            storage,
                            clip_uid=target_clip.clip_uid,
                            frames=target_frames,
                        )
                        target_frame_slots = tuple(
                            slot for slot, _ in target_frame_images
                        )
                        target_context = build_cross_pair_target_contact_sheet(
                            target_frame_images
                        )
                        target_crop = None
                    for donor in donors:
                        if active_judge is None:
                            judge_config = config.qwen.cross_pair_judge
                            if judge_config is None:
                                raise RuntimeError("cross-pair judge is not configured")
                            owned_judge = QwenCrossPairJudge(
                                judge_config,
                                repair_retries=config.pair.repair_retries,
                            )
                            active_judge = owned_judge
                        with Image.open(donor.image_path) as opened:
                            donor_image = opened.convert("RGBA")
                            donor_image.load()
                        counters["cross_pair_attempted"] += 1
                        try:
                            attempt = active_judge.decide(
                                target_clip_uid=target_clip.clip_uid,
                                target_entity=target_entity,
                                target_evidence_mode=target_evidence_mode,
                                target_context_image=target_context,
                                target_entity_crop=target_crop,
                                donor_clip_uid=donor.clip.clip_uid,
                                donor_entity=donor.entity,
                                donor_reference_image=donor_image,
                            )
                        except CrossPairJudgeFailure as exc:
                            _write_cross_pair_debug(
                                storage,
                                target_clip=target_clip,
                                target_entity=target_entity,
                                target_evidence_mode=target_evidence_mode,
                                target_frame_slots=target_frame_slots,
                                target_context_image=target_context,
                                donor=donor,
                                failure=exc,
                            )
                            raise
                        _write_cross_pair_debug(
                            storage,
                            target_clip=target_clip,
                            target_entity=target_entity,
                            target_evidence_mode=target_evidence_mode,
                            target_frame_slots=target_frame_slots,
                            target_context_image=target_context,
                            donor=donor,
                            attempt=attempt,
                        )
                        counters["cross_pair_repaired"] += int(
                            attempt.repair_attempts > 0
                        )
                        if attempt.decision.verdict != "accept":
                            continue
                        donor_state = donor.reference
                        states_by_id[target_entity.entity_id] = EntityReferenceState(
                            entity_id=target_entity.entity_id,
                            status="ready",
                            reference_scope=donor_state.reference_scope,
                            visible_region=donor_state.visible_region,
                            whole_entity_recognizable=(
                                donor_state.whole_entity_recognizable
                            ),
                            identity_features_visible=(
                                donor_state.identity_features_visible
                            ),
                            scope_reason=donor_state.scope_reason,
                            viewpoint=donor_state.viewpoint,
                            independent_reference_value=(
                                donor_state.independent_reference_value
                            ),
                            requires_substantial_invention=(
                                donor_state.requires_substantial_invention
                            ),
                            primary_identity_region_visible=(
                                donor_state.primary_identity_region_visible
                            ),
                            major_structure_visible=(
                                donor_state.major_structure_visible
                            ),
                            truncation_severity=donor_state.truncation_severity,
                            discrete_foreground_instance=(
                                donor_state.discrete_foreground_instance
                            ),
                            mask_matches_target=donor_state.mask_matches_target,
                            completion_needed_for_reference_use=(
                                donor_state.completion_needed_for_reference_use
                            ),
                            detached_target_fragments_present=(
                                donor_state.detached_target_fragments_present
                            ),
                            image_path=storage.relative_artifact_path(
                                storage.selected_entity_path(
                                    target_clip.clip_uid,
                                    target_entity.entity_id,
                                )
                            ),
                            source_frame_index=(donor_state.source_frame_index),
                            synthetic=False,
                            source_clip_uid=donor.clip.clip_uid,
                            source_entity_id=donor.entity.entity_id,
                            image_quality=(donor_state.image_quality or "high"),
                            completeness="complete",
                        )
                        temporary_donors[target_entity.entity_id] = donor.image_path
                        break
                if not temporary_donors:
                    continue
                entity_states = [
                    states_by_id[entity.entity_id]
                    for entity in target_clip.annotation.entities
                ]
                pairing = _pairing_from_references(
                    target_clip,
                    entity_states,
                    bind_ready_background=False,
                )
                if pairing.status == "ready":
                    pairing = pairing.model_copy(
                        update={
                            "background_token": (
                                background_guard.token_for_ready_pairing(
                                    clip=target_clip,
                                    frames=target_frames,
                                )
                            )
                        }
                    )
                references = ReferencesState(
                    entities=entity_states,
                    background=target_clip.references.background,
                )
                previous_status = target_clip.pairing.status
                _publish_cross_pair_result(
                    storage,
                    clip_uid=target_clip.clip_uid,
                    references=references,
                    pairing=pairing,
                    donor_paths=temporary_donors,
                )
                added = len(temporary_donors)
                newly_ready = sum(
                    original_statuses[entity_id] != "ready"
                    for entity_id in temporary_donors
                )
                counters["cross_pair_ready"] += added
                counters["entities_ready"] += newly_ready
                counters["entities_rejected"] -= newly_ready
                if previous_status != pairing.status:
                    counters[previous_status] -= 1
                    counters[pairing.status] += 1
                    if (
                        previous_status == "rejected"
                        and pairing.status == "ready"
                        and pairing.background_token is not None
                    ):
                        counters["backgrounds_bound"] += 1
            except Exception as exc:  # noqa: BLE001 - isolate target clips
                storage.cleanup_pair_artifacts(target_clip.clip_uid)
                storage.append_failure(
                    stage="pair",
                    clip_uid=target_clip.clip_uid,
                    reason=str(exc),
                    details=_failure_details(exc),
                )
                counters["failed"] += 1

    finally:
        if owned_judge is not None:
            owned_judge.close()


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
    if isinstance(
        exc,
        (EntityReferenceJudgeFailure, CrossPairJudgeFailure),
    ):
        return exc.to_dict()
    return {"exception_type": type(exc).__name__}


def pair_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    judge: EntityReferenceJudge | None = None,
    cross_pair_judge: CrossPairJudge | None = None,
    completion_backend: QwenReferenceCompletionBackend | None = None,
    completion_judge: QwenLocalizedCompletionJudge | None = None,
    completion_segmentation_backend: SegmentationBackend | None = None,
    background_final_judge: FinalBackgroundJudge | None = None,
) -> PairStats:
    config.validate()
    if storage.root != config.resolved_run_root:
        raise ValueError("storage run_root does not match pair configuration")
    if not config.pair.enabled:
        raise ValueError("V3 pair stage is disabled")
    counters = {field: 0 for field in PairStats.__dataclass_fields__}
    cross_pair_targets: set[str] = set()
    active_judge = judge
    owned_judge: QwenEntityReferenceJudge | None = None
    background_guard = _BackgroundFinalGuardRuntime(
        config,
        storage,
        counters,
        background_final_judge,
    )
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
                    (
                        candidates,
                        all_candidates_tiny,
                        all_candidates_fragmented,
                    ) = _build_entity_reference_candidates(
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
                                (
                                    "tiny_reference_candidates"
                                    if all_candidates_tiny
                                    else "fragmented_reference_candidates"
                                    if all_candidates_fragmented
                                    else "no_valid_reference_candidate"
                                ),
                            )
                        )
                        continue
                    source_images = _load_source_images(storage, candidates)
                    judged_candidates = candidates
                    if config.pair.reference_prefilter_mode == "conservative_v1":
                        try:
                            prefilter_result = prefilter_entity_reference_candidates(
                                entity,
                                candidates,
                                source_images,
                            )
                        except Exception as exc:  # noqa: BLE001 - required fail-open
                            counters["prefilter_candidates_examined"] += len(
                                candidates
                            )
                            counters["prefilter_fail_open_entities"] += 1
                            _write_prefilter_debug(
                                storage,
                                clip_uid=clip.clip_uid,
                                entity=entity,
                                original_candidates=candidates,
                                result=None,
                                error=exc,
                            )
                        else:
                            _record_prefilter_stats(counters, prefilter_result)
                            _write_prefilter_debug(
                                storage,
                                clip_uid=clip.clip_uid,
                                entity=entity,
                                original_candidates=candidates,
                                result=prefilter_result,
                            )
                            judged_candidates = list(
                                prefilter_result.retained_candidates
                            )
                            if not judged_candidates:
                                entity_states.append(
                                    _rejected_reference(
                                        entity.entity_id,
                                        "reference_prefilter_all_candidates_filtered",
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
                    attempt = active_judge.decide(
                        entity=entity,
                        candidates=judged_candidates,
                        source_images=source_images,
                    )
                    counters["repaired"] += int(attempt.repair_attempts > 0)
                    _write_debug_attempt(
                        storage,
                        clip_uid=clip.clip_uid,
                        entity=entity,
                        candidates=judged_candidates,
                        attempt=attempt,
                    )
                    decision = attempt.decision
                    decision_issues = validate_entity_reference_decision(
                        decision,
                        candidate_ids={
                            item.candidate_id for item in judged_candidates
                        },
                        reference_type=entity.reference_type,
                        candidate_by_id={
                            item.candidate_id: item for item in judged_candidates
                        },
                    )
                    if decision_issues:
                        messages = "; ".join(issue.message for issue in decision_issues)
                        raise ValueError(f"invalid judge decision: {messages}")
                    if decision.reference_scope == "reject":
                        entity_states.append(
                            EntityReferenceState(
                                entity_id=entity.entity_id,
                                status="rejected",
                                reference_scope="reject",
                                visible_region="custom",
                                whole_entity_recognizable=False,
                                identity_features_visible=False,
                                scope_reason=decision.scope_reason,
                                image_quality=decision.image_quality,
                                completeness=decision.completeness,
                                viewpoint=decision.viewpoint,
                                independent_reference_value=(
                                    decision.independent_reference_value
                                ),
                                requires_substantial_invention=(
                                    decision.requires_substantial_invention
                                ),
                                primary_identity_region_visible=(
                                    decision.primary_identity_region_visible
                                ),
                                major_structure_visible=(
                                    decision.major_structure_visible
                                ),
                                truncation_severity=decision.truncation_severity,
                                discrete_foreground_instance=(
                                    decision.discrete_foreground_instance
                                ),
                                mask_matches_target=decision.mask_matches_target,
                                completion_needed_for_reference_use=(
                                    decision.completion_needed_for_reference_use
                                ),
                                detached_target_fragments_present=(
                                    decision.detached_target_fragments_present
                                ),
                                synthetic=False,
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
                                decision=decision,
                            )
                        )
                        continue
                    selected = next(
                        candidate
                        for candidate in judged_candidates
                        if candidate.candidate_id == decision.selected_candidate_id
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
                            source_clip_uid=clip.clip_uid,
                            source_entity_id=entity.entity_id,
                            image_quality=decision.image_quality,
                            completeness=decision.completeness,
                            viewpoint=decision.viewpoint,
                            independent_reference_value=(
                                decision.independent_reference_value
                            ),
                            requires_substantial_invention=(
                                decision.requires_substantial_invention
                            ),
                            primary_identity_region_visible=(
                                decision.primary_identity_region_visible
                            ),
                            major_structure_visible=(
                                decision.major_structure_visible
                            ),
                            truncation_severity=decision.truncation_severity,
                            discrete_foreground_instance=(
                                decision.discrete_foreground_instance
                            ),
                            mask_matches_target=decision.mask_matches_target,
                            completion_needed_for_reference_use=(
                                decision.completion_needed_for_reference_use
                            ),
                            detached_target_fragments_present=(
                                decision.detached_target_fragments_present
                            ),
                        )
                    )
                retained = [
                    state.entity_id
                    for state in entity_states
                    if state.status == "ready"
                ]
                if not set(retained).intersection(clip.coverage.qualifying_entity_ids):
                    pairing = PairingState(
                        status="rejected",
                        reason="no_qualifying_ready_reference",
                    )
                else:
                    background_token = background_guard.token_for_ready_pairing(
                        clip=clip,
                        frames=frames,
                    )
                    entity_by_id = {
                        entity.entity_id: entity for entity in clip.annotation.entities
                    }
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(retained, entity_by_id),
                        background_token=background_token,
                    )
                references = ReferencesState(
                    entities=entity_states,
                    background=clip.references.background,
                )
                _publish_pair_result(
                    config,
                    storage,
                    clip_uid=clip.clip_uid,
                    references=references,
                    pairing=pairing,
                    temporary_images=temporary_images,
                )
                ready_count = sum(state.status == "ready" for state in entity_states)
                counters["entities_ready"] += ready_count
                counters["entities_rejected"] += len(entity_states) - ready_count
                counters[pairing.status] += 1
                counters["backgrounds_bound"] += int(
                    pairing.background_token is not None
                )
                cross_pair_targets.add(clip.clip_uid)
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
        _run_same_parent_cross_pair_fallback(
            config,
            storage,
            target_clip_uids=cross_pair_targets,
            counters=counters,
            judge=cross_pair_judge,
            background_guard=background_guard,
        )
        if not config.reference_edit.enabled:
            completion_stats = run_reference_completion_fallbacks(
                config,
                storage,
                target_clip_uids=cross_pair_targets,
                reference_judge=active_judge,
                completion_backend=completion_backend,
                completion_judge=completion_judge,
                segmentation_backend=completion_segmentation_backend,
            )
            counters["completion_attempted"] += completion_stats.attempted
            counters["completion_ready"] += completion_stats.ready
            counters["completion_rejected"] += completion_stats.rejected
        if config.pair.background_final_guard_mode == "qwen_v1":
            for clip_uid in sorted(
                cross_pair_targets - background_guard.evaluated_clip_uids
            ):
                completed_clip = storage.read_clip(clip_uid)
                if (
                    completed_clip.pairing is None
                    or completed_clip.pairing.status != "ready"
                    or not completed_clip.pairing.retained_entity_ids
                ):
                    continue
                background = completed_clip.references.background
                if background is None or background.status not in {
                    "clean_raw",
                    "ready_removed",
                }:
                    continue
                completed_frames = validate_sampled_frames(storage, clip_uid)
                background_token = background_guard.token_for_ready_pairing(
                    clip=completed_clip,
                    frames=completed_frames,
                )
                if background_token is not None:
                    storage.write_pairing(
                        clip_uid,
                        completed_clip.pairing.model_copy(
                            update={"background_token": background_token}
                        ),
                    )
                    counters["backgrounds_bound"] += 1
    finally:
        if owned_judge is not None:
            owned_judge.close()
        background_guard.close()
    stats = PairStats(**counters)
    storage.update_stage_counts("pair", stats.to_dict())
    return stats
