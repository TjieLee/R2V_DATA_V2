"""Deterministic conservative geometry policy for localized completion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from r2v_data_v2.v3.reference_completion_benchmark import (
    build_completion_canvas,
)
from r2v_data_v2.v3.reference_completion_publish import (
    compute_foreground_metrics,
)

CompletionDirection = Literal["top", "bottom", "left", "right"]

_DIRECTION_ORDER: tuple[CompletionDirection, ...] = (
    "top",
    "bottom",
    "left",
    "right",
)
_MIN_LARGEST_COMPONENT_RATIO = 0.75
_MAX_COMPONENT_COUNT = 4
_MIN_VISIBLE_AREA_RATIO = 0.01
_MIN_VISIBLE_PIXELS = 16
_PROTECTED_COMPONENT_AREA_RATIO = 0.03
_MIN_PROTECTED_COMPONENT_PIXELS = 512
_NEAR_COMPONENT_RATIO = 0.10
_NEAR_DIRECTION_RATIO = 0.15
_RECOVERY_EXPANSION_RATIO = 0.20
_RECOVERY_LATERAL_RATIO = 0.10
_MIN_DIRECTIONAL_GROWTH_RATIO = 0.02
_MAX_DIRECTIONAL_GROWTH_RATIO = 0.35
_OUTSIDE_CORRIDOR_TOLERANCE_RATIO = 0.005
_SIGNIFICANT_NEW_COMPONENT_RATIO = 0.01


@dataclass(frozen=True)
class _CanvasConfig:
    canvas_expand_ratio: float = _RECOVERY_EXPANSION_RATIO
    lateral_padding_ratio: float = _RECOVERY_LATERAL_RATIO
    mask_overlap_pixels: int = 0
    model_min_side: int = 16
    model_multiple: int = 16

    def validate(self) -> None:
        if not 0.10 <= self.canvas_expand_ratio <= 0.25:
            raise ValueError("recovery expansion ratio must be between 0.10 and 0.25")
        if not 0 <= self.lateral_padding_ratio <= 0.25:
            raise ValueError("recovery lateral ratio must be between 0 and 0.25")
        if self.mask_overlap_pixels != 0:
            raise ValueError("conservative recovery must not overlap source pixels")


@dataclass(frozen=True)
class _Component:
    component_id: str
    label: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ConservativeCompletionPolicy:
    eligible: bool
    completion_skipped_reason: str | None
    source_rgba: Image.Image
    cleaned_source_rgba: Image.Image
    completion_input_rgb: Image.Image
    protected_component_mask: Image.Image
    recovery_corridor_mask: Image.Image
    recovery_directions: tuple[CompletionDirection, ...]
    source_offset_xy: tuple[int, int]
    source_metrics: dict[str, object]
    source_component_stats: tuple[dict[str, object], ...]
    protected_component_ids: tuple[str, ...]

    def diagnostics(self) -> dict[str, object]:
        return {
            "source_component_stats": {
                "metrics": self.source_metrics,
                "components": list(self.source_component_stats),
            },
            "protected_component_ids": list(self.protected_component_ids),
            "recovery_direction": list(self.recovery_directions),
            "source_offset_xy": list(self.source_offset_xy),
            "completion_input_size": list(self.completion_input_rgb.size),
            "completion_skipped_reason": self.completion_skipped_reason,
        }


@dataclass(frozen=True)
class ConservativeCandidateValidation:
    mask: Image.Image
    passed: bool
    reasons: tuple[str, ...]
    report: dict[str, object]


def _binary_mask(image: Image.Image, *, mode_name: str) -> np.ndarray:
    if not isinstance(image, Image.Image) or image.mode != "L":
        raise TypeError(f"{mode_name} must be an L-mode PIL image")
    values = np.asarray(image, dtype=np.uint8)
    if not np.isin(values, (0, 255)).all():
        raise ValueError(f"{mode_name} must be binary")
    return values == 255


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, tuple[_Component, ...]]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    raw_components: list[tuple[int, int, tuple[int, int, int, int]]] = []
    next_label = 0
    for row in range(height):
        for column in range(width):
            if not binary[row, column] or labels[row, column] != 0:
                continue
            next_label += 1
            labels[row, column] = next_label
            stack = [(row, column)]
            area = 0
            left = right = column
            top = bottom = row
            while stack:
                current_row, current_column = stack.pop()
                area += 1
                left = min(left, current_column)
                right = max(right, current_column)
                top = min(top, current_row)
                bottom = max(bottom, current_row)
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == 0 and column_offset == 0:
                            continue
                        neighbor_row = current_row + row_offset
                        neighbor_column = current_column + column_offset
                        if (
                            0 <= neighbor_row < height
                            and 0 <= neighbor_column < width
                            and binary[neighbor_row, neighbor_column]
                            and labels[neighbor_row, neighbor_column] == 0
                        ):
                            labels[neighbor_row, neighbor_column] = next_label
                            stack.append((neighbor_row, neighbor_column))
            raw_components.append(
                (
                    next_label,
                    area,
                    (left, top, right + 1, bottom + 1),
                )
            )
    raw_components.sort(key=lambda item: (-item[1], item[2], item[0]))
    components = tuple(
        _Component(
            component_id=f"component_{index}",
            label=label,
            area=area,
            bbox=bbox,
        )
        for index, (label, area, bbox) in enumerate(raw_components, start=1)
    )
    return labels, components


def _bbox_gap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0)
    vertical = max(first[1] - second[3], second[1] - first[3], 0)
    return math.hypot(horizontal, vertical)


def _near_recovery_direction(
    bbox: tuple[int, int, int, int],
    main_bbox: tuple[int, int, int, int],
    directions: tuple[CompletionDirection, ...],
) -> bool:
    main_width = main_bbox[2] - main_bbox[0]
    main_height = main_bbox[3] - main_bbox[1]
    horizontal_margin = max(2, math.ceil(main_width * _NEAR_DIRECTION_RATIO))
    vertical_margin = max(2, math.ceil(main_height * _NEAR_DIRECTION_RATIO))
    return any(
        (
            direction == "top"
            and bbox[1] <= main_bbox[1] + vertical_margin
        )
        or (
            direction == "bottom"
            and bbox[3] >= main_bbox[3] - vertical_margin
        )
        or (
            direction == "left"
            and bbox[0] <= main_bbox[0] + horizontal_margin
        )
        or (
            direction == "right"
            and bbox[2] >= main_bbox[2] - horizontal_margin
        )
        for direction in directions
    )


def _white_rgb(source_rgba: Image.Image) -> Image.Image:
    source = np.asarray(source_rgba, dtype=np.uint8)
    output = np.full(source.shape[:2] + (3,), 255, dtype=np.uint8)
    visible = source[:, :, 3] == 255
    output[visible] = source[:, :, :3][visible]
    return Image.fromarray(output, mode="RGB")


def build_conservative_completion_policy(
    source_rgba: Image.Image,
    *,
    whole_entity_recognizable: bool,
) -> ConservativeCompletionPolicy:
    if not isinstance(source_rgba, Image.Image) or source_rgba.mode != "RGBA":
        raise TypeError("conservative completion source must be an RGBA image")
    source = np.asarray(source_rgba, dtype=np.uint8)
    alpha = source[:, :, 3]
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("conservative completion source alpha must be binary")
    foreground = alpha == 255
    labels, components = _label_components(foreground)
    source_metrics = compute_foreground_metrics(
        Image.fromarray(alpha.copy(), mode="L")
    )
    foreground_pixels = int(source_metrics["foreground_pixels"])
    component_count = int(source_metrics["component_count"])
    largest_component_ratio = (
        components[0].area / foreground_pixels
        if foreground_pixels and components
        else 0.0
    )
    source_metrics = {
        **source_metrics,
        "largest_component_ratio": largest_component_ratio,
        "minimum_largest_component_ratio": _MIN_LARGEST_COMPONENT_RATIO,
        "maximum_component_count": _MAX_COMPONENT_COUNT,
    }
    edge_touches = source_metrics["edge_touches"]
    if not isinstance(edge_touches, dict):
        raise TypeError("source edge-touch metrics must be a mapping")
    directions = tuple(
        direction
        for direction in _DIRECTION_ORDER
        if edge_touches.get(direction) is True
    )

    protected_threshold = max(
        _MIN_PROTECTED_COMPONENT_PIXELS,
        math.ceil(foreground_pixels * _PROTECTED_COMPONENT_AREA_RATIO),
    )
    main_bbox = components[0].bbox if components else (0, 0, 0, 0)
    main_scale = max(
        main_bbox[2] - main_bbox[0],
        main_bbox[3] - main_bbox[1],
        1,
    )
    near_threshold = max(2, math.ceil(main_scale * _NEAR_COMPONENT_RATIO))
    protected_labels: set[int] = set()
    component_stats: list[dict[str, object]] = []
    for index, component in enumerate(components):
        gap = _bbox_gap(component.bbox, main_bbox)
        protection_reasons: list[str] = []
        if index == 0:
            protection_reasons.append("main_component")
        if component.area >= protected_threshold:
            protection_reasons.append("minimum_area")
        if index > 0 and gap <= near_threshold:
            protection_reasons.append("near_main_component")
        if _near_recovery_direction(component.bbox, main_bbox, directions):
            protection_reasons.append("near_recovery_direction")
        protected = bool(protection_reasons)
        if protected:
            protected_labels.add(component.label)
        component_stats.append(
            {
                "component_id": component.component_id,
                "area_pixels": component.area,
                "area_ratio_vs_source_foreground": (
                    component.area / foreground_pixels
                    if foreground_pixels
                    else 0.0
                ),
                "bbox": list(component.bbox),
                "distance_to_main_pixels": gap,
                "protected": protected,
                "protection_reasons": protection_reasons,
            }
        )

    protected = np.isin(labels, list(protected_labels))
    cleaned = source.copy()
    cleaned[~protected, :3] = 255
    cleaned[~protected, 3] = 0
    cleaned_source = Image.fromarray(cleaned, mode="RGBA")

    minimum_visible_pixels = max(
        _MIN_VISIBLE_PIXELS,
        math.ceil(foreground.size * _MIN_VISIBLE_AREA_RATIO),
    )
    skipped_reason: str | None = None
    if whole_entity_recognizable is not True:
        skipped_reason = "completion_source_not_whole_recognizable"
    elif foreground_pixels < minimum_visible_pixels:
        skipped_reason = "completion_source_visible_region_too_small"
    elif component_count > _MAX_COMPONENT_COUNT or (
        largest_component_ratio < _MIN_LARGEST_COMPONENT_RATIO
    ):
        skipped_reason = "completion_source_fragmented"
    elif len(directions) not in {1, 2}:
        skipped_reason = "completion_missing_direction_unavailable"

    if directions and protected.any():
        canvas = build_completion_canvas(
            source_rgba=cleaned_source,
            context_rgb=None,
            completion_sides=directions,
            completion_start_ratio=1.0,
            config=_CanvasConfig(),
        )
        input_rgb = canvas.baseline_rgb
        protected_mask = canvas.visible_mask
        recovery_mask = canvas.candidate_region
        source_offset = canvas.source_offset_xy
    else:
        input_rgb = _white_rgb(cleaned_source)
        protected_mask = Image.fromarray(
            protected.astype(np.uint8) * 255,
            mode="L",
        )
        recovery_mask = Image.new("L", source_rgba.size, 0)
        source_offset = (0, 0)

    return ConservativeCompletionPolicy(
        eligible=skipped_reason is None,
        completion_skipped_reason=skipped_reason,
        source_rgba=source_rgba.copy(),
        cleaned_source_rgba=cleaned_source,
        completion_input_rgb=input_rgb,
        protected_component_mask=protected_mask,
        recovery_corridor_mask=recovery_mask,
        recovery_directions=directions,
        source_offset_xy=source_offset,
        source_metrics=source_metrics,
        source_component_stats=tuple(component_stats),
        protected_component_ids=tuple(
            item["component_id"]
            for item in component_stats
            if item["protected"] is True
        ),
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    if radius == 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for row_offset in range(radius * 2 + 1):
        for column_offset in range(radius * 2 + 1):
            result |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return result


def validate_conservative_completion_candidate(
    policy: ConservativeCompletionPolicy,
    candidate_mask: Image.Image,
) -> ConservativeCandidateValidation:
    if not policy.eligible:
        raise ValueError("cannot validate a candidate for an ineligible source")
    candidate = _binary_mask(candidate_mask, mode_name="candidate mask")
    protected = _binary_mask(
        policy.protected_component_mask,
        mode_name="protected component mask",
    )
    corridor = _binary_mask(
        policy.recovery_corridor_mask,
        mode_name="recovery corridor mask",
    )
    if candidate.shape != protected.shape or candidate.shape != corridor.shape:
        reason = "completion_abnormal_shape_growth"
        return ConservativeCandidateValidation(
            mask=policy.protected_component_mask.copy(),
            passed=False,
            reasons=(reason,),
            report={
                "candidate_size": list(candidate_mask.size),
                "expected_size": list(policy.completion_input_rgb.size),
                "reasons": [reason],
            },
        )

    source_pixels = int(protected.sum())
    new_region = candidate & ~protected
    outside_corridor = new_region & ~corridor
    labels, components = _label_components(new_region)
    protected_rows, protected_columns = np.nonzero(protected)
    source_bbox = (
        int(protected_columns.min()),
        int(protected_rows.min()),
        int(protected_columns.max()) + 1,
        int(protected_rows.max()) + 1,
    )
    source_scale = max(
        source_bbox[2] - source_bbox[0],
        source_bbox[3] - source_bbox[1],
        1,
    )
    adjacency_radius = max(1, math.ceil(source_scale * 0.01))
    adjacent_to_source = _dilate(protected, adjacency_radius) & ~protected
    significant_threshold = max(
        4,
        math.ceil(source_pixels * _SIGNIFICANT_NEW_COMPONENT_RATIO),
    )
    accepted_new = np.zeros_like(candidate, dtype=bool)
    component_stats: list[dict[str, object]] = []
    disconnected_significant = 0
    connected_significant = 0
    for component in components:
        component_mask = labels == component.label
        connected = bool(np.logical_and(component_mask, adjacent_to_source).any())
        inside_pixels = int(np.logical_and(component_mask, corridor).sum())
        outside_pixels = component.area - inside_pixels
        significant = component.area >= significant_threshold
        if connected:
            accepted_new |= component_mask & corridor
            connected_significant += int(significant)
        elif significant:
            disconnected_significant += 1
        component_stats.append(
            {
                "component_id": component.component_id,
                "area_pixels": component.area,
                "bbox": list(component.bbox),
                "connected_or_adjacent_to_source": connected,
                "inside_recovery_corridor_pixels": inside_pixels,
                "outside_recovery_corridor_pixels": outside_pixels,
                "significant": significant,
            }
        )

    outside_pixels = int(outside_corridor.sum())
    outside_tolerance = max(
        2,
        math.ceil(source_pixels * _OUTSIDE_CORRIDOR_TOLERANCE_RATIO),
    )
    accepted_new_pixels = int(accepted_new.sum())
    minimum_directional_growth = max(
        4,
        math.ceil(source_pixels * _MIN_DIRECTIONAL_GROWTH_RATIO),
    )
    directional_extensions = {
        "top": bool(accepted_new[: source_bbox[1], :].any()),
        "bottom": bool(accepted_new[source_bbox[3] :, :].any()),
        "left": bool(accepted_new[:, : source_bbox[0]].any()),
        "right": bool(accepted_new[:, source_bbox[2] :].any()),
    }
    matching_directions = [
        direction
        for direction in policy.recovery_directions
        if directional_extensions[direction]
    ]
    growth_ratio = (
        accepted_new_pixels / source_pixels if source_pixels else math.inf
    )
    reasons: list[str] = []
    if outside_pixels > outside_tolerance:
        reasons.append("completion_growth_outside_recovery_corridor")
    if disconnected_significant:
        reasons.append("completion_new_disconnected_fragments")
    if (
        accepted_new_pixels < minimum_directional_growth
        or not matching_directions
    ):
        reasons.append("completion_no_directional_improvement")
    if (
        growth_ratio > _MAX_DIRECTIONAL_GROWTH_RATIO
        or connected_significant > 1
    ):
        reasons.append("completion_abnormal_shape_growth")

    conservative_mask = protected | accepted_new
    report = {
        "source_foreground_pixels": source_pixels,
        "raw_candidate_foreground_pixels": int(candidate.sum()),
        "protected_source_pixels_restored": int(
            np.logical_and(protected, ~candidate).sum()
        ),
        "new_region_pixels": int(new_region.sum()),
        "accepted_new_region_pixels": accepted_new_pixels,
        "outside_recovery_corridor_pixels": outside_pixels,
        "outside_recovery_corridor_tolerance_pixels": outside_tolerance,
        "minimum_directional_growth_pixels": minimum_directional_growth,
        "growth_ratio_vs_source": growth_ratio,
        "maximum_growth_ratio_vs_source": _MAX_DIRECTIONAL_GROWTH_RATIO,
        "new_component_count": len(components),
        "connected_significant_component_count": connected_significant,
        "disconnected_significant_component_count": disconnected_significant,
        "new_components": component_stats,
        "directional_extensions": directional_extensions,
        "matching_recovery_directions": matching_directions,
        "reasons": reasons,
    }
    return ConservativeCandidateValidation(
        mask=Image.fromarray(conservative_mask.astype(np.uint8) * 255, mode="L"),
        passed=not reasons,
        reasons=tuple(reasons),
        report=report,
    )


def restore_protected_source_pixels(
    policy: ConservativeCompletionPolicy,
    candidate_rgb: Image.Image,
) -> Image.Image:
    if not isinstance(candidate_rgb, Image.Image) or candidate_rgb.mode != "RGB":
        raise TypeError("candidate must be an RGB image")
    if candidate_rgb.size != policy.completion_input_rgb.size:
        raise ValueError("candidate dimensions must match conservative input")
    protected = _binary_mask(
        policy.protected_component_mask,
        mode_name="protected component mask",
    )
    candidate = np.asarray(candidate_rgb, dtype=np.uint8).copy()
    baseline = np.asarray(policy.completion_input_rgb, dtype=np.uint8)
    candidate[protected] = baseline[protected]
    return Image.fromarray(candidate, mode="RGB")
