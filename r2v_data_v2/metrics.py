from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from r2v_data_v2.mask_utils import bbox_from_mask, touches_border


@dataclass(frozen=True)
class CandidateMetrics:
    effective_short_side: int
    mask_area_ratio: float
    border_touch: bool
    laplacian_variance: float
    tenengrad_sharpness: float
    exposure_score: float
    mask_area_continuity: float
    maximum_other_mask_overlap: float
    maximum_bbox_containment: float
    crop_subject_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def padded_crop_box(
    bbox_xyxy: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    padding_x = round((x2 - x1) * padding_ratio)
    padding_y = round((y2 - y1) * padding_ratio)
    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def _bbox_containment(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(1, (lx2 - lx1) * (ly2 - ly1))
    return intersection / left_area


def calculate_candidate_metrics(
    *,
    frame: np.ndarray,
    mask: np.ndarray,
    area_median: float,
    other_masks: list[np.ndarray] | None = None,
) -> CandidateMetrics:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != frame.shape[:2] or not binary.any():
        raise ValueError("candidate mask must be nonempty and match the frame")
    bbox = bbox_from_mask(binary)
    x1, y1, x2, y2 = bbox
    crop_box = padded_crop_box(
        bbox,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )
    cx1, cy1, cx2, cy2 = crop_box
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    foreground_pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[binary]
    clipped = np.mean((foreground_pixels <= 8) | (foreground_pixels >= 247))
    exposure = float(max(0.0, 1.0 - clipped))
    area_ratio = float(binary.mean())
    continuity = float(
        max(0.0, 1.0 - abs(area_ratio - area_median) / max(area_median, 1e-9))
    )
    maximum_overlap = 0.0
    maximum_containment = 0.0
    for other in other_masks or []:
        other_binary = np.asarray(other, dtype=bool)
        if other_binary.shape != binary.shape or not other_binary.any():
            continue
        maximum_overlap = max(
            maximum_overlap,
            float(np.logical_and(binary, other_binary).sum() / binary.sum()),
        )
        maximum_containment = max(
            maximum_containment,
            _bbox_containment(bbox, bbox_from_mask(other_binary)),
        )
    crop_area = max(1, (cx2 - cx1) * (cy2 - cy1))
    return CandidateMetrics(
        effective_short_side=min(x2 - x1, y2 - y1),
        mask_area_ratio=area_ratio,
        border_touch=touches_border(binary),
        laplacian_variance=laplacian,
        tenengrad_sharpness=tenengrad,
        exposure_score=exposure,
        mask_area_continuity=continuity,
        maximum_other_mask_overlap=maximum_overlap,
        maximum_bbox_containment=maximum_containment,
        crop_subject_ratio=float(binary[cy1:cy2, cx1:cx2].sum() / crop_area),
    )


def overlap_statistics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float]:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    intersection = float(np.logical_and(left_mask, right_mask).sum())
    union = float(np.logical_or(left_mask, right_mask).sum())
    return {
        "mask_iou": intersection / max(union, 1.0),
        "left_contained_in_right": intersection / max(float(left_mask.sum()), 1.0),
        "right_contained_in_left": intersection / max(float(right_mask.sum()), 1.0),
        "bbox_iou": _bbox_iou(
            bbox_from_mask(left_mask),
            bbox_from_mask(right_mask),
        ),
    }


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1)


def classify_entity_overlap(
    *,
    statistics: dict[str, float],
    temporal_cooccurrence: float,
    relation: str | None,
    child_has_independent_candidate: bool,
) -> Literal[
    "independent",
    "attached_accessory",
    "important_independent_object",
    "composite_candidate",
    "duplicate_entity",
]:
    mask_iou = statistics["mask_iou"]
    containment = max(
        statistics["left_contained_in_right"],
        statistics["right_contained_in_left"],
    )
    if mask_iou >= 0.85:
        return "duplicate_entity"
    attached_relation = relation in {"holding", "wearing", "carrying", "attached to"}
    if containment >= 0.75 and temporal_cooccurrence >= 0.6:
        if attached_relation and not child_has_independent_candidate:
            return "attached_accessory"
        if child_has_independent_candidate:
            return "important_independent_object"
        return "composite_candidate"
    if mask_iou >= 0.3 and temporal_cooccurrence >= 0.6:
        return "composite_candidate"
    return "independent"
