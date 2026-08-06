from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ContentGeometry:
    bbox_xyxy: tuple[int, int, int, int]
    width: int
    height: int
    area_pixels: int
    normalized_bbox: tuple[float, float, float, float]
    normalized_bbox_area: float
    center_xy: tuple[float, float]
    touches_canvas_boundary: bool


def content_geometry_from_mask(mask: np.ndarray) -> ContentGeometry:
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("content mask must be two-dimensional")
    if not np.isin(binary, (False, True, 0, 1)).all():
        raise ValueError("content mask must be binary")
    binary = binary.astype(bool, copy=False)
    rows, columns = np.nonzero(binary)
    if not rows.size:
        raise ValueError("content mask must not be empty")
    canvas_height, canvas_width = binary.shape
    x1 = int(columns.min())
    y1 = int(rows.min())
    x2 = int(columns.max()) + 1
    y2 = int(rows.max()) + 1
    width = x2 - x1
    height = y2 - y1
    normalized_bbox = (
        x1 / canvas_width,
        y1 / canvas_height,
        x2 / canvas_width,
        y2 / canvas_height,
    )
    return ContentGeometry(
        bbox_xyxy=(x1, y1, x2, y2),
        width=width,
        height=height,
        area_pixels=width * height,
        normalized_bbox=normalized_bbox,
        normalized_bbox_area=(width / canvas_width) * (height / canvas_height),
        center_xy=(
            (normalized_bbox[0] + normalized_bbox[2]) / 2,
            (normalized_bbox[1] + normalized_bbox[3]) / 2,
        ),
        touches_canvas_boundary=(
            x1 == 0 or y1 == 0 or x2 == canvas_width or y2 == canvas_height
        ),
    )


def content_geometry_from_rgba(image: Image.Image) -> ContentGeometry:
    if not isinstance(image, Image.Image) or image.mode != "RGBA":
        raise ValueError("source content geometry requires an RGBA image")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    return content_geometry_from_mask(alpha > 0)


def tiny_content_reason(
    geometry: ContentGeometry,
    *,
    minimum_area_pixels: int,
    minimum_long_side_pixels: int,
) -> str | None:
    for value, name in (
        (minimum_area_pixels, "minimum_area_pixels"),
        (minimum_long_side_pixels, "minimum_long_side_pixels"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    area_tiny = geometry.area_pixels < minimum_area_pixels
    long_side_tiny = max(geometry.width, geometry.height) < minimum_long_side_pixels
    if area_tiny and long_side_tiny:
        return "source_content_area_and_long_side_below_minimum"
    if area_tiny:
        return "source_content_area_below_minimum"
    if long_side_tiny:
        return "source_content_long_side_below_minimum"
    return None


def source_geometry_metadata(
    geometry: ContentGeometry,
    *,
    source_gate_reason: str | None,
) -> dict[str, object]:
    return {
        "source_content_bbox_xyxy": list(geometry.bbox_xyxy),
        "source_content_width": geometry.width,
        "source_content_height": geometry.height,
        "source_content_area_pixels": geometry.area_pixels,
        "source_tiny": source_gate_reason is not None,
        "source_gate_reason": source_gate_reason or "passed",
    }


def composition_geometry_metadata(
    source: ContentGeometry,
    candidate: ContentGeometry,
    *,
    minimum_scale_ratio: float,
    maximum_center_shift: float,
) -> dict[str, object]:
    if (
        not isinstance(minimum_scale_ratio, float)
        or not math.isfinite(minimum_scale_ratio)
        or not 0 < minimum_scale_ratio <= 1
    ):
        raise ValueError("minimum_scale_ratio must be a finite float in (0, 1]")
    if (
        not isinstance(maximum_center_shift, float)
        or not math.isfinite(maximum_center_shift)
        or not 0 <= maximum_center_shift <= math.sqrt(2)
    ):
        raise ValueError(
            "maximum_center_shift must be a finite float between 0 and sqrt(2)"
        )
    scale_ratio = candidate.normalized_bbox_area / source.normalized_bbox_area
    center_shift = math.dist(source.center_xy, candidate.center_xy)
    rejection_reason: str | None = None
    if scale_ratio < minimum_scale_ratio:
        rejection_reason = "entity_scale_collapsed"
    elif center_shift > maximum_center_shift:
        rejection_reason = "entity_shifted_off_layout"
    return {
        "source_normalized_bbox": list(source.normalized_bbox),
        "candidate_normalized_bbox": list(candidate.normalized_bbox),
        "source_normalized_bbox_area": source.normalized_bbox_area,
        "candidate_normalized_bbox_area": candidate.normalized_bbox_area,
        "candidate_scale_ratio": scale_ratio,
        "source_center_xy": list(source.center_xy),
        "candidate_center_xy": list(candidate.center_xy),
        "center_shift": center_shift,
        "geometry_gate_passed": rejection_reason is None,
        "geometry_rejection_reason": rejection_reason,
    }
