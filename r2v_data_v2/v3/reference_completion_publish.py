"""Fail-closed publication gates for localized reference completion.

This module is deliberately independent from the production V3 pipeline. It
consumes immutable artifacts from the ``localized_raw`` Qwen benchmark and
publishes a generated reference only when deterministic checks and three
independent structured judges all accept it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    build_structured_repair_prompt,
    parse_qwen_json_issues,
)
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.sam3_backend import (
    EntityTrackResult,
    SegmentationBackend,
)

ReferenceType = Literal["subject", "object", "group"]
PublicationStatus = Literal["auto_published", "rejected"]
JudgeKind = Literal["identity", "locality", "usability"]

QWEN_COMPLETION_BACKEND = "qwen_image_edit_2511"
QWEN_COMPLETION_MODE = "localized_raw"
ALLOWED_DATA_ROOT = Path("/mnt/workspace/litengjie/data").resolve()
DEFAULT_PUBLICATION_ROOT = (
    ALLOWED_DATA_ROOT / "reference_completion_publication_benchmarks"
).resolve()
DEFAULT_SAM3_CHECKPOINT = Path(
    "/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt"
)

_JUDGE_SYSTEM_PROMPTS: dict[JudgeKind, str] = {
    "identity": """You are the identity gate for a localized entity-reference
completion. Compare the immutable source entity with the segmented candidate.
Require the same subject, object, or group, no redesign or identity drift, and
no extra instance. For a group, preserve the original members and their
relationships. Return exactly one strict JSON object matching the supplied
schema. Set certain=false whenever identity cannot be established. verdict is
accept if and only if every boolean is true. Return JSON only.""",
    "locality": """You are the locality gate for a localized entity-reference
completion. Accept only a repair of a genuinely missing local part. Reject a
major redraw of visible content, composition expansion, a new scene, unrelated
content, or a change that does not improve the missing region. Return exactly
one strict JSON object matching the supplied schema. Set certain=false when
locality is uncertain. verdict is accept if and only if every boolean is true.
Return JSON only.""",
    "usability": """You are the usability gate for a segmented localized
entity-reference completion. Decide whether the candidate RGBA is a clean,
source-faithful reference. Require clean edges and plausible geometry,
anatomy, material, and group structure as applicable. Reject ghosting,
duplicated structures, fractures, extra limbs, isolated fragments, text,
logos, or watermarks. Return exactly one strict JSON object matching the
supplied schema. Set certain=false when usability is uncertain. verdict is
accept if and only if every boolean is true. Return JSON only.""",
}

_REFERENCE_TYPE_GUIDANCE: dict[ReferenceType, str] = {
    "subject": (
        "The same person, animal, or main subject must remain. Reject a second "
        "subject, extra face, body, or limb, and discontinuous pose, clothing, "
        "or silhouette."
    ),
    "object": (
        "The same object must remain. Require continuous geometry, material, "
        "texture, color, and viewpoint. Reject a second or redesigned object."
    ),
    "group": (
        "The same group and member relationships must remain. Multiple natural "
        "mask components are allowed; reject unrelated members or a new scene."
    ),
}


class PublicationManifestRecord(BaseModel):
    """One immutable localized completion selected for publication gating."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    clip_uid: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    reference_type: ReferenceType
    entity_phrase: str = Field(min_length=1)
    source_rgba_path: Path
    localized_result_path: Path

    @field_validator("clip_uid", "entity_id", "entity_phrase")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be non-empty")
        return stripped


class IdentityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    same_entity: StrictBool
    identity_preserved: StrictBool
    no_redesign: StrictBool
    no_extra_instance: StrictBool
    reference_type_consistent: StrictBool
    certain: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty_reason(value)

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> IdentityReview:
        flags = (
            self.same_entity,
            self.identity_preserved,
            self.no_redesign,
            self.no_extra_instance,
            self.reference_type_consistent,
            self.certain,
        )
        _require_matching_verdict(self.verdict, flags)
        return self


class LocalityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    local_missing_part_only: StrictBool
    visible_content_preserved: StrictBool
    no_composition_expansion: StrictBool
    no_new_scene_or_unrelated_content: StrictBool
    missing_region_improved: StrictBool
    certain: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty_reason(value)

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> LocalityReview:
        flags = (
            self.local_missing_part_only,
            self.visible_content_preserved,
            self.no_composition_expansion,
            self.no_new_scene_or_unrelated_content,
            self.missing_region_improved,
            self.certain,
        )
        _require_matching_verdict(self.verdict, flags)
        return self


class UsabilityReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    reference_usable: StrictBool
    boundary_clean: StrictBool
    geometry_and_structure_plausible: StrictBool
    no_ghosting_or_duplicate_structure: StrictBool
    no_fracture_extra_limb_or_fragment: StrictBool
    no_text_logo_or_watermark: StrictBool
    certain: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty_reason(value)

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> UsabilityReview:
        flags = (
            self.reference_usable,
            self.boundary_clean,
            self.geometry_and_structure_plausible,
            self.no_ghosting_or_duplicate_structure,
            self.no_fracture_extra_limb_or_fragment,
            self.no_text_logo_or_watermark,
            self.certain,
        )
        _require_matching_verdict(self.verdict, flags)
        return self


ReviewModel = IdentityReview | LocalityReview | UsabilityReview
ReviewType = TypeVar(
    "ReviewType",
    IdentityReview,
    LocalityReview,
    UsabilityReview,
)


def _nonempty_reason(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("reason must be non-empty")
    return stripped


def _require_matching_verdict(
    verdict: Literal["accept", "reject"],
    flags: Sequence[bool],
) -> None:
    if (verdict == "accept") != all(flags):
        raise ValueError("verdict must accept if and only if every flag is true")


@dataclass(frozen=True)
class SegmentationMaskCandidate:
    """A single SAM3 mask candidate before deterministic ranking."""

    mask: Image.Image | np.ndarray
    confidence: float
    object_id: str = ""



class CompletionPublicationJudge(Protocol):
    def review(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        candidate_mask: Image.Image,
        candidate_reference: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BaseModel: ...


@dataclass(frozen=True)
class PublicationConfig:
    """Thresholds recorded in every result for reproducibility."""

    allowed_input_root: Path = ALLOWED_DATA_ROOT
    publication_root: Path = DEFAULT_PUBLICATION_ROOT
    largest_component_ratio: float = 0.90
    max_secondary_component_ratio: float = 0.05
    group_min_largest_component_ratio: float = 0.15
    group_max_secondary_component_ratio: float = 1.0
    group_significant_component_ratio: float = 0.08
    group_max_component_gap_ratio: float = 0.30
    min_area_ratio_vs_source: float = 0.75
    max_area_ratio_vs_source: float = 2.00
    group_min_area_ratio_vs_source: float = 0.60
    group_max_area_ratio_vs_source: float = 2.50
    min_coverage_increase_ratio: float = 0.02
    max_local_bbox_extension_ratio: float = 0.25
    min_border_mean: float = 235.0
    max_border_std: float = 30.0
    near_white_channel_min: int = 225
    min_outside_near_white_ratio: float = 0.80
    max_outside_color_bins: int = 64
    color_quantization_step: int = 16

    def validate(self) -> None:
        if not isinstance(self.allowed_input_root, Path):
            raise TypeError("allowed_input_root must be a pathlib.Path")
        if not isinstance(self.publication_root, Path):
            raise TypeError("publication_root must be a pathlib.Path")
        ratios = {
            "largest_component_ratio": self.largest_component_ratio,
            "max_secondary_component_ratio": self.max_secondary_component_ratio,
            "group_min_largest_component_ratio": (
                self.group_min_largest_component_ratio
            ),
            "group_max_secondary_component_ratio": (
                self.group_max_secondary_component_ratio
            ),
            "group_significant_component_ratio": (
                self.group_significant_component_ratio
            ),
            "group_max_component_gap_ratio": self.group_max_component_gap_ratio,
            "min_coverage_increase_ratio": self.min_coverage_increase_ratio,
            "max_local_bbox_extension_ratio": (
                self.max_local_bbox_extension_ratio
            ),
            "min_outside_near_white_ratio": self.min_outside_near_white_ratio,
        }
        for name, value in ratios.items():
            _require_ratio(name, value)
        area_pairs = (
            (
                "area_ratio_vs_source",
                self.min_area_ratio_vs_source,
                self.max_area_ratio_vs_source,
            ),
            (
                "group_area_ratio_vs_source",
                self.group_min_area_ratio_vs_source,
                self.group_max_area_ratio_vs_source,
            ),
        )
        for name, minimum, maximum in area_pairs:
            if not _finite_number(minimum) or minimum <= 0:
                raise ValueError(f"minimum {name} must be finite and positive")
            if not _finite_number(maximum) or maximum < minimum:
                raise ValueError(f"maximum {name} must be at least its minimum")
        for name, value in (
            ("min_border_mean", self.min_border_mean),
            ("max_border_std", self.max_border_std),
        ):
            if not _finite_number(value) or not 0 <= value <= 255:
                raise ValueError(f"{name} must be between 0 and 255")
        for name, value, lower, upper in (
            ("near_white_channel_min", self.near_white_channel_min, 0, 255),
            ("max_outside_color_bins", self.max_outside_color_bins, 1, 4096),
            ("color_quantization_step", self.color_quantization_step, 1, 255),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")

    def thresholds_for(self, reference_type: ReferenceType) -> dict[str, object]:
        group = reference_type == "group"
        return {
            "largest_component_ratio": (
                self.group_min_largest_component_ratio
                if group
                else self.largest_component_ratio
            ),
            "max_secondary_component_ratio": (
                self.group_max_secondary_component_ratio
                if group
                else self.max_secondary_component_ratio
            ),
            "group_significant_component_ratio": (
                self.group_significant_component_ratio if group else None
            ),
            "group_max_component_gap_ratio": (
                self.group_max_component_gap_ratio if group else None
            ),
            "min_area_ratio_vs_source": (
                self.group_min_area_ratio_vs_source
                if group
                else self.min_area_ratio_vs_source
            ),
            "max_area_ratio_vs_source": (
                self.group_max_area_ratio_vs_source
                if group
                else self.max_area_ratio_vs_source
            ),
            "min_coverage_increase_ratio": self.min_coverage_increase_ratio,
            "max_local_bbox_extension_ratio": (
                self.max_local_bbox_extension_ratio
            ),
            "min_border_mean": self.min_border_mean,
            "max_border_std": self.max_border_std,
            "near_white_channel_min": self.near_white_channel_min,
            "min_outside_near_white_ratio": (
                self.min_outside_near_white_ratio
            ),
            "max_outside_color_bins": self.max_outside_color_bins,
            "color_quantization_step": self.color_quantization_step,
        }


@dataclass(frozen=True)
class PublicationStats:
    processed: int
    auto_published: int
    rejected: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _Preflight:
    record: PublicationManifestRecord
    source_path: Path
    source_sha256: str
    localized_result_path: Path
    candidate_path: Path
    candidate_sha256: str


@dataclass(frozen=True)
class _Component:
    area: int
    bbox: tuple[int, int, int, int]


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_ratio(name: str, value: object) -> None:
    if not _finite_number(value) or not 0 <= cast(float, value) <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_input_path(
    path: Path,
    *,
    field_name: str,
    allowed_root: Path,
) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    resolved = path.expanduser().resolve(strict=True)
    root = allowed_root.expanduser().resolve(strict=True)
    if not _is_at_or_below(resolved, root):
        raise ValueError(f"{field_name} must remain under {root}")
    return resolved


def _resolve_publication_run(path: Path, config: PublicationConfig) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("benchmark_root must be an absolute path")
    allowed = config.publication_root.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    if resolved == allowed or not _is_at_or_below(resolved, allowed):
        raise ValueError(
            "benchmark_root must be a run directory strictly below "
            f"{allowed}"
        )
    forbidden = {"selected", "r2v_v3_runs", "r2v_v3_datasets"}
    relative_parts = {part.casefold() for part in resolved.relative_to(allowed).parts}
    if relative_parts & forbidden:
        raise ValueError("benchmark_root targets a production directory")
    if resolved.exists():
        raise FileExistsError(f"benchmark_root already exists: {resolved}")
    return resolved


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_png(path: Path, image: Image.Image) -> str:
    image.save(path, format="PNG")
    with Image.open(path) as reopened:
        reopened.load()
        if reopened.format != "PNG" or reopened.mode != image.mode:
            raise RuntimeError("published PNG format or mode changed")
        if reopened.size != image.size:
            raise RuntimeError("published PNG dimensions changed")
    return _sha256_path(path)


def _load_source_rgba(path: Path) -> tuple[Image.Image, bytes, str]:
    if path.suffix.casefold() != ".png":
        raise ValueError("source_rgba_path must name a PNG")
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "RGBA":
            raise ValueError("source reference must be an RGBA PNG")
        image = opened.copy()
    pixels = np.asarray(image, dtype=np.uint8)
    alpha = pixels[:, :, 3]
    if not np.isin(alpha, (0, 255)).all() or not np.any(alpha == 255):
        raise ValueError("source reference must have non-empty binary alpha")
    if np.any(pixels[:, :, :3][alpha == 0] != 255):
        raise ValueError("source transparent RGB must be exact white")
    return image, raw, _sha256_bytes(raw)


def _load_candidate_rgb(path: Path) -> tuple[Image.Image, bytes, str]:
    if path.suffix.casefold() != ".png":
        raise ValueError("localized candidate must name a PNG")
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "RGB":
            raise ValueError("localized candidate must be an RGB PNG")
        if opened.width <= 0 or opened.height <= 0:
            raise ValueError("localized candidate dimensions must be positive")
        image = opened.copy()
    return image, raw, _sha256_bytes(raw)


def load_publication_manifest(
    path: Path,
    *,
    config: PublicationConfig,
) -> list[PublicationManifestRecord]:
    resolved = _resolve_input_path(
        path,
        field_name="manifest_path",
        allowed_root=config.allowed_input_root,
    )
    records: list[PublicationManifestRecord] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                record = PublicationManifestRecord.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"invalid publication manifest line {line_number}: {exc}"
                ) from exc
            if record.sample_id in seen_ids:
                raise ValueError(
                    f"duplicate publication sample_id: {record.sample_id}"
                )
            seen_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError("publication manifest contains no records")
    return sorted(records, key=lambda record: record.sample_id)


def _preflight_record(
    record: PublicationManifestRecord,
    config: PublicationConfig,
) -> _Preflight:
    source_path = _resolve_input_path(
        record.source_rgba_path,
        field_name="source_rgba_path",
        allowed_root=config.allowed_input_root,
    )
    _, _, source_sha256 = _load_source_rgba(source_path)
    localized_result_path = _resolve_input_path(
        record.localized_result_path,
        field_name="localized_result_path",
        allowed_root=config.allowed_input_root,
    )
    try:
        localized = json.loads(localized_result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("localized_result_path must contain valid JSON") from exc
    if not isinstance(localized, dict):
        raise TypeError("localized result must be one JSON object")
    if localized.get("backend") != QWEN_COMPLETION_BACKEND:
        raise ValueError("localized result backend must be qwen_image_edit_2511")
    if localized.get("mode") != QWEN_COMPLETION_MODE:
        raise ValueError("localized result mode must be localized_raw")
    hard_check = localized.get("hard_check")
    if not isinstance(hard_check, dict):
        raise TypeError("localized result hard_check must be an object")
    checks = hard_check.get("checks")
    reasons = hard_check.get("reasons")
    if (
        hard_check.get("status") != "passed"
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or not isinstance(reasons, list)
        or reasons
    ):
        raise ValueError("localized result contains a hard-check failure")
    localized_source = localized.get("source_path")
    if not isinstance(localized_source, str):
        raise TypeError("localized result source_path must be a string")
    result_source_path = _resolve_input_path(
        Path(localized_source),
        field_name="localized result source_path",
        allowed_root=config.allowed_input_root,
    )
    if result_source_path != source_path:
        raise ValueError("manifest and localized result source paths differ")
    if localized.get("source_sha256") != source_sha256:
        raise ValueError("localized result source SHA-256 does not match")
    candidate_value = localized.get("candidate_path")
    if not isinstance(candidate_value, str) or not candidate_value.strip():
        raise ValueError("localized result candidate_path must be non-empty")
    candidate_unresolved = Path(candidate_value)
    if not candidate_unresolved.is_absolute():
        candidate_unresolved = localized_result_path.parent / candidate_unresolved
    candidate_path = _resolve_input_path(
        candidate_unresolved,
        field_name="localized result candidate_path",
        allowed_root=config.allowed_input_root,
    )
    _, _, candidate_sha256 = _load_candidate_rgb(candidate_path)
    if localized.get("candidate_sha256") != candidate_sha256:
        raise ValueError("localized candidate SHA-256 does not match")
    return _Preflight(
        record=record,
        source_path=source_path,
        source_sha256=source_sha256,
        localized_result_path=localized_result_path,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
    )


def _normalize_mask(
    value: Image.Image | np.ndarray,
    *,
    expected_size: tuple[int, int],
) -> Image.Image:
    if isinstance(value, Image.Image):
        if value.mode != "L":
            raise ValueError("segmentation mask must use L mode")
        array = np.asarray(value)
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError("segmentation mask must be a PIL image or numpy array")
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("segmentation mask must have H-W shape")
    if (array.shape[1], array.shape[0]) != expected_size:
        raise ValueError("segmentation mask dimensions must match candidate RGB")
    if not np.issubdtype(array.dtype, np.bool_):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("segmentation mask pixels must be finite integers")
        unique = {int(item) for item in np.unique(array)}
        if not unique.issubset({0, 1, 255}):
            raise ValueError("segmentation mask must be binary")
        foreground = array != 0
    else:
        foreground = array
    return Image.fromarray(np.where(foreground, 255, 0).astype(np.uint8), mode="L")


def _components(mask: np.ndarray) -> list[_Component]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[_Component] = []
    for start_y, start_x in np.argwhere(mask):
        y0 = int(start_y)
        x0 = int(start_x)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        area = 0
        left = right = x0
        top = bottom = y0
        while stack:
            y, x = stack.pop()
            area += 1
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        components.append(
            _Component(area=area, bbox=(left, top, right + 1, bottom + 1))
        )
    return sorted(
        components,
        key=lambda component: (-component.area, component.bbox),
    )


def compute_foreground_metrics(mask: Image.Image) -> dict[str, object]:
    """Compute size-normalized geometry for source alpha or candidate mask."""

    if not isinstance(mask, Image.Image) or mask.mode != "L":
        raise TypeError("foreground metrics require an L-mode PIL image")
    array = np.asarray(mask, dtype=np.uint8)
    if not np.isin(array, (0, 255)).all():
        raise ValueError("foreground metrics require a binary mask")
    foreground = array == 255
    height, width = foreground.shape
    area = int(foreground.sum())
    components = _components(foreground)
    if area:
        ys, xs = np.nonzero(foreground)
        bbox: list[int] | None = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
        normalized_bbox: list[float] | None = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ]
        normalized_bbox_size: list[float] | None = [
            (bbox[2] - bbox[0]) / width,
            (bbox[3] - bbox[1]) / height,
        ]
    else:
        bbox = None
        normalized_bbox = None
        normalized_bbox_size = None
    edge_touches = {
        "top": bool(foreground[0, :].any()),
        "bottom": bool(foreground[-1, :].any()),
        "left": bool(foreground[:, 0].any()),
        "right": bool(foreground[:, -1].any()),
    }
    return {
        "canvas_size": [width, height],
        "foreground_pixels": area,
        "foreground_area_ratio": area / (width * height),
        "bbox": bbox,
        "normalized_bbox": normalized_bbox,
        "normalized_bbox_size": normalized_bbox_size,
        "edge_touches": edge_touches,
        "edge_touch_count": sum(edge_touches.values()),
        "component_count": len(components),
        "component_areas": [component.area for component in components],
        "component_bboxes": [list(component.bbox) for component in components],
    }


def select_segmented_mask(
    candidates: Sequence[SegmentationMaskCandidate],
    *,
    expected_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, object]]:
    """Select exactly one mask with stable confidence/area/index ordering."""

    ranked: list[tuple[float, int, int, str, Image.Image]] = []
    diagnostics: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, SegmentationMaskCandidate):
            raise TypeError("segmenter must return SegmentationMaskCandidate values")
        if not _finite_number(candidate.confidence):
            raise ValueError("segmentation confidence must be finite")
        mask = _normalize_mask(candidate.mask, expected_size=expected_size)
        area = int(np.count_nonzero(np.asarray(mask, dtype=np.uint8)))
        confidence = float(candidate.confidence)
        ranked.append((confidence, area, index, candidate.object_id, mask))
        diagnostics.append(
            {
                "original_index": index,
                "object_id": candidate.object_id,
                "confidence": confidence,
                "foreground_pixels": area,
            }
        )
    if not ranked:
        raise ValueError("SAM3 returned no segmentation candidates")
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]))
    selected = ranked[0]
    ordered = sorted(
        diagnostics,
        key=lambda value: (
            -cast(float, value["confidence"]),
            -cast(int, value["foreground_pixels"]),
            cast(int, value["original_index"]),
        ),
    )
    provenance = {
        "policy": "confidence_desc_area_desc_original_index_asc",
        "candidate_count": len(ranked),
        "selected_original_index": selected[2],
        "selected_object_id": selected[3],
        "ranked_candidates": ordered,
    }
    return selected[4].copy(), provenance


def _component_gap_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> float:
    horizontal = max(first[0] - second[2], second[0] - first[2], 0) / width
    vertical = max(first[1] - second[3], second[1] - first[3], 0) / height
    return math.hypot(horizontal, vertical)


def evaluate_mask_gate(
    mask: Image.Image,
    *,
    reference_type: ReferenceType,
    config: PublicationConfig,
) -> tuple[dict[str, object], bool, list[str]]:
    metrics = compute_foreground_metrics(mask)
    foreground = cast(int, metrics["foreground_pixels"])
    canvas_size = cast(list[int], metrics["canvas_size"])
    canvas_area = canvas_size[0] * canvas_size[1]
    component_areas = cast(list[int], metrics["component_areas"])
    largest_ratio = component_areas[0] / foreground if foreground else 0.0
    secondary_ratio = component_areas[1] / foreground if len(component_areas) > 1 else 0.0
    reasons: list[str] = []
    if foreground == 0:
        reasons.append("mask_empty")
    if foreground == canvas_area:
        reasons.append("mask_full")
    if foreground and not component_areas:
        reasons.append("mask_has_no_major_component")
    isolated_components: list[dict[str, object]] = []
    if foreground and component_areas:
        if reference_type == "group":
            if largest_ratio < config.group_min_largest_component_ratio:
                reasons.append("group_largest_component_ratio_below_min")
            if secondary_ratio > config.group_max_secondary_component_ratio:
                reasons.append("group_secondary_component_ratio_above_max")
            boxes = cast(list[list[int]], metrics["component_bboxes"])
            for index, (area, bbox) in enumerate(
                zip(component_areas[1:], boxes[1:]),
                start=1,
            ):
                area_ratio = area / foreground
                gap_ratio = _component_gap_ratio(
                    tuple(boxes[0]),
                    tuple(bbox),
                    width=canvas_size[0],
                    height=canvas_size[1],
                )
                if (
                    area_ratio >= config.group_significant_component_ratio
                    and gap_ratio > config.group_max_component_gap_ratio
                ):
                    isolated_components.append(
                        {
                            "component_index": index,
                            "area_ratio": area_ratio,
                            "gap_ratio": gap_ratio,
                        }
                    )
            if isolated_components:
                reasons.append("group_significant_component_isolated")
        else:
            if largest_ratio < config.largest_component_ratio:
                reasons.append("largest_component_ratio_below_min")
            if secondary_ratio > config.max_secondary_component_ratio:
                reasons.append("secondary_component_ratio_above_max")
    metrics.update(
        {
            "largest_component_ratio": largest_ratio,
            "secondary_component_ratio": secondary_ratio,
            "isolated_group_components": isolated_components,
            "thresholds": config.thresholds_for(reference_type),
        }
    )
    return metrics, not reasons, reasons


def _normalized_bbox_overlap_ratio(
    source: Sequence[float],
    candidate: Sequence[float],
) -> float:
    intersection_width = max(0.0, min(source[2], candidate[2]) - max(source[0], candidate[0]))
    intersection_height = max(0.0, min(source[3], candidate[3]) - max(source[1], candidate[1]))
    source_area = max(0.0, source[2] - source[0]) * max(0.0, source[3] - source[1])
    return (
        intersection_width * intersection_height / source_area
        if source_area
        else 0.0
    )


def evaluate_improvement_gate(
    source_metrics: dict[str, object],
    candidate_metrics: dict[str, object],
    *,
    reference_type: ReferenceType,
    config: PublicationConfig,
) -> tuple[dict[str, object], bool, list[str]]:
    source_area = cast(float, source_metrics["foreground_area_ratio"])
    candidate_area = cast(float, candidate_metrics["foreground_area_ratio"])
    area_ratio = candidate_area / source_area if source_area else math.inf
    if reference_type == "group":
        minimum = config.group_min_area_ratio_vs_source
        maximum = config.group_max_area_ratio_vs_source
    else:
        minimum = config.min_area_ratio_vs_source
        maximum = config.max_area_ratio_vs_source
    source_edges = cast(dict[str, bool], source_metrics["edge_touches"])
    candidate_edges = cast(dict[str, bool], candidate_metrics["edge_touches"])
    reduced_edges = sorted(
        edge
        for edge, touched in source_edges.items()
        if touched and not candidate_edges[edge]
    )
    source_bbox = cast(list[float] | None, source_metrics["normalized_bbox"])
    candidate_bbox = cast(
        list[float] | None,
        candidate_metrics["normalized_bbox"],
    )
    overlap_ratio = 0.0
    bbox_extension = 0.0
    nearby_local_fill = False
    if source_bbox is not None and candidate_bbox is not None:
        overlap_ratio = _normalized_bbox_overlap_ratio(source_bbox, candidate_bbox)
        extensions = (
            max(0.0, source_bbox[0] - candidate_bbox[0]),
            max(0.0, source_bbox[1] - candidate_bbox[1]),
            max(0.0, candidate_bbox[2] - source_bbox[2]),
            max(0.0, candidate_bbox[3] - source_bbox[3]),
        )
        bbox_extension = max(extensions)
        nearby_local_fill = (
            overlap_ratio >= 0.5
            and 0 < bbox_extension <= config.max_local_bbox_extension_ratio
        )
    coverage_increased = candidate_area >= source_area * (
        1.0 + config.min_coverage_increase_ratio
    )
    area_ok = minimum <= area_ratio <= maximum
    improvement_flags = {
        "edge_touches_reduced": bool(reduced_edges),
        "nearby_local_missing_region_filled": nearby_local_fill,
        "coverage_increased_without_abnormal_growth": coverage_increased and area_ok,
    }
    reasons: list[str] = []
    if area_ratio < minimum:
        reasons.append("candidate_area_ratio_below_min")
    elif area_ratio > maximum:
        reasons.append("candidate_area_ratio_above_max")
    if not any(improvement_flags.values()):
        reasons.append("no_local_completion_improvement")
    report = {
        "area_ratio_vs_source": area_ratio,
        "min_area_ratio_vs_source": minimum,
        "max_area_ratio_vs_source": maximum,
        "reduced_edge_touches": reduced_edges,
        "normalized_bbox_overlap_vs_source": overlap_ratio,
        "max_normalized_bbox_extension": bbox_extension,
        "improvements": improvement_flags,
    }
    return report, not reasons, reasons


def evaluate_background_gate(
    candidate_rgb: Image.Image,
    candidate_mask: Image.Image,
    *,
    config: PublicationConfig,
) -> tuple[dict[str, object], bool, list[str]]:
    if candidate_rgb.mode != "RGB" or candidate_mask.mode != "L":
        raise ValueError("background gate requires RGB candidate and L mask")
    if candidate_rgb.size != candidate_mask.size:
        raise ValueError("background gate image sizes must match")
    pixels = np.asarray(candidate_rgb, dtype=np.uint8)
    foreground = np.asarray(candidate_mask, dtype=np.uint8) == 255
    outside = pixels[~foreground]
    border_selector = np.zeros(foreground.shape, dtype=bool)
    border_selector[0, :] = True
    border_selector[-1, :] = True
    border_selector[:, 0] = True
    border_selector[:, -1] = True
    border = pixels[border_selector & ~foreground]
    if outside.size:
        near_white = np.all(outside >= config.near_white_channel_min, axis=1)
        near_white_ratio = float(near_white.mean())
        quantized = outside // config.color_quantization_step
        color_bins = int(np.unique(quantized, axis=0).shape[0])
        outside_mean = float(outside.mean())
        outside_std = float(outside.std())
    else:
        near_white_ratio = 0.0
        color_bins = 0
        outside_mean = 0.0
        outside_std = 0.0
    border_mean = float(border.mean()) if border.size else 0.0
    border_std = float(border.std()) if border.size else math.inf
    checks = {
        "outside_pixels_present": bool(outside.size),
        "outside_near_white_ratio": (
            near_white_ratio >= config.min_outside_near_white_ratio
        ),
        "border_mean": border_mean >= config.min_border_mean,
        "border_std": border_std <= config.max_border_std,
        "outside_color_diversity": color_bins <= config.max_outside_color_bins,
    }
    reasons = [
        f"background_{name}_failed" for name, passed in checks.items() if not passed
    ]
    metrics = {
        "outside_pixel_count": int(outside.shape[0]),
        "outside_mean": outside_mean,
        "outside_std": outside_std,
        "outside_near_white_ratio": near_white_ratio,
        "outside_color_bins": color_bins,
        "border_outside_pixel_count": int(border.shape[0]),
        "border_mean": border_mean,
        "border_std": border_std,
        "checks": checks,
    }
    return metrics, not reasons, reasons


def build_candidate_reference(
    candidate_rgb: Image.Image,
    candidate_mask: Image.Image,
) -> Image.Image:
    """Build source-faithful-size RGBA without resize or interpolation."""

    if candidate_rgb.mode != "RGB" or candidate_mask.mode != "L":
        raise ValueError("candidate reference requires RGB and L inputs")
    if candidate_rgb.size != candidate_mask.size:
        raise ValueError("candidate reference inputs must have matching sizes")
    alpha = np.asarray(candidate_mask, dtype=np.uint8)
    if not np.isin(alpha, (0, 255)).all():
        raise ValueError("candidate reference alpha must be binary")
    rgb = np.asarray(candidate_rgb, dtype=np.uint8).copy()
    rgb[alpha == 0] = 255
    return Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _source_on_white(source_rgba: Image.Image) -> Image.Image:
    pixels = np.asarray(source_rgba, dtype=np.uint8)
    result = np.full(pixels.shape[:2] + (3,), 255, dtype=np.uint8)
    visible = pixels[:, :, 3] == 255
    result[visible] = pixels[:, :, :3][visible]
    return Image.fromarray(result, mode="RGB")


_REVIEW_MODELS: dict[JudgeKind, type[BaseModel]] = {
    "identity": IdentityReview,
    "locality": LocalityReview,
    "usability": UsabilityReview,
}


class PublicationJudgeFailure(StructuredOutputFailure):
    pass


class QwenCompletionPublicationJudge:
    """One strict, deterministic Qwen-VL publication judge."""

    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        kind: JudgeKind,
        repair_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        if kind not in _REVIEW_MODELS:
            raise ValueError("unsupported publication judge kind")
        if (
            not isinstance(repair_retries, int)
            or isinstance(repair_retries, bool)
            or repair_retries < 0
        ):
            raise ValueError("repair_retries must be a non-negative integer")
        self.config = config
        self.kind = kind
        self.repair_retries = repair_retries
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _messages(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        candidate_mask: Image.Image,
        candidate_reference: Image.Image,
        request_text: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [{"type": "text", "text": request_text}]
        for label, image in (
            ("Image 1: immutable source entity on white", _source_on_white(source_rgba)),
            ("Image 2: localized raw RGB candidate", candidate_rgb),
            ("Image 3: deterministic SAM3 candidate mask", candidate_mask),
            ("Image 4: provisional segmented RGBA on white", candidate_reference),
        ):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(image)},
                }
            )
        return [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPTS[self.kind]},
            {"role": "user", "content": content},
        ]

    def _request(
        self,
        messages: list[dict[str, object]],
        model: type[BaseModel],
    ) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        schema_name = f"v3_completion_publication_{self.kind}_review"
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": model.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Qwen returned an empty {self.kind} review")
        return str(content)

    def review(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        candidate_mask: Image.Image,
        candidate_reference: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BaseModel:
        if source_rgba.mode != "RGBA":
            raise ValueError("publication judge source must be RGBA")
        if candidate_rgb.mode != "RGB" or candidate_mask.mode != "L":
            raise ValueError("publication judge candidate must be RGB with L mask")
        if candidate_reference.mode != "RGBA":
            raise ValueError("publication judge reference must be RGBA")
        if not (
            candidate_rgb.size
            == candidate_mask.size
            == candidate_reference.size
        ):
            raise ValueError("publication judge candidate dimensions must match")
        model = _REVIEW_MODELS[self.kind]
        original_request = (
            f"Run only the {self.kind} publication gate. "
            f"reference_type={reference_type}; entity_phrase={entity_phrase}. "
            f"{_REFERENCE_TYPE_GUIDANCE[reference_type]} Return the required "
            "strict JSON object."
        )
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(self.repair_retries + 1):
            request_text = original_request
            if attempt:
                request_text = build_structured_repair_prompt(
                    original_request=original_request,
                    invalid_response=raw_responses[-1],
                    validation_issues=issues,
                    json_schema=model.model_json_schema(),
                )
            try:
                raw = self._request(
                    self._messages(
                        source_rgba=source_rgba,
                        candidate_rgb=candidate_rgb,
                        candidate_mask=candidate_mask,
                        candidate_reference=candidate_reference,
                        request_text=request_text,
                    ),
                    model,
                )
            except Exception as exc:
                raise PublicationJudgeFailure(
                    raw_responses=raw_responses,
                    issues=[
                        ValidationIssue(
                            code="qwen_request_failed",
                            field=None,
                            message=str(exc),
                        )
                    ],
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            review, issues = parse_qwen_json_issues(raw, model)
            if review is not None and not issues:
                return review
        raise PublicationJudgeFailure(raw_responses=raw_responses, issues=issues)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _mask_from_track_result(
    result: EntityTrackResult,
    *,
    expected_size: tuple[int, int],
) -> tuple[Image.Image, dict[str, object]]:
    if not isinstance(result, EntityTrackResult):
        raise TypeError("SAM3 backend must return EntityTrackResult")
    if result.status != "ready":
        detail = f": {result.reason}" if result.reason else ""
        raise ValueError(f"SAM3 track status is {result.status}{detail}")
    slot_zero = [
        observation
        for observation in result.observations
        if observation.slot == 0
    ]
    if len(slot_zero) != 1:
        raise ValueError(
            "SAM3 ready track must contain exactly one slot 0 observation"
        )
    observation = slot_zero[0]
    if not observation.valid:
        raise ValueError("SAM3 slot 0 observation is invalid")
    return select_segmented_mask(
        (
            SegmentationMaskCandidate(
                mask=observation.mask,
                confidence=observation.confidence,
                object_id=observation.object_id,
            ),
        ),
        expected_size=expected_size,
    )


def _synthetic_review(kind: JudgeKind, reason: str) -> ReviewModel:
    if kind == "identity":
        return IdentityReview(
            verdict="reject",
            same_entity=False,
            identity_preserved=False,
            no_redesign=False,
            no_extra_instance=False,
            reference_type_consistent=False,
            certain=False,
            reason=reason,
        )
    if kind == "locality":
        return LocalityReview(
            verdict="reject",
            local_missing_part_only=False,
            visible_content_preserved=False,
            no_composition_expansion=False,
            no_new_scene_or_unrelated_content=False,
            missing_region_improved=False,
            certain=False,
            reason=reason,
        )
    return UsabilityReview(
        verdict="reject",
        reference_usable=False,
        boundary_clean=False,
        geometry_and_structure_plausible=False,
        no_ghosting_or_duplicate_structure=False,
        no_fracture_extra_limb_or_fragment=False,
        no_text_logo_or_watermark=False,
        certain=False,
        reason=reason,
    )


def _run_one_judge(
    kind: JudgeKind,
    judge: CompletionPublicationJudge,
    *,
    source_rgba: Image.Image,
    candidate_rgb: Image.Image,
    candidate_mask: Image.Image,
    candidate_reference: Image.Image,
    record: PublicationManifestRecord,
) -> tuple[ReviewModel, Literal["reviewed", "failed_closed"]]:
    expected = _REVIEW_MODELS[kind]
    try:
        review = judge.review(
            source_rgba=source_rgba.copy(),
            candidate_rgb=candidate_rgb.copy(),
            candidate_mask=candidate_mask.copy(),
            candidate_reference=candidate_reference.copy(),
            entity_phrase=record.entity_phrase,
            reference_type=record.reference_type,
        )
        if not isinstance(review, expected):
            raise TypeError(
                f"{kind} judge must return {expected.__name__}"
            )
        return cast(ReviewModel, review), "reviewed"
    except Exception as exc:  # noqa: BLE001 - every judge error fails closed
        return _synthetic_review(kind, f"judge_failed: {exc}"), "failed_closed"


def _verify_candidate_reference(
    path: Path,
    *,
    expected_size: tuple[int, int],
) -> str:
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "RGBA":
            raise RuntimeError("published candidate reference must be RGBA PNG")
        if opened.size != expected_size:
            raise RuntimeError("published candidate reference dimensions changed")
        pixels = np.asarray(opened, dtype=np.uint8)
    alpha = pixels[:, :, 3]
    if not np.isin(alpha, (0, 255)).all():
        raise RuntimeError("published candidate reference alpha is not binary")
    if np.any(pixels[:, :, :3][alpha == 0] != 255):
        raise RuntimeError("published transparent RGB is not exact white")
    return _sha256_path(path)


def _first_rejection_reason(
    *,
    hard_checks_passed: bool,
    mask_reasons: Sequence[str],
    improvement_reasons: Sequence[str],
    background_reasons: Sequence[str],
    reviews: dict[JudgeKind, ReviewModel],
    judge_statuses: dict[JudgeKind, str],
) -> str:
    if not hard_checks_passed:
        return "input_hard_checks_failed"
    for reasons in (mask_reasons, improvement_reasons, background_reasons):
        if reasons:
            return reasons[0]
    for kind in ("identity", "locality", "usability"):
        if judge_statuses[kind] == "failed_closed":
            return f"{kind}_judge_failed"
        if reviews[kind].verdict != "accept":
            return f"{kind}_judge_rejected"
    return "all_publication_gates_passed"


def _process_preflight(
    preflight: _Preflight,
    *,
    benchmark_root: Path,
    config: PublicationConfig,
    segmenter: SegmentationBackend,
    identity_judge: CompletionPublicationJudge,
    locality_judge: CompletionPublicationJudge,
    usability_judge: CompletionPublicationJudge,
) -> dict[str, object]:
    record = preflight.record
    final_directory = benchmark_root / record.sample_id
    temporary = benchmark_root / f".{record.sample_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        source_rgba, source_bytes, source_sha256 = _load_source_rgba(
            preflight.source_path
        )
        candidate_rgb, candidate_bytes, candidate_sha256 = _load_candidate_rgb(
            preflight.candidate_path
        )
        input_hashes_match = (
            source_sha256 == preflight.source_sha256
            and candidate_sha256 == preflight.candidate_sha256
        )
        source_output = temporary / "source_rgba.png"
        candidate_output = temporary / "candidate_rgb.png"
        source_output.write_bytes(source_bytes)
        candidate_output.write_bytes(candidate_bytes)
        if _sha256_path(source_output) != preflight.source_sha256:
            raise RuntimeError("transactional source copy hash changed")
        if _sha256_path(candidate_output) != preflight.candidate_sha256:
            raise RuntimeError("transactional candidate copy hash changed")

        segmentation_error: str | None = None
        try:
            track_result = segmenter.track(
                frame_paths=[candidate_output],
                entity_id=record.entity_id,
                reference_type=record.reference_type,
                grounding_prompt=record.entity_phrase,
            )
            candidate_mask, ranking = _mask_from_track_result(
                track_result,
                expected_size=candidate_rgb.size,
            )
        except Exception as exc:  # noqa: BLE001 - segmentation fails closed
            segmentation_error = f"segmentation_failed: {exc}"
            candidate_mask = Image.new("L", candidate_rgb.size, 0)
            ranking = {
                "policy": "confidence_desc_area_desc_original_index_asc",
                "candidate_count": 0,
                "selected_original_index": None,
                "selected_object_id": None,
                "ranked_candidates": [],
                "error": segmentation_error,
            }
        candidate_mask_path = temporary / "candidate_mask.png"
        candidate_mask_sha256 = _save_png(candidate_mask_path, candidate_mask)

        source_metrics = compute_foreground_metrics(
            source_rgba.getchannel("A")
        )
        mask_metrics, mask_gate_passed, mask_reasons = evaluate_mask_gate(
            candidate_mask,
            reference_type=record.reference_type,
            config=config,
        )
        mask_metrics["ranking"] = ranking
        if segmentation_error is not None:
            mask_gate_passed = False
            mask_reasons.insert(0, "segmentation_failed")
        candidate_metrics = {
            key: value
            for key, value in mask_metrics.items()
            if key not in {"ranking", "thresholds", "isolated_group_components"}
        }
        improvement_metrics, improvement_gate_passed, improvement_reasons = (
            evaluate_improvement_gate(
                source_metrics,
                candidate_metrics,
                reference_type=record.reference_type,
                config=config,
            )
        )
        background_metrics, background_gate_passed, background_reasons = (
            evaluate_background_gate(
                candidate_rgb,
                candidate_mask,
                config=config,
            )
        )
        candidate_reference = build_candidate_reference(
            candidate_rgb,
            candidate_mask,
        )
        hard_checks_passed = input_hashes_match
        deterministic_gates_passed = all(
            (
                hard_checks_passed,
                mask_gate_passed,
                improvement_gate_passed,
                background_gate_passed,
            )
        )

        judges: dict[JudgeKind, CompletionPublicationJudge] = {
            "identity": identity_judge,
            "locality": locality_judge,
            "usability": usability_judge,
        }
        reviews: dict[JudgeKind, ReviewModel] = {}
        judge_statuses: dict[JudgeKind, str] = {}
        for kind in ("identity", "locality", "usability"):
            typed_kind = cast(JudgeKind, kind)
            if deterministic_gates_passed:
                review, status = _run_one_judge(
                    typed_kind,
                    judges[typed_kind],
                    source_rgba=source_rgba,
                    candidate_rgb=candidate_rgb,
                    candidate_mask=candidate_mask,
                    candidate_reference=candidate_reference,
                    record=record,
                )
            else:
                review = _synthetic_review(
                    typed_kind,
                    "not_run: deterministic publication gate failed",
                )
                status = "not_run"
            reviews[typed_kind] = review
            judge_statuses[typed_kind] = status
            write_json_atomic(
                temporary / f"review_{typed_kind}.json",
                review.model_dump(mode="json"),
            )

        inputs_unchanged = (
            _sha256_path(preflight.source_path) == preflight.source_sha256
            and _sha256_path(preflight.candidate_path) == preflight.candidate_sha256
        )
        hard_checks_passed = hard_checks_passed and inputs_unchanged
        publication_gate = {
            "hard_checks_passed": hard_checks_passed,
            "mask_gate_passed": mask_gate_passed,
            "improvement_gate_passed": improvement_gate_passed,
            "background_gate_passed": background_gate_passed,
            "identity_gate_passed": reviews["identity"].verdict == "accept",
            "locality_gate_passed": reviews["locality"].verdict == "accept",
            "usability_gate_passed": reviews["usability"].verdict == "accept",
        }
        status: PublicationStatus = (
            "auto_published" if all(publication_gate.values()) else "rejected"
        )
        candidate_reference_path: str | None = None
        candidate_reference_sha256: str | None = None
        if status == "auto_published":
            reference_output = temporary / "candidate_reference.png"
            _save_png(reference_output, candidate_reference)
            candidate_reference_sha256 = _verify_candidate_reference(
                reference_output,
                expected_size=candidate_rgb.size,
            )
            candidate_reference_path = "candidate_reference.png"

        reason = _first_rejection_reason(
            hard_checks_passed=hard_checks_passed,
            mask_reasons=mask_reasons,
            improvement_reasons=improvement_reasons,
            background_reasons=background_reasons,
            reviews=reviews,
            judge_statuses=judge_statuses,
        )
        result: dict[str, object] = {
            "status": status,
            "backend": QWEN_COMPLETION_BACKEND,
            "mode": QWEN_COMPLETION_MODE,
            "sample_id": record.sample_id,
            "clip_uid": record.clip_uid,
            "entity_id": record.entity_id,
            "reference_type": record.reference_type,
            "entity_phrase": record.entity_phrase,
            "source_path": str(preflight.source_path),
            "source_sha256": preflight.source_sha256,
            "localized_result_path": str(preflight.localized_result_path),
            "candidate_rgb_path": "candidate_rgb.png",
            "candidate_rgb_sha256": preflight.candidate_sha256,
            "candidate_mask_path": "candidate_mask.png",
            "candidate_mask_sha256": candidate_mask_sha256,
            "candidate_reference_path": candidate_reference_path,
            "candidate_reference_sha256": candidate_reference_sha256,
            "mask_metrics": mask_metrics,
            "source_metrics": source_metrics,
            "candidate_metrics": candidate_metrics,
            "improvement_metrics": improvement_metrics,
            "background_metrics": background_metrics,
            "identity_review": reviews["identity"].model_dump(mode="json"),
            "locality_review": reviews["locality"].model_dump(mode="json"),
            "usability_review": reviews["usability"].model_dump(mode="json"),
            "judge_statuses": judge_statuses,
            "publication_gate": publication_gate,
            "thresholds": config.thresholds_for(record.reference_type),
            "rejection_reasons": [
                *mask_reasons,
                *improvement_reasons,
                *background_reasons,
            ],
            "reason": reason,
        }
        write_json_atomic(temporary / "result.json", result)
        temporary.replace(final_directory)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _publish_processing_failure(
    preflight: _Preflight,
    *,
    benchmark_root: Path,
    error: Exception,
) -> dict[str, object]:
    """Publish a deterministic rejection and let the remaining batch continue."""

    record = preflight.record
    final_directory = benchmark_root / record.sample_id
    temporary = benchmark_root / f".{record.sample_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        source_rgba, source_bytes, _ = _load_source_rgba(preflight.source_path)
        candidate_rgb, candidate_bytes, _ = _load_candidate_rgb(
            preflight.candidate_path
        )
        (temporary / "source_rgba.png").write_bytes(source_bytes)
        (temporary / "candidate_rgb.png").write_bytes(candidate_bytes)
        empty_mask = Image.new("L", candidate_rgb.size, 0)
        mask_sha256 = _save_png(temporary / "candidate_mask.png", empty_mask)
        reviews = {
            kind: _synthetic_review(kind, f"not_run: processing_failed: {error}")
            for kind in cast(tuple[JudgeKind, ...], ("identity", "locality", "usability"))
        }
        for kind, review in reviews.items():
            write_json_atomic(
                temporary / f"review_{kind}.json",
                review.model_dump(mode="json"),
            )
        result: dict[str, object] = {
            "status": "rejected",
            "backend": QWEN_COMPLETION_BACKEND,
            "mode": QWEN_COMPLETION_MODE,
            "sample_id": record.sample_id,
            "clip_uid": record.clip_uid,
            "entity_id": record.entity_id,
            "reference_type": record.reference_type,
            "entity_phrase": record.entity_phrase,
            "source_path": str(preflight.source_path),
            "source_sha256": preflight.source_sha256,
            "localized_result_path": str(preflight.localized_result_path),
            "candidate_rgb_path": "candidate_rgb.png",
            "candidate_rgb_sha256": preflight.candidate_sha256,
            "candidate_mask_path": "candidate_mask.png",
            "candidate_mask_sha256": mask_sha256,
            "candidate_reference_path": None,
            "candidate_reference_sha256": None,
            "mask_metrics": {},
            "source_metrics": compute_foreground_metrics(
                source_rgba.getchannel("A")
            ),
            "candidate_metrics": {},
            "improvement_metrics": {},
            "background_metrics": {},
            "identity_review": reviews["identity"].model_dump(mode="json"),
            "locality_review": reviews["locality"].model_dump(mode="json"),
            "usability_review": reviews["usability"].model_dump(mode="json"),
            "judge_statuses": {
                "identity": "not_run",
                "locality": "not_run",
                "usability": "not_run",
            },
            "publication_gate": {
                "hard_checks_passed": False,
                "mask_gate_passed": False,
                "improvement_gate_passed": False,
                "background_gate_passed": False,
                "identity_gate_passed": False,
                "locality_gate_passed": False,
                "usability_gate_passed": False,
            },
            "thresholds": {},
            "rejection_reasons": ["processing_failed"],
            "reason": "processing_failed",
        }
        write_json_atomic(temporary / "result.json", result)
        temporary.replace(final_directory)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _publication_summary(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    reference_types: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    judge_rejections: Counter[str] = Counter()
    for result in results:
        reference_types[str(result["reference_type"])] += 1
        if result["status"] == "rejected":
            rejection_reasons[str(result["reason"])] += 1
        gates = cast(dict[str, bool], result["publication_gate"])
        statuses = cast(dict[str, str], result["judge_statuses"])
        for kind in ("identity", "locality", "usability"):
            if statuses[kind] != "not_run" and not gates[f"{kind}_gate_passed"]:
                judge_rejections[kind] += 1
    auto_published = sum(result["status"] == "auto_published" for result in results)
    return {
        "processed": len(results),
        "auto_published": auto_published,
        "rejected": len(results) - auto_published,
        "reference_type_counts": dict(sorted(reference_types.items())),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "judge_rejection_counts": {
            kind: judge_rejections[kind]
            for kind in ("identity", "locality", "usability")
        },
    }


def run_reference_completion_publication(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    config: PublicationConfig,
    segmenter: SegmentationBackend,
    identity_judge: CompletionPublicationJudge,
    locality_judge: CompletionPublicationJudge,
    usability_judge: CompletionPublicationJudge,
) -> PublicationStats:
    """Preflight the whole batch, then publish each sample transactionally."""

    config.validate()
    resolved_root = _resolve_publication_run(benchmark_root, config)
    records = load_publication_manifest(manifest_path, config=config)
    preflights = tuple(_preflight_record(record, config) for record in records)
    resolved_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    for preflight in preflights:
        try:
            result = _process_preflight(
                preflight,
                benchmark_root=resolved_root,
                config=config,
                segmenter=segmenter,
                identity_judge=identity_judge,
                locality_judge=locality_judge,
                usability_judge=usability_judge,
            )
        except Exception as exc:  # noqa: BLE001 - continue remaining records
            result = _publish_processing_failure(
                preflight,
                benchmark_root=resolved_root,
                error=exc,
            )
        results.append(result)
    summary = _publication_summary(results)
    write_json_atomic(resolved_root / "publication_summary.json", summary)
    return PublicationStats(
        processed=cast(int, summary["processed"]),
        auto_published=cast(int, summary["auto_published"]),
        rejected=cast(int, summary["rejected"]),
    )
