from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import io
import json
import math
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

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

ReferenceType = Literal["subject", "object", "group"]
CompletionStrategy = Literal["text_guided", "shape_guided"]
CompletionSide = Literal["top", "bottom", "left", "right"]
CompletionCandidateRegionKind = Literal["completion_mask", "editable_region"]

DEFAULT_STRATEGIES: tuple[CompletionStrategy, ...] = (
    "text_guided",
    "shape_guided",
)
DEFAULT_SEEDS = (0, 17)
_SIDE_ORDER: tuple[CompletionSide, ...] = (
    "top",
    "bottom",
    "left",
    "right",
)

ALLOWED_INPUT_ROOT = Path("/mnt/workspace/litengjie/data").resolve()
ALLOWED_BENCHMARK_ROOT = (
    ALLOWED_INPUT_ROOT / "reference_completion_benchmarks"
).resolve()

JUDGE_SYSTEM_PROMPT = """You judge an entity-reference completion benchmark.

This task extends the same entity; it must not insert a second entity. The
visible source region must remain exactly the same and the completion must be
physically connected to it. For a person, require the same face, hairstyle,
clothing, skin tone, body type, pose direction, perspective, and lighting.
Reject when identity is uncertain. Reject a second face or person, duplicate or
overlapping bodies, extra limbs, disconnected parts, reconstructed occluders,
new salient entities, implausible structure, broken boundaries, or obvious
copy-paste artifacts. Reject when only background was generated and the missing
entity was not completed. Judge the images, not the prompt's identity claim.

Return one strict JSON object containing only verdict,
visible_source_preserved, same_entity_continued, identity_preserved,
exactly_one_entity, completion_plausible, completion_useful,
no_occluder_reconstructed, no_new_salient_entity, boundary_clean,
reference_usable, and reason. verdict is accept if and only if every boolean is
true. Return JSON only."""


class CompletionManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    clip_uid: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    reference_type: ReferenceType
    entity_phrase: str = Field(min_length=1)
    source_rgba_path: Path
    context_rgb_path: Optional[Path]
    completion_mask_path: Optional[Path]
    completion_sides: tuple[CompletionSide, ...]
    completion_start_ratio: float

    @field_validator("clip_uid", "entity_id", "entity_phrase")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be non-empty")
        return stripped

    @field_validator("completion_sides", mode="before")
    @classmethod
    def _validate_sides(cls, value: object) -> tuple[CompletionSide, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("completion_sides must be an explicit list")
        sides = tuple(value)
        if not sides:
            raise ValueError("completion_sides must be non-empty")
        if len(set(sides)) != len(sides):
            raise ValueError("completion_sides must not contain duplicates")
        unknown = [side for side in sides if side not in _SIDE_ORDER]
        if unknown:
            raise ValueError(f"unsupported completion sides: {unknown}")
        return tuple(side for side in _SIDE_ORDER if side in sides)

    @field_validator("completion_start_ratio")
    @classmethod
    def _validate_start_ratio(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("completion_start_ratio must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError("completion_start_ratio must be between 0 and 1")
        return value


class ReferenceCompletionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    visible_source_preserved: StrictBool
    same_entity_continued: StrictBool
    identity_preserved: StrictBool
    exactly_one_entity: StrictBool
    completion_plausible: StrictBool
    completion_useful: StrictBool
    no_occluder_reconstructed: StrictBool
    no_new_salient_entity: StrictBool
    boundary_clean: StrictBool
    reference_usable: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must be non-empty")
        return stripped

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> ReferenceCompletionReview:
        all_pass = all(
            (
                self.visible_source_preserved,
                self.same_entity_continued,
                self.identity_preserved,
                self.exactly_one_entity,
                self.completion_plausible,
                self.completion_useful,
                self.no_occluder_reconstructed,
                self.no_new_salient_entity,
                self.boundary_clean,
                self.reference_usable,
            )
        )
        if (self.verdict == "accept") != all_pass:
            raise ValueError(
                "verdict must be accept if and only if every quality flag is true"
            )
        return self


class ReferenceCompletionBackend(Protocol):
    def complete(
        self,
        *,
        input_rgb: Image.Image,
        completion_mask: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        strategy: CompletionStrategy,
        seed: int,
        fitting_degree: float,
        prompt: str,
        negative_prompt: str,
    ) -> Image.Image: ...


class ReferenceCompletionJudge(Protocol):
    def review(
        self,
        *,
        source_rgba: Image.Image,
        completion_canvas: Image.Image,
        completion_mask: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        completion_region_label: str = "completion mask",
        system_prompt_addendum: str = "",
    ) -> ReferenceCompletionReview: ...


class ReferenceCompletionJudgeFailure(StructuredOutputFailure):
    pass


@dataclass(frozen=True)
class PowerPaintV21CompletionConfig:
    powerpaint_repo_path: Path
    checkpoint_dir: Path
    device: str = "cuda"
    dtype: str = "float16"
    num_inference_steps: int = 40
    guidance_scale: float = 7.5
    brushnet_conditioning_scale: float = 1.0
    fitting_degree: float = 0.55
    enable_model_cpu_offload: bool = True
    local_files_only: bool = True
    canvas_expand_ratio: float = 0.75
    lateral_padding_ratio: float = 0.20
    mask_overlap_pixels: int = 0
    model_min_side: int = 640
    model_multiple: int = 8

    def validate(self) -> None:
        if not isinstance(self.powerpaint_repo_path, Path):
            raise TypeError("powerpaint_repo_path must be a pathlib.Path")
        if not isinstance(self.checkpoint_dir, Path):
            raise TypeError("checkpoint_dir must be a pathlib.Path")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.local_files_only is not True:
            raise ValueError("PowerPaint benchmark requires local_files_only=true")
        if not isinstance(self.enable_model_cpu_offload, bool):
            raise TypeError("enable_model_cpu_offload must be boolean")
        if (
            not isinstance(self.num_inference_steps, int)
            or isinstance(self.num_inference_steps, bool)
            or self.num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        for name, value, minimum, maximum in (
            ("guidance_scale", self.guidance_scale, 0.0, None),
            (
                "brushnet_conditioning_scale",
                self.brushnet_conditioning_scale,
                0.0,
                None,
            ),
            ("fitting_degree", self.fitting_degree, 0.0, 1.0),
            ("canvas_expand_ratio", self.canvas_expand_ratio, 0.0, None),
            ("lateral_padding_ratio", self.lateral_padding_ratio, 0.0, None),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and at least {minimum}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must be at most {maximum}")
        if self.canvas_expand_ratio <= 0:
            raise ValueError("canvas_expand_ratio must be positive")
        if (
            not isinstance(self.mask_overlap_pixels, int)
            or isinstance(self.mask_overlap_pixels, bool)
            or self.mask_overlap_pixels < 0
        ):
            raise ValueError("mask_overlap_pixels must be a non-negative integer")
        for name, value in (
            ("model_min_side", self.model_min_side),
            ("model_multiple", self.model_multiple),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CompletionCanvas:
    baseline_rgb: Image.Image
    visible_mask: Image.Image
    candidate_region: Image.Image
    candidate_region_kind: CompletionCandidateRegionKind
    source_offset_xy: tuple[int, int]
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]
    candidate_region_area_ratio: float

    @property
    def completion_mask(self) -> Image.Image:
        """Compatibility alias for PowerPaint completion-mask callers."""
        return self.candidate_region

    @property
    def completion_mask_area_ratio(self) -> float:
        """Compatibility alias for existing PowerPaint result metadata."""
        return self.candidate_region_area_ratio


@dataclass(frozen=True)
class _CompletionCanvasBase:
    baseline_rgb: Image.Image
    visible_mask: Image.Image
    visible_bbox: tuple[int, int, int, int]
    source_offset_xy: tuple[int, int]
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]


@dataclass(frozen=True)
class _CompletionPreflight:
    source_path: Path
    context_path: Path | None
    completion_mask_path: Path | None
    completion_mask_source_sha256: str | None


@dataclass(frozen=True)
class ModelSpaceTransform:
    canvas_size: tuple[int, int]
    model_size: tuple[int, int]
    scale_xy: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "canvas_size": list(self.canvas_size),
            "model_size": list(self.model_size),
            "scale_xy": list(self.scale_xy),
            "image_resize_filter": "lanczos",
            "mask_resize_filter": "nearest",
            "restore_filter": "lanczos",
        }


@dataclass(frozen=True)
class HardCheckReport:
    passed: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "passed" if self.passed else "failed",
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CompletionBenchmarkStats:
    processed: int
    accepted: int
    rejected: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionCandidateSpec:
    file_stem: str
    seed: int
    prompt: str
    negative_prompt: str
    attempt_metadata: dict[str, object]
    selection_metadata: dict[str, object]


@dataclass(frozen=True)
class CompletionBenchmarkRun:
    stats: CompletionBenchmarkStats
    results: tuple[dict[str, object], ...]


class CompletionCandidateGenerator(Protocol):
    def __call__(
        self,
        *,
        record: CompletionManifestRecord,
        candidate: CompletionCandidateSpec,
        input_rgb: Image.Image,
        candidate_region: Image.Image,
    ) -> Image.Image: ...


class CompletionCanvasRuntimeConfig(Protocol):
    canvas_expand_ratio: float
    lateral_padding_ratio: float
    mask_overlap_pixels: int
    model_min_side: int
    model_multiple: int

    def validate(self) -> None: ...


@dataclass(frozen=True)
class CompletionRunPolicy:
    candidate_region_kind: CompletionCandidateRegionKind = "completion_mask"
    preserve_outside_candidate_region: bool = True
    region_artifact_name: str = "completion_mask.png"
    judge_region_label: str = "completion mask"
    judge_system_prompt_addendum: str = ""

    def validate(self) -> None:
        if self.candidate_region_kind == "completion_mask":
            if self.region_artifact_name != "completion_mask.png":
                raise ValueError("completion-mask policy must publish completion_mask.png")
        elif self.candidate_region_kind == "editable_region":
            if self.preserve_outside_candidate_region:
                raise ValueError("editable-region policy cannot preserve its outside")
            if self.region_artifact_name != "editable_region.png":
                raise ValueError("editable-region policy must publish editable_region.png")
        else:
            raise ValueError("unsupported candidate region kind")
        if not self.judge_region_label.strip():
            raise ValueError("judge_region_label must be non-empty")


@dataclass(frozen=True)
class PowerPaintTaskPrompts:
    promptA: str
    promptB: str
    promptU: str
    negative_promptA: str
    negative_promptB: str
    negative_promptU: str


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _resolve_input_path(path: Path, *, field_name: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    resolved = path.expanduser().resolve(strict=True)
    if not _is_at_or_below(resolved, ALLOWED_INPUT_ROOT):
        raise ValueError(f"{field_name} must remain under {ALLOWED_INPUT_ROOT}")
    return resolved


def _resolve_benchmark_root(
    path: Path,
    *,
    allowed_benchmark_root: Path | None = None,
) -> Path:
    allowed_root = (
        ALLOWED_BENCHMARK_ROOT
        if allowed_benchmark_root is None
        else allowed_benchmark_root.resolve()
    )
    if not path.is_absolute():
        raise ValueError("benchmark_root must be an absolute path")
    resolved = path.expanduser().resolve(strict=False)
    if resolved == allowed_root or not _is_at_or_below(
        resolved,
        allowed_root,
    ):
        raise ValueError(
            "benchmark_root must be a run directory strictly below "
            f"{allowed_root}"
        )
    forbidden = {"selected", "r2v_v3_runs", "r2v_v3_datasets"}
    relative_parts = {
        part.casefold() for part in resolved.relative_to(allowed_root).parts
    }
    if relative_parts & forbidden:
        raise ValueError("benchmark_root targets a forbidden production directory")
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


def _save_png(path: Path, image: Image.Image) -> None:
    image.save(path, format="PNG")
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != image.mode:
            raise RuntimeError("published PNG format or mode changed unexpectedly")
        if opened.size != image.size:
            raise RuntimeError("published PNG dimensions changed unexpectedly")


def _load_source_rgba(path: Path) -> tuple[Image.Image, bytes, str]:
    if path.suffix.casefold() != ".png":
        raise ValueError("source reference must use a .png path")
    source_bytes = path.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as opened:
        opened.load()
        if opened.format != "PNG" or opened.mode != "RGBA":
            raise ValueError("source reference must be an RGBA PNG")
        source = opened.copy()
    alpha = np.asarray(source.getchannel("A"), dtype=np.uint8)
    unique = {int(value) for value in np.unique(alpha)}
    if not unique.issubset({0, 255}):
        raise ValueError("source alpha must contain only 0 and 255")
    if 255 not in unique:
        raise ValueError("source reference foreground is empty")
    return source, source_bytes, _sha256_bytes(source_bytes)


def _load_context_rgb(
    path: Path,
    *,
    expected_size: tuple[int, int],
) -> tuple[Image.Image, bytes, str]:
    context_bytes = path.read_bytes()
    with Image.open(io.BytesIO(context_bytes)) as opened:
        opened.load()
        if opened.format not in {"PNG", "JPEG"} or opened.mode != "RGB":
            raise ValueError("context reference must be an RGB PNG or JPEG")
        if opened.size != expected_size:
            raise ValueError("context dimensions must match source RGBA")
        context = opened.copy()
    return context, context_bytes, _sha256_bytes(context_bytes)


def _visible_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(alpha)
    if not len(xs):
        raise ValueError("source reference foreground is empty")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for y_offset in range(3):
        for x_offset in range(3):
            result |= padded[
                y_offset : y_offset + height,
                x_offset : x_offset + width,
            ]
    return result


def _build_completion_canvas_base(
    *,
    source_rgba: Image.Image,
    context_rgb: Image.Image | None,
    completion_sides: tuple[CompletionSide, ...],
    config: CompletionCanvasRuntimeConfig,
) -> _CompletionCanvasBase:
    if source_rgba.mode != "RGBA":
        raise ValueError("completion source must be RGBA")
    if not completion_sides:
        raise ValueError("completion_sides must be non-empty")
    if context_rgb is not None and (
        context_rgb.mode != "RGB" or context_rgb.size != source_rgba.size
    ):
        raise ValueError("context RGB must match source dimensions")
    config.validate()
    source_width, source_height = source_rgba.size
    expand_x = max(1, round(source_width * config.canvas_expand_ratio))
    expand_y = max(1, round(source_height * config.canvas_expand_ratio))
    left_expand = expand_x if "left" in completion_sides else 0
    right_expand = expand_x if "right" in completion_sides else 0
    top_expand = expand_y if "top" in completion_sides else 0
    bottom_expand = expand_y if "bottom" in completion_sides else 0
    offset_x, offset_y = left_expand, top_expand
    canvas_width = source_width + left_expand + right_expand
    canvas_height = source_height + top_expand + bottom_expand

    baseline = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    source_rectangle = np.s_[
        offset_y : offset_y + source_height,
        offset_x : offset_x + source_width,
    ]
    if context_rgb is not None:
        baseline[source_rectangle] = np.asarray(context_rgb, dtype=np.uint8)
    source = np.asarray(source_rgba, dtype=np.uint8)
    source_visible = source[:, :, 3] == 255
    baseline_region = baseline[source_rectangle]
    baseline_region[source_visible] = source[:, :, :3][source_visible]

    visible = np.zeros((canvas_height, canvas_width), dtype=bool)
    visible[source_rectangle] = source_visible
    x1, y1, x2, y2 = _visible_bbox(source_visible)
    x1 += offset_x
    x2 += offset_x
    y1 += offset_y
    y2 += offset_y

    return _CompletionCanvasBase(
        baseline_rgb=Image.fromarray(baseline),
        visible_mask=Image.fromarray(visible.astype(np.uint8) * 255),
        visible_bbox=(x1, y1, x2, y2),
        source_offset_xy=(offset_x, offset_y),
        source_size=source_rgba.size,
        canvas_size=(canvas_width, canvas_height),
    )


def _validate_completion_mask(
    mask_image: Image.Image,
    *,
    canvas_base: _CompletionCanvasBase,
) -> np.ndarray:
    if mask_image.mode != "L":
        raise ValueError("completion mask must use PIL mode L")
    if mask_image.size != canvas_base.canvas_size:
        raise ValueError("completion mask dimensions must match completion canvas")
    mask_values = np.asarray(mask_image, dtype=np.uint8)
    if not {int(value) for value in np.unique(mask_values)}.issubset({0, 255}):
        raise ValueError("completion mask must contain only 0 and 255")
    mask = mask_values == 255
    visible = np.asarray(canvas_base.visible_mask, dtype=np.uint8) == 255
    if not mask.any():
        raise ValueError("completion mask is empty")
    if mask.all():
        raise ValueError("completion mask covers the full canvas")
    if np.logical_and(mask, visible).any():
        raise ValueError("completion mask overlaps visible entity pixels")
    if not np.logical_and(mask, _dilate_one_pixel(visible) & ~visible).any():
        raise ValueError("completion mask is disconnected from visible entity")
    if np.array_equal(mask, ~visible):
        raise ValueError("completion mask cannot cover all transparent pixels")
    return mask


def _build_directional_completion_mask(
    *,
    canvas_base: _CompletionCanvasBase,
    completion_sides: tuple[CompletionSide, ...],
    completion_start_ratio: float,
    config: CompletionCanvasRuntimeConfig,
) -> Image.Image:
    visible = np.asarray(canvas_base.visible_mask, dtype=np.uint8) == 255
    x1, y1, x2, y2 = canvas_base.visible_bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    canvas_width, canvas_height = canvas_base.canvas_size
    lateral_x = round(bbox_width * config.lateral_padding_ratio)
    lateral_y = round(bbox_height * config.lateral_padding_ratio)
    overlap = config.mask_overlap_pixels
    mask = np.zeros_like(visible)

    if "bottom" in completion_sides:
        y_start = y1 + math.floor(completion_start_ratio * bbox_height) - overlap
        mask[
            max(0, y_start) : canvas_height,
            max(0, x1 - lateral_x) : min(canvas_width, x2 + lateral_x),
        ] = True
    if "top" in completion_sides:
        y_end = y2 - math.floor(completion_start_ratio * bbox_height) + overlap
        mask[
            0 : min(canvas_height, y_end),
            max(0, x1 - lateral_x) : min(canvas_width, x2 + lateral_x),
        ] = True
    if "right" in completion_sides:
        x_start = x1 + math.floor(completion_start_ratio * bbox_width) - overlap
        mask[
            max(0, y1 - lateral_y) : min(canvas_height, y2 + lateral_y),
            max(0, x_start) : canvas_width,
        ] = True
    if "left" in completion_sides:
        x_end = x2 - math.floor(completion_start_ratio * bbox_width) + overlap
        mask[
            max(0, y1 - lateral_y) : min(canvas_height, y2 + lateral_y),
            0 : min(canvas_width, x_end),
        ] = True

    mask[visible] = False
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    _validate_completion_mask(mask_image, canvas_base=canvas_base)
    return mask_image


def _load_explicit_completion_mask(
    path: Path,
    *,
    canvas_base: _CompletionCanvasBase,
) -> tuple[Image.Image, str]:
    if path.suffix.casefold() != ".png":
        raise ValueError("explicit completion mask must use a .png path")
    mask_bytes = path.read_bytes()
    with Image.open(io.BytesIO(mask_bytes)) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("explicit completion mask must be a PNG")
        if opened.mode != "L":
            raise ValueError("explicit completion mask must use PIL mode L")
        mask_image = opened.copy()
    _validate_completion_mask(mask_image, canvas_base=canvas_base)
    return mask_image, _sha256_bytes(mask_bytes)


def _finish_completion_canvas(
    *,
    canvas_base: _CompletionCanvasBase,
    completion_mask: Image.Image,
) -> CompletionCanvas:
    mask = _validate_completion_mask(
        completion_mask,
        canvas_base=canvas_base,
    )

    return CompletionCanvas(
        baseline_rgb=canvas_base.baseline_rgb,
        visible_mask=canvas_base.visible_mask,
        candidate_region=completion_mask.copy(),
        candidate_region_kind="completion_mask",
        source_offset_xy=canvas_base.source_offset_xy,
        source_size=canvas_base.source_size,
        canvas_size=canvas_base.canvas_size,
        candidate_region_area_ratio=float(mask.mean()),
    )


def build_completion_canvas(
    *,
    source_rgba: Image.Image,
    context_rgb: Image.Image | None,
    completion_sides: tuple[CompletionSide, ...],
    completion_start_ratio: float,
    config: CompletionCanvasRuntimeConfig,
    explicit_completion_mask: Image.Image | None = None,
) -> CompletionCanvas:
    canvas_base = _build_completion_canvas_base(
        source_rgba=source_rgba,
        context_rgb=context_rgb,
        completion_sides=completion_sides,
        config=config,
    )
    completion_mask = explicit_completion_mask
    if completion_mask is None:
        completion_mask = _build_directional_completion_mask(
            canvas_base=canvas_base,
            completion_sides=completion_sides,
            completion_start_ratio=completion_start_ratio,
            config=config,
        )
    return _finish_completion_canvas(
        canvas_base=canvas_base,
        completion_mask=completion_mask,
    )


def build_editable_completion_canvas(
    *,
    source_rgba: Image.Image,
    context_rgb: Image.Image | None,
    completion_sides: tuple[CompletionSide, ...],
    config: CompletionCanvasRuntimeConfig,
) -> CompletionCanvas:
    canvas_base = _build_completion_canvas_base(
        source_rgba=source_rgba,
        context_rgb=context_rgb,
        completion_sides=completion_sides,
        config=config,
    )
    visible = np.asarray(canvas_base.visible_mask, dtype=np.uint8) == 255
    editable = ~visible
    editable_region = Image.fromarray(editable.astype(np.uint8) * 255)
    return CompletionCanvas(
        baseline_rgb=canvas_base.baseline_rgb,
        visible_mask=canvas_base.visible_mask,
        candidate_region=editable_region,
        candidate_region_kind="editable_region",
        source_offset_xy=canvas_base.source_offset_xy,
        source_size=canvas_base.source_size,
        canvas_size=canvas_base.canvas_size,
        candidate_region_area_ratio=float(editable.mean()),
    )


def _align_up(value: float, multiple: int) -> int:
    return max(multiple, math.ceil(value / multiple) * multiple)


def build_model_space_transform(
    canvas_size: tuple[int, int],
    *,
    model_min_side: int,
    model_multiple: int,
) -> ModelSpaceTransform:
    width, height = canvas_size
    if width <= 0 or height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if model_min_side <= 0 or model_multiple <= 0:
        raise ValueError("model size settings must be positive")
    if width <= height:
        model_width = _align_up(model_min_side, model_multiple)
        model_height = _align_up(
            model_width * height / width,
            model_multiple,
        )
    else:
        model_height = _align_up(model_min_side, model_multiple)
        model_width = _align_up(
            model_height * width / height,
            model_multiple,
        )
    return ModelSpaceTransform(
        canvas_size=canvas_size,
        model_size=(model_width, model_height),
        scale_xy=(model_width / width, model_height / height),
    )


def resize_completion_inputs(
    canvas: CompletionCanvas,
    transform: ModelSpaceTransform,
) -> tuple[Image.Image, Image.Image]:
    if canvas.canvas_size != transform.canvas_size:
        raise ValueError("model transform does not match completion canvas")
    model_rgb = canvas.baseline_rgb.resize(
        transform.model_size,
        resample=Image.Resampling.LANCZOS,
    )
    model_region = canvas.candidate_region.resize(
        transform.model_size,
        resample=Image.Resampling.NEAREST,
    )
    region_values = np.asarray(model_region, dtype=np.uint8)
    if not {int(value) for value in np.unique(region_values)}.issubset({0, 255}):
        raise RuntimeError("model-space candidate region is not binary")
    return model_rgb, model_region


def build_completion_prompt(
    *,
    entity_phrase: str,
    reference_type: ReferenceType,
) -> str:
    phrase = entity_phrase.strip()
    if not phrase:
        raise ValueError("entity_phrase must be non-empty")
    templates = {
        "subject": (
            "Complete the same single person shown in the preserved visible "
            "region. Continue the same identity, face, hairstyle, clothing, body "
            "proportions, pose direction, lighting, and perspective. Reconstruct "
            "only the missing or occluded parts inside the completion mask. "
            "Produce exactly one person. Do not replace or redesign the visible "
            "person."
        ),
        "object": (
            "Complete the same single object shown in the preserved visible "
            "region. Continue its exact geometry, material, texture, color, scale, "
            "viewpoint, and lighting. Reconstruct only the missing or occluded "
            "parts inside the completion mask. Produce exactly one coherent object."
        ),
        "group": (
            "Complete only the missing parts of the same coherent group. Preserve "
            "the visible members, arrangement, style, geometry, and identity. Do "
            "not create an unrelated second group. Reconstruct only inside the "
            "completion mask."
        ),
    }
    return f"{templates[reference_type]} Entity description: {phrase}."


def build_completion_negative_prompt(reference_type: ReferenceType) -> str:
    shared = [
        "duplicate",
        "identity change",
        "different clothing",
        "new foreground object",
        "text",
        "watermark",
        "logo",
    ]
    if reference_type in {"subject", "group"}:
        shared.extend(
            [
                "second person",
                "extra person",
                "different person",
                "multiple bodies",
                "extra head",
                "extra face",
                "extra arms",
                "extra legs",
                "deformed anatomy",
                "disconnected body",
                "floating body parts",
            ]
        )
    return ", ".join(shared)


def build_powerpaint_task_prompts(
    *,
    strategy: CompletionStrategy,
    prompt: str,
    negative_prompt: str,
) -> PowerPaintTaskPrompts:
    if strategy == "text_guided":
        prompt_a = prompt_b = negative_a = negative_b = "P_obj"
    elif strategy == "shape_guided":
        prompt_a = negative_a = "P_shape"
        prompt_b = negative_b = "P_ctxt"
    else:
        raise ValueError(f"unsupported completion strategy: {strategy}")
    return PowerPaintTaskPrompts(
        promptA=prompt_a,
        promptB=prompt_b,
        promptU=prompt,
        negative_promptA=negative_a,
        negative_promptB=negative_b,
        negative_promptU=negative_prompt,
    )


def validate_powerpaint_v21_layout(
    config: PowerPaintV21CompletionConfig,
) -> dict[str, Path]:
    config.validate()
    repo = config.powerpaint_repo_path.expanduser().resolve(strict=False)
    checkpoint = config.checkpoint_dir.expanduser().resolve(strict=False)
    required = {
        "base_model": checkpoint / "realisticVisionV60B1_v51VAE",
        "brushnet_directory": checkpoint / "PowerPaint_Brushnet",
        "brushnet_weights": checkpoint
        / "PowerPaint_Brushnet"
        / "diffusion_pytorch_model.safetensors",
        "text_encoder_weights": checkpoint
        / "PowerPaint_Brushnet"
        / "pytorch_model.bin",
        "brushnet_source": repo / "powerpaint" / "models" / "BrushNet_CA.py",
        "unet_source": repo / "powerpaint" / "models" / "unet_2d_condition.py",
        "pipeline_source": repo
        / "powerpaint"
        / "pipelines"
        / "pipeline_PowerPaint_Brushnet_CA.py",
        "utils_source": repo / "powerpaint" / "utils" / "utils.py",
    }
    for name, path in required.items():
        expected_directory = name in {"base_model", "brushnet_directory"}
        exists = path.is_dir() if expected_directory else path.is_file()
        if not exists:
            kind = "directory" if expected_directory else "file"
            raise FileNotFoundError(
                f"PowerPaint v2-1 required {kind} is missing: {path}"
            )
    return required


@contextmanager
def _temporary_import_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


def _supports_parameter(callable_object: Any, name: str) -> bool:
    parameters = inspect.signature(callable_object).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _require_parameters(
    callable_object: Any,
    names: tuple[str, ...],
    *,
    operation: str,
) -> None:
    missing = [name for name in names if not _supports_parameter(callable_object, name)]
    if missing:
        raise RuntimeError(
            f"installed PowerPaint API lacks {operation} parameters {missing}"
        )


class PowerPaintV21ReferenceCompletionBackend:
    def __init__(
        self,
        config: PowerPaintV21CompletionConfig,
        *,
        pipeline: Any | None = None,
        torch_module: Any | None = None,
    ) -> None:
        config.validate()
        if (pipeline is None) != (torch_module is None):
            raise ValueError("pipeline and torch_module must be injected together")
        self.config = config
        self._pipeline = pipeline
        self._torch = torch_module
        self._load_error: BaseException | None = None

    def _load(self) -> None:
        paths = validate_powerpaint_v21_layout(self.config)
        repo = self.config.powerpaint_repo_path.expanduser().resolve()
        with _temporary_import_path(repo):
            brushnet_module = importlib.import_module("powerpaint.models.BrushNet_CA")
            unet_module = importlib.import_module("powerpaint.models.unet_2d_condition")
            pipeline_module = importlib.import_module(
                "powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA"
            )
            utils_module = importlib.import_module("powerpaint.utils.utils")
        import torch
        from diffusers import UniPCMultistepScheduler
        from safetensors.torch import load_model
        from transformers import CLIPTextModel

        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise RuntimeError(
                f"installed torch does not provide dtype {self.config.dtype}"
            )
        UNet2DConditionModel = unet_module.UNet2DConditionModel
        BrushNetModel = brushnet_module.BrushNetModel
        Pipeline = pipeline_module.StableDiffusionPowerPaintBrushNetPipeline
        TokenizerWrapper = utils_module.TokenizerWrapper
        add_tokens = utils_module.add_tokens
        base_model = str(paths["base_model"])
        common = {"local_files_only": True}
        unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            torch_dtype=dtype,
            **common,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            base_model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            **common,
        )
        brushnet = BrushNetModel.from_unet(unet)
        pipeline = Pipeline.from_pretrained(
            base_model,
            unet=unet,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=False,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        tokenizer = TokenizerWrapper(
            from_pretrained=base_model,
            subfolder="tokenizer",
            revision=None,
            local_files_only=True,
        )
        wrapped_tokenizer = object.__getattribute__(tokenizer, "wrapped")
        tokenizer._module_cls = type(wrapped_tokenizer)
        tokenizer._module_name = type(wrapped_tokenizer).__name__
        add_tokens(
            tokenizer=tokenizer,
            text_encoder=text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        pipeline.tokenizer = tokenizer
        for component_name, expected_component in (
            ("unet", unet),
            ("brushnet", brushnet),
            ("text_encoder_brushnet", text_encoder_brushnet),
            ("tokenizer", tokenizer),
        ):
            if getattr(pipeline, component_name, None) is not expected_component:
                raise RuntimeError(
                    "PowerPaint pipeline did not preserve the expected "
                    f"{component_name} component"
                )
        load_model(brushnet, str(paths["brushnet_weights"]))
        torch_load_parameters: dict[str, object] = {"map_location": "cpu"}
        if _supports_parameter(torch.load, "weights_only"):
            torch_load_parameters["weights_only"] = True
        text_state = torch.load(
            str(paths["text_encoder_weights"]),
            **torch_load_parameters,
        )
        if isinstance(text_state, dict) and set(text_state) == {"state_dict"}:
            text_state = text_state["state_dict"]
        incompatible_keys = text_encoder_brushnet.load_state_dict(
            text_state,
            strict=False,
        )
        for field_name in ("missing_keys", "unexpected_keys"):
            if not hasattr(incompatible_keys, field_name):
                raise TypeError(
                    "PowerPaint text encoder load_state_dict must return "
                    "torch-compatible incompatible keys"
                )
            keys = getattr(incompatible_keys, field_name)
            if not isinstance(keys, (list, tuple)):
                raise TypeError(
                    "PowerPaint text encoder incompatible keys must be lists or tuples"
                )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        progress = getattr(pipeline, "set_progress_bar_config", None)
        if callable(progress):
            progress(disable=True)
        if self.config.enable_model_cpu_offload:
            offload = getattr(pipeline, "enable_model_cpu_offload", None)
            if not callable(offload):
                raise TypeError("PowerPaint pipeline cannot enable model CPU offload")
            offload()
        else:
            to_method = getattr(pipeline, "to", None)
            if not callable(to_method):
                raise TypeError("PowerPaint pipeline has no device transfer API")
            pipeline = to_method(self.config.device)
        self._pipeline = pipeline
        self._torch = torch

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(
                "PowerPaint v2-1 benchmark backend previously failed to load"
            ) from self._load_error
        try:
            self._load()
        except Exception as exc:
            self._load_error = exc
            raise

    def complete(
        self,
        *,
        input_rgb: Image.Image,
        completion_mask: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        strategy: CompletionStrategy,
        seed: int,
        fitting_degree: float,
        prompt: str,
        negative_prompt: str,
    ) -> Image.Image:
        if input_rgb.mode != "RGB" or completion_mask.mode != "L":
            raise ValueError("PowerPaint requires RGB input and L mask")
        if input_rgb.size != completion_mask.size:
            raise ValueError("PowerPaint input and mask dimensions must match")
        mask_array = np.asarray(completion_mask, dtype=np.uint8)
        if not {int(value) for value in np.unique(mask_array)}.issubset({0, 255}):
            raise ValueError("PowerPaint completion mask must be binary")
        if not entity_phrase.strip() or reference_type not in {
            "subject",
            "object",
            "group",
        }:
            raise ValueError("PowerPaint entity metadata is invalid")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("PowerPaint seed must be a non-negative integer")
        if not math.isfinite(fitting_degree) or not 0.0 <= fitting_degree <= 1.0:
            raise ValueError("fitting_degree must be between 0 and 1")
        if not prompt.strip() or not negative_prompt.strip():
            raise ValueError("PowerPaint prompts must be non-empty")
        tasks = build_powerpaint_task_prompts(
            strategy=strategy,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        self._ensure_loaded()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("PowerPaint v2-1 backend is unavailable")

        input_array = np.asarray(input_rgb, dtype=np.uint8)
        normalized_mask = mask_array.astype(np.float32) / 255.0
        masked_input = np.rint(
            input_array.astype(np.float32) * (1.0 - normalized_mask[:, :, None])
        ).astype(np.uint8)
        masked_image = Image.fromarray(masked_input)
        mask_rgb = completion_mask.convert("RGB")
        generator = self._torch.Generator(device=self.config.device).manual_seed(seed)
        parameters = {
            **asdict(tasks),
            "tradoff": fitting_degree,
            "tradoff_nag": fitting_degree,
            "image": masked_image,
            "mask": mask_rgb,
            "num_inference_steps": self.config.num_inference_steps,
            "generator": generator,
            "brushnet_conditioning_scale": (self.config.brushnet_conditioning_scale),
            "guidance_scale": self.config.guidance_scale,
            "width": input_rgb.width,
            "height": input_rgb.height,
        }
        _require_parameters(
            self._pipeline.__call__,
            tuple(parameters),
            operation="StableDiffusionPowerPaintBrushNetPipeline.__call__",
        )
        result = self._pipeline(**parameters)
        images = getattr(result, "images", None)
        if (
            not isinstance(images, (list, tuple))
            or not images
            or not isinstance(images[0], Image.Image)
        ):
            raise TypeError("PowerPaint backend did not return a PIL image")
        output = images[0]
        if output.size != input_rgb.size:
            raise RuntimeError("PowerPaint output dimensions do not match model input")
        return output.convert("RGB")

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            del pipeline
        torch = self._torch
        self._torch = None
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _white_source(source_rgba: Image.Image) -> Image.Image:
    source = np.asarray(source_rgba, dtype=np.uint8)
    visible = source[:, :, 3] == 255
    result = np.full(source.shape[:2] + (3,), 255, dtype=np.uint8)
    result[visible] = source[:, :, :3][visible]
    return Image.fromarray(result)


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenReferenceCompletionJudge:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        repair_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        if (
            not isinstance(repair_retries, int)
            or isinstance(repair_retries, bool)
            or repair_retries < 0
        ):
            raise ValueError("repair_retries must be a non-negative integer")
        self.config = config
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
        completion_canvas: Image.Image,
        completion_mask: Image.Image,
        candidate_rgb: Image.Image,
        request_text: str,
        completion_region_label: str,
        system_prompt_addendum: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [{"type": "text", "text": request_text}]
        for label, image in (
            ("Image 1: source entity on white", _white_source(source_rgba)),
            ("Image 2: completion canvas input", completion_canvas),
            (f"Image 3: {completion_region_label}", completion_mask.convert("L")),
            ("Image 4: completed candidate", candidate_rgb),
        ):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(image)},
                }
            )
        system_prompt = JUDGE_SYSTEM_PROMPT
        if system_prompt_addendum.strip():
            system_prompt += "\n\n" + system_prompt_addendum.strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _request(self, messages: list[dict[str, object]]) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "v3_reference_completion_review",
                        "strict": True,
                        "schema": ReferenceCompletionReview.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty completion review")
        return str(content)

    def review(
        self,
        *,
        source_rgba: Image.Image,
        completion_canvas: Image.Image,
        completion_mask: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        completion_region_label: str = "completion mask",
        system_prompt_addendum: str = "",
    ) -> ReferenceCompletionReview:
        if source_rgba.mode != "RGBA":
            raise ValueError("completion judge source must be RGBA")
        if completion_canvas.mode != "RGB" or candidate_rgb.mode != "RGB":
            raise ValueError("completion judge canvas and candidate must be RGB")
        if completion_mask.mode != "L":
            raise ValueError("completion judge region must be L mode")
        if completion_canvas.size != candidate_rgb.size or (
            completion_mask.size != completion_canvas.size
        ):
            raise ValueError("completion judge canvas images must have matching sizes")
        if not completion_region_label.strip():
            raise ValueError("completion_region_label must be non-empty")
        original_request = (
            "Judge this immutable completion candidate. "
            f"reference_type={reference_type}; entity_phrase={entity_phrase}. "
            "Return the required strict JSON object."
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
                    json_schema=ReferenceCompletionReview.model_json_schema(),
                )
            try:
                raw = self._request(
                    self._messages(
                        source_rgba=source_rgba,
                        completion_canvas=completion_canvas,
                        completion_mask=completion_mask,
                        candidate_rgb=candidate_rgb,
                        request_text=request_text,
                        completion_region_label=completion_region_label,
                        system_prompt_addendum=system_prompt_addendum,
                    )
                )
            except Exception as exc:
                raise ReferenceCompletionJudgeFailure(
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
            review, issues = parse_qwen_json_issues(
                raw,
                ReferenceCompletionReview,
            )
            if review is not None and not issues:
                return review
        raise ReferenceCompletionJudgeFailure(
            raw_responses=raw_responses,
            issues=issues,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def _hard_check_candidate(
    *,
    generated_model_rgb: object,
    source_rgba: Image.Image,
    source_path: Path,
    source_sha256: str,
    canvas: CompletionCanvas,
    transform: ModelSpaceTransform,
    preserve_outside_candidate_region: bool = True,
) -> tuple[Image.Image | None, HardCheckReport]:
    checks = {
        "output_rgb": isinstance(generated_model_rgb, Image.Image)
        and generated_model_rgb.mode == "RGB",
        "dimensions_match_completion_canvas": False,
        "source_visible_pixels_exact": False,
        "no_invalid_pixels": False,
        "source_file_unchanged": False,
    }
    if preserve_outside_candidate_region:
        checks.update(
            {
                "outside_completion_mask_exact": False,
                "completion_mask_nonempty": False,
                "completion_mask_not_full_canvas": False,
                "completion_mask_connected_to_visible_entity": False,
                "candidate_changed_inside_completion_mask": False,
                "background_not_constant": False,
            }
        )
    else:
        checks.update(
            {
                "candidate_changed_outside_visible_region": False,
                "generated_region_not_constant": False,
            }
        )
    reasons: list[str] = []
    if not checks["output_rgb"]:
        reasons.append("generated_output_must_be_rgb_pil")
        return None, HardCheckReport(False, checks, tuple(reasons))
    assert isinstance(generated_model_rgb, Image.Image)
    if generated_model_rgb.size != transform.model_size:
        reasons.append("generated_model_size_mismatch")
        return None, HardCheckReport(False, checks, tuple(reasons))

    restored = generated_model_rgb.resize(
        canvas.canvas_size,
        resample=Image.Resampling.LANCZOS,
    )
    restored_pixels = np.asarray(restored, dtype=np.uint8)
    baseline = np.asarray(canvas.baseline_rgb, dtype=np.uint8)
    candidate_region = (
        np.asarray(canvas.candidate_region, dtype=np.uint8) == 255
    )
    visible_mask = np.asarray(canvas.visible_mask, dtype=np.uint8) == 255
    final = restored_pixels.copy()
    if preserve_outside_candidate_region:
        final[~candidate_region] = baseline[~candidate_region]

    source = np.asarray(source_rgba, dtype=np.uint8)
    source_visible = source[:, :, 3] == 255
    offset_x, offset_y = canvas.source_offset_xy
    source_height, source_width = source_visible.shape
    source_rectangle = np.s_[
        offset_y : offset_y + source_height,
        offset_x : offset_x + source_width,
    ]
    final_region = final[source_rectangle]
    final_region[source_visible] = source[:, :, :3][source_visible]
    candidate = Image.fromarray(final)

    checks["dimensions_match_completion_canvas"] = candidate.size == canvas.canvas_size
    checks["source_visible_pixels_exact"] = bool(
        np.array_equal(
            np.asarray(candidate)[visible_mask],
            baseline[visible_mask],
        )
    )
    if preserve_outside_candidate_region:
        checks["outside_completion_mask_exact"] = bool(
            np.array_equal(
                np.asarray(candidate)[~candidate_region],
                baseline[~candidate_region],
            )
        )
        checks["completion_mask_nonempty"] = bool(candidate_region.any())
        checks["completion_mask_not_full_canvas"] = not bool(
            candidate_region.all()
        )
        checks["completion_mask_connected_to_visible_entity"] = bool(
            np.logical_and(
                candidate_region,
                _dilate_one_pixel(visible_mask) & ~visible_mask,
            ).any()
        )
        model_baseline = np.asarray(
            canvas.baseline_rgb.resize(
                transform.model_size,
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        model_region = (
            np.asarray(
                canvas.candidate_region.resize(
                    transform.model_size,
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            == 255
        )
        checks["candidate_changed_inside_completion_mask"] = bool(
            np.any(
                np.asarray(generated_model_rgb, dtype=np.uint8)[model_region]
                != model_baseline[model_region]
            )
            and np.any(final[candidate_region] != baseline[candidate_region])
        )
        region_pixels = final[candidate_region]
        checks["background_not_constant"] = bool(
            region_pixels.size
            and np.unique(region_pixels.reshape(-1, 3), axis=0).shape[0] > 1
        )
    else:
        model_baseline = np.asarray(
            canvas.baseline_rgb.resize(
                transform.model_size,
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )
        model_region = (
            np.asarray(
                canvas.candidate_region.resize(
                    transform.model_size,
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )
            == 255
        )
        checks["candidate_changed_outside_visible_region"] = bool(
            candidate_region.any()
            and np.any(
                np.asarray(generated_model_rgb, dtype=np.uint8)[model_region]
                != model_baseline[model_region]
            )
            and np.any(final[candidate_region] != baseline[candidate_region])
        )
        generated_pixels = final[candidate_region]
        checks["generated_region_not_constant"] = bool(
            generated_pixels.size
            and np.unique(generated_pixels.reshape(-1, 3), axis=0).shape[0] > 1
        )
    checks["no_invalid_pixels"] = final.dtype == np.uint8 and bool(
        np.isfinite(final).all()
    )
    checks["source_file_unchanged"] = _sha256_path(source_path) == source_sha256
    reason_codes = {
        "dimensions_match_completion_canvas": "candidate_canvas_size_mismatch",
        "source_visible_pixels_exact": "source_visible_pixels_changed",
        "outside_completion_mask_exact": "pixels_changed_outside_completion_mask",
        "completion_mask_nonempty": "completion_mask_empty",
        "completion_mask_not_full_canvas": "completion_mask_full_canvas",
        "completion_mask_connected_to_visible_entity": (
            "completion_mask_disconnected_from_visible_entity"
        ),
        "candidate_changed_inside_completion_mask": (
            "candidate_unchanged_inside_completion_mask"
        ),
        "background_not_constant": "candidate_completion_is_constant",
        "candidate_changed_outside_visible_region": (
            "candidate_unchanged_outside_visible_region"
        ),
        "generated_region_not_constant": "generated_region_is_constant",
        "no_invalid_pixels": "candidate_has_invalid_pixels",
        "source_file_unchanged": "source_file_changed",
    }
    reasons.extend(
        reason_codes[name]
        for name, passed in checks.items()
        if not passed and name in reason_codes
    )
    return candidate, HardCheckReport(
        passed=all(checks.values()),
        checks=checks,
        reasons=tuple(reasons),
    )


def _synthetic_reject_review(reason: str) -> ReferenceCompletionReview:
    return ReferenceCompletionReview(
        verdict="reject",
        visible_source_preserved=False,
        same_entity_continued=False,
        identity_preserved=False,
        exactly_one_entity=False,
        completion_plausible=False,
        completion_useful=False,
        no_occluder_reconstructed=False,
        no_new_salient_entity=False,
        boundary_clean=False,
        reference_usable=False,
        reason=reason,
    )


def _validated_candidate_order(
    strategies: tuple[CompletionStrategy, ...],
    seeds: tuple[int, ...],
) -> tuple[tuple[CompletionStrategy, int], ...]:
    if not strategies or len(set(strategies)) != len(strategies):
        raise ValueError("completion strategies must be non-empty and unique")
    if any(strategy not in DEFAULT_STRATEGIES for strategy in strategies):
        raise ValueError("unsupported completion strategy")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("completion seeds must be non-empty and unique")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("completion seeds must be non-negative integers")
    return tuple((strategy, seed) for strategy in strategies for seed in seeds)


def load_completion_manifest(path: Path) -> list[CompletionManifestRecord]:
    resolved = _resolve_input_path(path, field_name="manifest_path")
    records: list[CompletionManifestRecord] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                record = CompletionManifestRecord.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"invalid completion manifest line {line_number}: {exc}"
                ) from exc
            if record.sample_id in seen_ids:
                raise ValueError(f"duplicate completion sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError("completion manifest contains no records")
    return records


def _build_record_canvas(
    record: CompletionManifestRecord,
    *,
    source_rgba: Image.Image,
    context_rgb: Image.Image | None,
    completion_mask_path: Path | None,
    config: CompletionCanvasRuntimeConfig,
    policy: CompletionRunPolicy,
) -> tuple[CompletionCanvas, str | None]:
    if policy.candidate_region_kind == "editable_region":
        return (
            build_editable_completion_canvas(
                source_rgba=source_rgba,
                context_rgb=context_rgb,
                completion_sides=record.completion_sides,
                config=config,
            ),
            None,
        )
    canvas_base = _build_completion_canvas_base(
        source_rgba=source_rgba,
        context_rgb=context_rgb,
        completion_sides=record.completion_sides,
        config=config,
    )
    completion_mask_sha256: str | None = None
    if completion_mask_path is None:
        completion_mask = _build_directional_completion_mask(
            canvas_base=canvas_base,
            completion_sides=record.completion_sides,
            completion_start_ratio=record.completion_start_ratio,
            config=config,
        )
    else:
        completion_mask, completion_mask_sha256 = _load_explicit_completion_mask(
            completion_mask_path,
            canvas_base=canvas_base,
        )
    return (
        _finish_completion_canvas(
            canvas_base=canvas_base,
            completion_mask=completion_mask,
        ),
        completion_mask_sha256,
    )


def _preflight_record(
    record: CompletionManifestRecord,
    *,
    config: CompletionCanvasRuntimeConfig,
    policy: CompletionRunPolicy,
) -> _CompletionPreflight:
    source_path = _resolve_input_path(
        record.source_rgba_path,
        field_name="source_rgba_path",
    )
    source, _, _ = _load_source_rgba(source_path)
    context_path: Path | None = None
    context_rgb: Image.Image | None = None
    if record.context_rgb_path is not None:
        context_path = _resolve_input_path(
            record.context_rgb_path,
            field_name="context_rgb_path",
        )
        context_rgb, _, _ = _load_context_rgb(
            context_path,
            expected_size=source.size,
        )
    completion_mask_path: Path | None = None
    if (
        policy.candidate_region_kind == "completion_mask"
        and record.completion_mask_path is not None
    ):
        completion_mask_path = _resolve_input_path(
            record.completion_mask_path,
            field_name="completion_mask_path",
        )
    _, completion_mask_sha256 = _build_record_canvas(
        record,
        source_rgba=source,
        context_rgb=context_rgb,
        completion_mask_path=completion_mask_path,
        config=config,
        policy=policy,
    )
    return _CompletionPreflight(
        source_path=source_path,
        context_path=context_path,
        completion_mask_path=completion_mask_path,
        completion_mask_source_sha256=completion_mask_sha256,
    )


def _process_record(
    record: CompletionManifestRecord,
    *,
    preflight: _CompletionPreflight,
    benchmark_root: Path,
    config: CompletionCanvasRuntimeConfig,
    judge: ReferenceCompletionJudge,
    candidate_factory: Callable[
        [CompletionManifestRecord],
        tuple[CompletionCandidateSpec, ...],
    ],
    generate_candidate: CompletionCandidateGenerator,
    result_metadata: dict[str, object],
    policy: CompletionRunPolicy,
) -> dict[str, object]:
    source_path = preflight.source_path
    context_path = preflight.context_path
    completion_mask_path = preflight.completion_mask_path
    source_rgba, source_bytes, source_sha256 = _load_source_rgba(source_path)
    context_rgb: Image.Image | None = None
    context_sha256: str | None = None
    if context_path is not None:
        context_rgb, _, context_sha256 = _load_context_rgb(
            context_path,
            expected_size=source_rgba.size,
        )
    canvas, completion_mask_sha256 = _build_record_canvas(
        record,
        source_rgba=source_rgba,
        context_rgb=context_rgb,
        completion_mask_path=completion_mask_path,
        config=config,
        policy=policy,
    )
    if completion_mask_sha256 != preflight.completion_mask_source_sha256:
        raise RuntimeError("explicit completion mask changed after preflight")
    transform = build_model_space_transform(
        canvas.canvas_size,
        model_min_side=config.model_min_side,
        model_multiple=config.model_multiple,
    )
    model_rgb, model_region = resize_completion_inputs(canvas, transform)
    final_directory = benchmark_root / record.sample_id
    temporary = benchmark_root / f".{record.sample_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    try:
        (temporary / "source_rgba.png").write_bytes(source_bytes)
        _save_png(temporary / "source_white.png", _white_source(source_rgba))
        if context_rgb is not None:
            _save_png(temporary / "context_rgb.png", context_rgb)
        _save_png(temporary / "baseline_canvas.png", canvas.baseline_rgb)
        _save_png(temporary / "visible_mask.png", canvas.visible_mask)
        _save_png(temporary / policy.region_artifact_name, canvas.candidate_region)
        if completion_mask_path is not None:
            with Image.open(temporary / "completion_mask.png") as published_mask:
                published_mask.load()
                if not np.array_equal(
                    np.asarray(published_mask, dtype=np.uint8),
                    np.asarray(canvas.candidate_region, dtype=np.uint8),
                ):
                    raise RuntimeError(
                        "published completion mask differs from explicit source"
                    )
        candidate_specs = candidate_factory(record)
        if not candidate_specs:
            raise ValueError("completion candidate plan must be non-empty")
        attempts: list[dict[str, object]] = []
        accepted: dict[str, object] | None = None
        for candidate_spec in candidate_specs:
            attempt_started = time.perf_counter()
            stem = candidate_spec.file_stem
            candidate_name = f"candidate_{stem}.png"
            review_name = f"review_{stem}.json"
            candidate_sha256: str | None = None
            judge_status = "not_run"
            reason = ""
            try:
                generated = generate_candidate(
                    record=record,
                    candidate=candidate_spec,
                    input_rgb=model_rgb.copy(),
                    candidate_region=model_region.copy(),
                )
                candidate, hard_check = _hard_check_candidate(
                    generated_model_rgb=generated,
                    source_rgba=source_rgba,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    canvas=canvas,
                    transform=transform,
                    preserve_outside_candidate_region=(
                        policy.preserve_outside_candidate_region
                    ),
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                candidate = None
                hard_check = HardCheckReport(
                    passed=False,
                    checks={},
                    reasons=(f"backend_failed: {exc}",),
                )
            if candidate is not None:
                candidate_path = temporary / candidate_name
                _save_png(candidate_path, candidate)
                candidate_sha256 = _sha256_path(candidate_path)

            if not hard_check.passed:
                reason = "; ".join(hard_check.reasons)
                review = _synthetic_reject_review("hard_check_failed: " + reason)
            else:
                assert candidate is not None
                try:
                    review = judge.review(
                        source_rgba=source_rgba.copy(),
                        completion_canvas=canvas.baseline_rgb.copy(),
                        completion_mask=canvas.candidate_region.copy(),
                        candidate_rgb=candidate.copy(),
                        entity_phrase=record.entity_phrase,
                        reference_type=record.reference_type,
                        completion_region_label=policy.judge_region_label,
                        system_prompt_addendum=policy.judge_system_prompt_addendum,
                    )
                    if not isinstance(review, ReferenceCompletionReview):
                        raise TypeError("judge must return ReferenceCompletionReview")
                    judge_status = "reviewed"
                    reason = review.reason
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    judge_status = "failed_closed"
                    reason = f"judge_failed: {exc}"
                    review = _synthetic_reject_review(reason)
            write_json_atomic(
                temporary / review_name,
                review.model_dump(mode="json"),
            )
            attempts.append(
                {
                    **candidate_spec.attempt_metadata,
                    "candidate_path": (
                        candidate_name if candidate is not None else None
                    ),
                    "candidate_sha256": candidate_sha256,
                    "hard_check": hard_check.to_dict(),
                    "judge_status": judge_status,
                    "judge_verdict": review.verdict,
                    "review_path": review_name,
                    "runtime_seconds": time.perf_counter() - attempt_started,
                    "reason": reason,
                }
            )
            if accepted is None and hard_check.passed and review.verdict == "accept":
                accepted = {
                    **candidate_spec.selection_metadata,
                    "candidate_path": candidate_name,
                    "candidate_sha256": candidate_sha256,
                }

        if _sha256_path(source_path) != source_sha256:
            raise RuntimeError("source reference changed during completion benchmark")
        if context_path is not None and (
            context_sha256 is None or _sha256_path(context_path) != context_sha256
        ):
            raise RuntimeError("context image changed during completion benchmark")
        if completion_mask_path is not None and (
            completion_mask_sha256 is None
            or _sha256_path(completion_mask_path) != completion_mask_sha256
        ):
            raise RuntimeError(
                "explicit completion mask changed during completion benchmark"
            )
        status = "accepted" if accepted is not None else "rejected"
        result: dict[str, object] = {
            "sample_id": record.sample_id,
            "clip_uid": record.clip_uid,
            "entity_id": record.entity_id,
            "reference_type": record.reference_type,
            "entity_phrase": record.entity_phrase,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "context_path": str(context_path) if context_path is not None else None,
            "context_sha256": context_sha256,
            "source_size": list(source_rgba.size),
            "canvas_size": list(canvas.canvas_size),
            "source_offset_xy": list(canvas.source_offset_xy),
            "completion_sides": list(record.completion_sides),
            "completion_start_ratio": record.completion_start_ratio,
            "completion_mask_mode": (
                "explicit" if completion_mask_path is not None else "directional"
            ),
            "completion_mask_source_path": (
                str(completion_mask_path) if completion_mask_path is not None else None
            ),
            "completion_mask_source_sha256": completion_mask_sha256,
            "completion_mask_area_ratio": canvas.candidate_region_area_ratio,
            "model_space_transform": transform.to_dict(),
            **result_metadata,
            "status": status,
            "accepted_candidate": accepted,
            "attempts": attempts,
            "runtime_seconds": time.perf_counter() - started,
            "reason": (
                "earliest accepted candidate selected"
                if accepted is not None
                else "all completion candidates were rejected"
            ),
        }
        write_json_atomic(temporary / "result.json", result)
        temporary.replace(final_directory)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_completion_benchmark(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    allowed_benchmark_root: Path | None,
    config: CompletionCanvasRuntimeConfig,
    judge: ReferenceCompletionJudge,
    candidate_factory: Callable[
        [CompletionManifestRecord],
        tuple[CompletionCandidateSpec, ...],
    ],
    generate_candidate: CompletionCandidateGenerator,
    record_validator: Callable[[CompletionManifestRecord], None] | None = None,
    result_metadata: dict[str, object] | None = None,
    policy: CompletionRunPolicy | None = None,
) -> CompletionBenchmarkRun:
    config.validate()
    selected_policy = policy or CompletionRunPolicy()
    selected_policy.validate()
    resolved_root = _resolve_benchmark_root(
        benchmark_root,
        allowed_benchmark_root=allowed_benchmark_root,
    )
    records = load_completion_manifest(manifest_path)
    if record_validator is not None:
        for record in records:
            record_validator(record)
    preflights = [
        _preflight_record(record, config=config, policy=selected_policy)
        for record in records
    ]
    resolved_root.mkdir(parents=True, exist_ok=False)
    accepted = rejected = 0
    results: list[dict[str, object]] = []
    for record, preflight in zip(records, preflights):
        result = _process_record(
            record,
            preflight=preflight,
            benchmark_root=resolved_root,
            config=config,
            judge=judge,
            candidate_factory=candidate_factory,
            generate_candidate=generate_candidate,
            result_metadata=dict(result_metadata or {}),
            policy=selected_policy,
        )
        results.append(result)
        accepted += int(result["status"] == "accepted")
        rejected += int(result["status"] == "rejected")
    return CompletionBenchmarkRun(
        stats=CompletionBenchmarkStats(
            processed=len(records),
            accepted=accepted,
            rejected=rejected,
        ),
        results=tuple(results),
    )


def run_reference_completion_benchmark(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    config: PowerPaintV21CompletionConfig,
    backend: ReferenceCompletionBackend,
    judge: ReferenceCompletionJudge,
    strategies: tuple[CompletionStrategy, ...] = DEFAULT_STRATEGIES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> CompletionBenchmarkStats:
    candidate_order = _validated_candidate_order(strategies, seeds)

    def candidate_factory(
        record: CompletionManifestRecord,
    ) -> tuple[CompletionCandidateSpec, ...]:
        prompt = build_completion_prompt(
            entity_phrase=record.entity_phrase,
            reference_type=record.reference_type,
        )
        negative_prompt = build_completion_negative_prompt(record.reference_type)
        return tuple(
            CompletionCandidateSpec(
                file_stem=f"{strategy}_seed_{seed}",
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                attempt_metadata={
                    "strategy": strategy,
                    "seed": seed,
                    "fitting_degree": config.fitting_degree,
                },
                selection_metadata={
                    "strategy": strategy,
                    "seed": seed,
                    "fitting_degree": config.fitting_degree,
                },
            )
            for strategy, seed in candidate_order
        )

    def generate_candidate(
        *,
        record: CompletionManifestRecord,
        candidate: CompletionCandidateSpec,
        input_rgb: Image.Image,
        candidate_region: Image.Image,
    ) -> Image.Image:
        strategy = candidate.attempt_metadata["strategy"]
        if strategy not in DEFAULT_STRATEGIES:
            raise ValueError("invalid PowerPaint completion strategy")
        return backend.complete(
            input_rgb=input_rgb,
            completion_mask=candidate_region,
            entity_phrase=record.entity_phrase,
            reference_type=record.reference_type,
            strategy=strategy,
            seed=candidate.seed,
            fitting_degree=config.fitting_degree,
            prompt=candidate.prompt,
            negative_prompt=candidate.negative_prompt,
        )

    run = run_completion_benchmark(
        manifest_path=manifest_path,
        benchmark_root=benchmark_root,
        allowed_benchmark_root=None,
        config=config,
        judge=judge,
        candidate_factory=candidate_factory,
        generate_candidate=generate_candidate,
        policy=CompletionRunPolicy(
            candidate_region_kind="completion_mask",
            preserve_outside_candidate_region=True,
            region_artifact_name="completion_mask.png",
            judge_region_label="completion mask",
        ),
    )
    return run.stats
