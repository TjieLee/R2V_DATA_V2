"""Isolated Boogu reference editing and fail-closed artifact publication.

Boogu produces a new, full-resolution reference image. This module therefore
never composites source pixels or masks into a generated candidate. SAM review
is an optional quality signal only.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import selectors
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

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
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.profiling import (
    get_model_profile_context,
    model_profile_context,
    profile_model_call,
    profiled_openai_call,
)
from r2v_data_v2.v3.reference_geometry import (
    composition_geometry_metadata,
    content_geometry_from_mask,
    content_geometry_from_rgba,
    source_geometry_metadata,
    tiny_content_reason,
)
from r2v_data_v2.v3.sam3_backend import SegmentationBackend

BooguEditOperation = Literal["complete_entity", "add_entity_background"]
ReferenceType = Literal["subject", "object", "group"]
EditStatus = Literal["accepted", "rejected"]
SamFailureKind = Literal[
    "none",
    "not_found",
    "multiple_instances",
    "excessive_area_growth",
    "fragmented",
    "backend_failure",
]

BOOGU_MODEL_NAME = "Boogu-Image-0.1-Edit-Turbo"
BOOGU_MODEL_REVISION = "hotfix-1k-20260708"
DEFAULT_BOOGU_CODE_ROOT = Path("/mnt/workspace/litengjie/data/vendor/Boogu-Image")
DEFAULT_BOOGU_PYTHON = Path(
    "/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python"
)
DEFAULT_BOOGU_MODEL_PATH = Path(
    "/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708"
)
DEFAULT_ALLOWED_SERVER_ROOT = Path("/mnt/workspace/litengjie/data")
DEFAULT_TARGET_AREA = 1024 * 1024
DEFAULT_ALIGNMENT = 16
DEFAULT_MIN_SOURCE_CONTENT_AREA_PIXELS = 128 * 128
DEFAULT_MIN_SOURCE_CONTENT_LONG_SIDE_PIXELS = 128
DEFAULT_MIN_CANDIDATE_SCALE_RATIO = 0.60
DEFAULT_MAX_CANDIDATE_CENTER_SHIFT = 0.20

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _TinySourceRejected(Exception):
    pass


def resolve_boogu_1k_size(
    source_width: int,
    source_height: int,
    *,
    target_area: int = DEFAULT_TARGET_AREA,
    alignment: int = DEFAULT_ALIGNMENT,
) -> tuple[int, int]:
    """Resolve an aligned, aspect-preserving canvas close to ``target_area``."""

    for name, value in (
        ("source_width", source_width),
        ("source_height", source_height),
        ("target_area", target_area),
        ("alignment", alignment),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    aspect_ratio = source_width / source_height
    raw_width = math.sqrt(target_area * aspect_ratio)
    raw_height = math.sqrt(target_area / aspect_ratio)
    if not all(math.isfinite(value) and value > 0 for value in (raw_width, raw_height)):
        raise ValueError("resolved Boogu dimensions must be finite and positive")

    width_units = max(1, round(raw_width / alignment))
    height_units = max(1, round(raw_height / alignment))
    candidates: list[tuple[tuple[float, float, int, int], int, int]] = []
    for width_delta in range(-3, 4):
        for height_delta in range(-3, 4):
            candidate_width = (width_units + width_delta) * alignment
            candidate_height = (height_units + height_delta) * alignment
            if candidate_width <= 0 or candidate_height <= 0:
                continue
            output_ratio = candidate_width / candidate_height
            ratio_error = abs(output_ratio - aspect_ratio) / aspect_ratio
            area_error = (
                abs(candidate_width * candidate_height - target_area) / target_area
            )
            score = (
                ratio_error + area_error,
                ratio_error,
                abs(candidate_width * candidate_height - target_area),
                candidate_width,
            )
            candidates.append((score, candidate_width, candidate_height))
    if not candidates:
        raise ValueError("could not resolve aligned Boogu output dimensions")
    _, width, height = min(candidates, key=lambda item: item[0])
    return width, height


class BooguCompletionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    same_physical_entity: StrictBool
    identity_preserved: StrictBool
    original_visible_attributes_preserved: StrictBool
    exactly_one_entity: StrictBool
    missing_parts_plausibly_completed: StrictBool
    no_duplicate_entity: StrictBool
    no_unrelated_entity: StrictBool
    no_severe_structure_artifact: StrictBool
    style_coherent: StrictBool
    resolution_usable: StrictBool
    reference_usable: StrictBool
    certain: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty(value, "reason")

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> BooguCompletionReview:
        flags = (
            self.same_physical_entity,
            self.identity_preserved,
            self.original_visible_attributes_preserved,
            self.exactly_one_entity,
            self.missing_parts_plausibly_completed,
            self.no_duplicate_entity,
            self.no_unrelated_entity,
            self.no_severe_structure_artifact,
            self.style_coherent,
            self.resolution_usable,
            self.reference_usable,
            self.certain,
        )
        _require_matching_verdict(self.verdict, flags)
        return self


class BooguBackgroundReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["accept", "reject"]
    exactly_one_target_entity: StrictBool
    identity_preserved: StrictBool
    entity_appearance_consistent: StrictBool
    no_duplicate_entity: StrictBool
    no_added_salient_entity: StrictBool
    no_unintended_completion_or_extension: StrictBool
    subject_scale_preserved: StrictBool
    subject_layout_preserved: StrictBool
    background_coherent: StrictBool
    background_improves_reference: StrictBool
    prefer_candidate_over_source: StrictBool
    background_style_consistent: StrictBool
    no_halo_or_seam: StrictBool
    subject_not_severely_redrawn: StrictBool
    reference_usable: StrictBool
    certain: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty(value, "reason")

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> BooguBackgroundReview:
        flags = (
            self.exactly_one_target_entity,
            self.identity_preserved,
            self.entity_appearance_consistent,
            self.no_duplicate_entity,
            self.no_added_salient_entity,
            self.no_unintended_completion_or_extension,
            self.subject_scale_preserved,
            self.subject_layout_preserved,
            self.background_coherent,
            self.background_improves_reference,
            self.prefer_candidate_over_source,
            self.background_style_consistent,
            self.no_halo_or_seam,
            self.subject_not_severely_redrawn,
            self.reference_usable,
            self.certain,
        )
        _require_matching_verdict(self.verdict, flags)
        return self


class BooguSamReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: StrictBool
    target_entity_present: StrictBool
    exactly_one_target_instance: StrictBool
    area_growth_acceptable: StrictBool
    fragmentation_acceptable: StrictBool
    reason: str = Field(min_length=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return _nonempty(value, "reason")

    @model_validator(mode="after")
    def _passed_matches_flags(self) -> BooguSamReview:
        expected = all(
            (
                self.target_entity_present,
                self.exactly_one_target_instance,
                self.area_growth_acceptable,
                self.fragmentation_acceptable,
            )
        )
        if self.passed != expected:
            raise ValueError("SAM review passed must match every quality flag")
        failure_kind = self.diagnostics.get("failure_kind")
        allowed_failure_kinds = {
            "none",
            "not_found",
            "multiple_instances",
            "excessive_area_growth",
            "fragmented",
            "backend_failure",
        }
        if failure_kind not in allowed_failure_kinds:
            raise ValueError("SAM review diagnostics require a valid failure_kind")
        if self.passed != (failure_kind == "none"):
            raise ValueError("SAM review failure_kind must match passed")
        return self


BooguQwenReview = BooguCompletionReview | BooguBackgroundReview


@dataclass(frozen=True)
class BooguEditOutput:
    png_bytes: bytes
    original_instruction: str
    rewritten_instruction: str | None
    effective_instruction: str
    worker_metadata: dict[str, Any] = field(default_factory=dict)


class BooguReferenceEditBackend(Protocol):
    def edit(
        self,
        *,
        source_rgb: Image.Image,
        instruction: str,
        width: int,
        height: int,
        thinking_enabled: bool,
    ) -> BooguEditOutput: ...


class BooguReferenceEditJudge(Protocol):
    def review(
        self,
        *,
        operation: BooguEditOperation,
        source_rgba: Image.Image,
        source_input_rgb: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BooguQwenReview: ...


class BooguSamReviewer(Protocol):
    def review(
        self,
        *,
        operation: BooguEditOperation,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BooguSamReview: ...


_COMPLETION_REVIEW_PROMPT = """You review a generated entity reference.
Image 1 is the source reference and Image 2 is the generated completion.
Accept only if Image 2 preserves the same physical entity and every visible
identity attribute, contains exactly one coherent target, plausibly completes
missing structure, adds no duplicate or unrelated entity, has no severe
structural artifact, and is a usable high-resolution reference. Judge visible
facts only and return one strict JSON object matching the supplied schema."""

_BACKGROUND_REVIEW_PROMPT = """You review a generated entity reference.
Image 1 is the source reference and Image 2 adds a clean supporting background.
Accept only if Image 2 contains exactly the same single target entity, preserves
its identity, appearance, scale, and layout, introduces no duplicate or salient
entity, and does not extend or complete the target unexpectedly. Reject if the
subject becomes smaller, moves toward a corner, or leaves large meaningless
empty space. The background must be coherent, improve the reference's
naturalness, clarity, or usefulness, and make Image 2 clearly preferable to
Image 1. A technically clean but implausible or unhelpful background must be
rejected. Judge visible facts only and return one strict JSON object matching
the supplied schema."""


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenBooguReferenceEditJudge:
    """Production structured Qwen reviewer for native Boogu candidates."""

    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        client: Any | None = None,
        repair_retries: int = 1,
    ) -> None:
        if repair_retries < 0:
            raise ValueError("repair_retries must be non-negative")
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self.repair_retries = repair_retries

    def review(
        self,
        *,
        operation: BooguEditOperation,
        source_rgba: Image.Image,
        source_input_rgb: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BooguQwenReview:
        model = (
            BooguCompletionReview
            if operation == "complete_entity"
            else BooguBackgroundReview
        )
        system_prompt = (
            _COMPLETION_REVIEW_PROMPT
            if operation == "complete_entity"
            else _BACKGROUND_REVIEW_PROMPT
        )
        source = (
            source_input_rgb
            if operation == "complete_entity"
            else source_rgba.convert("RGB")
        )
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Reference type: {reference_type}\n"
                    f"Entity phrase: {entity_phrase.strip()}"
                ),
            }
        ]
        for label, image in (
            ("Image 1: source reference", source),
            ("Image 2: generated candidate", candidate_rgb),
        ):
            content.extend(
                [
                    {"type": "text", "text": label},
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_url(image.convert("RGB"))},
                    },
                ]
            )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        with model_profile_context(
            retry_index=0,
            metadata={"edit_operation": operation},
        ):
            raw = self._request(messages, model)
        for attempt in range(self.repair_retries + 1):
            try:
                return model.model_validate_json(raw)
            except (TypeError, ValueError) as exc:
                if attempt >= self.repair_retries:
                    raise ValueError(f"invalid Qwen Boogu review: {exc}") from exc
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Repair the JSON to match the schema exactly. "
                            f"Validation error: {exc}"
                        ),
                    },
                ]
                with model_profile_context(
                    retry_index=attempt + 1,
                    metadata={"edit_operation": operation},
                ):
                    raw = self._request(repair_messages, model)
        raise AssertionError("unreachable")

    def _request(
        self,
        messages: list[dict[str, object]],
        model: type[BooguCompletionReview | BooguBackgroundReview],
    ) -> str:
        profile_context = get_model_profile_context()
        retry_index = profile_context.retry_index
        operation = str(profile_context.metadata.get("edit_operation", "unknown"))
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_boogu_reference_edit_review",
                            "strict": True,
                            "schema": model.model_json_schema(),
                        },
                    },
                ),
                component=(
                    "qwen_boogu_completion_review"
                    if operation == "complete_entity"
                    else "qwen_boogu_background_review"
                ),
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    "edit_operation": operation,
                    "response_format": "json_schema",
                },
            )
        except BadRequestError:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={"type": "json_object"},
                ),
                component=(
                    "qwen_boogu_completion_review"
                    if operation == "complete_entity"
                    else "qwen_boogu_background_review"
                ),
                operation="initial" if retry_index == 0 else "repair",
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={
                    "edit_operation": operation,
                    "response_format": "json_object",
                },
            )
        raw = response.choices[0].message.content
        if not raw:
            raise RuntimeError("Qwen returned an empty Boogu reference review")
        return str(raw)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class Sam3BooguReferenceReviewer:
    """Use SAM3 only as a post-generation identity/geometry quality guard."""

    def __init__(
        self,
        backend: SegmentationBackend,
        *,
        temporary_root: Path,
        max_area_growth_ratio: float,
        max_significant_components: int,
        min_candidate_scale_ratio: float = DEFAULT_MIN_CANDIDATE_SCALE_RATIO,
        max_candidate_center_shift: float = DEFAULT_MAX_CANDIDATE_CENTER_SHIFT,
    ) -> None:
        if not math.isfinite(max_area_growth_ratio) or max_area_growth_ratio < 1:
            raise ValueError("max_area_growth_ratio must be finite and at least 1")
        if max_significant_components < 1:
            raise ValueError("max_significant_components must be positive")
        if (
            not isinstance(min_candidate_scale_ratio, float)
            or not math.isfinite(min_candidate_scale_ratio)
            or not 0 < min_candidate_scale_ratio <= 1
        ):
            raise ValueError(
                "min_candidate_scale_ratio must be a finite float in (0, 1]"
            )
        if (
            not isinstance(max_candidate_center_shift, float)
            or not math.isfinite(max_candidate_center_shift)
            or not 0 <= max_candidate_center_shift <= math.sqrt(2)
        ):
            raise ValueError(
                "max_candidate_center_shift must be a finite float between "
                "0 and sqrt(2)"
            )
        self.backend = backend
        self.temporary_root = temporary_root.expanduser().resolve(strict=False)
        self.max_area_growth_ratio = max_area_growth_ratio
        self.max_significant_components = max_significant_components
        self.min_candidate_scale_ratio = min_candidate_scale_ratio
        self.max_candidate_center_shift = max_candidate_center_shift

    def review(
        self,
        *,
        operation: BooguEditOperation,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> BooguSamReview:
        del operation
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix="sam3-review-",
                dir=self.temporary_root,
            ) as temporary_name:
                directory = Path(temporary_name)
                frame_paths: list[Path] = []
                for slot in range(10):
                    path = directory / f"{slot:02d}.jpg"
                    candidate_rgb.save(path, format="JPEG", quality=95, subsampling=0)
                    frame_paths.append(path)
                result = self.backend.track(
                    frame_paths=frame_paths,
                    entity_id="e1",
                    reference_type=reference_type,
                    grounding_prompt=entity_phrase,
                )
        except Exception as exc:  # noqa: BLE001 - SAM is a fail-closed reviewer
            return _failed_sam_review(
                failure_kind="backend_failure",
                reason=f"sam3_backend_failure:{type(exc).__name__}:{exc}",
                diagnostics={"exception_type": type(exc).__name__},
            )
        if result.status != "ready" or not result.observations:
            failure_kind = _sam_track_failure_kind(
                status=result.status,
                reason=result.reason,
            )
            return _failed_sam_review(
                failure_kind=failure_kind,
                reason=f"sam3_target_not_ready:{result.reason or result.status}",
                diagnostics={"track_status": result.status},
            )
        observation = min(
            result.observations,
            key=lambda item: (abs(item.slot - 5), item.slot),
        )
        mask = np.asarray(observation.mask, dtype=bool)
        if mask.shape != (candidate_rgb.height, candidate_rgb.width):
            return _failed_sam_review(
                failure_kind="backend_failure",
                reason="sam3_candidate_mask_dimensions_do_not_match_image",
                diagnostics={"candidate_mask_shape": list(mask.shape)},
            )
        source_alpha = np.asarray(source_rgba.getchannel("A"), dtype=np.uint8) > 0
        source_ratio = float(source_alpha.mean())
        candidate_ratio = float(mask.mean())
        area_growth = math.inf if source_ratio <= 0 else candidate_ratio / source_ratio
        component_count = _significant_component_count(mask)
        target_present = bool(mask.any())
        exactly_one = (
            target_present
            and len({item.object_id for item in result.observations}) == 1
        )
        area_ok = area_growth <= self.max_area_growth_ratio
        fragmentation_ok = component_count <= self.max_significant_components
        passed = all((target_present, exactly_one, area_ok, fragmentation_ok))
        failure_kind = _sam_failure_kind(
            target_present=target_present,
            exactly_one=exactly_one,
            area_ok=area_ok,
            fragmentation_ok=fragmentation_ok,
        )
        geometry = (
            composition_geometry_metadata(
                content_geometry_from_rgba(source_rgba),
                content_geometry_from_mask(mask),
                minimum_scale_ratio=self.min_candidate_scale_ratio,
                maximum_center_shift=self.max_candidate_center_shift,
            )
            if target_present
            else {}
        )
        return BooguSamReview(
            passed=passed,
            target_entity_present=target_present,
            exactly_one_target_instance=exactly_one,
            area_growth_acceptable=area_ok,
            fragmentation_acceptable=fragmentation_ok,
            reason="sam3_review_passed" if passed else "sam3_review_failed",
            diagnostics={
                "source_area_ratio": source_ratio,
                "candidate_area_ratio": candidate_ratio,
                "area_growth_ratio": area_growth,
                "significant_component_count": component_count,
                "review_slot": observation.slot,
                "mask_usage": "review_only",
                "failure_kind": failure_kind,
                **geometry,
            },
        )


def _failed_sam_review(
    *,
    failure_kind: SamFailureKind,
    reason: str,
    diagnostics: dict[str, Any],
) -> BooguSamReview:
    return BooguSamReview(
        passed=False,
        target_entity_present=False,
        exactly_one_target_instance=False,
        area_growth_acceptable=False,
        fragmentation_acceptable=False,
        reason=reason,
        diagnostics={**diagnostics, "failure_kind": failure_kind},
    )


def _sam_failure_kind(
    *,
    target_present: bool,
    exactly_one: bool,
    area_ok: bool,
    fragmentation_ok: bool,
) -> SamFailureKind:
    if not target_present:
        return "not_found"
    if not exactly_one:
        return "multiple_instances"
    if not area_ok:
        return "excessive_area_growth"
    if not fragmentation_ok:
        return "fragmented"
    return "none"


def _sam_track_failure_kind(*, status: str, reason: str | None) -> SamFailureKind:
    if status == "not_found":
        return "not_found"
    normalized_reason = (reason or "").casefold()
    if status == "failed" and "multiple ambiguous instances" in normalized_reason:
        return "multiple_instances"
    return "backend_failure"


_GEOMETRY_METADATA_FIELDS = (
    "source_normalized_bbox",
    "candidate_normalized_bbox",
    "source_normalized_bbox_area",
    "candidate_normalized_bbox_area",
    "candidate_scale_ratio",
    "source_center_xy",
    "candidate_center_xy",
    "center_shift",
    "geometry_gate_passed",
    "geometry_rejection_reason",
)


def _sam_geometry_metadata(review: BooguSamReview | None) -> dict[str, object]:
    if review is None:
        return {}
    diagnostics = review.diagnostics
    present = {
        field: diagnostics[field]
        for field in _GEOMETRY_METADATA_FIELDS
        if field in diagnostics
    }
    if not present:
        return {}
    if set(present) != set(_GEOMETRY_METADATA_FIELDS):
        raise ValueError("SAM geometry diagnostics must be complete when present")
    if not isinstance(present["geometry_gate_passed"], bool):
        raise TypeError("SAM geometry_gate_passed must be a boolean")
    reason = present["geometry_rejection_reason"]
    if reason not in {None, "entity_scale_collapsed", "entity_shifted_off_layout"}:
        raise ValueError("SAM geometry rejection reason is invalid")
    if present["geometry_gate_passed"] != (reason is None):
        raise ValueError("SAM geometry gate result does not match its reason")
    return present


def _background_qwen_skip_reason(
    review: BooguSamReview | None,
    geometry_metadata: dict[str, object],
) -> str | None:
    if review is not None:
        failure_kind = review.diagnostics["failure_kind"]
        if not review.passed and failure_kind != "not_found":
            return f"sam_hard_failure:{failure_kind}"
    if geometry_metadata and not geometry_metadata["geometry_gate_passed"]:
        return (
            "geometry_hard_failure:"
            f"{geometry_metadata['geometry_rejection_reason']}"
        )
    return None


def _background_qwen_rejection_reason(review: BooguBackgroundReview) -> str:
    if not review.subject_scale_preserved:
        return "entity_scale_collapsed"
    if not review.subject_layout_preserved:
        return "entity_shifted_off_layout"
    if not review.background_improves_reference or not review.prefer_candidate_over_source:
        return "background_not_beneficial"
    return review.reason


def _significant_component_count(mask: np.ndarray) -> int:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("SAM review mask must be two-dimensional")
    visited = np.zeros(binary.shape, dtype=bool)
    minimum_area = max(16, int(binary.sum() * 0.02))
    count = 0
    height, width = binary.shape
    rows, columns = np.nonzero(binary)
    for row, column in zip(rows, columns):
        if visited[row, column]:
            continue
        stack = [(int(row), int(column))]
        visited[row, column] = True
        area = 0
        while stack:
            current_row, current_column = stack.pop()
            area += 1
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and binary[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        count += int(area >= minimum_area)
    return count


@dataclass(frozen=True)
class BooguWorkerConfig:
    python_executable: Path = DEFAULT_BOOGU_PYTHON
    code_root: Path = DEFAULT_BOOGU_CODE_ROOT
    model_path: Path = DEFAULT_BOOGU_MODEL_PATH
    model_revision: str = BOOGU_MODEL_REVISION
    device: str = "cuda:0"
    seed: int = 0
    timeout_seconds: int = 3600
    cuda_visible_devices: str = "0"
    allowed_server_root: Path = DEFAULT_ALLOWED_SERVER_ROOT
    temporary_root: Path | None = None
    worker_script: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "run_v3_boogu_reference_edit_worker.py"
        )
    )

    def validate(self) -> None:
        allowed_root = self.allowed_server_root.expanduser().resolve(strict=False)
        for name, path in (
            ("python_executable", self.python_executable),
            ("code_root", self.code_root),
            ("model_path", self.model_path),
        ):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute pathlib.Path")
            resolved = path.expanduser().resolve(strict=False)
            if resolved != allowed_root and allowed_root not in resolved.parents:
                raise ValueError(f"{name} must remain inside allowed_server_root")
        if (
            not isinstance(self.worker_script, Path)
            or not self.worker_script.is_absolute()
        ):
            raise ValueError("worker_script must be an absolute pathlib.Path")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not self.cuda_visible_devices.strip():
            raise ValueError("cuda_visible_devices must be non-empty")
        if self.temporary_root is not None:
            temporary_root = self.temporary_root.expanduser().resolve(strict=False)
            if (
                temporary_root != allowed_root
                and allowed_root not in temporary_root.parents
            ):
                raise ValueError(
                    "temporary_root must remain inside allowed_server_root"
                )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds < 1
        ):
            raise ValueError("timeout_seconds must be a positive integer")


class BooguSubprocessBackend:
    """Reuse one fail-closed JSONL worker for every edit in a stage."""

    def __init__(self, config: BooguWorkerConfig) -> None:
        config.validate()
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr_handle: TextIO | None = None
        self._stderr_path: Path | None = None

    @property
    def started(self) -> bool:
        return self._process is not None

    def start(self, *, stderr_log_path: Path) -> None:
        if self._process is not None:
            raise RuntimeError("Boogu worker is already started")
        stderr_path = stderr_log_path.expanduser().resolve(strict=False)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = stderr_path.open("a", encoding="utf-8")
        config = self.config
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        command = [
            str(config.python_executable),
            str(config.worker_script),
            "--serve",
            "--code-root",
            str(config.code_root),
            "--model-path",
            str(config.model_path),
            "--model-name",
            BOOGU_MODEL_NAME,
            "--model-revision",
            config.model_revision,
            "--device",
            config.device,
            "--seed",
            str(config.seed),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=config.code_root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                text=True,
                bufsize=1,
            )
        except Exception:
            stderr_handle.close()
            raise
        self._process = process
        self._stderr_handle = stderr_handle
        self._stderr_path = stderr_path
        try:
            response = self._read_response()
            if response != {
                "schema_version": 1,
                "type": "ready",
                "status": "ok",
            }:
                raise RuntimeError(f"invalid Boogu worker startup response: {response}")
        except Exception:
            self._terminate()
            raise

    def edit(
        self,
        *,
        source_rgb: Image.Image,
        instruction: str,
        width: int,
        height: int,
        thinking_enabled: bool,
    ) -> BooguEditOutput:
        if source_rgb.mode != "RGB":
            raise ValueError("Boogu source image must be RGB")
        instruction = _nonempty(instruction, "instruction")
        _validate_output_dimensions(width, height)
        if self._process is None:
            raise RuntimeError("Boogu worker must be started before editing")
        config = self.config
        temporary_root = (
            None
            if config.temporary_root is None
            else config.temporary_root.expanduser().resolve(strict=False)
        )
        if temporary_root is not None:
            temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="r2v-boogu-",
            dir=temporary_root,
        ) as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "source_input_rgb.png"
            output_path = temporary / "candidate.png"
            source_rgb.save(input_path, format="PNG")
            request_id = uuid.uuid4().hex
            request = {
                "schema_version": 1,
                "type": "edit",
                "request_id": request_id,
                "input_image_path": str(input_path),
                "output_image_path": str(output_path),
                "instruction": instruction,
                "thinking_enabled": thinking_enabled,
                "width": width,
                "height": height,
            }
            try:
                self._write_request(request)
                result = self._read_response()
            except Exception:
                self._terminate()
                raise
            if result.get("request_id") != request_id:
                self._terminate()
                raise RuntimeError("Boogu worker response request_id mismatch")
            if result.get("type") != "response":
                self._terminate()
                raise RuntimeError("Boogu worker returned an invalid response type")
            if result.get("status") == "error":
                raise RuntimeError(
                    "Boogu worker rejected request: "
                    f"{result.get('reason', 'generation failed')}"
                )
            if result.get("status") != "ok":
                self._terminate()
                raise RuntimeError("Boogu worker returned an invalid response status")
            if not output_path.is_file():
                raise RuntimeError("Boogu worker did not publish its output image")
            output_bytes = output_path.read_bytes()
            _validated_native_png(output_bytes, expected_size=(width, height))
            if result.get("original_instruction") != instruction:
                raise RuntimeError("Boogu worker changed original instruction metadata")
            rewritten = result.get("rewritten_instruction")
            if rewritten is not None and not isinstance(rewritten, str):
                raise TypeError("rewritten_instruction must be a string or null")
            effective = result.get("effective_instruction")
            if not isinstance(effective, str) or not effective.strip():
                raise ValueError("Boogu worker returned an empty effective instruction")
            returned_size = result.get("returned_size")
            if returned_size != [width, height]:
                raise RuntimeError(
                    "Boogu worker result dimensions do not match output: "
                    f"expected_size={[width, height]}, returned_size={returned_size}"
                )
            return BooguEditOutput(
                png_bytes=output_bytes,
                original_instruction=instruction,
                rewritten_instruction=(
                    rewritten.strip() if isinstance(rewritten, str) else None
                ),
                effective_instruction=effective.strip(),
                worker_metadata={
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "original_instruction",
                        "rewritten_instruction",
                        "effective_instruction",
                    }
                },
            )

    def _write_request(self, request: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Boogu worker stdin is unavailable")
        if process.poll() is not None:
            raise RuntimeError(
                f"Boogu worker exited before request with code {process.returncode}"
            )
        try:
            process.stdin.write(
                json.dumps(request, ensure_ascii=True, sort_keys=True) + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Boogu worker request pipe failed") from exc

    def _read_response(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Boogu worker stdout is unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.config.timeout_seconds):
                raise TimeoutError(
                    f"Boogu worker timed out after {self.config.timeout_seconds}s"
                )
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            returncode = process.poll()
            raise RuntimeError(
                "Boogu worker exited without a response: "
                f"returncode={returncode}, stderr_log={self._stderr_path}"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Boogu worker returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise TypeError("Boogu worker response must be a JSON object")
        return response

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        request_id = uuid.uuid4().hex
        try:
            self._write_request(
                {
                    "schema_version": 1,
                    "type": "shutdown",
                    "request_id": request_id,
                }
            )
            response = self._read_response()
            if response != {
                "schema_version": 1,
                "type": "shutdown",
                "request_id": request_id,
                "status": "ok",
            }:
                raise RuntimeError(
                    f"invalid Boogu worker shutdown response: {response}"
                )
            process.wait(timeout=self.config.timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Boogu worker exited with code {process.returncode}"
                )
        except Exception:
            self._terminate()
            raise
        finally:
            self._close_pipes()

    def _terminate(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._close_pipes()

    def _close_pipes(self) -> None:
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        self._process = None
        self._stderr_handle = None


@dataclass(frozen=True)
class BooguReferenceEditResult:
    status: EditStatus
    operation: BooguEditOperation
    candidate_path: Path | None
    final_reference_path: Path | None
    metadata_path: Path
    rejection_path: Path | None
    fallback_status: str


@dataclass(frozen=True)
class BooguFinalPublication:
    final_reference_path: Path
    final_metadata_path: Path


def publish_boogu_final_reference(
    *,
    run_root: Path,
    clip_uid: str,
    entity_id: str,
    candidate_path: Path,
    selected_metadata_path: Path,
    completion_metadata_path: Path | None = None,
    background_metadata_path: Path | None = None,
    background_fallback: Literal["completion_candidate"] | None = None,
    final_selection: Literal["background_candidate", "completion_candidate"],
    final_selection_reason: str,
) -> BooguFinalPublication:
    """Publish an accepted native candidate without changing its pixels."""

    clip_uid = _safe_component(clip_uid, "clip_uid")
    entity_id = _safe_component(entity_id, "entity_id")
    final_selection_reason = _nonempty(
        final_selection_reason,
        "final_selection_reason",
    )
    root = run_root.expanduser().resolve(strict=False)
    edit_dir = (root / "clips" / clip_uid / "reference_edit" / entity_id).resolve(
        strict=False
    )

    def resolve_evidence(path: Path, name: str) -> Path:
        resolved = path.expanduser().resolve(strict=False)
        if edit_dir not in resolved.parents or not resolved.is_file():
            raise ValueError(f"{name} must be an existing entity edit artifact")
        return resolved

    candidate = resolve_evidence(candidate_path, "candidate_path")
    selected_metadata = resolve_evidence(
        selected_metadata_path,
        "selected_metadata_path",
    )
    completion_metadata = (
        None
        if completion_metadata_path is None
        else resolve_evidence(completion_metadata_path, "completion_metadata_path")
    )
    background_metadata = (
        None
        if background_metadata_path is None
        else resolve_evidence(background_metadata_path, "background_metadata_path")
    )
    if background_fallback is not None and (
        background_fallback != "completion_candidate"
        or completion_metadata is None
        or background_metadata is None
    ):
        raise ValueError(
            "completion-candidate fallback requires completion and background metadata"
        )

    candidate_bytes = candidate.read_bytes()
    candidate_rgb = _validated_native_rgb_png(candidate_bytes)
    candidate_sha256 = _sha256_bytes(candidate_bytes)
    metadata = json.loads(selected_metadata.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "accepted"
        or metadata.get("generated_reference_sha256") != candidate_sha256
    ):
        raise ValueError("selected Boogu metadata does not accept the final candidate")

    final_path = edit_dir / "final_reference_1k.png"
    final_metadata_path = edit_dir / "final_metadata.json"
    _write_bytes_atomic(final_path, candidate_bytes)
    if final_path.read_bytes() != candidate_bytes:
        raise RuntimeError("final reference is not the native Boogu candidate")
    final_metadata = {
        **metadata,
        "final_reference_path": final_path.name,
        "final_reference_sha256": candidate_sha256,
        "final_dimensions": {
            "width": candidate_rgb.width,
            "height": candidate_rgb.height,
        },
        "completion_metadata_path": (
            completion_metadata.name if completion_metadata is not None else None
        ),
        "background_metadata_path": (
            background_metadata.name if background_metadata is not None else None
        ),
        "background_fallback": background_fallback,
        "final_selection": final_selection,
        "final_selection_reason": final_selection_reason,
    }
    write_json_atomic(final_metadata_path, final_metadata)
    return BooguFinalPublication(
        final_reference_path=final_path,
        final_metadata_path=final_metadata_path,
    )


def run_boogu_reference_edit(
    *,
    run_root: Path,
    clip_uid: str,
    entity_id: str,
    operation: BooguEditOperation,
    instruction: str,
    entity_phrase: str,
    grounding_prompt: str | None = None,
    reference_type: ReferenceType,
    backend: BooguReferenceEditBackend,
    judge: BooguReferenceEditJudge,
    sam_reviewer: BooguSamReviewer | None = None,
    target_area: int = DEFAULT_TARGET_AREA,
    alignment: int = DEFAULT_ALIGNMENT,
    min_source_content_area_pixels: int = DEFAULT_MIN_SOURCE_CONTENT_AREA_PIXELS,
    min_source_content_long_side_pixels: int = (
        DEFAULT_MIN_SOURCE_CONTENT_LONG_SIDE_PIXELS
    ),
    model_revision: str = BOOGU_MODEL_REVISION,
    fallback_status: str = "canonical_preserved",
    source_image_path: Path | None = None,
    geometry_source_image_path: Path | None = None,
    publish_final: bool = True,
    overwrite: bool = False,
) -> BooguReferenceEditResult:
    """Generate, review, and publish one native Boogu reference artifact."""

    if operation not in {"complete_entity", "add_entity_background"}:
        raise ValueError(f"unsupported Boogu edit operation: {operation}")
    instruction = _nonempty(instruction, "instruction")
    entity_phrase = _nonempty(entity_phrase, "entity_phrase")
    grounding_prompt = (
        entity_phrase
        if grounding_prompt is None
        else _nonempty(grounding_prompt, "grounding_prompt")
    )
    clip_uid = _safe_component(clip_uid, "clip_uid")
    entity_id = _safe_component(entity_id, "entity_id")
    root = run_root.expanduser().resolve(strict=False)
    canonical_path = root / "clips" / clip_uid / "selected" / f"{entity_id}.png"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical reference does not exist: {canonical_path}")
    actual_source_path = (
        canonical_path
        if source_image_path is None
        else source_image_path.expanduser().resolve(strict=False)
    )
    if root not in actual_source_path.parents or not actual_source_path.is_file():
        raise ValueError("Boogu source image must be an existing run artifact")
    geometry_source_path = (
        actual_source_path
        if geometry_source_image_path is None
        else geometry_source_image_path.expanduser().resolve(strict=False)
    )
    if root not in geometry_source_path.parents or not geometry_source_path.is_file():
        raise ValueError("Boogu geometry source must be an existing run artifact")
    edit_dir = root / "clips" / clip_uid / "reference_edit" / entity_id
    edit_dir.mkdir(parents=True, exist_ok=True)
    candidate_name = (
        "completion_candidate_1k.png"
        if operation == "complete_entity"
        else "background_candidate_1k.png"
    )
    metadata_name = (
        "completion_metadata.json"
        if operation == "complete_entity"
        else "background_metadata.json"
    )
    candidate_path = edit_dir / candidate_name
    metadata_path = edit_dir / metadata_name
    final_path = edit_dir / "final_reference_1k.png"
    final_metadata_path = edit_dir / "final_metadata.json"
    rejection_path = edit_dir / (
        "completion_rejection.json"
        if operation == "complete_entity"
        else "background_rejection.json"
    )
    if not overwrite and (candidate_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Boogu edit output already exists: {metadata_path}")
    if overwrite:
        stale_paths = [candidate_path, metadata_path, rejection_path]
        if publish_final:
            stale_paths.extend((final_path, final_metadata_path))
        for stale in stale_paths:
            stale.unlink(missing_ok=True)

    canonical_bytes = canonical_path.read_bytes()
    canonical_sha256 = _sha256_bytes(canonical_bytes)
    source_bytes = actual_source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    source_rgba = _load_source_rgba(source_bytes)
    geometry_source_bytes = geometry_source_path.read_bytes()
    geometry_source_sha256 = _sha256_bytes(geometry_source_bytes)
    geometry_source_rgba = _load_source_rgba(geometry_source_bytes)
    source_content_geometry = content_geometry_from_rgba(geometry_source_rgba)
    source_gate_reason = tiny_content_reason(
        source_content_geometry,
        minimum_area_pixels=min_source_content_area_pixels,
        minimum_long_side_pixels=min_source_content_long_side_pixels,
    )
    source_input_rgb = _white_composite(source_rgba)
    width, height = resolve_boogu_1k_size(
        source_rgba.width,
        source_rgba.height,
        target_area=target_area,
        alignment=alignment,
    )
    if actual_source_path == canonical_path:
        source_evidence_path = edit_dir / "source_rgba.png"
        _write_bytes_atomic(source_evidence_path, source_bytes)
    else:
        source_evidence_path = actual_source_path
    source_input_path = edit_dir / (
        "completion_source_input_rgb.png"
        if operation == "complete_entity"
        else "background_source_input_rgb.png"
    )
    _save_rgb_png_atomic(source_input_path, source_input_rgb)
    thinking_enabled = operation == "complete_entity"

    output: BooguEditOutput | None = None
    output_sha256: str | None = None
    qwen_review: BooguQwenReview | None = None
    qwen_review_skipped_reason: str | None = None
    sam_review: BooguSamReview | None = None
    sam_warning: str | None = None
    rejection_reason: str | None = None
    accepted = False
    try:
        if source_gate_reason is not None:
            rejection_reason = "tiny_source_entity"
            raise _TinySourceRejected
        with profile_model_call(
            component=(
                "boogu_complete_entity"
                if operation == "complete_entity"
                else "boogu_add_entity_background"
            ),
            operation=operation,
            retry_index=0,
            model=type(backend).__name__,
            input_text_chars=len(instruction),
            input_image_count=1,
            metadata={
                "width": width,
                "height": height,
                "thinking_enabled": thinking_enabled,
            },
        ):
            output = backend.edit(
                source_rgb=source_input_rgb.copy(),
                instruction=instruction,
                width=width,
                height=height,
                thinking_enabled=thinking_enabled,
            )
        if output.original_instruction != instruction:
            raise RuntimeError("backend changed original instruction metadata")
        candidate_rgb = _validated_native_png(
            output.png_bytes,
            expected_size=(width, height),
        )
        _write_bytes_atomic(candidate_path, output.png_bytes)
        output_sha256 = _sha256_bytes(output.png_bytes)

        def run_qwen_review() -> BooguQwenReview:
            review = judge.review(
                operation=operation,
                source_rgba=source_rgba.copy(),
                source_input_rgb=source_input_rgb.copy(),
                candidate_rgb=candidate_rgb.copy(),
                entity_phrase=entity_phrase,
                reference_type=reference_type,
            )
            _validate_qwen_review_type(operation, review)
            return review

        def run_sam_review() -> BooguSamReview | None:
            if sam_reviewer is None:
                return None
            with profile_model_call(
                component="sam3_boogu_review",
                operation=operation,
                retry_index=0,
                model=type(sam_reviewer).__name__,
                input_text_chars=len(entity_phrase),
                input_image_count=2,
                metadata={"reference_type": reference_type},
            ):
                review = sam_reviewer.review(
                    operation=operation,
                    source_rgba=geometry_source_rgba.copy(),
                    candidate_rgb=candidate_rgb.copy(),
                    entity_phrase=entity_phrase,
                    reference_type=reference_type,
                )
            if not isinstance(review, BooguSamReview):
                raise TypeError("sam_reviewer must return BooguSamReview")
            return review

        if operation == "add_entity_background":
            sam_review = run_sam_review()
            geometry_metadata = _sam_geometry_metadata(sam_review)
            qwen_review_skipped_reason = _background_qwen_skip_reason(
                sam_review,
                geometry_metadata,
            )
            if qwen_review_skipped_reason is None:
                qwen_review = run_qwen_review()
        else:
            qwen_review = run_qwen_review()
            sam_review = run_sam_review()
            geometry_metadata = _sam_geometry_metadata(sam_review)

        qwen_accepted = qwen_review is not None and qwen_review.verdict == "accept"
        sam_accepted = sam_review is None or sam_review.passed
        if geometry_metadata and not geometry_metadata["geometry_gate_passed"]:
            sam_accepted = False
        if (
            qwen_accepted
            and operation == "add_entity_background"
            and sam_review is not None
            and sam_review.diagnostics["failure_kind"] == "not_found"
        ):
            sam_accepted = True
            sam_warning = "target_not_found"
        accepted = qwen_accepted and sam_accepted
        if not accepted:
            rejection_reason = (
                str(geometry_metadata["geometry_rejection_reason"])
                if geometry_metadata
                and not geometry_metadata["geometry_gate_passed"]
                else qwen_review_skipped_reason
                if qwen_review_skipped_reason is not None
                else _background_qwen_rejection_reason(qwen_review)
                if isinstance(qwen_review, BooguBackgroundReview)
                and qwen_review.verdict == "reject"
                else qwen_review.reason
                if qwen_review is not None and qwen_review.verdict == "reject"
                else sam_review.reason
                if sam_review is not None
                else "candidate_rejected"
            )
    except _TinySourceRejected:
        pass
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        accepted = False
        rejection_reason = f"boogu_reference_edit_failed: {exc}"

    source_ratio = source_rgba.width / source_rgba.height
    output_ratio = width / height
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "backend": "boogu_image_0_1_edit_turbo",
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "status": "accepted" if accepted else "rejected",
        "operation": operation,
        "entity_phrase": entity_phrase,
        "grounding_prompt": grounding_prompt,
        "source_dimensions": {
            "width": source_rgba.width,
            "height": source_rgba.height,
        },
        "resolved_output_dimensions": {"width": width, "height": height},
        "target_area": target_area,
        "alignment": alignment,
        "source_aspect_ratio": source_ratio,
        "output_aspect_ratio": output_ratio,
        "aspect_ratio_error": abs(output_ratio - source_ratio) / source_ratio,
        "output_pixel_count": width * height,
        "model_name": BOOGU_MODEL_NAME,
        "model_revision": model_revision,
        "original_instruction": instruction,
        "rewritten_instruction": (
            output.rewritten_instruction if output is not None else None
        ),
        "effective_instruction": (
            output.effective_instruction if output is not None else instruction
        ),
        "thinking_enabled": thinking_enabled,
        "output_sha256": output_sha256,
        "canonical_source_sha256": canonical_sha256,
        "source_image_path": source_evidence_path.relative_to(root).as_posix(),
        "source_image_sha256": source_sha256,
        "source_geometry_image_path": geometry_source_path.relative_to(root).as_posix(),
        "source_geometry_image_sha256": geometry_source_sha256,
        "source_input_rgb_path": source_input_path.relative_to(root).as_posix(),
        "generated_reference_sha256": output_sha256,
        "qwen_review": (
            qwen_review.model_dump(mode="json") if qwen_review is not None else None
        ),
        "qwen_review_skipped_reason": qwen_review_skipped_reason,
        "sam_review": (
            sam_review.model_dump(mode="json") if sam_review is not None else None
        ),
        "sam_warning": sam_warning,
        "sam_mask_usage": "review_only",
        "fallback_status": "not_used" if accepted else fallback_status,
        "candidate_path": candidate_name if candidate_path.is_file() else None,
        "worker_metadata": output.worker_metadata if output is not None else {},
        **source_geometry_metadata(
            source_content_geometry,
            source_gate_reason=source_gate_reason,
        ),
        **_sam_geometry_metadata(sam_review),
    }
    write_json_atomic(metadata_path, metadata)

    if _sha256_bytes(canonical_path.read_bytes()) != canonical_sha256:
        raise RuntimeError("canonical reference changed during Boogu reference edit")

    if accepted:
        if not candidate_path.is_file() or output_sha256 is None:
            raise RuntimeError("accepted Boogu edit has no candidate artifact")
        publication = None
        if publish_final:
            publication = publish_boogu_final_reference(
                run_root=root,
                clip_uid=clip_uid,
                entity_id=entity_id,
                candidate_path=candidate_path,
                selected_metadata_path=metadata_path,
                completion_metadata_path=(
                    metadata_path if operation == "complete_entity" else None
                ),
                background_metadata_path=(
                    metadata_path if operation == "add_entity_background" else None
                ),
                final_selection=(
                    "completion_candidate"
                    if operation == "complete_entity"
                    else "background_candidate"
                ),
                final_selection_reason=(
                    "accepted_completion_candidate"
                    if operation == "complete_entity"
                    else "accepted_background_candidate_preferred_over_source"
                ),
            )
        rejection_path.unlink(missing_ok=True)
        return BooguReferenceEditResult(
            status="accepted",
            operation=operation,
            candidate_path=candidate_path,
            final_reference_path=(
                publication.final_reference_path if publication is not None else None
            ),
            metadata_path=metadata_path,
            rejection_path=None,
            fallback_status="not_used",
        )

    if publish_final:
        final_path.unlink(missing_ok=True)
        final_metadata_path.unlink(missing_ok=True)
    rejection = {
        "schema_version": 1,
        "status": "rejected",
        "operation": operation,
        "reason": rejection_reason or "candidate_rejected",
        "canonical_source_sha256": canonical_sha256,
        "candidate_sha256": output_sha256,
        "fallback_status": fallback_status,
    }
    write_json_atomic(rejection_path, rejection)
    return BooguReferenceEditResult(
        status="rejected",
        operation=operation,
        candidate_path=candidate_path if candidate_path.is_file() else None,
        final_reference_path=None,
        metadata_path=metadata_path,
        rejection_path=rejection_path,
        fallback_status=fallback_status,
    )


def _validate_qwen_review_type(
    operation: BooguEditOperation,
    review: BooguQwenReview,
) -> None:
    expected = (
        BooguCompletionReview
        if operation == "complete_entity"
        else BooguBackgroundReview
    )
    if not isinstance(review, expected):
        raise TypeError(f"judge must return {expected.__name__} for {operation}")


def _validated_native_png(
    png_bytes: bytes,
    *,
    expected_size: tuple[int, int],
) -> Image.Image:
    if not isinstance(png_bytes, bytes) or not png_bytes:
        raise ValueError("Boogu output must contain PNG bytes")
    try:
        with Image.open(io.BytesIO(png_bytes)) as loaded:
            loaded.load()
            image_format = loaded.format
            mode = loaded.mode
            size = loaded.size
            candidate = loaded.copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Boogu output is not a readable image: {exc}") from exc
    if image_format != "PNG":
        raise ValueError(f"Boogu output must be PNG, got {image_format}")
    if mode != "RGB":
        raise ValueError(f"Boogu output must be RGB, got {mode}")
    if size != expected_size:
        raise ValueError(
            "Boogu output dimensions do not match resolved 1K size: "
            f"expected_size={expected_size}, returned_size={size}"
        )
    return candidate


def _validated_native_rgb_png(png_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(png_bytes)) as loaded:
            size = loaded.size
    except OSError as exc:
        raise ValueError(f"Boogu output is not a readable image: {exc}") from exc
    return _validated_native_png(png_bytes, expected_size=size)


def _load_source_rgba(source_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(source_bytes)) as loaded:
            if loaded.format != "PNG":
                raise ValueError("Boogu source reference must be PNG")
            loaded.load()
            return loaded.convert("RGBA")
    except OSError as exc:
        raise ValueError(f"Boogu source reference is not a readable PNG: {exc}") from exc


def _white_composite(source_rgba: Image.Image) -> Image.Image:
    if source_rgba.mode != "RGBA":
        raise ValueError("source image must be RGBA before white compositing")
    white = Image.new("RGB", source_rgba.size, (255, 255, 255))
    white.paste(source_rgba.convert("RGB"), mask=source_rgba.getchannel("A"))
    return white


def _validate_output_dimensions(width: int, height: int) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (width, height)
    ):
        raise ValueError("Boogu output dimensions must be positive integers")


def _save_rgb_png_atomic(path: Path, image: Image.Image) -> None:
    if image.mode != "RGB":
        raise ValueError("artifact image must be RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    _write_bytes_atomic(path, buffer.getvalue())


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsafe path characters")
    return value


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()


def _require_matching_verdict(
    verdict: Literal["accept", "reject"],
    flags: tuple[bool, ...],
) -> None:
    if (verdict == "accept") != all(flags):
        raise ValueError("verdict must accept if and only if every flag is true")
