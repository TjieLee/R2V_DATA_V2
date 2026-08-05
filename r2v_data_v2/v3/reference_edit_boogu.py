"""Isolated Boogu reference editing and fail-closed artifact publication.

Boogu produces a new, full-resolution reference image. This module therefore
never composites source pixels or masks into a generated candidate. SAM review
is an optional quality signal only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

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

BooguEditOperation = Literal["complete_entity", "add_entity_background"]
ReferenceType = Literal["subject", "object", "group"]
EditStatus = Literal["accepted", "rejected"]

BOOGU_MODEL_NAME = "Boogu-Image-0.1-Edit-Turbo"
BOOGU_MODEL_REVISION = "hotfix-1k-20260708"
DEFAULT_BOOGU_CODE_ROOT = Path("/mnt/workspace/litengjie/data/vendor/Boogu-Image")
DEFAULT_BOOGU_PYTHON = Path(
    "/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python"
)
DEFAULT_BOOGU_MODEL_PATH = Path(
    "/mnt/workspace/litengjie/data/models/"
    "Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708"
)
DEFAULT_ALLOWED_SERVER_ROOT = Path("/mnt/workspace/litengjie/data")
DEFAULT_TARGET_AREA = 1024 * 1024
DEFAULT_ALIGNMENT = 16

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
            area_error = abs(
                candidate_width * candidate_height - target_area
            ) / target_area
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
    background_coherent: StrictBool
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
            self.background_coherent,
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


@dataclass(frozen=True)
class BooguWorkerConfig:
    python_executable: Path = DEFAULT_BOOGU_PYTHON
    code_root: Path = DEFAULT_BOOGU_CODE_ROOT
    model_path: Path = DEFAULT_BOOGU_MODEL_PATH
    model_revision: str = BOOGU_MODEL_REVISION
    device: str = "cuda:0"
    seed: int = 0
    timeout_seconds: int = 3600
    allowed_server_root: Path = DEFAULT_ALLOWED_SERVER_ROOT
    worker_script: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
        / "tools"
        / "run_v3_boogu_reference_edit_worker.py"
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
        if not isinstance(self.worker_script, Path) or not self.worker_script.is_absolute():
            raise ValueError("worker_script must be an absolute pathlib.Path")
        if not self.model_revision.strip():
            raise ValueError("model_revision must be non-empty")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.timeout_seconds, int)
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds < 1
        ):
            raise ValueError("timeout_seconds must be a positive integer")


class BooguSubprocessBackend:
    """Run Boogu through its configured Linux Python without shell activation."""

    def __init__(self, config: BooguWorkerConfig) -> None:
        config.validate()
        self.config = config

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
        config = self.config
        with tempfile.TemporaryDirectory(prefix="r2v-boogu-") as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "source_input_rgb.png"
            output_path = temporary / "candidate.png"
            result_path = temporary / "result.json"
            request_path = temporary / "request.json"
            source_rgb.save(input_path, format="PNG")
            request = {
                "schema_version": 1,
                "code_root": str(config.code_root),
                "model_path": str(config.model_path),
                "model_name": BOOGU_MODEL_NAME,
                "model_revision": config.model_revision,
                "device": config.device,
                "seed": config.seed,
                "input_image_path": str(input_path),
                "output_image_path": str(output_path),
                "result_path": str(result_path),
                "instruction": instruction,
                "thinking_enabled": thinking_enabled,
                "width": width,
                "height": height,
            }
            request_path.write_text(
                json.dumps(request, ensure_ascii=True, sort_keys=True),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONNOUSERSITE"] = "1"
            subprocess.run(
                [
                    str(config.python_executable),
                    str(config.worker_script),
                    "--request",
                    str(request_path),
                ],
                cwd=config.code_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
            if not result_path.is_file() or not output_path.is_file():
                raise RuntimeError("Boogu worker did not publish its result artifacts")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise TypeError("Boogu worker result must be a JSON object")
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


@dataclass(frozen=True)
class BooguReferenceEditResult:
    status: EditStatus
    operation: BooguEditOperation
    candidate_path: Path | None
    final_reference_path: Path | None
    metadata_path: Path
    rejection_path: Path | None
    fallback_status: str


def run_boogu_reference_edit(
    *,
    run_root: Path,
    clip_uid: str,
    entity_id: str,
    operation: BooguEditOperation,
    instruction: str,
    entity_phrase: str,
    reference_type: ReferenceType,
    backend: BooguReferenceEditBackend,
    judge: BooguReferenceEditJudge,
    sam_reviewer: BooguSamReviewer | None = None,
    target_area: int = DEFAULT_TARGET_AREA,
    alignment: int = DEFAULT_ALIGNMENT,
    model_revision: str = BOOGU_MODEL_REVISION,
    fallback_status: str = "canonical_preserved",
    overwrite: bool = False,
) -> BooguReferenceEditResult:
    """Generate, review, and publish one native Boogu reference artifact."""

    if operation not in {"complete_entity", "add_entity_background"}:
        raise ValueError(f"unsupported Boogu edit operation: {operation}")
    instruction = _nonempty(instruction, "instruction")
    entity_phrase = _nonempty(entity_phrase, "entity_phrase")
    clip_uid = _safe_component(clip_uid, "clip_uid")
    entity_id = _safe_component(entity_id, "entity_id")
    root = run_root.expanduser().resolve(strict=False)
    canonical_path = root / "clips" / clip_uid / "selected" / f"{entity_id}.png"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical reference does not exist: {canonical_path}")
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
    rejection_path = edit_dir / "rejection.json"
    if not overwrite and (candidate_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"Boogu edit output already exists: {metadata_path}")
    if overwrite:
        for stale in (
            candidate_path,
            metadata_path,
            final_path,
            final_metadata_path,
            rejection_path,
        ):
            stale.unlink(missing_ok=True)

    canonical_bytes = canonical_path.read_bytes()
    canonical_sha256 = _sha256_bytes(canonical_bytes)
    source_rgba = _load_source_rgba(canonical_bytes)
    source_input_rgb = _white_composite(source_rgba)
    width, height = resolve_boogu_1k_size(
        source_rgba.width,
        source_rgba.height,
        target_area=target_area,
        alignment=alignment,
    )
    _write_bytes_atomic(edit_dir / "source_rgba.png", canonical_bytes)
    _save_rgb_png_atomic(edit_dir / "source_input_rgb.png", source_input_rgb)
    thinking_enabled = operation == "complete_entity"

    output: BooguEditOutput | None = None
    output_sha256: str | None = None
    qwen_review: BooguQwenReview | None = None
    sam_review: BooguSamReview | None = None
    rejection_reason: str | None = None
    try:
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
        qwen_review = judge.review(
            operation=operation,
            source_rgba=source_rgba.copy(),
            source_input_rgb=source_input_rgb.copy(),
            candidate_rgb=candidate_rgb.copy(),
            entity_phrase=entity_phrase,
            reference_type=reference_type,
        )
        _validate_qwen_review_type(operation, qwen_review)
        if sam_reviewer is not None:
            sam_review = sam_reviewer.review(
                operation=operation,
                source_rgba=source_rgba.copy(),
                candidate_rgb=candidate_rgb.copy(),
                entity_phrase=entity_phrase,
                reference_type=reference_type,
            )
            if not isinstance(sam_review, BooguSamReview):
                raise TypeError("sam_reviewer must return BooguSamReview")
        accepted = qwen_review.verdict == "accept" and (
            sam_review is None or sam_review.passed
        )
        if not accepted:
            rejection_reason = (
                qwen_review.reason
                if qwen_review.verdict == "reject"
                else sam_review.reason
                if sam_review is not None
                else "candidate_rejected"
            )
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        accepted = False
        rejection_reason = f"boogu_reference_edit_failed: {exc}"

    source_ratio = source_rgba.width / source_rgba.height
    output_ratio = width / height
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted" if accepted else "rejected",
        "operation": operation,
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
        "qwen_review": (
            qwen_review.model_dump(mode="json") if qwen_review is not None else None
        ),
        "sam_review": (
            sam_review.model_dump(mode="json") if sam_review is not None else None
        ),
        "sam_mask_usage": "review_only",
        "fallback_status": "not_used" if accepted else fallback_status,
        "candidate_path": candidate_name if candidate_path.is_file() else None,
        "worker_metadata": output.worker_metadata if output is not None else {},
    }
    write_json_atomic(metadata_path, metadata)

    if _sha256_bytes(canonical_path.read_bytes()) != canonical_sha256:
        raise RuntimeError("canonical reference changed during Boogu reference edit")

    if accepted:
        if not candidate_path.is_file() or output_sha256 is None:
            raise RuntimeError("accepted Boogu edit has no candidate artifact")
        _write_bytes_atomic(final_path, candidate_path.read_bytes())
        if final_path.read_bytes() != candidate_path.read_bytes():
            raise RuntimeError("final reference is not the native Boogu candidate")
        final_metadata = {
            **metadata,
            "final_reference_path": final_path.name,
            "final_reference_sha256": output_sha256,
        }
        write_json_atomic(final_metadata_path, final_metadata)
        rejection_path.unlink(missing_ok=True)
        return BooguReferenceEditResult(
            status="accepted",
            operation=operation,
            candidate_path=candidate_path,
            final_reference_path=final_path,
            metadata_path=metadata_path,
            rejection_path=None,
            fallback_status="not_used",
        )

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


def _load_source_rgba(source_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(source_bytes)) as loaded:
            if loaded.format != "PNG":
                raise ValueError("canonical reference must be PNG")
            loaded.load()
            return loaded.convert("RGBA")
    except OSError as exc:
        raise ValueError(f"canonical reference is not a readable PNG: {exc}") from exc


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
