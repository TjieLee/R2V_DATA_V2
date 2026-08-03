from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import math
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

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
JudgeVerdict = Literal["accept", "reject"]

ALLOWED_INPUT_ROOT = Path("/mnt/workspace/litengjie/data").resolve()
ALLOWED_BENCHMARK_ROOT = (ALLOWED_INPUT_ROOT / "reference_inpaint_benchmarks").resolve()
DEFAULT_SEEDS = (0, 17)

JUDGE_SYSTEM_PROMPT = """You judge an entity-reference background benchmark.

You receive exactly three images: the original source-faithful entity reference
shown on white, one generated RGB candidate, and the binary entity alpha mask.
Judge visible evidence only. The supplied entity must remain unchanged, must not
be duplicated, and must remain usable as a reference. The generated background
must be natural, secondary, and free of new people, animals, vehicles, products,
text, signs, or other salient entities. Reject halos, seams, implausible shadows,
or copy-paste boundaries.

Return one strict JSON object containing only verdict, entity_unchanged,
no_duplicate_entity, no_new_salient_entity, background_natural, boundary_clean,
reference_usable, and reason. verdict is accept if and only if every boolean is
true. Return JSON only."""


class BenchmarkManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    clip_uid: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    reference_type: ReferenceType
    entity_phrase: str = Field(min_length=1)
    source_rgba_path: Path

    @field_validator("clip_uid", "entity_id", "entity_phrase")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be non-empty")
        return stripped


class ReferenceBackgroundReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: JudgeVerdict
    entity_unchanged: StrictBool
    no_duplicate_entity: StrictBool
    no_new_salient_entity: StrictBool
    background_natural: StrictBool
    boundary_clean: StrictBool
    reference_usable: StrictBool
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _reason_must_be_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must be non-empty")
        return stripped

    @model_validator(mode="after")
    def _verdict_matches_flags(self) -> ReferenceBackgroundReview:
        all_pass = all(
            (
                self.entity_unchanged,
                self.no_duplicate_entity,
                self.no_new_salient_entity,
                self.background_natural,
                self.boundary_clean,
                self.reference_usable,
            )
        )
        if (self.verdict == "accept") != all_pass:
            raise ValueError(
                "verdict must be accept if and only if every quality flag is true"
            )
        return self


class ReferenceBackgroundInpainter(Protocol):
    def generate(
        self,
        *,
        source_rgba: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        seed: int,
        prompt: str,
    ) -> Image.Image: ...


class ReferenceBackgroundJudge(Protocol):
    def review(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        alpha_mask: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> ReferenceBackgroundReview: ...


class ReferenceBackgroundJudgeFailure(StructuredOutputFailure):
    pass


@dataclass(frozen=True)
class QwenImageEditBenchmarkConfig:
    model_path: Path
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: str = " "

    def validate(self) -> None:
        if not isinstance(self.model_path, Path):
            raise TypeError("model_path must be a pathlib.Path")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not self.dtype.strip():
            raise ValueError("dtype must be non-empty")
        if (
            not isinstance(self.num_inference_steps, int)
            or isinstance(self.num_inference_steps, bool)
            or self.num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        for name, value in (
            ("true_cfg_scale", self.true_cfg_scale),
            ("guidance_scale", self.guidance_scale),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


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
class BenchmarkStats:
    processed: int
    accepted: int
    rejected: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_absolute_path(path: Path, *, field_name: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")


def _resolve_input_path(path: Path, *, field_name: str) -> Path:
    _require_absolute_path(path, field_name=field_name)
    resolved = path.expanduser().resolve(strict=True)
    if not _is_at_or_below(resolved, ALLOWED_INPUT_ROOT):
        raise ValueError(f"{field_name} must remain under {ALLOWED_INPUT_ROOT}")
    return resolved


def _resolve_benchmark_root(path: Path) -> Path:
    _require_absolute_path(path, field_name="benchmark_root")
    resolved = path.expanduser().resolve(strict=False)
    if resolved == ALLOWED_BENCHMARK_ROOT or not _is_at_or_below(
        resolved,
        ALLOWED_BENCHMARK_ROOT,
    ):
        raise ValueError(
            "benchmark_root must be a run directory strictly below "
            f"{ALLOWED_BENCHMARK_ROOT}"
        )
    relative = resolved.relative_to(ALLOWED_BENCHMARK_ROOT)
    if "selected" in {part.casefold() for part in relative.parts}:
        raise ValueError("benchmark_root must not write into selected/")
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
    with Image.open(path) as published:
        published.load()
        if published.format != "PNG" or published.mode != image.mode:
            raise RuntimeError("published image format or mode changed unexpectedly")
        if published.size != image.size:
            raise RuntimeError("published image dimensions changed unexpectedly")


def _white_baseline(source_rgba: Image.Image) -> Image.Image:
    source = np.asarray(source_rgba, dtype=np.uint8)
    foreground = source[:, :, 3] == 255
    baseline = np.full(source.shape[:2] + (3,), 255, dtype=np.uint8)
    baseline[foreground] = source[:, :, :3][foreground]
    return Image.fromarray(baseline)


def build_reference_background_prompt(
    *,
    entity_phrase: str,
    reference_type: ReferenceType,
) -> str:
    phrase = entity_phrase.strip()
    if not phrase:
        raise ValueError("entity_phrase must be non-empty")
    kind = {
        "subject": "subject",
        "object": "object",
        "group": "coherent group",
    }[reference_type]
    return (
        "Generate a clean, natural, non-salient background around the supplied "
        f"{kind}, described as: {phrase}. Preserve the supplied entity exactly "
        "and use generated pixels only for its transparent background. Do not "
        "add another person, animal, vehicle, product, text, sign, or duplicate "
        "of the entity. Do not change its pose, clothing, anatomy, geometry, "
        "texture, or color. Keep the background visually plausible but secondary. "
        "Avoid studio cutout halos, seams, inconsistent shadows, and copy-paste "
        "appearance."
    )


def pad_image_to_multiple(
    image: Image.Image,
    *,
    multiple: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if image.mode != "RGB":
        raise ValueError("padding requires an RGB image")
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0:
        raise ValueError("multiple must be a positive integer")
    width, height = image.size
    target_width = ((width + multiple - 1) // multiple) * multiple
    target_height = ((height + multiple - 1) // multiple) * multiple
    horizontal = target_width - width
    vertical = target_height - height
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    crop_box = (left, top, left + width, top + height)
    if not any((left, right, top, bottom)):
        return image.copy(), crop_box
    padded = np.pad(
        np.asarray(image),
        ((top, bottom), (left, right), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded), crop_box


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
            f"installed API does not support {operation} parameters {missing}"
        )


class QwenImageEditReferenceBackgroundInpainter:
    """Optional local-only experimental backend without the remover LoRA."""

    def __init__(
        self,
        config: QwenImageEditBenchmarkConfig,
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
        model_path = self.config.model_path.expanduser().resolve(strict=False)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Qwen Image Edit model path does not exist: {model_path}"
            )
        import torch
        from diffusers import QwenImageEditPlusPipeline

        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise RuntimeError(
                f"installed torch does not provide dtype {self.config.dtype}"
            )
        _require_parameters(
            QwenImageEditPlusPipeline.from_pretrained,
            ("torch_dtype", "local_files_only"),
            operation="QwenImageEditPlusPipeline.from_pretrained",
        )
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        progress = getattr(pipeline, "set_progress_bar_config", None)
        if callable(progress):
            progress(disable=True)
        to_method = getattr(pipeline, "to", None)
        if not callable(to_method):
            raise TypeError("Qwen Image Edit pipeline has no device transfer API")
        self._pipeline = to_method(self.config.device)
        self._torch = torch

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(
                "Qwen Image Edit benchmark backend previously failed to load"
            ) from self._load_error
        try:
            self._load()
        except Exception as exc:
            self._load_error = exc
            raise

    def generate(
        self,
        *,
        source_rgba: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        seed: int,
        prompt: str,
    ) -> Image.Image:
        if not isinstance(source_rgba, Image.Image) or source_rgba.mode != "RGBA":
            raise ValueError("Qwen benchmark input must be an RGBA image")
        if not entity_phrase.strip() or reference_type not in {
            "subject",
            "object",
            "group",
        }:
            raise ValueError("Qwen benchmark entity metadata is invalid")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        self._ensure_loaded()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("Qwen Image Edit benchmark backend is unavailable")

        original_size = source_rgba.size
        baseline = _white_baseline(source_rgba)
        multiple = int(self._pipeline.vae_scale_factor) * 2
        padded, crop_box = pad_image_to_multiple(baseline, multiple=multiple)
        generator = self._torch.Generator(device=self.config.device).manual_seed(seed)
        parameters = {
            "image": [padded],
            "height": padded.height,
            "width": padded.width,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": self.config.true_cfg_scale,
            "negative_prompt": self.config.negative_prompt,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "num_images_per_prompt": 1,
        }
        _require_parameters(
            self._pipeline.__call__,
            tuple(parameters),
            operation="QwenImageEditPlusPipeline.__call__",
        )
        result = self._pipeline(**parameters)
        images = getattr(result, "images", None)
        if (
            not isinstance(images, (list, tuple))
            or not images
            or not isinstance(images[0], Image.Image)
        ):
            raise TypeError("Qwen benchmark backend did not return a PIL image")
        edited = images[0]
        if edited.size != padded.size:
            raise RuntimeError(
                "Qwen benchmark output dimensions do not match padded input"
            )
        cropped = edited.crop(crop_box).convert("RGB")
        if cropped.size != original_size:
            raise RuntimeError("Qwen benchmark crop dimensions do not match source")
        return cropped

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            del pipeline
        torch = self._torch
        self._torch = None
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class QwenReferenceBackgroundJudge:
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
        candidate_rgb: Image.Image,
        alpha_mask: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
        request_text: str,
    ) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [
            {"type": "text", "text": request_text},
        ]
        for label, image in (
            ("Image 1: original entity on white", _white_baseline(source_rgba)),
            ("Image 2: generated RGB candidate", candidate_rgb),
            ("Image 3: binary entity alpha mask", alpha_mask.convert("L")),
        ):
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _png_data_url(image)},
                }
            )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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
                        "name": "v3_reference_background_review",
                        "strict": True,
                        "schema": ReferenceBackgroundReview.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty benchmark review")
        return str(content)

    def review(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        alpha_mask: Image.Image,
        entity_phrase: str,
        reference_type: ReferenceType,
    ) -> ReferenceBackgroundReview:
        if source_rgba.mode != "RGBA" or candidate_rgb.mode != "RGB":
            raise ValueError("judge requires RGBA source and RGB candidate")
        if (
            source_rgba.size != candidate_rgb.size
            or alpha_mask.size != source_rgba.size
        ):
            raise ValueError("judge images must have matching dimensions")
        original_request = (
            "Judge this immutable benchmark candidate. "
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
                    json_schema=ReferenceBackgroundReview.model_json_schema(),
                )
            try:
                raw = self._request(
                    self._messages(
                        source_rgba=source_rgba,
                        candidate_rgb=candidate_rgb,
                        alpha_mask=alpha_mask,
                        entity_phrase=entity_phrase,
                        reference_type=reference_type,
                        request_text=request_text,
                    )
                )
            except Exception as exc:
                raise ReferenceBackgroundJudgeFailure(
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
                ReferenceBackgroundReview,
            )
            if review is not None and not issues:
                return review
        raise ReferenceBackgroundJudgeFailure(
            raw_responses=raw_responses,
            issues=issues,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


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
    unique_alpha = {int(value) for value in np.unique(alpha)}
    if not unique_alpha.issubset({0, 255}):
        raise ValueError("source alpha must contain only 0 and 255")
    if 255 not in unique_alpha:
        raise ValueError("source reference foreground is empty")
    if 0 not in unique_alpha:
        raise ValueError("source reference has no background generation region")
    return source, source_bytes, _sha256_bytes(source_bytes)


def _candidate_from_generated(
    source_rgba: Image.Image,
    generated_rgb: object,
) -> tuple[Image.Image | None, HardCheckReport]:
    checks = {
        "pil_rgb": isinstance(generated_rgb, Image.Image)
        and generated_rgb.mode == "RGB",
        "size_matches_source": False,
        "finite_uint8_pixels": False,
        "foreground_rgb_exact": False,
        "background_nonconstant": False,
        "opaque_boundary": False,
    }
    reasons: list[str] = []
    if not checks["pil_rgb"]:
        reasons.append("generated_output_must_be_rgb_pil")
        return None, HardCheckReport(False, checks, tuple(reasons))
    assert isinstance(generated_rgb, Image.Image)
    checks["size_matches_source"] = generated_rgb.size == source_rgba.size
    if not checks["size_matches_source"]:
        reasons.append("generated_output_size_mismatch")
        return None, HardCheckReport(False, checks, tuple(reasons))

    generated = np.asarray(generated_rgb)
    checks["finite_uint8_pixels"] = generated.dtype == np.uint8 and bool(
        np.isfinite(generated).all()
    )
    if not checks["finite_uint8_pixels"]:
        reasons.append("generated_output_has_invalid_pixels")
        return None, HardCheckReport(False, checks, tuple(reasons))

    source = np.asarray(source_rgba, dtype=np.uint8)
    entity_mask = source[:, :, 3] == 255
    background_mask = ~entity_mask
    composited = generated.copy()
    composited[entity_mask] = source[:, :, :3][entity_mask]
    candidate = Image.fromarray(composited)
    checks["foreground_rgb_exact"] = bool(
        np.array_equal(
            np.asarray(candidate)[entity_mask],
            source[:, :, :3][entity_mask],
        )
    )
    background_pixels = composited[background_mask]
    checks["background_nonconstant"] = bool(
        background_pixels.size
        and np.unique(background_pixels.reshape(-1, 3), axis=0).shape[0] > 1
    )
    checks["opaque_boundary"] = candidate.mode == "RGB"
    if not checks["foreground_rgb_exact"]:
        reasons.append("source_foreground_pixels_changed")
    if not checks["background_nonconstant"]:
        reasons.append("generated_background_is_constant")
    if not checks["opaque_boundary"]:
        reasons.append("candidate_contains_transparency")
    return candidate, HardCheckReport(
        passed=all(checks.values()),
        checks=checks,
        reasons=tuple(reasons),
    )


def _synthetic_reject_review(reason: str) -> ReferenceBackgroundReview:
    return ReferenceBackgroundReview(
        verdict="reject",
        entity_unchanged=False,
        no_duplicate_entity=False,
        no_new_salient_entity=False,
        background_natural=False,
        boundary_clean=False,
        reference_usable=False,
        reason=reason,
    )


def _validated_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not seeds or len(seeds) > 2:
        raise ValueError("benchmark requires one or two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be unique")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("benchmark seeds must be non-negative integers")
    return seeds


def load_benchmark_manifest(path: Path) -> list[BenchmarkManifestRecord]:
    resolved = _resolve_input_path(path, field_name="manifest_path")
    records: list[BenchmarkManifestRecord] = []
    seen_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                record = BenchmarkManifestRecord.model_validate(payload)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid benchmark manifest line {line_number}: {exc}"
                ) from exc
            if record.sample_id in seen_ids:
                raise ValueError(f"duplicate benchmark sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError("benchmark manifest contains no records")
    return records


def _process_record(
    record: BenchmarkManifestRecord,
    *,
    benchmark_root: Path,
    backend: ReferenceBackgroundInpainter,
    judge: ReferenceBackgroundJudge,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    source_path = _resolve_input_path(
        record.source_rgba_path,
        field_name="source_rgba_path",
    )
    source_rgba, source_bytes, source_sha256 = _load_source_rgba(source_path)
    final_directory = benchmark_root / record.sample_id
    if final_directory.exists():
        raise FileExistsError(
            f"benchmark sample output already exists: {final_directory}"
        )
    temporary = benchmark_root / (f".{record.sample_id}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    try:
        (temporary / "source_rgba.png").write_bytes(source_bytes)
        _save_png(temporary / "baseline_white.png", _white_baseline(source_rgba))
        alpha_mask = source_rgba.getchannel("A")
        prompt = build_reference_background_prompt(
            entity_phrase=record.entity_phrase,
            reference_type=record.reference_type,
        )
        attempts: list[dict[str, object]] = []
        accepted: dict[str, object] | None = None
        for seed in seeds:
            attempt_started = time.perf_counter()
            candidate_name = f"candidate_seed_{seed}.png"
            review_name = f"review_seed_{seed}.json"
            candidate_sha256: str | None = None
            judge_status = "not_run"
            reason = ""
            try:
                generated = backend.generate(
                    source_rgba=source_rgba.copy(),
                    entity_phrase=record.entity_phrase,
                    reference_type=record.reference_type,
                    seed=seed,
                    prompt=prompt,
                )
                candidate, hard_check = _candidate_from_generated(
                    source_rgba,
                    generated,
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
                        candidate_rgb=candidate.copy(),
                        alpha_mask=alpha_mask.copy(),
                        entity_phrase=record.entity_phrase,
                        reference_type=record.reference_type,
                    )
                    if not isinstance(review, ReferenceBackgroundReview):
                        raise TypeError("judge must return ReferenceBackgroundReview")
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
            attempt_record: dict[str, object] = {
                "seed": seed,
                "candidate_path": candidate_name if candidate is not None else None,
                "candidate_sha256": candidate_sha256,
                "hard_check": hard_check.to_dict(),
                "judge_status": judge_status,
                "judge_verdict": review.verdict,
                "review_path": review_name,
                "runtime_seconds": time.perf_counter() - attempt_started,
                "reason": reason,
            }
            attempts.append(attempt_record)
            if accepted is None and hard_check.passed and review.verdict == "accept":
                accepted = {
                    "seed": seed,
                    "candidate_path": candidate_name,
                    "candidate_sha256": candidate_sha256,
                }

        if _sha256_path(source_path) != source_sha256:
            raise RuntimeError("source reference changed during benchmark")
        status = "accepted" if accepted is not None else "rejected"
        result: dict[str, object] = {
            "sample_id": record.sample_id,
            "clip_uid": record.clip_uid,
            "entity_id": record.entity_id,
            "reference_type": record.reference_type,
            "entity_phrase": record.entity_phrase,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "status": status,
            "accepted_candidate": accepted,
            "attempts": attempts,
            "runtime_seconds": time.perf_counter() - started,
            "reason": (
                f"first accepted seed: {accepted['seed']}"
                if accepted is not None
                else "no candidate passed hard checks and judge review"
            ),
        }
        write_json_atomic(temporary / "result.json", result)
        temporary.replace(final_directory)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_reference_inpaint_benchmark(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    backend: ReferenceBackgroundInpainter,
    judge: ReferenceBackgroundJudge,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> BenchmarkStats:
    resolved_root = _resolve_benchmark_root(benchmark_root)
    validated_seeds = _validated_seeds(seeds)
    records = load_benchmark_manifest(manifest_path)
    resolved_root.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    for record in records:
        result = _process_record(
            record,
            benchmark_root=resolved_root,
            backend=backend,
            judge=judge,
            seeds=validated_seeds,
        )
        accepted += int(result["status"] == "accepted")
        rejected += int(result["status"] == "rejected")
    return BenchmarkStats(
        processed=len(records),
        accepted=accepted,
        rejected=rejected,
    )
