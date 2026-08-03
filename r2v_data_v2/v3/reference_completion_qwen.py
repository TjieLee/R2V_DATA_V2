from __future__ import annotations

import inspect
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.reference_completion_benchmark import (
    ALLOWED_INPUT_ROOT,
    CompletionBenchmarkStats,
    CompletionCandidateSpec,
    CompletionManifestRecord,
    ReferenceCompletionJudge,
    run_completion_benchmark,
)

QWEN_COMPLETION_BACKEND = "qwen_image_edit_2511"
DEFAULT_QWEN_MODEL_PATH = Path(
    "/mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511"
)
ALLOWED_QWEN_BENCHMARK_ROOT = (
    ALLOWED_INPUT_ROOT / "reference_completion_qwen_benchmarks"
).resolve()
DEFAULT_QWEN_SEEDS = (0, 17)

DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT = (
    "Complete the missing parts of the same single person shown in this image. "
    "Preserve the person's identity, face, hairstyle, clothing, accessories, "
    "pose, body proportions, lighting, perspective, and all visible appearance "
    "details. Extend only the incomplete body and clothing naturally into the "
    "empty region. Do not create another person, another face, duplicate body "
    "parts, new clothing, or new foreground objects. Do not alter the already "
    "visible person. Keep the background plain white."
)
DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT = (
    "second person, extra person, different person, different face, duplicate "
    "face, duplicate body, extra arms, extra hands, extra legs, disconnected "
    "limbs, deformed anatomy, identity change, different clothing, new "
    "foreground object, text, watermark, logo"
)

_REVIEW_BOOLEAN_FIELDS = (
    "visible_source_preserved",
    "same_entity_continued",
    "identity_preserved",
    "exactly_one_entity",
    "completion_plausible",
    "completion_useful",
    "no_occluder_reconstructed",
    "no_new_salient_entity",
    "boundary_clean",
    "reference_usable",
)


class QwenReferenceCompletionBackend(Protocol):
    def complete(
        self,
        *,
        input_rgb: Image.Image,
        entity_phrase: str,
        seed: int,
        prompt: str,
        negative_prompt: str,
    ) -> Image.Image: ...


@dataclass(frozen=True)
class QwenImageEdit2511CompletionConfig:
    model_path: Path = DEFAULT_QWEN_MODEL_PATH
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    local_files_only: bool = True
    canvas_expand_ratio: float = 0.75
    lateral_padding_ratio: float = 0.20
    mask_overlap_pixels: int = 0
    model_min_side: int = 1024
    model_multiple: int = 16

    def validate(self) -> None:
        if not isinstance(self.model_path, Path):
            raise TypeError("model_path must be a pathlib.Path")
        if not self.model_path.expanduser().resolve(strict=False).is_dir():
            raise FileNotFoundError(
                f"Qwen Image Edit model path does not exist: {self.model_path}"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if self.local_files_only is not True:
            raise ValueError("Qwen completion requires local_files_only=true")
        if (
            not isinstance(self.num_inference_steps, int)
            or isinstance(self.num_inference_steps, bool)
            or self.num_inference_steps <= 0
        ):
            raise ValueError("num_inference_steps must be a positive integer")
        for name, value in (
            ("true_cfg_scale", self.true_cfg_scale),
            ("guidance_scale", self.guidance_scale),
            ("canvas_expand_ratio", self.canvas_expand_ratio),
            ("lateral_padding_ratio", self.lateral_padding_ratio),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
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


def _supports_explicit_parameter(callable_object: Any, name: str) -> bool:
    return name in inspect.signature(callable_object).parameters


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
        raise RuntimeError(f"installed API lacks {operation} parameters {missing}")


class QwenImageEdit2511ReferenceCompletionBackend:
    """Local whole-image editing backend constrained by hard compositing."""

    def __init__(
        self,
        config: QwenImageEdit2511CompletionConfig,
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
        import torch
        from diffusers import QwenImageEditPlusPipeline

        model_path = self.config.model_path.expanduser().resolve()
        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise RuntimeError(
                f"installed torch does not provide dtype {self.config.dtype}"
            )
        dtype_parameter = (
            "dtype"
            if _supports_explicit_parameter(
                QwenImageEditPlusPipeline.from_pretrained,
                "dtype",
            )
            else "torch_dtype"
        )
        _require_parameters(
            QwenImageEditPlusPipeline.from_pretrained,
            (dtype_parameter, "local_files_only"),
            operation="QwenImageEditPlusPipeline.from_pretrained",
        )
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            str(model_path),
            **{
                dtype_parameter: dtype,
                "local_files_only": True,
            },
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
                "Qwen completion backend previously failed to load"
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
        entity_phrase: str,
        seed: int,
        prompt: str,
        negative_prompt: str,
    ) -> Image.Image:
        if not isinstance(input_rgb, Image.Image) or input_rgb.mode != "RGB":
            raise ValueError("Qwen completion input must be an RGB image")
        if not entity_phrase.strip():
            raise ValueError("entity_phrase must be non-empty")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not prompt.strip() or not negative_prompt.strip():
            raise ValueError("Qwen completion prompts must be non-empty")
        self._ensure_loaded()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("Qwen completion backend is unavailable")

        generator = self._torch.Generator(device=self.config.device).manual_seed(seed)
        parameters = {
            "image": [input_rgb],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": input_rgb.height,
            "width": input_rgb.width,
            "generator": generator,
            "true_cfg_scale": self.config.true_cfg_scale,
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
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
            raise TypeError("Qwen completion backend did not return a PIL image")
        output = images[0]
        if output.mode != "RGB":
            raise TypeError("Qwen completion output must use RGB mode")
        if output.size != input_rgb.size:
            raise RuntimeError(
                "Qwen completion output dimensions do not match model input"
            )
        return output.copy()

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            del pipeline
        torch = self._torch
        self._torch = None
        if (
            torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()


def _validate_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Qwen completion seeds must be non-empty and unique")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("Qwen completion seeds must be non-negative integers")
    return seeds


def _validate_qwen_record(record: CompletionManifestRecord) -> None:
    if record.reference_type != "subject":
        raise ValueError("Qwen completion supports only reference_type=subject")
    if record.completion_mask_path is None:
        raise ValueError("Qwen completion requires completion_mask_path")


def _full_prompt(prompt: str, entity_phrase: str) -> str:
    return f"{prompt.strip()} Entity description: {entity_phrase.strip()}."


def _summary_from_results(
    benchmark_root: Path,
    results: tuple[dict[str, object], ...],
    stats: CompletionBenchmarkStats,
) -> dict[str, object]:
    hard_rejections: Counter[str] = Counter()
    judge_rejections: Counter[str] = Counter()
    for result in results:
        sample_id = str(result["sample_id"])
        attempts = result.get("attempts")
        if not isinstance(attempts, list):
            raise TypeError("completion result attempts must be a list")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise TypeError("completion result attempt must be an object")
            hard_check = attempt.get("hard_check")
            if not isinstance(hard_check, dict):
                raise TypeError("completion hard_check must be an object")
            if hard_check.get("status") == "failed":
                reasons = hard_check.get("reasons")
                if not isinstance(reasons, list):
                    raise TypeError("completion hard-check reasons must be a list")
                hard_rejections.update(str(reason) for reason in reasons)
                continue
            if attempt.get("judge_status") != "reviewed" or (
                attempt.get("judge_verdict") != "reject"
            ):
                continue
            review_path = attempt.get("review_path")
            if not isinstance(review_path, str):
                raise TypeError("completion review_path must be a string")
            review = json.loads(
                (benchmark_root / sample_id / review_path).read_text(
                    encoding="utf-8"
                )
            )
            for field_name in _REVIEW_BOOLEAN_FIELDS:
                if review.get(field_name) is False:
                    judge_rejections[field_name] += 1
    return {
        "backend": QWEN_COMPLETION_BACKEND,
        "processed": stats.processed,
        "accepted": stats.accepted,
        "rejected": stats.rejected,
        "hard_check_rejection_counts": dict(sorted(hard_rejections.items())),
        "judge_rejection_flag_counts": dict(sorted(judge_rejections.items())),
    }


def run_qwen_reference_completion_benchmark(
    *,
    manifest_path: Path,
    benchmark_root: Path,
    config: QwenImageEdit2511CompletionConfig,
    backend: QwenReferenceCompletionBackend,
    judge: ReferenceCompletionJudge,
    prompt: str = DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT,
    negative_prompt: str = DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT,
    seeds: tuple[int, ...] = DEFAULT_QWEN_SEEDS,
) -> CompletionBenchmarkStats:
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    if not prompt:
        raise ValueError("Qwen completion prompt must be non-empty")
    if not negative_prompt:
        raise ValueError("Qwen completion negative prompt must be non-empty")
    seeds = _validate_seeds(seeds)

    def candidate_factory(
        record: CompletionManifestRecord,
    ) -> tuple[CompletionCandidateSpec, ...]:
        final_prompt = _full_prompt(prompt, record.entity_phrase)
        return tuple(
            CompletionCandidateSpec(
                file_stem=f"qwen_seed_{seed}",
                seed=seed,
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                attempt_metadata={
                    "backend": QWEN_COMPLETION_BACKEND,
                    "candidate_id": f"qwen_seed_{seed}",
                    "seed": seed,
                    "prompt": final_prompt,
                    "negative_prompt": negative_prompt,
                },
                selection_metadata={
                    "backend": QWEN_COMPLETION_BACKEND,
                    "candidate_id": f"qwen_seed_{seed}",
                    "seed": seed,
                },
            )
            for seed in seeds
        )

    def generate_candidate(
        *,
        record: CompletionManifestRecord,
        candidate: CompletionCandidateSpec,
        input_rgb: Image.Image,
        completion_mask: Image.Image,
    ) -> Image.Image:
        # Qwen edits the whole RGB image. The mask is intentionally not forwarded.
        del completion_mask
        return backend.complete(
            input_rgb=input_rgb,
            entity_phrase=record.entity_phrase,
            seed=candidate.seed,
            prompt=candidate.prompt,
            negative_prompt=candidate.negative_prompt,
        )

    run = run_completion_benchmark(
        manifest_path=manifest_path,
        benchmark_root=benchmark_root,
        allowed_benchmark_root=ALLOWED_QWEN_BENCHMARK_ROOT,
        config=config,
        judge=judge,
        candidate_factory=candidate_factory,
        generate_candidate=generate_candidate,
        record_validator=_validate_qwen_record,
        result_metadata={"backend": QWEN_COMPLETION_BACKEND},
    )
    resolved_root = benchmark_root.expanduser().resolve()
    write_json_atomic(
        resolved_root / "benchmark_summary.json",
        _summary_from_results(resolved_root, run.results, run.stats),
    )
    return run.stats
