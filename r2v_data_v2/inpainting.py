from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw

from prompts.qwen_background_fill_prompt import BACKGROUND_FILL_PROMPT
from prompts.qwen_background_inpainting_review_prompt import (
    BACKGROUND_INPAINTING_LOCAL_REVIEW_PROMPT,
    BACKGROUND_INPAINTING_REVIEW_PROMPT,
)
from prompts.qwen_inpainting_consistency_prompt import (
    INPAINTING_CONSISTENCY_PROMPT,
)
from r2v_data_v2.config import (
    InpaintingConfig,
    PipelineConfig,
    QwenConfig,
    _qwen_services,
    has_background_inpainting_repair_judge,
    has_inpainting_semantic_validator,
)
from r2v_data_v2.image_utils import (
    image_data_uri,
    image_extension,
)
from r2v_data_v2.mask_utils import save_mask_png
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic
from r2v_data_v2.schemas import (
    BackgroundFillPrompt,
    BackgroundInpaintingReview,
    InpaintingSemanticReview,
    MaskedContentOutcome,
)
from r2v_data_v2.semantic_alignment import Siglip2Aligner
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    request_structured_output,
)
from r2v_data_v2.visual_embedding import (
    DinoV3Embedder,
    embedding_cosine_similarity,
    save_selected_dinov3_embedding,
)

BACKGROUND_INPAINT_PROMPT = (
    "Continuous background scenery and textures matching the surrounding "
    "environment, perspective, lighting, color, depth, texture, and camera style"
)
BACKGROUND_INPAINT_SUFFIX = (
    "Seamless natural continuation of the surrounding scene, with consistent "
    "geometry, perspective, scale, texture, color, lighting, depth, and camera "
    "characteristics."
)
BACKGROUND_FILL_BANNED_WORDS = frozenset(
    {
        "remove",
        "replace",
        "erase",
        "without",
        "no",
        "empty",
        "foreground",
        "object",
        "mask",
        "hole",
    }
)
BACKGROUND_FILL_FOREGROUND_WORDS = frozenset(
    {
        "animal",
        "animals",
        "airplane",
        "aircraft",
        "bicycle",
        "bike",
        "bird",
        "body",
        "boat",
        "bus",
        "car",
        "cat",
        "child",
        "dog",
        "face",
        "fish",
        "horse",
        "logo",
        "motorcycle",
        "person",
        "people",
        "product",
        "ship",
        "sign",
        "subject",
        "train",
        "truck",
        "vehicle",
        "vessel",
        "whale",
    }
)
BACKGROUND_COMPARISON_VERSION = "1"

ENTITY_INPAINT_PROMPT = (
    "Restore only the masked damaged or occluded region of {description}. "
    "Preserve the exact same entity identity, shape, materials, colors, "
    "viewpoint, lighting, scale, and image style. Do not modify unmasked "
    "regions and do not add new parts beyond what is necessary to complete "
    "the local damage."
)


def _consistency_review_prompt(reference_phrase: str, mode: str) -> str:
    if mode == "background_hole_fill":
        return BACKGROUND_INPAINTING_REVIEW_PROMPT.format(
            reference_phrase=reference_phrase
        )
    if mode == "background_hole_fill_local":
        return BACKGROUND_INPAINTING_LOCAL_REVIEW_PROMPT.format(
            reference_phrase=reference_phrase
        )
    if mode == "entity_local_repair":
        return INPAINTING_CONSISTENCY_PROMPT.format(
            reference_phrase=reference_phrase,
            repair_mode=mode,
        )
    raise ValueError(f"unsupported inpainting repair mode: {mode}")


INPAINTING_SOURCE_METADATA_VERSION = "6"


class InpaintBackend(Protocol):
    def inpaint(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image: ...


class ConsistencyValidator(Protocol):
    def __call__(
        self,
        *,
        original: Image.Image,
        repaired: Image.Image,
        repair_mask: Image.Image,
        reference: dict[str, object],
        mode: str,
    ) -> dict[str, object]: ...


class InpaintingSemanticJudge(Protocol):
    def review(
        self,
        *,
        original_path: Path,
        repaired_path: Path,
        repair_mask_path: Path,
        comparison_sheet_path: Path | None = None,
        reference_phrase: str,
        mode: str,
    ) -> InpaintingSemanticReview | BackgroundInpaintingReview: ...


class BackgroundFillPromptGenerator(Protocol):
    def generate(
        self,
        *,
        original_path: Path,
        generation_mask_path: Path,
        forbidden_texts: list[str],
    ) -> tuple[BackgroundFillPrompt, dict[str, object]]: ...


class InpaintingDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class InpaintingStats:
    processed: int = 0
    skipped_disabled: int = 0
    skipped_no_repair_needed: int = 0
    repaired: int = 0
    fallback_to_raw: int = 0
    rejected: int = 0
    failed: int = 0


class NoOpInpaintBackend:
    def inpaint(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        del mask, prompt, seed
        return image.copy()


class Flux1FillBackend:
    def __init__(self, config: InpaintingConfig) -> None:
        self.config = config
        self._pipeline: object | None = None
        self._torch: object | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        model_path = self.config.model_path
        if model_path is None:
            raise ValueError("inpainting.model_path is required for FLUX.1 Fill")
        resolved = model_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"FLUX.1 Fill model path does not exist: {resolved}"
            )
        try:
            import torch
            from diffusers import FluxFillPipeline
        except ImportError as exc:
            raise InpaintingDependencyError(
                "FLUX.1 Fill dependencies are unavailable; install "
                "requirements-inpaint.txt without replacing the server torch build"
            ) from exc
        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {self.config.dtype}")
        self._pipeline = FluxFillPipeline.from_pretrained(
            str(resolved),
            torch_dtype=dtype,
            local_files_only=True,
        )
        self._pipeline.to(self.config.device)
        self._torch = torch

    def inpaint(
        self,
        *,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        self._load()
        if self._pipeline is None or self._torch is None:
            raise RuntimeError("FLUX.1 Fill failed to initialize")
        source = image.convert("RGB")
        source_mask = mask.convert("L")
        if source.size != source_mask.size:
            raise ValueError("FLUX image and mask dimensions must match")
        width, height = source.size
        padded_width = ((width + 15) // 16) * 16
        padded_height = ((height + 15) // 16) * 16
        image_array = np.asarray(source)
        mask_array = np.asarray(source_mask)
        padded_image = Image.fromarray(
            np.pad(
                image_array,
                (
                    (0, padded_height - height),
                    (0, padded_width - width),
                    (0, 0),
                ),
                mode="edge",
            )
        )
        padded_mask = Image.fromarray(
            np.pad(
                mask_array,
                (
                    (0, padded_height - height),
                    (0, padded_width - width),
                ),
                mode="constant",
                constant_values=0,
            )
        )
        generator = self._torch.Generator(device=self.config.device).manual_seed(
            seed
        )
        result = self._pipeline(
            prompt=prompt,
            image=padded_image,
            mask_image=padded_mask,
            height=padded_height,
            width=padded_width,
            generator=generator,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            max_sequence_length=self.config.max_sequence_length,
        )
        if not getattr(result, "images", None):
            raise RuntimeError("FLUX.1 Fill returned no image")
        generated = result.images[0].convert("RGB")
        if generated.size != (padded_width, padded_height):
            raise ValueError("FLUX.1 Fill returned unexpected padded dimensions")
        return generated.crop((0, 0, width, height))


def _background_fill_prompt_issues(
    result: BackgroundFillPrompt,
    *,
    forbidden_texts: list[str],
) -> list[ValidationIssue]:
    prompt = result.fill_prompt.strip()
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", prompt)
    issues: list[ValidationIssue] = []
    if not 12 <= len(words) <= 35:
        issues.append(
            ValidationIssue(
                code="background_fill_word_count",
                field="fill_prompt",
                message="fill_prompt must contain 12-35 English words",
            )
        )
    if re.sub(r"[A-Za-z\s,.;:'-]", "", prompt):
        issues.append(
            ValidationIssue(
                code="background_fill_non_english",
                field="fill_prompt",
                message="fill_prompt must use English text only",
            )
        )
    tokens = {word.casefold() for word in words}
    prohibited = sorted(
        tokens & (BACKGROUND_FILL_BANNED_WORDS | BACKGROUND_FILL_FOREGROUND_WORDS)
    )
    if prohibited:
        issues.append(
            ValidationIssue(
                code="background_fill_prohibited_terms",
                field="fill_prompt",
                message=f"prohibited terms: {', '.join(prohibited)}",
            )
        )
    normalized_prompt = " ".join(words).casefold()
    for forbidden in forbidden_texts:
        forbidden_words = re.findall(r"[A-Za-z]+", forbidden)
        normalized_forbidden = " ".join(forbidden_words).casefold()
        if len(forbidden_words) >= 2 and normalized_forbidden in normalized_prompt:
            issues.append(
                ValidationIssue(
                    code="background_fill_copied_source_text",
                    field="fill_prompt",
                    message="fill_prompt copies caption or reference text",
                )
            )
            break
        forbidden_tokens = [
            word.casefold() for word in forbidden_words
        ]
        if len(forbidden_tokens) >= 4 and any(
            " ".join(forbidden_tokens[index : index + 4])
            in normalized_prompt
            for index in range(len(forbidden_tokens) - 3)
        ):
            issues.append(
                ValidationIssue(
                    code="background_fill_copied_source_text",
                    field="fill_prompt",
                    message="fill_prompt copies caption or reference text",
                )
            )
            break
        source_tokens = {
            word.casefold()
            for word in forbidden_words
            if len(word) >= 4
        }
        copied_foreground = sorted(
            tokens
            & source_tokens
            & BACKGROUND_FILL_FOREGROUND_WORDS
        )
        if copied_foreground:
            issues.append(
                ValidationIssue(
                    code="background_fill_copied_foreground",
                    field="fill_prompt",
                    message=(
                        "fill_prompt copied foreground terms: "
                        f"{', '.join(copied_foreground)}"
                    ),
                )
            )
            break
    return issues


class QwenBackgroundFillPromptGenerator:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(
        self,
        *,
        prompt: str,
        original_path: Path,
        generation_mask_path: Path,
    ) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri(original_path)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri(generation_mask_path)
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": min(384, self.config.max_tokens),
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": BackgroundFillPrompt.__name__,
                        "strict": True,
                        "schema": BackgroundFillPrompt.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty background fill prompt")
        return content

    def generate(
        self,
        *,
        original_path: Path,
        generation_mask_path: Path,
        forbidden_texts: list[str],
    ) -> tuple[BackgroundFillPrompt, dict[str, object]]:
        result = request_structured_output(
            request=lambda request_text: self._request(
                prompt=request_text,
                original_path=original_path,
                generation_mask_path=generation_mask_path,
            ),
            original_request=BACKGROUND_FILL_PROMPT,
            model=BackgroundFillPrompt,
            validate=lambda value: _background_fill_prompt_issues(
                value,
                forbidden_texts=forbidden_texts,
            ),
        )
        return result, {
            "model": self.config.model,
            "request_prompt_sha256": _sha256_text(BACKGROUND_FILL_PROMPT),
            "validation": "passed",
        }


class QwenInpaintingConsistencyJudge:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(
        self,
        *,
        prompt: str,
        original_path: Path,
        repaired_path: Path,
        repair_mask_path: Path,
        comparison_sheet_path: Path | None,
        review_model: (
            type[InpaintingSemanticReview | BackgroundInpaintingReview]
        ),
    ) -> str:
        content: list[dict[str, object]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri(original_path)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri(repaired_path)},
            },
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri(repair_mask_path)},
            },
        ]
        if comparison_sheet_path is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri(comparison_sheet_path)
                    },
                }
            )
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": min(512, self.config.max_tokens),
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": review_model.__name__,
                        "strict": True,
                        "schema": review_model.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty inpainting review")
        return content

    def review(
        self,
        *,
        original_path: Path,
        repaired_path: Path,
        repair_mask_path: Path,
        comparison_sheet_path: Path | None = None,
        reference_phrase: str,
        mode: str,
    ) -> InpaintingSemanticReview | BackgroundInpaintingReview:
        prompt = _consistency_review_prompt(reference_phrase, mode)
        review_model = (
            BackgroundInpaintingReview
            if mode
            in {"background_hole_fill", "background_hole_fill_local"}
            else InpaintingSemanticReview
        )
        return request_structured_output(
            request=lambda request_text: self._request(
                prompt=request_text,
                original_path=original_path,
                repaired_path=repaired_path,
                repair_mask_path=repair_mask_path,
                comparison_sheet_path=comparison_sheet_path,
                review_model=review_model,
            ),
            original_request=prompt,
            model=review_model,
        )


def _background_review_passes(
    review: BackgroundInpaintingReview,
    *,
    require_reference_phrase: bool = True,
) -> bool:
    return (
        review.masked_content_outcome
        == MaskedContentOutcome.REMOVED_TO_BACKGROUND
        and review.background_continuity_preserved
        and (
            review.reference_phrase_supported
            or not require_reference_phrase
        )
        and not review.artifact_types
    )


def _background_review_payload(
    review: BackgroundInpaintingReview,
) -> dict[str, object]:
    payload = review.model_dump(mode="json")
    outcome = review.masked_content_outcome
    payload.update(
        {
            "masked_foreground_removed": (
                outcome == MaskedContentOutcome.REMOVED_TO_BACKGROUND
            ),
            "new_salient_objects": (
                outcome == MaskedContentOutcome.REPLACED_BY_NEW_OBJECT
            ),
            "visible_seam_or_artifact": bool(review.artifact_types),
        }
    )
    return payload


def _mask_boundary(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    kernel_size = max(3, (2 * pixels) + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(source.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(source.astype(np.uint8), kernel).astype(bool)
    return dilated & ~eroded


def _red_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    result[mask] = (0.55 * result[mask]) + (
        0.45 * np.asarray([255, 0, 0])
    )
    return np.rint(result).astype(np.uint8)


def _background_comparison_sheet(
    *,
    original: Image.Image,
    repaired: Image.Image,
    generation_mask: Image.Image,
) -> Image.Image:
    original_array = np.asarray(original.convert("RGB"))
    repaired_array = np.asarray(repaired.convert("RGB"))
    mask = np.asarray(generation_mask.convert("L")) >= 128
    if (
        original_array.shape != repaired_array.shape
        or original_array.shape[:2] != mask.shape
    ):
        raise ValueError("comparison images and mask must have matching sizes")
    original_overlay = _red_mask_overlay(original_array, mask)
    repaired_overlay = repaired_array.copy()
    repaired_overlay[_mask_boundary(mask, pixels=2)] = (255, 0, 0)
    difference = np.abs(
        original_array.astype(np.int16) - repaired_array.astype(np.int16)
    ).mean(axis=2)
    heatmap = np.zeros_like(original_array)
    heatmap[..., 0] = np.clip(difference * 3.0, 0, 255).astype(np.uint8)
    heatmap[..., 1] = np.clip(
        (difference - 32.0) * 2.0, 0, 255
    ).astype(np.uint8)
    heatmap[~mask] = 0
    boundary = _mask_boundary(mask, pixels=5)
    height, width = mask.shape
    coordinates = np.argwhere(boundary)
    if coordinates.size:
        y0, x0 = coordinates.min(axis=0)
        y1, x1 = coordinates.max(axis=0) + 1
        context = max(2, round(0.15 * max(y1 - y0, x1 - x0)))
        crop_box = (
            max(0, int(x0) - context),
            max(0, int(y0) - context),
            min(width, int(x1) + context),
            min(height, int(y1) + context),
        )
    else:
        crop_box = (0, 0, width, height)
    half_width = max(1, width // 2)
    boundary_panels = []
    for image_array in (original_array, repaired_array):
        overlay = Image.fromarray(_red_mask_overlay(image_array, boundary))
        boundary_panels.append(
            overlay.crop(crop_box).resize(
                (half_width, height),
                Image.Resampling.NEAREST,
            )
        )
    boundary_comparison = Image.new(
        "RGB",
        (half_width * 2, height),
        (24, 24, 24),
    )
    boundary_comparison.paste(boundary_panels[0], (0, 0))
    boundary_comparison.paste(boundary_panels[1], (half_width, 0))
    cell_width = width
    sheet = Image.new("RGB", (cell_width * 2, height * 2), (24, 24, 24))
    panels = (
        Image.fromarray(original_overlay),
        Image.fromarray(repaired_overlay),
        Image.fromarray(heatmap),
        boundary_comparison,
    )
    for index, panel in enumerate(panels):
        x = (index % 2) * cell_width
        y = (index // 2) * height
        sheet.paste(panel.resize((cell_width, height)), (x, y))
    draw = ImageDraw.Draw(sheet)
    labels = (
        "ORIGINAL + MASK",
        "REPAIRED + CONTOUR",
        "MASKED ABS DIFFERENCE",
        "BOUNDARY RING: ORIGINAL | REPAIRED",
    )
    for index, label in enumerate(labels):
        x = (index % 2) * cell_width + 6
        y = (index // 2) * height + 5
        draw.rectangle((x - 3, y - 2, x + 230, y + 12), fill=(0, 0, 0))
        draw.text((x, y), label, fill=(255, 255, 255))
    return sheet


def _background_pixel_diagnostics(
    *,
    original: np.ndarray,
    repaired: np.ndarray,
    source_mask: np.ndarray,
    generation_mask: np.ndarray,
    destination: Path,
) -> dict[str, object]:
    if original.shape != repaired.shape:
        raise ValueError("diagnostic images must have matching sizes")
    pixel_l1 = np.abs(
        original.astype(np.float32) - repaired.astype(np.float32)
    ).mean(axis=2)
    source = np.asarray(source_mask, dtype=bool)
    generation = np.asarray(generation_mask, dtype=bool)
    boundary = _mask_boundary(generation, pixels=3)
    heatmap = np.zeros_like(original)
    heatmap[..., 0] = np.clip(pixel_l1 * 3.0, 0, 255).astype(np.uint8)
    heatmap[..., 1] = np.clip(
        (pixel_l1 - 32.0) * 2.0, 0, 255
    ).astype(np.uint8)
    heatmap[~generation] = 0
    _save_lossless_atomic(Image.fromarray(heatmap), destination)
    return {
        "masked_mean_l1": (
            float(pixel_l1[source].mean()) if source.any() else 0.0
        ),
        "masked_changed_pixel_ratio": (
            float((pixel_l1[source] > 12.0).mean()) if source.any() else 0.0
        ),
        "generation_mask_changed_pixel_ratio": (
            float((pixel_l1[generation] > 12.0).mean())
            if generation.any()
            else 0.0
        ),
        "boundary_ring_mean_l1": (
            float(pixel_l1[boundary].mean()) if boundary.any() else 0.0
        ),
        "difference_heatmap_path": str(destination),
    }


def _background_local_review_images(
    *,
    original: Image.Image,
    repaired: Image.Image,
    generation_mask: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int, int, int]]:
    original_rgb = original.convert("RGB")
    repaired_rgb = repaired.convert("RGB")
    mask_l = generation_mask.convert("L")
    if original_rgb.size != repaired_rgb.size or original_rgb.size != mask_l.size:
        raise ValueError("background review images and mask must have matching sizes")
    mask_array = np.asarray(mask_l) >= 128
    coordinates = np.argwhere(mask_array)
    if coordinates.size == 0:
        raise ValueError("background local review requires a non-empty mask")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    box_width = int(x1 - x0)
    box_height = int(y1 - y0)
    context_x = max(1, round(0.25 * box_width))
    context_y = max(1, round(0.25 * box_height))
    width, height = original_rgb.size
    crop_box = (
        max(0, int(x0) - context_x),
        max(0, int(y0) - context_y),
        min(width, int(x1) + context_x),
        min(height, int(y1) + context_y),
    )
    original_crop = original_rgb.crop(crop_box)
    repaired_crop = repaired_rgb.crop(crop_box)
    mask_crop = mask_l.crop(crop_box)
    crop_width, crop_height = original_crop.size
    long_side = max(crop_width, crop_height)
    if long_side < 768:
        scale = 768 / long_side
        resized = (
            max(1, round(crop_width * scale)),
            max(1, round(crop_height * scale)),
        )
        original_crop = original_crop.resize(resized, Image.Resampling.LANCZOS)
        repaired_crop = repaired_crop.resize(resized, Image.Resampling.LANCZOS)
        mask_crop = mask_crop.resize(resized, Image.Resampling.NEAREST)
    return original_crop, repaired_crop, mask_crop, crop_box


def _background_local_review_components(
    *,
    original: Image.Image,
    repaired: Image.Image,
    generation_mask: Image.Image,
) -> list[
    tuple[
        int,
        Image.Image,
        Image.Image,
        Image.Image,
        tuple[int, int, int, int],
    ]
]:
    mask_l = generation_mask.convert("L")
    mask_array = np.asarray(mask_l) >= 128
    component_count, labels = cv2.connectedComponents(
        mask_array.astype(np.uint8),
        connectivity=8,
    )
    components = []
    for component_index, label in enumerate(range(1, component_count)):
        component_mask = labels == label
        (
            local_original,
            local_repaired,
            local_mask,
            crop_box,
        ) = _background_local_review_images(
            original=original,
            repaired=repaired,
            generation_mask=Image.fromarray(
                component_mask.astype(np.uint8) * 255
            ),
        )
        components.append(
            (
                component_index,
                local_original,
                local_repaired,
                local_mask,
                crop_box,
            )
        )
    if not components:
        raise ValueError("background local review requires a non-empty mask")
    return components


class ProductionConsistencyValidator:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        dino_embedder: DinoV3Embedder | None = None,
        siglip_aligner: Siglip2Aligner | None = None,
        qwen_judge: InpaintingSemanticJudge | None = None,
    ) -> None:
        self.config = config
        self.dino = dino_embedder
        self.siglip = siglip_aligner
        self.qwen = qwen_judge
        self._owns_dino = False
        self._owns_siglip = False

    def _ensure_models(self) -> list[str]:
        unavailable: list[str] = []
        if self.dino is None and self.config.ranking.evaluators.dinov3.enabled:
            try:
                self.dino = DinoV3Embedder(self.config.ranking)
                self._owns_dino = True
            except Exception:  # noqa: BLE001
                unavailable.append("dinov3_validator_unavailable")
        if self.siglip is None and self.config.ranking.evaluators.siglip2.enabled:
            try:
                self.siglip = Siglip2Aligner(
                    self.config.ranking.siglip2_model_path,
                    self.config.ranking.siglip2_batch_size,
                )
                self._owns_siglip = True
            except Exception:  # noqa: BLE001
                unavailable.append("siglip_validator_unavailable")
        if self.qwen is None:
            repair_config = _qwen_services(self.config.qwen).repair_judge
            if repair_config is not None:
                self.qwen = QwenInpaintingConsistencyJudge(repair_config)
        if self.dino is not None and self.siglip is not None:
            return []
        if self.qwen is not None:
            return []
        if self.dino is None:
            unavailable.append("dinov3_validator_unavailable")
        if self.siglip is None:
            unavailable.append("siglip_validator_unavailable")
        return list(dict.fromkeys(unavailable))

    def __call__(
        self,
        *,
        original: Image.Image,
        repaired: Image.Image,
        repair_mask: Image.Image,
        reference: dict[str, object],
        mode: str,
        diagnostics_dir: Path | None = None,
    ) -> dict[str, object]:
        reasons = self._ensure_models()
        dino_similarity: float | None = None
        raw_siglip: float | None = None
        repaired_siglip: float | None = None
        phrase = str(
            reference.get("phrase")
            or reference.get("canonical_label")
            or "reference image"
        )
        semantic_original = original
        semantic_repaired = repaired
        semantic_input = "full_frame"
        if mode == "entity_local_repair":
            semantic_original, semantic_repaired = _entity_neutral_images(
                original=original,
                repaired=repaired,
                repair_mask=repair_mask,
                reference=reference,
            )
            semantic_input = "masked_neutral_crop"
        if self.dino is not None:
            try:
                embeddings = self.dino.encode(
                    [semantic_original, semantic_repaired]
                )
                if len(embeddings) != 2:
                    raise RuntimeError("unexpected DINO embedding count")
                dino_similarity = embedding_cosine_similarity(
                    embeddings[0],
                    embeddings[1],
                )
                if mode == "entity_local_repair" and (
                    dino_similarity
                    < self.config.inpainting.consistency.minimum_dino_similarity
                ):
                    reasons.append("dino_similarity")
            except Exception:  # noqa: BLE001
                reasons.append("dinov3_validation_failed")
        if self.siglip is not None:
            try:
                scores = self.siglip.score(
                    [semantic_original, semantic_repaired],
                    phrase,
                    [],
                )
                if len(scores) != 2:
                    raise RuntimeError("unexpected SigLIP score count")
                raw_siglip = float(scores[0].target_similarity)
                repaired_siglip = float(scores[1].target_similarity)
                if mode == "entity_local_repair" and (
                    repaired_siglip
                    < self.config.inpainting.consistency.minimum_siglip_similarity
                ):
                    reasons.append("siglip_similarity")
                if (
                    raw_siglip - repaired_siglip
                    > self.config.inpainting.consistency.maximum_siglip_similarity_drop
                ):
                    reasons.append("siglip_similarity_drop")
            except Exception:  # noqa: BLE001
                reasons.append("siglip_validation_failed")

        qwen_review: dict[str, object] | None = None
        qwen_full_review: dict[str, object] | None = None
        qwen_local_reviews: list[dict[str, object]] = []
        qwen_local_review: dict[str, object] | None = None
        qwen_local_crop_box: tuple[int, int, int, int] | None = None
        comparison_sheet_paths: list[str] = []
        comparison_sheet_fingerprints: list[str] = []
        if self.qwen is not None:
            temporary_root = Path(
                str(reference.get("raw_canonical_path", "."))
            ).parent
            original_path = temporary_root / ".inpainting_qwen_original.png"
            repaired_path = temporary_root / ".inpainting_qwen_repaired.png"
            repair_mask_path = (
                temporary_root / ".inpainting_qwen_repair_mask.png"
            )
            local_original_path = (
                temporary_root / ".inpainting_qwen_local_original.png"
            )
            local_repaired_path = (
                temporary_root / ".inpainting_qwen_local_repaired.png"
            )
            local_mask_path = (
                temporary_root / ".inpainting_qwen_local_mask.png"
            )
            full_comparison_path = (
                diagnostics_dir / "comparison_full.png"
                if diagnostics_dir is not None
                else temporary_root / ".inpainting_qwen_comparison.png"
            )
            temporary_comparison_paths: list[Path] = []
            try:
                _save_lossless_atomic(original, original_path)
                _save_lossless_atomic(repaired, repaired_path)
                _save_lossless_atomic(
                    repair_mask.convert("L"),
                    repair_mask_path,
                )
                if mode == "background_hole_fill":
                    if diagnostics_dir is not None:
                        diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    _save_lossless_atomic(
                        _background_comparison_sheet(
                            original=original,
                            repaired=repaired,
                            generation_mask=repair_mask,
                        ),
                        full_comparison_path,
                    )
                    comparison_sheet_paths.append(str(full_comparison_path))
                    comparison_sheet_fingerprints.append(
                        _sha256_path(full_comparison_path)
                    )
                    if diagnostics_dir is None:
                        temporary_comparison_paths.append(full_comparison_path)
                    full_passes = False
                    local_passes = True
                    try:
                        full_review = self.qwen.review(
                            original_path=original_path,
                            repaired_path=repaired_path,
                            repair_mask_path=repair_mask_path,
                            comparison_sheet_path=full_comparison_path,
                            reference_phrase=phrase,
                            mode=mode,
                        )
                        if isinstance(
                            full_review,
                            BackgroundInpaintingReview,
                        ):
                            qwen_full_review = _background_review_payload(
                                full_review
                            )
                            full_passes = _background_review_passes(
                                full_review
                            )
                        else:
                            reasons.append("qwen_full_review_schema")
                    except Exception:  # noqa: BLE001
                        reasons.append("qwen_full_validation_failed")
                    try:
                        local_components = _background_local_review_components(
                            original=original,
                            repaired=repaired,
                            generation_mask=repair_mask,
                        )
                    except Exception:  # noqa: BLE001
                        local_components = []
                        local_passes = False
                        reasons.append("qwen_local_validation_failed")
                    for (
                        component_index,
                        local_original,
                        local_repaired,
                        local_mask,
                        crop_box,
                    ) in local_components:
                        local_record: dict[str, object] = {
                            "component_index": component_index,
                            "crop_box": list(crop_box),
                            "review": None,
                        }
                        local_comparison_path = (
                            diagnostics_dir
                            / f"comparison_local_{component_index:02d}.png"
                            if diagnostics_dir is not None
                            else temporary_root
                            / (
                                ".inpainting_qwen_local_comparison_"
                                f"{component_index:02d}.png"
                            )
                        )
                        try:
                            _save_lossless_atomic(
                                local_original,
                                local_original_path,
                            )
                            _save_lossless_atomic(
                                local_repaired,
                                local_repaired_path,
                            )
                            _save_lossless_atomic(local_mask, local_mask_path)
                            _save_lossless_atomic(
                                _background_comparison_sheet(
                                    original=local_original,
                                    repaired=local_repaired,
                                    generation_mask=local_mask,
                                ),
                                local_comparison_path,
                            )
                            local_comparison_fingerprint = _sha256_path(
                                local_comparison_path
                            )
                            comparison_sheet_paths.append(
                                str(local_comparison_path)
                            )
                            comparison_sheet_fingerprints.append(
                                local_comparison_fingerprint
                            )
                            if diagnostics_dir is None:
                                temporary_comparison_paths.append(
                                    local_comparison_path
                                )
                            local_review = self.qwen.review(
                                original_path=local_original_path,
                                repaired_path=local_repaired_path,
                                repair_mask_path=local_mask_path,
                                comparison_sheet_path=local_comparison_path,
                                reference_phrase=phrase,
                                mode="background_hole_fill_local",
                            )
                            if isinstance(
                                local_review,
                                BackgroundInpaintingReview,
                            ):
                                local_record["review"] = (
                                    _background_review_payload(local_review)
                                )
                                if not _background_review_passes(
                                    local_review,
                                    require_reference_phrase=False,
                                ):
                                    local_passes = False
                            else:
                                local_passes = False
                                reasons.append("qwen_local_review_schema")
                        except Exception:  # noqa: BLE001
                            local_passes = False
                            reasons.append("qwen_local_validation_failed")
                        qwen_local_reviews.append(local_record)
                    if len(qwen_local_reviews) == 1:
                        only_local = qwen_local_reviews[0]
                        only_review = only_local["review"]
                        if isinstance(only_review, dict):
                            qwen_local_review = only_review
                        qwen_local_crop_box = tuple(
                            int(value)
                            for value in only_local["crop_box"]
                        )
                    if not (full_passes and local_passes):
                        reasons.append("qwen_background_consistency")
                else:
                    review = self.qwen.review(
                        original_path=original_path,
                        repaired_path=repaired_path,
                        repair_mask_path=repair_mask_path,
                        reference_phrase=phrase,
                        mode=mode,
                    )
                    qwen_review = review.model_dump(mode="json")
                    if not isinstance(
                        review,
                        InpaintingSemanticReview,
                    ) or not (
                        review.same_semantic_content
                        and review.identity_preserved
                        and review.reference_phrase_supported
                        and not review.new_salient_objects
                    ):
                        reasons.append("qwen_semantic_consistency")
            except Exception:  # noqa: BLE001
                reasons.append("qwen_validation_failed")
            finally:
                original_path.unlink(missing_ok=True)
                repaired_path.unlink(missing_ok=True)
                repair_mask_path.unlink(missing_ok=True)
                local_original_path.unlink(missing_ok=True)
                local_repaired_path.unlink(missing_ok=True)
                local_mask_path.unlink(missing_ok=True)
                for path in temporary_comparison_paths:
                    path.unlink(missing_ok=True)

        reasons = list(dict.fromkeys(reasons))
        return {
            "accepted": not reasons,
            "dino_similarity": dino_similarity,
            "raw_siglip_similarity": raw_siglip,
            "repaired_siglip_similarity": repaired_siglip,
            "siglip_similarity_drop": (
                raw_siglip - repaired_siglip
                if raw_siglip is not None and repaired_siglip is not None
                else None
            ),
            "qwen_review": qwen_review,
            "qwen_full_review": qwen_full_review,
            "qwen_local_reviews": qwen_local_reviews,
            "qwen_local_review": qwen_local_review,
            "qwen_local_crop_box": qwen_local_crop_box,
            "comparison_sheet_paths": comparison_sheet_paths,
            "comparison_sheet_fingerprints": comparison_sheet_fingerprints,
            "comparison_sheet_version": BACKGROUND_COMPARISON_VERSION,
            "semantic_input": semantic_input,
            "rejection_reasons": reasons,
        }

    def close(self) -> None:
        if self._owns_dino and self.dino is not None:
            self.dino.close()
        if self._owns_siglip and self.siglip is not None:
            self.siglip.close()


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_lossless_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        # Lossless storage is required for the strict unmasked-pixel contract.
        image.save(temporary, format="PNG")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_jpeg_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        image.save(temporary, format="JPEG", quality=95)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config_fingerprint(config: PipelineConfig) -> str:
    repair_judge = _qwen_services(config.qwen).repair_judge
    payload = {
        "inpainting": asdict(config.inpainting),
        "ranking_evaluators": asdict(config.ranking.evaluators),
        "dinov3_repo_dir": config.ranking.dinov3_repo_dir,
        "dinov3_model_path": config.ranking.dinov3_model_path,
        "dinov3_model_name": config.ranking.dinov3_model_name,
        "siglip2_model_path": config.ranking.siglip2_model_path,
        "repair_judge": (
            asdict(repair_judge) if repair_judge is not None else None
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_signature(
    *,
    reference: dict[str, object],
    source_image_path: Path,
    source_mask_path: Path,
    config_fingerprint: str,
) -> dict[str, object]:
    reference_phrase = str(reference.get("phrase") or "")
    canonical_label = str(reference.get("canonical_label") or "")
    reference_type = str(reference.get("reference_type") or "entity")
    mode = (
        "background_hole_fill"
        if reference_type == "background"
        else "entity_local_repair"
    )
    repair_prompt = _inpainting_prompt(reference, mode)
    consistency_prompt = _consistency_review_prompt(
        reference_phrase or canonical_label or "reference image",
        mode,
    )
    local_consistency_prompt = (
        _consistency_review_prompt(
            reference_phrase or canonical_label or "reference image",
            "background_hole_fill_local",
        )
        if mode == "background_hole_fill"
        else None
    )
    prompt_payload = json.dumps(
        {
            "repair_prompt": repair_prompt,
            "consistency_prompt": consistency_prompt,
            "local_consistency_prompt": local_consistency_prompt,
            "background_fill_prompt_request": (
                BACKGROUND_FILL_PROMPT
                if mode == "background_hole_fill"
                else None
            ),
            "background_comparison_version": BACKGROUND_COMPARISON_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    version_fingerprint = _sha256_text(
        f"r2v_data_v2.inpainting.source_metadata."
        f"{INPAINTING_SOURCE_METADATA_VERSION}"
    )
    return {
        "source_image_sha256": _sha256_path(source_image_path),
        "mask_sha256": _sha256_path(source_mask_path),
        "source_frame_index": reference.get("source_frame_index"),
        "config_fingerprint": config_fingerprint,
        "reference_phrase": reference_phrase,
        "canonical_label": canonical_label,
        "reference_type": reference_type,
        "inpainting_prompt_sha256": _sha256_text(repair_prompt),
        "consistency_prompt_sha256": _sha256_text(consistency_prompt),
        "local_consistency_prompt_sha256": (
            _sha256_text(local_consistency_prompt)
            if local_consistency_prompt is not None
            else None
        ),
        "prompt_fingerprint": _sha256_text(prompt_payload),
        "background_fill_prompt_request_sha256": (
            _sha256_text(BACKGROUND_FILL_PROMPT)
            if mode == "background_hole_fill"
            else None
        ),
        "background_comparison_version": (
            BACKGROUND_COMPARISON_VERSION
            if mode == "background_hole_fill"
            else None
        ),
        "source_metadata_version": INPAINTING_SOURCE_METADATA_VERSION,
        "version_fingerprint": version_fingerprint,
    }


def _metadata_matches_source(
    metadata: dict[str, object],
    signature: dict[str, object],
    reference: dict[str, object],
) -> bool:
    if any(metadata.get(key) != value for key, value in signature.items()):
        return False
    candidate = metadata.get("candidate_path")
    if isinstance(candidate, str) and candidate and not Path(candidate).is_file():
        return False
    if metadata.get("accepted") is True:
        canonical = reference.get("canonical_path")
        return (
            isinstance(canonical, str)
            and bool(canonical)
            and Path(canonical).is_file()
        )
    return True


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"inpainting mask is unreadable: {path}")
    if mask.shape != shape:
        raise ValueError("inpainting mask dimensions must match the raw reference")
    return mask >= 128


def _entity_neutral_images(
    *,
    original: Image.Image,
    repaired: Image.Image,
    repair_mask: Image.Image,
    reference: dict[str, object],
) -> tuple[Image.Image, Image.Image]:
    original_array = np.asarray(original.convert("RGB"))
    repaired_array = np.asarray(repaired.convert("RGB"))
    if original_array.shape != repaired_array.shape:
        raise ValueError("semantic validation images must have matching dimensions")
    mask_value = reference.get("mask_raw_path") or reference.get("mask_path")
    if not isinstance(mask_value, str) or not mask_value:
        raise ValueError("entity semantic validation requires an entity mask")
    subject_mask = _read_mask(Path(mask_value), original_array.shape[:2])
    repair = np.asarray(repair_mask.convert("L")) >= 128
    if repair.shape != subject_mask.shape:
        raise ValueError("repair mask dimensions must match the entity mask")
    repaired_subject_mask = subject_mask | repair
    neutral_original = np.full_like(original_array, 204)
    neutral_repaired = np.full_like(repaired_array, 204)
    neutral_original[subject_mask] = original_array[subject_mask]
    neutral_repaired[repaired_subject_mask] = repaired_array[
        repaired_subject_mask
    ]
    return Image.fromarray(neutral_original), Image.fromarray(neutral_repaired)


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return np.asarray(mask, dtype=bool).copy()
    kernel_size = 2 * pixels + 1
    return cv2.dilate(
        np.asarray(mask, dtype=np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        iterations=1,
    ).astype(bool)


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    inverse = (~source).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    filled = source.copy()
    height, width = source.shape
    for label in range(1, count):
        x, y, component_width, component_height, _ = stats[label]
        if (
            x > 0
            and y > 0
            and x + component_width < width
            and y + component_height < height
        ):
            filled[labels == label] = True
    return filled


def _background_generation_mask(
    source_mask: np.ndarray,
    *,
    mask_dilation_pixels: int,
    adaptive_mask_dilation_ratio: float = 0.04,
) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    height, width = source.shape
    margin = max(
        mask_dilation_pixels,
        round(adaptive_mask_dilation_ratio * min(height, width)),
    )
    filled = _fill_internal_holes(source)
    closed = cv2.morphologyEx(
        filled.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed.astype(np.uint8),
        8,
    )
    if count <= 1:
        return closed

    component_labels = list(range(1, count))
    parents = {label: label for label in component_labels}

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    expanded_boxes: dict[int, tuple[int, int, int, int]] = {}
    for label in component_labels:
        x, y, component_width, component_height, _ = stats[label]
        expanded_boxes[label] = (
            max(0, int(x) - margin),
            max(0, int(y) - margin),
            min(width, int(x + component_width) + margin),
            min(height, int(y + component_height) + margin),
        )
    for index, left_label in enumerate(component_labels):
        left_x0, left_y0, left_x1, left_y1 = expanded_boxes[left_label]
        for right_label in component_labels[index + 1 :]:
            right_x0, right_y0, right_x1, right_y1 = expanded_boxes[
                right_label
            ]
            if (
                left_x0 <= right_x1
                and right_x0 <= left_x1
                and left_y0 <= right_y1
                and right_y0 <= left_y1
            ):
                union(left_label, right_label)

    groups: dict[int, list[int]] = {}
    for label in component_labels:
        groups.setdefault(find(label), []).append(label)

    generation = np.zeros_like(source)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * margin + 1, 2 * margin + 1),
    )
    for group in groups.values():
        points = np.concatenate(
            [
                np.column_stack(np.where(labels == label))[:, ::-1]
                for label in group
            ]
        ).astype(np.int32)
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        component_hull = np.zeros_like(source, dtype=np.uint8)
        cv2.fillConvexPoly(component_hull, hull, 1)
        dilated_hull = cv2.dilate(component_hull, kernel, iterations=1)
        generation |= dilated_hull.astype(bool)
    return generation


def _entity_repair_mask(
    subject_mask: np.ndarray,
    *,
    config: InpaintingConfig,
) -> tuple[np.ndarray, list[str]]:
    subject = np.asarray(subject_mask, dtype=bool)
    inverse = (~subject).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    holes = np.zeros_like(subject)
    height, width = subject.shape
    for label in range(1, count):
        x, y, component_width, component_height, _ = stats[label]
        if (
            x > 0
            and y > 0
            and x + component_width < width
            and y + component_height < height
        ):
            holes[labels == label] = True
    closed = cv2.morphologyEx(
        subject.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)
    repair = holes | (closed & ~subject)
    reasons: list[str] = []
    if not repair.any():
        return repair, reasons

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(repair.astype(np.uint8), 8)
    )
    maximum_component = (
        config.entity.maximum_component_area_ratio * repair.size
    )
    reliable = np.zeros_like(repair)
    for label in range(1, component_count):
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        if area <= maximum_component:
            reliable[component_labels == label] = True
        else:
            reasons.append("repair_component_too_large")
    repair = reliable
    if (
        repair.any()
        and config.entity.require_reliable_repair_mask
        and (
            repair[0].any()
            or repair[-1].any()
            or repair[:, 0].any()
            or repair[:, -1].any()
        )
    ):
        reasons.append("repair_mask_touches_border")
        repair[:] = False
    return repair, list(dict.fromkeys(reasons))


def _repair_mask_for_reference(
    reference: dict[str, object],
    *,
    raw_shape: tuple[int, int],
    config: InpaintingConfig,
) -> tuple[
    np.ndarray | None,
    str | None,
    list[str],
    float | None,
    float | None,
]:
    reference_type = str(reference.get("reference_type", "entity"))
    mask_path = Path(str(reference["mask_path"]))
    source_mask = _read_mask(mask_path, raw_shape)
    if reference_type == "background":
        source_foreground_area_ratio = float(source_mask.mean())
        if not config.background.enabled or not reference.get(
            "needs_inpainting", False
        ):
            return None, None, [], source_foreground_area_ratio, None
        if (
            source_foreground_area_ratio
            > config.background.maximum_hole_area_ratio
        ):
            return (
                None,
                "background_hole_fill",
                ["source_foreground_area_ratio"],
                source_foreground_area_ratio,
                None,
            )
        generation_mask = _background_generation_mask(
            source_mask,
            mask_dilation_pixels=config.mask_dilation_pixels,
            adaptive_mask_dilation_ratio=config.adaptive_mask_dilation_ratio,
        )
        generation_mask_area_ratio = float(generation_mask.mean())
        reasons = []
        if (
            generation_mask_area_ratio
            > config.background.maximum_generation_mask_area_ratio
        ):
            reasons.append("generation_mask_area_ratio")
        return (
            generation_mask,
            "background_hole_fill",
            reasons,
            source_foreground_area_ratio,
            generation_mask_area_ratio,
        )
    if not config.entity.enabled:
        return None, None, [], None, None
    completeness = float(
        (
            reference.get("visual_review")
            if isinstance(reference.get("visual_review"), dict)
            else {}
        ).get("completeness", 1.0)
    )
    if completeness < config.entity.minimum_completeness_before_repair:
        return (
            None,
            None,
            ["entity_completeness_too_low_for_local_repair"],
            None,
            None,
        )
    repair, reasons = _entity_repair_mask(source_mask, config=config)
    if repair.any():
        repair = _dilate(repair, config.mask_dilation_pixels)
    if float(repair.mean()) > config.entity.maximum_repair_area_ratio:
        reasons.append("repair_area_ratio")
    return (
        repair,
        "entity_local_repair",
        list(dict.fromkeys(reasons)),
        None,
        None,
    )


def _hard_composite(
    *,
    original: np.ndarray,
    generated: np.ndarray,
    core_mask: np.ndarray,
    feather_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    if original.shape != generated.shape:
        raise ValueError("inpainting output dimensions must match the raw image")
    core = np.asarray(core_mask, dtype=bool)
    if core.shape != original.shape[:2]:
        raise ValueError("repair mask dimensions must match the raw image")
    final = original.copy()
    final[core] = generated[core]
    ring = np.zeros_like(core)
    if feather_pixels > 0:
        kernel_size = 2 * feather_pixels + 1
        inner = cv2.erode(
            core.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        ring = core & ~inner
        if ring.any():
            distance = cv2.distanceTransform(
                core.astype(np.uint8),
                cv2.DIST_L2,
                3,
            )
            alpha = np.clip(
                distance / (feather_pixels + 1.0),
                0.0,
                1.0,
            )
            blended = (
                original.astype(np.float32) * (1.0 - alpha[..., None])
                + generated.astype(np.float32) * alpha[..., None]
            )
            final[ring] = np.rint(blended[ring]).astype(np.uint8)
    return final, ring


def _inpainting_prompt(
    reference: dict[str, object],
    mode: str,
    *,
    background_fill_prompt: str | None = None,
) -> str:
    if mode == "background_hole_fill":
        fill_prompt = (
            BACKGROUND_INPAINT_PROMPT
            if background_fill_prompt is None
            else background_fill_prompt
        )
        normalized = fill_prompt.rstrip(" .")
        return (
            f"{normalized}. {BACKGROUND_INPAINT_SUFFIX}"
            if normalized
            else BACKGROUND_INPAINT_SUFFIX
        )
    return ENTITY_INPAINT_PROMPT.format(
        description=str(
            reference.get("phrase")
            or reference.get("canonical_label")
            or "the subject"
        )
    )


def _background_forbidden_texts(
    reference: dict[str, object],
    output_root: Path,
) -> list[str]:
    values = [
        str(reference.get("phrase") or ""),
        str(reference.get("canonical_label") or ""),
    ]
    clip_uid = str(reference.get("clip_uid") or "")
    annotation_path = output_root / "annotations" / f"{clip_uid}.json"
    if annotation_path.is_file():
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            annotation = None
        if isinstance(annotation, dict):
            values.extend(
                str(annotation.get(key) or "")
                for key in ("caption", "video_summary")
            )
            entities = annotation.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        values.extend(
                            str(entity.get(key) or "")
                            for key in (
                                "canonical_label",
                                "reference_phrase",
                                "grounding_prompt",
                            )
                        )
    return list(dict.fromkeys(value for value in values if value.strip()))


def _resolve_background_fill_prompt(
    *,
    config: PipelineConfig,
    reference: dict[str, object],
    original_path: Path,
    generation_mask_path: Path,
    generator: BackgroundFillPromptGenerator | None,
) -> tuple[str, dict[str, object]]:
    prompt_mode = config.inpainting.background.prompt_mode
    if prompt_mode == "generic":
        return BACKGROUND_INPAINT_PROMPT, {
            "source": "generic",
            "visible_background_elements": [],
            "reason": "configured deterministic generic prompt",
            "validator": {"validation": "not_requested"},
        }
    active_generator = generator
    if active_generator is None:
        repair_judge = _qwen_services(config.qwen).repair_judge
        if repair_judge is not None:
            active_generator = QwenBackgroundFillPromptGenerator(repair_judge)
    if active_generator is None:
        return BACKGROUND_INPAINT_PROMPT, {
            "source": "generic_fallback",
            "visible_background_elements": [],
            "reason": "background prompt generator unavailable",
            "validator": {
                "validation": "failed",
                "error": "background_prompt_generator_unavailable",
            },
        }
    forbidden_texts = _background_forbidden_texts(
        reference,
        config.output_root,
    )
    try:
        result, validator_metadata = active_generator.generate(
            original_path=original_path,
            generation_mask_path=generation_mask_path,
            forbidden_texts=forbidden_texts,
        )
        return result.fill_prompt.strip(), {
            "source": "qwen_local_background",
            "visible_background_elements": result.visible_background_elements,
            "reason": result.reason,
            "validator": validator_metadata,
        }
    except StructuredOutputFailure as exc:
        error: object = exc.to_dict()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return BACKGROUND_INPAINT_PROMPT, {
        "source": "generic_fallback",
        "visible_background_elements": [],
        "reason": "Qwen background prompt generation failed validation",
        "validator": {"validation": "failed", "error": error},
    }


def _reference_artifacts(output_root: Path) -> list[Path]:
    reference_root = output_root / "references"
    return sorted(
        [
            *reference_root.glob("*/*/metadata.json"),
            *reference_root.glob("*/*/reference_metadata.json"),
        ]
    )


def _canonical_copy_path(artifact: Path, raw_path: Path) -> Path:
    return artifact.parent / f"canonical{image_extension(raw_path)}"


def _ensure_entity_raw_artifacts(
    *,
    reference: dict[str, object],
    artifact: Path,
) -> None:
    if reference.get("reference_type", "entity") != "entity":
        return
    snapshots = (
        ("mask.png", "mask_raw.png", "mask_path", "mask_raw_path", True),
        (
            "foreground_rgba.png",
            "foreground_rgba_raw.png",
            "foreground_rgba_path",
            "foreground_rgba_raw_path",
            True,
        ),
        (
            "neutral_background.jpg",
            "neutral_background_raw.jpg",
            "neutral_background_path",
            "neutral_background_raw_path",
            True,
        ),
        (
            "dinov3_embedding.npy",
            "dinov3_embedding_raw.npy",
            "dinov3_embedding_path",
            "dinov3_embedding_raw_path",
            False,
        ),
    )
    for (
        source_name,
        raw_name,
        source_key,
        raw_key,
        required,
    ) in snapshots:
        source_value = reference.get(source_key)
        source = (
            Path(source_value)
            if isinstance(source_value, str) and source_value
            else artifact.parent / source_name
        )
        raw = artifact.parent / raw_name
        if not raw.is_file():
            if (
                not source.is_file()
                and source_name
                in {"foreground_rgba.png", "neutral_background.jpg"}
            ):
                raw_canonical_value = reference.get("raw_canonical_path")
                mask_raw = artifact.parent / "mask_raw.png"
                if (
                    isinstance(raw_canonical_value, str)
                    and raw_canonical_value
                    and Path(raw_canonical_value).is_file()
                    and mask_raw.is_file()
                ):
                    image = np.asarray(
                        Image.open(raw_canonical_value).convert("RGB")
                    )
                    subject_mask = _read_mask(mask_raw, image.shape[:2])
                    if source_name == "foreground_rgba.png":
                        rgba = np.dstack(
                            (image, subject_mask.astype(np.uint8) * 255)
                        )
                        _save_lossless_atomic(Image.fromarray(rgba), source)
                    else:
                        neutral = np.full_like(image, 204)
                        neutral[subject_mask] = image[subject_mask]
                        _save_jpeg_atomic(Image.fromarray(neutral), source)
            if source.is_file():
                _copy_atomic(source, raw)
            elif required:
                raise FileNotFoundError(
                    f"entity source artifact is missing: {source}"
                )
        reference[raw_key] = str(raw) if raw.is_file() else None


def _restore_entity_raw_artifacts(
    *,
    reference: dict[str, object],
    artifact: Path,
) -> None:
    if reference.get("reference_type", "entity") != "entity":
        return
    snapshots = (
        ("mask_raw.png", "mask.png", "mask_path", True),
        (
            "foreground_rgba_raw.png",
            "foreground_rgba.png",
            "foreground_rgba_path",
            True,
        ),
        (
            "neutral_background_raw.jpg",
            "neutral_background.jpg",
            "neutral_background_path",
            True,
        ),
        (
            "dinov3_embedding_raw.npy",
            "dinov3_embedding.npy",
            "dinov3_embedding_path",
            False,
        ),
    )
    for raw_name, destination_name, metadata_key, required in snapshots:
        raw = artifact.parent / raw_name
        destination = artifact.parent / destination_name
        if raw.is_file():
            _copy_atomic(raw, destination)
            reference[metadata_key] = str(destination)
        else:
            destination.unlink(missing_ok=True)
            reference[metadata_key] = None
            if required:
                raise FileNotFoundError(
                    f"immutable entity artifact is missing: {raw}"
                )


def _restore_from_raw(
    *,
    reference: dict[str, object],
    artifact: Path,
    raw_path: Path,
    write_artifact: bool = True,
) -> Path:
    _restore_entity_raw_artifacts(
        reference=reference,
        artifact=artifact,
    )
    canonical_path = _canonical_copy_path(artifact, raw_path)
    for stale in (artifact.parent / "canonical.jpg", artifact.parent / "canonical.png"):
        if stale != canonical_path:
            stale.unlink(missing_ok=True)
    _copy_atomic(raw_path, canonical_path)
    needs_inpainting = (
        reference.get("reference_type", "entity") == "background"
        and bool(reference.get("needs_inpainting", False))
    )
    reference.update(
        {
            "canonical_path": str(canonical_path),
            "inpainted": False,
            "status": "pending_inpainting" if needs_inpainting else "ready",
            "rejected": False,
        }
    )
    reference.pop("inpainting_metadata_path", None)
    if write_artifact:
        write_json_atomic(artifact, reference)
    return canonical_path


def _clear_stale_inpainting_artifacts(artifact: Path) -> None:
    for filename in (
        "repair_mask.png",
        "generation_mask.png",
        "canonical_repaired.jpg",
        "canonical_repaired.png",
        "canonical_repaired_candidate.png",
        "inpainting_metadata.json",
    ):
        (artifact.parent / filename).unlink(missing_ok=True)
    candidate_root = artifact.parent / "inpainting_candidates"
    if candidate_root.is_dir():
        shutil.rmtree(candidate_root)


def _publish_rejection(
    *,
    reference: dict[str, object],
    artifact: Path,
    raw_path: Path,
    metadata: dict[str, object],
    config: InpaintingConfig,
) -> bool:
    canonical_path = _restore_from_raw(
        reference=reference,
        artifact=artifact,
        raw_path=raw_path,
        write_artifact=False,
    )
    fallback = (
        reference.get("reference_type", "entity") != "background"
        and config.consistency.fallback_to_raw
    )
    metadata["fallback_to_raw"] = fallback
    reference.update(
        {
            "canonical_path": str(canonical_path),
            "inpainted": False,
            "inpainting_metadata_path": str(
                artifact.parent / "inpainting_metadata.json"
            ),
            "status": "ready" if fallback else "rejected",
            "rejected": not fallback,
        }
    )
    write_json_atomic(artifact.parent / "inpainting_metadata.json", metadata)
    write_json_atomic(artifact, reference)
    return fallback


def _metadata_base(
    *,
    config: InpaintingConfig,
    mode: str | None,
    repair_area_ratio: float,
    source_foreground_area_ratio: float | None,
    generation_mask_area_ratio: float | None,
    source_mask_path: Path,
    generation_mask_path: Path | None,
    source_signature: dict[str, object],
) -> dict[str, object]:
    return {
        "backend": config.backend,
        "model_path": (
            str(config.model_path) if config.model_path is not None else None
        ),
        "mode": mode,
        "seed": config.seed,
        "source_foreground_area_ratio": source_foreground_area_ratio,
        "generation_mask_area_ratio": generation_mask_area_ratio,
        "repair_area_ratio": repair_area_ratio,
        "source_mask_path": str(source_mask_path),
        "generation_mask_path": (
            str(generation_mask_path)
            if generation_mask_path is not None
            else None
        ),
        "candidate_path": None,
        "unmasked_l1_diff": 0.0,
        "dino_similarity": None,
        "accepted": False,
        "fallback_to_raw": False,
        **source_signature,
    }


def _regenerate_entity_artifacts(
    *,
    artifact: Path,
    reference: dict[str, object],
    repaired: np.ndarray,
    original_mask: np.ndarray,
    repair_mask: np.ndarray,
    dino_embedder: DinoV3Embedder | None,
) -> None:
    updated_mask = np.asarray(original_mask | repair_mask, dtype=bool)
    mask_path = artifact.parent / "mask.png"
    rgba_path = artifact.parent / "foreground_rgba.png"
    neutral_path = artifact.parent / "neutral_background.jpg"
    embedding_path = artifact.parent / "dinov3_embedding.npy"

    rgba = np.dstack((repaired, updated_mask.astype(np.uint8) * 255))
    neutral = np.full_like(repaired, 204)
    neutral[updated_mask] = repaired[updated_mask]
    selected_embedding: np.ndarray | None = None
    if dino_embedder is not None:
        embeddings = dino_embedder.encode([Image.fromarray(neutral)])
        if len(embeddings) != 1:
            raise RuntimeError("DINOv3 returned an unexpected embedding count")
        selected_embedding = embeddings[0]

    save_mask_png(mask_path, updated_mask)
    _save_lossless_atomic(Image.fromarray(rgba), rgba_path)
    _save_jpeg_atomic(Image.fromarray(neutral), neutral_path)
    if selected_embedding is not None:
        save_selected_dinov3_embedding(embedding_path, selected_embedding)
        reference["dinov3_embedding_path"] = str(embedding_path)
    else:
        previous_embedding = reference.get("dinov3_embedding_path")
        if isinstance(previous_embedding, str) and previous_embedding:
            previous_path = Path(previous_embedding).resolve(strict=False)
            if previous_path.parent == artifact.parent.resolve():
                previous_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        reference["dinov3_embedding_path"] = None
    reference.update(
        {
            "mask_path": str(mask_path),
            "foreground_rgba_path": str(rgba_path),
            "neutral_background_path": str(neutral_path),
        }
    )


def run_inpainting(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    backend: InpaintBackend | None = None,
    validator: ConsistencyValidator | None = None,
    dino_embedder: DinoV3Embedder | None = None,
    background_prompt_generator: BackgroundFillPromptGenerator | None = None,
) -> InpaintingStats:
    if not config.inpainting.enabled:
        return InpaintingStats(skipped_disabled=1)
    if not config.inpainting.consistency.preserve_unmasked_pixels:
        raise ValueError(
            "inpainting.consistency.preserve_unmasked_pixels must remain true"
        )
    if (
        config.inpainting.backend == "flux1_fill"
        and config.inpainting.background.enabled
        and not has_background_inpainting_repair_judge(config)
    ):
        raise ValueError(
            "production background_hole_fill requires qwen.repair_judge"
        )
    if (
        config.inpainting.backend == "flux1_fill"
        and not has_inpainting_semantic_validator(config)
    ):
        raise ValueError(
            "production inpainting requires DINOv3+SigLIP2 or an explicit "
            "qwen.repair_judge consistency validator"
        )
    output_root = config.ensure_output_root()
    engine = backend
    if engine is None:
        engine = (
            NoOpInpaintBackend()
            if config.inpainting.backend == "noop"
            else Flux1FillBackend(config.inpainting)
        )
    active_validator = validator
    owned_validator: ProductionConsistencyValidator | None = None
    if active_validator is None:
        owned_validator = ProductionConsistencyValidator(
            config,
            dino_embedder=dino_embedder,
        )
        active_validator = owned_validator

    config_fingerprint = _config_fingerprint(config)
    processed = skipped = repaired_count = fallback_count = rejected = failed = 0
    for artifact in _reference_artifacts(output_root):
        reference = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(reference, dict):
            raise TypeError(f"reference metadata must be an object: {artifact}")
        reference.setdefault("status", "ready")
        raw_path = Path(
            str(reference.get("raw_canonical_path") or reference["canonical_path"])
        )
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw canonical reference is missing: {raw_path}")
        if "raw_canonical_path" not in reference:
            new_raw_path = (
                artifact.parent / f"canonical_raw{image_extension(raw_path)}"
            )
            _copy_atomic(raw_path, new_raw_path)
            raw_path = new_raw_path
            reference["raw_canonical_path"] = str(raw_path)
        _ensure_entity_raw_artifacts(
            reference=reference,
            artifact=artifact,
        )
        source_mask_value = (
            reference.get("mask_raw_path")
            if reference.get("reference_type", "entity") == "entity"
            else reference.get("mask_path")
        )
        if not isinstance(source_mask_value, str) or not source_mask_value:
            raise ValueError(f"reference mask path is missing: {artifact}")
        source_mask_path = Path(source_mask_value)
        source_signature = _source_signature(
            reference=reference,
            source_image_path=raw_path,
            source_mask_path=source_mask_path,
            config_fingerprint=config_fingerprint,
        )
        write_json_atomic(artifact, reference)
        inpainting_metadata_path = artifact.parent / "inpainting_metadata.json"
        stale_metadata = False
        if inpainting_metadata_path.is_file() and not overwrite:
            existing = json.loads(
                inpainting_metadata_path.read_text(encoding="utf-8")
            )
            if not isinstance(existing, dict):
                raise TypeError(
                    f"inpainting metadata must be an object: "
                    f"{inpainting_metadata_path}"
                )
            if _metadata_matches_source(existing, source_signature, reference):
                processed += 1
                if existing.get("accepted") is True:
                    repaired_count += 1
                elif existing.get("fallback_to_raw") is True:
                    fallback_count += 1
                else:
                    rejected += 1
                continue
            stale_metadata = True
            _clear_stale_inpainting_artifacts(artifact)
            _restore_from_raw(
                reference=reference,
                artifact=artifact,
                raw_path=raw_path,
            )
        if (
            reference.get("status") == "rejected"
            and not overwrite
            and not stale_metadata
        ):
            skipped += 1
            continue
        if overwrite:
            _clear_stale_inpainting_artifacts(artifact)
            _restore_from_raw(
                reference=reference,
                artifact=artifact,
                raw_path=raw_path,
            )
        mode: str | None = None
        repair_area_ratio = 0.0
        source_foreground_area_ratio: float | None = None
        generation_mask_area_ratio: float | None = None
        generation_mask_path: Path | None = None
        candidate_path: Path | None = None
        try:
            original_image = Image.open(raw_path).convert("RGB")
            original = np.asarray(original_image).copy()
            (
                repair_mask,
                mode,
                mask_reasons,
                source_foreground_area_ratio,
                generation_mask_area_ratio,
            ) = _repair_mask_for_reference(
                reference,
                raw_shape=original.shape[:2],
                config=config.inpainting,
            )
            repair_area_ratio = (
                float(repair_mask.mean()) if repair_mask is not None else 0.0
            )
            if mask_reasons or (
                repair_mask is not None
                and mode is not None
                and repair_mask.any()
            ):
                processed += 1
            if (
                mode == "background_hole_fill"
                and repair_mask is not None
                and repair_mask.any()
            ):
                generation_mask_path = (
                    artifact.parent / "generation_mask.png"
                )
                _save_lossless_atomic(
                    Image.fromarray(repair_mask.astype(np.uint8) * 255),
                    generation_mask_path,
                )
            if repair_mask is None or mode is None or not repair_mask.any():
                if mask_reasons:
                    metadata = _metadata_base(
                        config=config.inpainting,
                        mode=mode,
                        repair_area_ratio=repair_area_ratio,
                        source_foreground_area_ratio=(
                            source_foreground_area_ratio
                        ),
                        generation_mask_area_ratio=(
                            generation_mask_area_ratio
                        ),
                        source_mask_path=source_mask_path,
                        generation_mask_path=generation_mask_path,
                        source_signature=source_signature,
                    )
                    metadata["rejection_reasons"] = mask_reasons
                    did_fallback = _publish_rejection(
                        reference=reference,
                        artifact=artifact,
                        raw_path=raw_path,
                        metadata=metadata,
                        config=config.inpainting,
                    )
                    rejected += 1
                    if did_fallback:
                        fallback_count += 1
                    _append_jsonl(
                        output_root / "logs" / "inpainting_rejected.jsonl",
                        {
                            "clip_uid": reference.get("clip_uid"),
                            "reference_id": reference.get(
                                "reference_id", reference.get("entity_id")
                            ),
                            "reasons": mask_reasons,
                        },
                    )
                else:
                    _restore_from_raw(
                        reference=reference,
                        artifact=artifact,
                        raw_path=raw_path,
                    )
                    skipped += 1
                continue
            if mask_reasons:
                metadata = _metadata_base(
                    config=config.inpainting,
                    mode=mode,
                    repair_area_ratio=repair_area_ratio,
                    source_foreground_area_ratio=(
                        source_foreground_area_ratio
                    ),
                    generation_mask_area_ratio=generation_mask_area_ratio,
                    source_mask_path=source_mask_path,
                    generation_mask_path=generation_mask_path,
                    source_signature=source_signature,
                )
                metadata["rejection_reasons"] = mask_reasons
                did_fallback = _publish_rejection(
                    reference=reference,
                    artifact=artifact,
                    raw_path=raw_path,
                    metadata=metadata,
                    config=config.inpainting,
                )
                rejected += 1
                if did_fallback:
                    fallback_count += 1
                _append_jsonl(
                    output_root / "logs" / "inpainting_rejected.jsonl",
                    {
                        "clip_uid": reference.get("clip_uid"),
                        "reference_id": reference.get(
                            "reference_id", reference.get("entity_id")
                        ),
                        "reasons": mask_reasons,
                    },
                )
                continue

            repair_mask_image = Image.fromarray(
                repair_mask.astype(np.uint8) * 255,
            )
            _save_lossless_atomic(
                repair_mask_image,
                artifact.parent / "repair_mask.png",
            )
            fill_prompt: str | None = None
            fill_prompt_metadata: dict[str, object] | None = None
            if mode == "background_hole_fill":
                if generation_mask_path is None:
                    raise RuntimeError("background generation mask was not saved")
                fill_prompt, fill_prompt_metadata = (
                    _resolve_background_fill_prompt(
                        config=config,
                        reference=reference,
                        original_path=raw_path,
                        generation_mask_path=generation_mask_path,
                        generator=background_prompt_generator,
                    )
                )
            production_prompt = _inpainting_prompt(
                reference,
                mode,
                background_fill_prompt=fill_prompt,
            )
            candidate_seeds = (
                config.inpainting.background.candidate_seeds
                if mode == "background_hole_fill"
                else [config.inpainting.seed]
            )
            candidate_records: list[dict[str, object]] = []
            candidate_arrays: dict[int, np.ndarray] = {}
            selected_candidate_index: int | None = None
            source_mask = _read_mask(source_mask_path, original.shape[:2])
            for candidate_index, seed in enumerate(candidate_seeds):
                candidate_dir = (
                    artifact.parent
                    / "inpainting_candidates"
                    / f"candidate_{candidate_index:02d}_seed_{seed}"
                )
                seed_candidate_path = candidate_dir / "candidate.png"
                record: dict[str, object] = {
                    "candidate_index": candidate_index,
                    "seed": seed,
                    "guidance_scale": config.inpainting.guidance_scale,
                    "num_inference_steps": config.inpainting.num_inference_steps,
                    "strength": config.inpainting.strength,
                    "max_sequence_length": (
                        config.inpainting.max_sequence_length
                    ),
                    "candidate_path": str(seed_candidate_path),
                    "prompt_sha256": _sha256_text(production_prompt),
                    "accepted": False,
                    "rejection_reasons": [],
                }
                try:
                    generated_image = engine.inpaint(
                        image=original_image,
                        mask=repair_mask_image,
                        prompt=production_prompt,
                        seed=seed,
                    ).convert("RGB")
                    generated = np.asarray(generated_image)
                    final, feather_ring = _hard_composite(
                        original=original,
                        generated=generated,
                        core_mask=repair_mask,
                        feather_pixels=config.inpainting.feather_pixels,
                    )
                    _save_lossless_atomic(
                        Image.fromarray(final),
                        seed_candidate_path,
                    )
                    protected = ~(repair_mask | feather_ring)
                    unmasked_l1_diff = (
                        float(
                            np.abs(
                                original[protected].astype(np.float32)
                                - final[protected].astype(np.float32)
                            ).mean()
                        )
                        if protected.any()
                        else 0.0
                    )
                    diagnostics: dict[str, object] = {}
                    if mode == "background_hole_fill":
                        diagnostics = _background_pixel_diagnostics(
                            original=original,
                            repaired=final,
                            source_mask=source_mask,
                            generation_mask=repair_mask,
                            destination=candidate_dir
                            / "difference_heatmap.png",
                        )
                    if isinstance(
                        active_validator,
                        ProductionConsistencyValidator,
                    ):
                        validation = active_validator(
                            original=original_image,
                            repaired=Image.fromarray(final),
                            repair_mask=repair_mask_image,
                            reference=reference,
                            mode=mode,
                            diagnostics_dir=(
                                candidate_dir
                                if mode == "background_hole_fill"
                                else None
                            ),
                        )
                    else:
                        validation = active_validator(
                            original=original_image,
                            repaired=Image.fromarray(final),
                            repair_mask=repair_mask_image,
                            reference=reference,
                            mode=mode,
                        )
                    rejection_reasons = [
                        str(reason)
                        for reason in validation.get(
                            "rejection_reasons", []
                        )
                    ]
                    if (
                        unmasked_l1_diff
                        > config.inpainting.consistency.maximum_unmasked_l1_diff
                    ):
                        rejection_reasons.append(
                            "unmasked_pixel_difference"
                        )
                    dino_similarity_value = validation.get("dino_similarity")
                    dino_similarity = (
                        float(dino_similarity_value)
                        if isinstance(dino_similarity_value, (int, float))
                        else None
                    )
                    if (
                        mode == "entity_local_repair"
                        and dino_similarity is not None
                        and dino_similarity
                        < config.inpainting.consistency.minimum_dino_similarity
                    ):
                        rejection_reasons.append("dino_similarity")
                    if validation.get("accepted") is not True:
                        rejection_reasons.append("consistency_validator")
                    if isinstance(engine, NoOpInpaintBackend):
                        rejection_reasons.append("noop_backend_test_only")
                    rejection_reasons = list(
                        dict.fromkeys(rejection_reasons)
                    )
                    record.update(
                        {
                            **diagnostics,
                            "unmasked_l1_diff": unmasked_l1_diff,
                            "dino_similarity": dino_similarity,
                            "accepted": not rejection_reasons,
                            "rejection_reasons": rejection_reasons,
                            "validator": validation,
                        }
                    )
                    candidate_arrays[candidate_index] = final
                except InpaintingDependencyError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    record.update(
                        {
                            "rejection_reasons": [
                                "candidate_generation_or_validation_failed"
                            ],
                            "error": str(exc),
                        }
                    )
                validator_payload = record.get("validator")
                record["source_signature"] = {
                    **source_signature,
                    "fill_prompt_sha256": (
                        _sha256_text(fill_prompt)
                        if fill_prompt is not None
                        else None
                    ),
                    "production_prompt_sha256": _sha256_text(
                        production_prompt
                    ),
                    "prompt_source": (
                        fill_prompt_metadata.get("source")
                        if fill_prompt_metadata is not None
                        else "entity_template"
                    ),
                    "seed": seed,
                    "guidance_scale": config.inpainting.guidance_scale,
                    "num_inference_steps": (
                        config.inpainting.num_inference_steps
                    ),
                    "candidate_index": candidate_index,
                    "comparison_sheet_fingerprints": (
                        validator_payload.get(
                            "comparison_sheet_fingerprints"
                        )
                        if isinstance(validator_payload, dict)
                        else None
                    ),
                    "diagnostics": {
                        key: record.get(key)
                        for key in (
                            "masked_mean_l1",
                            "masked_changed_pixel_ratio",
                            "generation_mask_changed_pixel_ratio",
                            "boundary_ring_mean_l1",
                        )
                        if key in record
                    },
                }
                candidate_records.append(record)
                if record["accepted"] is True and (
                    mode != "background_hole_fill"
                    or config.inpainting.background.stop_after_first_accepted
                ):
                    selected_candidate_index = candidate_index
                    break
            accepted_records = [
                record
                for record in candidate_records
                if record.get("accepted") is True
            ]
            if (
                selected_candidate_index is None
                and accepted_records
            ):
                selected = max(
                    accepted_records,
                    key=lambda record: (
                        float(record.get("dino_similarity") or -1.0),
                        float(record.get("masked_changed_pixel_ratio") or 0.0),
                        -int(record["candidate_index"]),
                    ),
                )
                selected_candidate_index = int(
                    selected["candidate_index"]
                )
            debug_record = (
                next(
                    (
                        record
                        for record in candidate_records
                        if record["candidate_index"]
                        == selected_candidate_index
                    ),
                    None,
                )
                or (candidate_records[-1] if candidate_records else None)
            )
            if debug_record is None:
                raise RuntimeError("inpainting produced no candidate records")
            candidate_destination = (
                artifact.parent / "canonical_repaired_candidate.png"
            )
            debug_candidate_path = Path(str(debug_record["candidate_path"]))
            if debug_candidate_path.is_file():
                _copy_atomic(debug_candidate_path, candidate_destination)
            candidate_path = candidate_destination
            selected_record = next(
                (
                    record
                    for record in candidate_records
                    if record["candidate_index"] == selected_candidate_index
                ),
                debug_record,
            )
            accepted = selected_candidate_index is not None
            rejection_reasons = (
                []
                if accepted
                else list(
                    dict.fromkeys(
                        str(reason)
                        for record in candidate_records
                        for reason in record.get("rejection_reasons", [])
                    )
                )
            )
            final = candidate_arrays.get(int(selected_record["candidate_index"]))
            dino_similarity_value = selected_record.get("dino_similarity")
            dino_similarity = (
                float(dino_similarity_value)
                if isinstance(dino_similarity_value, (int, float))
                else None
            )
            metadata = _metadata_base(
                config=config.inpainting,
                mode=mode,
                repair_area_ratio=repair_area_ratio,
                source_foreground_area_ratio=source_foreground_area_ratio,
                generation_mask_area_ratio=generation_mask_area_ratio,
                source_mask_path=source_mask_path,
                generation_mask_path=generation_mask_path,
                source_signature=source_signature,
            )
            metadata.update(
                {
                    "candidate_path": str(candidate_path),
                    "candidates": candidate_records,
                    "selected_candidate_index": selected_candidate_index,
                    "selected_seed": (
                        selected_record.get("seed") if accepted else None
                    ),
                    "seed": (
                        selected_record.get("seed")
                        if accepted
                        else config.inpainting.seed
                    ),
                    "guidance_scale": config.inpainting.guidance_scale,
                    "num_inference_steps": (
                        config.inpainting.num_inference_steps
                    ),
                    "fill_prompt": fill_prompt,
                    "production_prompt": production_prompt,
                    "prompt_source": (
                        fill_prompt_metadata.get("source")
                        if fill_prompt_metadata is not None
                        else "entity_template"
                    ),
                    "prompt_metadata": fill_prompt_metadata,
                    "fill_prompt_sha256": (
                        _sha256_text(fill_prompt)
                        if fill_prompt is not None
                        else None
                    ),
                    "unmasked_l1_diff": selected_record.get(
                        "unmasked_l1_diff", 0.0
                    ),
                    "dino_similarity": dino_similarity,
                    "accepted": accepted,
                    "rejection_reasons": rejection_reasons,
                    "validator": selected_record.get("validator"),
                    "diagnostics": {
                        key: selected_record.get(key)
                        for key in (
                            "masked_mean_l1",
                            "masked_changed_pixel_ratio",
                            "generation_mask_changed_pixel_ratio",
                            "boundary_ring_mean_l1",
                            "difference_heatmap_path",
                        )
                        if key in selected_record
                    },
                    "selected_candidate_source_signature": (
                        {
                            **source_signature,
                            "fill_prompt_sha256": (
                                _sha256_text(fill_prompt)
                                if fill_prompt is not None
                                else None
                            ),
                            "production_prompt_sha256": _sha256_text(
                                production_prompt
                            ),
                            "prompt_source": (
                                fill_prompt_metadata.get("source")
                                if fill_prompt_metadata is not None
                                else "entity_template"
                            ),
                            "seed": selected_record.get("seed"),
                            "guidance_scale": (
                                config.inpainting.guidance_scale
                            ),
                            "num_inference_steps": (
                                config.inpainting.num_inference_steps
                            ),
                            "candidate_index": selected_record.get(
                                "candidate_index"
                            ),
                            "comparison_sheet_fingerprints": (
                                (
                                    selected_record.get("validator")
                                    if isinstance(
                                        selected_record.get("validator"),
                                        dict,
                                    )
                                    else {}
                                ).get("comparison_sheet_fingerprints")
                            ),
                            "diagnostics": {
                                key: selected_record.get(key)
                                for key in (
                                    "masked_mean_l1",
                                    "masked_changed_pixel_ratio",
                                    "generation_mask_changed_pixel_ratio",
                                    "boundary_ring_mean_l1",
                                )
                                if key in selected_record
                            },
                        }
                        if accepted
                        else None
                    ),
                    "lossless_storage": True,
                }
            )
            if accepted:
                if final is None:
                    raise RuntimeError("selected inpainting candidate is missing")
                repaired_path = artifact.parent / "canonical_repaired.png"
                _copy_atomic(candidate_path, repaired_path)
                (artifact.parent / "canonical_repaired.jpg").unlink(
                    missing_ok=True
                )
                if reference.get("reference_type", "entity") == "entity":
                    entity_dino = dino_embedder
                    if (
                        entity_dino is None
                        and isinstance(
                            active_validator,
                            ProductionConsistencyValidator,
                        )
                    ):
                        entity_dino = active_validator.dino
                    _regenerate_entity_artifacts(
                        artifact=artifact,
                        reference=reference,
                        repaired=final,
                        original_mask=_read_mask(
                            Path(
                                str(
                                    reference.get("mask_raw_path")
                                    or reference["mask_path"]
                                )
                            ),
                            final.shape[:2],
                        ),
                        repair_mask=repair_mask,
                        dino_embedder=entity_dino,
                    )
                write_json_atomic(inpainting_metadata_path, metadata)
                reference.update(
                    {
                        "canonical_path": str(repaired_path),
                        "inpainted": True,
                        "inpainting_metadata_path": str(
                            inpainting_metadata_path
                        ),
                        "status": "ready",
                        "rejected": False,
                    }
                )
                write_json_atomic(artifact, reference)
                repaired_count += 1
            else:
                did_fallback = _publish_rejection(
                    reference=reference,
                    artifact=artifact,
                    raw_path=raw_path,
                    metadata=metadata,
                    config=config.inpainting,
                )
                rejected += 1
                if did_fallback:
                    fallback_count += 1
                _append_jsonl(
                    output_root / "logs" / "inpainting_rejected.jsonl",
                    {
                        "clip_uid": reference.get("clip_uid"),
                        "reference_id": reference.get(
                            "reference_id", reference.get("entity_id")
                        ),
                        "reasons": rejection_reasons,
                    },
                )
        except InpaintingDependencyError:
            if owned_validator is not None:
                owned_validator.close()
            raise
        except Exception as exc:  # noqa: BLE001
            failed += 1
            _append_jsonl(
                output_root / "logs" / "inpainting_failed.jsonl",
                {
                    "clip_uid": reference.get("clip_uid"),
                    "reference_id": reference.get(
                        "reference_id", reference.get("entity_id")
                    ),
                    "error": str(exc),
                },
            )
            metadata = _metadata_base(
                config=config.inpainting,
                mode=mode,
                repair_area_ratio=repair_area_ratio,
                source_foreground_area_ratio=source_foreground_area_ratio,
                generation_mask_area_ratio=generation_mask_area_ratio,
                source_mask_path=source_mask_path,
                generation_mask_path=generation_mask_path,
                source_signature=source_signature,
            )
            metadata.update(
                {
                    "candidate_path": (
                        str(candidate_path)
                        if candidate_path is not None
                        else None
                    ),
                    "rejection_reasons": ["inpainting_runtime_failure"],
                    "error": str(exc),
                }
            )
            did_fallback = _publish_rejection(
                reference=reference,
                artifact=artifact,
                raw_path=raw_path,
                metadata=metadata,
                config=config.inpainting,
            )
            rejected += 1
            if did_fallback:
                fallback_count += 1
    reconcile_references(output_root)
    if owned_validator is not None:
        owned_validator.close()
    return InpaintingStats(
        processed=processed,
        skipped_no_repair_needed=skipped,
        repaired=repaired_count,
        fallback_to_raw=fallback_count,
        rejected=rejected,
        failed=failed,
    )
