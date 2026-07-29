from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image

from prompts.qwen_inpainting_consistency_prompt import (
    INPAINTING_CONSISTENCY_PROMPT,
)
from r2v_data_v2.config import (
    InpaintingConfig,
    PipelineConfig,
    QwenConfig,
    _qwen_services,
    has_inpainting_semantic_validator,
)
from r2v_data_v2.image_utils import (
    image_data_uri,
    image_extension,
)
from r2v_data_v2.mask_utils import save_mask_png
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic
from r2v_data_v2.schemas import InpaintingSemanticReview
from r2v_data_v2.semantic_alignment import Siglip2Aligner
from r2v_data_v2.structured_output import request_structured_output
from r2v_data_v2.visual_embedding import (
    DinoV3Embedder,
    embedding_cosine_similarity,
    save_selected_dinov3_embedding,
)

BACKGROUND_INPAINT_PROMPT = (
    "Remove the masked foreground subjects and reconstruct the original "
    "background consistently with the surrounding scene. Preserve the same "
    "perspective, lighting, color palette, depth, textures, and camera style. "
    "Do not add new salient objects."
)

ENTITY_INPAINT_PROMPT = (
    "Restore only the masked damaged or occluded region of {description}. "
    "Preserve the exact same entity identity, shape, materials, colors, "
    "viewpoint, lighting, scale, and image style. Do not modify unmasked "
    "regions and do not add new parts beyond what is necessary to complete "
    "the local damage."
)

INPAINTING_SOURCE_METADATA_VERSION = "2"


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
        reference_phrase: str,
        mode: str,
    ) -> InpaintingSemanticReview: ...


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
        )
        if not getattr(result, "images", None):
            raise RuntimeError("FLUX.1 Fill returned no image")
        generated = result.images[0].convert("RGB")
        if generated.size != (padded_width, padded_height):
            raise ValueError("FLUX.1 Fill returned unexpected padded dimensions")
        return generated.crop((0, 0, width, height))


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
                            "image_url": {"url": image_data_uri(repaired_path)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri(repair_mask_path)},
                        },
                    ],
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
                        "name": "inpainting_semantic_review",
                        "strict": True,
                        "schema": InpaintingSemanticReview.model_json_schema(),
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
        reference_phrase: str,
        mode: str,
    ) -> InpaintingSemanticReview:
        prompt = INPAINTING_CONSISTENCY_PROMPT.format(
            reference_phrase=reference_phrase,
            repair_mode=mode,
        )
        return request_structured_output(
            request=lambda request_text: self._request(
                prompt=request_text,
                original_path=original_path,
                repaired_path=repaired_path,
                repair_mask_path=repair_mask_path,
            ),
            original_request=prompt,
            model=InpaintingSemanticReview,
        )


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
                if (
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
                if (
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
        if self.qwen is not None:
            temporary_root = Path(
                str(reference.get("raw_canonical_path", "."))
            ).parent
            original_path = temporary_root / ".inpainting_qwen_original.png"
            repaired_path = temporary_root / ".inpainting_qwen_repaired.png"
            repair_mask_path = (
                temporary_root / ".inpainting_qwen_repair_mask.png"
            )
            try:
                _save_lossless_atomic(original, original_path)
                _save_lossless_atomic(repaired, repaired_path)
                _save_lossless_atomic(
                    repair_mask.convert("L"),
                    repair_mask_path,
                )
                review = self.qwen.review(
                    original_path=original_path,
                    repaired_path=repaired_path,
                    repair_mask_path=repair_mask_path,
                    reference_phrase=phrase,
                    mode=mode,
                )
                qwen_review = review.model_dump(mode="json")
                if not (
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
    consistency_prompt = INPAINTING_CONSISTENCY_PROMPT.format(
        reference_phrase=(
            reference_phrase or canonical_label or "reference image"
        ),
        repair_mode=mode,
    )
    prompt_payload = json.dumps(
        {
            "repair_prompt": repair_prompt,
            "consistency_prompt": consistency_prompt,
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
        "prompt_fingerprint": _sha256_text(prompt_payload),
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
) -> tuple[np.ndarray | None, str | None, list[str]]:
    reference_type = str(reference.get("reference_type", "entity"))
    mask_path = Path(str(reference["mask_path"]))
    source_mask = _read_mask(mask_path, raw_shape)
    if reference_type == "background":
        if not config.background.enabled or not reference.get(
            "needs_inpainting", False
        ):
            return None, None, []
        repair = _dilate(source_mask, config.mask_dilation_pixels)
        reasons = []
        if float(repair.mean()) > config.background.maximum_hole_area_ratio:
            reasons.append("repair_area_ratio")
        return repair, "background_hole_fill", reasons
    if not config.entity.enabled:
        return None, None, []
    completeness = float(
        (
            reference.get("visual_review")
            if isinstance(reference.get("visual_review"), dict)
            else {}
        ).get("completeness", 1.0)
    )
    if completeness < config.entity.minimum_completeness_before_repair:
        return None, None, ["entity_completeness_too_low_for_local_repair"]
    repair, reasons = _entity_repair_mask(source_mask, config=config)
    if repair.any():
        repair = _dilate(repair, config.mask_dilation_pixels)
    if float(repair.mean()) > config.entity.maximum_repair_area_ratio:
        reasons.append("repair_area_ratio")
    return repair, "entity_local_repair", list(dict.fromkeys(reasons))


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
) -> str:
    if mode == "background_hole_fill":
        phrase = str(reference.get("phrase", "")).strip()
        suffix = f" The original scene is described as: {phrase}." if phrase else ""
        return BACKGROUND_INPAINT_PROMPT + suffix
    return ENTITY_INPAINT_PROMPT.format(
        description=str(
            reference.get("phrase")
            or reference.get("canonical_label")
            or "the subject"
        )
    )


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
        "canonical_repaired.jpg",
        "canonical_repaired.png",
        "inpainting_metadata.json",
    ):
        (artifact.parent / filename).unlink(missing_ok=True)


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
    source_signature: dict[str, object],
) -> dict[str, object]:
    return {
        "backend": config.backend,
        "model_path": (
            str(config.model_path) if config.model_path is not None else None
        ),
        "mode": mode,
        "seed": config.seed,
        "repair_area_ratio": repair_area_ratio,
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
) -> InpaintingStats:
    if not config.inpainting.enabled:
        return InpaintingStats(skipped_disabled=1)
    if not config.inpainting.consistency.preserve_unmasked_pixels:
        raise ValueError(
            "inpainting.consistency.preserve_unmasked_pixels must remain true"
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
        source_signature = _source_signature(
            reference=reference,
            source_image_path=raw_path,
            source_mask_path=Path(source_mask_value),
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
        try:
            original_image = Image.open(raw_path).convert("RGB")
            original = np.asarray(original_image).copy()
            repair_mask, mode, mask_reasons = _repair_mask_for_reference(
                reference,
                raw_shape=original.shape[:2],
                config=config.inpainting,
            )
            if repair_mask is None or mode is None or not repair_mask.any():
                if mask_reasons:
                    metadata = _metadata_base(
                        config=config.inpainting,
                        mode=mode,
                        repair_area_ratio=(
                            float(repair_mask.mean())
                            if repair_mask is not None
                            else 0.0
                        ),
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
            processed += 1
            repair_area_ratio = float(repair_mask.mean())
            if mask_reasons:
                metadata = _metadata_base(
                    config=config.inpainting,
                    mode=mode,
                    repair_area_ratio=repair_area_ratio,
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
            generated_image = engine.inpaint(
                image=original_image,
                mask=repair_mask_image,
                prompt=_inpainting_prompt(reference, mode),
                seed=config.inpainting.seed,
            ).convert("RGB")
            generated = np.asarray(generated_image)
            final, feather_ring = _hard_composite(
                original=original,
                generated=generated,
                core_mask=repair_mask,
                feather_pixels=config.inpainting.feather_pixels,
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
            validation = active_validator(
                original=original_image,
                repaired=Image.fromarray(final),
                repair_mask=repair_mask_image,
                reference=reference,
                mode=mode,
            )
            rejection_reasons = [
                str(reason)
                for reason in validation.get("rejection_reasons", [])
            ]
            if (
                unmasked_l1_diff
                > config.inpainting.consistency.maximum_unmasked_l1_diff
            ):
                rejection_reasons.append("unmasked_pixel_difference")
            dino_similarity_value = validation.get("dino_similarity")
            dino_similarity = (
                float(dino_similarity_value)
                if isinstance(dino_similarity_value, (int, float))
                else None
            )
            if (
                dino_similarity is not None
                and dino_similarity
                < config.inpainting.consistency.minimum_dino_similarity
            ):
                rejection_reasons.append("dino_similarity")
            if validation.get("accepted") is not True:
                rejection_reasons.append("consistency_validator")
            if isinstance(engine, NoOpInpaintBackend):
                rejection_reasons.append("noop_backend_test_only")
            rejection_reasons = list(dict.fromkeys(rejection_reasons))
            accepted = not rejection_reasons
            metadata = _metadata_base(
                config=config.inpainting,
                mode=mode,
                repair_area_ratio=repair_area_ratio,
                source_signature=source_signature,
            )
            metadata.update(
                {
                    "unmasked_l1_diff": unmasked_l1_diff,
                    "dino_similarity": dino_similarity,
                    "accepted": accepted,
                    "rejection_reasons": rejection_reasons,
                    "validator": validation,
                    "lossless_storage": True,
                }
            )
            if accepted:
                repaired_path = artifact.parent / "canonical_repaired.png"
                _save_lossless_atomic(Image.fromarray(final), repaired_path)
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
                mode=None,
                repair_area_ratio=0.0,
                source_signature=source_signature,
            )
            metadata.update(
                {
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
