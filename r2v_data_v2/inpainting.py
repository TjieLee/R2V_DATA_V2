from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from r2v_data_v2.config import InpaintingConfig, PipelineConfig
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic

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
        generator = self._torch.Generator(device=self.config.device).manual_seed(
            seed
        )
        result = self._pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        )
        if not getattr(result, "images", None):
            raise RuntimeError("FLUX.1 Fill returned no image")
        return result.images[0].convert("RGB")


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


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"inpainting mask is unreadable: {path}")
    if mask.shape != shape:
        raise ValueError("inpainting mask dimensions must match the raw reference")
    return mask >= 128


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


def _fallback_to_raw(
    *,
    reference: dict[str, object],
    artifact: Path,
    raw_path: Path,
    canonical_path: Path,
    metadata: dict[str, object],
    rejected: bool,
) -> None:
    _copy_atomic(raw_path, canonical_path)
    reference.update(
        {
            "canonical_path": str(canonical_path),
            "inpainted": False,
            "inpainting_metadata_path": str(
                artifact.parent / "inpainting_metadata.json"
            ),
            "rejected": rejected,
        }
    )
    write_json_atomic(artifact.parent / "inpainting_metadata.json", metadata)
    write_json_atomic(artifact, reference)


def run_inpainting(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    backend: InpaintBackend | None = None,
    validator: ConsistencyValidator | None = None,
) -> InpaintingStats:
    if not config.inpainting.enabled:
        return InpaintingStats(skipped_disabled=1)
    if not config.inpainting.consistency.preserve_unmasked_pixels:
        raise ValueError(
            "inpainting.consistency.preserve_unmasked_pixels must remain true"
        )
    output_root = config.ensure_output_root()
    engine = backend
    if engine is None:
        engine = (
            NoOpInpaintBackend()
            if config.inpainting.backend == "noop"
            else Flux1FillBackend(config.inpainting)
        )

    processed = skipped = repaired_count = fallback_count = rejected = failed = 0
    for artifact in _reference_artifacts(output_root):
        reference = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(reference, dict):
            raise TypeError(f"reference metadata must be an object: {artifact}")
        inpainting_metadata_path = artifact.parent / "inpainting_metadata.json"
        if inpainting_metadata_path.is_file() and not overwrite:
            existing = json.loads(
                inpainting_metadata_path.read_text(encoding="utf-8")
            )
            processed += 1
            if existing.get("accepted") is True:
                repaired_count += 1
            elif existing.get("fallback_to_raw") is True:
                fallback_count += 1
            continue
        raw_path = Path(
            str(reference.get("raw_canonical_path") or reference["canonical_path"])
        )
        canonical_path = artifact.parent / "canonical.jpg"
        if not raw_path.is_file():
            raise FileNotFoundError(f"raw canonical reference is missing: {raw_path}")
        if "raw_canonical_path" not in reference:
            new_raw_path = artifact.parent / "canonical_raw.jpg"
            _copy_atomic(raw_path, new_raw_path)
            raw_path = new_raw_path
            reference["raw_canonical_path"] = str(raw_path)
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
                    metadata: dict[str, object] = {
                        "backend": config.inpainting.backend,
                        "model_path": (
                            str(config.inpainting.model_path)
                            if config.inpainting.model_path is not None
                            else None
                        ),
                        "mode": mode,
                        "seed": config.inpainting.seed,
                        "repair_area_ratio": (
                            float(repair_mask.mean())
                            if repair_mask is not None
                            else 0.0
                        ),
                        "unmasked_l1_diff": 0.0,
                        "dino_similarity": None,
                        "accepted": False,
                        "fallback_to_raw": (
                            config.inpainting.consistency.fallback_to_raw
                        ),
                        "rejection_reasons": mask_reasons,
                    }
                    _fallback_to_raw(
                        reference=reference,
                        artifact=artifact,
                        raw_path=raw_path,
                        canonical_path=canonical_path,
                        metadata=metadata,
                        rejected=not (
                            config.inpainting.consistency.fallback_to_raw
                        ),
                    )
                    rejected += 1
                    if config.inpainting.consistency.fallback_to_raw:
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
                    skipped += 1
                continue
            processed += 1
            repair_area_ratio = float(repair_mask.mean())
            if mask_reasons:
                metadata = {
                    "backend": config.inpainting.backend,
                    "model_path": (
                        str(config.inpainting.model_path)
                        if config.inpainting.model_path is not None
                        else None
                    ),
                    "mode": mode,
                    "seed": config.inpainting.seed,
                    "repair_area_ratio": repair_area_ratio,
                    "unmasked_l1_diff": 0.0,
                    "dino_similarity": None,
                    "accepted": False,
                    "fallback_to_raw": (
                        config.inpainting.consistency.fallback_to_raw
                    ),
                    "rejection_reasons": mask_reasons,
                }
                _fallback_to_raw(
                    reference=reference,
                    artifact=artifact,
                    raw_path=raw_path,
                    canonical_path=canonical_path,
                    metadata=metadata,
                    rejected=not config.inpainting.consistency.fallback_to_raw,
                )
                rejected += 1
                if config.inpainting.consistency.fallback_to_raw:
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
            validation = (
                validator(
                    original=original_image,
                    repaired=Image.fromarray(final),
                    repair_mask=repair_mask_image,
                    reference=reference,
                    mode=mode,
                )
                if validator is not None
                else {"accepted": True}
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
            rejection_reasons = list(dict.fromkeys(rejection_reasons))
            accepted = not rejection_reasons
            metadata = {
                "backend": config.inpainting.backend,
                "model_path": (
                    str(config.inpainting.model_path)
                    if config.inpainting.model_path is not None
                    else None
                ),
                "mode": mode,
                "seed": config.inpainting.seed,
                "repair_area_ratio": repair_area_ratio,
                "unmasked_l1_diff": unmasked_l1_diff,
                "dino_similarity": dino_similarity,
                "accepted": accepted,
                "fallback_to_raw": (
                    not accepted
                    and config.inpainting.consistency.fallback_to_raw
                ),
                "rejection_reasons": rejection_reasons,
                "validator": validation,
                "lossless_storage": True,
            }
            if accepted:
                repaired_path = artifact.parent / "canonical_repaired.jpg"
                _save_lossless_atomic(Image.fromarray(final), repaired_path)
                _copy_atomic(repaired_path, canonical_path)
                write_json_atomic(inpainting_metadata_path, metadata)
                reference.update(
                    {
                        "canonical_path": str(canonical_path),
                        "inpainted": True,
                        "inpainting_metadata_path": str(
                            inpainting_metadata_path
                        ),
                        "rejected": False,
                    }
                )
                write_json_atomic(artifact, reference)
                repaired_count += 1
            else:
                _fallback_to_raw(
                    reference=reference,
                    artifact=artifact,
                    raw_path=raw_path,
                    canonical_path=canonical_path,
                    metadata=metadata,
                    rejected=not config.inpainting.consistency.fallback_to_raw,
                )
                rejected += 1
                if config.inpainting.consistency.fallback_to_raw:
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
            if config.inpainting.consistency.fallback_to_raw:
                _copy_atomic(raw_path, canonical_path)
                reference["canonical_path"] = str(canonical_path)
                reference["inpainted"] = False
                write_json_atomic(artifact, reference)
                fallback_count += 1
    reconcile_references(output_root)
    return InpaintingStats(
        processed=processed,
        skipped_no_repair_needed=skipped,
        repaired=repaired_count,
        fallback_to_raw=fallback_count,
        rejected=rejected,
        failed=failed,
    )
