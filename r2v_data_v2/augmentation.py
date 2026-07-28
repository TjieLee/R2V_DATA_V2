from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.config import PipelineConfig
from r2v_data_v2.manifest import iter_source_records

Editor = Callable[[Path, Path, Path, int], Path]
VariantValidator = Callable[[Path, Path, Path, str], dict[str, object]]


@dataclass(frozen=True)
class AugmentationStats:
    disabled: bool = False
    processed: int = 0
    skipped_existing: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0


def _foreground_core(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask >= 250, dtype=np.uint8)
    eroded = cv2.erode(binary, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return eroded.astype(bool) if eroded.any() else binary.astype(bool)


def foreground_core_similarity(
    *,
    canonical_path: Path,
    candidate_path: Path,
    mask_path: Path,
) -> float:
    canonical = cv2.imread(str(canonical_path))
    candidate = cv2.imread(str(candidate_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if canonical is None or candidate is None or mask is None:
        raise FileNotFoundError("augmentation input or output image is unreadable")
    if canonical.shape != candidate.shape or canonical.shape[:2] != mask.shape:
        return 0.0
    foreground_core = _foreground_core(mask)
    if not foreground_core.any():
        return 0.0
    difference = np.abs(
        canonical[foreground_core].astype(np.float32)
        - candidate[foreground_core].astype(np.float32)
    )
    return float(max(0.0, 1.0 - difference.mean() / 255.0))


def _restore_foreground_core(
    *,
    canonical_path: Path,
    candidate_path: Path,
    mask_path: Path,
) -> None:
    canonical = cv2.imread(str(canonical_path))
    candidate = cv2.imread(str(candidate_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if canonical is None or candidate is None or mask is None:
        raise FileNotFoundError("augmentation input or output image is unreadable")
    if canonical.shape != candidate.shape or canonical.shape[:2] != mask.shape:
        raise ValueError("generated variant dimensions must match the canonical image")
    foreground_core = _foreground_core(mask)
    candidate[foreground_core] = canonical[foreground_core]
    if not cv2.imwrite(str(candidate_path), candidate):
        raise OSError(f"failed to preserve foreground core in {candidate_path}")


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _try_variant(
    *,
    reference: dict[str, object],
    variant_type: str,
    variant_index: int,
    destination: Path,
    editor: Editor | None,
    validator: VariantValidator | None,
) -> tuple[dict[str, object] | None, str | None]:
    if editor is None or validator is None:
        return None, f"{variant_type} editor and identity validator are required"
    canonical = Path(str(reference["canonical_path"]))
    rgba = Path(str(reference["foreground_rgba_path"]))
    mask = Path(str(reference["mask_path"]))
    destination.mkdir(parents=True, exist_ok=True)
    candidate = Path(editor(canonical, rgba, destination, variant_index)).resolve()
    resolved_destination = destination.resolve()
    if resolved_destination not in candidate.parents:
        raise ValueError(
            "augmentation editor output must remain inside its destination"
        )
    _restore_foreground_core(
        canonical_path=canonical,
        candidate_path=candidate,
        mask_path=mask,
    )
    core_similarity = foreground_core_similarity(
        canonical_path=canonical,
        candidate_path=candidate,
        mask_path=mask,
    )
    validation = validator(canonical, candidate, mask, variant_type)
    if core_similarity < 0.98 or validation.get("accepted") is not True:
        candidate.unlink(missing_ok=True)
        return None, f"{variant_type} failed identity or foreground-core validation"
    return (
        {
            "clip_uid": reference["clip_uid"],
            "entity_id": reference["entity_id"],
            "variant_type": variant_type,
            "variant_index": variant_index,
            "image_path": str(candidate),
            "foreground_core_similarity": core_similarity,
            "validation": validation,
        },
        None,
    )


def _update_final_samples(
    final_path: Path,
    variants: dict[str, list[dict[str, object]]],
) -> None:
    if not final_path.is_file():
        return
    temporary = final_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in iter_source_records(final_path):
            sample["augmentation_variants"] = variants.get(
                str(sample["clip_uid"]),
                [],
            )
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    temporary.replace(final_path)


def augment_references(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    background_editor: Editor | None = None,
    viewpoint_editor: Editor | None = None,
    validator: VariantValidator | None = None,
) -> AugmentationStats:
    output_root = config.ensure_output_root()
    if not config.augmentation.enabled:
        return AugmentationStats(disabled=True)
    reference_path = output_root / "manifests" / "references.jsonl"
    augmentation_path = output_root / "manifests" / "augmentations.jsonl"
    if not reference_path.is_file():
        raise FileNotFoundError("run Stage 04 before augmentation")
    if overwrite:
        augmentation_path.unlink(missing_ok=True)
    existing_variants: dict[tuple[str, str, str, int], dict[str, object]] = {}
    if augmentation_path.is_file():
        for variant in iter_source_records(augmentation_path):
            key = (
                str(variant["clip_uid"]),
                str(variant["entity_id"]),
                str(variant["variant_type"]),
                int(variant["variant_index"]),
            )
            existing_variants[key] = variant
    processed = skipped = accepted = rejected = failed = 0
    variants_by_clip: dict[str, list[dict[str, object]]] = {}
    for variant in existing_variants.values():
        variants_by_clip.setdefault(str(variant["clip_uid"]), []).append(variant)
    for reference in iter_source_records(reference_path):
        clip = str(reference["clip_uid"])
        entity = str(reference["entity_id"])
        destination = output_root / "references" / clip / entity / "augmented"
        jobs = [
            (
                "generated_background",
                index,
                background_editor,
            )
            for index in range(config.augmentation.generated_background_count)
        ]
        jobs.extend(
            ("viewpoint", index, viewpoint_editor)
            for index in range(config.augmentation.viewpoint_count)
        )
        for variant_type, index, editor in jobs:
            key = (clip, entity, variant_type, index)
            existing = existing_variants.get(key)
            if existing is not None and Path(str(existing["image_path"])).is_file():
                skipped += 1
                continue
            try:
                variant, reason = _try_variant(
                    reference=reference,
                    variant_type=variant_type,
                    variant_index=index,
                    destination=destination,
                    editor=editor,
                    validator=validator,
                )
                if variant is None:
                    rejected += 1
                    _append_jsonl(
                        output_root / "logs" / "augmentation_failed.jsonl",
                        {
                            "clip_uid": clip,
                            "entity_id": entity,
                            "variant_type": variant_type,
                            "variant_index": index,
                            "error": reason,
                        },
                    )
                    continue
                _append_jsonl(augmentation_path, variant)
                variants_by_clip.setdefault(clip, []).append(variant)
                accepted += 1
            except Exception as exc:  # noqa: BLE001 - preserve canonical on failure
                _append_jsonl(
                    output_root / "logs" / "augmentation_failed.jsonl",
                    {
                        "clip_uid": clip,
                        "entity_id": entity,
                        "variant_type": variant_type,
                        "variant_index": index,
                        "error": str(exc),
                    },
                )
                failed += 1
        processed += 1
    _update_final_samples(
        output_root / "manifests" / "final_samples.jsonl",
        variants_by_clip,
    )
    return AugmentationStats(False, processed, skipped, accepted, rejected, failed)


def stats_dict(stats: AugmentationStats) -> dict[str, int | bool]:
    return asdict(stats)
