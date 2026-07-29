from __future__ import annotations

from pathlib import Path

INPAINTING_ARTIFACT_NAMES = (
    "repair_mask.png",
    "canonical_repaired.jpg",
    "canonical_repaired.png",
    "inpainting_metadata.json",
    "mask_raw.png",
    "foreground_rgba_raw.png",
    "neutral_background_raw.jpg",
    "dinov3_embedding_raw.npy",
)


def invalidate_inpainting_artifacts(reference_dir: Path) -> None:
    for filename in INPAINTING_ARTIFACT_NAMES:
        (reference_dir / filename).unlink(missing_ok=True)
