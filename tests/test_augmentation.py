from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.augmentation import augment_references
from r2v_data_v2.config import load_config


def _write_config(tmp_path: Path, *, enabled: bool) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"dataset_json: {tmp_path / 'source.jsonl'}\n"
        f"output_root: {tmp_path / 'output'}\n"
        "augmentation:\n"
        f"  enabled: {str(enabled).lower()}\n"
        "  generated_background_count: 1\n"
        "  viewpoint_count: 0\n",
        encoding="utf-8",
    )
    return config_path


def test_disabled_augmentation_does_not_call_or_load_editors(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, enabled=False)

    stats = augment_references(load_config(config_path))

    assert stats.disabled
    assert stats.processed == 0
    assert stats.skipped_existing == 0
    assert not (tmp_path / "output" / "manifests" / "augmentations.jsonl").exists()


def test_augmentation_restores_core_and_skips_existing_variant(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path, enabled=True))
    reference_dir = tmp_path / "output" / "references" / "clip-1" / "e1"
    reference_dir.mkdir(parents=True)
    canonical = np.full((32, 32, 3), (20, 100, 220), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[6:26, 6:26] = 255
    cv2.imwrite(str(reference_dir / "canonical.png"), canonical)
    cv2.imwrite(str(reference_dir / "foreground_rgba.png"), canonical)
    cv2.imwrite(str(reference_dir / "mask.png"), mask)
    manifest = tmp_path / "output" / "manifests" / "references.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "entity_id": "e1",
                "canonical_path": str(reference_dir / "canonical.png"),
                "foreground_rgba_path": str(reference_dir / "foreground_rgba.png"),
                "mask_path": str(reference_dir / "mask.png"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def editor(
        canonical_path: Path,
        rgba_path: Path,
        destination: Path,
        index: int,
    ) -> Path:
        nonlocal calls
        del canonical_path, rgba_path
        calls += 1
        candidate = destination / f"background_{index}.png"
        cv2.imwrite(str(candidate), np.zeros_like(canonical))
        return candidate

    def validator(
        canonical_path: Path,
        candidate_path: Path,
        mask_path: Path,
        variant_type: str,
    ) -> dict[str, object]:
        del canonical_path, candidate_path, mask_path, variant_type
        return {"accepted": True}

    first = augment_references(
        config,
        background_editor=editor,
        validator=validator,
    )
    second = augment_references(
        config,
        background_editor=editor,
        validator=validator,
    )

    candidate = cv2.imread(
        str(reference_dir / "augmented" / "background_0.png"),
    )
    assert first.accepted == 1
    assert second.skipped_existing == 1
    assert calls == 1
    assert np.array_equal(candidate[10:22, 10:22], canonical[10:22, 10:22])
    variant = json.loads(
        (tmp_path / "output" / "manifests" / "augmentations.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert variant["pre_restore_core_similarity"] < 1.0
    assert variant["post_restore_core_similarity"] >= 0.995
    assert "foreground_core_similarity" not in variant
