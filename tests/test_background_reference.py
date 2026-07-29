from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.background_reference import build_background_references
from r2v_data_v2.config import BackgroundConfig, PipelineConfig
from r2v_data_v2.mask_utils import encode_mask


def _annotation(*, with_entity: bool) -> dict[str, object]:
    entities: list[dict[str, object]] = []
    if with_entity:
        entities.append(
            {
                "entity_id": "e1",
                "phrase": "a red bicycle",
                "grounding_prompt": "red bicycle",
                "canonical_label": "red bicycle",
                "category": "vehicle",
                "reference_worthy": True,
                "salience": "primary",
                "genericity": "descriptive",
                "name_evidence": "none",
                "separability": "independent",
                "selection_reason": "primary subject",
                "ref_token": "<ref_subject_1>",
            }
        )
    return {
        "clip_uid": "clip-1",
        "caption": "A red bicycle is parked beside a brick wall.",
        "prompt_with_refs": (
            "<ref_subject_1> is parked beside a brick wall."
            if with_entity
            else "A quiet brick courtyard at noon."
        ),
        "entities": entities,
        "relations": [],
        "background": {
            "phrase": "a quiet brick courtyard at noon",
            "grounding_prompt": "quiet brick courtyard, daylight",
            "reference_worthy": True,
            "ref_token": None,
        },
    }


def _write_fixture(
    output_root: Path,
    *,
    foreground_ratio: float | None,
) -> None:
    manifest = output_root / "manifests" / "annotations.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(_annotation(with_entity=foreground_ratio is not None)) + "\n",
        encoding="utf-8",
    )
    frame_dir = output_root / "frames" / "clip-1"
    frame_dir.mkdir(parents=True)
    y, x = np.indices((100, 100))
    frame = np.stack(
        (
            70 + (x % 80),
            80 + (y % 80),
            90 + ((x + y) % 80),
        ),
        axis=-1,
    ).astype(np.uint8)
    for slot in range(10):
        assert cv2.imwrite(str(frame_dir / f"frame_{slot:02d}.jpg"), frame)
    (frame_dir / "frames.json").write_text(
        json.dumps({"sampled_indices": list(range(100, 110))}),
        encoding="utf-8",
    )
    if foreground_ratio is None:
        return
    height = round(100 * foreground_ratio)
    encoded: dict[str, object] = {}
    for slot in range(10):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:height, :] = True
        encoded[f"frame_{slot:02d}"] = encode_mask(mask)
    candidate_dir = output_root / "candidates" / "clip-1" / "e1"
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "top_masks.rle.json").write_text(
        json.dumps(encoded),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    background: BackgroundConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        background=background or BackgroundConfig(),
    )


def _metadata(output_root: Path) -> dict[str, object]:
    return json.loads(
        (
            output_root
            / "references"
            / "clip-1"
            / "bg1"
            / "reference_metadata.json"
        ).read_text(encoding="utf-8")
    )


def test_background_only_clip_produces_raw_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=None)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.processed == 1
    assert stats.raw_background_count == 1
    assert metadata["reference_type"] == "background"
    assert metadata["reference_id"] == "bg1"
    assert metadata["ref_token"] == "<ref_bg_1>"
    assert metadata["entity_id"] is None
    assert metadata["needs_inpainting"] is False
    assert metadata["inpainted"] is False
    assert Path(str(metadata["raw_canonical_path"])).is_file()
    assert Path(str(metadata["canonical_path"])).is_file()
    assert len(list((config.output_root / "references" / "clip-1").iterdir())) == 1


def test_low_foreground_ratio_keeps_raw_frame(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.04)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.raw_background_count == 1
    assert metadata["foreground_area_ratio"] == 0.04
    assert metadata["needs_inpainting"] is False


def test_medium_foreground_ratio_marks_candidate_for_inpainting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_fixture(config.output_root, foreground_ratio=0.10)

    stats = build_background_references(config)
    metadata = _metadata(config.output_root)

    assert stats.needs_inpainting_count == 1
    assert metadata["foreground_area_ratio"] == 0.10
    assert metadata["needs_inpainting"] is True


def test_hole_ratio_over_limit_is_quality_rejection(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        background=BackgroundConfig(maximum_hole_area_ratio=0.08),
    )
    _write_fixture(config.output_root, foreground_ratio=0.10)

    stats = build_background_references(config)

    assert stats.no_valid_candidate == 1
    assert stats.failed == 0
    assert not (
        config.output_root
        / "references"
        / "clip-1"
        / "bg1"
        / "reference_metadata.json"
    ).exists()
    ranking = json.loads(
        (
            config.output_root
            / "background_candidates"
            / "clip-1"
            / "ranking_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        "hole_area_ratio" in candidate["rejection_reasons"]
        for candidate in ranking["candidates"]
    )


def test_disabled_background_stage_is_noop(tmp_path: Path) -> None:
    config = _config(tmp_path, background=BackgroundConfig(enabled=False))

    stats = build_background_references(config)

    assert stats.processed == 0
    assert not config.output_root.exists()
