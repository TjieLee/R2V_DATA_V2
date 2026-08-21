from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from r2v_data_v2.v3.mask_codec import encode_binary_mask
from tools.compare_v3_sam3_compile_runs import compare_runs


def _write_run(
    root: Path,
    *,
    shifted: bool = False,
    source_frame_index: int = 5,
    retained_entity_ids: list[str] | None = None,
) -> None:
    clip_root = root / "clips" / "clip-a"
    clip_root.mkdir(parents=True)
    clip = {
        "clip_uid": "clip-a",
        "coverage": {"passed": True, "qualifying_entity_ids": ["e1"]},
        "references": {
            "entities": [
                {
                    "entity_id": "e1",
                    "status": "ready",
                    "reason": None,
                    "source_clip_uid": "clip-a",
                    "source_entity_id": "e1",
                    "source_frame_index": source_frame_index,
                    "source_frame_slot": 5,
                    "source_image_path": "frames/05.jpg",
                }
            ]
        },
        "pairing": {
            "status": "ready",
            "retained_entity_ids": (
                ["e1"] if retained_entity_ids is None else retained_entity_ids
            ),
        },
    }
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1 + int(shifted) : 3 + int(shifted)] = 1
    frames = [
        {
            "slot": slot,
            "rle": encode_binary_mask(mask).model_dump(mode="json"),
        }
        for slot in range(10)
    ]
    masks = {
        "clip_uid": "clip-a",
        "entities": {
            "e1": {
                "status": "ready",
                "reason": None,
                "reference_type": "subject",
                "grounding_prompt": "the person",
                "backend_object_ids": ["1"],
                "frames": frames,
            }
        },
    }
    (clip_root / "clip.json").write_text(json.dumps(clip), encoding="utf-8")
    (clip_root / "masks.rle.json").write_text(
        json.dumps(masks),
        encoding="utf-8",
    )


def test_compile_comparison_reports_exact_artifacts(tmp_path: Path) -> None:
    eager = tmp_path / "eager"
    compiled = tmp_path / "compiled"
    _write_run(eager)
    _write_run(compiled)

    result = compare_runs(eager, compiled)

    assert result["all_exact"] is True
    assert result["exactly_equal_clips"] == 1
    assert result["exactly_equal_entity_tracks"] == 1
    assert result["exact_masks"] == 1
    assert result["minimum_nonexact_mask_iou"] is None


def test_compile_comparison_reports_mask_difference_and_iou(tmp_path: Path) -> None:
    eager = tmp_path / "eager"
    compiled = tmp_path / "compiled"
    _write_run(eager)
    _write_run(compiled, shifted=True)

    result = compare_runs(eager, compiled)

    assert result["all_exact"] is False
    assert result["exactly_equal_clips"] == 0
    assert result["exact_masks"] == 0
    assert result["minimum_nonexact_mask_iou"] == 1 / 3
    entity = result["clips"][0]["entities"]["e1"]
    assert entity["status_reason_exact"] is True
    assert entity["provenance_exact"] is True


def test_compile_comparison_detects_reference_and_retained_differences(
    tmp_path: Path,
) -> None:
    eager = tmp_path / "eager"
    compiled = tmp_path / "compiled"
    _write_run(eager)
    _write_run(compiled, source_frame_index=8, retained_entity_ids=[])

    result = compare_runs(eager, compiled)

    assert result["differing_reference_selections"] == 1
    assert result["differing_retained_entity_sets"] == 1
