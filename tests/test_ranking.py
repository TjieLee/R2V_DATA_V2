from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.config import RankingConfig
from r2v_data_v2.metrics import CandidateMetrics, classify_entity_overlap
from r2v_data_v2.ranking import (
    _candidate_sheet_label,
    _ensure_best_frame_has_no_hard_rejection,
    _top_candidate_sheet,
    rank_candidates,
)
from r2v_data_v2.schemas import CandidateVisualReview


def _metrics(*, border_touch: bool, sharpness: float) -> CandidateMetrics:
    return CandidateMetrics(
        effective_short_side=300,
        mask_area_ratio=0.2,
        border_touch=border_touch,
        laplacian_variance=sharpness,
        tenengrad_sharpness=sharpness,
        exposure_score=0.9,
        mask_area_continuity=0.9,
        maximum_other_mask_overlap=0.0,
        maximum_bbox_containment=0.0,
        crop_subject_ratio=0.7,
    )


def _review(slot: int, visual_quality: float = 0.8) -> CandidateVisualReview:
    return CandidateVisualReview(
        frame_slot=slot,
        completeness=0.9,
        recognizability=0.9,
        occlusion=0.1,
        mask_quality=0.9,
        visual_quality=visual_quality,
        identity_features_visible=True,
        rejection_reasons=[],
    )


def test_hard_gate_beats_weighted_score() -> None:
    config = RankingConfig()
    invalid = (
        0,
        10,
        0.99,
        _metrics(border_touch=True, sharpness=1000),
        _review(0, 1.0),
    )
    valid = (
        1,
        20,
        0.70,
        _metrics(border_touch=False, sharpness=10),
        _review(1, 0.5),
    )

    ranked = rank_candidates([invalid, valid], config=config)

    assert ranked[0].frame_slot == 1
    assert ranked[0].hard_rejection_reasons == ()
    assert "border_touch" in ranked[1].hard_rejection_reasons


def test_low_completeness_is_hard_failure() -> None:
    review = _review(0).model_copy(update={"completeness": 0.2})
    ranked = rank_candidates(
        [(0, 10, 0.9, _metrics(border_touch=False, sharpness=10), review)],
        config=RankingConfig(),
    )
    assert "incomplete" in ranked[0].hard_rejection_reasons


def test_entity_overlap_classification() -> None:
    attached = classify_entity_overlap(
        statistics={
            "mask_iou": 0.4,
            "left_contained_in_right": 0.85,
            "right_contained_in_left": 0.2,
            "bbox_iou": 0.5,
        },
        temporal_cooccurrence=0.9,
        relation="holding",
        child_has_independent_candidate=False,
    )
    important = classify_entity_overlap(
        statistics={
            "mask_iou": 0.4,
            "left_contained_in_right": 0.85,
            "right_contained_in_left": 0.2,
            "bbox_iou": 0.5,
        },
        temporal_cooccurrence=0.9,
        relation="holding",
        child_has_independent_candidate=True,
    )
    duplicate = classify_entity_overlap(
        statistics={
            "mask_iou": 0.9,
            "left_contained_in_right": 0.95,
            "right_contained_in_left": 0.95,
            "bbox_iou": 0.9,
        },
        temporal_cooccurrence=1.0,
        relation=None,
        child_has_independent_candidate=False,
    )

    assert attached == "attached_accessory"
    assert important == "important_independent_object"
    assert duplicate == "duplicate_entity"


def test_candidate_sheet_contains_three_panels_and_metric_label(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame.jpg"
    frame = np.full((120, 180, 3), (30, 120, 220), dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), frame)
    mask = np.zeros((120, 180), dtype=bool)
    mask[25:100, 55:135] = True
    metrics = _metrics(border_touch=False, sharpness=123.4)
    record: dict[str, object] = {
        "frame_slot": 2,
        "sam_confidence": 0.91,
    }
    destination = tmp_path / "sheet.jpg"

    _top_candidate_sheet(
        frame_paths={2: frame_path},
        masks={2: mask},
        candidates=[(record, metrics)],
        destination=destination,
    )

    with Image.open(destination) as sheet:
        assert sheet.size == (1440, 362)
    label = _candidate_sheet_label(record, metrics)
    assert "frame_slot=2" in label
    assert "SAM=0.910" in label
    assert "effective_short_side=300" in label
    assert "mask_area_ratio=0.2000" in label
    assert "Tenengrad=123.4" in label
    assert "border_touch=false" in label


def test_qwen_best_frame_must_not_have_hard_rejection() -> None:
    ranked = rank_candidates(
        [
            (
                0,
                10,
                0.99,
                _metrics(border_touch=True, sharpness=100),
                _review(0),
            ),
            (
                1,
                20,
                0.90,
                _metrics(border_touch=False, sharpness=80),
                _review(1),
            ),
        ],
        config=RankingConfig(),
    )

    _ensure_best_frame_has_no_hard_rejection(ranked, 1)
    with pytest.raises(ValueError, match="hard rejection"):
        _ensure_best_frame_has_no_hard_rejection(ranked, 0)
