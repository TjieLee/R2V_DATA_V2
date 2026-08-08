from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.reference_prefilter as prefilter_module
from r2v_data_v2.v3.pair import EntityReferenceCandidate
from r2v_data_v2.v3.reference_prefilter import (
    NEAR_SILHOUETTE_RULE,
    RELATIVE_BLUR_V2_RULE,
    _subject_near_silhouette,
    _subject_relative_blur_v2,
    prefilter_entity_reference_candidates,
)
from r2v_data_v2.v3.schemas import AnnotationEntity


def _entity(reference_type: str = "subject") -> AnnotationEntity:
    return AnnotationEntity(
        entity_id="e1",
        reference_type=reference_type,
        phrase="visible entity",
        grounding_prompt="the visible entity",
    )


def _candidate(index: int) -> EntityReferenceCandidate:
    mask = np.ones((6, 8), dtype=bool)
    return EntityReferenceCandidate(
        candidate_id=f"candidate_{index}",
        entity_id="e1",
        frame_slot=index,
        source_frame_index=index * 10,
        image_path=f"frame-{index}.png",
        mask=mask,
        bbox_xyxy=(0, 0, 8, 6),
        area_pixels=int(mask.sum()),
        area_ratio=1.0,
        bbox_fill_ratio=1.0,
        border_contact_count=4,
        normalized_center_distance=0.0,
    )


def _metrics(
    *,
    luma: float = 90.0,
    dark_fraction: float = 0.0,
    laplacian: float = 100.0,
    tenengrad: float = 2000.0,
) -> dict[str, object]:
    return {
        "status": "succeeded",
        "backend": "cheap_cv",
        "luma_mean": luma,
        "dark_fraction_32": dark_fraction,
        "laplacian_variance": laplacian,
        "tenengrad_mean": tenengrad,
    }


def _source_images(count: int) -> dict[str, Image.Image]:
    return {
        f"frame-{index}.png": Image.new("RGB", (8, 6), (index, index, index))
        for index in range(1, count + 1)
    }


def _install_metrics(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[int, dict[str, object]],
) -> None:
    def fake_metrics(source: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        assert mask.shape == source.shape[:2]
        return dict(values[int(source[0, 0, 0])])

    monkeypatch.setattr(
        prefilter_module,
        "cheap_foreground_technical_metrics",
        fake_metrics,
    )


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (_metrics(luma=15, dark_fraction=0.95, laplacian=5, tenengrad=100), True),
        (_metrics(luma=16, dark_fraction=0.95, laplacian=5, tenengrad=100), False),
        (_metrics(luma=15, dark_fraction=0.94, laplacian=5, tenengrad=100), False),
        (_metrics(luma=15, dark_fraction=0.95, laplacian=6, tenengrad=100), False),
        (_metrics(luma=15, dark_fraction=0.95, laplacian=5, tenengrad=101), False),
    ],
)
def test_near_silhouette_v1_keeps_frozen_conjunction(
    metrics: dict[str, object],
    expected: bool,
) -> None:
    assert _subject_near_silhouette(metrics) is expected


@pytest.mark.parametrize(
    ("metrics", "laplacian_ratio", "tenengrad_ratio", "expected"),
    [
        (_metrics(laplacian=50, tenengrad=1500), 0.35, 0.50, True),
        (_metrics(laplacian=51, tenengrad=1500), 0.35, 0.50, False),
        (_metrics(laplacian=50, tenengrad=1501), 0.35, 0.50, False),
        (_metrics(laplacian=50, tenengrad=1500), 0.36, 0.50, False),
        (_metrics(laplacian=50, tenengrad=1500), 0.35, 0.51, False),
    ],
)
def test_relative_blur_v2_keeps_frozen_relative_and_absolute_thresholds(
    metrics: dict[str, object],
    laplacian_ratio: float,
    tenengrad_ratio: float,
    expected: bool,
) -> None:
    assert (
        _subject_relative_blur_v2(
            metrics,
            laplacian_ratio=laplacian_ratio,
            tenengrad_ratio=tenengrad_ratio,
        )
        is expected
    )


def test_three_candidate_relative_blur_filters_without_renumbering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_candidate(index) for index in range(1, 4)]
    _install_metrics(
        monkeypatch,
        {
            1: _metrics(laplacian=100, tenengrad=1000),
            2: _metrics(laplacian=30, tenengrad=400),
            3: _metrics(laplacian=90, tenengrad=900),
        },
    )

    result = prefilter_entity_reference_candidates(
        _entity(),
        candidates,
        _source_images(3),
    )

    assert [item.candidate_id for item in result.original_candidates] == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
    ]
    assert [item.candidate_id for item in result.retained_candidates] == [
        "candidate_1",
        "candidate_3",
    ]
    assert result.decisions[1].flagged_by == (RELATIVE_BLUR_V2_RULE,)
    assert result.decisions[1].laplacian_ratio == pytest.approx(0.30)
    assert result.decisions[1].tenengrad_ratio == pytest.approx(0.40)
    assert candidates[1].candidate_id == "candidate_2"


def test_two_candidate_near_silhouette_applies_but_relative_blur_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_candidate(1), _candidate(2)]
    _install_metrics(
        monkeypatch,
        {
            1: _metrics(
                luma=10,
                dark_fraction=0.99,
                laplacian=4,
                tenengrad=40,
            ),
            2: _metrics(laplacian=100, tenengrad=100),
        },
    )

    result = prefilter_entity_reference_candidates(
        _entity(),
        candidates,
        _source_images(2),
    )

    assert [item.candidate_id for item in result.retained_candidates] == [
        "candidate_2"
    ]
    first = result.decisions[0]
    assert first.flagged_by == (NEAR_SILHOUETTE_RULE,)
    assert first.laplacian_ratio == pytest.approx(0.04)
    assert first.tenengrad_ratio == pytest.approx(0.40)
    assert first.relative_blur_v2_applicable is False
    assert first.relative_blur_v2_inapplicable_reason == "requires_three_candidates"
    assert RELATIVE_BLUR_V2_RULE not in first.flagged_by


@pytest.mark.parametrize("reference_type", ["object", "group"])
def test_object_and_group_are_never_measured_or_filtered(
    monkeypatch: pytest.MonkeyPatch,
    reference_type: str,
) -> None:
    def unexpected_metrics(*args: object) -> dict[str, object]:
        del args
        raise AssertionError("object/group technical metrics must not run")

    monkeypatch.setattr(
        prefilter_module,
        "cheap_foreground_technical_metrics",
        unexpected_metrics,
    )
    candidates = [_candidate(index) for index in range(1, 4)]

    result = prefilter_entity_reference_candidates(
        _entity(reference_type),
        candidates,
        _source_images(3),
    )

    assert result.retained_candidates == tuple(candidates)
    assert all(not decision.flagged for decision in result.decisions)
    assert all(
        decision.relative_blur_v2_inapplicable_reason == "subject_only"
        for decision in result.decisions
    )


def test_metric_failure_is_raised_for_pair_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prefilter_module,
        "cheap_foreground_technical_metrics",
        lambda *args: (_ for _ in ()).throw(RuntimeError("metric failed")),
    )

    with pytest.raises(RuntimeError, match="metric failed"):
        prefilter_entity_reference_candidates(
            _entity(),
            [_candidate(1), _candidate(2), _candidate(3)],
            _source_images(3),
        )


def test_prefilter_does_not_mutate_candidate_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(1)
    snapshot = replace(candidate)
    _install_metrics(monkeypatch, {1: _metrics()})

    prefilter_entity_reference_candidates(
        _entity(),
        [candidate],
        _source_images(1),
    )

    assert candidate == snapshot
