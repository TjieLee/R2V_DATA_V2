from __future__ import annotations

import pytest

from r2v_data_v2.config import MetricNormalizationConfig
from r2v_data_v2.normalization import normalize_metric_values


def test_identity_clips_values_to_unit_interval() -> None:
    assert normalize_metric_values(
        [-0.2, 0.3, 1.4],
        MetricNormalizationConfig(method="identity"),
    ) == [0.0, 0.3, 1.0]


def test_minmax_maps_range_and_constant_values() -> None:
    policy = MetricNormalizationConfig(method="minmax")
    assert normalize_metric_values([10.0, 20.0, 30.0], policy) == [
        0.0,
        0.5,
        1.0,
    ]
    assert normalize_metric_values([7.0, 7.0], policy) == [0.5, 0.5]


def test_robust_minmax_limits_single_extreme_value() -> None:
    values = [float(value) for value in range(9)] + [1000.0]
    robust = normalize_metric_values(
        values,
        MetricNormalizationConfig(method="robust_minmax"),
    )
    ordinary = normalize_metric_values(
        values,
        MetricNormalizationConfig(method="minmax"),
    )

    assert robust[-1] == 1.0
    assert robust[4] > ordinary[4]


def test_robust_minmax_falls_back_below_four_candidates() -> None:
    assert normalize_metric_values(
        [10.0, 20.0, 30.0],
        MetricNormalizationConfig(method="robust_minmax"),
    ) == [0.0, 0.5, 1.0]


def test_fixed_range_uses_configured_bounds() -> None:
    assert normalize_metric_values(
        [0.5, 0.6, 0.775, 0.95, 1.0],
        MetricNormalizationConfig(
            method="fixed_range",
            minimum=0.60,
            maximum=0.95,
        ),
    ) == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
