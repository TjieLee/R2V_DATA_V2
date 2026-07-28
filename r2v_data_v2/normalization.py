from __future__ import annotations

import numpy as np

from r2v_data_v2.config import MetricNormalizationConfig

_EPSILON = 1e-12


def normalize_metric_values(
    values: list[float],
    policy: MetricNormalizationConfig,
) -> list[float]:
    if not values:
        return []
    raw = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("metric values must be finite")

    if policy.method == "identity":
        normalized = np.clip(raw, 0.0, 1.0)
    elif policy.method == "minmax":
        normalized = _minmax(raw, float(np.min(raw)), float(np.max(raw)))
    elif policy.method == "robust_minmax":
        if len(raw) < 4:
            low, high = float(np.min(raw)), float(np.max(raw))
        else:
            low = float(np.quantile(raw, 0.10))
            high = float(np.quantile(raw, 0.90))
        normalized = _minmax(raw, low, high)
    elif policy.method == "fixed_range":
        if (
            policy.minimum is None
            or policy.maximum is None
            or policy.minimum >= policy.maximum
        ):
            raise ValueError("fixed_range requires minimum < maximum")
        normalized = np.clip(
            (raw - policy.minimum) / (policy.maximum - policy.minimum),
            0.0,
            1.0,
        )
    else:
        raise ValueError(f"unsupported normalization method: {policy.method}")
    return [float(value) for value in normalized]


def _minmax(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high - low < _EPSILON:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)
