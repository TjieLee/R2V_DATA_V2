from __future__ import annotations

import numpy as np

from r2v_data_v2.v3.schemas import MaskRle


def encode_binary_mask(mask: np.ndarray) -> MaskRle:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("mask dimensions must be positive")
    if not np.isin(value, (0, 1)).all():
        raise ValueError("mask must contain only binary values")
    flattened = value.astype(np.uint8, copy=False).reshape(-1)
    changes = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
    boundaries = np.concatenate(
        (
            np.array([0], dtype=changes.dtype),
            changes,
            np.array([flattened.size], dtype=changes.dtype),
        )
    )
    counts = np.diff(boundaries).tolist()
    if flattened[0] == 1:
        counts.insert(0, 0)
    return MaskRle(
        size=(int(value.shape[0]), int(value.shape[1])),
        counts=counts,
    )


def decode_binary_mask(rle: MaskRle | dict[str, object]) -> np.ndarray:
    encoded = (
        rle
        if isinstance(rle, MaskRle)
        else MaskRle.model_validate(rle)
    )
    counts = np.asarray(encoded.counts, dtype=np.intp)
    flattened = np.repeat(
        np.arange(counts.size, dtype=np.uint8) & 1,
        counts,
    ).astype(bool, copy=False)
    height, width = encoded.size
    return flattened.reshape((height, width))
