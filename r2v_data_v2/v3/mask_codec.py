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
    counts: list[int] = []
    current = 0
    run_length = 0
    for item in flattened:
        pixel = int(item)
        if pixel == current:
            run_length += 1
            continue
        counts.append(run_length)
        current = pixel
        run_length = 1
    counts.append(run_length)
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
    values: list[np.ndarray] = []
    for index, count in enumerate(encoded.counts):
        values.append(
            np.full(count, index % 2, dtype=bool)
        )
    flattened = np.concatenate(values)
    height, width = encoded.size
    return flattened.reshape((height, width))
