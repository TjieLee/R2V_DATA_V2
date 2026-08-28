from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from pydantic import ValidationError

from r2v_data_v2.v3.mask_codec import (
    decode_binary_mask,
    encode_binary_mask,
)
from r2v_data_v2.v3.schemas import MaskRle


def _scalar_encode_binary_mask_reference(mask: np.ndarray) -> MaskRle:
    """Exact copy of the pre-vectorization encoder, used only as an oracle."""
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


def _assert_exact_equivalence(mask: np.ndarray) -> None:
    expected = _scalar_encode_binary_mask_reference(mask)
    actual = encode_binary_mask(mask)

    assert actual.model_dump(mode="json") == expected.model_dump(mode="json")
    decoded = decode_binary_mask(actual)
    assert decoded.dtype == np.bool_
    assert decoded.shape == mask.shape
    assert np.array_equal(decoded, mask.astype(bool))


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)],
)
def test_binary_mask_codec_exhaustive_small_masks(
    shape: tuple[int, int],
) -> None:
    pixel_count = shape[0] * shape[1]
    for pixels in product((0, 1), repeat=pixel_count):
        _assert_exact_equivalence(
            np.asarray(pixels, dtype=np.uint8).reshape(shape)
        )


def test_binary_mask_codec_preserves_zero_run_and_edge_patterns() -> None:
    patterns = [
        np.zeros((3, 5), dtype=bool),
        np.ones((3, 5), dtype=bool),
        np.asarray([[1, 1, 0]], dtype=np.uint8),
        np.asarray([[0, 1, 0, 1, 0, 1]], dtype=np.uint8),
        np.asarray([[1, 0, 1, 0, 1, 0]], dtype=np.uint8),
        np.asarray([[0, 0, 1], [1, 0, 0]], dtype=np.uint8),
        np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.uint8),
        np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.uint8),
    ]
    for mask in patterns:
        _assert_exact_equivalence(mask)

    assert encode_binary_mask(patterns[2]).counts == [0, 2, 1]


def test_binary_mask_codec_realistic_stress_patterns() -> None:
    rows, columns = np.indices((101, 137))
    rectangle = np.zeros((101, 137), dtype=bool)
    rectangle[17:83, 29:119] = True
    tiny_components = (rows % 11 == 0) & (columns % 13 == 0)
    rng = np.random.default_rng(20260829)
    patterns = [
        (rows + columns) % 2 == 0,
        rows % 6 < 3,
        columns % 6 < 3,
        rectangle,
        tiny_components,
        rng.random((101, 137)) < 0.01,
        rng.random((101, 137)) < 0.99,
    ]

    for mask in patterns:
        _assert_exact_equivalence(mask)


def test_binary_mask_codec_randomized_exact_equivalence() -> None:
    rng = np.random.default_rng(20260829)
    foreground_ratios = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    tested = 0

    for shape in (
        (1, 127),
        (127, 1),
        (10, 10),
        (37, 53),
        (480, 854),
        (720, 1280),
    ):
        for ratio in foreground_ratios:
            _assert_exact_equivalence(rng.random(shape) < ratio)
            tested += 1

    for _ in range(200):
        shape = (
            int(rng.integers(1, 34)),
            int(rng.integers(1, 34)),
        )
        ratio = float(rng.choice(foreground_ratios))
        _assert_exact_equivalence(rng.random(shape) < ratio)
        tested += 1

    assert tested >= 200


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int64])
def test_binary_mask_codec_accepts_existing_binary_dtypes(
    dtype: type[np.generic],
) -> None:
    mask = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=dtype)

    _assert_exact_equivalence(mask)


@pytest.mark.parametrize("invalid_value", [2, -1, 0.5])
def test_binary_mask_codec_rejects_non_binary_values(
    invalid_value: float,
) -> None:
    with pytest.raises(ValueError, match="mask must contain only binary values"):
        encode_binary_mask(np.asarray([[0, invalid_value]]))


def test_binary_mask_codec_preserves_shape_validation() -> None:
    with pytest.raises(ValueError, match="mask must be a two-dimensional array"):
        encode_binary_mask(np.zeros((2, 2, 1), dtype=bool))
    with pytest.raises(ValueError, match="mask dimensions must be positive"):
        encode_binary_mask(np.zeros((0, 2), dtype=bool))
    with pytest.raises(ValueError, match="mask dimensions must be positive"):
        encode_binary_mask(np.zeros((2, 0), dtype=bool))


def test_decode_binary_mask_rejects_malformed_count_total() -> None:
    with pytest.raises(ValidationError, match="do not match"):
        decode_binary_mask({"size": [2, 3], "counts": [2, 3]})
