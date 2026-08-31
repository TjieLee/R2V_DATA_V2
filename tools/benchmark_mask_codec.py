"""Benchmark scalar and vectorized V3 binary-mask RLE codecs."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.mask_codec import (
    decode_binary_mask,
    encode_binary_mask,
)
from r2v_data_v2.v3.schemas import MaskRle


def _scalar_encode_binary_mask_reference(mask: np.ndarray) -> MaskRle:
    value = np.asarray(mask)
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


def _timed(
    function: Callable[[np.ndarray], MaskRle],
    masks: Sequence[np.ndarray],
) -> tuple[float, list[MaskRle]]:
    started = time.perf_counter()
    outputs = [function(mask) for mask in masks]
    return time.perf_counter() - started, outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark scalar and vectorized V3 binary-mask RLE codecs."
    )
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--masks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.height < 1 or args.width < 1 or args.masks < 1:
        raise ValueError("height, width, and masks must be positive")

    rng = np.random.default_rng(args.seed)
    ratios = (0.01, 0.1, 0.5, 0.9, 0.99)
    masks = [
        rng.random((args.height, args.width)) < ratios[index % len(ratios)]
        for index in range(args.masks)
    ]

    scalar_seconds, scalar_results = _timed(
        _scalar_encode_binary_mask_reference,
        masks,
    )
    vectorized_seconds, vectorized_results = _timed(
        encode_binary_mask,
        masks,
    )
    decode_started = time.perf_counter()
    decoded = [decode_binary_mask(result) for result in vectorized_results]
    decode_seconds = time.perf_counter() - decode_started

    exact_equal = all(
        scalar.model_dump(mode="json") == vectorized.model_dump(mode="json")
        and np.array_equal(mask, restored)
        for mask, scalar, vectorized, restored in zip(
            masks,
            scalar_results,
            vectorized_results,
            decoded,
            strict=True,
        )
    )
    if not exact_equal:
        raise RuntimeError("scalar/vectorized mask codec outputs differ")

    pixel_count = args.height * args.width * args.masks
    print(f"scalar_encode_seconds={scalar_seconds:.6f}")
    print(f"vectorized_encode_seconds={vectorized_seconds:.6f}")
    print(f"encode_speedup={scalar_seconds / vectorized_seconds:.3f}")
    print(f"vectorized_decode_seconds={decode_seconds:.6f}")
    print(
        "vectorized_encode_pixels_per_second="
        f"{pixel_count / vectorized_seconds:.0f}"
    )
    print(
        "vectorized_decode_pixels_per_second="
        f"{pixel_count / decode_seconds:.0f}"
    )
    print("exact_equal=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
