from __future__ import annotations

import math

import numpy as np


def _erode_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("foreground mask must be a non-empty two-dimensional array")
    if pixels < 0:
        raise ValueError("mask erosion pixels must be non-negative")
    if pixels == 0:
        return binary.copy()
    size = pixels * 2 + 1
    padded = np.pad(binary, pixels, mode="constant", constant_values=False)
    eroded = np.ones_like(binary, dtype=bool)
    for row_offset in range(size):
        for column_offset in range(size):
            eroded &= padded[
                row_offset : row_offset + binary.shape[0],
                column_offset : column_offset + binary.shape[1],
            ]
    return eroded


def cheap_foreground_technical_metrics(
    source_rgb: np.ndarray,
    foreground_mask: np.ndarray,
) -> dict[str, object]:
    source = np.asarray(source_rgb)
    mask = np.asarray(foreground_mask, dtype=bool)
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("technical metric source must be an H-W uint8 RGB array")
    if mask.shape != source.shape[:2] or not mask.any():
        raise ValueError("technical metric mask must match source and be non-empty")

    source_float = source.astype(np.float64, copy=False)
    luma = (
        0.299 * source_float[..., 0]
        + 0.587 * source_float[..., 1]
        + 0.114 * source_float[..., 2]
    )
    foreground_luma = luma[mask]
    foreground_rgb = source_float[mask]
    channel_max = np.max(foreground_rgb, axis=1)
    channel_min = np.min(foreground_rgb, axis=1)
    saturation = np.divide(
        channel_max - channel_min,
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 0,
    )

    gradient_mask = _erode_binary_mask(mask, 2)
    gradient_mask_fallback = not gradient_mask.any()
    if gradient_mask_fallback:
        gradient_mask = mask
    padded = np.pad(luma, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    sobel_x = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    sobel_y = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    gradient_squared = sobel_x * sobel_x + sobel_y * sobel_y
    gradient_magnitude = np.sqrt(gradient_squared)
    rows, columns = np.nonzero(mask)
    percentiles = np.quantile(
        foreground_luma,
        (0.05, 0.10, 0.50, 0.90, 0.95),
    )
    values: dict[str, object] = {
        "status": "succeeded",
        "backend": "cheap_cv",
        "foreground_pixel_count": int(mask.sum()),
        "luma_mean": float(np.mean(foreground_luma)),
        "luma_std": float(np.std(foreground_luma)),
        "luma_p05": float(percentiles[0]),
        "luma_p10": float(percentiles[1]),
        "luma_p50": float(percentiles[2]),
        "luma_p90": float(percentiles[3]),
        "luma_p95": float(percentiles[4]),
        "dark_fraction_16": float(np.mean(foreground_luma <= 16.0)),
        "dark_fraction_32": float(np.mean(foreground_luma <= 32.0)),
        "dark_fraction_48": float(np.mean(foreground_luma <= 48.0)),
        "black_clip_fraction": float(np.mean(foreground_luma <= 1.0)),
        "white_clip_fraction": float(np.mean(foreground_luma >= 254.0)),
        "rms_contrast": float(np.std(foreground_luma)),
        "laplacian_variance": float(np.var(laplacian[gradient_mask])),
        "tenengrad_mean": float(np.mean(gradient_squared[gradient_mask])),
        "edge_density": float(
            np.mean(gradient_magnitude[gradient_mask] >= 20.0)
        ),
        "saturation_mean": float(np.mean(saturation)),
        "saturation_std": float(np.std(saturation)),
        "foreground_bbox_width": int(columns.max() - columns.min() + 1),
        "foreground_bbox_height": int(rows.max() - rows.min() + 1),
        "gradient_mask_erosion_pixels": 2,
        "gradient_foreground_pixel_count": int(gradient_mask.sum()),
        "gradient_mask_fallback": gradient_mask_fallback,
        "edge_threshold_luma": 20.0,
    }
    if not all(
        math.isfinite(value)
        for value in values.values()
        if isinstance(value, float)
    ):
        raise ValueError("technical metrics must be finite")
    return values
