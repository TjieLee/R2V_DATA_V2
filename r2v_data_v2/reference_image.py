from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def build_neutral_subject_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    background_value: int = 204,
) -> Image.Image:
    """Build an RGB subject crop from an OpenCV BGR frame."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HWC BGR image")
    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.shape != frame.shape[:2]:
        raise ValueError("mask dimensions must match frame dimensions")
    if not 0 <= background_value <= 255:
        raise ValueError("background_value must be between 0 and 255")
    x1, y1, x2, y2 = crop_box
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ValueError("crop_box must be nonempty and inside the frame")

    crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    crop_mask = binary_mask[y1:y2, x1:x2]
    neutral = np.full(crop_rgb.shape, background_value, dtype=np.uint8)
    neutral[crop_mask] = crop_rgb[crop_mask]
    return Image.fromarray(neutral)
