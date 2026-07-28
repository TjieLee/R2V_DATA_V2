from __future__ import annotations

import numpy as np

from r2v_data_v2.reference_image import build_neutral_subject_crop


def test_neutral_subject_crop_preserves_rgb_foreground() -> None:
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    frame[1:3, 2:4] = (10, 20, 30)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    image = build_neutral_subject_crop(frame, mask, (1, 0, 5, 4))
    pixels = np.asarray(image)

    assert image.size == (4, 4)
    assert pixels[1, 1].tolist() == [30, 20, 10]
    assert pixels[0, 0].tolist() == [204, 204, 204]
