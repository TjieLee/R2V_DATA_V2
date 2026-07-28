from __future__ import annotations

import numpy as np
import pytest

from r2v_data_v2.video_io import resize_without_upscaling, sample_frame_indices


def test_fixed_ten_frame_indices_match_specification() -> None:
    assert sample_frame_indices(240) == [12, 36, 60, 84, 108, 132, 156, 180, 204, 228]
    assert sample_frame_indices(10) == list(range(10))


def test_short_video_keeps_ten_bounded_slots() -> None:
    indices = sample_frame_indices(3)

    assert len(indices) == 10
    assert min(indices) >= 0
    assert max(indices) == 2
    assert len(set(indices)) < len(indices)


def test_frame_indices_reject_empty_video() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        sample_frame_indices(0)


def test_resize_preserves_aspect_ratio_and_never_upscales() -> None:
    large = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = np.zeros((240, 320, 3), dtype=np.uint8)

    assert resize_without_upscaling(large, 1280).shape == (720, 1280, 3)
    assert resize_without_upscaling(small, 1280) is small
