from __future__ import annotations

from pathlib import Path

import numpy as np

from r2v_data_v2.config import Sam3Config
from r2v_data_v2.mask_utils import (
    decode_mask,
    encode_mask,
    fill_small_enclosed_holes,
)
from r2v_data_v2.sam3_backend import Sam3Backend


def test_mask_codec_is_lossless() -> None:
    random = np.random.default_rng(42)
    mask = random.random((73, 119)) > 0.7
    encoded = encode_mask(mask)
    decoded = decode_mask(encoded)

    assert encoded["encoding"] == "packbits_zlib_base64"
    assert np.array_equal(decoded, mask)


def test_only_small_enclosed_holes_are_filled() -> None:
    mask = np.ones((100, 100), dtype=bool)
    mask[40, 40] = False
    mask[50:60, 50:60] = False
    mask[:, 10] = False

    filled = fill_small_enclosed_holes(mask, maximum_area_ratio=0.002)

    assert filled[40, 40]
    assert not filled[55, 55]
    assert not filled[0, 10]


class _FakePredictor:
    def __init__(self) -> None:
        self.closed = False

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        if request["type"] == "start_session":
            return {"session_id": "fake"}
        if request["type"] == "add_prompt":
            return {
                "frame_index": 0,
                "outputs": self._outputs(0),
            }
        if request["type"] == "close_session":
            self.closed = True
            return {"is_success": True}
        raise AssertionError(request)

    def handle_stream_request(self, request: dict[str, object]):
        assert request["type"] == "propagate_in_video"
        for frame_slot in range(1, 8):
            yield {"frame_index": frame_slot, "outputs": self._outputs(frame_slot)}

    @staticmethod
    def _outputs(frame_slot: int) -> dict[str, np.ndarray]:
        mask = np.zeros((32, 48), dtype=bool)
        mask[4:28, 8 + frame_slot : 28 + frame_slot] = True
        return {
            "out_binary_masks": mask[None, None, ...],
            "out_probs": np.array([0.9]),
            "out_obj_ids": np.array([7]),
        }


def test_fake_sam_tracks_same_object_over_eight_frames(tmp_path: Path) -> None:
    predictor = _FakePredictor()
    backend = Sam3Backend(Sam3Config(), predictor=predictor)
    observations = backend.track(frames_dir=tmp_path, grounding_prompt="red coat")

    assert len(observations) == 8
    assert {item.frame_slot for item in observations} == set(range(8))
    assert {item.object_id for item in observations} == {7}
    assert predictor.closed
