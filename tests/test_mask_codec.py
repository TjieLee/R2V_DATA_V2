from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


class _AnchorPredictor:
    def __init__(
        self,
        *,
        anchor_outputs: dict[int, dict[str, np.ndarray]],
        propagation_outputs: dict[int, dict[str, np.ndarray]],
    ) -> None:
        self.anchor_outputs = anchor_outputs
        self.propagation_outputs = propagation_outputs
        self.session_count = 0
        self.closed_sessions: set[str] = set()

    @staticmethod
    def empty_outputs() -> dict[str, np.ndarray]:
        return {
            "out_binary_masks": np.empty((0, 1, 32, 48), dtype=bool),
            "out_probs": np.empty((0,), dtype=float),
            "out_obj_ids": np.empty((0,), dtype=int),
        }

    @staticmethod
    def outputs(
        *,
        object_ids: list[int],
        confidences: list[float],
    ) -> dict[str, np.ndarray]:
        masks = np.zeros((len(object_ids), 1, 32, 48), dtype=bool)
        for index in range(len(object_ids)):
            masks[index, 0, 6 + index : 26 + index, 10:34] = True
        return {
            "out_binary_masks": masks,
            "out_probs": np.asarray(confidences),
            "out_obj_ids": np.asarray(object_ids),
        }

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        if request["type"] == "start_session":
            self.session_count += 1
            return {"session_id": f"session-{self.session_count}"}
        if request["type"] == "add_prompt":
            slot = int(request["frame_index"])
            return {
                "frame_index": slot,
                "outputs": self.anchor_outputs.get(slot, self.empty_outputs()),
            }
        if request["type"] == "close_session":
            self.closed_sessions.add(str(request["session_id"]))
            return {"is_success": True}
        raise AssertionError(request)

    def handle_stream_request(self, request: dict[str, object]):
        assert request["type"] == "propagate_in_video"
        for slot, outputs in sorted(self.propagation_outputs.items()):
            yield {"frame_index": slot, "outputs": outputs}


def test_sam_anchor_can_start_when_entity_first_appears_in_slot_five(
    tmp_path: Path,
) -> None:
    outputs = _AnchorPredictor.outputs(object_ids=[7], confidences=[0.96])
    predictor = _AnchorPredictor(
        anchor_outputs={5: outputs},
        propagation_outputs={
            4: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.91]),
            6: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.92]),
        },
    )
    backend = Sam3Backend(Sam3Config(), predictor=predictor)

    observations = backend.track(frames_dir=tmp_path, grounding_prompt="late subject")

    assert {item.frame_slot for item in observations} == {4, 5, 6}
    assert predictor.session_count == 9
    assert len(predictor.closed_sessions) == 9


def test_low_margin_multiple_instances_do_not_make_a_valid_anchor(
    tmp_path: Path,
) -> None:
    predictor = _AnchorPredictor(
        anchor_outputs={
            5: _AnchorPredictor.outputs(
                object_ids=[7, 8],
                confidences=[0.90, 0.85],
            )
        },
        propagation_outputs={},
    )
    backend = Sam3Backend(Sam3Config(), predictor=predictor)

    assert backend.find_best_anchor(tmp_path, "ambiguous people") is None


def test_sam_object_id_switch_is_rejected(tmp_path: Path) -> None:
    predictor = _AnchorPredictor(
        anchor_outputs={
            5: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.96])
        },
        propagation_outputs={
            6: _AnchorPredictor.outputs(object_ids=[9], confidences=[0.90])
        },
    )
    backend = Sam3Backend(Sam3Config(), predictor=predictor)

    with pytest.raises(
        ValueError,
        match="object identity switched",
    ):
        backend.track(frames_dir=tmp_path, grounding_prompt="tracked person")


def test_sam_requires_two_visible_frames(tmp_path: Path) -> None:
    predictor = _AnchorPredictor(
        anchor_outputs={
            5: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.96])
        },
        propagation_outputs={},
    )
    backend = Sam3Backend(Sam3Config(minimum_visible_frames=2), predictor=predictor)

    assert backend.track(frames_dir=tmp_path, grounding_prompt="single frame") == []
