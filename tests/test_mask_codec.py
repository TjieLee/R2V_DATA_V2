from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.config import PipelineConfig, RankingConfig, Sam3Config
from r2v_data_v2.mask_utils import (
    decode_mask,
    encode_mask,
    fill_small_enclosed_holes,
    save_mask_contact_sheet,
)
from r2v_data_v2.sam3_backend import (
    Sam3Backend,
    SamObservation,
    _write_candidates,
    extract_manifest_candidates,
)


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
        for frame_slot in range(1, 10):
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


def test_fake_sam_tracks_same_object_over_ten_frames(tmp_path: Path) -> None:
    predictor = _FakePredictor()
    backend = Sam3Backend(Sam3Config(), predictor=predictor)
    observations = backend.track(frames_dir=tmp_path, grounding_prompt="red coat")

    assert len(observations) == 10
    assert {item.frame_slot for item in observations} == set(range(10))
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


def test_sam_anchor_can_start_when_entity_first_appears_in_slot_eight(
    tmp_path: Path,
) -> None:
    outputs = _AnchorPredictor.outputs(object_ids=[7], confidences=[0.96])
    predictor = _AnchorPredictor(
        anchor_outputs={8: outputs},
        propagation_outputs={
            7: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.91]),
            9: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.92]),
        },
    )
    backend = Sam3Backend(Sam3Config(), predictor=predictor)

    observations = backend.track(frames_dir=tmp_path, grounding_prompt="late subject")

    assert {item.frame_slot for item in observations} == {7, 8, 9}
    assert predictor.session_count == 11
    assert len(predictor.closed_sessions) == 11


def test_ten_frame_mask_contact_sheet_uses_all_slots(tmp_path: Path) -> None:
    frame_paths = []
    masks = {}
    candidates = []
    for slot in range(10):
        path = tmp_path / f"frame_{slot:02d}.png"
        Image.new("RGB", (48, 32), color=(slot * 20, 40, 80)).save(path)
        frame_paths.append(path)
        mask = np.zeros((32, 48), dtype=bool)
        mask[8:24, 12:36] = True
        masks[slot] = mask
        candidates.append(
            {
                "frame_slot": slot,
                "bbox_xyxy": [12, 8, 36, 24],
            }
        )
    destination = tmp_path / "contact_sheet.jpg"

    save_mask_contact_sheet(
        frame_paths=frame_paths,
        candidates=candidates,
        masks=masks,
        destination=destination,
    )

    with Image.open(destination) as sheet:
        assert sheet.size == (48 * 5, 32 * 2)


def test_all_ten_valid_candidate_masks_are_saved_as_rle(tmp_path: Path) -> None:
    frame_paths = []
    candidates = []
    masks = {}
    for slot in range(10):
        path = tmp_path / f"frame_{slot:02d}.png"
        Image.new("RGB", (48, 32), color=(slot * 20, 40, 80)).save(path)
        frame_paths.append(path)
        mask = np.zeros((32, 48), dtype=bool)
        mask[6:26, 10 + slot : 30 + slot] = True
        masks[slot] = mask
        candidates.append(
            {
                "frame_slot": slot,
                "source_frame_index": slot * 10,
                "bbox_xyxy": [10 + slot, 6, 30 + slot, 26],
                "mask_area_ratio": float(mask.mean()),
                "sam_confidence": 0.99 - slot * 0.01,
                "touches_border": False,
                "visible": True,
                "effective_short_side": 20,
                "mask_rle_key": None,
            }
        )

    _write_candidates(
        output_root=tmp_path,
        clip_uid="clip-1",
        entity_id="e1",
        candidates=candidates,
        masks=masks,
        frame_paths=frame_paths,
        save_top_k=10,
    )

    candidate_dir = tmp_path / "candidates" / "clip-1" / "e1"
    encoded = json.loads(
        (candidate_dir / "top_masks.rle.json").read_text(encoding="utf-8")
    )
    records = [
        json.loads(line)
        for line in (candidate_dir / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(encoded) == 10
    assert all(record["mask_rle_key"] in encoded for record in records)
    for slot in range(10):
        assert np.array_equal(decode_mask(encoded[f"frame_{slot:02d}"]), masks[slot])
    assert not list(candidate_dir.glob("mask*.png"))


def test_tracked_masks_preserve_slots_filtered_from_entity_candidates(
    tmp_path: Path,
) -> None:
    frame_paths: list[Path] = []
    tracked_masks: dict[int, np.ndarray] = {}
    for slot in range(2):
        path = tmp_path / f"frame_{slot:02d}.png"
        Image.new("RGB", (48, 32), color=(20, 40, 80)).save(path)
        frame_paths.append(path)
        mask = np.zeros((32, 48), dtype=bool)
        if slot == 0:
            mask[2:4, 4:44] = True
        else:
            mask[6:26, 10:38] = True
        tracked_masks[slot] = mask
    accepted_mask = tracked_masks[1]
    candidates = [
        {
            "frame_slot": 1,
            "source_frame_index": 10,
            "bbox_xyxy": [10, 6, 38, 26],
            "mask_area_ratio": float(accepted_mask.mean()),
            "sam_confidence": 0.99,
            "touches_border": False,
            "visible": True,
            "effective_short_side": 20,
            "mask_rle_key": None,
        }
    ]

    _write_candidates(
        output_root=tmp_path,
        clip_uid="clip-1",
        entity_id="e1",
        candidates=candidates,
        masks={1: accepted_mask},
        frame_paths=frame_paths,
        save_top_k=10,
        tracked_masks=tracked_masks,
        mask_coverage={
            0: {
                "tracked": True,
                "mask_available": True,
                "candidate_accepted": False,
                "filtered_reasons": ["effective_short_side"],
            },
            1: {
                "tracked": True,
                "mask_available": True,
                "candidate_accepted": True,
                "filtered_reasons": [],
            },
        },
    )

    candidate_dir = tmp_path / "candidates" / "clip-1" / "e1"
    candidate_masks = json.loads(
        (candidate_dir / "top_masks.rle.json").read_text(encoding="utf-8")
    )
    all_tracked_masks = json.loads(
        (candidate_dir / "tracked_masks.rle.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (candidate_dir / "mask_coverage.json").read_text(encoding="utf-8")
    )
    assert set(candidate_masks) == {"frame_01"}
    assert set(all_tracked_masks) == {"frame_00", "frame_01"}
    assert np.array_equal(
        decode_mask(all_tracked_masks["frame_00"]),
        tracked_masks[0],
    )
    assert coverage["slots"]["frame_00"]["filtered_reasons"] == [
        "effective_short_side"
    ]


def _write_sam_stage_fixture(output_root: Path) -> None:
    manifest = output_root / "manifests" / "annotations.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "caption": "A red bicycle is parked.",
                "prompt_with_refs": "A red bicycle <ref_subject_1> is parked.",
                "entities": [
                    {
                        "entity_id": "e1",
                        "phrase": "a red bicycle",
                        "grounding_prompt": "red bicycle",
                        "canonical_label": "red bicycle",
                        "category": "vehicle",
                        "reference_worthy": True,
                        "salience": "primary",
                        "genericity": "descriptive",
                        "name_evidence": "none",
                        "separability": "independent",
                        "selection_reason": "primary subject",
                        "ref_token": "<ref_subject_1>",
                    }
                ],
                "relations": [],
                "background": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frame_dir = output_root / "frames" / "clip-1"
    frame_dir.mkdir(parents=True)
    for slot in range(10):
        Image.new("RGB", (200, 200), (80, 100, 120)).save(
            frame_dir / f"frame_{slot:02d}.jpg"
        )
    (frame_dir / "frames.json").write_text(
        json.dumps({"sampled_indices": list(range(10))}),
        encoding="utf-8",
    )


def test_single_strict_candidate_is_published_separately_from_coverage(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
        sam3=Sam3Config(minimum_visible_frames=2),
        ranking=RankingConfig(minimum_effective_short_side=20),
    )
    _write_sam_stage_fixture(config.output_root)
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True

    class _Backend:
        def track(
            self,
            *,
            frames_dir: Path,
            grounding_prompt: str,
        ) -> list[SamObservation]:
            assert frames_dir.name == "clip-1"
            assert grounding_prompt == "red bicycle"
            return [
                SamObservation(
                    frame_slot=0,
                    mask=mask,
                    confidence=0.95,
                    object_id=7,
                )
            ]

    extraction = extract_manifest_candidates(
        config,
        backend=_Backend(),  # type: ignore[arg-type]
    )
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    status = json.loads(
        (candidate_dir / "candidate_status.json").read_text(encoding="utf-8")
    )
    tracked = json.loads(
        (candidate_dir / "tracked_masks.rle.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (candidate_dir / "mask_coverage.json").read_text(encoding="utf-8")
    )

    assert extraction.processed == 1
    assert extraction.no_valid_candidate == 0
    assert status["status"] == "ready"
    assert status["candidate_count"] == 1
    assert status["published_candidate_count"] == 1
    assert status["sampled_frame_count"] == 10
    assert status["visible_frame_count"] == 1
    assert status["required_visible_frames"] == 8
    assert status["temporal_coverage_passed"] is False
    assert (candidate_dir / "candidates.jsonl").read_text(
        encoding="utf-8"
    ).strip()
    assert set(
        json.loads(
            (candidate_dir / "top_masks.rle.json").read_text(
                encoding="utf-8"
            )
        )
    ) == {"frame_00"}
    assert set(tracked) == {"frame_00"}
    assert coverage["slots"]["frame_00"]["candidate_accepted"] is True
    assert coverage["visible_frame_count"] == 1
    assert coverage["temporal_coverage_passed"] is False


def test_failed_sam_overwrite_removes_all_stale_candidate_artifacts(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "output",
    )
    _write_sam_stage_fixture(config.output_root)
    candidate_dir = config.output_root / "candidates" / "clip-1" / "e1"
    selected_dir = candidate_dir / "selected"
    selected_dir.mkdir(parents=True)
    clip_coverage = candidate_dir.parent / "entity_coverage.json"
    clip_coverage.write_text("stale", encoding="utf-8")
    for name in (
        "candidates.jsonl",
        "top_masks.rle.json",
        "tracked_masks.rle.json",
        "mask_coverage.json",
        "candidate_status.json",
        "ranking_metadata.json",
    ):
        (candidate_dir / name).write_text("stale", encoding="utf-8")
    (selected_dir / "top_candidates.jpg").write_bytes(b"stale")

    class _FailingBackend:
        def track(
            self,
            *,
            frames_dir: Path,
            grounding_prompt: str,
        ) -> list[SamObservation]:
            del frames_dir, grounding_prompt
            raise RuntimeError("tracking failed")

    stats = extract_manifest_candidates(
        config,
        overwrite=True,
        backend=_FailingBackend(),  # type: ignore[arg-type]
    )

    assert stats.sam_failed == 1
    assert list(candidate_dir.iterdir()) == []
    assert not clip_coverage.exists()


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


def test_sam_returns_tracking_observations_before_visibility_gate(
    tmp_path: Path,
) -> None:
    predictor = _AnchorPredictor(
        anchor_outputs={
            5: _AnchorPredictor.outputs(object_ids=[7], confidences=[0.96])
        },
        propagation_outputs={},
    )
    backend = Sam3Backend(Sam3Config(minimum_visible_frames=2), predictor=predictor)

    observations = backend.track(
        frames_dir=tmp_path,
        grounding_prompt="single frame",
    )

    assert [observation.frame_slot for observation in observations] == [5]


def test_sam_tracked_object_survives_with_extra_objects(
    tmp_path: Path,
) -> None:
    predictor = _AnchorPredictor(
        anchor_outputs={
            5: _AnchorPredictor.outputs(
                object_ids=[7],
                confidences=[0.96],
            )
        },
        propagation_outputs={
            6: _AnchorPredictor.outputs(
                object_ids=[7, 9],
                confidences=[0.90, 0.85],
            )
        },
    )
    backend = Sam3Backend(
        Sam3Config(minimum_visible_frames=1),
        predictor=predictor,
    )

    observations = backend.track(
        frames_dir=tmp_path,
        grounding_prompt="target object",
    )

    assert observations
    assert {item.object_id for item in observations} == {7}
