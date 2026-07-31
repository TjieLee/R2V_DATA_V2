from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
import r2v_data_v2.v3.segment as v3_segment_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    Sam3Config,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import (
    decode_binary_mask,
    encode_binary_mask,
)
from r2v_data_v2.v3.rank import build_coverage_state
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
    Sam3SegmentationBackend,
    mask_iou,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    MaskRle,
    SampledFrame,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.segment import segment_clips
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3


@dataclass
class FakeSegmentationBackend:
    results: dict[str, EntityTrackResult | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)
    close_calls: int = 0

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
    ) -> EntityTrackResult:
        self.calls.append(
            {
                "frame_paths": frame_paths,
                "entity_id": entity_id,
                "reference_type": reference_type,
                "grounding_prompt": grounding_prompt,
            }
        )
        result = self.results[entity_id]
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class FakeSam3Predictor:
    outputs_by_slot: dict[int, dict[str, object]]
    anchor_outputs_by_session: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    propagation_by_direction: dict[
        str,
        list[dict[str, object]],
    ] = field(default_factory=dict)
    prompt_slots: list[int] = field(default_factory=list)
    prompt_session_ids: list[str] = field(default_factory=list)
    propagation_directions: list[str] = field(default_factory=list)
    propagation_session_ids: list[str] = field(default_factory=list)
    closed_session_ids: list[str] = field(default_factory=list)
    next_session_id: int = 0

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        request_type = request["type"]
        if request_type == "start_session":
            self.next_session_id += 1
            return {"session_id": f"session-{self.next_session_id}"}
        if request_type == "close_session":
            self.closed_session_ids.append(str(request["session_id"]))
            return {}
        if request_type == "add_prompt":
            slot = int(request["frame_index"])
            session_id = str(request["session_id"])
            self.prompt_slots.append(slot)
            self.prompt_session_ids.append(session_id)
            return {
                "frame_index": slot,
                "outputs": self.anchor_outputs_by_session.get(
                    session_id,
                    self.outputs_by_slot.get(slot, _sam_outputs()),
                ),
            }
        raise AssertionError(f"unexpected predictor request: {request_type}")

    def handle_stream_request(
        self,
        request: dict[str, object],
    ) -> list[dict[str, object]]:
        assert request["type"] == "propagate_in_video"
        direction = str(request["propagation_direction"])
        assert direction in {"forward", "backward"}
        self.propagation_directions.append(direction)
        self.propagation_session_ids.append(str(request["session_id"]))
        return self.propagation_by_direction.get(direction, [])


def _sam_outputs(
    *masks: np.ndarray,
) -> dict[str, object]:
    if not masks:
        binary_masks = np.zeros((0, 1, 12, 16), dtype=np.uint8)
    else:
        binary_masks = np.stack(masks, axis=0)[:, None].astype(np.uint8)
    return {
        "out_binary_masks": binary_masks,
        "out_probs": np.asarray(
            [0.95 - (index * 0.05) for index in range(len(masks))],
            dtype=np.float32,
        ),
        "out_obj_ids": np.asarray(
            [f"object-{index}" for index in range(len(masks))]
        ),
    }


def _sam_output(
    mask: np.ndarray,
    *,
    object_id: str = "object-0",
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "out_binary_masks": mask[None, None].astype(np.uint8),
        "out_probs": np.asarray([confidence], dtype=np.float32),
        "out_obj_ids": np.asarray([object_id]),
    }


def _stream_response(
    slot: int,
    mask: np.ndarray,
    *,
    object_id: str = "object-0",
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "frame_index": slot,
        "outputs": _sam_output(
            mask,
            object_id=object_id,
            confidence=confidence,
        ),
    }


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug_overlays: bool = False,
    with_sam3_model: bool = True,
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = (writable / "models").resolve()
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(v3_config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(v3_config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(
        v3_config_module,
        "ALLOWED_PRETRAINED_ROOT",
        pretrained,
    )
    monkeypatch.setattr(
        v3_config_module,
        "ALLOWED_USER_MODEL_ROOT",
        user_models,
    )
    dataset_json = dataset_root / "source.jsonl"
    dataset_json.write_text("", encoding="utf-8")
    model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / "segment",
        export_root=writable / "datasets" / "segment-v1",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(model)),
            instruction_writer=QwenServiceConfig(model=str(model)),
        ),
        sam3=Sam3Config(
            model_path=(
                user_models / "sam3" / "checkpoint.pt"
                if with_sam3_model
                else None
            ),
            save_debug_overlays=debug_overlays,
        ),
        remove=RemoveConfig(
            base_model_path=pretrained
            / "Qwen"
            / "Qwen-Image-Edit-2511",
            adapter_path=user_models
            / "Qwen-Image-Edit-2511-Object-Remover",
        ),
    )
    config.validate()
    return config


def _entity(
    entity_id: str,
    *,
    reference_type: str = "subject",
) -> AnnotationEntity:
    return AnnotationEntity(
        entity_id=entity_id,
        reference_type=reference_type,
        phrase=f"entity {entity_id}",
        grounding_prompt=f"distinct grounding for {entity_id}",
    )


def _storage_with_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entities: list[AnnotationEntity],
    debug_overlays: bool = False,
    with_sam3_model: bool = True,
) -> tuple[RunStorage, list[Path]]:
    storage = RunStorage(
        _config(
            tmp_path,
            monkeypatch,
            debug_overlays=debug_overlays,
            with_sam3_model=with_sam3_model,
        )
    )
    storage.initialize(git_commit="segment-test")
    video_path = storage.config.dataset_json.parent / "videos" / "clip-1.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake-video")
    storage.create_clip(
        clip_uid="clip-1",
        source=ClipSource(
            video_path=str(video_path),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    storage.write_annotation(
        "clip-1",
        AnnotationState(
            status="ready",
            t2v_caption="Several visible entities move through the frame.",
            entities=entities,
        ),
    )
    frames_dir = storage.frames_dir("clip-1")
    frames_dir.mkdir(parents=True)
    frame_records: list[SampledFrame] = []
    frame_paths: list[Path] = []
    for slot in range(10):
        path = frames_dir / f"{slot:02d}.jpg"
        Image.new(
            "RGB",
            (16, 12),
            (slot * 10, slot * 5, slot * 3),
        ).save(path, format="JPEG", quality=95)
        frame_paths.append(path)
        frame_records.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 3,
                timestamp_seconds=slot * 0.125,
                image_path=f"frames/{slot:02d}.jpg",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    artifact = SampledFramesArtifact(
        clip_uid="clip-1",
        width=16,
        height=12,
        frames=frame_records,
    )
    write_json_atomic(
        storage.frames_manifest_path("clip-1"),
        artifact.model_dump(mode="json"),
    )
    return storage, frame_paths


def _mask(
    *,
    x1: int = 2,
    y1: int = 3,
    x2: int = 7,
    y2: int = 9,
) -> np.ndarray:
    value = np.zeros((12, 16), dtype=bool)
    value[y1:y2, x1:x2] = True
    return value


def _ready(
    observations: list[BackendMaskObservation],
    *,
    group_tracks_verified: bool = False,
) -> EntityTrackResult:
    return EntityTrackResult(
        status="ready",
        observations=tuple(observations),
        group_tracks_verified=group_tracks_verified,
    )


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((7, 9), dtype=bool),
        np.pad(np.ones((2, 3), dtype=bool), ((2, 3), (1, 5))),
        np.eye(8, dtype=bool),
    ],
)
def test_binary_mask_rle_round_trip(mask: np.ndarray) -> None:
    encoded = encode_binary_mask(mask)

    assert np.array_equal(decode_binary_mask(encoded), mask)


def test_mask_rle_rejects_invalid_size_and_non_binary_input() -> None:
    with pytest.raises(ValidationError, match="do not match"):
        MaskRle(size=(3, 4), counts=[11])
    with pytest.raises(ValueError, match="binary"):
        encode_binary_mask(np.array([[0, 2]], dtype=np.uint8))
    with pytest.raises(ValueError, match="two-dimensional"):
        encode_binary_mask(np.zeros((2, 2, 1), dtype=bool))


def test_segment_uses_grounding_prompt_and_writes_ten_computed_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1")
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
    )
    mask = _mask()
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [
                    BackendMaskObservation(
                        slot=4,
                        mask=mask,
                        confidence=0.83,
                        object_id="track-7",
                    )
                ]
            )
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")
    tracked = artifact.entities["e1"]
    frame = tracked.frames[4]

    assert stats.processed == 1
    assert backend.calls == [
        {
            "frame_paths": frame_paths,
            "entity_id": "e1",
            "reference_type": "subject",
            "grounding_prompt": entity.grounding_prompt,
        }
    ]
    assert len(tracked.frames) == 10
    assert [item.slot for item in tracked.frames] == list(range(10))
    assert frame.present is True
    assert frame.area_pixels == int(mask.sum())
    assert frame.area_ratio == pytest.approx(float(mask.mean()))
    assert frame.bbox_xyxy == (2, 3, 7, 9)
    assert np.array_equal(decode_binary_mask(frame.rle), mask)


def test_one_entity_failure_does_not_block_other_entities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2", reference_type="object")]
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=entities,
    )
    backend = FakeSegmentationBackend(
        {
            "e1": RuntimeError("SAM3 entity failure"),
            "e2": _ready(
                [
                    BackendMaskObservation(
                        slot=2,
                        mask=_mask(),
                        confidence=0.9,
                        object_id="object-2",
                    )
                ]
            ),
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")

    assert stats.processed == 1
    assert stats.entities_failed == 1
    assert stats.entities_ready == 1
    assert artifact.entities["e1"].status == "failed"
    assert artifact.entities["e2"].status == "ready"
    assert [call["entity_id"] for call in backend.calls] == ["e1", "e2"]


def test_zero_entity_clip_writes_empty_masks_without_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[],
    )
    backend = FakeSegmentationBackend({})

    stats = segment_clips(storage.config, storage, backend=backend)

    assert stats.processed == 1
    assert backend.calls == []
    assert storage.read_masks("clip-1").entities == {}


def test_subject_masks_do_not_union_multiple_backend_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [
                    BackendMaskObservation(0, _mask(), 0.9, "one"),
                    BackendMaskObservation(
                        0,
                        _mask(x1=9, x2=12),
                        0.8,
                        "two",
                    ),
                ]
            )
        }
    )

    segment_clips(storage.config, storage, backend=backend)

    result = storage.read_masks("clip-1").entities["e1"]
    assert result.status == "failed"
    assert "cannot be unioned" in result.reason
    assert not any(frame.present for frame in result.frames)


def test_multi_object_group_is_unverified_and_never_unioned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="group")],
    )
    first = _mask(x1=1, y1=2, x2=4, y2=5)
    second = _mask(x1=9, y1=6, x2=13, y2=10)
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [
                    BackendMaskObservation(3, first, 0.91, "member-a"),
                    BackendMaskObservation(3, second, 0.86, "member-b"),
                ],
                group_tracks_verified=True,
            )
        }
    )

    segment_clips(storage.config, storage, backend=backend)

    result = storage.read_masks("clip-1").entities["e1"]
    assert result.status == "failed"
    assert result.reason == "unverified_multi_object_group"
    assert result.backend_object_ids == []
    assert not any(frame.present for frame in result.frames)


def test_sam3_anchor_probe_order_stops_at_first_valid_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor({2: _sam_outputs(_mask())})
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert predictor.prompt_slots == [5, 2, 2, 2]
    assert not {7, 0, 9}.intersection(predictor.prompt_slots)


def test_sam3_anchor_probe_order_uses_only_configured_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor({})
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "not_found"
    assert predictor.prompt_slots == [5, 2, 7, 0, 9]


def test_sam3_multi_object_group_fails_without_propagation_or_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="group")],
    )
    predictor = FakeSam3Predictor(
        {5: _sam_outputs(_mask(), _mask(x1=9, x2=12))}
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="group",
        grounding_prompt="two visible people",
    )

    assert result.status == "failed"
    assert result.reason == "unverified_multi_object_group"
    assert result.group_tracks_verified is False
    assert predictor.prompt_slots == [5]


def test_sam3_single_track_group_is_not_marked_group_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="group")],
    )
    predictor = FakeSam3Predictor({5: _sam_outputs(_mask())})
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="group",
        grounding_prompt="a compact group",
    )

    assert result.status == "ready"
    assert result.group_tracks_verified is False


def test_sam3_collects_forward_and_backward_tracks_for_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1")
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
    )
    mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(slot, mask) for slot in range(5, 10)
            ],
            "backward": [
                _stream_response(slot, mask) for slot in range(4, -1, -1)
            ],
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id=entity.entity_id,
        reference_type=entity.reference_type,
        grounding_prompt=entity.grounding_prompt,
    )

    assert result.status == "ready"
    assert [item.slot for item in result.observations] == list(range(10))
    assert predictor.propagation_directions == ["forward", "backward"]
    assert predictor.propagation_session_ids == predictor.prompt_session_ids[-2:]
    assert len(set(predictor.propagation_session_ids)) == 2
    assert all(
        session_id in predictor.closed_session_ids
        for session_id in predictor.propagation_session_ids
    )
    assert predictor.prompt_slots[-2:] == [5, 5]
    tracked = v3_segment_module._entity_masks_from_result(
        entity,
        result,
        height=12,
        width=16,
    )
    coverage = build_coverage_state(
        artifact=TrackedMasksArtifact(
            clip_uid="clip-1",
            height=12,
            width=16,
            entities={entity.entity_id: tracked},
        ),
        entities=[entity],
        required_visible_frames=(
            storage.config.coverage.required_visible_frames
        ),
    )

    assert coverage.required_visible_frames == 7
    assert coverage.entity_visibility_summary["e1"].visible_frame_count == 10


def test_mask_iou_rejects_shape_mismatch_and_empty_union() -> None:
    mask = _mask()

    assert mask_iou(mask, mask.copy()) == pytest.approx(1.0)
    assert mask_iou(
        np.zeros_like(mask),
        np.zeros_like(mask),
    ) == 0.0
    with pytest.raises(ValueError, match="equal mask shapes"):
        mask_iou(mask, np.zeros((8, 8), dtype=bool))


def test_sam3_session_local_object_ids_are_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask, object_id="probe-0")},
        anchor_outputs_by_session={
            "session-2": _sam_output(mask, object_id="forward-0"),
            "session-3": _sam_output(mask, object_id="backward-7"),
        },
        propagation_by_direction={
            "forward": [
                _stream_response(6, mask, object_id="forward-0")
            ],
            "backward": [
                _stream_response(4, mask, object_id="backward-7")
            ],
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert {item.object_id for item in result.observations} == {"forward-0"}
    assert [item.slot for item in result.observations] == [4, 5, 6]


def test_sam3_anchor_identity_mismatch_between_sessions_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    left = _mask(x1=1, y1=2, x2=5, y2=8)
    right = _mask(x1=10, y1=2, x2=14, y2=8)
    predictor = FakeSam3Predictor(
        {5: _sam_output(left)},
        anchor_outputs_by_session={
            "session-2": _sam_output(left, object_id="forward-0"),
            "session-3": _sam_output(right, object_id="backward-0"),
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "failed"
    assert result.reason == "anchor_identity_mismatch_between_directions"
    assert set(predictor.propagation_session_ids).issubset(
        predictor.closed_session_ids
    )


@pytest.mark.parametrize(
    ("empty_direction", "expected_slots"),
    [
        ("forward", list(range(6))),
        ("backward", list(range(5, 10))),
    ],
)
def test_sam3_empty_direction_preserves_other_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_direction: str,
    expected_slots: list[int],
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    mask = _mask()
    streams = {
        "forward": [
            _stream_response(slot, mask) for slot in range(5, 10)
        ],
        "backward": [
            _stream_response(slot, mask) for slot in range(4, -1, -1)
        ],
    }
    streams[empty_direction] = []
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask)},
        propagation_by_direction=streams,
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert [item.slot for item in result.observations] == expected_slots
    assert predictor.propagation_directions == ["forward", "backward"]


def test_sam3_consistent_duplicate_keeps_anchor_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    anchor_mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(anchor_mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(5, anchor_mask, confidence=0.8)
            ],
            "backward": [
                _stream_response(5, anchor_mask, confidence=0.7)
            ],
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    slot_five = [item for item in result.observations if item.slot == 5]
    assert len(slot_five) == 1
    assert slot_five[0].confidence == pytest.approx(0.9)
    assert np.array_equal(slot_five[0].mask, anchor_mask)


def test_sam3_conflicting_bidirectional_duplicate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    anchor_mask = _mask()
    conflict_mask = _mask(x1=10, y1=7, x2=14, y2=11)
    predictor = FakeSam3Predictor(
        {5: _sam_output(anchor_mask)},
        propagation_by_direction={
            "forward": [_stream_response(5, anchor_mask)],
            "backward": [_stream_response(5, conflict_mask)],
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "failed"
    assert result.reason == "conflicting_bidirectional_mask"


@pytest.mark.parametrize(
    ("changed_direction", "expected_reason"),
    [
        (
            "forward",
            "sam3_object_identity_changed_during_forward_propagation",
        ),
        (
            "backward",
            "sam3_object_identity_changed_during_backward_propagation",
        ),
    ],
)
def test_sam3_object_id_change_in_either_direction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_direction: str,
    expected_reason: str,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask)},
        propagation_by_direction={
            changed_direction: [
                _stream_response(
                    6 if changed_direction == "forward" else 4,
                    mask,
                    object_id="changed-object",
                )
            ]
        },
    )
    backend = Sam3SegmentationBackend(
        storage.config.sam3,
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "failed"
    assert result.reason == expected_reason
    assert set(predictor.propagation_session_ids).issubset(
        predictor.closed_session_ids
    )


def test_segment_closes_only_its_owned_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )

    class FakeOwnedBackend:
        instances: ClassVar[list[FakeOwnedBackend]] = []

        def __init__(self, config: Sam3Config) -> None:
            assert config == storage.config.sam3
            self.closed = False
            self.instances.append(self)

        def track(
            self,
            *,
            frame_paths: list[Path],
            entity_id: str,
            reference_type: str,
            grounding_prompt: str,
        ) -> EntityTrackResult:
            return _ready(
                [BackendMaskObservation(1, _mask(), 0.9, "track-1")]
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        v3_segment_module,
        "Sam3SegmentationBackend",
        FakeOwnedBackend,
    )

    segment_clips(storage.config, storage)

    assert len(FakeOwnedBackend.instances) == 1
    assert FakeOwnedBackend.instances[0].closed is True


def test_segment_does_not_close_injected_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [BackendMaskObservation(1, _mask(), 0.9, "track-1")]
            )
        }
    )

    segment_clips(storage.config, storage, backend=backend)

    assert backend.close_calls == 0


def test_segment_requires_model_path_when_creating_real_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
        with_sam3_model=False,
    )

    with pytest.raises(
        ValueError,
        match="sam3.model_path must be configured before segment runs",
    ):
        segment_clips(storage.config, storage)


def test_segment_fake_backend_allows_missing_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
        with_sam3_model=False,
    )
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [BackendMaskObservation(1, _mask(), 0.9, "track-1")]
            )
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)

    assert storage.config.sam3.model_path is None
    assert stats.processed == 1
    assert storage.read_masks("clip-1").entities["e1"].status == "ready"


@pytest.mark.parametrize("enabled", [False, True])
def test_segment_debug_overlays_follow_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
        debug_overlays=enabled,
    )
    frame_bytes = {path.name: path.read_bytes() for path in frame_paths}
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [BackendMaskObservation(1, _mask(), 0.9, "track-1")]
            )
        }
    )

    segment_clips(storage.config, storage, backend=backend)

    debug_dir = storage.clip_dir("clip-1") / "debug" / "segment"
    assert debug_dir.is_dir() is enabled
    if enabled:
        assert len(list(debug_dir.glob("e1_*.jpg"))) == 10
        assert (debug_dir / "contact_sheet_e1.jpg").is_file()
    assert {path.name: path.read_bytes() for path in frame_paths} == frame_bytes


def test_pipeline_accepts_fake_segment_backend_and_runs_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1")
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
    )
    backend = FakeSegmentationBackend(
        {
            "e1": _ready(
                [
                    BackendMaskObservation(
                        slot=slot,
                        mask=_mask(),
                        confidence=0.9,
                        object_id="track-1",
                    )
                    for slot in range(7)
                ]
            )
        }
    )
    config_path = tmp_path / "v3.yaml"
    config_path.write_text(
        f"dataset_json: {storage.config.dataset_json}\n"
        f"run_root: {storage.config.run_root}\n"
        f"export_root: {storage.config.export_root}\n"
        "source:\n"
        "  limit: 10\n"
        "sam3:\n"
        f"  model_path: {storage.config.sam3.model_path}\n",
        encoding="utf-8",
    )

    result = run_pipeline_v3(
        config_path=config_path,
        stages=("segment", "rank"),
        overwrite=True,
        git_commit="segment-test",
        segmentation_backend=backend,
    )

    assert result["completed_stages"] == ["segment", "rank"]
    coverage = storage.read_clip("clip-1").coverage
    assert coverage is not None
    assert coverage.passed is True
