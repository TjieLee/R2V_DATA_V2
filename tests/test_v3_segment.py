from __future__ import annotations

import hashlib
import json
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
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler
from r2v_data_v2.v3.rank import build_coverage_state
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
    MultiInstanceAnchorDecision,
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
    TrackedEntityMasks,
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


@dataclass
class FakeAnchorSelector:
    result: MultiInstanceAnchorDecision | Exception
    calls: list[dict[str, object]] = field(default_factory=list)

    def select(self, **kwargs: object) -> MultiInstanceAnchorDecision:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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
        "out_obj_ids": np.asarray([f"object-{index}" for index in range(len(masks))]),
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
            candidate_judge=QwenServiceConfig(model=str(model)),
            background_remove_judge=QwenServiceConfig(model=str(model)),
        ),
        sam3=Sam3Config(
            model_path=(
                user_models / "sam3" / "checkpoint.pt" if with_sam3_model else None
            ),
            save_debug_overlays=debug_overlays,
        ),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "Qwen-Image-Edit-2511-Object-Remover",
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


def _published_track(
    entity_id: str,
    masks_by_slot: dict[int, np.ndarray],
    *,
    reference_type: str = "subject",
    confidence: float = 0.9,
) -> TrackedEntityMasks:
    entity = _entity(entity_id, reference_type=reference_type)
    first_mask = next(iter(masks_by_slot.values()))
    return v3_segment_module._entity_masks_from_result(
        entity,
        _ready(
            [
                BackendMaskObservation(
                    slot=slot,
                    mask=mask,
                    confidence=confidence,
                    object_id=f"track-{entity_id}",
                )
                for slot, mask in masks_by_slot.items()
            ]
        ),
        height=first_mask.shape[0],
        width=first_mask.shape[1],
    )


def _overlap_mask_pair(intersection_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((1, 200), dtype=bool)
    second = np.zeros((1, 200), dtype=bool)
    first[:, :100] = True
    second[:, 100 - intersection_pixels : 200 - intersection_pixels] = True
    return first, second


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


def test_segment_profiles_one_track_call_per_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2", reference_type="object")]
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=entities,
    )
    mask = _mask()
    backend = FakeSegmentationBackend(
        {
            entity.entity_id: _ready(
                [
                    BackendMaskObservation(
                        slot=4,
                        mask=mask,
                        confidence=0.83,
                        object_id=f"track-{entity.entity_id}",
                    )
                ]
            )
            for entity in entities
        }
    )
    profiler = V3Profiler(tmp_path / "profile", git_commit="abc123")

    with active_profiler(profiler):
        stats = segment_clips(storage.config, storage, backend=backend)

    assert stats.processed == 1
    events = [
        json.loads(line)
        for line in profiler.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 2
    assert {event["component"] for event in events} == {"sam3_segment_track"}
    assert {event["metadata"]["entity_id"] for event in events} == {"e1", "e2"}
    assert {event["metadata"]["reference_type"] for event in events} == {
        "subject",
        "object",
    }


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


@pytest.mark.parametrize("fallback_slot", [4, 6, 1])
def test_sam3_progressive_anchor_finds_remaining_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fallback_slot: int,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor({fallback_slot: _sam_outputs(_mask())})
    backend = Sam3SegmentationBackend(
        Sam3Config(anchor_search_mode="progressive_v1"),
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert predictor.prompt_slots[:5] == [5, 2, 7, 0, 9]
    assert fallback_slot in predictor.prompt_slots
    counters = backend.anchor_search_counters()
    assert counters["anchor_fallback_attempted"] == 1
    assert counters["anchor_fallback_hits"] == 1
    assert (
        counters["anchor_probe_calls"]
        == predictor.prompt_slots.index(fallback_slot) + 1
    )


def test_sam3_progressive_anchor_uses_unique_after_ambiguous_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor(
        {
            5: _sam_outputs(_mask(), _mask(x1=9, x2=12)),
            4: _sam_outputs(_mask()),
        }
    )
    backend = Sam3SegmentationBackend(
        Sam3Config(anchor_search_mode="progressive_v1"),
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert 4 in predictor.prompt_slots


def test_sam3_progressive_anchor_all_absent_reports_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor({})
    backend = Sam3SegmentationBackend(
        Sam3Config(anchor_search_mode="progressive_v1"),
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "not_found"
    assert predictor.prompt_slots == [5, 2, 7, 0, 9, 4, 6, 3, 8, 1]
    assert backend.anchor_search_counters() == {
        "anchor_fast_path_hits": 0,
        "anchor_fallback_attempted": 1,
        "anchor_fallback_hits": 0,
        "anchor_all_frames_not_found": 1,
        "anchor_probe_calls": 10,
    }


def test_sam3_progressive_anchor_all_ambiguous_preserves_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    ambiguous = _sam_outputs(_mask(), _mask(x1=9, x2=12))
    predictor = FakeSam3Predictor({slot: ambiguous for slot in range(10)})
    backend = Sam3SegmentationBackend(
        Sam3Config(anchor_search_mode="progressive_v1"),
        predictor=predictor,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "failed"
    assert "multiple ambiguous instances" in str(result.reason)


def test_sam3_unique_anchor_fast_path_makes_no_qwen_selection_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    selector = FakeAnchorSelector(
        MultiInstanceAnchorDecision("select", 1, "unused")
    )
    predictor = FakeSam3Predictor({5: _sam_outputs(_mask())})
    backend = Sam3SegmentationBackend(
        Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
        predictor=predictor,
        anchor_selector=selector,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        entity_phrase="a woman in a red coat",
        grounding_prompt="woman near center",
    )

    assert result.status == "ready"
    assert selector.calls == []
    assert backend.recall_rescue_counters()["multi_instance_rescue_attempted"] == 0


def test_sam3_qwen_selects_one_of_two_subject_anchor_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    first = _mask(x1=1, x2=5)
    second = _mask(x1=9, x2=13)
    selector = FakeAnchorSelector(
        MultiInstanceAnchorDecision("select", 2, "candidate 2 matches")
    )
    predictor = FakeSam3Predictor({5: _sam_outputs(first, second)})
    backend = Sam3SegmentationBackend(
        Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
        predictor=predictor,
        anchor_selector=selector,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        entity_phrase="the woman in the blue jacket",
        grounding_prompt="woman standing on the right",
    )

    assert result.status == "ready"
    assert len(selector.calls) == 1
    assert selector.calls[0]["entity_phrase"] == "the woman in the blue jacket"
    assert selector.calls[0]["grounding_prompt"] == "woman standing on the right"
    assert np.array_equal(result.observations[0].mask, second)
    assert backend.recall_rescue_counters() == {
        "multi_instance_rescue_attempted": 1,
        "multi_instance_rescue_selected": 1,
        "multi_instance_rescue_rejected": 0,
        "propagation_identity_switch_detected": 0,
        "partial_track_salvage_attempted": 0,
        "partial_track_salvage_ready": 0,
        "partial_track_salvage_insufficient": 0,
    }


@pytest.mark.parametrize(
    ("decision", "expected_reason"),
    (
        (
            MultiInstanceAnchorDecision("uncertain", None, "both cars plausible"),
            "multi_instance_anchor_uncertain",
        ),
        (
            MultiInstanceAnchorDecision("select", 3, "invalid candidate"),
            "multi_instance_anchor_invalid_candidate_id",
        ),
    ),
)
def test_sam3_multi_instance_uncertain_or_invalid_selection_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: MultiInstanceAnchorDecision,
    expected_reason: str,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="object")],
    )
    predictor = FakeSam3Predictor(
        {5: _sam_outputs(_mask(x1=1, x2=5), _mask(x1=9, x2=13))}
    )
    selector = FakeAnchorSelector(decision)
    backend = Sam3SegmentationBackend(
        Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
        predictor=predictor,
        anchor_selector=selector,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="object",
        entity_phrase="the parked blue car",
        grounding_prompt="blue car near the curb",
    )

    assert result.status == "failed"
    assert result.reason == expected_reason
    assert len(selector.calls) == 1


def test_sam3_multi_instance_judge_failure_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    predictor = FakeSam3Predictor(
        {5: _sam_outputs(_mask(x1=1, x2=5), _mask(x1=9, x2=13))}
    )
    selector = FakeAnchorSelector(RuntimeError("Qwen unavailable"))
    backend = Sam3SegmentationBackend(
        Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
        predictor=predictor,
        anchor_selector=selector,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        entity_phrase="the woman in the blue jacket",
        grounding_prompt="woman on the right",
    )

    assert result.status == "failed"
    assert result.reason == "multi_instance_anchor_judge_failed"


def test_sam3_multi_instance_group_never_calls_qwen_or_unions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="group")],
    )
    selector = FakeAnchorSelector(
        MultiInstanceAnchorDecision("select", 1, "must remain unused")
    )
    predictor = FakeSam3Predictor(
        {5: _sam_outputs(_mask(x1=1, x2=5), _mask(x1=9, x2=13))}
    )
    backend = Sam3SegmentationBackend(
        Sam3Config(multi_instance_rescue_mode="qwen_anchor_select_v1"),
        predictor=predictor,
        anchor_selector=selector,
    )

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="group",
        entity_phrase="two women",
        grounding_prompt="two women near center",
    )

    assert result.status == "failed"
    assert result.reason == "unverified_multi_object_group"
    assert selector.calls == []


def test_sam3_multi_object_group_fails_without_propagation_or_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1", reference_type="group")],
    )
    predictor = FakeSam3Predictor({5: _sam_outputs(_mask(), _mask(x1=9, x2=12))})
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
            "forward": [_stream_response(slot, mask) for slot in range(6, 10)],
            "backward": [_stream_response(slot, mask) for slot in range(4, -1, -1)],
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
        required_visible_frames=(storage.config.coverage.required_visible_frames),
    )

    assert coverage.required_visible_frames == 7
    assert coverage.entity_visibility_summary["e1"].visible_frame_count == 10


def test_mask_iou_rejects_shape_mismatch_and_empty_union() -> None:
    mask = _mask()

    assert mask_iou(mask, mask.copy()) == pytest.approx(1.0)
    assert (
        mask_iou(
            np.zeros_like(mask),
            np.zeros_like(mask),
        )
        == 0.0
    )
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
            "forward": [_stream_response(6, mask, object_id="forward-0")],
            "backward": [_stream_response(4, mask, object_id="backward-7")],
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
    assert set(predictor.propagation_session_ids).issubset(predictor.closed_session_ids)


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
        "forward": [_stream_response(slot, mask) for slot in range(6, 10)],
        "backward": [_stream_response(slot, mask) for slot in range(4, -1, -1)],
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


def test_sam3_propagation_anchor_duplicates_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[_entity("e1")],
    )
    anchor_mask = _mask()
    forward_anchor_copy = anchor_mask.copy()
    forward_anchor_copy[3, 2:4] = False
    backward_anchor_copy = anchor_mask.copy()
    backward_anchor_copy[8, 5:7] = False
    assert mask_iou(anchor_mask, forward_anchor_copy) < 0.95
    assert mask_iou(anchor_mask, backward_anchor_copy) < 0.95
    predictor = FakeSam3Predictor(
        {5: _sam_output(anchor_mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(5, forward_anchor_copy),
                *[_stream_response(slot, anchor_mask) for slot in range(6, 10)],
            ],
            "backward": [
                _stream_response(5, backward_anchor_copy),
                *[_stream_response(slot, anchor_mask) for slot in range(4, -1, -1)],
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
    assert [item.slot for item in result.observations] == list(range(10))
    slot_five = [item for item in result.observations if item.slot == 5]
    assert len(slot_five) == 1
    assert slot_five[0].confidence == pytest.approx(0.9)
    assert np.array_equal(slot_five[0].mask, anchor_mask)


def test_sam3_propagation_uses_strict_directional_slot_ownership(
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
        {5: _sam_output(mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(4, mask, confidence=0.11),
                _stream_response(5, mask, confidence=0.12),
                _stream_response(6, mask, confidence=0.86),
            ],
            "backward": [
                _stream_response(6, mask, confidence=0.13),
                _stream_response(5, mask, confidence=0.14),
                _stream_response(4, mask, confidence=0.84),
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
    by_slot = {item.slot: item for item in result.observations}
    assert sorted(by_slot) == [4, 5, 6]
    assert by_slot[4].confidence == pytest.approx(0.84)
    assert by_slot[5].confidence == pytest.approx(0.9)
    assert by_slot[6].confidence == pytest.approx(0.86)


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
            "forward": [_stream_response(4, anchor_mask)],
            "backward": [_stream_response(4, conflict_mask)],
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
    assert set(predictor.propagation_session_ids).issubset(predictor.closed_session_ids)


def test_late_forward_identity_switch_salvages_verified_track_and_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1")
    storage, _frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
    )
    mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(6, mask),
                _stream_response(7, mask),
                _stream_response(8, mask, object_id="changed-object"),
                _stream_response(9, mask),
            ],
            "backward": [_stream_response(slot, mask) for slot in range(4, -1, -1)],
        },
    )
    backend = Sam3SegmentationBackend(storage.config.sam3, predictor=predictor)

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")
    track = artifact.entities[entity.entity_id]
    coverage = build_coverage_state(
        artifact=artifact,
        entities=[entity],
        required_visible_frames=storage.config.coverage.required_visible_frames,
    )

    assert track.status == "ready"
    assert [frame.slot for frame in track.frames if frame.present] == list(range(8))
    assert track.frames[8].present is False
    assert track.frames[8].track_valid is False
    assert track.frames[9].present is False
    assert track.frames[9].track_valid is False
    assert "changed-object" not in track.backend_object_ids
    assert all(
        "changed-object" not in frame.backend_object_ids for frame in track.frames
    )
    assert coverage.passed is True
    assert coverage.entity_visibility_summary["e1"].visible_frame_count == 8
    assert stats.propagation_identity_switch_detected == 1
    assert stats.partial_track_salvage_attempted == 1
    assert stats.partial_track_salvage_ready == 1
    assert stats.partial_track_salvage_insufficient == 0


def test_partial_identity_switch_track_below_coverage_remains_non_qualifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entity = _entity("e1")
    storage, _frame_paths = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=[entity],
    )
    mask = _mask()
    predictor = FakeSam3Predictor(
        {5: _sam_output(mask)},
        propagation_by_direction={
            "forward": [
                _stream_response(6, mask, object_id="changed-object"),
                _stream_response(7, mask),
            ],
            "backward": [
                _stream_response(4, mask),
                _stream_response(3, mask),
            ],
        },
    )
    backend = Sam3SegmentationBackend(storage.config.sam3, predictor=predictor)

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")
    coverage = build_coverage_state(
        artifact=artifact,
        entities=[entity],
        required_visible_frames=storage.config.coverage.required_visible_frames,
    )

    assert artifact.entities["e1"].status == "ready"
    assert [
        frame.slot for frame in artifact.entities["e1"].frames if frame.present
    ] == [3, 4, 5]
    assert coverage.passed is False
    assert coverage.entity_visibility_summary["e1"].visible_frame_count == 3
    assert stats.partial_track_salvage_ready == 0
    assert stats.partial_track_salvage_insufficient == 1


def test_no_identity_switch_keeps_recall_counters_zero(
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
        {5: _sam_output(mask)},
        propagation_by_direction={
            "forward": [_stream_response(6, mask)],
            "backward": [_stream_response(4, mask)],
        },
    )
    backend = Sam3SegmentationBackend(storage.config.sam3, predictor=predictor)

    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="distinct subject",
    )

    assert result.status == "ready"
    assert [observation.slot for observation in result.observations] == [4, 5, 6]
    assert all(observation.valid for observation in result.observations)
    assert all(value == 0 for value in backend.recall_rescue_counters().values())


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
            return _ready([BackendMaskObservation(1, _mask(), 0.9, "track-1")])

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
        {"e1": _ready([BackendMaskObservation(1, _mask(), 0.9, "track-1")])}
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
        {"e1": _ready([BackendMaskObservation(1, _mask(), 0.9, "track-1")])}
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
        {"e1": _ready([BackendMaskObservation(1, _mask(), 0.9, "track-1")])}
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
        "qwen:\n"
        "  candidate_judge:\n"
        f"    model: {storage.config.qwen.candidate_judge.model}\n"
        "  background_remove_judge:\n"
        f"    model: {storage.config.qwen.background_remove_judge.model}\n"
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


def test_cross_entity_duplicate_requires_sustained_overlap() -> None:
    first_mask = np.zeros((1, 200), dtype=bool)
    second_mask = np.zeros((1, 200), dtype=bool)
    first_mask[:, :195] = True
    second_mask[:, 5:] = True
    duplicate = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: first_mask for slot in range(5)}),
        "e2",
        _published_track("e2", {slot: second_mask for slot in range(5)}),
        first_annotation_index=0,
        second_annotation_index=1,
    )
    single_frame = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {0: first_mask}),
        "e2",
        _published_track("e2", {0: second_mask}),
        first_annotation_index=0,
        second_annotation_index=1,
    )

    assert duplicate.is_duplicate is True
    assert duplicate.common_valid_frame_count == 5
    assert duplicate.median_mask_iou == pytest.approx(0.95)
    assert single_frame.is_duplicate is False


def test_cross_entity_duplicate_requires_median_iou_threshold() -> None:
    first_mask, lower_overlap = _overlap_mask_pair(90)
    decision = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: first_mask for slot in range(3)}),
        "e2",
        _published_track("e2", {slot: lower_overlap for slot in range(3)}),
        first_annotation_index=0,
        second_annotation_index=1,
    )

    assert decision.median_mask_iou < 0.85
    assert decision.is_duplicate is False


def test_cross_entity_duplicate_requires_high_overlap_frame_fraction() -> None:
    pairs = [_overlap_mask_pair(intersection) for intersection in (100, 98, 95, 88, 88)]
    decision = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: pair[0] for slot, pair in enumerate(pairs)}),
        "e2",
        _published_track("e2", {slot: pair[1] for slot, pair in enumerate(pairs)}),
        first_annotation_index=0,
        second_annotation_index=1,
    )

    assert decision.median_mask_iou >= 0.85
    assert decision.high_overlap_frame_ratio == pytest.approx(3 / 5)
    assert decision.is_duplicate is False


def test_cross_entity_duplicate_winner_uses_track_quality_then_order() -> None:
    mask = np.ones((8, 8), dtype=bool)
    more_frames = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: mask for slot in range(5)}, confidence=0.7),
        "e2",
        _published_track("e2", {slot: mask for slot in range(4)}, confidence=0.99),
        first_annotation_index=0,
        second_annotation_index=1,
    )
    higher_confidence = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: mask for slot in range(4)}, confidence=0.7),
        "e2",
        _published_track("e2", {slot: mask for slot in range(4)}, confidence=0.9),
        first_annotation_index=0,
        second_annotation_index=1,
    )
    earlier_annotation = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: mask for slot in range(4)}),
        "e2",
        _published_track("e2", {slot: mask for slot in range(4)}),
        first_annotation_index=0,
        second_annotation_index=1,
    )

    assert more_frames.winner_entity_id == "e1"
    assert higher_confidence.winner_entity_id == "e2"
    assert earlier_annotation.winner_entity_id == "e1"


def test_group_tracks_are_excluded_from_cross_entity_duplicate_gate() -> None:
    mask = np.ones((8, 8), dtype=bool)
    decision = v3_segment_module.compare_cross_entity_tracks(
        "e1",
        _published_track("e1", {slot: mask for slot in range(5)}),
        "e2",
        _published_track(
            "e2",
            {slot: mask for slot in range(5)},
            reference_type="group",
        ),
        first_annotation_index=0,
        second_annotation_index=1,
    )

    assert decision.is_duplicate is False


def test_segment_marks_duplicate_loser_failed_without_reordering_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [_entity("e1"), _entity("e2")]
    storage, _ = _storage_with_frames(
        tmp_path,
        monkeypatch,
        entities=entities,
    )
    mask = _mask()
    backend = FakeSegmentationBackend(
        {
            entity.entity_id: _ready(
                [
                    BackendMaskObservation(
                        slot=slot,
                        mask=mask,
                        confidence=0.9,
                        object_id=f"track-{entity.entity_id}",
                    )
                    for slot in range(5)
                ]
            )
            for entity in entities
        }
    )

    stats = segment_clips(storage.config, storage, backend=backend)
    artifact = storage.read_masks("clip-1")

    assert list(artifact.entities) == ["e1", "e2"]
    assert artifact.entities["e1"].status == "ready"
    assert artifact.entities["e2"].status == "failed"
    assert artifact.entities["e2"].reason == "duplicate_cross_entity_track:e1"
    assert stats.entities_ready == 1
    assert stats.entities_failed == 1
