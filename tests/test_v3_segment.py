from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.config as v3_config_module
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
from r2v_data_v2.v3.sam3_backend import (
    BackendMaskObservation,
    EntityTrackResult,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    MaskRle,
    SampledFrame,
    SampledFramesArtifact,
)
from r2v_data_v2.v3.segment import segment_clips
from r2v_data_v2.v3.storage import RunStorage
from run_pipeline_v3 import run_pipeline_v3


@dataclass
class FakeSegmentationBackend:
    results: dict[str, EntityTrackResult | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

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


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug_overlays: bool = False,
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
        sam3=Sam3Config(save_debug_overlays=debug_overlays),
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
) -> tuple[RunStorage, list[Path]]:
    storage = RunStorage(
        _config(
            tmp_path,
            monkeypatch,
            debug_overlays=debug_overlays,
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


def test_group_can_union_backend_verified_group_tracks(
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
    frame = result.frames[3]
    assert result.status == "ready"
    assert result.backend_object_ids == ["member-a", "member-b"]
    assert frame.backend_object_ids == ["member-a", "member-b"]
    assert frame.backend_confidences == [0.91, 0.86]
    assert frame.confidence == 0.86
    assert frame.area_pixels == int(np.logical_or(first, second).sum())


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
        "  limit: 10\n",
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
