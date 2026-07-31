from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.v3.config import (
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.frames import (
    DecodedFrameInfo,
    DecodedVideoFrame,
    deterministic_frame_indices,
    sample_frames,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    ClipSource,
    CoverageState,
    EntityReferenceState,
    EntityVisibilitySummary,
    ExportState,
    InstructionState,
    PairingState,
    ReferencesState,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass
class FakeFrameDecoder:
    frame_counts: dict[str, int]
    inspect_calls: list[str] = field(default_factory=list)
    decode_calls: list[tuple[str, tuple[int, ...]]] = field(
        default_factory=list
    )

    def inspect(self, video_path: Path) -> list[DecodedFrameInfo]:
        self.inspect_calls.append(video_path.name)
        count = self.frame_counts[video_path.name]
        return [
            DecodedFrameInfo(
                source_frame_index=index,
                timestamp_seconds=(index * 0.041) + (index % 3) * 0.001,
                width=18,
                height=12,
            )
            for index in range(count)
        ]

    def decode_indices(
        self,
        video_path: Path,
        source_frame_indices: list[int],
    ) -> list[DecodedVideoFrame]:
        self.decode_calls.append(
            (video_path.name, tuple(source_frame_indices))
        )
        return [
            DecodedVideoFrame(
                source_frame_index=index,
                image=Image.new(
                    "RGB",
                    (18, 12),
                    (index % 255, (index * 3) % 255, (index * 7) % 255),
                ),
            )
            for index in source_frame_indices
        ]


def _config(
    tmp_path: Path,
    monkeypatch,
    *,
    run_name: str = "pilot",
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
    annotation_model = pretrained / "Qwen" / "Qwen3-VL-32B-Instruct"
    config = V3Config(
        dataset_json=dataset_json,
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / f"{run_name}-v1",
        source=SourceConfig(limit=100),
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=str(annotation_model)),
            instruction_writer=QwenServiceConfig(model=str(annotation_model)),
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


def _storage_with_clip(
    tmp_path: Path,
    monkeypatch,
    *,
    clip_uid: str = "clip-1",
    entities: bool = True,
) -> RunStorage:
    storage = RunStorage(_config(tmp_path, monkeypatch))
    storage.initialize(git_commit="frames-test")
    video_path = (
        storage.config.dataset_json.parent / "videos" / f"{clip_uid}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-video")
    storage.create_clip(
        clip_uid=clip_uid,
        source=ClipSource(
            video_path=str(video_path),
            parent_video_id="parent",
            clip_suffix="1_0",
            source_index=0,
            caption_raw="",
            metadata={},
        ),
    )
    annotation_entities = (
        [
            AnnotationEntity(
                entity_id="e1",
                reference_type="subject",
                phrase="a person",
                grounding_prompt="person in a blue coat",
            )
        ]
        if entities
        else []
    )
    storage.write_annotation(
        clip_uid,
        AnnotationState(
            status="ready",
            t2v_caption="A person crosses a plaza.",
            entities=annotation_entities,
        ),
    )
    return storage


def test_deterministic_indices_are_unique_and_include_endpoints() -> None:
    indices = deterministic_frame_indices(37)

    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 36
    assert all(
        indices[index] < indices[index + 1]
        for index in range(len(indices) - 1)
    )


def test_frames_stage_writes_ten_chronological_hashed_jpegs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(tmp_path, monkeypatch)
    decoder = FakeFrameDecoder({"clip-1.mp4": 23})

    stats = sample_frames(storage.config, storage, decoder=decoder)
    artifact = storage.read_frames("clip-1")

    assert storage.config.sam3.model_path is None
    assert stats.processed == 1
    assert artifact.sampled_frame_count == 10
    assert len(artifact.frames) == 10
    source_indices = [
        frame.source_frame_index for frame in artifact.frames
    ]
    assert source_indices[0] == 0
    assert source_indices[-1] == 22
    assert len(set(source_indices)) == 10
    assert source_indices == sorted(source_indices)
    timestamps = [frame.timestamp_seconds for frame in artifact.frames]
    assert all(
        timestamps[index] < timestamps[index + 1]
        for index in range(len(timestamps) - 1)
    )
    for frame in artifact.frames:
        path = storage.clip_dir("clip-1") / frame.image_path
        assert path.is_file()
        assert frame.image_path == f"frames/{frame.slot:02d}.jpg"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame.sha256
    manifest = json.loads(
        storage.frames_manifest_path("clip-1").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "r2v.v3.frames.1"


def test_frames_stage_is_idempotent_without_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(tmp_path, monkeypatch)
    decoder = FakeFrameDecoder({"clip-1.mp4": 20})

    first = sample_frames(storage.config, storage, decoder=decoder)
    before = {
        path.name: path.read_bytes()
        for path in storage.frames_dir("clip-1").iterdir()
        if path.is_file()
    }
    second = sample_frames(storage.config, storage, decoder=decoder)

    assert first.processed == 1
    assert second.skipped_existing == 1
    assert decoder.inspect_calls == ["clip-1.mp4"]
    assert {
        path.name: path.read_bytes()
        for path in storage.frames_dir("clip-1").iterdir()
        if path.is_file()
    } == before


def test_short_video_failure_does_not_block_other_clips(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(
        tmp_path,
        monkeypatch,
        clip_uid="short",
    )
    second_video = storage.config.dataset_json.parent / "videos" / "valid.mp4"
    second_video.write_bytes(b"fake-video")
    storage.create_clip(
        clip_uid="valid",
        source=ClipSource(
            video_path=str(second_video),
            parent_video_id="parent",
            clip_suffix="2_0",
            source_index=1,
            caption_raw="",
            metadata={},
        ),
    )
    storage.write_annotation(
        "valid",
        AnnotationState(
            status="ready",
            t2v_caption="A table remains in view.",
        ),
    )
    decoder = FakeFrameDecoder({"short.mp4": 9, "valid.mp4": 10})

    stats = sample_frames(storage.config, storage, decoder=decoder)

    assert stats.processed == 1
    assert stats.failed == 1
    assert not storage.frames_manifest_path("short").exists()
    assert storage.frames_manifest_path("valid").is_file()
    failure = json.loads(
        storage.failures_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert failure["clip_uid"] == "short"
    assert failure["stage"] == "frames"


def test_overwrite_frames_invalidates_masks_and_all_downstream_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(tmp_path, monkeypatch)
    decoder = FakeFrameDecoder({"clip-1.mp4": 12})
    sample_frames(storage.config, storage, decoder=decoder)
    storage.write_masks(
        "clip-1",
        TrackedMasksArtifact(
            clip_uid="clip-1",
            height=12,
            width=18,
        ),
    )
    storage.write_coverage(
        "clip-1",
        CoverageState(
            passed=False,
            entity_visibility_summary={
                "e1": EntityVisibilitySummary(
                    status="not_found",
                    visible_frame_count=0,
                    coverage_ratio=0.0,
                    qualifies=False,
                    per_frame_area_ratio=[0.0] * 10,
                    per_frame_confidence=[None] * 10,
                )
            },
        ),
    )
    storage.write_references(
        "clip-1",
        ReferencesState(
            entities=[
                EntityReferenceState(
                    entity_id="e1",
                    status="rejected",
                    reference_scope="reject",
                    visible_region="central",
                    whole_entity_recognizable=False,
                    identity_features_visible=False,
                    scope_reason="not suitable",
                )
            ]
        ),
    )
    storage.write_pairing(
        "clip-1",
        PairingState(status="rejected", reason="not paired"),
    )
    storage.write_instruction(
        "clip-1",
        InstructionState(status="failed", reason="not instructed"),
    )
    storage.write_export(
        "clip-1",
        ExportState(accepted=False, reason="not exported"),
    )

    stats = sample_frames(
        storage.config,
        storage,
        overwrite=True,
        decoder=decoder,
    )
    clip = storage.read_clip("clip-1")

    assert stats.processed == 1
    assert not storage.masks_path("clip-1").exists()
    assert clip.coverage is None
    assert clip.references == ReferencesState()
    assert clip.pairing is None
    assert clip.instruction is None
    assert clip.export == ExportState()


def test_zero_entity_ready_clip_still_samples_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(
        tmp_path,
        monkeypatch,
        entities=False,
    )
    decoder = FakeFrameDecoder({"clip-1.mp4": 10})

    stats = sample_frames(storage.config, storage, decoder=decoder)

    assert stats.processed == 1
    assert storage.read_frames("clip-1").sampled_frame_count == 10


def test_changed_annotation_removes_frames_and_masks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage = _storage_with_clip(tmp_path, monkeypatch)
    sample_frames(
        storage.config,
        storage,
        decoder=FakeFrameDecoder({"clip-1.mp4": 10}),
    )
    storage.write_masks(
        "clip-1",
        TrackedMasksArtifact(
            clip_uid="clip-1",
            height=12,
            width=18,
        ),
    )
    annotation = storage.read_clip("clip-1").annotation
    assert annotation is not None

    storage.write_annotation(
        "clip-1",
        annotation.model_copy(
            update={"t2v_caption": "A person stops in the plaza."}
        ),
    )

    assert not storage.frames_dir("clip-1").exists()
    assert not storage.masks_path("clip-1").exists()
