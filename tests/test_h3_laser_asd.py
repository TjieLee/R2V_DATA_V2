from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from r2v_data_v2.h3 import laser_asd as laser_module
from r2v_data_v2.h3.association import associate_face_tracks_to_entities
from r2v_data_v2.h3.fusion import fuse_audio_entity_bindings
from r2v_data_v2.h3.laser_asd import (
    LASER_ASD_UPSTREAM_COMMIT,
    LaserASDRuntimeConfig,
    LaserASDRuntimeError,
    LaserASDSubprocessBackend,
    PrecomputedLaserASDBackend,
    normalize_laser_asd_evidence,
)
from r2v_data_v2.h3.laser_pilot import run_h3_laser_audio_binding_pilot
from r2v_data_v2.h3.lr_asd import (
    PrecomputedSpeechActivityBackend,
    normalize_lr_asd_evidence,
)
from r2v_data_v2.h3.pilot_schemas import (
    LaserASDNativeArtifact,
    LaserASDNativeSample,
    LaserASDNativeTrack,
    LRASDNativeArtifact,
    LRASDNativeSample,
    LRASDNativeTrack,
    SpeechActivityArtifact,
    SpeechActivityInterval,
)
from r2v_data_v2.h3.schemas import ASDModelProvenance
from r2v_data_v2.h3.visual_clip_contract import VisualClipRecord
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.schemas import (
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from tools import build_h3_asd_backend_comparison_review as comparison_module
from tools.build_h3_asd_backend_comparison_review import build_comparison_review
from tools.run_h3_laser_loconet_bridge import (
    LIP_LANDMARK_INDICES,
    _install_gdown_blocker,
    _load_verified_checkpoint,
    _offline_vggish_construction,
    _render_visualization,
    _stage_s3fd_model,
)


def _laser_artifact(
    *,
    clip_uid: str = "clip-1",
    source_video: Path,
    audio_path: Path,
    score: float = 0.7,
    debug_visualization_path: Path | None = None,
) -> LaserASDNativeArtifact:
    samples = [
        LaserASDNativeSample(
            frame_index=index,
            timestamp_seconds=index / 25,
            bbox_xyxy=(2.0, 2.0, 8.0, 8.0),
            detection_confidence=0.95,
            raw_backend_score=score,
            backend_native_active=score >= 0,
            landmark_available=index % 2 == 0,
        )
        for index in range(10)
    ]
    return LaserASDNativeArtifact(
        clip_uid=clip_uid,
        source_video_path=str(source_video),
        model_video_path=str(source_video.parent / f"{clip_uid}.model.avi"),
        audio_path=str(audio_path),
        debug_visualization_path=(
            str(debug_visualization_path)
            if debug_visualization_path is not None
            else None
        ),
        model_provenance=ASDModelProvenance(
            backend="laser_asd_loconet",
            model_identifier="plnguyen2908/LASER_ASD LoCoNet+LASER",
            checkpoint_path="/models/laser.model",
            checkpoint_sha256="a" * 64,
        ),
        config_path="/models/multi.yaml",
        config_sha256="b" * 64,
        landmark_model_path="/models/face_landmarker.task",
        landmark_model_sha256="c" * 64,
        s3fd_model_path="/models/sfd_face.pth",
        s3fd_model_sha256="d" * 64,
        resolved_n_channel=1,
        resolved_layer=1,
        device="cuda:0",
        cuda_visible_devices="6",
        mediapipe_version="1.0.0",
        torch_version="2.5.1+cu121",
        width=20,
        height=20,
        duration_seconds=0.4,
        landmark_sample_count=10,
        landmark_available_count=5,
        tracks=[LaserASDNativeTrack(face_track_id="face_1", samples=samples)],
    )


def _speech(clip_uid: str, audio_path: Path) -> SpeechActivityArtifact:
    return SpeechActivityArtifact(
        clip_uid=clip_uid,
        backend="silero_vad",
        model_identifier="silero_vad.jit",
        source_audio_path=str(audio_path),
        duration_seconds=0.4,
        intervals=[SpeechActivityInterval(start_time=0.0, end_time=0.4)],
    )


def _visual_artifacts(
    clip_uid: str,
) -> tuple[SampledFramesArtifact, TrackedMasksArtifact]:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:12, 1:12] = 1
    rows, columns = np.nonzero(mask)
    frames = SampledFramesArtifact(
        clip_uid=clip_uid,
        width=20,
        height=20,
        frames=[
            SampledFrame(
                slot=slot,
                source_frame_index=slot,
                timestamp_seconds=slot / 25,
                image_path=f"frames/{slot:02d}.jpg",
                sha256="0" * 64,
            )
            for slot in range(10)
        ],
    )
    masks = TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=20,
        height=20,
        entities={
            "e1": TrackedEntityMasks(
                status="ready",
                reference_type="subject",
                grounding_prompt="person",
                backend_object_ids=["object-e1"],
                frames=[
                    TrackedMaskFrame(
                        slot=slot,
                        present=True,
                        confidence=0.95,
                        backend_confidences=[0.95],
                        backend_object_ids=["object-e1"],
                        area_pixels=int(mask.sum()),
                        area_ratio=float(mask.mean()),
                        bbox_xyxy=(
                            int(columns.min()),
                            int(rows.min()),
                            int(columns.max()) + 1,
                            int(rows.max()) + 1,
                        ),
                        rle=encode_binary_mask(mask),
                    )
                    for slot in range(10)
                ],
            )
        },
    )
    return frames, masks


def _visual_clip(clip_uid: str, source_video: Path) -> VisualClipRecord:
    return VisualClipRecord.model_validate(
        {
            "schema_version": "r2v.v3.clip.2",
            "clip_uid": clip_uid,
            "source": {
                "video_path": str(source_video),
                "metadata": {
                    "source_relative_video_path": f"show/{clip_uid}.mp4",
                    "source_relative_source_video_path": "show/source.mp4",
                },
            },
            "annotation": {
                "status": "ready",
                "entities": [
                    {
                        "entity_id": "e1",
                        "reference_type": "subject",
                        "phrase": "person",
                    }
                ],
            },
            "coverage": {"passed": True},
            "references": {
                "entities": [
                    {
                        "entity_id": "e1",
                        "status": "ready",
                        "image_path": f"clips/{clip_uid}/selected/e1.png",
                    }
                ]
            },
            "pairing": {"status": "ready", "retained_entity_ids": ["e1"]},
        }
    )


def _write_pilot_clip(run_root: Path, clip_uid: str, source_video: Path) -> None:
    run_root.mkdir(exist_ok=True)
    (run_root / "run.json").write_text("{}\n", encoding="utf-8")
    clip_dir = run_root / "clips" / clip_uid
    clip_dir.mkdir(parents=True)
    clip = _visual_clip(clip_uid, source_video)
    (clip_dir / "clip.json").write_text(
        clip.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    frames, masks = _visual_artifacts(clip_uid)
    (clip_dir / "frames").mkdir()
    (clip_dir / "frames/frames.json").write_text(
        frames.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (clip_dir / "masks.rle.json").write_text(
        masks.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _runtime_tree(root: Path) -> LaserASDRuntimeConfig:
    code_root = root / "LASER_ASD"
    for relative in (
        "README.md",
        "LoCoNet/demoLoCoNet_landmark.py",
        "LoCoNet/landmark_loconet.py",
        "create_landmark.py",
    ):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    python = root / "laser-python"
    model = root / "laser.model"
    config = root / "multi.yaml"
    landmark = root / "face_landmarker.task"
    s3fd = root / "sfd_face.pth"
    for path in (python, model, config, landmark, s3fd):
        path.write_bytes(b"fixture")
    return LaserASDRuntimeConfig(
        code_root=code_root,
        python_path=python,
        model_path=model,
        config_path=config,
        landmark_model_path=landmark,
        s3fd_model_path=s3fd,
        device="cuda:0",
        cuda_visible_devices="6",
    )


def _git_result(command: list[str]) -> SimpleNamespace:
    if command[-2:] == ["rev-parse", "HEAD"]:
        return SimpleNamespace(returncode=0, stdout=LASER_ASD_UPSTREAM_COMMIT + "\n")
    return SimpleNamespace(returncode=0, stdout="")


def test_laser_runtime_config_is_local_pinned_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_tree(tmp_path)
    monkeypatch.setattr(
        laser_module.subprocess,
        "run",
        lambda command, **kwargs: _git_result(command),
    )
    config.validate()
    config.model_path.unlink()
    with pytest.raises(FileNotFoundError):
        config.validate()


@pytest.mark.parametrize(
    "field_name",
    ["model_path", "config_path", "landmark_model_path", "s3fd_model_path"],
)
def test_laser_runtime_requires_every_local_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    config = _runtime_tree(tmp_path)
    monkeypatch.setattr(
        laser_module.subprocess,
        "run",
        lambda command, **kwargs: _git_result(command),
    )
    getattr(config, field_name).unlink()
    with pytest.raises(FileNotFoundError):
        config.validate()


def test_laser_runtime_rejects_tracked_changes_but_ignores_untracked_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_tree(tmp_path)
    calls = 0

    def clean_then_dirty(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return _git_result(command)
        calls += 1
        return SimpleNamespace(returncode=0, stdout="" if calls == 1 else " M LoCoNet/x.py\n")

    monkeypatch.setattr(laser_module.subprocess, "run", clean_then_dirty)
    config.validate()
    with pytest.raises(ValueError, match="modified tracked files"):
        config.validate()


def test_laser_runtime_requires_explicit_isolated_cuda_device(tmp_path: Path) -> None:
    config = _runtime_tree(tmp_path)
    invalid = LaserASDRuntimeConfig(
        **{
            **config.__dict__,
            "device": "cuda:1",
            "cuda_visible_devices": "6",
        }
    )
    with pytest.raises(ValueError, match="outside isolated CUDA visibility"):
        invalid.validate()


def test_bridge_stages_local_s3fd_and_blocks_gdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sfd_face.pth"
    source.write_bytes(b"s3fd")
    staged = _stage_s3fd_model(
        runtime_root=tmp_path / "runtime", source=source.resolve()
    )
    assert staged.is_symlink()
    assert staged.resolve() == source.resolve()
    original = SimpleNamespace(download=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "gdown", original)
    _install_gdown_blocker()
    with pytest.raises(RuntimeError, match="forbids gdown"):
        sys.modules["gdown"].download("https://example.invalid")  # type: ignore[attr-defined]


def test_vggish_construction_cannot_use_torch_hub_download() -> None:
    calls: list[str] = []

    class FakeVGGish:
        def __init__(self, *, pretrained: bool = True) -> None:
            if pretrained:
                fake_torch.hub.load_state_dict_from_url("https://example.invalid")

    fake_torch = SimpleNamespace(
        hub=SimpleNamespace(
            load_state_dict_from_url=lambda url: calls.append(str(url))
        )
    )
    with _offline_vggish_construction(fake_torch, FakeVGGish):
        FakeVGGish()
    assert calls == []
    FakeVGGish()
    assert calls == ["https://example.invalid"]


class _TensorShape:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


class _CheckpointModel:
    def __init__(self) -> None:
        self.state = {
            "model.body.weight": _TensorShape(2, 2),
            "model.landmark_bottleneck.weight": _TensorShape(1, 2),
            "model.bottle_neck.weight": _TensorShape(1, 1),
            "model.norm.running_mean": _TensorShape(1),
            "model.norm.running_var": _TensorShape(1),
            "model.norm.num_batches_tracked": _TensorShape(),
        }

    def state_dict(self) -> dict[str, _TensorShape]:
        return self.state

    def named_parameters(self) -> list[tuple[str, _TensorShape]]:
        return [
            (key, value)
            for key, value in self.state.items()
            if not key.startswith("model.norm.")
        ]

    def load_state_dict(
        self, state: dict[str, object], *, strict: bool
    ) -> SimpleNamespace:
        assert strict is False
        missing = sorted(set(self.state).difference(state))
        unexpected = sorted(set(state).difference(self.state))
        return SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)


def test_checkpoint_rejects_missing_laser_landmark_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CheckpointModel()
    checkpoint = dict(model.state)
    checkpoint.pop("model.landmark_bottleneck.weight")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: checkpoint),
    )
    with pytest.raises(ValueError, match="landmark_bottleneck"):
        _load_verified_checkpoint(
            model, tmp_path / "laser.model", n_channel=1, layer=1
        )


def test_checkpoint_requires_all_inference_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _CheckpointModel()
    checkpoint = dict(model.state)
    checkpoint.pop("model.norm.running_var")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: checkpoint),
    )
    with pytest.raises(ValueError, match="missing inference state"):
        _load_verified_checkpoint(
            model, tmp_path / "laser.model", n_channel=1, layer=1
        )


def test_bridge_uses_exact_upstream_82_lip_landmark_indices() -> None:
    assert len(LIP_LANDMARK_INDICES) == 82
    assert LIP_LANDMARK_INDICES[:11] == (
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        291,
    )


def test_laser_visualization_publishes_h264_aac() -> None:
    source = inspect.getsource(_render_visualization)
    assert '"libx264"' in source
    assert '"yuv420p"' in source
    assert '"aac"' in source
    assert '"copy"' not in source
    assert LIP_LANDMARK_INDICES[-10:] == (
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        78,
    )


def test_laser_native_schema_preserves_score_and_determinism(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    artifact = _laser_artifact(source_video=source, audio_path=audio)
    assert artifact.schema_version == "r2v.h3.laser_asd_native.2"
    assert artifact.s3fd_model_sha256 == "d" * 64
    assert artifact.resolved_n_channel == 1
    assert artifact.resolved_layer == 1
    assert artifact.score_semantics == "laser_loconet_native_score"
    assert artifact.tracks[0].samples[0].backend_native_active is True
    payload = artifact.model_dump(mode="json")
    payload["tracks"][0]["samples"][0]["backend_native_active"] = False
    with pytest.raises(ValidationError, match="score >= 0"):
        LaserASDNativeArtifact.model_validate(payload)
    payload = artifact.model_dump(mode="json")
    payload["tracks"][0]["samples"] = list(
        reversed(payload["tracks"][0]["samples"])
    )
    with pytest.raises(ValidationError, match="deterministic order"):
        LaserASDNativeArtifact.model_validate(payload)


def test_laser_normalization_reuses_association_and_fusion(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "audio.wav"
    source.write_bytes(b"video")
    audio.write_bytes(b"audio")
    native = _laser_artifact(source_video=source, audio_path=audio)
    frames, masks = _visual_artifacts("clip-1")
    associations = associate_face_tracks_to_entities(
        frames=frames,
        masks=masks,
        tracks=native.tracks,
    )
    evidence = normalize_laser_asd_evidence(
        native, _speech("clip-1", audio), associations
    )
    assert associations[0].entity_id == "e1"
    assert evidence.active_speaker_intervals[0].face_scores[0].score_semantics == (
        "laser_loconet_native_score"
    )
    bindings = fuse_audio_entity_bindings(evidence, known_entity_ids={"e1"})
    assert bindings[0].status == "bound"
    assert "laser_asd_native_decision_unvalidated" in (
        bindings[0].evidence.reason_codes
    )


def test_laser_subprocess_bridge_command_records_explicit_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _runtime_tree(tmp_path)
    monkeypatch.setattr(LaserASDRuntimeConfig, "validate", lambda self: None)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    captured: list[str] = []
    captured_environment: dict[str, str] = {}

    def fake_run(command: list[str], **kwargs: object) -> None:
        captured.extend(command)
        captured_environment.update(kwargs["environment"])  # type: ignore[arg-type]
        output = Path(command[command.index("--output") + 1])
        audio = output.parent / "audio.wav"
        audio.write_bytes(b"audio")
        payload = _laser_artifact(
            source_video=source, audio_path=audio
        ).model_dump(mode="json")
        payload["model_provenance"]["checkpoint_path"] = command[
            command.index("--model-path") + 1
        ]
        payload["model_provenance"]["checkpoint_sha256"] = command[
            command.index("--checkpoint-sha256") + 1
        ]
        payload["config_path"] = command[command.index("--config-path") + 1]
        payload["config_sha256"] = command[
            command.index("--config-sha256") + 1
        ]
        payload["landmark_model_path"] = command[
            command.index("--landmark-model-path") + 1
        ]
        payload["landmark_model_sha256"] = command[
            command.index("--landmark-model-sha256") + 1
        ]
        payload["s3fd_model_path"] = command[
            command.index("--s3fd-model-path") + 1
        ]
        payload["s3fd_model_sha256"] = command[
            command.index("--s3fd-model-sha256") + 1
        ]
        payload["device"] = command[command.index("--device") + 1]
        payload["cuda_visible_devices"] = captured_environment[
            "CUDA_VISIBLE_DEVICES"
        ]
        output.write_text(
            LaserASDNativeArtifact.model_validate(payload).model_dump_json(),
            encoding="utf-8",
        )

    monkeypatch.setattr(laser_module, "_run_laser_command", fake_run)
    backend = LaserASDSubprocessBackend(config)
    backend.analyze(
        clip_uid="clip-1",
        source_video_path=source,
        work_dir=tmp_path / "work",
    )

    for flag in (
        "--model-path",
        "--config-path",
        "--landmark-model-path",
        "--s3fd-model-path",
        "--checkpoint-sha256",
        "--config-sha256",
        "--landmark-model-sha256",
        "--s3fd-model-sha256",
        "--upstream-commit",
        "--device",
    ):
        assert flag in captured
    assert captured[captured.index("--upstream-commit") + 1] == (
        LASER_ASD_UPSTREAM_COMMIT
    )
    assert captured_environment["CUDA_VISIBLE_DEVICES"] == "6"
    assert captured_environment["HF_HUB_OFFLINE"] == "1"
    assert captured_environment["TRANSFORMERS_OFFLINE"] == "1"


class _FailingLaserBackend(PrecomputedLaserASDBackend):
    def analyze(
        self,
        *,
        clip_uid: str,
        source_video_path: Path,
        work_dir: Path,
    ) -> LaserASDNativeArtifact:
        if clip_uid == "clip-1":
            raise LaserASDRuntimeError("isolated LASER fixture failure")
        return super().analyze(
            clip_uid=clip_uid,
            source_video_path=source_video_path,
            work_dir=work_dir,
        )


def test_laser_pilot_isolates_clips_and_does_not_publish_voice_quality(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    source_1 = tmp_path / "clip-1.mp4"
    source_2 = tmp_path / "clip-2.mp4"
    audio_1 = tmp_path / "clip-1.wav"
    audio_2 = tmp_path / "clip-2.wav"
    visualization = tmp_path / "laser.mp4"
    for path in (source_1, source_2, audio_1, audio_2, visualization):
        path.write_bytes(b"fixture")
    _write_pilot_clip(run_root, "clip-1", source_1)
    _write_pilot_clip(run_root, "clip-2", source_2)
    native_1 = _laser_artifact(
        clip_uid="clip-1", source_video=source_1, audio_path=audio_1
    )
    native_2 = _laser_artifact(
        clip_uid="clip-2",
        source_video=source_2,
        audio_path=audio_2,
        debug_visualization_path=visualization,
    )
    source_before = {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    summary = run_h3_laser_audio_binding_pilot(
        run_root=run_root,
        output_root=tmp_path / "laser-pilot",
        laser_backend=_FailingLaserBackend(
            {"clip-1": native_1, "clip-2": native_2}
        ),
        speech_backend=PrecomputedSpeechActivityBackend(
            {"clip-2": _speech("clip-2", audio_2)}
        ),
        limit=2,
    )

    assert summary.clips_succeeded == 1
    assert summary.clips_failed == 1
    assert summary.asd_runtime_failures == 1
    output = tmp_path / "laser-pilot"
    assert (output / "review/clip-2/laser_asd_native.json").is_file()
    assert (output / "review/clip-2/visualization.mp4").is_file()
    assert not (output / "voice_reference_quality.jsonl").exists()
    failure = json.loads((output / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["clip_uid"] == "clip-1"
    assert source_before == {
        path.relative_to(run_root): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }


def test_lr_asd_schema_and_native_normalization_contract_remain_unchanged() -> None:
    sample = LRASDNativeSample(
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
        detection_confidence=0.9,
        raw_class1_logit=0.1,
        backend_native_active=True,
    )
    artifact = LRASDNativeArtifact(
        clip_uid="legacy",
        source_video_path="/source.mp4",
        model_video_path="/model.avi",
        audio_path="/audio.wav",
        model_provenance=ASDModelProvenance(
            backend="lr_asd",
            model_identifier="Junhua-Liao/LR-ASD",
            checkpoint_path="/model.pt",
            checkpoint_sha256="d" * 64,
        ),
        width=4,
        height=4,
        duration_seconds=0.04,
        tracks=[LRASDNativeTrack(face_track_id="face_1", samples=[sample])],
    )
    assert artifact.schema_version == "r2v.h3.lr_asd_native.1"
    assert artifact.score_semantics == "lr_asd_native_class_1_logit"
    assert "laser" not in artifact.model_dump_json()
    evidence = normalize_lr_asd_evidence(
        artifact,
        SpeechActivityArtifact(
            clip_uid="legacy",
            backend="silero_vad",
            model_identifier="silero_vad.jit",
            source_audio_path="/audio.wav",
            duration_seconds=0.04,
            intervals=[SpeechActivityInterval(start_time=0.0, end_time=0.04)],
        ),
        [],
    )
    score = evidence.active_speaker_intervals[0].face_scores[0]
    assert score.score_semantics == "lr_asd_native_class_1_logit"
    assert score.raw_backend_score == pytest.approx(0.1)


def test_model_free_backend_comparison_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lr_root = tmp_path / "lr"
    laser_root = tmp_path / "laser"
    for root in (lr_root, laser_root):
        review = root / "review/clip-1"
        review.mkdir(parents=True)
        (review / "source.mp4").write_bytes(b"source")
        (review / "visualization.mp4").write_bytes(root.name.encode())
        (review / "audio_binding.json").write_text(
            json.dumps({"clip_uid": "clip-1", "bindings": []}),
            encoding="utf-8",
        )
    before = {
        path: path.read_bytes()
        for root in (lr_root, laser_root)
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        comparison_module,
        "_transcode_review_media",
        lambda source, destination, **kwargs: destination.write_bytes(
            source.read_bytes()
        ),
    )
    summary = build_comparison_review(
        lr_asd_root=lr_root,
        laser_root=laser_root,
        output_root=tmp_path / "comparison",
    )
    assert summary["clip_ids"] == ["clip-1"]
    assert "No automatic accuracy metric" in (
        tmp_path / "comparison/review.html"
    ).read_text(encoding="utf-8")
    assert "<h3>Source</h3>" in (
        tmp_path / "comparison/review.html"
    ).read_text(encoding="utf-8")
    assert all(path.read_bytes() == value for path, value in before.items())


def test_comparison_review_transcodes_browser_compatible_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "destination.mp4"
    source.write_bytes(b"source")
    captured: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        captured.extend(command)
        destination.write_bytes(b"h264")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(comparison_module.subprocess, "run", fake_run)
    comparison_module._transcode_review_media(
        source, destination, ffmpeg="ffmpeg"
    )
    assert "libx264" in captured
    assert "yuv420p" in captured
    assert "aac" in captured
