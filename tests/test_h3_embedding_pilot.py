from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.h3.audio_backends import (
    EmbeddingResult,
    FaceEmbeddingResult,
    PersistentSubprocessEmbeddingBackend,
    PrecomputedEmbeddingBackend,
    fingerprint_local_model_path,
)
from r2v_data_v2.h3.audio_schemas import FileAsset, VoiceReferenceArtifact
from r2v_data_v2.h3.embedding_pilot import (
    EmbeddingPilotInput,
    EmbeddingPilotOccurrence,
    load_embedding_pilot_inputs,
    run_embedding_pilot,
)
from r2v_data_v2.h3.primary_voice import PrimaryVoiceReferenceSelection
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioEntityBinding,
    AudioTrackMetadata,
    BindingEvidence,
    H3AudioBindingIR,
    H3TaskSpecification,
    PictureAsset,
    SemanticSubject,
)
from tools.run_h3_face_embedding_worker import (
    InsightFaceWorker,
    validate_face_model_pack,
)
from tools.run_h3_speaker_embedding_worker import validate_speaker_model_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(
    root: Path,
    occurrence_id: str,
    *,
    image_color: str = "red",
) -> EmbeddingPilotInput:
    clip_uid, entity_id = occurrence_id.split("/")
    image = root / "sources" / clip_uid / f"{entity_id}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), image_color).save(image)
    voice = root / "voices" / clip_uid / f"{entity_id}.flac"
    voice.parent.mkdir(parents=True, exist_ok=True)
    voice.write_bytes(f"voice:{occurrence_id}".encode())
    return EmbeddingPilotInput(
        entity_occurrence_id=occurrence_id,
        clip_uid=clip_uid,
        entity_id=entity_id,
        visual_reference_path=image,
        visual_reference_sha256=_sha256(image),
        primary_voice_reference_path=voice,
        primary_voice_reference_sha256=_sha256(voice),
        primary_voice_duration_seconds=1.25,
        local_binding_valid=True,
    )


class _RecordingFaceBackend:
    model_identifier = "test/face"
    checkpoint_sha256 = "a" * 64

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        failures: dict[str, str] | None = None,
    ) -> None:
        self.vectors = vectors
        self.failures = failures or {}
        self.requests: list[Path] = []

    def embed_face(
        self,
        *,
        entity_occurrence_id: str,
        image_path: Path,
    ) -> FaceEmbeddingResult:
        self.requests.append(image_path)
        failure = self.failures.get(entity_occurrence_id)
        if failure == "raise":
            raise RuntimeError("isolated face failure")
        if failure is not None:
            return FaceEmbeddingResult(status="unavailable", reason=failure)
        return FaceEmbeddingResult(
            status="available",
            embedding=EmbeddingResult(
                vector=self.vectors[entity_occurrence_id],
                model_identifier=self.model_identifier,
                checkpoint_sha256=self.checkpoint_sha256,
                backend_metadata={"backend_version": "test-1"},
            ),
            face_crop=Image.new("RGB", (8, 8), "white"),
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _run(
    root: Path,
    inputs: list[EmbeddingPilotInput],
    face_vectors: dict[str, np.ndarray],
    voice_vectors: dict[str, np.ndarray],
    *,
    failures: dict[str, str] | None = None,
    name: str = "embedding_pilot20",
    top_k: int = 2,
) -> tuple[Path, _RecordingFaceBackend]:
    face = _RecordingFaceBackend(face_vectors, failures)
    speaker = PrecomputedEmbeddingBackend(
        voice_vectors,
        model_identifier="test/voice",
        checkpoint_sha256="b" * 64,
    )
    output = root / name
    run_embedding_pilot(
        inputs=inputs,
        output_root=output,
        face_backend=face,
        speaker_backend=speaker,
        top_k=top_k,
    )
    return output, face


def test_pilot_uses_one_canonical_reference_and_normalizes_embeddings(
    tmp_path: Path,
) -> None:
    item = _input(tmp_path, "clip-a/e1")
    unrelated_frame = tmp_path / "frames" / "01.jpg"
    unrelated_frame.parent.mkdir()
    unrelated_frame.write_bytes(b"must-not-be-read")
    output, face = _run(
        tmp_path,
        [item],
        {item.entity_occurrence_id: np.array([3.0, 4.0])},
        {item.entity_occurrence_id: np.array([0.0, 5.0])},
    )

    assert face.requests == [item.visual_reference_path]
    face_vector = np.load(output / "embeddings/face/clip-a/e1.npy")
    voice_vector = np.load(output / "embeddings/voice/clip-a/e1.npy")
    assert face_vector.dtype == np.float32
    assert voice_vector.dtype == np.float32
    assert np.array_equal(face_vector, np.array([0.6, 0.8], dtype=np.float32))
    assert np.array_equal(voice_vector, np.array([0.0, 1.0], dtype=np.float32))
    assert unrelated_frame.read_bytes() == b"must-not-be-read"


@pytest.mark.parametrize(
    ("backend_reason", "published_reason"),
    [
        ("face_not_found", "face_not_found_in_canonical_reference"),
        ("multiple_faces", "multiple_faces_in_canonical_reference"),
    ],
)
def test_face_detection_failures_are_unavailable_and_fail_closed(
    tmp_path: Path,
    backend_reason: str,
    published_reason: str,
) -> None:
    item = _input(tmp_path, "clip-a/e1")
    output, _ = _run(
        tmp_path,
        [item],
        {item.entity_occurrence_id: np.array([1.0, 0.0])},
        {item.entity_occurrence_id: np.array([0.0, 1.0])},
        failures={item.entity_occurrence_id: backend_reason},
    )

    occurrence = _read_jsonl(output / "occurrences.jsonl")[0]
    assert occurrence["face"]["status"] == "unavailable"  # type: ignore[index]
    assert occurrence["face"]["failure_reason"] == published_reason  # type: ignore[index]
    assert not (output / "embeddings/face/clip-a/e1.npy").exists()
    assert (output / "embeddings/voice/clip-a/e1.npy").exists()


def test_model_failure_isolates_one_occurrence(tmp_path: Path) -> None:
    inputs = [_input(tmp_path, "clip-a/e1"), _input(tmp_path, "clip-b/e1")]
    vectors = {
        item.entity_occurrence_id: np.array([1.0, index], dtype=np.float32)
        for index, item in enumerate(inputs)
    }
    output, _ = _run(
        tmp_path,
        inputs,
        vectors,
        vectors,
        failures={"clip-a/e1": "raise"},
    )
    occurrences = {
        row["entity_occurrence_id"]: row
        for row in _read_jsonl(output / "occurrences.jsonl")
    }

    assert occurrences["clip-a/e1"]["face"]["status"] == "failed"  # type: ignore[index]
    assert occurrences["clip-b/e1"]["face"]["status"] == "available"  # type: ignore[index]
    assert occurrences["clip-a/e1"]["speaker"]["status"] == "available"  # type: ignore[index]


def test_similarity_and_joint_retrieval_are_exact_and_deterministic(
    tmp_path: Path,
) -> None:
    ids = ["clip-a/e1", "clip-a/e2", "clip-b/e1", "clip-c/e1"]
    inputs = [_input(tmp_path, occurrence_id) for occurrence_id in ids]
    face_vectors = {
        "clip-a/e1": np.array([1.0, 0.0]),
        "clip-a/e2": np.array([0.99, 0.01]),
        "clip-b/e1": np.array([0.9, 0.1]),
        "clip-c/e1": np.array([0.0, 1.0]),
    }
    voice_vectors = {
        "clip-a/e1": np.array([1.0, 0.0]),
        "clip-a/e2": np.array([0.98, 0.02]),
        "clip-b/e1": np.array([0.8, 0.2]),
        "clip-c/e1": np.array([0.0, 1.0]),
    }
    output, _ = _run(tmp_path, inputs, face_vectors, voice_vectors, top_k=1)
    faces = _read_jsonl(output / "face_similarity.jsonl")
    joint = _read_jsonl(output / "joint_candidates.jsonl")

    assert len(faces) == 6
    assert all(row["left_occurrence_id"] < row["right_occurrence_id"] for row in faces)
    same_clip = next(
        row
        for row in faces
        if row["left_occurrence_id"] == "clip-a/e1"
        and row["right_occurrence_id"] == "clip-a/e2"
    )
    assert same_clip["same_clip"] is True
    assert all(
        str(row["anchor_occurrence_id"]).split("/", maxsplit=1)[0]
        != str(row["candidate_occurrence_id"]).split("/", maxsplit=1)[0]
        for row in joint
    )
    anchor = next(
        row
        for row in joint
        if row["anchor_occurrence_id"] == "clip-a/e1"
    )
    assert anchor["candidate_occurrence_id"] == "clip-b/e1"
    assert anchor["face_anchor_to_candidate_rank"] == 1
    assert anchor["voice_anchor_to_candidate_rank"] == 1
    expected = float(
        np.dot(
            face_vectors["clip-a/e1"] / np.linalg.norm(face_vectors["clip-a/e1"]),
            face_vectors["clip-b/e1"] / np.linalg.norm(face_vectors["clip-b/e1"]),
        )
    )
    assert float(anchor["face_similarity"]) == pytest.approx(expected)
    assert anchor["same_clip"] is False
    assert "accepted" not in anchor
    assert "same_person" not in anchor

    repeat, _ = _run(
        tmp_path,
        list(reversed(inputs)),
        face_vectors,
        voice_vectors,
        name="embedding-repeat",
        top_k=1,
    )
    for relative in (
        "occurrences.jsonl",
        "face_similarity.jsonl",
        "voice_similarity.jsonl",
        "joint_candidates.jsonl",
        "summary.json",
        "embeddings/face/clip-a/e1.npy",
        "embeddings/voice/clip-b/e1.npy",
    ):
        assert (output / relative).read_bytes() == (repeat / relative).read_bytes()


def test_loader_skips_occurrences_without_primary_voice_reference(
    tmp_path: Path,
) -> None:
    visual_root = tmp_path / "visual-run"
    image = visual_root / "clips/clip-a/selected/e1.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "blue").save(image)
    audio_pilot = tmp_path / "audio-pilot"
    clip_dir = audio_pilot / "clips/clip-a"
    clip_dir.mkdir(parents=True)
    binding = AudioEntityBinding(
        start_time=0.0,
        end_time=1.0,
        entity_id="e1",
        face_track_id="face_1",
        status="bound",
        confidence=0.95,
        evidence=BindingEvidence(
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=True,
            association_confidence=0.95,
        ),
    )
    sidecar = AudioBindingSidecar(
        clip_uid="clip-a",
        source_run_root=str(visual_root),
        source_video_path="/source/clip-a.mp4",
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid="clip-a",
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path="/source/clip-a.mp4",
                full_audio_path="/audio/clip-a.wav",
                duration_seconds=1.0,
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
        bindings=[binding],
        h3_ir=H3AudioBindingIR(
            clip_uid="clip-a",
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[
                PictureAsset(
                    picture_id="picture_1",
                    entity_id="e1",
                    path="clips/clip-a/selected/e1.png",
                )
            ],
            subjects=[
                SemanticSubject(
                    subject_id="subject_1",
                    entity_id="e1",
                    reference_type="subject",
                    phrase="a person",
                    source_assets=["picture_1"],
                )
            ],
            audio_assets=[],
            bindings=[binding],
        ),
    )
    (clip_dir / "audio_binding.json").write_text(sidecar.model_dump_json())
    voice_root = tmp_path / "primary-voice"
    voice = voice_root / "voice_refs/clip-a/e1/voice_ref_1.flac"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"selected-voice")
    artifact = VoiceReferenceArtifact(
        voice_reference_id="voice_ref_1",
        entity_occurrence_id="clip-a/e1",
        source_turn_id="turn_1",
        source_start=0.0,
        source_end=1.0,
        source_start_sample=0,
        source_end_sample=16000,
        asset=FileAsset(
            path="voice_refs/clip-a/e1/voice_ref_1.flac",
            sha256=_sha256(voice),
            byte_size=voice.stat().st_size,
            media_type="audio/flac",
        ),
        quality_score=1.0,
    )
    selections = [
        PrimaryVoiceReferenceSelection(
            clip_uid="clip-a",
            entity_id="e1",
            entity_occurrence_id="clip-a/e1",
            primary_voice_reference=artifact,
            accepted_turn_ids=["turn_1"],
            policy_version="v1",
            policy_fingerprint="c" * 64,
        ),
        PrimaryVoiceReferenceSelection(
            clip_uid="missing-clip",
            entity_id="e2",
            entity_occurrence_id="missing-clip/e2",
            reason_codes=["no_voice_reference_passed_quality_gate"],
            policy_version="v1",
            policy_fingerprint="c" * 64,
        ),
    ]
    (voice_root / "primary_voice_references.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in selections)
    )

    inputs = load_embedding_pilot_inputs(
        audio_pilot_root=audio_pilot,
        primary_voice_root=voice_root,
    )

    assert [item.entity_occurrence_id for item in inputs] == ["clip-a/e1"]
    assert inputs[0].visual_reference_path == image
    assert inputs[0].primary_voice_reference_path == voice


def test_persistent_backend_loads_one_worker_for_multiple_requests(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """
import json
import pathlib
import sys

counter = pathlib.Path(sys.argv[1])
counter.write_text(counter.read_text() + "start\\n" if counter.exists() else "start\\n")
for line in sys.stdin:
    request = json.loads(line)
    request_id = request["request_id"]
    if request["operation"] == "shutdown":
        print(json.dumps({"request_id": request_id, "status": "shutdown"}), flush=True)
        break
    print(json.dumps({
        "request_id": request_id,
        "status": "available",
        "model_identifier": "fake/speaker",
        "model_fingerprint": "f" * 64,
        "embedding": [3.0, 4.0],
        "dimension": 2,
        "dtype": "float32",
        "backend_metadata": {"backend_version": "fake"},
    }), flush=True)
""".lstrip()
    )
    counter = tmp_path / "starts.txt"
    backend = PersistentSubprocessEmbeddingBackend(
        executable=[sys.executable, str(worker), str(counter)],
        model_identifier="fake/speaker",
        checkpoint_sha256="f" * 64,
        timeout_seconds=10,
        diagnostics_root=tmp_path / "logs",
    )
    try:
        first = backend.embed_speaker(
            entity_occurrence_id="clip-a/e1",
            audio_path=tmp_path / "one.flac",
        )
        second = backend.embed_speaker(
            entity_occurrence_id="clip-b/e1",
            audio_path=tmp_path / "two.flac",
        )
    finally:
        backend.close()

    assert counter.read_text().splitlines() == ["start"]
    assert np.array_equal(first.vector, second.vector)
    assert first.backend_metadata == {
        "backend": "persistent_subprocess",
        "backend_version": "fake",
    }


def test_model_contracts_fail_before_optional_model_imports(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_face_model_pack(tmp_path / "missing-face-root", "antelopev2")
    with pytest.raises(FileNotFoundError):
        validate_speaker_model_path(tmp_path / "missing-speaker")
    model = tmp_path / "model"
    model.mkdir()
    (model / "hyperparams.yaml").write_text("modules: {}\n")
    (model / "weights.ckpt").write_bytes(b"weights")
    assert len(fingerprint_local_model_path(model)) == 64
    assert validate_speaker_model_path(model) == model


@pytest.mark.parametrize(
    ("face_count", "reason"),
    [
        (0, "face_not_found_in_canonical_reference"),
        (2, "multiple_faces_in_canonical_reference"),
    ],
)
def test_insightface_worker_fails_closed_without_choosing_a_largest_face(
    tmp_path: Path,
    face_count: int,
    reason: str,
) -> None:
    class FakeCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imread(path: str, mode: int) -> np.ndarray:
            del path, mode
            return np.zeros((8, 8, 3), dtype=np.uint8)

    worker = object.__new__(InsightFaceWorker)
    worker.args = SimpleNamespace(
        det_threshold=0.5,
        model_identifier="test/arcface",
        model_fingerprint="e" * 64,
    )
    worker.cv2 = FakeCV2()
    worker.app = SimpleNamespace(
        get=lambda image: [
            SimpleNamespace(det_score=0.99 - index * 0.1)
            for index in range(face_count)
        ]
    )
    image = tmp_path / "canonical.png"
    image.write_bytes(b"fixture")

    response = worker.process(
        {
            "image_path": str(image),
            "face_crop_output_path": str(tmp_path / "face.png"),
        }
    )

    assert response["status"] == "unavailable"
    assert response["reason"] == reason
    assert "embedding" not in response


def test_schema_has_no_cross_pair_acceptance_and_thresholds_are_uncalibrated(
    tmp_path: Path,
) -> None:
    item = _input(tmp_path, "clip-a/e1")
    output, _ = _run(
        tmp_path,
        [item],
        {item.entity_occurrence_id: np.array([1.0, 0.0])},
        {item.entity_occurrence_id: np.array([0.0, 1.0])},
    )
    summary = json.loads((output / "summary.json").read_text())
    occurrence = EmbeddingPilotOccurrence.model_validate_json(
        (output / "occurrences.jsonl").read_text().strip()
    )

    assert summary["thresholds_calibrated"] is False
    assert not any("threshold" in key for key in summary if key != "thresholds_calibrated")
    assert occurrence.face.status == "available"
    assert "cross_pair_eligible" not in occurrence.model_dump()
    assert "same_person" not in occurrence.model_dump()
    assert "same_voice" not in occurrence.model_dump()


def test_worker_launch_preserves_python_symlink_text(tmp_path: Path) -> None:
    target = Path(sys.executable)
    link = tmp_path / "embedding-python"
    os.symlink(target, link)
    backend = PersistentSubprocessEmbeddingBackend(
        executable=[str(link), "unused.py"],
        model_identifier="test",
        checkpoint_sha256="d" * 64,
        timeout_seconds=1,
        diagnostics_root=tmp_path / "logs",
    )

    assert backend.executable[0] == str(link)
