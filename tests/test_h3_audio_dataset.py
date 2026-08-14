from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from r2v_data_v2.h3 import audio_backends as audio_backends_module
from r2v_data_v2.h3 import audio_export as audio_export_module
from r2v_data_v2.h3.audio_backends import (
    ExternalSubprocessEmbeddingBackend,
    PrecomputedAudioMediaBackend,
    PrecomputedEmbeddingBackend,
)
from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    build_audio_clip_binding_dataset,
    coalesce_audio_bindings,
    load_clip_bindings,
    select_primary_voice_turns,
)
from r2v_data_v2.h3.audio_export import (
    export_h3_audio_dataset,
    publish_audio_pair_dataset,
)
from r2v_data_v2.h3.audio_pairing import (
    AudioPairingConfig,
    NumpyBlockwiseTopKIndex,
    build_audio_pair_samples,
    build_pairwise_evidence,
)
from r2v_data_v2.h3.audio_schemas import (
    AudioClipBinding,
    AudioStreamProvenance,
    EmbeddingAsset,
    EntityOccurrence,
    FileAsset,
    FullAudioArtifact,
    LocalBindingSummary,
    ProducerProvenance,
    SourceVideoProvenance,
    SpeechTurn,
    TranscriptProvenance,
    VisualReferenceProvenance,
    VisualSourceProvenance,
    VoiceReferenceArtifact,
)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(root: Path, relative: str, data: bytes, media_type: str) -> FileAsset:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return FileAsset(
        path=relative,
        sha256=_sha256(path),
        byte_size=len(data),
        media_type=media_type,
    )


def _embedding(
    root: Path,
    relative: str,
    vector: list[float],
    model: str,
) -> EmbeddingAsset:
    array = np.asarray(vector, dtype=np.float32)
    array /= np.linalg.norm(array)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    asset = FileAsset(
        path=relative,
        sha256=_sha256(path),
        byte_size=path.stat().st_size,
        media_type="application/x-npy",
    )
    return EmbeddingAsset(
        **asset.model_dump(mode="python"),
        model_identifier=model,
        dimension=array.size,
    )


def _frame_binding(
    start: float,
    end: float,
    *,
    status: str = "bound",
    entity_id: str | None = "e1",
    face_track_id: str | None = "face_1",
    confidence: float = 0.9,
    clean: bool = True,
) -> AudioEntityBinding:
    if status != "bound":
        entity_id = None
        face_track_id = None
        clean = False
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        entity_id=entity_id,
        face_track_id=face_track_id,
        status=status,
        confidence=confidence,
        evidence=BindingEvidence(
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=clean,
        ),
    )


def _producer() -> ProducerProvenance:
    return ProducerProvenance(
        producer="test",
        version="v1",
        config_fingerprint=hashlib.sha256(b"test-config").hexdigest(),
    )


def _canonical_binding(
    root: Path,
    clip_uid: str,
    *,
    face: list[float] | None = None,
    voice: list[float] | None = None,
    text: list[float] | None = None,
    transcript: str | None = None,
    source_video_bytes: bytes | None = None,
    entity_id: str = "e1",
) -> AudioClipBinding:
    occurrence_id = f"{clip_uid}/{entity_id}"
    video_path = root / "source_videos" / f"{clip_uid}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(source_video_bytes or f"video:{clip_uid}".encode())
    visual_path = root / "visual_references" / clip_uid / f"{entity_id}.png"
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), "navy").save(visual_path)
    visual_asset = FileAsset(
        path=visual_path.relative_to(root).as_posix(),
        sha256=_sha256(visual_path),
        byte_size=visual_path.stat().st_size,
        media_type="image/png",
    )
    full_audio = _asset(
        root,
        f"full_audio/{clip_uid}.flac",
        f"full:{clip_uid}".encode(),
        "audio/flac",
    )
    frame_binding = _asset(
        root,
        f"diagnostics/{clip_uid}/frame_bindings.json",
        b"{}\n",
        "application/json",
    )
    face_asset = None
    face_crop = None
    if face is not None:
        face_asset = _embedding(
            root,
            f"embeddings/face/{clip_uid}/{entity_id}.npy",
            face,
            "fake-arcface",
        )
        face_crop = _asset(
            root,
            f"face_crops/{clip_uid}/{entity_id}.png",
            visual_path.read_bytes(),
            "image/png",
        )
    voice_asset = None
    voice_reference = None
    if voice is not None:
        voice_file = _asset(
            root,
            f"voice_refs/{clip_uid}/{entity_id}/voice_ref_1.flac",
            f"voice:{clip_uid}:{entity_id}".encode(),
            "audio/flac",
        )
        voice_reference = VoiceReferenceArtifact(
            voice_reference_id="voice_ref_1",
            entity_occurrence_id=occurrence_id,
            source_turn_id="turn_1",
            source_start=0.0,
            source_end=1.0,
            source_start_sample=0,
            source_end_sample=48000,
            asset=voice_file,
            quality_score=0.95,
        )
        voice_asset = _embedding(
            root,
            f"embeddings/voice/{clip_uid}/{entity_id}.npy",
            voice,
            "fake-speaker",
        )
    text_asset = None
    if text is not None:
        text_asset = _embedding(
            root,
            f"embeddings/text/{clip_uid}/{entity_id}.npy",
            text,
            "fake-text",
        )
    speech_turns = []
    if voice_reference is not None:
        speech_turns = [
            SpeechTurn(
                turn_id="turn_1",
                clip_uid=clip_uid,
                start_time=0.0,
                end_time=1.0,
                start_sample=0,
                end_sample=48000,
                start_frame=0,
                end_frame=24,
                entity_id=entity_id,
                face_track_id="face_1",
                status="bound",
                binding_confidence=0.95,
                source_binding_ids=["binding_1"],
                source_frame_ranges=[(0, 24)],
                voice_reference_eligible=True,
                text=transcript,
                text_status="auto" if transcript else "missing",
            )
        ]
    occurrence = EntityOccurrence(
        entity_occurrence_id=occurrence_id,
        entity_id=entity_id,
        reference_type="subject",
        phrase="person in a blue jacket",
        grounding_prompt="person in a blue jacket",
        identity_text="person in a blue jacket" if text is not None else None,
        identity_text_specific=text is not None,
        visual_reference=VisualReferenceProvenance(
            image_asset=visual_asset,
            token="<ref_subject_1>",
            reference_scope="full",
            visible_region="whole",
            source_frame_index=5,
            source_clip_uid=clip_uid,
            source_entity_id=entity_id,
            synthetic=False,
        ),
        face_identity_status="available" if face_asset else "unavailable",
        face_crop_asset=face_crop,
        face_embedding_asset=face_asset,
        identity_text_embedding_asset=text_asset,
        primary_voice_reference=voice_reference,
        voice_embedding_asset=voice_asset,
        local_binding_summary=LocalBindingSummary(
            bound_turn_ids=["turn_1"] if voice_reference else [],
            total_bound_seconds=1.0 if voice_reference else 0.0,
            best_binding_confidence=0.95 if voice_reference else 0.0,
            high_confidence_bound=voice_reference is not None,
        ),
        in_pair_eligible=voice_reference is not None,
        cross_pair_eligible=voice_reference is not None and face_asset is not None,
        reason_codes=[] if voice_reference else ["primary_voice_reference_unavailable"],
    )
    return AudioClipBinding(
        clip_binding_id=f"clip_binding/{clip_uid}",
        clip_uid=clip_uid,
        sample_id=clip_uid,
        parent_video_id=f"parent-{clip_uid}",
        clip_suffix="0",
        source_video=SourceVideoProvenance(
            path=str(video_path),
            sha256=_sha256(video_path),
        ),
        visual_source=VisualSourceProvenance(
            run_root="/read-only/run",
            export_root="/read-only/export",
            visual_sample_id=clip_uid,
            visual_sample_sha256=hashlib.sha256(clip_uid.encode()).hexdigest(),
        ),
        full_audio=FullAudioArtifact(
            asset=full_audio,
            stream=AudioStreamProvenance(
                stream_index=0,
                codec_name="aac",
                original_sample_rate_hz=48000,
                original_channels=2,
                duration_seconds=2.0,
                time_base="1/48000",
            ),
            output_sample_rate_hz=48000,
            output_channels=2,
            output_format="flac",
        ),
        raw_frame_bindings_path=frame_binding.path,
        speech_turns=speech_turns,
        entity_occurrences=[occurrence],
        transcript_provenance=TranscriptProvenance(
            backend="fixture" if transcript else None,
            status="precomputed" if transcript else "missing",
        ),
        visual_t2v_caption=f"Caption for {clip_uid}.",
        visual_r2v_instruction=f"Use the person from {clip_uid}.",
        producer_provenance=_producer(),
    )


def _write_bindings(root: Path, bindings: list[AudioClipBinding]) -> None:
    (root / "clip_bindings.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in bindings),
        encoding="utf-8",
    )
    (root / "pair_samples.jsonl").write_text("", encoding="utf-8")
    (root / "pair_report.json").write_text("{}\n", encoding="utf-8")
    (root / "dataset.json").write_text("{}\n", encoding="utf-8")


def _two_subject_binding(root: Path, clip_uid: str) -> AudioClipBinding:
    first = _canonical_binding(
        root,
        clip_uid,
        face=[1.0, 0.0],
        voice=[1.0, 0.0],
        transcript="first subject words",
    )
    second = _canonical_binding(
        root,
        clip_uid,
        face=[0.0, 1.0],
        voice=[0.0, 1.0],
        transcript="second subject words",
        entity_id="e2",
    )
    second_turn = second.speech_turns[0].model_copy(
        update={
            "turn_id": "turn_2",
            "start_time": 1.0,
            "end_time": 2.0,
            "start_sample": 48000,
            "end_sample": 96000,
            "start_frame": 25,
            "end_frame": 49,
        }
    )
    second_occurrence = second.entity_occurrences[0]
    assert second_occurrence.primary_voice_reference is not None
    second_occurrence = second_occurrence.model_copy(
        update={
            "primary_voice_reference": (
                second_occurrence.primary_voice_reference.model_copy(
                    update={
                        "source_turn_id": "turn_2",
                        "source_start": 1.0,
                        "source_end": 2.0,
                        "source_start_sample": 48000,
                        "source_end_sample": 96000,
                    }
                )
            ),
            "local_binding_summary": LocalBindingSummary(
                bound_turn_ids=["turn_2"],
                total_bound_seconds=1.0,
                best_binding_confidence=0.95,
                high_confidence_bound=True,
            ),
        }
    )
    payload = first.model_dump(mode="python")
    payload["speech_turns"] = [first.speech_turns[0], second_turn]
    payload["entity_occurrences"] = [
        first.entity_occurrences[0],
        second_occurrence,
    ]
    return AudioClipBinding.model_validate(payload)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def _visual_and_sidecar_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    run_root = tmp_path / "visual-run"
    visual_root = tmp_path / "visual-export"
    sidecar_root = tmp_path / "sidecar"
    source_video = tmp_path / "public" / "clip-a.mp4"
    source_video.parent.mkdir()
    source_video.write_bytes(b"source-video")
    (run_root / "clips" / "clip-a").mkdir(parents=True)
    (run_root / "clips" / "clip-a" / "clip.json").write_text(
        json.dumps(
            {
                "annotation": {
                    "entities": [
                        {
                            "entity_id": "e1",
                            "reference_type": "subject",
                            "phrase": "person in a blue jacket",
                            "grounding_prompt": "person in a blue jacket",
                        }
                    ]
                },
                "pairing": {"retained_entity_ids": ["e1"]},
            }
        ),
        encoding="utf-8",
    )
    reference_path = visual_root / "references" / "clip-a" / "subject_1.png"
    reference_path.parent.mkdir(parents=True)
    Image.new("RGBA", (13, 11), (20, 40, 80, 170)).save(reference_path)
    (visual_root / "samples.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "clip-a",
                "target_video": str(source_video),
                "t2v_caption": "A person speaks to the camera.",
                "r2v_instruction": "Use the person in the reference.",
                "references": [
                    {
                        "type": "entity",
                        "entity_id": "e1",
                        "image_path": "references/clip-a/subject_1.png",
                        "token": "<ref_subject_1>",
                        "scope": "full",
                        "visible_region": "whole",
                        "source_frame_index": 5,
                        "source_clip_uid": "clip-a",
                        "source_entity_id": "e1",
                        "synthetic": False,
                    }
                ],
                "source": {"parent_video_id": "parent-a", "clip_suffix": "0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frame_binding = _frame_binding(0.0, 1.0, confidence=0.95)
    sidecar_record = AudioBindingSidecar(
        clip_uid="clip-a",
        source_run_root=str(run_root),
        source_video_path=str(source_video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid="clip-a",
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(source_video),
                full_audio_path="evidence/full.wav",
                duration_seconds=2.0,
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
        bindings=[frame_binding],
        h3_ir=H3AudioBindingIR(
            clip_uid="clip-a",
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[
                PictureAsset(
                    picture_id="picture_1",
                    entity_id="e1",
                    path="immutable.png",
                )
            ],
            subjects=[
                SemanticSubject(
                    subject_id="subject_1",
                    entity_id="e1",
                    reference_type="subject",
                    phrase="person in a blue jacket",
                    source_assets=["picture_1"],
                )
            ],
            audio_assets=[],
            bindings=[frame_binding],
        ),
    )
    sidecar_path = sidecar_root / "clips" / "clip-a" / "audio_binding.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text(sidecar_record.model_dump_json(), encoding="utf-8")
    full_audio = tmp_path / "fixtures" / "full.flac"
    voice_audio = tmp_path / "fixtures" / "voice.flac"
    full_audio.parent.mkdir()
    full_audio.write_bytes(b"full-audio-fixture")
    voice_audio.write_bytes(b"voice-audio-fixture")
    return run_root, visual_root, sidecar_root, full_audio, voice_audio


def test_coalescing_preserves_sources_and_applies_duration_after_merge() -> None:
    turns = coalesce_audio_bindings(
        [_frame_binding(0.0, 0.3), _frame_binding(0.32, 0.7)],
        clip_uid="clip-a",
        sample_rate_hz=16000,
        maximum_gap_seconds=0.05,
        minimum_voice_reference_duration_seconds=0.5,
    )

    assert len(turns) == 1
    assert turns[0].source_binding_ids == ["binding_1", "binding_2"]
    assert turns[0].source_frame_ranges == [(0, 7), (8, 17)]
    assert turns[0].voice_reference_eligible is True


@pytest.mark.parametrize("status", ["overlap", "offscreen", "ambiguous", "no_speech"])
def test_non_bound_status_breaks_speech_turn_coalescing(status: str) -> None:
    turns = coalesce_audio_bindings(
        [
            _frame_binding(0.0, 0.4),
            _frame_binding(0.4, 0.8, status=status),
            _frame_binding(0.8, 1.2),
        ],
        clip_uid="clip-a",
        sample_rate_hz=16000,
        maximum_gap_seconds=0.1,
        minimum_voice_reference_duration_seconds=0.5,
    )

    assert [item.status for item in turns] == ["bound", status, "bound"]


def test_gap_entity_and_face_changes_do_not_merge() -> None:
    bindings = [
        _frame_binding(0.0, 0.3),
        _frame_binding(0.5, 0.8),
        _frame_binding(0.8, 1.1, entity_id="e2", face_track_id="face_2"),
    ]

    turns = coalesce_audio_bindings(
        bindings,
        clip_uid="clip-a",
        sample_rate_hz=16000,
        maximum_gap_seconds=0.1,
        minimum_voice_reference_duration_seconds=0.5,
    )

    assert len(turns) == 3


def test_primary_voice_selection_is_deterministic_and_one_per_entity() -> None:
    turns = coalesce_audio_bindings(
        [_frame_binding(0.0, 0.7, confidence=0.9), _frame_binding(1.0, 2.0)],
        clip_uid="clip-a",
        sample_rate_hz=16000,
        maximum_gap_seconds=0.05,
        minimum_voice_reference_duration_seconds=0.5,
    )

    selected = select_primary_voice_turns(
        turns,
        entity_order=["e1"],
        minimum_binding_confidence=0.8,
    )

    assert selected == {"e1": turns[1]}


def test_blockwise_top_k_is_deterministic_without_full_matrix() -> None:
    vectors = np.eye(17, dtype=np.float32)
    vectors[1] = vectors[0]
    index = NumpyBlockwiseTopKIndex(block_size=4)

    first_indices, first_scores = index.search(vectors, top_k=3)
    second_indices, second_scores = index.search(vectors, top_k=3)

    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first_scores, second_scores)
    assert tuple(index.maximum_similarity_block_shape) == (1, 4)
    assert first_indices[0, 0] == 1


def test_text_similarity_cannot_rescue_low_face_score(tmp_path: Path) -> None:
    bindings = [
        _canonical_binding(
            tmp_path,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            text=[1.0, 0.0],
        ),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[0.0, 1.0],
            voice=[1.0, 0.0],
            text=[1.0, 0.0],
        ),
    ]

    edges = build_pairwise_evidence(bindings, audio_root=tmp_path)

    assert edges
    assert all(item.same_person.status == "rejected" for item in edges)
    assert all(item.same_voice.status == "accepted" for item in edges)


def test_strict_cross_pair_requires_face_and_voice_and_preserves_target_text(
    tmp_path: Path,
) -> None:
    bindings = [
        _canonical_binding(
            tmp_path,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            transcript="target words",
        ),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            transcript="reference words",
        ),
    ]

    samples, _, report = build_audio_pair_samples(bindings, audio_root=tmp_path)
    cross_a = next(
        item
        for item in samples
        if item.pair_kind == "cross_pair" and item.target.clip_uid == "clip-a"
    )

    assert report["in_pair_count"] == 2
    assert report["cross_pair_count"] == 2
    assert "target words" in cross_a.annotation_draft.text
    assert "reference words" not in cross_a.annotation_draft.text
    assert cross_a.source_clip_binding_ids == [
        "clip_binding/clip-a",
        "clip_binding/clip-b",
    ]


def test_no_face_keeps_in_pair_but_blocks_strict_cross_pair(tmp_path: Path) -> None:
    bindings = [
        _canonical_binding(tmp_path, "clip-a", voice=[1.0, 0.0]),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
    ]

    samples, _, _ = build_audio_pair_samples(bindings, audio_root=tmp_path)

    assert any(
        item.pair_kind == "in_pair" and item.target.clip_uid == "clip-a"
        for item in samples
    )
    assert not any(
        item.pair_kind == "cross_pair" and item.target.clip_uid == "clip-a"
        for item in samples
    )


def test_face_only_and_voice_only_matches_never_enter_strict_export(
    tmp_path: Path,
) -> None:
    face_only = [
        _canonical_binding(
            tmp_path,
            "face-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
        _canonical_binding(
            tmp_path,
            "face-b",
            face=[1.0, 0.0],
            voice=[0.0, 1.0],
        ),
    ]
    voice_only = [
        _canonical_binding(
            tmp_path,
            "voice-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
        _canonical_binding(
            tmp_path,
            "voice-b",
            face=[0.0, 1.0],
            voice=[1.0, 0.0],
        ),
    ]

    face_samples, face_edges, _ = build_audio_pair_samples(
        face_only,
        audio_root=tmp_path,
    )
    voice_samples, voice_edges, _ = build_audio_pair_samples(
        voice_only,
        audio_root=tmp_path,
    )

    assert any(edge.same_person.status == "accepted" for edge in face_edges)
    assert any(edge.same_voice.status == "rejected" for edge in face_edges)
    assert any(edge.same_voice.status == "accepted" for edge in voice_edges)
    assert any(edge.same_person.status == "rejected" for edge in voice_edges)
    assert all(sample.pair_kind == "in_pair" for sample in face_samples)
    assert all(sample.pair_kind == "in_pair" for sample in voice_samples)


def test_candidate_and_nonmutual_edges_do_not_enter_strict_export(tmp_path: Path) -> None:
    bindings = [
        _canonical_binding(
            tmp_path,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[0.8, 0.6],
            voice=[0.8, 0.6],
        ),
        _canonical_binding(
            tmp_path,
            "clip-c",
            face=[0.8, 0.6],
            voice=[0.8, 0.6],
        ),
    ]
    config = AudioPairingConfig(top_k=1)

    samples, edges, _ = build_audio_pair_samples(
        bindings,
        audio_root=tmp_path,
        config=config,
    )
    edge = next(
        item
        for item in edges
        if item.target_entity_occurrence_id == "clip-a/e1"
        and item.reference_entity_occurrence_id == "clip-b/e1"
    )

    assert edge.same_person.status != "accepted"
    assert "face_not_mutual_top_k" in edge.same_person.reason_codes
    assert not any(
        sample.pair_kind == "cross_pair" and sample.target.clip_uid == "clip-a"
        for sample in samples
    )


def test_missing_one_subject_reference_blocks_cross_but_not_in_pair(
    tmp_path: Path,
) -> None:
    target = _two_subject_binding(tmp_path, "clip-a")
    reference = _canonical_binding(
        tmp_path,
        "clip-b",
        face=[1.0, 0.0],
        voice=[1.0, 0.0],
    )

    samples, _, report = build_audio_pair_samples(
        [target, reference],
        audio_root=tmp_path,
    )

    target_in_pair = next(
        sample
        for sample in samples
        if sample.pair_kind == "in_pair" and sample.target.clip_uid == "clip-a"
    )
    assert len(target_in_pair.subjects) == 2
    assert not any(
        sample.pair_kind == "cross_pair" and sample.target.clip_uid == "clip-a"
        for sample in samples
    )
    assert report["incomplete_cross_pair_targets"] == [
        {
            "clip_uid": "clip-a",
            "reason": "not_all_speaking_subjects_have_strict_reference",
        }
    ]


def test_zero_cross_variant_policy_keeps_only_in_pair(tmp_path: Path) -> None:
    bindings = [
        _canonical_binding(
            tmp_path,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
        ),
    ]

    samples, _, _ = build_audio_pair_samples(
        bindings,
        audio_root=tmp_path,
        config=AudioPairingConfig(max_cross_pair_variants_per_target=0),
    )

    assert [item.pair_id for item in samples] == [
        "in_pair/clip-a",
        "in_pair/clip-b",
    ]


def test_near_duplicate_source_video_is_not_cross_paired(tmp_path: Path) -> None:
    duplicate = b"same-source-video"
    bindings = [
        _canonical_binding(
            tmp_path,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            source_video_bytes=duplicate,
        ),
        _canonical_binding(
            tmp_path,
            "clip-b",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            source_video_bytes=duplicate,
        ),
    ]

    samples, _, _ = build_audio_pair_samples(bindings, audio_root=tmp_path)

    assert {item.pair_kind for item in samples} == {"in_pair"}


def test_missing_transcript_does_not_render_empty_dialogue(tmp_path: Path) -> None:
    binding = _canonical_binding(
        tmp_path,
        "clip-a",
        face=[1.0, 0.0],
        voice=[1.0, 0.0],
    )

    samples, _, _ = build_audio_pair_samples([binding], audio_root=tmp_path)

    assert len(samples) == 1
    assert "<d></d>" not in samples[0].annotation_draft.text
    assert samples[0].annotation_draft.annotation_status == "draft"
    assert samples[0].annotation_draft.is_final_annotation is False


def test_h3_export_is_relative_hash_checked_and_tree_exact(tmp_path: Path) -> None:
    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    bindings = [
        _canonical_binding(
            binding_root,
            "clip-a",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            transcript="target words",
        ),
        _canonical_binding(
            binding_root,
            "clip-b",
            face=[1.0, 0.0],
            voice=[1.0, 0.0],
            transcript="reference words",
        ),
    ]
    _write_bindings(binding_root, bindings)
    pair_root = tmp_path / "pairs"
    publish_audio_pair_dataset(
        audio_binding_root=binding_root,
        output_root=pair_root,
    )
    output_root = tmp_path / "h3"

    samples = export_h3_audio_dataset(
        audio_root=pair_root,
        output_root=output_root,
    )

    assert len(samples) == 4
    assert all(not Path(sample.target_video.path).is_absolute() for sample in samples)
    assert all(
        not Path(asset.path).is_absolute()
        for sample in samples
        for asset in [
            sample.target_full_audio,
            *sample.pictures,
            *sample.voice_reference_audio,
        ]
    )
    declared = {"dataset.json", "samples.jsonl", "pair_report.json"}
    for sample in samples:
        declared.add(sample.target_video.path)
        declared.add(sample.target_full_audio.path)
        declared.update(item.path for item in sample.pictures)
        declared.update(item.path for item in sample.voice_reference_audio)
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert actual == declared


def test_h3_overwrite_failure_preserves_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    binding = _canonical_binding(
        binding_root,
        "clip-a",
        face=[1.0, 0.0],
        voice=[1.0, 0.0],
    )
    _write_bindings(binding_root, [binding])
    pair_root = tmp_path / "pairs"
    publish_audio_pair_dataset(
        audio_binding_root=binding_root,
        output_root=pair_root,
    )
    output_root = tmp_path / "h3"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    def fail_copy(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated copy failure")

    monkeypatch.setattr(audio_export_module.shutil, "copyfile", fail_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        export_h3_audio_dataset(
            audio_root=pair_root,
            output_root=output_root,
            overwrite=True,
        )

    assert _tree_hashes(output_root) == {"sentinel.txt": _sha256(sentinel)}


def test_audio_asset_path_traversal_fails_closed(tmp_path: Path) -> None:
    binding = _canonical_binding(
        tmp_path,
        "clip-a",
        face=[1.0, 0.0],
        voice=[1.0, 0.0],
    )
    payload = binding.model_dump(mode="json")
    payload["full_audio"]["asset"]["path"] = "../outside.flac"

    with pytest.raises(ValidationError, match="safe export-relative path"):
        AudioClipBinding.model_validate(payload)


def test_visual_stage_order_remains_frozen() -> None:
    pipeline_source = Path("run_pipeline_v3.py").read_text(encoding="utf-8")

    assert '"audio_binding"' not in pipeline_source
    assert '"audio_pair"' not in pipeline_source
    assert '"export_h3"' not in pipeline_source


def test_precomputed_backends_materialize_hashable_assets(tmp_path: Path) -> None:
    full_source = tmp_path / "source.flac"
    voice_source = tmp_path / "voice.flac"
    full_source.write_bytes(b"full-audio")
    voice_source.write_bytes(b"voice-audio")
    media = PrecomputedAudioMediaBackend(
        {"clip-a": full_source},
        {"clip-a/e1": voice_source},
        {
            "clip-a": AudioStreamProvenance(
                stream_index=0,
                codec_name="aac",
                original_sample_rate_hz=48000,
                original_channels=2,
                duration_seconds=2.0,
                time_base="1/48000",
            )
        },
    )
    embeddings = PrecomputedEmbeddingBackend(
        {"clip-a/e1": np.asarray([3.0, 4.0], dtype=np.float32)},
        model_identifier="fixture",
    )

    full = media.materialize_full_audio(
        clip_uid="clip-a",
        source_video_path=tmp_path / "unused.mp4",
        destination=tmp_path / "out" / "full.flac",
        sample_rate_hz=48000,
        channels=2,
        output_format="flac",
    )
    voice = media.extract_voice_reference(
        clip_uid="clip-a",
        entity_id="e1",
        full_audio_path=full.path,
        start_time=0.0,
        end_time=1.0,
        destination=tmp_path / "out" / "voice.flac",
        sample_rate_hz=16000,
        output_format="flac",
    )
    embedding = embeddings.embed_speaker(
        entity_occurrence_id="clip-a/e1",
        audio_path=voice,
    )

    assert full.path.read_bytes() == b"full-audio"
    assert voice.read_bytes() == b"voice-audio"
    assert np.array_equal(embedding.vector, np.asarray([3.0, 4.0], dtype=np.float32))


def test_external_embedding_contract_records_command_and_normalized_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command == ["adapter", "--serve-one"]
        request = json.loads(str(kwargs["input"]))
        requests.append(request)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "request_id": request["request_id"],
                    "embedding": [1.0, 0.0],
                    "dimension": 2,
                    "dtype": "float32",
                    "normalized": True,
                }
            ),
            stderr="adapter diagnostic",
        )

    monkeypatch.setattr(audio_backends_module.subprocess, "run", fake_run)
    backend = ExternalSubprocessEmbeddingBackend(
        executable=["adapter", "--serve-one"],
        model_identifier="server-speaker-v1",
        checkpoint_sha256="a" * 64,
        timeout_seconds=10.0,
        diagnostics_root=tmp_path / "logs",
    )

    result = backend.embed_speaker(
        entity_occurrence_id="clip-a/e1",
        audio_path=tmp_path / "voice.flac",
    )

    assert requests[0]["operation"] == "speaker_embedding"
    assert result.backend_metadata == {
        "backend": "external_subprocess",
        "executable": ["adapter", "--serve-one"],
    }
    assert len(list((tmp_path / "logs").glob("*.stdout.log"))) == 1
    assert len(list((tmp_path / "logs").glob("*.stderr.log"))) == 1


def test_audio_config_thresholds_are_explicitly_uncalibrated() -> None:
    assert AudioBindingProductionConfig().fingerprint()
    assert AudioPairingConfig().fingerprint()
    assert _producer().thresholds_calibrated is False


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="the frozen Visual/H3 baseline requires Python 3.10+ strict zip",
)
def test_precomputed_end_to_end_binding_uses_one_immutable_visual_reference(
    tmp_path: Path,
) -> None:
    run_root, visual_root, sidecar_root, full_audio, voice_audio = (
        _visual_and_sidecar_fixture(tmp_path)
    )
    source_hashes = {
        "run": _tree_hashes(run_root),
        "visual": _tree_hashes(visual_root),
        "sidecar": _tree_hashes(sidecar_root),
    }
    source_reference = visual_root / "references" / "clip-a" / "subject_1.png"
    output_root = tmp_path / "audio-output"
    media = PrecomputedAudioMediaBackend(
        {"clip-a": full_audio},
        {"clip-a/e1": voice_audio},
        {
            "clip-a": AudioStreamProvenance(
                stream_index=0,
                codec_name="aac",
                original_sample_rate_hz=48000,
                original_channels=2,
                duration_seconds=2.0,
                time_base="1/48000",
            )
        },
    )
    face = PrecomputedEmbeddingBackend(
        {"clip-a/e1": np.asarray([1.0, 0.0], dtype=np.float32)},
        model_identifier="fake-arcface",
    )
    speaker = PrecomputedEmbeddingBackend(
        {"clip-a/e1": np.asarray([1.0, 0.0], dtype=np.float32)},
        model_identifier="fake-speaker",
    )

    outputs = build_audio_clip_binding_dataset(
        run_root=run_root,
        visual_export_root=visual_root,
        sidecar_root=sidecar_root,
        output_root=output_root,
        audio_backend=media,
        face_backend=face,
        speaker_backend=speaker,
    )

    occurrence = outputs[0].entity_occurrences[0]
    copied_reference = output_root / occurrence.visual_reference.image_asset.path
    assert copied_reference.read_bytes() == source_reference.read_bytes()
    assert occurrence.visual_reference.synthetic is False
    assert occurrence.face_embedding_asset is not None
    assert occurrence.voice_embedding_asset is not None
    assert occurrence.primary_voice_reference is not None
    assert occurrence.in_pair_eligible is True
    assert occurrence.cross_pair_eligible is True
    assert source_hashes == {
        "run": _tree_hashes(run_root),
        "visual": _tree_hashes(visual_root),
        "sidecar": _tree_hashes(sidecar_root),
    }


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="the frozen Visual/H3 baseline requires Python 3.10+ strict zip",
)
def test_clip_failure_is_isolated_and_recorded(tmp_path: Path) -> None:
    run_root, visual_root, sidecar_root, full_audio, voice_audio = (
        _visual_and_sidecar_fixture(tmp_path)
    )
    source_b = tmp_path / "public" / "clip-b.mp4"
    source_b.write_bytes(b"source-video-b")
    clip_a_payload = json.loads(
        (run_root / "clips" / "clip-a" / "clip.json").read_text(encoding="utf-8")
    )
    clip_b_path = run_root / "clips" / "clip-b" / "clip.json"
    clip_b_path.parent.mkdir()
    clip_b_path.write_text(json.dumps(clip_a_payload), encoding="utf-8")
    sample_a = json.loads(
        (visual_root / "samples.jsonl").read_text(encoding="utf-8")
    )
    sample_b = dict(sample_a)
    sample_b["sample_id"] = "clip-b"
    sample_b["target_video"] = str(source_b)
    sample_b["source"] = {"parent_video_id": "parent-b", "clip_suffix": "0"}
    sample_b["references"] = [dict(sample_a["references"][0])]
    sample_b["references"][0]["source_clip_uid"] = "clip-b"
    (visual_root / "samples.jsonl").write_text(
        json.dumps(sample_a) + "\n" + json.dumps(sample_b) + "\n",
        encoding="utf-8",
    )
    sidecar_a_path = sidecar_root / "clips" / "clip-a" / "audio_binding.json"
    sidecar_b_payload = json.loads(sidecar_a_path.read_text(encoding="utf-8"))
    sidecar_b_payload["clip_uid"] = "clip-b"
    sidecar_b_payload["source_video_path"] = str(source_b)
    sidecar_b_payload["evidence"]["clip_uid"] = "clip-b"
    sidecar_b_payload["evidence"]["audio"]["source_video_path"] = str(source_b)
    sidecar_b_payload["h3_ir"]["clip_uid"] = "clip-b"
    sidecar_b_path = sidecar_root / "clips" / "clip-b" / "audio_binding.json"
    sidecar_b_path.parent.mkdir()
    sidecar_b_path.write_text(json.dumps(sidecar_b_payload), encoding="utf-8")
    media = PrecomputedAudioMediaBackend(
        {"clip-a": full_audio},
        {"clip-a/e1": voice_audio},
        {
            "clip-a": AudioStreamProvenance(
                stream_index=0,
                codec_name="aac",
                original_sample_rate_hz=48000,
                original_channels=2,
                duration_seconds=2.0,
                time_base="1/48000",
            )
        },
    )
    face = PrecomputedEmbeddingBackend(
        {"clip-a/e1": np.asarray([1.0, 0.0], dtype=np.float32)},
        model_identifier="fake-arcface",
    )
    speaker = PrecomputedEmbeddingBackend(
        {"clip-a/e1": np.asarray([1.0, 0.0], dtype=np.float32)},
        model_identifier="fake-speaker",
    )
    output_root = tmp_path / "audio-output"

    outputs = build_audio_clip_binding_dataset(
        run_root=run_root,
        visual_export_root=visual_root,
        sidecar_root=sidecar_root,
        output_root=output_root,
        audio_backend=media,
        face_backend=face,
        speaker_backend=speaker,
    )

    assert [item.clip_uid for item in outputs] == ["clip-a"]
    failure = json.loads((output_root / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["clip_uid"] == "clip-b"
    assert failure["failure_type"] == "KeyError"


def test_clip_binding_jsonl_order_is_deterministic(tmp_path: Path) -> None:
    bindings = [
        _canonical_binding(tmp_path, "clip-b", voice=[1.0, 0.0]),
        _canonical_binding(tmp_path, "clip-a", voice=[1.0, 0.0]),
    ]
    path = tmp_path / "bindings.jsonl"
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in bindings),
        encoding="utf-8",
    )

    assert [item.clip_uid for item in load_clip_bindings(path)] == ["clip-b", "clip-a"]
