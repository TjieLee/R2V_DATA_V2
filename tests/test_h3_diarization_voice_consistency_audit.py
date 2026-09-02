from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from r2v_data_v2.h3.audio_backends import EmbeddingResult
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationBoundaryReconciliation,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    VoiceConsistencyAuditRecord,
    VoiceConsistencyAuditSummary,
    build_review_candidates,
    cosine_similarity,
    duration_bucket,
    run_diarization_voice_consistency_audit,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalSubjectVoice,
    FinalVisualReference,
)

_SAMPLE_RATE = 16000
_MODEL_HASH = "f" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, seconds: float = 3.0) -> None:
    samples = np.arange(round(seconds * _SAMPLE_RATE), dtype=np.int32)
    pcm = ((samples % 200) - 100).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def _raw(
    *,
    source_audio: Path,
    segment_id: str,
    start: float,
    end: float,
) -> RawDiarizationSegment:
    start_sample = round(start * _SAMPLE_RATE)
    end_sample = round(end * _SAMPLE_RATE)
    return RawDiarizationSegment(
        target_clip_uid="clip-a",
        segment_id=segment_id,
        speaker_cluster_id="speaker_0",
        backend_speaker_label="SPEAKER_00",
        backend_reported_start_time=start,
        backend_reported_end_time=end,
        backend_reported_start_sample=start_sample,
        backend_reported_end_sample=end_sample,
        start_time=start,
        end_time=end,
        source_start_sample=start_sample,
        source_end_sample=end_sample,
        source_audio_path=str(source_audio),
        source_audio_sha256=_sha256(source_audio),
        source_sample_rate_hz=_SAMPLE_RATE,
        backend="fixture_diarizen",
        model_identifier="fixture/diarizen",
        model_fingerprint="a" * 64,
        backend_configuration_fingerprint="b" * 64,
        boundary_reconciliation=DiarizationBoundaryReconciliation(
            adjusted=False,
            end_clamped=False,
            end_overrun_samples=0,
            end_overrun_seconds=0,
        ),
    )


def _bound(
    raw: RawDiarizationSegment,
    *,
    entity_id: str,
    direct_anchor_seconds: float,
) -> BoundDiarizationSegment:
    direct_samples = round(direct_anchor_seconds * _SAMPLE_RATE)
    return BoundDiarizationSegment(
        target_clip_uid=raw.target_clip_uid,
        segment_id=raw.segment_id,
        speaker_cluster_id=raw.speaker_cluster_id,
        start_time=raw.start_time,
        end_time=raw.end_time,
        source_start_sample=raw.source_start_sample,
        source_end_sample=raw.source_end_sample,
        cluster_binding_status="candidate_mapped",
        entity_id=entity_id,
        entity_occurrence_id=f"{raw.target_clip_uid}/{entity_id}",
        direct_anchor_samples=direct_samples,
        direct_anchor_seconds=direct_samples / _SAMPLE_RATE,
        identity_scope=(
            "direct_anchor_present"
            if direct_samples > 0
            else "cluster_propagated_only"
        ),
    )


def _reference(tmp_path: Path, entity_id: str, index: int) -> FinalVisualReference:
    artifact = tmp_path / f"{entity_id}.png"
    artifact.write_bytes(entity_id.encode())
    return FinalVisualReference(
        image_id=f"image_{index}",
        image_index=index,
        kind="subject",
        image_path=f"references/{entity_id}.png",
        image_artifact_path=str(artifact),
        entity_id=entity_id,
        source_frame_index=0,
        scope="full",
        visible_region="whole",
        synthetic=False,
    )


def _sample(
    tmp_path: Path,
    *,
    source_audio: Path,
    voice_path: Path,
) -> FinalH3SampleV2:
    video = tmp_path / "clip-a.mp4"
    video.write_bytes(b"video")
    return FinalH3SampleV2(
        sample_id="clip-a/in_pair",
        pair_id="in_pair/clip-a",
        pair_type="in_pair",
        clip_uid="clip-a",
        clip_display_path="category/show/clip-a",
        media_collection_relpath="category/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip-a",
        shard_id="shard-a",
        target_video=str(video),
        target_full_audio_path=str(source_audio),
        target_full_audio_sha256=_sha256(source_audio),
        r2v_instruction="Use Image 1 and Image 2.",
        visual_references=[
            _reference(tmp_path, "e1", 1),
            _reference(tmp_path, "e2", 2),
        ],
        subject_voices=[
            FinalSubjectVoice(
                subject_index=1,
                entity_id="e1",
                target_occurrence_id="clip-a/e1",
                voice_reference_path=str(voice_path),
                voice_reference_sha256=_sha256(voice_path),
                source_start=0.0,
                source_end=1.0,
                source_start_sample=0,
                source_end_sample=32000,
                sample_mapping_policy="round_time_seconds_times_32000_v1",
                voice_source="target",
            )
        ],
        speech_segments=[],
    )


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),  # type: ignore[attr-defined]
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


class _FakeSpeakerBackend:
    model_identifier = "speechbrain/spkrec-ecapa-voxceleb"
    checkpoint_sha256 = _MODEL_HASH

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def embed_speaker(
        self,
        *,
        entity_occurrence_id: str,
        audio_path: Path,
    ) -> EmbeddingResult:
        self.calls.append((entity_occurrence_id, audio_path))
        if entity_occurrence_id.startswith("primary/"):
            vector = np.array([3.0, 0.0], dtype=np.float32)
        elif entity_occurrence_id.endswith("segment_0001"):
            vector = np.array([2.0, 0.0], dtype=np.float32)
        else:
            vector = np.array([0.0, 4.0], dtype=np.float32)
        return EmbeddingResult(
            vector=vector,
            model_identifier=self.model_identifier,
            checkpoint_sha256=self.checkpoint_sha256,
            backend_metadata={"backend": "fixture"},
        )


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    production = tmp_path / "production"
    source_audio = tmp_path / "source.wav"
    voice = tmp_path / "voice.flac"
    _write_wav(source_audio)
    voice.write_bytes(b"voice-reference")
    raw = [
        _raw(
            source_audio=source_audio,
            segment_id="segment_0001",
            start=0.0,
            end=1.0,
        ),
        _raw(
            source_audio=source_audio,
            segment_id="segment_0002",
            start=1.0,
            end=2.0,
        ),
        _raw(
            source_audio=source_audio,
            segment_id="segment_0003",
            start=2.0,
            end=3.0,
        ),
    ]
    bound = [
        _bound(raw[0], entity_id="e1", direct_anchor_seconds=0.2),
        _bound(raw[1], entity_id="e1", direct_anchor_seconds=0.0),
        _bound(raw[2], entity_id="e2", direct_anchor_seconds=0.0),
    ]
    sample = _sample(tmp_path, source_audio=source_audio, voice_path=voice)
    raw_path = production / "diarization/raw_segments.jsonl"
    bound_path = production / "diarization/bound_segments.jsonl"
    samples_path = production / "h3/samples.jsonl"
    _write_jsonl(raw_path, raw)
    _write_jsonl(bound_path, bound)
    _write_jsonl(samples_path, [sample])
    return production, [raw_path, bound_path, samples_path, source_audio, voice]


def _records(path: Path) -> list[VoiceConsistencyAuditRecord]:
    return [
        VoiceConsistencyAuditRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_cosine_similarity_normalizes_inputs() -> None:
    assert cosine_similarity(
        np.array([3.0, 0.0], dtype=np.float32),
        np.array([9.0, 0.0], dtype=np.float32),
    ) == pytest.approx(1.0)
    assert cosine_similarity(
        np.array([3.0, 0.0], dtype=np.float32),
        np.array([0.0, 4.0], dtype=np.float32),
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="norm must be positive"):
        cosine_similarity(np.zeros(2), np.ones(2))


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.749999, "<0.75s"),
        (0.75, "0.75-1.0s"),
        (0.999999, "0.75-1.0s"),
        (1.0, "1.0-2.0s"),
        (1.999999, "1.0-2.0s"),
        (2.0, ">=2.0s"),
    ],
)
def test_duration_bucket_boundaries(duration: float, expected: str) -> None:
    assert duration_bucket(duration) == expected


def test_audit_caches_primary_voice_and_preserves_scope_and_inputs(
    tmp_path: Path,
) -> None:
    production, inputs = _fixture(tmp_path)
    before = {str(path): _sha256(path) for path in inputs}
    backend = _FakeSpeakerBackend()
    output = tmp_path / "audit"
    summary = run_diarization_voice_consistency_audit(
        audio_production_root=production,
        speaker_backend=backend,
        output_root=output,
    )
    after = {str(path): _sha256(path) for path in inputs}
    assert after == before
    assert summary == VoiceConsistencyAuditSummary.model_validate_json(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary.audited_segment_count == 2
    assert summary.skipped_segment_count == 1
    assert summary.primary_voice_embedding_call_count == 1
    assert summary.segment_embedding_call_count == 2
    assert summary.model_call_count == 3
    assert sum(call[0].startswith("primary/") for call in backend.calls) == 1
    records = _records(output / "records.jsonl")
    assert [item.segment_id for item in records] == [
        "segment_0001",
        "segment_0002",
    ]
    assert records[0].identity_scope == "direct_anchor_present"
    assert records[0].direct_anchor_seconds == pytest.approx(0.2)
    assert records[0].cosine_similarity == pytest.approx(1.0)
    assert records[1].identity_scope == "cluster_propagated_only"
    assert records[1].direct_anchor_seconds == 0
    assert records[1].cosine_similarity == pytest.approx(0.0)
    with wave.open(str(output / records[0].segment_audio_path), "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 16000
    skipped = [
        json.loads(line)
        for line in (output / "skipped.jsonl").read_text().splitlines()
    ]
    assert skipped[0]["segment_id"] == "segment_0003"
    assert skipped[0]["reason_code"] == "missing_target_primary_voice"
    assert summary.skip_reason_counts == {"missing_target_primary_voice": 1}
    assert set(summary.duration_bucket_distributions["direct_anchor_present"]) == {
        "<0.75s",
        "0.75-1.0s",
        "1.0-2.0s",
        ">=2.0s",
    }
    assert (
        summary.duration_bucket_distributions["direct_anchor_present"][
            "1.0-2.0s"
        ].count
        == 1
    )
    assert (
        summary.duration_bucket_distributions["cluster_propagated_only"][
            "1.0-2.0s"
        ].count
        == 1
    )
    assert summary.similarity_threshold_applied is False
    assert summary.binding_modified is False
    assert summary.production_artifacts_modified is False


def test_audit_outputs_are_deterministic(tmp_path: Path) -> None:
    production, _ = _fixture(tmp_path)
    outputs = [tmp_path / "audit-a", tmp_path / "audit-b"]
    for output in outputs:
        run_diarization_voice_consistency_audit(
            audio_production_root=production,
            speaker_backend=_FakeSpeakerBackend(),
            output_root=output,
        )
    for name in (
        "records.jsonl",
        "skipped.jsonl",
        "review_candidates.jsonl",
        "summary.json",
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_default_output_root_is_independent_audit_directory(tmp_path: Path) -> None:
    production, _ = _fixture(tmp_path)
    run_diarization_voice_consistency_audit(
        audio_production_root=production,
        speaker_backend=_FakeSpeakerBackend(),
    )
    output = production / "diarization_voice_consistency_audit_v1"
    assert (output / "records.jsonl").is_file()
    assert (output / "summary.json").is_file()


def test_review_candidates_deduplicate_selection_groups(tmp_path: Path) -> None:
    production, _ = _fixture(tmp_path)
    output = tmp_path / "audit"
    run_diarization_voice_consistency_audit(
        audio_production_root=production,
        speaker_backend=_FakeSpeakerBackend(),
        output_root=output,
    )
    candidates = build_review_candidates(_records(output / "records.jsonl"))
    assert len(candidates) == 2
    propagated = next(
        item for item in candidates if item.identity_scope == "cluster_propagated_only"
    )
    assert propagated.selection_groups == [
        "propagated_lowest",
        "propagated_middle",
        "propagated_highest",
    ]
    direct = next(
        item for item in candidates if item.identity_scope == "direct_anchor_present"
    )
    assert direct.selection_groups == ["direct_anchor_lowest_control"]
