from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.h3.asr_transcription import (
    ASR_HUMAN_QA_LABELS,
    ASRBackendResult,
    ASRDecoderDiagnostics,
    ASRHumanQACounts,
    ASRHumanQAExport,
    ASRHumanQALabel,
    ASRTranscriptionFailure,
    ASRTurnRecord,
    ASRTurnSegmentationProvenance,
    FasterWhisperASRBackend,
    WhisperASRConfig,
    asr_output_root,
    build_asr_inventory,
    prepare_asr_audio,
    run_asr_transcription,
)
from r2v_data_v2.h3.audio_production import (
    H3ProductionInPair,
    H3ProductionInPairSubject,
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
from tools.run_h3_asr_transcription import _model_identity, _parser, _reuse_existing
from tools.run_h3_asr_transcription import main as asr_main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_pcm16_wav(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate_hz: int = 16000,
    channels: int = 1,
) -> None:
    values = np.asarray(samples, dtype="<i2")
    if values.ndim == 1:
        values = values.reshape(-1, channels)
    assert values.shape[1] == channels
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate_hz)
        destination.writeframes(values.tobytes())


def _binding(
    *,
    start: float,
    end: float,
    entity_id: str,
    face_track_id: str,
) -> AudioEntityBinding:
    return AudioEntityBinding(
        start_time=start,
        end_time=end,
        entity_id=entity_id,
        face_track_id=face_track_id,
        status="bound",
        confidence=0.93,
        evidence=BindingEvidence(
            active_face_track_ids=[face_track_id],
            face_speaking_probabilities={face_track_id: 0.91},
            association_confidence=0.95,
            audio_quality_usable=True,
            synchronization_plausible=True,
            clean_training_eligible=True,
        ),
    )


def _write_target(
    pairs_root: Path,
    *,
    clip_uid: str,
    intervals: list[tuple[float, float, str]],
    sample_rate_hz: int = 16000,
    channels: int = 1,
    samples: np.ndarray | None = None,
) -> H3ProductionInPair:
    media_root = pairs_root.parent / "fixture_media"
    video = media_root / f"{clip_uid}.mp4"
    audio = media_root / f"{clip_uid}.wav"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(f"video:{clip_uid}".encode())
    frame_count = max(
        sample_rate_hz * 2,
        round(max(end for _, end, _ in intervals) * sample_rate_hz) + 10,
    )
    if samples is None:
        mono = np.arange(frame_count, dtype=np.int32) % 20000 - 10000
        if channels == 1:
            samples = mono.astype(np.int16)
        else:
            samples = np.column_stack(
                [mono, -mono],
            ).astype(np.int16)
    _write_pcm16_wav(
        audio,
        samples,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )
    bindings = [
        _binding(
            start=start,
            end=end,
            entity_id=entity_id,
            face_track_id=f"face_{index}",
        )
        for index, (start, end, entity_id) in enumerate(intervals, start=1)
    ]
    entity_ids = list(dict.fromkeys(entity_id for _, _, entity_id in intervals))
    pictures = [
        PictureAsset(
            picture_id=f"picture_{index}",
            entity_id=entity_id,
            path=f"references/{clip_uid}/{entity_id}.png",
        )
        for index, entity_id in enumerate(entity_ids, start=1)
    ]
    subjects = [
        SemanticSubject(
            subject_id=f"subject_{index}",
            entity_id=entity_id,
            reference_type="subject",
            phrase=f"person {index}",
            source_assets=[f"picture_{index}"],
        )
        for index, entity_id in enumerate(entity_ids, start=1)
    ]
    sidecar = AudioBindingSidecar(
        clip_uid=clip_uid,
        source_run_root="/read-only/visual-run",
        source_video_path=str(video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid=clip_uid,
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(video),
                full_audio_path=str(audio),
                duration_seconds=frame_count / sample_rate_hz,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            ),
        ),
        bindings=bindings,
        h3_ir=H3AudioBindingIR(
            clip_uid=clip_uid,
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=pictures,
            subjects=subjects,
            audio_assets=[],
            bindings=bindings,
        ),
    )
    sidecar_path = (
        pairs_root.parent / "audio" / "clips" / clip_uid / "audio_binding.json"
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return H3ProductionInPair(
        pair_id=f"in_pair/{clip_uid}",
        target_clip_uid=clip_uid,
        target_video_path=str(video),
        target_full_audio_path=str(audio),
        target_audio_binding_path=str(sidecar_path),
        subjects=[
            H3ProductionInPairSubject(
                subject_index=index,
                target_occurrence_id=f"{clip_uid}/{entity_id}",
                target_entity_id=entity_id,
                target_visual_reference_path=(
                    f"visual/{clip_uid}/{entity_id}.png"
                ),
                target_primary_voice_reference_path=(
                    f"primary_voice/{clip_uid}/{entity_id}.flac"
                ),
            )
            for index, entity_id in enumerate(entity_ids, start=1)
        ],
    )


def _write_pairs(
    root: Path,
    specs: list[tuple[str, list[tuple[float, float, str]]]],
) -> Path:
    pairs_root = root / "production" / "pairs"
    pairs_root.mkdir(parents=True)
    pairs = [
        _write_target(pairs_root, clip_uid=clip_uid, intervals=intervals)
        for clip_uid, intervals in specs
    ]
    (pairs_root / "in_pairs.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in reversed(pairs)),
        encoding="utf-8",
    )
    (pairs_root / "cross_pairs.jsonl").write_text(
        json.dumps(
            {
                "target_clip_uid": specs[0][0],
                "donor_clip_uid": "donor-only",
                "donor_video_path": "/must/not/reach/asr.mp4",
                "target_primary_voice_reference_path": "/must/not/reach/asr.flac",
                "embedding_path": "/must/not/reach/asr.npy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pairs_root


def _diagnostics() -> ASRDecoderDiagnostics:
    return ASRDecoderDiagnostics(
        detected_language="en",
        language_probability=0.97,
        avg_log_probability=-0.2,
        no_speech_probability=0.03,
        compression_ratio=1.1,
        decoder_segment_count=1,
    )


class _FakeBackend:
    def __init__(
        self,
        results: list[ASRBackendResult | ASRTranscriptionFailure] | None = None,
    ) -> None:
        self.results = list(results or [])
        self.calls: list[tuple[np.ndarray, int]] = []
        self.provenance = WhisperASRConfig(
            model_identifier="fixture/whisper-large-v3",
            device="cpu",
            compute_type="int8",
        ).provenance()

    def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
    ) -> ASRBackendResult:
        self.calls.append((audio.copy(), sample_rate_hz))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, ASRTranscriptionFailure):
                raise result
            return result
        return ASRBackendResult(
            text=f"transcript-{len(self.calls)}",
            language="en",
            diagnostics=_diagnostics(),
        )


def _read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_inventory_uses_only_in_pairs_and_preserves_authoritative_turns(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(
        tmp_path,
        [
            ("clip-b", [(0.2, 0.8, "e1")]),
            ("clip-a", [(0.125, 0.225, "e1"), (0.5, 1.0, "e2")]),
        ],
    )

    inventory = build_asr_inventory(pairs_root=pairs, mode="production")

    assert [item.target_clip_uid for item in inventory.targets] == [
        "clip-a",
        "clip-b",
    ]
    assert [(item.target_clip_uid, item.turn_id) for item in inventory.jobs] == [
        ("clip-a", "turn_1"),
        ("clip-a", "turn_2"),
        ("clip-b", "turn_1"),
    ]
    first = inventory.jobs[0]
    assert (
        first.entity_id,
        first.entity_occurrence_id,
        first.start_time,
        first.end_time,
    ) == ("e1", "clip-a/e1", 0.125, 0.225)
    assert (first.source_start_sample, first.source_end_sample) == (2000, 3600)
    assert inventory.source_target_count == inventory.selected_target_count == 2
    assert inventory.selected_turn_count == 3
    assert inventory.parent_quota_applied is False
    assert inventory.donor_media_used is False
    assert first.segment_provenance.model_dump() == {
        "boundary_source": "frozen_audio_binding_turns_v1",
        "source_segment_id": "turn_1",
        "speaker_cluster_id": None,
        "entity_binding_source": "lr_asd_visual_entity_binding_v1",
    }
    assert "donor-only" not in inventory.model_dump_json()
    backend = _FakeBackend()
    run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr",
        backend=backend,
    )
    assert len(backend.calls) == inventory.selected_turn_count == 3


def test_replacement_segment_inventory_does_not_change_whisper_inference_contract(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", [(0.0, 0.4, "e1")])])
    inventory = build_asr_inventory(pairs_root=pairs, mode="production")
    replacement_job = inventory.jobs[0].model_copy(
        update={
            "segment_provenance": ASRTurnSegmentationProvenance(
                boundary_source="diarization_speaker_segments_v1",
                source_segment_id="speaker_2/segment_7",
                speaker_cluster_id="speaker_2",
                entity_binding_source="clip_overlap_entity_mapping_v1",
            )
        }
    )
    replacement_inventory = inventory.model_copy(update={"jobs": [replacement_job]})
    backend = _FakeBackend()

    run_asr_transcription(
        inventory=replacement_inventory,
        output_root=tmp_path / "asr",
        backend=backend,
    )

    assert len(backend.calls) == 1
    waveform, sample_rate_hz = backend.calls[0]
    assert isinstance(waveform, np.ndarray)
    assert sample_rate_hz == 16000
    record = ASRTurnRecord.model_validate(
        _read_records(tmp_path / "asr" / "turns.jsonl")[0]
    )
    assert record.entity_occurrence_id == "clip-a/e1"
    assert record.segment_provenance.speaker_cluster_id == "speaker_2"
    assert record.segment_provenance.source_segment_id == "speaker_2/segment_7"


def test_production_keeps_every_target_and_turn_without_parent_quota(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(
        tmp_path,
        [
            (f"clip-{index:02d}", [(0.0, 0.4, "e1")])
            for index in range(6)
        ],
    )

    inventory = build_asr_inventory(pairs_root=pairs, mode="production")

    assert inventory.selected_target_count == inventory.source_target_count == 6
    assert inventory.selected_turn_count == 6
    assert inventory.selection_mode == "complete_target_inventory_v1"
    assert inventory.bounded_selection_applied is False
    assert inventory.parent_quota_applied is False


def test_pilot20_matches_multi_subject_first_target_selection(tmp_path: Path) -> None:
    specs = [
        (f"clip-{index:02d}", [(0.0, 0.4, "e1")])
        for index in range(22)
    ]
    specs[-1] = (
        "clip-99",
        [(0.0, 0.4, "e1"), (0.8, 1.2, "e2")],
    )
    pairs = _write_pairs(tmp_path, specs)

    first = build_asr_inventory(pairs_root=pairs, mode="pilot20")
    second = build_asr_inventory(pairs_root=pairs, mode="pilot20")

    assert first == second
    assert first.selected_target_count == 20
    assert first.targets[0].target_clip_uid == "clip-99"
    assert [item.target_clip_uid for item in first.targets[1:]] == [
        f"clip-{index:02d}" for index in range(19)
    ]
    assert first.selected_turn_count == 21


def test_exact_source_crop_and_short_turn_reach_audio_only_backend(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "production" / "pairs"
    pairs_root.mkdir(parents=True)
    samples = np.arange(16000, dtype=np.int16) - 8000
    pair = _write_target(
        pairs_root,
        clip_uid="clip-a",
        intervals=[(0.125, 0.225, "e1")],
        samples=samples,
    )
    (pairs_root / "in_pairs.jsonl").write_text(
        pair.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (pairs_root / "cross_pairs.jsonl").write_text("", encoding="utf-8")
    inventory = build_asr_inventory(pairs_root=pairs_root, mode="production")
    backend = _FakeBackend()

    summary = run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr",
        backend=backend,
    )

    assert summary.transcribed_count == summary.turn_count == 1
    assert len(backend.calls) == 1
    waveform, sample_rate = backend.calls[0]
    np.testing.assert_array_equal(
        waveform,
        samples[2000:3600].astype(np.float32) / 32768.0,
    )
    assert sample_rate == 16000
    record = ASRTurnRecord.model_validate(_read_records(tmp_path / "asr" / "turns.jsonl")[0])
    assert record.entity_occurrence_id == "clip-a/e1"
    assert (record.start_time, record.end_time) == (0.125, 0.225)
    assert (record.source_start_sample, record.source_end_sample) == (2000, 3600)
    assert record.preprocessing.padding_seconds == 0
    assert record.task == "transcribe"


def test_preprocessing_downmixes_and_resamples_after_exact_source_crop(
    tmp_path: Path,
) -> None:
    pairs_root = tmp_path / "production" / "pairs"
    pairs_root.mkdir(parents=True)
    stereo = np.column_stack(
        [np.full(8000, 1000, dtype=np.int16), np.full(8000, 3000, dtype=np.int16)]
    )
    pair = _write_target(
        pairs_root,
        clip_uid="clip-a",
        intervals=[(0.25, 0.5, "e1")],
        sample_rate_hz=8000,
        channels=2,
        samples=stereo,
    )
    (pairs_root / "in_pairs.jsonl").write_text(pair.model_dump_json() + "\n")
    (pairs_root / "cross_pairs.jsonl").write_text("")
    job = build_asr_inventory(pairs_root=pairs_root, mode="production").jobs[0]

    prepared = prepare_asr_audio(job)

    assert (job.source_start_sample, job.source_end_sample) == (2000, 4000)
    assert prepared.waveform.shape == (4000,)
    assert prepared.waveform[0] == pytest.approx(2000 / 32768.0)
    assert prepared.preprocessing.resampled is True
    assert prepared.preprocessing.downmixed is True


def test_empty_output_is_uncertain_and_backend_error_fails_closed(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(
        tmp_path,
        [
            (
                "clip-a",
                [(0.0, 0.4, "e1"), (0.8, 1.2, "e2")],
            )
        ],
    )
    inventory = build_asr_inventory(pairs_root=pairs, mode="production")
    backend = _FakeBackend(
        [
            ASRBackendResult(
                text="   ",
                language="zh",
                diagnostics=_diagnostics(),
            ),
            ASRTranscriptionFailure(
                code="asr_backend_failed",
                reason="fixture GPU failure",
            ),
        ]
    )

    summary = run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr",
        backend=backend,
    )
    records = _read_records(tmp_path / "asr" / "turns.jsonl")

    assert [item["status"] for item in records] == ["uncertain", "failed"]
    assert all(item["text"] is None for item in records)
    assert records[0]["failure"] is None
    assert records[0]["warnings"] == ["empty_transcript"]
    assert records[1]["failure"]["code"] == "asr_backend_failed"
    assert summary.uncertain_count == summary.failed_count == 1
    assert summary.failure_reason_counts == {"asr_backend_failed": 1}


def test_source_audio_hash_change_is_detected_before_backend_call(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", [(0.0, 0.4, "e1")])])
    inventory = build_asr_inventory(pairs_root=pairs, mode="production")
    Path(inventory.jobs[0].source_audio_path).write_bytes(b"changed")
    backend = _FakeBackend()

    summary = run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr",
        backend=backend,
    )

    record = _read_records(tmp_path / "asr" / "turns.jsonl")[0]
    assert record["status"] == "failed"
    assert record["failure"]["code"] == "source_audio_changed"
    assert summary.backend_call_count == 0
    assert backend.calls == []
    assert not (tmp_path / "asr" / "review_media" / "clip-a" / "turn_1.wav").exists()
    assert "[unavailable]" in (tmp_path / "asr" / "review.html").read_text(
        encoding="utf-8"
    )


def test_faster_whisper_backend_is_local_transcribe_without_vad_or_carryover() -> None:
    factory_calls = []
    transcribe_calls = []

    class _Model:
        def transcribe(self, audio, **kwargs):
            transcribe_calls.append((audio.copy(), kwargs))
            return (
                iter(
                    [
                        SimpleNamespace(
                            text=" original language",
                            avg_logprob=-0.1,
                            no_speech_prob=0.02,
                            compression_ratio=1.05,
                        )
                    ]
                ),
                SimpleNamespace(language="zh", language_probability=0.98),
            )

    def factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return _Model()

    backend = FasterWhisperASRBackend(
        WhisperASRConfig(
            model_identifier="/local/whisper-large-v3",
            device="cuda:3",
            compute_type="float16",
        ),
        model_factory=factory,
    )
    result = backend.transcribe(
        audio=np.zeros(320, dtype=np.float32),
        sample_rate_hz=16000,
    )

    assert factory_calls == [
        (
            ("/local/whisper-large-v3",),
            {
                "device": "cuda",
                "device_index": 3,
                "compute_type": "float16",
                "local_files_only": True,
            },
        )
    ]
    assert transcribe_calls[0][1] == {
        "task": "transcribe",
        "condition_on_previous_text": False,
        "vad_filter": False,
        "word_timestamps": False,
    }
    assert "translate" not in json.dumps(transcribe_calls[0][1])
    assert result.text == "original language"
    assert result.language == "zh"


def test_outputs_and_review_are_deterministic_and_pairs_remain_read_only(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(
        tmp_path,
        [("clip-a", [(0.0, 0.4, "e1")]), ("clip-b", [(0.2, 0.7, "e1")])],
    )
    before = _tree_hashes(pairs)
    inventory = build_asr_inventory(pairs_root=pairs, mode="production")

    run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr-one",
        backend=_FakeBackend(),
    )
    run_asr_transcription(
        inventory=inventory,
        output_root=tmp_path / "asr-two",
        backend=_FakeBackend(),
    )

    assert (tmp_path / "asr-one" / "turns.jsonl").read_bytes() == (
        tmp_path / "asr-two" / "turns.jsonl"
    ).read_bytes()
    review = (tmp_path / "asr-one" / "review.html").read_text(encoding="utf-8")
    assert review == (tmp_path / "asr-two" / "review.html").read_text(
        encoding="utf-8"
    )
    assert all(label in review for label in ("CORRECT", "WRONG", "UNCERTAIN"))
    assert "exact crop" in review
    assert _tree_hashes(pairs) == before


def test_review_page_exports_deterministic_human_qa_contract(tmp_path: Path) -> None:
    pairs = _write_pairs(
        tmp_path,
        [("clip-a", [(0.0, 0.4, "e1"), (0.8, 1.2, "e2")])],
    )
    inventory = build_asr_inventory(pairs_root=pairs, mode="pilot20")
    output_root = tmp_path / "asr"
    run_asr_transcription(
        inventory=inventory,
        output_root=output_root,
        backend=_FakeBackend(),
    )

    review = (output_root / "review.html").read_text(encoding="utf-8")
    metadata_text = review.split("const qaMetadata = ", 1)[1].split(
        ";\nconst qaAllowedLabels",
        1,
    )[0]
    metadata = json.loads(metadata_text)

    assert "Export QA JSON" in review
    assert "Clear QA labels" in review
    assert "qa-progress" in review
    assert all(
        name in review
        for name in (
            "qa-correct-count",
            "qa-wrong-count",
            "qa-uncertain-count",
            "qa-unlabeled-count",
        )
    )
    assert metadata["schema_version"] == "r2v.h3.asr_human_qa.1"
    assert metadata["inventory_fingerprint"] == inventory.inventory_fingerprint
    assert metadata["mode"] == "pilot20"
    assert metadata["allowed_labels"] == list(ASR_HUMAN_QA_LABELS)
    assert [item["turn_id"] for item in metadata["turns"]] == [
        "turn_1",
        "turn_2",
    ]
    assert "asr_pilot20_human_qa.json" in review
    assert "restoreQALabels();" in review
    assert "updateQACounts();" in review
    assert "onchange='saveLabel(this)'" in review
    assert "window.confirm(" in review
    assert "localStorage.removeItem(turn.storage_key)" in review
    assert "localStorage.clear(" not in review


def test_human_qa_schema_reconciles_labels_and_unlabeled_count() -> None:
    export = ASRHumanQAExport(
        inventory_fingerprint="a" * 64,
        mode="pilot20",
        label_count=2,
        total_turn_count=3,
        counts=ASRHumanQACounts(
            CORRECT=1,
            WRONG=1,
            UNCERTAIN=0,
            UNLABELED=1,
        ),
        labels=[
            ASRHumanQALabel(
                target_clip_uid="clip-a",
                turn_id="turn_1",
                entity_occurrence_id="clip-a/e1",
                label="CORRECT",
            ),
            ASRHumanQALabel(
                target_clip_uid="clip-a",
                turn_id="turn_2",
                entity_occurrence_id="clip-a/e2",
                label="WRONG",
            ),
        ],
    )

    assert export.counts.UNLABELED == 1
    label_schema = ASRHumanQALabel.model_json_schema()["properties"]["label"]
    assert label_schema["enum"] == list(ASR_HUMAN_QA_LABELS)
    with pytest.raises(ValueError):
        ASRHumanQALabel(
            target_clip_uid="clip-a",
            turn_id="turn_1",
            entity_occurrence_id="clip-a/e1",
            label="ACCEPT",  # type: ignore[arg-type]
        )


def test_review_only_regeneration_uses_no_inventory_builder_or_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", [(0.125, 0.225, "e1")])])
    inventory = build_asr_inventory(pairs_root=pairs, mode="pilot20")
    output_root = tmp_path / "asr_pilot20"
    run_asr_transcription(
        inventory=inventory,
        output_root=output_root,
        backend=_FakeBackend(),
    )
    immutable_names = ("inventory.json", "turns.jsonl", "summary.json")
    before = {name: (output_root / name).read_bytes() for name in immutable_names}
    turn_media = output_root / "review_media" / "clip-a" / "turn_1.wav"
    turn_media.unlink()
    (output_root / "review.html").write_text("stale review", encoding="utf-8")

    def unexpected_call(*args, **kwargs):
        raise AssertionError("review-only regeneration must not construct inference")

    monkeypatch.setattr(
        "tools.run_h3_asr_transcription.build_asr_inventory",
        unexpected_call,
    )
    monkeypatch.setattr(
        "tools.run_h3_asr_transcription.FasterWhisperASRBackend",
        unexpected_call,
    )

    result = asr_main(
        [
            "--audio-run-root",
            str(tmp_path),
            "--mode",
            "pilot20",
            "--regenerate-review",
        ]
    )

    assert result == {
        "mode": "pilot20",
        "review_regenerated": True,
        "model_loaded": False,
        "backend_calls": 0,
        "turn_count": 1,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "regenerated_turn_media_count": 1,
    }
    assert json.loads(capsys.readouterr().out) == result
    assert {name: (output_root / name).read_bytes() for name in immutable_names} == before
    assert turn_media.is_file()
    assert "Export QA JSON" in (output_root / "review.html").read_text(
        encoding="utf-8"
    )
    with wave.open(str(turn_media), "rb") as source:
        assert source.getframerate() == 16000
        assert source.getnframes() == 1600
        recreated = np.frombuffer(source.readframes(1600), dtype="<i2")
    expected = (np.arange(2000, 3600, dtype=np.int32) % 20000 - 10000).astype(
        np.int16
    )
    np.testing.assert_array_equal(recreated, expected)


def test_dry_run_has_no_model_and_reports_fixed_output_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_pairs(tmp_path, [("clip-a", [(0.0, 0.4, "e1")])])
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)

    result = asr_main(
        [
            "--audio-run-root",
            str(tmp_path),
            "--mode",
            "pilot20",
            "--dry-run",
        ]
    )

    assert result["selected_target_count"] == 1
    assert result["selected_turn_count"] == 1
    assert result["parent_quota_applied"] is False
    assert result["donor_media_used"] is False
    assert result["output_root"] == str(tmp_path / "asr_pilot20")
    assert json.loads(capsys.readouterr().out)["selected_turn_count"] == 1


def test_cli_has_no_limit_parent_quota_or_translation_options() -> None:
    destinations = {action.dest for action in _parser()._actions}

    assert "limit" not in destinations
    assert "max_clips_per_parent" not in destinations
    assert "task" not in destinations
    assert "translate" not in destinations
    assert {
        "model",
        "model_path",
        "model_fingerprint",
        "device",
        "compute_type",
        "regenerate_review",
    } <= destinations
    assert asr_output_root(Path("/tmp/audio"), mode="production") == (
        Path("/tmp/audio").resolve() / "production" / "asr"
    )


def test_local_model_identity_uses_checkpoint_content_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "whisper-large-v3-ct2"
    model.mkdir()
    (model / "model.bin").write_bytes(b"fixture checkpoint")
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_MODEL_FINGERPRINT", raising=False)
    arguments = SimpleNamespace(
        model=None,
        model_path=model,
        model_fingerprint=None,
    )

    identifier, fingerprint = _model_identity(arguments)

    assert identifier == str(model.resolve())
    assert fingerprint is not None and len(fingerprint) == 64
    arguments.model_fingerprint = "0" * 64
    with pytest.raises(ValueError, match="does not match local checkpoint"):
        _model_identity(arguments)


def test_existing_output_reuse_requires_matching_inventory_and_config(
    tmp_path: Path,
) -> None:
    pairs = _write_pairs(tmp_path, [("clip-a", [(0.0, 0.4, "e1")])])
    inventory = build_asr_inventory(pairs_root=pairs, mode="production")
    output = tmp_path / "asr"
    backend = _FakeBackend()
    expected = run_asr_transcription(
        inventory=inventory,
        output_root=output,
        backend=backend,
    )

    assert _reuse_existing(
        output_root=output,
        inventory=inventory,
        backend_config=WhisperASRConfig(
            model_identifier="fixture/whisper-large-v3",
            device="cpu",
            compute_type="int8",
        ),
    ) == expected
    with pytest.raises(ValueError, match="different model/config"):
        _reuse_existing(
            output_root=output,
            inventory=inventory,
            backend_config=WhisperASRConfig(
                model_identifier="fixture/whisper-large-v3",
                device="cpu",
                compute_type="float32",
            ),
        )
    with pytest.raises(ValueError, match="different model/config"):
        _reuse_existing(
            output_root=output,
            inventory=inventory,
            backend_config=WhisperASRConfig(
                model_identifier="fixture/whisper-large-v3",
                checkpoint_fingerprint="a" * 64,
                device="cpu",
                compute_type="int8",
            ),
        )
    changed_pairs = _write_pairs(
        tmp_path / "changed",
        [("clip-b", [(0.0, 0.4, "e1")])],
    )
    changed_inventory = build_asr_inventory(
        pairs_root=changed_pairs,
        mode="production",
    )
    with pytest.raises(ValueError, match="different inputs"):
        _reuse_existing(
            output_root=output,
            inventory=changed_inventory,
            backend_config=WhisperASRConfig(
                model_identifier="fixture/whisper-large-v3",
                device="cpu",
                compute_type="int8",
            ),
        )


def test_asr_runtime_does_not_import_dots3_or_openai_backend() -> None:
    repository = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (repository / path).read_text(encoding="utf-8")
        for path in (
            "r2v_data_v2/h3/asr_transcription.py",
            "tools/run_h3_asr_transcription.py",
        )
    )

    assert "semantic_augmentation" not in source
    assert "from openai" not in source
    assert "dots3" not in source.lower()
