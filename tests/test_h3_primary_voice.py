from __future__ import annotations

import hashlib
import json
import math
import wave
from array import array
from pathlib import Path

import pytest

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.pilot_schemas import (
    AssociationConfidenceDiagnostics,
    LRASDScoreDiagnostics,
    VoiceReferenceClipDiagnostics,
    VoiceReferenceTurnDiagnostics,
)
from r2v_data_v2.h3.primary_voice import (
    VoiceReferenceQualityPolicy,
    assess_voice_reference_turn,
    export_primary_voice_references,
    select_primary_voice_assessments,
)
from r2v_data_v2.h3.schemas import (
    AudioBindingEvidence,
    AudioBindingSidecar,
    AudioTrackMetadata,
    H3AudioBindingIR,
    H3TaskSpecification,
    PictureAsset,
    SemanticSubject,
)


def _turn(
    *,
    clip_uid: str = "clip-a",
    entity_id: str = "e1",
    turn_id: str = "turn_1",
    start_time: float = 0.0,
    duration: float = 1.0,
    association: float = 0.9,
    lr_mean: float = 0.7,
    lr_p10: float = 0.4,
    rms_dbfs: float = -20.0,
    clipping_ratio: float = 0.0,
    noise_available: bool = True,
    snr_db: float | None = 20.0,
) -> VoiceReferenceTurnDiagnostics:
    end_time = start_time + duration
    if noise_available:
        assert snr_db is not None
        noise_dbfs = rms_dbfs - snr_db
        noise_rms = 10 ** (noise_dbfs / 20)
        noise_samples = 3200
        noise_duration = 0.2
    else:
        noise_dbfs = None
        noise_rms = None
        noise_samples = 0
        noise_duration = 0.0
        snr_db = None
    return VoiceReferenceTurnDiagnostics(
        clip_uid=clip_uid,
        turn_id=turn_id,
        entity_id=entity_id,
        face_track_id="face_1",
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        sample_count=round(duration * 16000),
        rms_amplitude=10 ** (rms_dbfs / 20),
        rms_dbfs=rms_dbfs,
        peak_amplitude=0.2,
        peak_dbfs=20 * math.log10(0.2),
        clipping_ratio=clipping_ratio,
        local_noise_context_available=noise_available,
        local_noise_sample_count=noise_samples,
        local_noise_duration_seconds=noise_duration,
        local_noise_rms_amplitude=noise_rms,
        local_noise_rms_dbfs=noise_dbfs,
        estimated_snr_db=snr_db,
        lr_asd_raw_native_score=LRASDScoreDiagnostics(
            mean=lr_mean,
            min=min(lr_p10, lr_mean),
            p10=lr_p10,
        ),
        association_confidence=AssociationConfidenceDiagnostics(
            mean=association,
            min=association,
        ),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array("h", samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(values.tobytes())


class _SampleSliceBackend:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def extract_voice_reference(self, **request: object) -> Path:
        self.requests.append(request)
        source = Path(str(request["source_audio_path"]))
        start = int(request["source_start_sample"])
        end = int(request["source_end_sample"])
        destination = Path(str(request["destination"]))
        with wave.open(str(source), "rb") as input_audio:
            input_audio.setpos(start)
            frames = input_audio.readframes(end - start)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(frames)
        return destination


def _write_pilot(
    root: Path,
    turns: list[VoiceReferenceTurnDiagnostics],
) -> None:
    clip_uid = "clip-a"
    video = root / "runtime" / clip_uid / "source.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    audio = root / "runtime" / clip_uid / "audio.wav"
    _write_wav(audio, [index % 1000 for index in range(64000)])
    subjects = [
        SemanticSubject(
            subject_id=f"subject_{index}",
            entity_id=f"e{index}",
            reference_type="subject",
            phrase=f"person {index}",
            source_assets=[f"picture_{index}"],
        )
        for index in (1, 2)
    ]
    sidecar = AudioBindingSidecar(
        clip_uid=clip_uid,
        source_run_root=str(root),
        source_video_path=str(video),
        status="ready",
        evidence=AudioBindingEvidence(
            clip_uid=clip_uid,
            audio=AudioTrackMetadata(
                status="ready",
                source_video_path=str(video),
                full_audio_path=str(audio),
                duration_seconds=4.0,
                sample_rate_hz=16000,
                channels=1,
            ),
        ),
        h3_ir=H3AudioBindingIR(
            clip_uid=clip_uid,
            task=H3TaskSpecification(components=["reference_generation"]),
            picture_assets=[
                PictureAsset(
                    picture_id=f"picture_{index}",
                    entity_id=f"e{index}",
                    path=f"picture_{index}.png",
                )
                for index in (1, 2)
            ],
            subjects=subjects,
            audio_assets=[],
            bindings=[],
        ),
    )
    clip_dir = root / "clips" / clip_uid
    clip_dir.mkdir(parents=True)
    (clip_dir / "audio_binding.json").write_text(
        sidecar.model_dump_json(),
        encoding="utf-8",
    )
    report = VoiceReferenceClipDiagnostics(
        clip_uid=clip_uid,
        source_audio_path=audio.relative_to(root).as_posix(),
        status="ready",
        candidate_turns=turns,
    )
    (clip_dir / "voice_reference_quality.json").write_text(
        report.model_dump_json(),
        encoding="utf-8",
    )


def test_formal_policy_is_frozen_and_calibrated() -> None:
    policy = VoiceReferenceQualityPolicy()

    assert policy.thresholds_calibrated is True
    assert policy.minimum_duration_seconds == 1.0
    assert policy.minimum_association_confidence == 0.85
    assert policy.minimum_lr_asd_mean == 0.50
    assert policy.minimum_lr_asd_p10 == 0.20
    assert policy.minimum_rms_dbfs == -40.0
    assert policy.maximum_clipping_ratio == 0.0001
    assert policy.require_local_noise_context is True
    assert policy.minimum_estimated_snr_db == 10.0
    assert len(policy.fingerprint()) == 64


def test_quality_assessment_reports_every_failed_hard_gate() -> None:
    assessment = assess_voice_reference_turn(
        _turn(
            duration=0.9,
            association=0.8,
            lr_mean=0.4,
            lr_p10=0.1,
            rms_dbfs=-45.0,
            clipping_ratio=0.001,
            noise_available=False,
        )
    )

    assert assessment.status == "rejected"
    assert assessment.reason_codes == [
        "voice_duration_too_short",
        "voice_association_confidence_low",
        "voice_lr_asd_mean_low",
        "voice_lr_asd_p10_low",
        "voice_rms_too_low",
        "voice_clipping_excessive",
        "voice_noise_context_unavailable",
    ]
    assert assessment.metrics.estimated_snr_db is None
    assert assessment.thresholds_calibrated is True


def test_low_snr_rejects_and_missing_noise_is_not_guessed() -> None:
    low_snr = assess_voice_reference_turn(_turn(snr_db=7.28))
    unavailable = assess_voice_reference_turn(_turn(noise_available=False))

    assert low_snr.reason_codes == ["voice_snr_too_low"]
    assert unavailable.reason_codes == ["voice_noise_context_unavailable"]
    assert unavailable.metrics.estimated_snr_db is None


def test_primary_selection_uses_frozen_deterministic_ranking() -> None:
    lower_snr = assess_voice_reference_turn(
        _turn(turn_id="turn_1", start_time=0.0, duration=2.0, snr_db=15.0)
    )
    higher_snr = assess_voice_reference_turn(
        _turn(turn_id="turn_2", start_time=2.0, duration=1.0, snr_db=20.0)
    )

    selected = select_primary_voice_assessments(
        [lower_snr, higher_snr],
        entity_order=["e1"],
    )

    assert selected["e1"].turn_id == "turn_2"


def test_offline_export_is_read_only_and_publishes_exact_selected_samples(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "voice_quality_pilot20"
    turns = [
        _turn(turn_id="turn_1", start_time=0.0, snr_db=15.0),
        _turn(turn_id="turn_2", start_time=2.0, snr_db=20.0),
        _turn(
            entity_id="e2",
            turn_id="turn_3",
            start_time=3.0,
            snr_db=7.0,
        ),
    ]
    _write_pilot(pilot, turns)
    before = _tree_hashes(pilot)
    backend = _SampleSliceBackend()
    output = tmp_path / "primary-voice"

    summary = export_primary_voice_references(
        pilot_root=pilot,
        output_root=output,
        audio_backend=backend,
    )

    assert _tree_hashes(pilot) == before
    assert summary.candidate_turn_count == 3
    assert summary.accepted_turn_count == 2
    assert summary.rejected_turn_count == 1
    assert summary.entity_occurrence_count == 2
    assert summary.occurrences_with_primary_voice_reference == 1
    assert summary.occurrences_without_primary_voice_reference == 1
    assert summary.rejection_reason_counts == {"voice_snr_too_low": 1}
    assert summary.selected_reference_rows[0]["source_turn_id"] == "turn_2"
    assert backend.requests[0]["source_start_sample"] == 32000
    assert backend.requests[0]["source_end_sample"] == 48000
    selections = [
        json.loads(line)
        for line in (output / "primary_voice_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert selections[0]["primary_voice_reference"]["source_start_sample"] == 32000
    assert selections[0]["primary_voice_reference"]["quality_score"] == 1.0
    assert selections[0]["reason_codes"] == []
    assert selections[1]["primary_voice_reference"] is None
    assert selections[1]["reason_codes"] == [
        "no_voice_reference_passed_quality_gate",
        "voice_snr_too_low",
    ]
    second_output = tmp_path / "primary-voice-repeat"
    export_primary_voice_references(
        pilot_root=pilot,
        output_root=second_output,
        audio_backend=_SampleSliceBackend(),
    )
    for relative in (
        "voice_quality_assessments.jsonl",
        "primary_voice_references.jsonl",
        "summary.json",
        "voice_refs/clip-a/e1/voice_ref_1.flac",
    ):
        assert (output / relative).read_bytes() == (second_output / relative).read_bytes()


def test_ffmpeg_backend_slices_exact_pcm_samples_before_flac_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "audio.wav"
    _write_wav(source, list(range(100)))
    destination = tmp_path / "voice.flac"
    backend = FFmpegAudioMediaBackend(ffmpeg="fake-ffmpeg")
    captured: dict[str, object] = {}

    def fake_publish(command: list[str], output: Path) -> None:
        slice_path = Path(command[command.index("-i") + 1])
        with wave.open(str(slice_path), "rb") as input_audio:
            values = array("h")
            values.frombytes(input_audio.readframes(input_audio.getnframes()))
        captured["command"] = command
        captured["samples"] = list(values)
        output.write_bytes(b"lossless-flac")

    monkeypatch.setattr(backend, "_publish_command", fake_publish)

    backend.extract_voice_reference(
        clip_uid="clip-a",
        entity_id="e1",
        full_audio_path=source,
        start_time=10 / 16000,
        end_time=20 / 16000,
        destination=destination,
        sample_rate_hz=16000,
        output_format="flac",
        source_audio_path=source,
        source_start_sample=10,
        source_end_sample=20,
    )

    assert captured["samples"] == list(range(10, 20))
    assert captured["command"][-3:] == ["-c:a", "flac", "-y"]
    assert destination.read_bytes() == b"lossless-flac"
    assert not list(tmp_path.glob("*.pcm-slice-*.wav"))
