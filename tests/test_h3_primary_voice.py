from __future__ import annotations

import hashlib
import json
import math
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    run_jea_primary_voice_stage,
)
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
from r2v_data_v2.h3.visual_production_source import ReadableClipIdentity


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
        destination = Path(str(request["destination"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if request.get("source_audio_path") is not None:
            source = Path(str(request["source_audio_path"]))
            start = int(request["source_start_sample"])
            end = int(request["source_end_sample"])
            if request.get("channels") == 2:
                destination.write_bytes(f"stereo:{start}:{end}".encode())
            else:
                with wave.open(str(source), "rb") as input_audio:
                    input_audio.setpos(start)
                    frames = input_audio.readframes(end - start)
                destination.write_bytes(frames)
        else:
            destination.write_bytes(Path(str(request["full_audio_path"])).read_bytes())
        return destination


class _FailingSampleSliceBackend:
    def extract_voice_reference(self, **request: object) -> Path:
        destination = Path(str(request["destination"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partial")
        raise RuntimeError("simulated primary voice export failure")


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


def _write_jea_canonical(
    pilot: Path,
    identity: ReadableClipIdentity,
) -> None:
    video = pilot / "runtime/clip-a/source.mp4"
    canonical_audio = pilot / "full_audio/clip-a.flac"
    canonical_audio.parent.mkdir(parents=True, exist_ok=True)
    canonical_audio.write_bytes(b"native-32k-stereo-fixture")
    record = CanonicalAudioClip(
        **identity.model_dump(mode="python"),
        target_video_path=str(video),
        target_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        target_full_audio_path=str(canonical_audio),
        target_full_audio_sha256=hashlib.sha256(
            canonical_audio.read_bytes()
        ).hexdigest(),
        frame_count=128000,
        target_duration_seconds=4.0,
        subject_reference_count=1,
    )
    (pilot / "canonical_clips.jsonl").write_text(
        record.model_dump_json() + "\n",
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


def test_jea_primary_voice_publishes_unicode_readable_asset_path(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "voice-quality"
    _write_pilot(pilot, [_turn()])
    output = tmp_path / "primary_voice"
    identity = ReadableClipIdentity(
        clip_uid="clip-a",
        clip_display_path="01/不惑之旅/不惑之旅 第一集/片段 01",
        media_collection_relpath="01/不惑之旅",
        media_collection_name="不惑之旅",
        episode_name="不惑之旅 第一集",
        clip_name="片段 01",
        shard_id="shard-1",
    )
    _write_jea_canonical(pilot, identity)
    backend = _SampleSliceBackend()

    summary = run_jea_primary_voice_stage(
        visual_inventory=SimpleNamespace(
            clips=[SimpleNamespace(identity=identity)]
        ),
        audio_root=pilot,
        output_root=output,
        audio_backend=backend,
    )

    expected = Path("01/不惑之旅/不惑之旅 第一集/片段 01/e1.flac")
    assert (output / expected).is_file()
    selection = json.loads(
        (output / "primary_voice_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    reference = selection["primary_voice_reference"]
    assert reference["asset"]["path"] == expected.as_posix()
    assert summary.selected_reference_rows[0]["asset_path"] == expected.as_posix()
    assert len(backend.requests) == 1
    assert backend.requests[0]["source_start_sample"] == 0
    assert backend.requests[0]["source_end_sample"] == 32000
    assert backend.requests[0]["sample_rate_hz"] == 32000
    assert backend.requests[0]["channels"] == 2
    assert Path(str(backend.requests[0]["full_audio_path"])) == Path(
        json.loads(
            (pilot / "canonical_clips.jsonl").read_text(encoding="utf-8")
        )["target_full_audio_path"]
    )
    assert reference["source_start_sample"] == 0
    assert reference["source_end_sample"] == 32000
    assert reference["quality_metadata"]["source_sample_rate_hz"] == 32000
    assert reference["quality_metadata"]["source_channels"] == 2


def test_failed_jea_primary_voice_export_leaves_no_completed_directory(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "voice-quality"
    _write_pilot(pilot, [_turn()])
    output = tmp_path / "primary_voice"
    identity = ReadableClipIdentity(
        clip_uid="clip-a",
        clip_display_path="01/创世纪 全系列/创世纪 第一集/片段 01",
        media_collection_relpath="01/创世纪 全系列",
        media_collection_name="创世纪 全系列",
        episode_name="创世纪 第一集",
        clip_name="片段 01",
        shard_id="shard-1",
    )
    _write_jea_canonical(pilot, identity)

    with pytest.raises(RuntimeError, match="simulated primary voice export failure"):
        run_jea_primary_voice_stage(
            visual_inventory=SimpleNamespace(
                clips=[SimpleNamespace(identity=identity)]
            ),
            audio_root=pilot,
            output_root=output,
            audio_backend=_FailingSampleSliceBackend(),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".primary_voice.tmp-*"))


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


def test_ffmpeg_backend_h3_voice_uses_exact_samples_and_native_stereo_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "h3-full-audio.flac"
    source.write_bytes(b"32-khz-stereo-source")
    destination = tmp_path / "h3-voice.flac"
    backend = FFmpegAudioMediaBackend(ffmpeg="fake-ffmpeg", ffprobe="fake-ffprobe")
    captured: list[str] = []

    def fake_publish(command: list[str], output: Path) -> None:
        captured.extend(command)
        output.write_bytes(b"32-khz-stereo-voice")

    monkeypatch.setattr(backend, "_publish_command", fake_publish)
    monkeypatch.setattr(
        backend,
        "probe_audio_file",
        lambda path: SimpleNamespace(
            sample_rate_hz=32000,
            channels=2,
            frame_count=64000 if path == source else 48000,
            duration_seconds=2.0 if path == source else 1.5,
            format_name="flac",
        ),
    )

    backend.extract_voice_reference(
        clip_uid="clip-a",
        entity_id="e1",
        full_audio_path=source,
        start_time=0.25,
        end_time=1.75,
        destination=destination,
        sample_rate_hz=32000,
        channels=2,
        output_format="flac",
        source_audio_path=source,
        source_start_sample=8000,
        source_end_sample=56000,
    )

    assert captured[captured.index("-i") + 1] == str(source)
    assert "-ss" not in captured
    assert "-to" not in captured
    assert captured[captured.index("-af") + 1] == (
        "atrim=start_sample=8000:end_sample=56000,asetpts=PTS-STARTPTS"
    )
    assert captured[captured.index("-ar") + 1] == "32000"
    assert captured[captured.index("-ac") + 1] == "2"
    assert destination.read_bytes() == b"32-khz-stereo-voice"
