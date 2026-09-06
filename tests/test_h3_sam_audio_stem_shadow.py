from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from r2v_data_v2.h3.audio_backends import AudioFileProbe
from r2v_data_v2.h3.audio_schemas import FileAsset, VoiceReferenceArtifact
from r2v_data_v2.h3.diarization_binding import (
    DIARIZATION_CANONICAL_PREPROCESSING_VERSION,
    DiarizationBackendProvenance,
    DiarizationBackendSegment,
    DiarizationInventory,
    DiarizationTargetClip,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MIMO25_INVENTORY_VERSION,
    MimoClipJob,
    MimoInventory,
    MimoReferenceImage,
    MimoReferenceSelection,
    MimoSegmentEvidence,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    _inventory as _mimo_inventory,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    _job as _mimo_job,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MATERIALIZER_VERSION,
    MIMO25_POLICY_VERSION,
    MIMO25_PROMPT_VERSION,
    MIMO25_SCHEMA_VERSION,
    MimoBackendConfig,
    MimoMediaResolver,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_FACT_PROMPT_VERSION,
    FFmpegStemViewBackend,
    MimoStemFactsRecord,
    MusicInterval,
    MusicStemFacts,
    OpenAIStemFactsBackend,
    SFXContinuousLayer,
    SFXEvent,
    SFXStemFacts,
    SpeechStemFacts,
    SpeechStemSegmentFact,
    StemAwareOpenAIMimo25Backend,
    StemFactJob,
    StemFactsBackendProvenance,
    StemFactSegment,
    StemView,
    build_stem_reconcile_jobs,
    load_stem_fact_records,
    run_mimo25_stem_facts_shadow,
    stem_reconcile_auxiliary_contract,
)
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAM_AUDIO_SHADOW_ROOT_NAME,
    OfficialSAMAudioBackend,
    SAMAudioSeparationResult,
    SAMAudioStemRecord,
    StemReferenceRequest,
    build_sam_audio_stem_inventory,
    build_shadow_primary_voice_reference_requests,
    build_stem_diarization_inventory,
    export_stem_native_references,
    load_stem_shadow,
    require_shadow_output_path,
    run_sam_audio_stem_shadow,
    run_stem_diarization_shadow,
    run_stem_qwen3_asr_shadow,
    sam_audio_configuration,
    sha256_file,
    stem_shadow_root,
)


def _write_wav(
    path: Path,
    *,
    frame_count: int = 32000,
    sample_rate_hz: int = 32000,
    channels: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(b"\0\0" * channels * frame_count)


def _probe(path: Path) -> AudioFileProbe:
    with wave.open(str(path), "rb") as handle:
        return AudioFileProbe(
            sample_rate_hz=handle.getframerate(),
            channels=handle.getnchannels(),
            frame_count=handle.getnframes(),
            duration_seconds=handle.getnframes() / handle.getframerate(),
            format_name="wav",
        )


class _Media:
    def __init__(self) -> None:
        self.reference_sources: list[Path] = []

    def probe_audio_file(self, path: Path) -> AudioFileProbe:
        return _probe(path)

    def extract_voice_reference(
        self,
        *,
        clip_uid: str,
        entity_id: str,
        full_audio_path: Path,
        start_time: float,
        end_time: float,
        destination: Path,
        sample_rate_hz: int,
        output_format: str,
        channels: int = 1,
        source_audio_path: Path | None = None,
        source_start_sample: int | None = None,
        source_end_sample: int | None = None,
    ) -> Path:
        del clip_uid, entity_id, start_time, end_time, output_format
        assert source_audio_path is not None
        assert source_start_sample is not None and source_end_sample is not None
        assert source_audio_path == full_audio_path
        self.reference_sources.append(source_audio_path)
        _write_wav(
            destination,
            frame_count=source_end_sample - source_start_sample,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
        )
        return destination


class _Canonicalizer:
    def __init__(self, *, duration_delta: float = 0.0) -> None:
        self.sources: list[Path] = []
        self.duration_delta = duration_delta

    def canonicalize(
        self,
        *,
        raw_stem_path: Path,
        canonical_stem_path: Path,
    ) -> tuple[AudioFileProbe, list[str]]:
        self.sources.append(raw_stem_path)
        raw = _probe(raw_stem_path)
        frame_count = raw.frame_count + round(self.duration_delta * 32000)
        _write_wav(canonical_stem_path, frame_count=frame_count)
        return _probe(canonical_stem_path), ["fake-raw-stem-only-conversion"]


class _SAM:
    def __init__(self, configuration: object) -> None:
        self.configuration = configuration
        self.calls: list[tuple[Path, str, Path, Path]] = []

    def separate(
        self,
        *,
        clip_uid: str,
        source_audio_path: Path,
        prompt: str,
        target_path: Path,
        residual_path: Path,
    ) -> SAMAudioSeparationResult:
        del clip_uid
        self.calls.append((source_audio_path, prompt, target_path, residual_path))
        probe = _probe(source_audio_path)
        _write_wav(
            target_path,
            frame_count=probe.frame_count,
            sample_rate_hz=probe.sample_rate_hz,
            channels=probe.channels,
        )
        _write_wav(
            residual_path,
            frame_count=probe.frame_count,
            sample_rate_hz=probe.sample_rate_hz,
            channels=probe.channels,
        )
        return SAMAudioSeparationResult(
            target_path=str(target_path),
            residual_path=str(residual_path),
            verification_state="unverified",
            runtime_seconds=0.1,
            backend_metadata={"fake": True},
        )


class _SecondPassFailSAM(_SAM):
    def separate(
        self,
        *,
        clip_uid: str,
        source_audio_path: Path,
        prompt: str,
        target_path: Path,
        residual_path: Path,
    ) -> SAMAudioSeparationResult:
        if self.calls:
            self.calls.append((source_audio_path, prompt, target_path, residual_path))
            raise RuntimeError("second pass failed")
        return super().separate(
            clip_uid=clip_uid,
            source_audio_path=source_audio_path,
            prompt=prompt,
            target_path=target_path,
            residual_path=residual_path,
        )


def _canonical_fixture(tmp_path: Path) -> tuple[Path, CanonicalAudioClip, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "target.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "audio" / "full_audio" / "clip.flac"
    _write_wav(audio)
    record = CanonicalAudioClip(
        clip_uid="clip-1",
        clip_display_path="01/show/episode/clip.mp4",
        media_collection_relpath="01/show",
        media_collection_name="show",
        episode_name="episode",
        clip_name="clip.mp4",
        shard_id="01",
        target_video_path=str(video.resolve()),
        target_video_sha256=sha256_file(video),
        target_full_audio_path=str(audio.resolve()),
        target_full_audio_sha256=sha256_file(audio),
        frame_count=32000,
        target_duration_seconds=1.0,
        subject_reference_count=0,
    )
    manifest = tmp_path / "audio" / "canonical_clips.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return manifest, record, video


def _configuration(tmp_path: Path):
    code = tmp_path / "sam-code"
    model = tmp_path / "sam-model"
    code.mkdir()
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return sam_audio_configuration(
        implementation_root=code,
        model_path=model,
        model_name="facebook/sam-audio-large",
        device="cuda:0",
        reranking_candidates=1,
    )


def _run_stems(
    tmp_path: Path,
    *,
    route: str = "music_first",
    both: bool = False,
    canonicalizer: _Canonicalizer | None = None,
) -> tuple[Path, object, _SAM, _Canonicalizer]:
    manifest, _, _ = _canonical_fixture(tmp_path)
    configuration = _configuration(tmp_path)
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=configuration,
        route=route,
        run_both_routes=both,
    )
    backend = _SAM(configuration)
    active_canonicalizer = canonicalizer or _Canonicalizer()
    output = tmp_path / SAM_AUDIO_SHADOW_ROOT_NAME
    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=output,
        backend=backend,
        canonicalizer=active_canonicalizer,
        raw_probe_backend=_Media(),
    )
    return output, inventory, backend, active_canonicalizer


def test_current_production_versions_and_shadow_root_are_unchanged(tmp_path: Path) -> None:
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_unified_av_reconcile_v22"
    assert MIMO25_POLICY_VERSION == "h3_mimo25_av_authority_contract_v16"
    assert MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.13"
    assert MIMO25_MATERIALIZER_VERSION == "h3_mimo25_materializer_v16"
    assert stem_shadow_root(tmp_path) == tmp_path / SAM_AUDIO_SHADOW_ROOT_NAME


def test_shadow_output_guard_rejects_current_production_directories(
    tmp_path: Path,
) -> None:
    shadow = stem_shadow_root(tmp_path)
    assert require_shadow_output_path(
        shadow_root=shadow,
        output_path=shadow / "asr",
    ) == shadow / "asr"
    with pytest.raises(ValueError, match="shadow root"):
        require_shadow_output_path(
            shadow_root=shadow,
            output_path=tmp_path / "asr",
        )


def test_music_first_is_two_call_full_clip_cascade_with_no_recursive_sfx(
    tmp_path: Path,
) -> None:
    output, inventory, backend, _ = _run_stems(tmp_path)
    source = Path(inventory.jobs[0].source_audio_path)
    assert backend.calls[0][:2] == (source, "music soundtrack")
    assert backend.calls[1][1] == "human voices"
    assert backend.calls[1][0].name == "residual_1.raw.wav"
    assert "stems/clip-1/music_first" in backend.calls[1][0].as_posix()
    assert len(backend.calls) == 2
    _, records, summary = load_stem_shadow(output)
    assert [item.stem_type for item in records[0].stems] == ["music", "speech", "sfx"]
    assert records[0].route == "music_first"
    assert summary.model_call_count == 2
    assert records[0].calls[1].input_sha256 == records[0].calls[0].residual_sha256


def test_voice_first_mapping_and_route_provenance(tmp_path: Path) -> None:
    output, _, backend, _ = _run_stems(tmp_path, route="voice_first")
    assert [item[1] for item in backend.calls] == ["human voices", "music soundtrack"]
    _, records, _ = load_stem_shadow(output)
    assert records[0].route == "voice_first"
    assert records[0].stem("speech").raw_stem_path.endswith("speech.raw.wav")
    assert records[0].stem("music").raw_stem_path.endswith("music.raw.wav")


def test_both_routes_are_explicit_and_bounded(tmp_path: Path) -> None:
    output, _, backend, _ = _run_stems(tmp_path, both=True)
    _, records, summary = load_stem_shadow(output)
    assert [item.route for item in records] == ["music_first", "voice_first"]
    assert len(backend.calls) == 4
    assert summary.route_counts == {"music_first": 1, "voice_first": 1}


def test_published_paths_are_rebased_from_atomic_temporary_root(tmp_path: Path) -> None:
    output, _, _, _ = _run_stems(tmp_path)
    _, records, _ = load_stem_shadow(output)
    payload = records[0].model_dump_json()
    assert ".tmp-" not in payload
    assert all(Path(item.canonical_stem_path).is_file() for item in records[0].stems)


def test_canonical_stems_derive_only_from_raw_stems(tmp_path: Path) -> None:
    _, inventory, _, canonicalizer = _run_stems(tmp_path)
    source = Path(inventory.jobs[0].source_audio_path)
    assert len(canonicalizer.sources) == 3
    assert source not in canonicalizer.sources
    assert all("/stems/clip-1/music_first/" in item.as_posix() for item in canonicalizer.sources)


def test_duration_mismatch_fails_closed_per_route(tmp_path: Path) -> None:
    output, _, _, _ = _run_stems(
        tmp_path,
        canonicalizer=_Canonicalizer(duration_delta=0.5),
    )
    _, records, _ = load_stem_shadow(output)
    assert records[0].separation_state == "failure"
    assert records[0].stems == []
    assert "duration differs" in (records[0].failure_reason or "")


def test_second_pass_failure_retains_first_call_provenance_and_attempt_count(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _canonical_fixture(tmp_path)
    configuration = _configuration(tmp_path)
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=configuration,
    )
    output = tmp_path / SAM_AUDIO_SHADOW_ROOT_NAME
    summary = run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=output,
        backend=_SecondPassFailSAM(configuration),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=_Media(),
    )
    _, records, _ = load_stem_shadow(output)
    assert summary.model_call_count == 2
    assert records[0].separation_state == "failure"
    assert records[0].model_call_count == 2
    assert len(records[0].calls) == 1
    assert records[0].calls[0].prompt == "music soundtrack"
    assert (output / "stems/clip-1/music_first/separation.json").is_file()


def test_case_manifest_preserves_exact_selection_order(tmp_path: Path) -> None:
    manifest, record, _ = _canonical_fixture(tmp_path)
    second = record.model_copy(
        update={
            "clip_uid": "clip-2",
            "clip_display_path": "01/show/episode/clip2.mp4",
            "clip_name": "clip2.mp4",
        }
    )
    manifest.write_text(
        second.model_dump_json() + "\n" + record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "schema_version": "r2v.h3.mimo25_case_manifest.1",
                "clip_uids": ["clip-1", "clip-2"],
            }
        ),
        encoding="utf-8",
    )
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=_configuration(tmp_path),
        case_manifest_path=case,
    )
    assert [item.clip_uid for item in inventory.jobs] == ["clip-1", "clip-2"]
    assert inventory.selection_mode == "explicit_case_manifest"


def test_official_backend_is_local_only_and_uses_official_two_output_api() -> None:
    source = Path(OfficialSAMAudioBackend.__module__.replace(".", "/"))
    del source
    code = Path("r2v_data_v2/h3/sam_audio_stem_shadow.py").read_text(encoding="utf-8")
    assert 'local_files_only": True' in code
    assert "model.separate(" in code
    assert "result.target" in code and "result.residual" in code


def test_stem_native_reference_cuts_the_stem_not_original_mix(tmp_path: Path) -> None:
    output, _, _, _ = _run_stems(tmp_path)
    _, records, _ = load_stem_shadow(output)
    media = _Media()
    references = export_stem_native_references(
        records=records,
        requests=[
            StemReferenceRequest(
                reference_id="voice-e1",
                clip_uid="clip-1",
                route="music_first",
                stem_type="speech",
                start_time=0.1,
                end_time=0.5,
            )
        ],
        output_root=output / "references",
        media_backend=media,
    )
    speech = records[0].stem("speech")
    assert media.reference_sources == [Path(speech.canonical_stem_path)]
    assert references[0].source_stem_sha256 == speech.canonical_stem_sha256
    assert references[0].start_sample == 3200
    assert references[0].end_sample == 16000


def _primary_voice_fixture(root: Path) -> None:
    voice = root / "clips" / "clip-1" / "e1.flac"
    _write_wav(voice)
    policy_fingerprint = "f" * 64
    artifact = VoiceReferenceArtifact(
        voice_reference_id="voice_ref_1",
        entity_occurrence_id="clip-1/e1",
        source_turn_id="turn_1",
        source_start=0.0,
        source_end=1.0,
        source_start_sample=0,
        source_end_sample=32000,
        asset=FileAsset(
            path="clips/clip-1/e1.flac",
            sha256=sha256_file(voice),
            byte_size=voice.stat().st_size,
            media_type="audio/flac",
        ),
        quality_score=1.0,
        quality_metadata={
            "assessment": {
                "status": "accepted",
                "turn_id": "turn_1",
                "entity_occurrence_id": "clip-1/e1",
            },
            "policy_version": "voice_reference_quality_v1",
            "policy_fingerprint": policy_fingerprint,
            "thresholds_calibrated": True,
        },
    )
    selection = PrimaryVoiceReferenceSelection(
        clip_uid="clip-1",
        entity_id="e1",
        entity_occurrence_id="clip-1/e1",
        primary_voice_reference=artifact,
        accepted_turn_ids=["turn_1"],
        candidate_turn_ids=["turn_1"],
        policy_version="voice_reference_quality_v1",
        policy_fingerprint=policy_fingerprint,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "primary_voice_references.jsonl").write_text(
        selection.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        PrimaryVoiceReferenceExportSummary(
            policy_version="voice_reference_quality_v1",
            policy_fingerprint=policy_fingerprint,
            candidate_turn_count=1,
            accepted_turn_count=1,
            rejected_turn_count=0,
            entity_occurrence_count=1,
            occurrences_with_primary_voice_reference=1,
            occurrences_without_primary_voice_reference=0,
            rejection_reason_counts={},
            selected_reference_rows=[],
        ).model_dump_json(),
        encoding="utf-8",
    )


def test_shadow_primary_voice_reuses_quality_selected_interval_and_cuts_speech_stem(
    tmp_path: Path,
) -> None:
    output, _, _, _ = _run_stems(tmp_path)
    _, records, _ = load_stem_shadow(output)
    primary = tmp_path / "primary_voice"
    _primary_voice_fixture(primary)
    requests = build_shadow_primary_voice_reference_requests(
        primary_voice_root=primary,
        records=records,
        route="music_first",
        selected_clip_uids=["clip-1"],
    )
    assert len(requests) == 1
    assert requests[0].reference_kind == "shadow_primary_voice"
    assert requests[0].source_turn_id == "turn_1"
    assert requests[0].source_quality_policy_version == "voice_reference_quality_v1"

    media = _Media()
    references = export_stem_native_references(
        records=records,
        requests=requests,
        output_root=output / "references",
        media_backend=media,
    )
    speech = records[0].stem("speech")
    assert media.reference_sources == [Path(speech.canonical_stem_path)]
    assert references[0].entity_occurrence_id == "clip-1/e1"
    assert references[0].source_turn_id == "turn_1"
    assert references[0].start_sample == 0
    assert references[0].end_sample == 32000


def _production_diarization_inventory(
    tmp_path: Path,
    canonical: CanonicalAudioClip,
) -> DiarizationInventory:
    target = DiarizationTargetClip(
        target_clip_uid=canonical.clip_uid,
        target_video_path=canonical.target_video_path,
        source_audio_path=canonical.target_full_audio_path,
        source_audio_sha256=canonical.target_full_audio_sha256,
        source_sample_rate_hz=32000,
        source_channels=2,
        source_frame_count=canonical.frame_count,
        visual_references=[],
    )
    visual_inventory = tmp_path / "visual-samples.jsonl"
    visual_inventory.write_text("{}\n", encoding="utf-8")
    canonical_manifest = tmp_path / "audio/canonical_clips.jsonl"
    visual_hash = sha256_file(visual_inventory)
    canonical_hash = sha256_file(canonical_manifest)
    fingerprint = _diarization_inventory_fingerprint(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=[target],
        source_inventory_kind="canonical_audio_manifest",
        source_visual_inventory_sha256=visual_hash,
        source_canonical_audio_manifest_sha256=canonical_hash,
    )
    return DiarizationInventory(
        mode="production",
        source_inventory_kind="canonical_audio_manifest",
        source_visual_production_root=str(tmp_path / "visual"),
        source_visual_inventory_path=str(visual_inventory),
        source_visual_inventory_sha256=visual_hash,
        source_canonical_audio_manifest_path=str(canonical_manifest),
        source_canonical_audio_manifest_sha256=canonical_hash,
        inventory_fingerprint=fingerprint,
        source_target_count=1,
        selected_target_count=1,
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=[target],
    )


class _Diarization:
    provenance = DiarizationBackendProvenance(
        backend="fake_diarizen",
        model_identifier="fake",
        model_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
        input_profile="canonical_32k_stereo",
        input_preprocessing=DIARIZATION_CANONICAL_PREPROCESSING_VERSION,
        source_sample_rate_hz=32000,
        source_channels=2,
    )

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def diarize(self, *, clip_uid: str, audio_path: Path):
        del clip_uid
        self.paths.append(audio_path)
        return [DiarizationBackendSegment(start_time=0.0, end_time=0.5, speaker_label="A")]


def _run_diarization(tmp_path: Path) -> tuple[Path, Path, SAMAudioStemRecord]:
    stem_root, inventory, _, _ = _run_stems(tmp_path)
    _, stem_records, _ = load_stem_shadow(stem_root)
    canonical = CanonicalAudioClip.model_validate_json(
        Path(inventory.source_canonical_audio_manifest_path)
        .read_text(encoding="utf-8")
        .strip()
    )
    production_root = tmp_path / "production-diarization"
    production_root.mkdir()
    production_inventory = _production_diarization_inventory(tmp_path, canonical)
    (production_root / "inventory.json").write_text(
        production_inventory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    backend = _Diarization()
    output = stem_root / "diarization"
    run_stem_diarization_shadow(
        stem_root=stem_root,
        production_diarization_root=production_root,
        backend=backend,
        route="music_first",
        output_root=output,
    )
    assert backend.paths == [Path(stem_records[0].stem("speech").canonical_stem_path)]
    return stem_root, output, stem_records[0]


def test_stem_diarization_uses_speech_stem_and_preserves_production_inventory(
    tmp_path: Path,
) -> None:
    manifest, canonical, _ = _canonical_fixture(tmp_path)
    del manifest
    production = _production_diarization_inventory(tmp_path, canonical)
    production_bytes = production.model_dump_json().encode()
    stem_root, inventory, _, _ = _run_stems(tmp_path / "run")
    _, records, _ = load_stem_shadow(stem_root)
    shadow = build_stem_diarization_inventory(
        stem_inventory=inventory,
        stem_records=records,
        production_diarization_inventory=production,
        route="music_first",
    )
    assert shadow.targets[0].source_audio_path == records[0].stem("speech").canonical_stem_path
    assert production.targets[0].source_audio_path == canonical.target_full_audio_path
    assert production.model_dump_json().encode() == production_bytes


class _Qwen:
    configuration = Qwen3ASRConfiguration(
        local_model_path="/local/fake-qwen3-asr",
        device="cpu",
    )

    def transcribe(self, *, waveform: np.ndarray, sample_rate_hz: int):
        assert waveform.ndim == 1 and sample_rate_hz == 16000
        return "exact transcript", "English"


def test_stem_diarization_and_asr_use_speech_stem_without_touching_production(
    tmp_path: Path,
) -> None:
    stem_root, diarization, stem_record = _run_diarization(tmp_path)
    observed: list[Path] = []

    def loader(path: Path, start: float, end: float):
        del start, end
        observed.append(path)
        return np.zeros(8000, dtype=np.float32), 16000

    summary, provenance = run_stem_qwen3_asr_shadow(
        stem_diarization_root=diarization,
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=stem_root / "asr",
        segment_audio_loader=loader,
    )
    assert observed == [Path(stem_record.stem("speech").canonical_stem_path)]
    assert summary.transcribed_count == 1
    assert provenance.asr_source_kind == "sam_audio_speech_stem_segments"
    row = json.loads((stem_root / "asr/segments.jsonl").read_text(encoding="utf-8"))
    assert row["source_audio_path"] == stem_record.stem("speech").canonical_stem_path


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (
            {
                "clip_duration_seconds": 2.0,
                "music_status": "present",
                "intervals": [
                    {
                        "start_time": 0.1,
                        "end_time": 1.5,
                        "description": "low sustained score",
                        "confidence": "high",
                    }
                ],
            },
            True,
        ),
        ({"clip_duration_seconds": 2.0, "music_status": "present", "intervals": []}, False),
        (
            {
                "clip_duration_seconds": 2.0,
                "music_status": "absent",
                "intervals": [
                    {
                        "start_time": 0.1,
                        "end_time": 1.0,
                        "description": "invented",
                        "confidence": "low",
                    }
                ],
            },
            False,
        ),
    ],
)
def test_music_status_interval_contract(value: dict[str, object], accepted: bool) -> None:
    if accepted:
        MusicStemFacts.model_validate(value)
    else:
        with pytest.raises(ValidationError):
            MusicStemFacts.model_validate(value)


def test_music_intervals_must_be_sorted_non_overlapping_and_bounded() -> None:
    with pytest.raises(ValidationError):
        MusicStemFacts(
            clip_duration_seconds=2.0,
            music_status="present",
            intervals=[
                MusicInterval(start_time=1.0, end_time=1.5, description="b", confidence="high"),
                MusicInterval(start_time=0.0, end_time=0.5, description="a", confidence="high"),
            ],
        )
    with pytest.raises(ValidationError):
        MusicStemFacts(
            clip_duration_seconds=1.0,
            music_status="present",
            intervals=[
                MusicInterval(start_time=0.0, end_time=1.1, description="a", confidence="high")
            ],
        )


def test_sfx_layers_and_events_are_ordered_and_bounded() -> None:
    valid = SFXStemFacts(
        clip_duration_seconds=2.0,
        continuous_layers=[
            SFXContinuousLayer(
                start_time=0.0,
                end_time=2.0,
                description="quiet room tone",
                confidence="medium",
            )
        ],
        events=[
            SFXEvent(
                start_time=0.5,
                end_time=0.6,
                category="physical",
                description="door click",
                confidence="high",
            )
        ],
    )
    assert valid.events[0].category == "physical"
    with pytest.raises(ValidationError):
        SFXStemFacts(
            clip_duration_seconds=1.0,
            continuous_layers=[],
            events=[
                SFXEvent(
                    start_time=0.0,
                    end_time=1.1,
                    category="other",
                    description="bad",
                    confidence="low",
                )
            ],
        )


class _ViewBackend:
    def create(
        self,
        *,
        clip_uid: str,
        route: str,
        stem_type: str,
        original_video_path: Path,
        source_stem_path: Path,
        destination: Path,
    ) -> StemView:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"proxy")
        return StemView(
            clip_uid=clip_uid,
            route=route,
            stem_type=stem_type,
            original_video_path=str(original_video_path),
            original_video_sha256=sha256_file(original_video_path),
            source_stem_path=str(source_stem_path),
            source_stem_sha256=sha256_file(source_stem_path),
            view_path=str(destination),
            view_sha256=sha256_file(destination),
        )


class _FactsBackend:
    provenance = StemFactsBackendProvenance(
        backend="fake",
        model="fake",
        base_url="local",
        media_mode="base64",
        configuration_fingerprint="d" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, *, job: StemFactJob, stem_type: str, view: StemView):
        del view
        self.calls.append(stem_type)
        if stem_type == "speech":
            return SpeechStemFacts(
                clip_duration_seconds=job.clip_duration_seconds,
                segment_facts=[
                    SpeechStemSegmentFact(
                        segment_id=item.segment_id,
                        vocal_composition="one voice",
                        delivery_style="calm",
                        confidence="high",
                    )
                    for item in job.segments
                ],
            )
        if stem_type == "music":
            return MusicStemFacts(
                clip_duration_seconds=job.clip_duration_seconds,
                music_status="present",
                intervals=[
                    MusicInterval(
                        start_time=0.1,
                        end_time=0.4,
                        description="a low score fades in",
                        confidence="medium",
                    )
                ],
            )
        return SFXStemFacts(
            clip_duration_seconds=job.clip_duration_seconds,
            continuous_layers=[],
            events=[],
        )


def _run_facts(tmp_path: Path) -> tuple[Path, MimoStemFactsRecord]:
    stem_root, _, _ = _run_diarization(tmp_path)
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=stem_root / "diarization",
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=stem_root / "asr",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
    )
    backend = _FactsBackend()
    summary = run_mimo25_stem_facts_shadow(
        stem_root=stem_root,
        route="music_first",
        backend=backend,
        view_backend=_ViewBackend(),
        output_root=stem_root / "mimo_stem_facts",
    )
    records = load_stem_fact_records(stem_root / "mimo_stem_facts")
    assert backend.calls == ["speech", "music", "sfx"]
    assert summary.model_call_count == 3
    return stem_root, records[0]


def test_stem_facts_use_full_timeline_and_keep_music_timing(tmp_path: Path) -> None:
    stem_root, record = _run_facts(tmp_path)
    assert record.music.intervals[0].start_time == 0.1
    assert record.music.intervals[0].end_time == 0.4
    assert all(Path(item.view_path).is_file() for item in record.stem_views)
    assert (stem_root / "mimo_stem_facts/records.jsonl").is_file()


def test_stem_reconcile_contract_keeps_original_av_above_stem_facts(tmp_path: Path) -> None:
    _, record = _run_facts(tmp_path)
    contract = stem_reconcile_auxiliary_contract(record)
    assert contract["authority_order"][0] == "original_target_full_av"
    assert "cannot override contradictory original AV" in " ".join(contract["rules"])
    assert contract["stem_facts"]["music"]["intervals"][0]["start_time"] == 0.1


class _Completions:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.raw))]
        )


def test_stem_fact_openai_transport_uses_derived_video_with_embedded_stem_only(
    tmp_path: Path,
) -> None:
    video = tmp_path / "music.mp4"
    stem = tmp_path / "music.wav"
    video.write_bytes(b"video")
    _write_wav(stem)
    response = MusicStemFacts(
        clip_duration_seconds=1.0,
        music_status="absent",
        intervals=[],
    ).model_dump_json()
    completions = _Completions(response)
    backend = OpenAIStemFactsBackend(
        model="mimo-v2.5",
        base_url="http://localhost/v1",
        api_key="test",
        media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    values = {
        "clip_uid": "clip-1",
        "route": "music_first",
        "target_video_path": str(video),
        "target_video_sha256": sha256_file(video),
        "clip_duration_seconds": 1.0,
        "speech_stem_path": str(stem),
        "speech_stem_sha256": sha256_file(stem),
        "music_stem_path": str(stem),
        "music_stem_sha256": sha256_file(stem),
        "sfx_stem_path": str(stem),
        "sfx_stem_sha256": sha256_file(stem),
        "segments": [],
    }
    job = StemFactJob(
        **values,
        job_fingerprint=__import__("hashlib").sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    view = StemView(
        clip_uid="clip-1",
        route="music_first",
        stem_type="music",
        original_video_path=str(video),
        original_video_sha256=sha256_file(video),
        source_stem_path=str(stem),
        source_stem_sha256=sha256_file(stem),
        view_path=str(video),
        view_sha256=sha256_file(video),
    )
    backend.extract(job=job, stem_type="music", view=view)
    request = completions.requests[0]
    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["video_url", "text"]
    assert request["extra_body"]["use_audio_in_video"] is True
    assert request["response_format"]["type"] == "json_schema"
    assert MIMO25_STEM_FACT_PROMPT_VERSION in backend.provenance.prompt_version
    assert "asr_text" not in content[1]["text"]
    assert "entity_id" not in content[1]["text"]


def test_speech_stem_fact_prompt_includes_authoritative_shadow_asr_without_output_field(
    tmp_path: Path,
) -> None:
    video = tmp_path / "speech.mp4"
    stem = tmp_path / "speech.wav"
    video.write_bytes(b"video")
    _write_wav(stem)
    segment = StemFactSegment(
        segment_id="segment_0001",
        speaker_cluster_id="speaker_0",
        start_time=0.0,
        end_time=1.0,
        asr_status="transcribed",
        asr_text="authoritative words",
        asr_language="English",
    )
    values = {
        "clip_uid": "clip-1",
        "route": "music_first",
        "target_video_path": str(video),
        "target_video_sha256": sha256_file(video),
        "clip_duration_seconds": 1.0,
        "speech_stem_path": str(stem),
        "speech_stem_sha256": sha256_file(stem),
        "music_stem_path": str(stem),
        "music_stem_sha256": sha256_file(stem),
        "sfx_stem_path": str(stem),
        "sfx_stem_sha256": sha256_file(stem),
        "segments": [segment.model_dump(mode="json")],
    }
    job = StemFactJob(
        **values,
        job_fingerprint=__import__("hashlib").sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    prompt = OpenAIStemFactsBackend._prompt(job, "speech")
    assert "authoritative words" in prompt
    assert "do not rewrite" in prompt
    schema = json.dumps(SpeechStemFacts.model_json_schema(), sort_keys=True)
    assert "asr_text" not in schema


def _mimo_job_fixture(tmp_path: Path) -> MimoClipJob:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "target.mp4"
    audio = tmp_path / "mix.flac"
    image = tmp_path / "reference.png"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    image.write_bytes(b"image")
    reference = MimoReferenceImage(
        image_index=1,
        picture_label="<Picture 1>",
        source_image_index=1,
        source_image_id="image_1",
        source_image_label="<Image 1>",
        kind="subject",
        entity_id="e1",
        image_artifact_path=str(image),
        image_sha256="1" * 64,
    )
    values = {
        "clip_uid": "clip-1",
        "r2v_instruction": "Recreate the observed target clip.",
        "target_video_path": str(video),
        "target_video_sha256": "2" * 64,
        "target_full_audio_path": str(audio),
        "target_full_audio_sha256": "3" * 64,
        "target_duration_seconds": 1.0,
        "reference_selection": MimoReferenceSelection(
            original_picture_count=1,
            selected_picture_count=1,
            selected_source_image_indexes=[1],
            selected_source_image_ids=["image_1"],
            dropped_references=[],
        ).model_dump(mode="json"),
        "reference_images": [reference.model_dump(mode="json")],
        "reference_subjects": [
            RecaptionSubjectContract(
                subject_index=1,
                subject_label="<Subject 1>",
                kind="entity",
                entity_id="e1",
                source_picture_labels=["<Picture 1>"],
            ).model_dump(mode="json")
        ],
        "segments": [
            MimoSegmentEvidence(
                segment_id="old-segment",
                start_time=0.0,
                end_time=1.0,
                source_start_sample=0,
                source_end_sample=32000,
                source_sample_rate_hz=32000,
                source_speaker_cluster_id="old-speaker",
                identity_scope="unresolved",
                direct_anchor_seconds=0.0,
                cluster_binding_status="unbound",
                overlapping_visible_entities=[],
                direct_support_seconds_by_entity={},
                competing_visible_speaker_evidence=[],
                asr_status="empty",
            ).model_dump(mode="json")
        ],
        "source_h3_sample_ids": ["clip-1/in_pair"],
    }
    return _mimo_job(values)


def _mimo_inventory_fixture(tmp_path: Path) -> MimoInventory:
    job = _mimo_job_fixture(tmp_path)
    values = {
        "schema_version": MIMO25_INVENTORY_VERSION,
        "inventory_scope": "explicit_case_subset",
        "canonical_wide_coverage": False,
        "source_visual_inventory_sha256": "1" * 64,
        "source_canonical_audio_manifest_sha256": "2" * 64,
        "source_diarization_inventory_sha256": "3" * 64,
        "source_diarization_raw_segments_sha256": "4" * 64,
        "source_diarization_bound_segments_sha256": "5" * 64,
        "source_qwen3_asr_segments_sha256": "6" * 64,
        "source_binding_audit_segments_sha256": "7" * 64,
        "source_h3_samples_sha256": "8" * 64,
        "clip_count": 1,
        "jobs": [job.model_dump(mode="json")],
    }
    return _mimo_inventory(values)


def test_stem_reconcile_jobs_keep_original_av_and_use_shadow_speech_timeline(
    tmp_path: Path,
) -> None:
    stem_root, diarization, _ = _run_diarization(tmp_path)
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=diarization,
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=stem_root / "asr",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(8000, dtype=np.float32),
            16000,
        ),
    )
    base = _mimo_inventory_fixture(tmp_path / "base")
    jobs = build_stem_reconcile_jobs(
        base_inventory=base,
        stem_diarization_root=diarization,
        stem_asr_root=stem_root / "asr",
    )
    assert jobs[0].target_video_path == base.jobs[0].target_video_path
    assert jobs[0].target_full_audio_path == base.jobs[0].target_full_audio_path
    assert [item.segment_id for item in jobs[0].segments] == ["segment_0001"]
    assert jobs[0].segments[0].asr_text == "exact transcript"


def test_stem_aware_mimo_request_keeps_original_video_as_model_media(tmp_path: Path) -> None:
    _, facts = _run_facts(tmp_path / "facts")
    job = _mimo_job_fixture(tmp_path / "mimo")
    resolver = MimoMediaResolver(mode="base64", media_root=tmp_path)
    backend = StemAwareOpenAIMimo25Backend(
        MimoBackendConfig(
            media_resolver=resolver,
            api_key="test",
            transport="sglang",
            base_url="http://localhost/v1",
        ),
        stem_facts_by_clip={"clip-1": facts},
        client=SimpleNamespace(),
    )
    media = backend._media_content(job, include_audio_fallback=False)
    video_rows = [item for item in media if item["type"] == "video_url"]
    assert len(video_rows) == 1
    assert not any(item["type"] in {"audio_url", "input_audio"} for item in media)
    prompt = backend._prompt(job)
    assert "original_target_full_av" in prompt
    assert "cannot override contradictory original AV" in prompt


def test_shadow_paths_do_not_write_current_production_directories(tmp_path: Path) -> None:
    output, _, _, _ = _run_stems(tmp_path)
    assert output.name == SAM_AUDIO_SHADOW_ROOT_NAME
    for name in ("diarization", "asr", "h3", "mimo25_av_reconcile_v5"):
        assert not (tmp_path / name).exists()


def test_speech_fact_schema_contains_no_asr_text_or_entity_identity() -> None:
    schema = SpeechStemFacts.model_json_schema()
    payload = json.dumps(schema, sort_keys=True)
    assert "asr_text" not in payload
    assert "entity_id" not in payload


def test_no_model_or_network_dependencies_are_imported_by_fake_pipeline() -> None:
    source = Path("r2v_data_v2/h3/sam_audio_stem_shadow.py").read_text(
        encoding="utf-8"
    )
    assert "import requests" not in source
    assert "from_pretrained(self.configuration.model_path" in source
    assert "local_files_only" in source
    assert not any(Path(".").glob("**/*.ckpt"))


def test_ffmpeg_view_contract_keeps_video_stream_copy_and_marks_audio_proxy() -> None:
    source = Path("r2v_data_v2/h3/mimo25_stem_shadow.py").read_text(encoding="utf-8")
    assert '"-c:v",\n                    "copy"' in source
    assert "aac_320k_annotation_proxy" in source
    assert FFmpegStemViewBackend.__name__ == "FFmpegStemViewBackend"
