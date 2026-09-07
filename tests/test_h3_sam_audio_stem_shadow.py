from __future__ import annotations

import json
import os
import shutil
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    RawDiarizationSegment,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MIMO25_INVENTORY_VERSION,
    MimoCaseManifest,
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
    MimoBackendFailure,
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
    StemFactsBackendFailure,
    StemFactsBackendProvenance,
    StemFactsBackendResult,
    StemFactSegment,
    StemView,
    build_stem_reconcile_jobs,
    load_stem_fact_records,
    run_mimo25_stem_facts_shadow,
    run_mimo25_stem_reconcile_shadow,
    stem_reconcile_auxiliary_contract,
    validate_stem_facts_lineage,
)
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration
from r2v_data_v2.h3.qwen38_h3_recaption import RecaptionSubjectContract
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAM_AUDIO_SHADOW_ROOT_NAME,
    STEM_ALIGNMENT_TOLERANCE_SECONDS,
    STEM_CANONICAL_CHANNEL_POLICY,
    STEM_TIMELINE_POLICY,
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
    selected_stem_records,
    separation_skips,
    sha256_file,
    stem_separation_root,
    stem_shadow_root,
    validate_stem_asr_lineage,
    validate_stem_diarization_lineage,
)
from tools.run_h3_mimo25_stem_reconcile_shadow import (
    _parser as _stem_reconcile_parser,
)
from tools.run_h3_mimo25_stem_reconcile_shadow import _validate_stage_closure
from tools.run_h3_sam_audio_stem_shadow import _parser as _sam_stem_parser


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
            channels=1,
        )
        _write_wav(
            residual_path,
            frame_count=probe.frame_count,
            sample_rate_hz=probe.sample_rate_hz,
            channels=1,
        )
        return SAMAudioSeparationResult(
            target_path=str(target_path),
            residual_path=str(residual_path),
            verification_state="unverified",
            runtime_seconds=0.1,
            backend_metadata={"fake": True},
        )


class _DurationDriftSAM(_SAM):
    def __init__(self, configuration: object, duration_delta: float) -> None:
        super().__init__(configuration)
        self.duration_delta = duration_delta

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
        first_call = not self.calls
        self.calls.append((source_audio_path, prompt, target_path, residual_path))
        probe = _probe(source_audio_path)
        frame_count = probe.frame_count + (
            round(self.duration_delta * probe.sample_rate_hz) if first_call else 0
        )
        _write_wav(
            target_path,
            frame_count=frame_count,
            sample_rate_hz=probe.sample_rate_hz,
            channels=1,
        )
        _write_wav(
            residual_path,
            frame_count=frame_count,
            sample_rate_hz=probe.sample_rate_hz,
            channels=1,
        )
        return SAMAudioSeparationResult(
            target_path=str(target_path),
            residual_path=str(residual_path),
            verification_state="unverified",
            runtime_seconds=0.1,
            backend_metadata={"fake": True, "duration_delta": self.duration_delta},
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


class _OneClipFailSAM(_SAM):
    def __init__(self, configuration: object, failed_clip_uid: str) -> None:
        super().__init__(configuration)
        self.failed_clip_uid = failed_clip_uid

    def separate(
        self,
        *,
        clip_uid: str,
        source_audio_path: Path,
        prompt: str,
        target_path: Path,
        residual_path: Path,
    ) -> SAMAudioSeparationResult:
        if clip_uid == self.failed_clip_uid:
            self.calls.append((source_audio_path, prompt, target_path, residual_path))
            raise RuntimeError("synthetic per-clip SAM failure")
        return super().separate(
            clip_uid=clip_uid,
            source_audio_path=source_audio_path,
            prompt=prompt,
            target_path=target_path,
            residual_path=residual_path,
        )


class _VariantSAM(_SAM):
    def separate(self, **kwargs: object) -> SAMAudioSeparationResult:
        result = super().separate(**kwargs)
        return result.model_copy(update={"backend_metadata": {"fake": "variant-b"}})


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


def _multi_canonical_fixture(
    tmp_path: Path,
    *,
    clip_uids: list[str],
) -> tuple[Path, list[CanonicalAudioClip], Path]:
    manifest, source, _ = _canonical_fixture(tmp_path)
    records = [
        source.model_copy(
            update={
                "clip_uid": clip_uid,
                "clip_display_path": f"01/show/episode/{clip_uid}.mp4",
                "clip_name": f"{clip_uid}.mp4",
            }
        )
        for clip_uid in sorted(clip_uids)
    ]
    manifest.write_text(
        "".join(item.model_dump_json() + "\n" for item in records),
        encoding="utf-8",
    )
    case = tmp_path / "case.json"
    case.write_text(
        json.dumps(
            {
                "schema_version": "r2v.h3.mimo25_case_manifest.1",
                "clip_uids": clip_uids,
            }
        ),
        encoding="utf-8",
    )
    return manifest, records, case


def _configuration(tmp_path: Path):
    code = tmp_path / "sam-code"
    model = tmp_path / "sam-model"
    t5 = tmp_path / "t5-base"
    for relative in (
        "sam_audio/__init__.py",
        "sam_audio/processor.py",
        "sam_audio/model/base.py",
        "sam_audio/model/config.py",
        "sam_audio/model/model.py",
    ):
        path = code / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "facebook/sam-audio-large",
                "text_encoder": {"name": "google/t5-base", "dim": 768},
            }
        ),
        encoding="utf-8",
    )
    (model / "checkpoint.pt").write_bytes(b"sam checkpoint")
    t5.mkdir(parents=True, exist_ok=True)
    (t5 / "config.json").write_text("{}", encoding="utf-8")
    (t5 / "tokenizer.json").write_text("{}", encoding="utf-8")
    (t5 / "model.safetensors").write_bytes(b"t5 checkpoint")
    return sam_audio_configuration(
        implementation_root=code,
        model_path=model,
        model_name="facebook/sam-audio-large",
        t5_base_path=t5,
        device="cuda:0",
        reranking_candidates=1,
    )


def _run_stems(
    tmp_path: Path,
    *,
    route: str = "music_first",
    both: bool = False,
    canonicalizer: _Canonicalizer | None = None,
    raw_duration_delta: float | None = None,
    shadow_run_id: str | None = None,
) -> tuple[Path, object, _SAM, _Canonicalizer]:
    manifest, _, _ = _canonical_fixture(tmp_path)
    configuration = _configuration(tmp_path)
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=configuration,
        route=route,
        run_both_routes=both,
    )
    backend = (
        _SAM(configuration)
        if raw_duration_delta is None
        else _DurationDriftSAM(configuration, raw_duration_delta)
    )
    active_canonicalizer = canonicalizer or _Canonicalizer()
    output = stem_separation_root(tmp_path, shadow_run_id)
    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=output,
        backend=backend,
        canonicalizer=active_canonicalizer,
        raw_probe_backend=_Media(),
    )
    return output, inventory, backend, active_canonicalizer


def test_current_shadow_contract_versions_and_root(tmp_path: Path) -> None:
    assert MIMO25_PROMPT_VERSION == "h3_mimo25_unified_av_reconcile_v23"
    assert MIMO25_POLICY_VERSION == "h3_mimo25_av_authority_contract_v17"
    assert MIMO25_SCHEMA_VERSION == "r2v.h3.mimo25_av_annotation.14"
    assert MIMO25_MATERIALIZER_VERSION == "h3_mimo25_materializer_v17"
    assert stem_shadow_root(tmp_path) == tmp_path / SAM_AUDIO_SHADOW_ROOT_NAME
    assert stem_separation_root(tmp_path) == (
        tmp_path / SAM_AUDIO_SHADOW_ROOT_NAME / "separation"
    )


@pytest.mark.parametrize(
    "run_id",
    ["", " ", ".", "..", "/tmp/run", "../run", "a/b", "a\\b", " a", "a ",
     "a\nb", "-run", "a" * 65],
)
def test_named_shadow_rejects_invalid_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run ID"):
        stem_shadow_root(tmp_path, run_id)
    with pytest.raises(ValueError, match="run ID"):
        stem_separation_root(tmp_path, run_id)


def test_named_shadow_roots_and_output_ownership(tmp_path: Path) -> None:
    default = stem_shadow_root(tmp_path)
    selected = stem_shadow_root(tmp_path, "random10-v1")
    other = stem_shadow_root(tmp_path, "random20-v1")
    assert selected == default / "runs/random10-v1"
    assert stem_separation_root(tmp_path, "random10-v1") == selected / "separation"
    assert selected != other
    assert stem_shadow_root(tmp_path, "a" * 64).name == "a" * 64
    for stage in ("separation", "diarization", "asr", "mimo_stem_facts",
                  "mimo_reconcile", "references"):
        assert require_shadow_output_path(
            shadow_root=selected, output_path=selected / stage,
        ) == selected / stage
        for outside in (default / stage, other / stage, tmp_path / stage):
            with pytest.raises(ValueError, match="shadow root"):
                require_shadow_output_path(shadow_root=selected, output_path=outside)
    other.mkdir(parents=True)
    selected.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="redirect"):
        stem_shadow_root(tmp_path, "random10-v1")


def test_two_named_separations_preserve_default_and_each_other(tmp_path: Path) -> None:
    default, _, _, _ = _run_stems(tmp_path)
    before = {path: path.read_bytes() for path in default.rglob("*") if path.is_file()}
    first, _, _, _ = _run_stems(tmp_path, shadow_run_id="pilot-a")
    first_before = {
        path: path.read_bytes() for path in first.rglob("*") if path.is_file()
    }
    second, _, _, _ = _run_stems(tmp_path, shadow_run_id="pilot-b")
    assert len({default, first, second}) == 3
    for path, content in (before | first_before).items():
        assert path.read_bytes() == content
    for root in (default, first, second):
        _, records, _ = load_stem_shadow(root)
        assert all(
            root in Path(stem.canonical_stem_path).parents
            for record in records for stem in record.stems
        )


@pytest.mark.parametrize("run_id", [None, "random10-v1"])
def test_six_shadow_clis_use_selected_run_and_preserve_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_id: str | None,
) -> None:
    from tools import (
        export_h3_sam_audio_stem_references as references_cli,
    )
    from tools import (
        run_h3_mimo25_stem_facts_shadow as facts_cli,
    )
    from tools import (
        run_h3_mimo25_stem_reconcile_shadow as reconcile_cli,
    )
    from tools import (
        run_h3_sam_audio_stem_shadow as separation_cli,
    )
    from tools import (
        run_h3_stem_diarization_shadow as diarization_cli,
    )
    from tools import (
        run_h3_stem_qwen3_asr_shadow as asr_cli,
    )

    separation, diarization, asr, _, order = _run_three_clip_shadow_to_asr(
        tmp_path, shadow_run_id=run_id,
    )
    shadow = stem_shadow_root(tmp_path, run_id)
    common = ["--audio-production-root", str(tmp_path),
              "--case-manifest", str(tmp_path / "case.json")]
    if run_id is not None:
        common += ["--shadow-run-id", run_id]
        assert not (stem_shadow_root(tmp_path) / "separation").exists()
    dry = [*common, "--dry-run"]
    result = separation_cli.main([
        *dry, "--sam-audio-code-root", str(tmp_path / "sam-code"),
        "--sam-audio-model-path", str(tmp_path / "sam-model"),
        "--sam-audio-t5-base-path", str(tmp_path / "t5-base"),
    ])
    assert result["output_root"] == str(separation)
    assert result["clip_uids"] == order
    shutil.copytree(tmp_path / "production-diarization", tmp_path / "diarization")
    result = diarization_cli.main([*dry, "--allow-unverified"])
    assert result["output_root"] == str(diarization)
    assert result["target_clip_uids"] == order

    def no_asr_backend(**kwargs):
        raise AssertionError("dry run must not construct the isolated ASR backend")

    monkeypatch.setattr(asr_cli, "_isolated_backend", no_asr_backend)
    result = asr_cli.main([
        *dry, "--visual-production-root", "/visual/source", "--allow-unverified",
    ])
    assert result["diarization_root"] == str(diarization)
    assert result["output_root"] == str(asr)
    result = facts_cli.main([*dry, "--allow-unverified"])
    assert result["output_root"] == str(shadow / "mimo_stem_facts")
    assert result["clip_uids"] == order
    facts_summary = run_mimo25_stem_facts_shadow(
        stem_root=separation, route="music_first", backend=_FactsBackend(),
        view_backend=_ViewBackend(), output_root=shadow / "mimo_stem_facts",
        allow_unverified=True,
    )
    assert facts_summary.source_clip_uids == order
    assert facts_summary.source_stem_root == str(separation)
    assert facts_summary.source_stem_diarization_root == str(diarization)
    assert facts_summary.source_stem_asr_root == str(asr)
    base = _mimo_inventory_for_clip_order(tmp_path / "base", order)
    monkeypatch.setattr(reconcile_cli, "build_mimo25_inventory", lambda **kwargs: base)
    result = reconcile_cli.main([
        *dry, "--visual-production-root", "/visual/source",
        "--visual-runs-root", "/visual/runs", "--allow-unverified",
    ])
    assert result["output_root"] == str(shadow / "mimo_reconcile")
    assert result["clip_uids"] == order
    reference_manifest = tmp_path / "references.json"
    reference_manifest.write_text(json.dumps({
        "schema_version": "r2v.h3.sam_audio_stem_reference_manifest.1",
        "requests": [{
            "reference_id": clip, "clip_uid": clip, "route": "music_first",
            "stem_type": "speech", "start_time": 0, "end_time": 0.5,
        } for clip in order],
    }), encoding="utf-8")
    monkeypatch.setattr(references_cli, "FFmpegAudioMediaBackend", lambda **kwargs: _Media())
    result = references_cli.main([
        *common, "--reference-manifest", str(reference_manifest), "--allow-unverified",
    ])
    assert result["output_root"] == str(shadow / "references")
    assert result["reference_ids"] == order
    for reference in result["references"]:
        assert str(separation) in reference["source_stem_path"]


@pytest.mark.parametrize("source_run", [None, "pilot-other"])
@pytest.mark.parametrize("stage", ["separation", "diarization", "asr"])
def test_named_run_rejects_copied_cross_run_lineage(
    tmp_path: Path, source_run: str | None, stage: str,
) -> None:
    separation, diarization, asr, _, _ = _run_three_clip_shadow_to_asr(
        tmp_path, shadow_run_id=source_run,
    )
    selected = stem_shadow_root(tmp_path, "pilot-current")
    selected.mkdir(parents=True)
    source = {"separation": separation, "diarization": diarization, "asr": asr}[stage]
    copied = selected / stage
    shutil.copytree(source, copied)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="shadow (root|run)"):
        if stage == "separation":
            load_stem_shadow(copied)
        elif stage == "diarization":
            validate_stem_diarization_lineage(copied, expected_shadow_root=selected)
        else:
            validate_stem_asr_lineage(copied, expected_shadow_root=selected)
    assert all(path.read_bytes() == data for path, data in before.items())


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


@pytest.mark.parametrize("duration_delta", [0.032, -0.025])
def test_bounded_sam_duration_drift_preserves_actual_stem_extent(
    tmp_path: Path,
    duration_delta: float,
) -> None:
    output, _, _, _ = _run_stems(
        tmp_path,
        raw_duration_delta=duration_delta,
    )
    _, records, _ = load_stem_shadow(output)
    assert records[0].separation_state == "unverified"
    for stem in records[0].stems:
        assert stem.raw_duration_delta_seconds == pytest.approx(duration_delta)
        assert stem.canonical_duration_delta_seconds == pytest.approx(duration_delta)
        assert stem.canonical_frame_count == stem.raw_frame_count
        assert stem.canonical_frame_count != stem.source_end_sample
        assert stem.alignment_tolerance_seconds == STEM_ALIGNMENT_TOLERANCE_SECONDS
        assert stem.alignment_offset_seconds == 0
        assert stem.timeline_policy == STEM_TIMELINE_POLICY
        assert stem.canonical_channel_policy == STEM_CANONICAL_CHANNEL_POLICY
        assert stem.raw_channels == 1
        assert stem.canonical_channels == 2
        assert stem.alignment_state == "normal_bounded_drift"
        assert stem.canonical_uncovered_tail_seconds == pytest.approx(
            max(0.0, -duration_delta)
        )
        assert stem.conversion_settings == ["fake-raw-stem-only-conversion"]


def test_eighty_millisecond_sam_drift_is_accepted_with_warning(
    tmp_path: Path,
) -> None:
    output, _, _, _ = _run_stems(tmp_path, raw_duration_delta=0.080)
    _, records, _ = load_stem_shadow(output)
    assert records[0].separation_state == "unverified"
    assert {
        item.alignment_state for item in records[0].stems
    } == {"accepted_with_alignment_warning"}


def test_one_hundred_twenty_millisecond_sam_drift_fails_closed(
    tmp_path: Path,
) -> None:
    output, _, _, _ = _run_stems(tmp_path, raw_duration_delta=0.120)
    _, records, _ = load_stem_shadow(output)
    assert records[0].separation_state == "failure"
    assert "duration differs" in (records[0].failure_reason or "")


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
    output = stem_separation_root(tmp_path)
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


def test_separation_overwrite_preserves_downstream_stage_sentinels(
    tmp_path: Path,
) -> None:
    output, inventory, _, _ = _run_stems(tmp_path)
    sentinels = []
    for name in (
        "diarization",
        "asr",
        "stem_views",
        "mimo_stem_facts",
        "mimo_reconcile",
        "references",
    ):
        sentinel = output.parent / name / "sentinel.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text(name, encoding="utf-8")
        sentinels.append(sentinel)

    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=output,
        backend=_SAM(inventory.model_configuration),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=_Media(),
        overwrite=True,
    )

    assert [item.read_text(encoding="utf-8") for item in sentinels] == [
        "diarization",
        "asr",
        "stem_views",
        "mimo_stem_facts",
        "mimo_reconcile",
        "references",
    ]


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


def test_official_backend_matches_current_local_api_and_list_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = _configuration(tmp_path)
    source = tmp_path / "source.wav"
    _write_wav(source)
    calls: dict[str, object] = {}

    class Tensor:
        def __init__(self, label: str, shape: tuple[int, ...] = (4,)) -> None:
            self.label = label
            self.shape = shape
            self.ndim = len(shape)

        def float(self) -> Tensor:
            calls.setdefault("float", []).append(self.label)
            return self

        def unsqueeze(self, dimension: int) -> Tensor:
            assert dimension == 0
            return Tensor(self.label, (1, *self.shape))

        def cpu(self) -> Tensor:
            return self

    class FiniteResult:
        def all(self) -> FiniteResult:
            return self

        def item(self) -> bool:
            return True

    class Batch:
        def to(self, device: str) -> Batch:
            calls["batch_device"] = device
            return self

    class Model:
        @classmethod
        def from_pretrained(
            cls,
            model_name_or_path: str,
            *,
            local_files_only: bool,
            text_encoder: dict[str, object],
            visual_ranker: object,
            text_ranker: object,
            span_predictor: object,
        ) -> Model:
            calls["model_load"] = {
                "model_name_or_path": model_name_or_path,
                "local_files_only": local_files_only,
                "text_encoder": text_encoder,
                "visual_ranker": visual_ranker,
                "text_ranker": text_ranker,
                "span_predictor": span_predictor,
            }
            return cls()

        def eval(self) -> Model:
            return self

        def to(self, device: str) -> Model:
            calls["model_device"] = device
            return self

        def separate(self, batch: Batch, **kwargs: object) -> object:
            calls["separate"] = (batch, kwargs)
            return SimpleNamespace(
                target=[Tensor("target")],
                residual=[Tensor("residual")],
            )

    class Processor:
        audio_sampling_rate = 48000

        @classmethod
        def from_pretrained(cls, model_name_or_path: str) -> Processor:
            calls["processor_load"] = model_name_or_path
            return cls()

        def __call__(self, **kwargs: object) -> Batch:
            calls["processor_call"] = kwargs
            return Batch()

    sam_audio = ModuleType("sam_audio")
    sam_audio.SAMAudio = Model
    sam_audio.SAMAudioProcessor = Processor
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    torch.isfinite = lambda tensor: FiniteResult()
    torchaudio = ModuleType("torchaudio")

    def save(path: str, tensor: Tensor, sample_rate: int) -> None:
        calls.setdefault("saved", []).append((path, tensor.label, sample_rate))
        Path(path).write_bytes(tensor.label.encode())

    torchaudio.save = save
    monkeypatch.setitem(sys.modules, "sam_audio", sam_audio)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    target = tmp_path / "target.wav"
    residual = tmp_path / "residual.wav"
    OfficialSAMAudioBackend(configuration).separate(
        clip_uid="clip-1",
        source_audio_path=source,
        prompt="music soundtrack",
        target_path=target,
        residual_path=residual,
    )

    model_load = calls["model_load"]
    assert isinstance(model_load, dict)
    assert model_load == {
        "model_name_or_path": configuration.model_path,
        "local_files_only": True,
        "text_encoder": {
            "name": configuration.t5_base_path,
            "dim": 768,
        },
        "visual_ranker": None,
        "text_ranker": None,
        "span_predictor": None,
    }
    assert calls["processor_load"] == configuration.model_path
    assert [item[1] for item in calls["saved"]] == ["target", "residual"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert json.loads(Path(configuration.model_config_path).read_text())["text_encoder"] == {
        "name": "google/t5-base",
        "dim": 768,
    }


def test_local_model_identifier_cannot_disagree_with_checkpoint_config(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    with pytest.raises(ValueError, match="model identifier differs"):
        sam_audio_configuration(
            implementation_root=Path(configuration.implementation_root),
            model_path=Path(configuration.model_path),
            model_name="another-org/sam-audio-large",
            t5_base_path=Path(configuration.t5_base_path),
            device="cuda:0",
            reranking_candidates=1,
        )


def test_local_model_identifier_without_config_name_uses_matching_path_basename(
    tmp_path: Path,
) -> None:
    existing = _configuration(tmp_path)
    original_model = Path(existing.model_path)
    model = original_model.with_name("sam-audio-small-tv")
    original_model.rename(model)
    raw = json.loads((model / "config.json").read_text(encoding="utf-8"))
    raw.pop("_name_or_path")
    (model / "config.json").write_text(json.dumps(raw), encoding="utf-8")

    supplied = sam_audio_configuration(
        implementation_root=Path(existing.implementation_root),
        model_path=model,
        model_name="facebook/sam-audio-small-tv",
        t5_base_path=Path(existing.t5_base_path),
        device="cuda:0",
        reranking_candidates=1,
    )
    derived = sam_audio_configuration(
        implementation_root=Path(existing.implementation_root),
        model_path=model,
        model_name=None,
        t5_base_path=Path(existing.t5_base_path),
        device="cuda:0",
        reranking_candidates=1,
    )
    assert supplied.model_name == "facebook/sam-audio-small-tv"
    assert derived.model_name == "sam-audio-small-tv"
    with pytest.raises(ValueError, match="model identifier differs"):
        sam_audio_configuration(
            implementation_root=Path(existing.implementation_root),
            model_path=model,
            model_name="facebook/sam-audio-large-tv",
            t5_base_path=Path(existing.t5_base_path),
            device="cuda:0",
            reranking_candidates=1,
        )


def test_disabled_rankers_require_exactly_one_reranking_candidate(
    tmp_path: Path,
) -> None:
    existing = _configuration(tmp_path)
    with pytest.raises(ValueError, match="runtime configuration is incomplete"):
        sam_audio_configuration(
            implementation_root=Path(existing.implementation_root),
            model_path=Path(existing.model_path),
            model_name=existing.model_name,
            t5_base_path=Path(existing.t5_base_path),
            device="cuda:0",
            reranking_candidates=2,
        )
    with pytest.raises(SystemExit):
        _sam_stem_parser().parse_args(
            [
                "--audio-production-root",
                str(tmp_path),
                "--sam-audio-code-root",
                existing.implementation_root,
                "--sam-audio-model-path",
                existing.model_path,
                "--sam-audio-t5-base-path",
                existing.t5_base_path,
                "--sam-reranking-candidates",
                "2",
            ]
        )


@pytest.mark.parametrize(
    ("values", "expected_shape", "error"),
    [
        ([0.0, 0.5], (1, 2), None),
        ([[0.0, 0.5]], (1, 2), None),
        ([[0.0, 0.5], [0.5, 0.0]], None, "shape"),
        ([], None, "empty"),
        ([0.0, float("nan")], None, "finite"),
        ([0.0, float("inf")], None, "finite"),
    ],
)
def test_official_sam_waveform_normalization_is_strict_mono_float(
    values: list[object],
    expected_shape: tuple[int, int] | None,
    error: str | None,
) -> None:
    class Tensor:
        def __init__(self, payload: object) -> None:
            self.values = np.asarray(payload)

        @property
        def ndim(self) -> int:
            return self.values.ndim

        @property
        def shape(self) -> tuple[int, ...]:
            return self.values.shape

        def float(self) -> Tensor:
            result = Tensor(self.values.astype(np.float32))
            return result

        def unsqueeze(self, dimension: int) -> Tensor:
            return Tensor(np.expand_dims(self.values, dimension))

        def cpu(self) -> Tensor:
            return self

    torch = SimpleNamespace(isfinite=lambda value: np.isfinite(value.values))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            OfficialSAMAudioBackend._normalized_waveform(
                Tensor(values),
                torch=torch,
                label="target",
            )
        return
    output = OfficialSAMAudioBackend._normalized_waveform(
        Tensor(values),
        torch=torch,
        label="target",
    )
    assert output.shape == expected_shape
    assert output.values.dtype == np.float32


def test_unverified_stems_require_explicit_downstream_opt_in(tmp_path: Path) -> None:
    output, _, _, _ = _run_stems(tmp_path)
    _, records, _ = load_stem_shadow(output)
    with pytest.raises(ValueError, match="explicit allow_unverified"):
        selected_stem_records(records, route="music_first")
    with pytest.raises(ValueError, match="explicit allow_unverified"):
        export_stem_native_references(
            records=records,
            requests=[
                StemReferenceRequest(
                    reference_id="manual-e1",
                    clip_uid="clip-1",
                    route="music_first",
                    stem_type="speech",
                    start_time=0.0,
                    end_time=0.5,
                )
            ],
            output_root=output.parent / "references",
            media_backend=_Media(),
        )


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
        allow_unverified=True,
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
        allow_unverified=True,
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
        allow_unverified=True,
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
    return _production_diarization_inventory_for_records(tmp_path, [canonical])


def _production_diarization_inventory_for_records(
    tmp_path: Path,
    canonical: list[CanonicalAudioClip],
) -> DiarizationInventory:
    targets = [
        DiarizationTargetClip(
            target_clip_uid=item.clip_uid,
            target_video_path=item.target_video_path,
            source_audio_path=item.target_full_audio_path,
            source_audio_sha256=item.target_full_audio_sha256,
            source_sample_rate_hz=32000,
            source_channels=2,
            source_frame_count=item.frame_count,
            visual_references=[],
        )
        for item in canonical
    ]
    visual_inventory = tmp_path / "visual-samples.jsonl"
    visual_inventory.write_text("{}\n", encoding="utf-8")
    canonical_manifest = tmp_path / "audio/canonical_clips.jsonl"
    visual_hash = sha256_file(visual_inventory)
    canonical_hash = sha256_file(canonical_manifest)
    fingerprint = _diarization_inventory_fingerprint(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
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
        source_target_count=len(targets),
        selected_target_count=len(targets),
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
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


class _TailDiarization(_Diarization):
    def diarize(self, *, clip_uid: str, audio_path: Path):
        del clip_uid
        self.paths.append(audio_path)
        return [
            DiarizationBackendSegment(
                start_time=0.8,
                end_time=1.02,
                speaker_label="A",
            )
        ]


class _MixedDiarization(_Diarization):
    def diarize(self, *, clip_uid: str, audio_path: Path):
        self.paths.append(audio_path)
        if clip_uid == "clip-a":
            raise RuntimeError("synthetic per-clip DiariZen failure")
        if clip_uid == "clip-m":
            return []
        return [
            DiarizationBackendSegment(
                start_time=0.0,
                end_time=0.5,
                speaker_label="A",
            )
        ]


def _run_diarization(
    tmp_path: Path,
    *,
    raw_duration_delta: float | None = None,
    backend: _Diarization | None = None,
) -> tuple[Path, Path, SAMAudioStemRecord]:
    stem_root, inventory, _, _ = _run_stems(
        tmp_path,
        raw_duration_delta=raw_duration_delta,
    )
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
    active_backend = backend or _Diarization()
    output = stem_root.parent / "diarization"
    run_stem_diarization_shadow(
        stem_root=stem_root,
        production_diarization_root=production_root,
        backend=active_backend,
        route="music_first",
        output_root=output,
        allow_unverified=True,
    )
    assert active_backend.paths == [
        Path(stem_records[0].stem("speech").canonical_stem_path)
    ]
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
        allow_unverified=True,
    )
    assert shadow.targets[0].source_audio_path == records[0].stem("speech").canonical_stem_path
    assert production.targets[0].source_audio_path == canonical.target_full_audio_path
    assert production.model_dump_json().encode() == production_bytes


@pytest.mark.parametrize(
    ("duration_delta", "expected_frames"),
    [(0.032, 32000), (-0.025, 31200)],
)
def test_stem_diarization_uses_original_and_available_stem_intersection(
    tmp_path: Path,
    duration_delta: float,
    expected_frames: int,
) -> None:
    stem_root, inventory, _, _ = _run_stems(
        tmp_path,
        raw_duration_delta=duration_delta,
    )
    _, records, _ = load_stem_shadow(stem_root)
    canonical = CanonicalAudioClip.model_validate_json(
        Path(inventory.source_canonical_audio_manifest_path)
        .read_text(encoding="utf-8")
        .strip()
    )
    shadow = build_stem_diarization_inventory(
        stem_inventory=inventory,
        stem_records=records,
        production_diarization_inventory=_production_diarization_inventory(
            tmp_path,
            canonical,
        ),
        route="music_first",
        allow_unverified=True,
    )

    assert shadow.targets[0].source_frame_count == expected_frames
    assert records[0].stem("speech").canonical_frame_count == round(
        (1.0 + duration_delta) * 32000
    )


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
        output_root=stem_root.parent / "asr",
        segment_audio_loader=loader,
        allow_unverified=True,
    )
    assert observed == [Path(stem_record.stem("speech").canonical_stem_path)]
    assert summary.transcribed_count == 1
    assert provenance.asr_source_kind == "sam_audio_speech_stem_segments"
    row = json.loads(
        (stem_root.parent / "asr/segments.jsonl").read_text(encoding="utf-8")
    )
    assert row["source_audio_path"] == stem_record.stem("speech").canonical_stem_path


def test_stem_diarization_clamps_codec_tail_to_original_target_eof(
    tmp_path: Path,
) -> None:
    _, diarization, _ = _run_diarization(
        tmp_path,
        raw_duration_delta=0.032,
        backend=_TailDiarization(),
    )
    row = RawDiarizationSegment.model_validate_json(
        (diarization / "raw_segments.jsonl").read_text(encoding="utf-8")
    )
    assert row.backend_reported_end_time == 1.02
    assert row.end_time == 1.0
    assert row.source_end_sample == 32000
    assert row.boundary_reconciliation.end_clamped is True


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
        temperature=0.2,
        max_completion_tokens=4096,
        configuration_fingerprint="d" * 64,
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, *, job: StemFactJob, stem_type: str, view: StemView):
        self.calls.append(stem_type)
        if stem_type == "speech":
            facts = SpeechStemFacts(
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
        elif stem_type == "music":
            facts = MusicStemFacts(
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
        else:
            facts = SFXStemFacts(
                clip_duration_seconds=job.clip_duration_seconds,
                continuous_layers=[],
                events=[],
            )
        return StemFactsBackendResult(
            facts=facts,
            raw_response=facts.model_dump_json(),
            diagnostics={"view_sha256": view.view_sha256},
        )


class _OneClipFailFactsBackend(_FactsBackend):
    def __init__(self, failed_clip_uid: str) -> None:
        super().__init__()
        self.failed_clip_uid = failed_clip_uid

    def extract(self, *, job: StemFactJob, stem_type: str, view: StemView):
        if job.clip_uid == self.failed_clip_uid and stem_type == "music":
            self.calls.append(stem_type)
            raise StemFactsBackendFailure(
                code="stem_fact_structured_output_failed",
                reason="synthetic malformed music facts",
                raw_response="{not-json",
                diagnostics={"view_sha256": view.view_sha256},
            )
        return super().extract(job=job, stem_type=stem_type, view=view)


class _OutOfExtentMusicFactsBackend(_FactsBackend):
    def extract(self, *, job: StemFactJob, stem_type: str, view: StemView):
        result = super().extract(job=job, stem_type=stem_type, view=view)
        if stem_type != "music":
            return result
        facts = MusicStemFacts(
            clip_duration_seconds=job.clip_duration_seconds,
            music_status="present",
            intervals=[
                MusicInterval(
                    start_time=0.9,
                    end_time=0.99,
                    description="tail outside the available shorter stem",
                    confidence="high",
                )
            ],
        )
        return StemFactsBackendResult(
            facts=facts,
            raw_response=facts.model_dump_json(),
            diagnostics=result.diagnostics,
        )


class _FailingViewBackend(_ViewBackend):
    def create(self, **kwargs: object) -> StemView:
        raise RuntimeError("view creation failed")


def _run_facts(
    tmp_path: Path,
    *,
    raw_duration_delta: float | None = None,
    facts_backend: _FactsBackend | None = None,
) -> tuple[Path, MimoStemFactsRecord]:
    stem_root, _, _ = _run_diarization(
        tmp_path,
        raw_duration_delta=raw_duration_delta,
    )
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=stem_root.parent / "diarization",
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=stem_root.parent / "asr",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
        allow_unverified=True,
    )
    backend = facts_backend or _FactsBackend()
    summary = run_mimo25_stem_facts_shadow(
        stem_root=stem_root,
        route="music_first",
        backend=backend,
        view_backend=_ViewBackend(),
        output_root=stem_root.parent / "mimo_stem_facts",
        allow_unverified=True,
    )
    records = load_stem_fact_records(stem_root.parent / "mimo_stem_facts")
    assert backend.calls == ["speech", "music", "sfx"]
    assert summary.model_call_count == 3
    return stem_root, records[0]


def test_stem_facts_use_full_timeline_and_keep_music_timing(tmp_path: Path) -> None:
    stem_root, record = _run_facts(tmp_path)
    assert record.music.intervals[0].start_time == 0.1
    assert record.music.intervals[0].end_time == 0.4
    assert all(Path(item.view_path).is_file() for item in record.stem_views)
    assert (stem_root.parent / "mimo_stem_facts/records.jsonl").is_file()


def test_stem_fact_interval_beyond_shorter_available_media_fails_closed(
    tmp_path: Path,
) -> None:
    _, record = _run_facts(
        tmp_path,
        raw_duration_delta=-0.025,
        facts_backend=_OutOfExtentMusicFactsBackend(),
    )
    assert record.status == "failed"
    assert record.failing_stem_type == "music"
    assert record.failure_reason == "music stem fact exceeds available source media"


def test_failed_facts_rerun_preserves_published_records_and_views(
    tmp_path: Path,
) -> None:
    stem_root, record = _run_facts(tmp_path)
    facts_root = stem_root.parent / "mimo_stem_facts"
    records_before = (facts_root / "records.jsonl").read_bytes()
    views_before = {
        Path(item.view_path): Path(item.view_path).read_bytes()
        for item in record.stem_views
    }

    summary = run_mimo25_stem_facts_shadow(
        stem_root=stem_root,
        route="music_first",
        backend=_FactsBackend(),
        view_backend=_FailingViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
        overwrite=True,
    )

    assert (facts_root / "records.jsonl").read_bytes() != records_before
    assert summary.ready_count == 0
    assert summary.failed_count == 1
    failed = load_stem_fact_records(facts_root)[0]
    assert failed.status == "failed"
    assert failed.failure_reason == "view creation failed"
    assert failed.model_call_count == 0
    assert {path: path.read_bytes() for path in views_before} == views_before


def test_stem_reconcile_contract_keeps_original_av_above_stem_facts(tmp_path: Path) -> None:
    _, record = _run_facts(tmp_path)
    contract = stem_reconcile_auxiliary_contract(record)
    assert contract["authority_order"][0] == "original_target_full_av"
    assert "cannot override contradictory original AV" in " ".join(contract["rules"])
    assert contract["stem_facts"]["music"]["intervals"][0]["start_time"] == 0.1
    assert contract["policy_version"] == "h3_mimo25_stem_reconcile_v2"
    rules = " ".join(contract["rules"])
    assert "Negative stem evidence is non-confirmatory" in rules
    assert "Music stem absent does NOT establish" in rules
    assert "Empty SFX items do NOT establish" in rules
    assert "Separator labels are never semantic truth" in rules


def test_negative_stem_facts_do_not_override_positive_original_av(tmp_path: Path) -> None:
    from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft
    from tests.test_h3_mimo25_av_shadow import (
        _materialize_sample,
        _playback_inputs,
        _record_fixture,
    )

    class NegativeFacts(_FactsBackend):
        def extract(self, *, job, stem_type, view):
            result = super().extract(job=job, stem_type=stem_type, view=view)
            if stem_type != "music":
                return result
            facts = MusicStemFacts(clip_duration_seconds=job.clip_duration_seconds,
                                   music_status="absent", intervals=[])
            return StemFactsBackendResult(facts=facts, raw_response=facts.model_dump_json(),
                                          diagnostics=result.diagnostics)

    _, facts = _run_facts(tmp_path / "stems", facts_backend=NegativeFacts())
    contract = stem_reconcile_auxiliary_contract(facts)
    assert contract["stem_facts"]["music"]["music_status"] == "absent"
    assert contract["stem_facts"]["sfx"]["events"] == []
    assert contract["stem_facts"]["sfx"]["continuous_layers"] == []
    target = tmp_path / "original"
    target.mkdir()
    sample, job, record = _playback_inputs(target)
    values = record.annotation.model_dump(mode="json")
    values["audio_observation"]["audio_semantics"].update(
        non_diegetic_music_status="present", non_diegetic_music="A faint musical score continues.",
    )
    annotation = MimoAVAnnotationDraft.model_validate(values)
    _, text, _ = _materialize_sample(sample, job, _record_fixture(target, annotation, job=job))
    assert "overall_soundscape:\nA quiet room tone accompanies a short clink." in text
    assert "non_diegetic_music:\nA faint musical score continues." in text


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
        "speech_stem_available_duration_seconds": 1.0,
        "music_stem_available_duration_seconds": 1.0,
        "sfx_stem_available_duration_seconds": 1.0,
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
    assert request["temperature"] == 0.2
    assert backend.provenance.temperature == 0.2
    assert MIMO25_STEM_FACT_PROMPT_VERSION in backend.provenance.prompt_version
    assert "asr_text" not in content[1]["text"]
    assert "entity_id" not in content[1]["text"]


def test_stem_reconcile_cli_keeps_current_completion_budget() -> None:
    arguments = _stem_reconcile_parser().parse_args(
        [
            "--visual-production-root",
            "/visual",
            "--visual-runs-root",
            "/runs",
            "--audio-production-root",
            "/audio",
            "--case-manifest",
            "/case.json",
        ]
    )
    assert arguments.max_completion_tokens == 32768
    assert arguments.temperature == 0.0


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
        "speech_stem_available_duration_seconds": 1.0,
        "music_stem_available_duration_seconds": 1.0,
        "sfx_stem_available_duration_seconds": 1.0,
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


def _mimo_inventory_for_clip_order(
    tmp_path: Path,
    clip_uids: list[str],
) -> MimoInventory:
    template = _mimo_job_fixture(tmp_path)
    jobs: list[MimoClipJob] = []
    for clip_uid in clip_uids:
        values = template.model_dump(mode="json", exclude={"request_fingerprint"})
        values["clip_uid"] = clip_uid
        values["source_h3_sample_ids"] = [f"{clip_uid}/in_pair"]
        jobs.append(_mimo_job(values))
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
        "clip_count": len(jobs),
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    return _mimo_inventory(values)


class _FailingReconcileBackend:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[str] = []
        self.provenance = StemAwareOpenAIMimo25Backend(
            MimoBackendConfig(
                media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
                api_key="test",
                transport="sglang",
                base_url="http://localhost/v1",
            ),
            stem_facts_by_clip={},
            client=SimpleNamespace(),
        ).provenance

    def reconcile(self, job: MimoClipJob, **_: object) -> object:
        self.calls.append(job.clip_uid)
        raise MimoBackendFailure(
            code="synthetic_failure",
            reason="synthetic reconcile failure",
        )


def test_stem_reconcile_jobs_keep_original_av_and_use_shadow_speech_timeline(
    tmp_path: Path,
) -> None:
    stem_root, diarization, _ = _run_diarization(tmp_path)
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=diarization,
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=stem_root.parent / "asr",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(8000, dtype=np.float32),
            16000,
        ),
        allow_unverified=True,
    )
    base = _mimo_inventory_fixture(tmp_path / "base")
    jobs = build_stem_reconcile_jobs(
        base_inventory=base,
        stem_diarization_root=diarization,
        stem_asr_root=stem_root.parent / "asr",
    )
    assert jobs[0].target_video_path == base.jobs[0].target_video_path
    assert jobs[0].target_full_audio_path == base.jobs[0].target_full_audio_path
    assert [item.segment_id for item in jobs[0].segments] == ["segment_0001"]
    assert jobs[0].segments[0].asr_text == "exact transcript"


def test_failed_clip_is_skipped_while_manifest_order_reaches_reconcile(
    tmp_path: Path,
) -> None:
    clip_order = ["clip-z", "clip-a", "clip-m"]
    manifest, canonical_records, case = _multi_canonical_fixture(
        tmp_path,
        clip_uids=clip_order,
    )
    configuration = _configuration(tmp_path)
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=configuration,
        case_manifest_path=case,
    )
    separation = stem_separation_root(tmp_path)
    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=separation,
        backend=_OneClipFailSAM(configuration, "clip-a"),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=_Media(),
    )
    loaded_inventory, stem_records, _ = load_stem_shadow(separation)
    assert loaded_inventory.clip_uids == clip_order
    assert [item.clip_uid for item in stem_records] == clip_order
    assert [item.clip_uid for item in separation_skips(
        inventory=loaded_inventory,
        records=stem_records,
        route="music_first",
    )] == ["clip-a"]

    canonical_by_clip = {item.clip_uid: item for item in canonical_records}
    production_root = tmp_path / "production-diarization"
    production_root.mkdir()
    production_inventory = _production_diarization_inventory_for_records(
        tmp_path,
        [canonical_by_clip[item] for item in clip_order],
    )
    (production_root / "inventory.json").write_text(
        production_inventory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    shadow_root = separation.parent
    diarization_root = shadow_root / "diarization"
    diarization_provenance = run_stem_diarization_shadow(
        stem_root=separation,
        production_diarization_root=production_root,
        backend=_Diarization(),
        route="music_first",
        output_root=diarization_root,
        allow_unverified=True,
    )
    assert diarization_provenance.clip_uids == clip_order
    assert diarization_provenance.usable_clip_uids == ["clip-z", "clip-m"]
    assert [item.clip_uid for item in diarization_provenance.skipped_clips] == [
        "clip-a"
    ]

    _, asr_provenance = run_stem_qwen3_asr_shadow(
        stem_diarization_root=diarization_root,
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=shadow_root / "asr",
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
        allow_unverified=True,
    )
    assert asr_provenance.source_clip_uids == clip_order
    assert asr_provenance.clip_uids == ["clip-z", "clip-m"]

    facts_summary = run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=_FactsBackend(),
        view_backend=_ViewBackend(),
        output_root=shadow_root / "mimo_stem_facts",
        allow_unverified=True,
    )
    facts = load_stem_fact_records(shadow_root / "mimo_stem_facts")
    assert facts_summary.source_clip_uids == clip_order
    assert facts_summary.clip_uids == ["clip-z", "clip-m"]
    assert [item.clip_uid for item in facts] == ["clip-z", "clip-m"]

    jobs = build_stem_reconcile_jobs(
        base_inventory=_mimo_inventory_for_clip_order(tmp_path / "base", clip_order),
        stem_diarization_root=diarization_root,
        stem_asr_root=shadow_root / "asr",
    )
    assert [item.clip_uid for item in jobs] == ["clip-z", "clip-m"]
    backend = _FailingReconcileBackend(tmp_path)
    reconcile_summary = run_mimo25_stem_reconcile_shadow(
        jobs=jobs,
        stem_facts=facts,
        backend=backend,
        output_root=shadow_root / "mimo_reconcile",
        source_clip_uids=clip_order,
        skipped_clips=diarization_provenance.skipped_clips,
        route="music_first",
        allow_unverified=True,
    )
    assert backend.calls == ["clip-z", "clip-m"]
    assert reconcile_summary.clip_uids == clip_order
    assert reconcile_summary.processed_clip_uids == ["clip-z", "clip-m"]
    assert reconcile_summary.skipped_clip_count == 1


def _run_three_clip_shadow_to_asr(
    tmp_path: Path,
    *,
    diarization_backend: _Diarization | None = None,
    shadow_run_id: str | None = None,
) -> tuple[Path, Path, Path, object, list[str]]:
    clip_order = ["clip-z", "clip-a", "clip-m"]
    manifest, canonical_records, case = _multi_canonical_fixture(
        tmp_path,
        clip_uids=clip_order,
    )
    configuration = _configuration(tmp_path)
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest,
        model_configuration=configuration,
        case_manifest_path=case,
    )
    separation = stem_separation_root(tmp_path, shadow_run_id)
    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=separation,
        backend=_SAM(configuration),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=_Media(),
    )
    production_root = tmp_path / "production-diarization"
    production_root.mkdir()
    production_inventory = _production_diarization_inventory_for_records(
        tmp_path,
        canonical_records,
    )
    (production_root / "inventory.json").write_text(
        production_inventory.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    shadow = separation.parent
    diarization = shadow / "diarization"
    run_stem_diarization_shadow(
        stem_root=separation,
        production_diarization_root=production_root,
        backend=diarization_backend or _Diarization(),
        route="music_first",
        output_root=diarization,
        allow_unverified=True,
    )
    asr = shadow / "asr"
    run_stem_qwen3_asr_shadow(
        stem_diarization_root=diarization,
        source_visual_production_root="/visual/source",
        backend=_Qwen(),
        output_root=asr,
        segment_audio_loader=lambda path, start, end: (
            np.zeros(round((end - start) * 16000), dtype=np.float32),
            16000,
        ),
        route="music_first",
        allow_unverified=True,
    )
    return separation, diarization, asr, inventory, clip_order


def test_diarization_clip_failure_propagates_without_hiding_usable_clips(
    tmp_path: Path,
) -> None:
    separation, diarization, asr, _, clip_order = _run_three_clip_shadow_to_asr(
        tmp_path,
        diarization_backend=_MixedDiarization(),
    )
    shadow = separation.parent
    diarization_provenance, _, _ = validate_stem_diarization_lineage(diarization)
    assert diarization_provenance.ready_clip_uids == ["clip-z"]
    assert diarization_provenance.empty_clip_uids == ["clip-m"]
    assert diarization_provenance.failed_clip_uids == ["clip-a"]
    assert diarization_provenance.usable_clip_uids == ["clip-z", "clip-m"]
    assert diarization_provenance.clip_results_sha256 == sha256_file(
        diarization / "clip_results.jsonl"
    )
    readable_targets = [
        json.loads(line)
        for line in (diarization / "readable_targets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [item["clip_uid"] for item in readable_targets] == ["clip-z", "clip-m"]
    assert [
        item.clip_uid for item in diarization_provenance.diarization_failed_clips
    ] == ["clip-a"]

    asr_provenance, _ = validate_stem_asr_lineage(asr)
    assert asr_provenance.clip_uids == ["clip-z", "clip-m"]
    assert [item.clip_uid for item in asr_provenance.diarization_failed_clips] == [
        "clip-a"
    ]
    asr_rows = [
        json.loads(line)
        for line in (asr / "segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [item["clip_uid"] for item in asr_rows] == ["clip-z"]

    facts_root = shadow / "mimo_stem_facts"
    facts_backend = _FactsBackend()
    facts_summary = run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=facts_backend,
        view_backend=_ViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
    )
    facts = load_stem_fact_records(facts_root)
    assert facts_summary.clip_uids == ["clip-z", "clip-m"]
    assert facts_summary.model_call_count == 6
    assert [item.clip_uid for item in facts_summary.diarization_failed_clips] == [
        "clip-a"
    ]

    jobs = build_stem_reconcile_jobs(
        base_inventory=_mimo_inventory_for_clip_order(tmp_path / "base", clip_order),
        stem_diarization_root=diarization,
        stem_asr_root=asr,
    )
    assert [item.clip_uid for item in jobs] == ["clip-z", "clip-m"]
    backend = _FailingReconcileBackend(tmp_path)
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=jobs,
        stem_facts=facts,
        backend=backend,
        output_root=shadow / "mimo_reconcile",
        source_clip_uids=clip_order,
        diarization_failed_clips=facts_summary.diarization_failed_clips,
        route="music_first",
        allow_unverified=True,
    )
    assert backend.calls == ["clip-z", "clip-m"]
    assert summary.processed_clip_uids == ["clip-z", "clip-m"]
    assert summary.skipped_clip_count == 1
    assert [item.clip_uid for item in summary.diarization_failed_clips] == [
        "clip-a"
    ]


def test_reconcile_preflight_closes_case_and_current_separation_roots(
    tmp_path: Path,
) -> None:
    separation, diarization, asr, inventory, clip_order = (
        _run_three_clip_shadow_to_asr(tmp_path)
    )
    facts_root = separation.parent / "mimo_stem_facts"
    run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=_FactsBackend(),
        view_backend=_ViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
    )
    diarization_provenance, _, _ = validate_stem_diarization_lineage(
        diarization
    )
    facts_summary, _ = validate_stem_facts_lineage(
        facts_root=facts_root,
        stem_diarization_root=diarization,
        stem_asr_root=asr,
        route="music_first",
    )
    case_manifest = MimoCaseManifest(clip_uids=clip_order)
    _validate_stage_closure(
        case_manifest=case_manifest,
        stem_inventory=inventory,
        separation_root=separation,
        diarization_provenance=diarization_provenance,
        facts_summary=facts_summary,
    )

    with pytest.raises(ValueError, match="case manifest differs"):
        _validate_stage_closure(
            case_manifest=MimoCaseManifest(clip_uids=list(reversed(clip_order))),
            stem_inventory=inventory,
            separation_root=separation,
            diarization_provenance=diarization_provenance,
            facts_summary=facts_summary,
        )

    other_separation = tmp_path / "other-separation"
    other_separation.mkdir()
    with pytest.raises(ValueError, match="DiariZen source root differs"):
        _validate_stage_closure(
            case_manifest=case_manifest,
            stem_inventory=inventory,
            separation_root=separation,
            diarization_provenance=diarization_provenance.model_copy(
                update={"source_stem_root": str(other_separation)}
            ),
            facts_summary=facts_summary,
        )
    with pytest.raises(ValueError, match="facts source root differs"):
        _validate_stage_closure(
            case_manifest=case_manifest,
            stem_inventory=inventory,
            separation_root=separation,
            diarization_provenance=diarization_provenance,
            facts_summary=facts_summary.model_copy(
                update={"source_stem_root": str(other_separation)}
            ),
        )


def test_one_facts_failure_is_audited_and_reconcile_processes_other_clips(
    tmp_path: Path,
) -> None:
    separation, diarization, asr, _, clip_order = _run_three_clip_shadow_to_asr(
        tmp_path
    )
    shadow = separation.parent
    facts_root = shadow / "mimo_stem_facts"
    backend = _OneClipFailFactsBackend("clip-a")
    facts_summary = run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=backend,
        view_backend=_ViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
    )
    _, facts = validate_stem_facts_lineage(
        facts_root=facts_root,
        stem_diarization_root=diarization,
        stem_asr_root=asr,
        route="music_first",
    )
    assert facts_summary.ready_clip_uids == ["clip-z", "clip-m"]
    assert facts_summary.failed_clip_uids == ["clip-a"]
    assert facts_summary.ready_count == 2
    assert facts_summary.failed_count == 1
    assert facts_summary.model_call_count == 8
    failed = next(item for item in facts if item.clip_uid == "clip-a")
    assert failed.status == "failed"
    assert failed.failing_stem_type == "music"
    assert failed.failure_code == "stem_fact_structured_output_failed"
    assert [item.status for item in failed.raw_responses] == ["ready", "failed"]
    assert failed.raw_responses[-1].raw_response == "{not-json"
    raw = json.loads(
        (facts_root / "raw_responses/clip-a.json").read_text(encoding="utf-8")
    )
    assert raw["calls"][-1]["raw_response"] == "{not-json"

    jobs = build_stem_reconcile_jobs(
        base_inventory=_mimo_inventory_for_clip_order(tmp_path / "base", clip_order),
        stem_diarization_root=diarization,
        stem_asr_root=asr,
        route="music_first",
    )
    reconcile_backend = _FailingReconcileBackend(tmp_path)
    summary = run_mimo25_stem_reconcile_shadow(
        jobs=jobs,
        stem_facts=facts,
        backend=reconcile_backend,
        output_root=shadow / "mimo_reconcile",
        source_clip_uids=clip_order,
        route="music_first",
        allow_unverified=True,
    )
    assert reconcile_backend.calls == ["clip-z", "clip-m"]
    assert summary.route == "music_first"
    assert summary.processed_clip_uids == ["clip-z", "clip-m"]
    assert summary.skipped_clip_count == 1
    assert [item.clip_uid for item in summary.facts_failed_clips] == ["clip-a"]


def test_separation_overwrite_invalidates_complete_downstream_lineage_before_model(
    tmp_path: Path,
) -> None:
    separation, diarization, asr, inventory, clip_order = (
        _run_three_clip_shadow_to_asr(tmp_path)
    )
    shadow = separation.parent
    facts_root = shadow / "mimo_stem_facts"
    facts_backend = _FactsBackend()
    run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=facts_backend,
        view_backend=_ViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
    )
    old_records_hash = sha256_file(separation / "records.jsonl")
    run_sam_audio_stem_shadow(
        inventory=inventory,
        output_root=separation,
        backend=_VariantSAM(inventory.model_configuration),
        canonicalizer=_Canonicalizer(),
        raw_probe_backend=_Media(),
        overwrite=True,
    )
    assert sha256_file(separation / "records.jsonl") != old_records_hash

    with pytest.raises(ValueError, match="source separation records changed"):
        validate_stem_diarization_lineage(diarization)
    model = _FailingReconcileBackend(tmp_path)
    with pytest.raises(ValueError, match="source separation records changed"):
        build_stem_reconcile_jobs(
            base_inventory=_mimo_inventory_for_clip_order(
                tmp_path / "stale-base",
                clip_order,
            ),
            stem_diarization_root=diarization,
            stem_asr_root=asr,
            route="music_first",
        )
    assert model.calls == []


def test_changed_diarization_provenance_invalidates_existing_asr(tmp_path: Path) -> None:
    _, diarization, asr, _, _ = _run_three_clip_shadow_to_asr(tmp_path)
    provenance_path = diarization / "stem_provenance.json"
    provenance_path.write_text(
        provenance_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source DiariZen provenance changed"):
        validate_stem_asr_lineage(asr)


def test_changed_asr_provenance_invalidates_existing_facts(tmp_path: Path) -> None:
    separation, diarization, asr, _, _ = _run_three_clip_shadow_to_asr(tmp_path)
    facts_root = separation.parent / "mimo_stem_facts"
    run_mimo25_stem_facts_shadow(
        stem_root=separation,
        route="music_first",
        backend=_FactsBackend(),
        view_backend=_ViewBackend(),
        output_root=facts_root,
        allow_unverified=True,
    )
    provenance_path = asr / "stem_provenance.json"
    provenance_path.write_text(
        provenance_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stem facts source lineage changed"):
        validate_stem_facts_lineage(
            facts_root=facts_root,
            stem_diarization_root=diarization,
            stem_asr_root=asr,
            route="music_first",
        )


def test_route_mismatch_fails_before_stem_facts_model_calls(tmp_path: Path) -> None:
    separation, _, _, _, _ = _run_three_clip_shadow_to_asr(tmp_path)
    backend = _FactsBackend()
    with pytest.raises(ValueError, match="route records differ"):
        run_mimo25_stem_facts_shadow(
            stem_root=separation,
            route="voice_first",
            backend=backend,
            view_backend=_ViewBackend(),
            output_root=separation.parent / "wrong-route-facts",
            allow_unverified=True,
        )
    assert backend.calls == []


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
    assert output.name == "separation"
    assert output.parent.name == SAM_AUDIO_SHADOW_ROOT_NAME
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
