import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3 import t2va_existing_audio as existing
from r2v_data_v2.h3 import t2va_shadow as t2va
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.resolved_audio_stems import downstream_stem_root
from r2v_data_v2.h3.sam_audio_stem_shadow import stem_shadow_root
from r2v_data_v2.naming import clip_uid
from tests.test_h3_t2va_shadow import config
from tools import run_h3_t2va_shadow as cli


def _published_audio(tmp_path, monkeypatch):
    production = tmp_path / "audio-production"
    run_id = "fixed20-audio"
    shadow = stem_shadow_root(production, run_id)
    resolved_root = downstream_stem_root(production, run_id)
    (production / "audio").mkdir(parents=True)
    (shadow / "diarization").mkdir(parents=True)
    (shadow / "asr").mkdir(parents=True)
    resolved_root.mkdir(parents=True)
    clips, records = [], []
    for index in range(25):
        name = "movie_3_2.mp4" if index == 2 else f"movie_{index + 1}.mp4"
        video = tmp_path / name
        full = production / "audio" / f"movie_{index + 1}.flac"
        uid = clip_uid(video)
        clip = CanonicalAudioClip(
            clip_uid=uid,
            clip_display_path=f"movie/{video.stem}",
            media_collection_relpath="movie",
            media_collection_name="movie",
            episode_name="movie",
            clip_name=video.stem,
            shard_id="shard",
            target_video_path=str(video),
            target_video_sha256="a" * 64,
            target_full_audio_path=str(full),
            target_full_audio_sha256="b" * 64,
            frame_count=32000,
            target_duration_seconds=1.0,
            subject_reference_count=0,
        )
        clips.append(clip)
        stems = {}
        for kind in ("speech", "music", "sfx"):
            stem_path = resolved_root / f"{uid}-{kind}.wav"
            stems[kind] = SimpleNamespace(
                canonical_stem_path=str(stem_path),
                canonical_stem_sha256="c" * 64,
                source_audio_path=str(full),
                source_audio_sha256="b" * 64,
                source_end_sample=32000,
            )
        records.append(
            SimpleNamespace(
                clip_uid=uid,
                status="ready",
                record_fingerprint="d" * 64,
                stem=lambda kind, values=stems: values[kind],
            )
        )
    wanted = [clip.clip_uid for clip in reversed(clips[2:22])]
    for clip in clips[2:22]:
        Path(clip.target_video_path).write_bytes(b"video")
        Path(clip.target_full_audio_path).write_bytes(b"audio")
        for kind in ("speech", "music", "sfx"):
            Path(records[clips.index(clip)].stem(kind).canonical_stem_path).write_bytes(
                b"stem"
            )
    canonical_path = production / "audio/canonical_clips.jsonl"
    canonical_path.write_text("".join(clip.model_dump_json() + "\n" for clip in clips))
    (shadow / "diarization/raw_segments.jsonl").write_text("")
    (shadow / "asr/segments.jsonl").write_text("")
    (shadow / "diarization/stem_provenance.json").write_text(
        json.dumps({"raw_segments_sha256": "e" * 64})
    )
    (shadow / "asr/stem_provenance.json").write_text(
        json.dumps({"clip_uids": [clip.clip_uid for clip in clips],
                    "segments_sha256": "f" * 64})
    )
    monkeypatch.setattr(
        existing,
        "load_resolved_stems",
        lambda root: (
            SimpleNamespace(
                source_canonical_audio_manifest_sha256="1" * 64,
                clip_uids=[clip.clip_uid for clip in clips],
            ),
            records,
            None,
        ),
    )
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"clip_uids": wanted}))
    return production, run_id, case, wanted


def test_fixed20_selects_published_audio_in_case_order_without_source_scan(
    tmp_path, monkeypatch
):
    production, run_id, case, wanted = _published_audio(tmp_path, monkeypatch)
    inventory = existing.build_t2va_inventory_from_existing_audio(
        audio_production_root=production,
        audio_shadow_run_id=run_id,
        case_manifest=case,
        t2va_run_id="v26-fixed20",
        backend=config(tmp_path).provenance(),
    )
    assert inventory.clip_uids == wanted
    assert len(inventory.jobs) == 20
    assert [job.target_video_path for job in inventory.jobs] == [
        str(tmp_path / ("movie_3_2.mp4" if index == 3 else f"movie_{index}.mp4"))
        for index in range(22, 2, -1)
    ]
    assert all(job.upstream_failure is None for job in inventory.jobs)
    assert all(job.audio_evidence is not None for job in inventory.jobs)
    assert inventory.shot_selection.shot_manifest_path == str(
        production / "audio/canonical_clips.jsonl"
    )
    assert inventory.shot_selection.case_manifest_path == str(case.resolve())


def test_fixed20_rejects_missing_and_duplicate_case_uids(tmp_path, monkeypatch):
    production, run_id, case, wanted = _published_audio(tmp_path, monkeypatch)
    options = {
        "audio_production_root": production,
        "audio_shadow_run_id": run_id,
        "case_manifest": case,
        "t2va_run_id": "v26-fixed20",
        "backend": config(tmp_path).provenance(),
    }
    case.write_text(json.dumps({"clip_uids": [*wanted, "missing"]}))
    with pytest.raises(ValueError, match="missing"):
        existing.build_t2va_inventory_from_existing_audio(**options)
    case.write_text(json.dumps({"clip_uids": [wanted[0], wanted[0]]}))
    with pytest.raises(ValueError, match="duplicate"):
        existing.build_t2va_inventory_from_existing_audio(**options)


def test_fixed20_cli_dry_run_needs_no_shot_manifest(tmp_path, monkeypatch):
    production, run_id, case, wanted = _published_audio(tmp_path, monkeypatch)
    result = cli.main(
        [
            "--existing-audio-cases",
            "--audio-production-root", str(production),
            "--audio-shadow-run-id", run_id,
            "--case-manifest", str(case),
            "--t2va-run-id", "v26-fixed20",
            "--base-url", "http://127.0.0.1:8092/v1",
            "--media-root", str(tmp_path),
            "--dry-run",
        ]
    )
    assert result["clip_uids"] == wanted
    assert result["model_call_count"] == 0


def test_fixed20_shadow_run_does_not_rehash_published_sources(tmp_path, monkeypatch):
    production, run_id, case, _ = _published_audio(tmp_path, monkeypatch)
    inventory = existing.build_t2va_inventory_from_existing_audio(
        audio_production_root=production,
        audio_shadow_run_id=run_id,
        case_manifest=case,
        t2va_run_id="v26-fixed20",
        backend=config(tmp_path).provenance(),
    )
    monkeypatch.setattr(
        t2va, "_check_sources", lambda *_: pytest.fail("source files rehashed")
    )
    original_hash = t2va.sha256_file

    def output_hash(path):
        assert "t2va_shadow_v1" in Path(path).parts
        return original_hash(path)

    monkeypatch.setattr(t2va, "sha256_file", output_hash)

    class FailedBackend:
        def provenance(self):
            return inventory.backend

        def annotate(self, job, request_fingerprint):
            return t2va.T2VARawResponse(
                clip_uid=job.clip_uid,
                request_fingerprint=request_fingerprint,
                model_call_count=0,
                response=None,
                finish_reason=None,
                usage={},
                warnings=[],
                error="synthetic backend failure",
            )

    summary = t2va.run_t2va_shadow(
        inventory, FailedBackend(), verify_source_media=False
    )
    assert summary.failed_count == 20
