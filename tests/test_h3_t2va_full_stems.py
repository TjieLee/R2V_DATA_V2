from pathlib import Path

import numpy as np
import soundfile as sf

from tests import test_h3_auk_speech_shadow as fixtures

ffmpeg = fixtures.ffmpeg
setup = fixtures.setup


def test_sam_model_loaded_once_and_inventory_bound_per_request(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from tests.test_h3_sam_audio_stem_shadow import _canonical_fixture, _configuration

    manifest, _, _ = _canonical_fixture(tmp_path)
    config = _configuration(tmp_path)
    inventory = sam.build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest, model_configuration=config
    )
    loads = []

    class Backend:
        def __init__(self, configuration):
            self.configuration = configuration

        def _load(self):
            loads.append(self.configuration)

    monkeypatch.setattr(sam, "OfficialSAMAudioBackend", Backend)
    configuration = {"model": config.model_dump(mode="json")}
    with stems.SAMWorker(configuration) as worker:
        backend = worker.backend
        for name in ("a", "b", "c"):
            path = tmp_path / f"{name}.json"
            path.write_text(inventory.model_dump_json())
            worker.bind_request({**configuration, "inventory_path": str(path)})
            assert worker.inventory == inventory
            assert worker.backend is backend
        assert loads == [config]


def test_sam_model_passes_finish_before_cpu_canonicalization(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from r2v_data_v2.h3.t2va_full_workers import _output_files
    from tests.test_h3_sam_audio_stem_shadow import (
        _SAM,
        _canonical_fixture,
        _Canonicalizer,
        _configuration,
        _Media,
    )

    manifest, _, _ = _canonical_fixture(tmp_path)
    config = _configuration(tmp_path)
    inventory = sam.build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest, model_configuration=config,
        route="music_first",
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(inventory.model_dump_json())
    worker = stems.SAMWorker({
        "model": config.model_dump(mode="json"),
        "inventory_path": str(inventory_path),
    })
    backend = _SAM(config)
    worker.backend = backend
    canonicalizer = _Canonicalizer()
    monkeypatch.setattr(sam, "FFmpegStemCanonicalizer", lambda **_kwargs: canonicalizer)
    monkeypatch.setattr(stems, "FFmpegAudioMediaBackend", lambda **_kwargs: _Media())
    job = {"source": inventory.jobs[0].model_dump(mode="json")}
    output = tmp_path / "worker-output"
    outcomes = worker.infer(job, output)
    assert len(backend.calls) == 2
    assert canonicalizer.sources == []
    result = worker.finalize(job, output, outcomes)
    record = sam.SAMAudioStemRecord.model_validate(result["record"])
    assert record.model_call_count == 2
    assert len(canonicalizer.sources) == 3
    assert [item.stem_type for item in record.stems] == ["music", "speech", "sfx"]
    assert _output_files(output, known_hashes=worker.output_digests(result)) == (
        _output_files(output)
    )


def test_auk_replay_publishes_full_inventory(setup, tmp_path, ffmpeg):
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from tests.test_h3_auk_speech_shadow import FakeBackend

    attempted = []

    def execute(stage_root, jobs, **kwargs):
        backend = FakeBackend(setup.model_configuration)
        result = {}
        for item in jobs:
            attempted.append(item["job_id"])
            raw = stage_root / (item["job_id"] + ".wav")
            raw.parent.mkdir(parents=True, exist_ok=True)
            response = backend.generate(auk.AukJob.model_validate(item["source"]), raw)
            result[item["job_id"]] = {
                "status": "ready",
                "result": {"raw_path": str(raw), "response": response},
            }
        return result

    stems.run_auk(
        setup,
        tmp_path / "work",
        ["0", "1"],
        eligible={"a", "c"},
        ffmpeg=ffmpeg,
        execute=execute,
    )
    _, records, _ = auk.load_auk_shadow(
        auk.auk_stage_root(Path(setup.audio_production_root), setup.shadow_run_id)
    )
    assert attempted == [uid for uid in setup.clip_uids if uid in {"a", "c"}]
    assert [r.clip_uid for r in records] == setup.clip_uids
    assert {r.clip_uid: r.status for r in records} == {
        "a": "ready",
        "b": "failed",
        "c": "ready",
    }


def test_auk_cpu_finalizer_canonicalizes_one_time(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import t2va_full_stems as stems

    raw = tmp_path / "speech.wav"
    sf.write(raw, np.zeros((3200, 2), dtype=np.int16), 32000, subtype="PCM_16")
    calls = []

    def canonicalize(source, output, frames, *, ffmpeg):
        calls.append((source, frames, ffmpeg))
        sf.write(output, np.zeros((frames, 2), dtype=np.int16), 32000, subtype="PCM_16")
        return 0

    monkeypatch.setattr(auk, "canonicalize_speech", canonicalize)
    worker = object.__new__(stems.AukWorker)
    worker.configuration = {"ffmpeg": "unused"}
    job = {
        "source": {
            "clip_uid": "a",
            "clip_display_path": "a",
            "target_video_path": str(tmp_path / "a.mp4"),
            "target_video_sha256": "0" * 64,
            "source_audio_path": str(tmp_path / "audio.flac"),
            "source_audio_sha256": "0" * 64,
            "source_frame_count": 3200,
            "source_duration_seconds": 0.1,
        }
    }
    result = worker.finalize(job, tmp_path, {"raw_path": str(raw), "response": {}})
    assert result["canonical_adjustment_samples"] == 0
    assert result["raw_duration_delta_seconds"] == 0
    assert calls == [(raw, 3200, "unused")]


def test_prepared_resolve_publishes_without_second_resolve(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import resolved_audio_stems as resolved
    from r2v_data_v2.h3.t2va_production import atomic_json

    root = tmp_path / "audio"
    destination = resolved.downstream_stem_root(root, "pilot")
    destination.parent.mkdir(parents=True)
    job = auk.AukJob(
        clip_uid="a",
        clip_display_path="a",
        target_video_path=str(tmp_path / "a.mp4"),
        target_video_sha256="0" * 64,
        source_audio_path=str(tmp_path / "a.flac"),
        source_audio_sha256="1" * 64,
        source_frame_count=3200,
        source_duration_seconds=0.1,
    )
    inventory = auk._signed(
        resolved.ResolvedStemInventory,
        {
            "source_sam_root": str(destination.parent / "separation"),
            "source_auk_root": str(destination.parent / "auk_speech_v1"),
            "source_sam_inventory_fingerprint": "0" * 64,
            "source_auk_inventory_fingerprint": "1" * 64,
            "source_sam_records_sha256": "2" * 64,
            "source_auk_records_sha256": "3" * 64,
            "source_canonical_audio_manifest_path": str(tmp_path / "clips.jsonl"),
            "source_canonical_audio_manifest_sha256": "4" * 64,
            "clip_uids": ["a"],
            "jobs": [job.model_dump(mode="json")],
        },
        "inventory_fingerprint",
    )
    record = auk._signed(
        resolved.ResolvedStemRecord,
        {
            "clip_uid": "a",
            "inventory_fingerprint": inventory.inventory_fingerprint,
            "source_sam_record_fingerprint": "5" * 64,
            "source_auk_record_fingerprint": "6" * 64,
            "status": "failed",
            "failure_reason": "upstream failure",
        },
        "record_fingerprint",
    )
    summary = resolved.ResolvedStemSummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        clip_uids=["a"], ready_count=0, failed_count=1,
    )
    prepared = tmp_path / "prepared.json"
    atomic_json(prepared, {
        "inventory": inventory.model_dump(mode="json"),
        "records": [record.model_dump(mode="json")],
        "summary": summary.model_dump(mode="json"),
    })
    monkeypatch.setattr(resolved, "_resolve", lambda *_: (_ for _ in ()).throw(AssertionError("replayed")))
    assert resolved.resolve_audio_stems(
        audio_production_root=root, shadow_run_id="pilot", prepared_path=prepared
    ) == summary
    assert resolved.load_resolved_stems(destination) == (inventory, [record], summary)
