import json

import pytest

from r2v_data_v2.h3 import t2va_full_production as full
from r2v_data_v2.h3 import t2va_production as production
from tests import test_h3_auk_speech_shadow as auk_fixtures
from tests.test_h3_t2va_shadow import Audio, shot_manifest

setup = auk_fixtures.setup
ffmpeg = auk_fixtures.ffmpeg


@pytest.mark.parametrize("failure_count", [1, 3])
def test_raw_video_to_snapshot_and_retry_without_repeat_models(
    tmp_path, monkeypatch, setup, ffmpeg, failure_count
):
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    from r2v_data_v2.h3 import audio_backends
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_speech as speech
    from r2v_data_v2.h3 import t2va_full_workers as workers
    from r2v_data_v2.h3.ta2va_shadow import TA2VAProfileBackend
    from tests import test_h3_t2va_shadow as shared
    from tests.test_h3_sam_audio_stem_shadow import (
        _Canonicalizer,
        _configuration,
        _Diarization,
        _Qwen,
    )
    from tests.test_h3_ta2va_shadow import ProfileClient

    manifest = shot_manifest(tmp_path)
    root = tmp_path / "full"
    config = shared.config(tmp_path)
    calls = Counter()

    class Semantic(shared.T2VAMimoBackend):
        def annotate(self, job, request_fingerprint):
            calls[("t2va", job.clip_uid)] += 1
            backend = shared.T2VAMimoBackend(
                config, client=shared.Client([shared.draft_for(job).model_dump_json()])
            )
            return backend.annotate(job, request_fingerprint)

    class Media(Audio):
        def __init__(self, **kwargs):
            pass

        def probe_audio_file(self, path):
            info = sf.info(path)
            return audio_backends.AudioFileProbe(
                sample_rate_hz=info.samplerate,
                channels=info.channels,
                frame_count=info.frames,
                duration_seconds=info.duration,
                format_name=info.format.lower(),
            )

    class FakeSAM:
        def __init__(self, configuration):
            self.configuration = configuration

        def separate(
            self, *, clip_uid, source_audio_path, prompt, target_path, residual_path
        ):
            info = sf.info(source_audio_path)
            for path in (target_path, residual_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    path,
                    np.zeros(info.frames),
                    info.samplerate,
                    format="WAV",
                    subtype="PCM_16",
                )
            return sam.SAMAudioSeparationResult(
                target_path=str(target_path),
                residual_path=str(residual_path),
                verification_state="unverified",
                runtime_seconds=0.1,
                backend_metadata={"fake": True},
            )

    monkeypatch.setattr(audio_backends, "FFmpegAudioMediaBackend", Media)
    monkeypatch.setattr(
        speech,
        "_diar_configuration",
        lambda: {
            "provenance": _Diarization.provenance.model_dump(mode="json"),
            "environment": {},
        },
    )
    monkeypatch.setattr(
        speech,
        "_asr_configuration",
        lambda: {
            "configuration": _Qwen.configuration.model_dump(mode="json"),
            "environment": {},
        },
    )
    cache, failed = {}, set(shared.shot_ids(tmp_path)[:failure_count])

    def execute(stage_root, jobs, *, factory, configuration, **kwargs):
        result = {}
        for job in jobs:
            key = (factory, job["job_id"])
            if key in cache:
                result[job["job_id"]] = cache[key]
                continue
            calls[key] += 1
            if factory.endswith(":SAMWorker") and job["job_id"] in failed:
                result[job["job_id"]] = {
                    "status": "failed",
                    "failure_reason": "synthetic separation failure",
                    "result": None,
                }
                continue
            folder = stage_root / job["job_id"]
            folder.mkdir(parents=True, exist_ok=True)
            if factory.endswith(":SAMWorker"):
                inventory = sam.SAMAudioStemInventory.model_validate_json(
                    Path(configuration["inventory_path"]).read_text()
                )
                record = sam._separate_one_route(
                    inventory=inventory,
                    job=sam.SAMAudioStemJob.model_validate(job["source"]),
                    route="music_first",
                    output_root=folder,
                    backend=FakeSAM(inventory.model_configuration),
                    canonicalizer=_Canonicalizer(),
                    raw_probe_backend=Media(),
                )
                value = {
                    "record": record.model_dump(mode="json"),
                    "output_root": str(folder),
                }
            elif factory.endswith(":AukWorker"):
                raw = folder / "speech.wav"
                response = auk_fixtures.FakeBackend(setup.model_configuration).generate(
                    auk.AukJob.model_validate(job["source"]), raw
                )
                value = {"raw_path": str(raw), "response": response}
            elif factory.endswith(":diarizen_worker"):
                value = {
                    "segments": [{"start_time": 0, "end_time": 1, "speaker_label": "A"}]
                }
            else:
                value = {"text": "Hello.", "language": "English"}
            cache[key] = {"status": "ready", "result": value, "failure_reason": None}
            result[job["job_id"]] = cache[key]
        return result

    monkeypatch.setattr(workers, "execute_stage", execute)
    pipeline = full.FullPipeline(
        root=root,
        index=production.build_source_index(manifest, root),
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0", "1"],
        sam_configuration=_configuration(tmp_path),
        auk_configuration=setup.model_configuration,
        backend=Semantic(config, client=shared.Client([])),
        profiles=TA2VAProfileBackend(config, client=ProfileClient()),
        allow_unverified=True,
        ffmpeg=ffmpeg,
    )
    full.run_assigned_shards(root, [0], pipeline)
    assert sum(v for (k, _), v in calls.items() if k == "t2va") == 3 - failure_count
    failed.clear()
    full.run_assigned_shards(root, [0], pipeline)
    assert sum(v for (k, _), v in calls.items() if k == "t2va") == 3
    before = calls.copy()
    full.run_assigned_shards(root, [0], pipeline)
    assert calls == before
    snapshot = production.build_snapshot(root, "ready")
    assert len(list(production.complete_rows(snapshot / "t2va.jsonl"))) == 3
    assert len(list(production.complete_rows(snapshot / "ta2va_full_audio.jsonl"))) == 3


def test_full_supervisor_stage_order(tmp_path, capsys):
    events = []

    class Pipeline:
        def stage(self, name, shard_id):
            events.append((shard_id, name))
            return {"ready": 2, "failed": 1, "clip_uids": ["inventory-only"]}

    result = full.run_assigned_shards(tmp_path, [59, 12], Pipeline())
    assert "inventory-only" not in capsys.readouterr().out
    assert "inventory-only" not in json.dumps(result)
    assert events == [(shard, stage) for shard in (59, 12) for stage in full.STAGES]
    assert full.STAGES == (
        "canonical",
        "sam",
        "auk",
        "resolve",
        "diarizen",
        "asr",
        "mimo",
    )


def test_shard_selection_preserves_bad_row_positions(tmp_path):
    manifest = shot_manifest(tmp_path)
    lines = manifest.read_bytes().splitlines(keepends=True)
    manifest.write_bytes(lines[0] + b"not json\n" + b"[]\n" + lines[2])
    root = tmp_path / "out"
    selection = full.shard_selection(
        root, production.build_source_index(manifest, root), 0, tmp_path, tmp_path
    )
    assert [s.source_index for s in selection.shots] == [0, 3]
    assert [r["source_index"] for r in selection.excluded_rows] == [1, 2]


def test_full_cli_dry_run_does_not_require_models(tmp_path):
    from tools.run_h3_t2va_full_production import main

    manifest = shot_manifest(tmp_path)
    result = main(
        [
            "--shot-manifest",
            str(manifest),
            "--clips-root",
            str(tmp_path),
            "--source-videos-root",
            str(tmp_path),
            "--production-root",
            str(tmp_path / "out"),
            "--media-root",
            str(tmp_path),
            "--shards",
            "0",
            "--dry-run",
        ]
    )
    assert result["shards"] == [0]
    assert result["request_workers"] == 1
    assert result["gpu_ids"] == [str(i) for i in range(8)]
    assert result["model_call_count"] == 0


def test_full_supervisor_worker_crash_stops_next_stage(tmp_path):
    from r2v_data_v2.h3.t2va_full_workers import WorkerStageError

    events = []

    class Pipeline:
        def stage(self, name, shard_id):
            events.append(name)
            if name == "auk":
                raise WorkerStageError("worker died")
            return {}

    with pytest.raises(WorkerStageError):
        full.run_assigned_shards(tmp_path, [0, 1], Pipeline())
    assert events == ["canonical", "sam", "auk"]


def test_sam_full_inventory_publication(tmp_path):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems
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
        canonical_audio_manifest_path=manifest, model_configuration=config
    )

    def execute(stage_root, jobs, **kwargs):
        backend = _SAM(config)
        results = {}
        for job in jobs:
            folder = stage_root / job["job_id"]
            folder.mkdir(parents=True)
            record = sam._separate_one_route(
                inventory=inventory,
                job=sam.SAMAudioStemJob.model_validate(job["source"]),
                route="music_first",
                output_root=folder,
                backend=backend,
                canonicalizer=_Canonicalizer(),
                raw_probe_backend=_Media(),
            )
            results[job["job_id"]] = {
                "status": "ready",
                "result": {
                    "record": record.model_dump(mode="json"),
                    "output_root": str(folder),
                },
            }
        return results

    output = tmp_path / "shadow/separation"
    stems.run_sam(inventory, output, tmp_path / "work", ["0", "1"], execute=execute)
    loaded, records, summary = sam.load_stem_shadow(output)
    assert loaded == inventory
    assert summary.record_count == 1
    assert records[0].calls[0].prompt == "music soundtrack"
    assert records[0].calls[1].prompt == "human voices"
    assert all(
        __import__("pathlib").Path(s.canonical_stem_path).is_relative_to(output)
        for s in records[0].stems
    )


def test_bootstrap_from_index_and_resume(tmp_path):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    assert [s.source_index for s in selection.shots] == [0, 1, 2]
    calls = []

    class Media(Audio):
        def materialize_full_audio(self, **kwargs):
            calls.append(kwargs["clip_uid"])
            return super().materialize_full_audio(**kwargs)

    audio = full.bootstrap_audio(
        root / "shards" / production.shard_name(0), selection, Media()
    )
    assert len(calls) == 3
    rows = list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))
    assert [r["clip_uid"] for r in rows] == [s.clip_uid for s in selection.shots]
    assert all(r["subject_reference_count"] == 0 for r in rows)
    full.bootstrap_audio(root / "shards" / production.shard_name(0), selection, Media())
    assert len(calls) == 3
    inventory = json.loads((audio / "diarization/inventory.json").read_text())
    assert inventory["source_inventory_kind"] == "jea_shot_manifest"
    assert all(t["visual_references"] == [] for t in inventory["targets"])


def test_all_canonical_failures_are_local(tmp_path):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    selection = full.shard_selection(
        root, production.build_source_index(manifest, root), 0, tmp_path, tmp_path
    )

    class Failed(Audio):
        def materialize_full_audio(self, **kwargs):
            raise ValueError("no audio stream")

    audio = full.bootstrap_audio(
        root / "shards" / production.shard_name(0), selection, Failed()
    )
    assert list(production.complete_rows(audio / "audio/canonical_clips.jsonl")) == []


def test_bootstrap_isolates_failures_and_retries(tmp_path):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    selection = full.shard_selection(
        root, production.build_source_index(manifest, root), 0, tmp_path, tmp_path
    )
    calls = []

    class Media(Audio):
        def materialize_full_audio(self, **kwargs):
            calls.append(kwargs["clip_uid"])
            if kwargs["clip_uid"] == selection.shots[1].clip_uid:
                raise ValueError("bad audio")
            super().materialize_full_audio(**kwargs)

    shard = root / "shards" / production.shard_name(0)
    audio = full.bootstrap_audio(shard, selection, Media())
    assert (
        len(list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))) == 2
    )
    full.bootstrap_audio(shard, selection, Media())
    assert len(calls) == 4
    full.bootstrap_audio(shard, selection, Audio())
    assert (
        len(list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))) == 3
    )
