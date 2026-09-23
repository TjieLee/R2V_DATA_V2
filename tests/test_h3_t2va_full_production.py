import json

import pytest

from r2v_data_v2.h3 import t2va_full_production as full
from r2v_data_v2.h3 import t2va_production as production
from tests import test_h3_auk_speech_shadow as auk_fixtures
from tests.test_h3_t2va_shadow import Audio, shot_manifest

setup = auk_fixtures.setup
ffmpeg = auk_fixtures.ffmpeg


@pytest.mark.parametrize("failure_count", [1, 3])
def test_raw_video_to_snapshot_keeps_terminal_failures_without_repeat_models(
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
    shard = root / "shards" / production.shard_name(0)
    assert (shard / "COMPLETE").exists()
    failed.clear()
    full.run_assigned_shards(root, [0], pipeline)
    assert sum(v for (k, _), v in calls.items() if k == "t2va") == 3 - failure_count
    before = calls.copy()
    full.run_assigned_shards(root, [0], pipeline)
    assert calls == before
    snapshot = production.build_snapshot(root, "ready")
    assert len(list(production.complete_rows(snapshot / "t2va.jsonl"))) == 3 - failure_count
    assert len(list(production.complete_rows(snapshot / "ta2va_full_audio.jsonl"))) == 3 - failure_count


def test_resume_first_schedule_uses_only_shard_directory_metadata(tmp_path):
    assigned = [7, 2, 9, 4, 1, 6]

    complete = tmp_path / "shards" / production.shard_name(2)
    complete.mkdir(parents=True)
    (complete / "COMPLETE").write_text("not-json")

    for shard_id in (9, 1, 6):
        shard = tmp_path / "shards" / production.shard_name(shard_id)
        shard.mkdir(parents=True)
        # Existing content is deliberately irrelevant to scheduling.
        (shard / "arbitrary.partial").write_text("not-json")

    order, summary = full.resume_first_assigned_shards(tmp_path, assigned)

    # Relative order is stable inside each class. No marker content is parsed.
    assert order == [9, 1, 6, 7, 4]
    assert summary == {
        "unfinished": 3,
        "fresh": 2,
        "complete_skipped": 1,
    }


def test_resume_first_happens_after_rank_partition(tmp_path):
    assigned = [
        production.shard_order(start=0, end=19, seed=42, rank=rank, world_size=3)
        for rank in range(3)
    ]
    assert set(assigned[0]).isdisjoint(assigned[1])
    assert set(assigned[0]).isdisjoint(assigned[2])
    assert set(assigned[1]).isdisjoint(assigned[2])

    # Mark one shard in each already-owned slice unfinished. Reordering cannot
    # move a shard across rank ownership.
    for shard_ids in assigned:
        state = (
            tmp_path
            / "shards"
            / production.shard_name(shard_ids[-1])
            / "stage_state"
        )
        state.mkdir(parents=True)
        (state / "asr.json").touch()

    prioritized = [
        full.resume_first_assigned_shards(tmp_path, shard_ids)[0]
        for shard_ids in assigned
    ]
    for before, after in zip(assigned, prioritized, strict=True):
        assert after[0] == before[-1]
        assert set(after) == set(before)


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


def test_node_lifetime_does_not_start_persistent_upstream_pools(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from r2v_data_v2.h3 import t2va_full_worker_pool as pools

    class ForbiddenManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("full production must use stage-local upstream workers")

    monkeypatch.setattr(pools, "PersistentPoolManager", ForbiddenManager)
    config = SimpleNamespace(model_dump=lambda **kw: {"model": "frozen"})
    pipeline = full.FullPipeline(
        root=tmp_path,
        index={},
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0", "5"],
        sam_configuration=config,
        auk_configuration=config,
        backend=None,
        profiles=None,
    )
    with pipeline:
        assert pipeline.pools is None
    assert pipeline.pools is None


def test_downstream_evidence_ignores_runtime_provenance_for_resume():
    from r2v_data_v2.h3.t2va_full_downstream import _model_agnostic_evidence

    job = {
        "clip_uid": "clip",
        "clip_display_path": "episode/clip",
        "target_video_path": "/video.mp4",
        "target_video_sha256": "1" * 64,
        "target_duration_seconds": 5.0,
        "speech_facts": [
            {
                "segment_id": "s1",
                "source_speaker_cluster": "speaker",
                "start_time": 0.0,
                "end_time": 1.0,
                "language": "English",
                "text": "hello",
            }
        ],
        "audio_evidence": {
            "resolved_root": "/resolved",
            "resolved_record_fingerprint": "2" * 64,
            "full_audio_path": "/full.wav",
            "full_audio_sha256": "3" * 64,
            "speech_path": "/speech.wav",
            "speech_sha256": "4" * 64,
            "music_path": "/music.wav",
            "music_sha256": "5" * 64,
            "sfx_path": "/sfx.wav",
            "sfx_sha256": "6" * 64,
        },
        "upstream_failure": None,
    }
    source = {
        "source_index": 1,
        "source_row_sha256": "a" * 64,
        "clip_uid": "clip",
        "video": "/video.mp4",
    }
    old = {
        "population": "source",
        "source": source,
        "policy": {
            "version": "full-downstream-per-clip-v1",
            "backend": {"model": "mimo-v2.5"},
            "profiles": {"model": "mimo-v2.5"},
            "allow_unverified": True,
        },
        "dependencies": {
            "job": job,
            "resolved": {"clip_uid": "clip", "record_fingerprint": "7" * 64},
            "diarization_backend": {"model_identifier": "old-diar"},
            "sam": {"configuration": {"model": "old-sam"}},
            "auk": {"configuration": {"model": "old-auk"}},
            "asr": [{"model_identifier": "old-asr", "package": "old"}],
        },
    }
    current_job = json.loads(json.dumps(job))
    current_job["audio_evidence"]["resolved_record_fingerprint"] = "8" * 64
    current = {
        "population": "source",
        "source": source,
        "policy": {
            "version": "full-downstream-per-clip-v3-content-only",
            "allow_unverified": True,
        },
        "dependencies": {"job": current_job},
    }

    assert _model_agnostic_evidence(old) == _model_agnostic_evidence(current)


def test_downstream_dependency_sidecar_is_audit_only(tmp_path):
    from r2v_data_v2.h3.t2va_full_downstream import _preflight

    shard = tmp_path / "shard"
    sidecars = shard / "downstream_dependencies"
    sidecars.mkdir(parents=True)
    row = {
        "source_index": 1,
        "source_row_sha256": "a" * 64,
        "clip_uid": "clip",
        "video": "/video.mp4",
    }
    current = {
        "population": "new-source",
        "source": row,
        "policy": {"version": "current", "allow_unverified": True},
        "dependencies": {"job": {"clip_uid": "clip", "speech_facts": []}},
    }
    production.atomic_json(
        sidecars / "clip.json",
        {
            "population": "old-source",
            "source": row,
            "policy": {"version": "old", "allow_unverified": True},
            "dependencies": {"sam": {"configuration": {"model": "old"}}},
        },
    )

    class Processor:
        def evidence(self, _row):
            return current

        def identity(self, _row):
            return "current-identity"

    _preflight(shard, [row], Processor())

    assert json.loads((sidecars / "clip.json").read_text()) == current


def test_preflight_never_reopens_complete_exports(tmp_path):
    from r2v_data_v2.h3.t2va_full_downstream import _preflight

    shard = tmp_path / "shard"
    row = {"source_index": 0, "clip_uid": "clip", "video": "/video.mp4"}
    production.atomic_json(shard / "sources.json", [row])
    production.atomic_json(shard / "COMPLETE", {"source_count": 1})
    export = production.training_row(row["video"], "caption")
    production.append_row(shard / "exports/t2va.jsonl", export)

    class Processor:
        def evidence(self, _row):
            return {"source": row, "policy": {}}

        def identity(self, _row):
            return "identity"

    _preflight(shard, [row], Processor())
    assert (shard / "COMPLETE").exists()
    assert list(production.complete_rows(shard / "exports/t2va.jsonl")) == [export]
    assert not (shard / "exports/t2va.jsonl.partial").exists()


def test_downstream_terminal_fast_path_skips_preparation(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import t2va_full_downstream as downstream
    from tests.test_h3_t2va_production import _historical_states

    rows, shard = _historical_states(
        tmp_path, [("ready", "failed"), ("failed", "skipped")]
    )
    monkeypatch.setattr(
        production,
        "prepare_shard",
        lambda *args, **kwargs: pytest.fail("terminal shard prepared upstream media"),
    )
    states = downstream.run_downstream(
        tmp_path, {}, 0, tmp_path / "audio", tmp_path, tmp_path,
        backend=None, profiles=None,
    )
    assert set(states) == {row["clip_uid"] for row in rows}
    assert (shard / "COMPLETE").exists()


def test_full_launcher_filters_historical_terminal_shard_before_workers(tmp_path):
    from tests.test_h3_t2va_production import _historical_states

    _, shard = _historical_states(tmp_path, [("failed", "skipped")])
    remaining = full.seal_terminal_assigned_shards(tmp_path, [0, 1])
    assert remaining == [1]
    assert (shard / "COMPLETE").exists()


def test_full_cli_terminal_shard_needs_no_model_dependencies(tmp_path, monkeypatch):
    from tests.test_h3_t2va_production import _historical_states
    from tools import run_h3_t2va_full_production as cli

    root = tmp_path / "production"
    _, shard = _historical_states(root, [("failed", "skipped")])
    production.atomic_json(
        root / "manifests/source_index.json",
        {"shard_size": production.SHARD_SIZE, "shards": [{}], "source_record_count": 1},
    )
    monkeypatch.setattr(
        production,
        "build_source_index",
        lambda *_: pytest.fail("terminal shard revalidated source manifest SHA"),
    )
    result = cli.main([
        "--shot-manifest", str(tmp_path / "shots.jsonl"),
        "--clips-root", str(tmp_path / "clips"),
        "--source-videos-root", str(tmp_path / "videos"),
        "--media-root", str(tmp_path),
        "--production-root", str(root),
        "--shards", "0",
        "--mimo-sglang", str(tmp_path / "missing-sglang"),
        "--mimo-checkpoint", str(tmp_path / "missing-checkpoint"),
    ])
    assert result == {}
    assert (shard / "COMPLETE").exists()


def test_mimo_stage_is_wrapped_by_per_shard_lifecycle(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from r2v_data_v2.h3 import t2va_full_downstream

    events = []

    class Lifecycle:
        @contextmanager
        def stage(self, shard_id):
            events.append(("enter", shard_id))
            try:
                yield
            finally:
                events.append(("exit", shard_id))

    def run_downstream(*args, **kwargs):
        events.append(("run", args[2]))
        return {
            "clip": {
                "t2va_status": "ready",
                "ta2va_status": "ready",
            }
        }

    monkeypatch.setattr(t2va_full_downstream, "run_downstream", run_downstream)
    config = SimpleNamespace(model_dump=lambda **kw: {})
    pipeline = full.FullPipeline(
        root=tmp_path,
        index={},
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0"],
        sam_configuration=config,
        auk_configuration=config,
        backend=None,
        profiles=None,
        mimo_lifecycle=Lifecycle(),
    )
    pipeline.prepared[7] = full.PreparedShard(tmp_path / "audio", {"ready": 1})
    result = pipeline.stage("mimo", 7)
    assert events == [("enter", 7), ("run", 7), ("exit", 7)]
    assert result["t2va_ready"] == result["ta2va_ready"] == 1


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


@pytest.mark.parametrize("overlong_duration", [200.000001, 200.1])
def test_shard_selection_skips_over_200_seconds_before_video_hash(
    tmp_path, monkeypatch, overlong_duration
):
    manifest = shot_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row, duration in zip(rows, (199.9, 200.0, overlong_duration), strict=True):
        row["duration"] = duration
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    original_hash = full.sha256_file

    def no_rejected_video_hash(path):
        if __import__("pathlib").Path(path).name == "movie_3.mp4":
            pytest.fail("over-200-second video was hashed")
        return original_hash(path)

    monkeypatch.setattr(full, "sha256_file", no_rejected_video_hash)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    assert [shot.duration_seconds for shot in selection.shots] == [199.9, 200.0]
    assert selection.excluded_rows == [
        {"source_index": 2, "reason": "clip_duration_over_200s"}
    ]


def test_cached_shard_selection_migrates_long_clip_to_terminal_skip(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from r2v_data_v2.h3 import t2va_shadow
    from r2v_data_v2.h3.t2va_source import select_t2va_shots

    manifest = shot_manifest(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()[:2]]
    rows[0]["duration"] = 100.0
    rows[1]["duration"] = 314.0
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    shard_manifest = production.materialize_shard(index, 0, root)
    initial = select_t2va_shots(
        shard_manifest, clips_root=tmp_path, source_videos_root=tmp_path
    )
    assert len(initial.shots) == 2
    cached = root / "shards" / production.shard_name(0) / "source/selection.json"
    production.atomic_json(cached, initial.model_dump(mode="json"))
    original_hash = full.sha256_file

    def no_video_rehash(path):
        if __import__("pathlib").Path(path).suffix == ".mp4":
            pytest.fail("cached selection migration rehashed a video")
        return original_hash(path)

    monkeypatch.setattr(full, "sha256_file", no_video_rehash)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    assert [shot.source_index for shot in selection.shots] == [0]
    assert selection.excluded_rows == [
        {"source_index": 1, "reason": "clip_duration_over_200s"}
    ]
    assert json.loads(cached.read_text())["shots"] == [
        selection.shots[0].model_dump(mode="json")
    ]

    monkeypatch.setattr(
        t2va_shadow,
        "build_t2va_inventory",
        lambda **_kwargs: SimpleNamespace(jobs=[]),
    )
    monkeypatch.setattr(t2va_shadow, "write_json", lambda *_args: None)
    projected, _ = production.prepare_shard(
        root,
        index,
        0,
        audio_production_root=tmp_path / "audio",
        audio_shadow_run_id="test",
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        backend=None,
    )
    skipped = projected[1]
    assert skipped["upstream_failure"] == "clip_duration_over_200s"
    assert "preparation_error" not in skipped

    class Processor:
        def identity(self, _row):
            return "test-identity"

        def process(self, *_args):
            pytest.fail("duration-excluded clip reached a model stage")

    state = production.process_shard(root, 0, [skipped], Processor())[skipped["clip_uid"]]
    assert (state["t2va_status"], state["ta2va_status"]) == ("skipped", "skipped")
    assert state["failure_stage"] == "upstream"
    assert state["failure_reason"] == "clip_duration_over_200s"


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
    assert result["canonical_workers"] == 16
    assert result["mimo_model"] == "mimo-v2.5"
    assert result["gpu_ids"] == [str(i) for i in range(8)]
    assert result["model_call_count"] == 0
    single = main(
        [
            "--shot-manifest", str(manifest),
            "--clips-root", str(tmp_path),
            "--source-videos-root", str(tmp_path),
            "--production-root", str(tmp_path / "out"),
            "--media-root", str(tmp_path),
            "--shards", "0", "--dry-run",
            "--mimo-model", "mimo-v2.6-flash-rl",
            "--mimo-call-mode", "single",
        ]
    )
    assert single["mimo_model"] == "mimo-v2.6-flash-rl"
    assert single["mimo_call_mode"] == "single"


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


def test_shard_selection_resume_reuses_cached_video_hashes(tmp_path, monkeypatch):
    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    first = full.shard_selection(root, index, 0, tmp_path, tmp_path)

    original = full.sha256_file

    def no_video_rehash(path):
        path = __import__("pathlib").Path(path)
        if path.suffix == ".mp4":
            pytest.fail("resume rehashed cached target video")
        return original(path)

    monkeypatch.setattr(full, "sha256_file", no_video_rehash)
    resumed = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    assert resumed == first


def test_bootstrap_resume_does_not_rehash_cached_media(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import t2va_source

    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    shard = root / "shards" / production.shard_name(0)
    audio = full.bootstrap_audio(shard, selection, Audio())

    full_hash = full.sha256_file
    source_hash = t2va_source.sha256_file

    def no_cached_media_full(path):
        path = __import__("pathlib").Path(path)
        if path.suffix in {".mp4", ".flac"}:
            pytest.fail(f"resume rehashed cached media: {path}")
        return full_hash(path)

    def no_cached_media_source(path):
        path = __import__("pathlib").Path(path)
        if path.suffix in {".mp4", ".flac"}:
            pytest.fail(f"resume rehashed cached media: {path}")
        return source_hash(path)

    monkeypatch.setattr(full, "sha256_file", no_cached_media_full)
    monkeypatch.setattr(t2va_source, "sha256_file", no_cached_media_source)
    resumed = full.bootstrap_audio(shard, selection, Audio())
    assert resumed == audio


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


def test_parallel_canonical_is_bounded_ordered_and_failure_isolated(tmp_path):
    import threading
    import time

    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    selection = full.shard_selection(
        root, production.build_source_index(manifest, root), 0, tmp_path, tmp_path
    )
    lock = threading.Lock()
    active = maximum = 0
    finished = []
    ids = [s.clip_uid for s in selection.shots]

    class Media(Audio):
        def materialize_full_audio(self, **kwargs):
            nonlocal active, maximum
            uid = kwargs["clip_uid"]
            with lock:
                active += 1
                maximum = max(active, maximum)
            try:
                time.sleep(0.1 if uid == ids[0] else 0.01)
                if uid == ids[2]:
                    raise ValueError("bad one")
                super().materialize_full_audio(**kwargs)
                finished.append(uid)
            finally:
                with lock:
                    active -= 1

    audio = full.bootstrap_audio(
        root / "shards" / production.shard_name(0),
        selection,
        Media(),
        canonical_workers=2,
    )
    assert 1 < maximum <= 2
    assert finished == [ids[1], ids[0]]
    rows = list(production.complete_rows(audio / "audio/canonical_clips.jsonl"))
    assert [r["clip_uid"] for r in rows] == ids[:2]


def test_canonical_workers_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="canonical workers"):
        full.bootstrap_audio(tmp_path, None, None, canonical_workers=0)


def test_canonical_consumption_refreshes_recovered_inputs_under_ownership(
    tmp_path, monkeypatch
):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems

    manifest = shot_manifest(tmp_path)
    root = tmp_path / "out"
    index = production.build_source_index(manifest, root)
    selection = full.shard_selection(root, index, 0, tmp_path, tmp_path)
    shard = root / "shards" / production.shard_name(0)
    audio_root = full.bootstrap_audio(shard, selection, Audio())
    pipeline = full.FullPipeline(
        root=root,
        index=index,
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        gpu_ids=["0"],
        sam_configuration=None,
        auk_configuration=None,
        backend=None,
        profiles=None,
    )
    # The previous preparation failed; a competing preparation has since
    # published the now-ready canonical clips before consumer lock acquisition.
    pipeline.prefetch = object()
    pipeline.prepared[0] = full.PreparedShard(
        audio_root, {"ready": 0, "failed": 3, "skipped": 0}
    )
    monkeypatch.setattr(sam, "build_sam_audio_stem_inventory", lambda **kw: object())
    monkeypatch.setattr(stems, "run_sam", lambda *args, **kw: {"ran": True})
    with production.file_lock(shard / "invocation.lock"):
        assert pipeline.stage("canonical", 0) == {"ready": 3, "failed": 0, "skipped": 0}
        assert pipeline.stage("sam", 0) == {"ran": True}
