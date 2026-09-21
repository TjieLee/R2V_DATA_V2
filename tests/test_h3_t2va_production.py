import json
from collections import Counter

import pytest

from r2v_data_v2.h3 import t2va_production as production
from tests import test_h3_t2va_shadow as shared

finalized = shared.finalized
setup = shared.setup
ffmpeg = shared.ffmpeg
job = shared.job


def test_shard_59_and_schedule():
    assert production.SHARD_SIZE == 2000
    assert production.shard_bounds(0) == (0, 1999)
    assert production.shard_bounds(1) == (2000, 3999)
    assert production.shard_bounds(59) == (118000, 119999)
    assert production.shard_name(59) == "shard-000118000-000119999"
    schedules = [
        production.shard_order(start=0, end=379, seed=20260917, rank=r, world_size=4)
        for r in range(4)
    ]
    assert len(set().union(*map(set, schedules))) == 380
    assert sum(map(len, schedules)) == 380
    assert schedules[0] == production.shard_order(
        start=0, end=379, seed=20260917, rank=0, world_size=4
    )
    assert production.shard_order(explicit=[59, 12, 87], rank=99, world_size=1) == [
        59,
        12,
        87,
    ]


def test_index_and_seek_preserve_raw_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    with source.open("wb") as handle:
        for i in range(20003):
            handle.write((json.dumps({"index": i}) + "\n").encode())
    root = tmp_path / "output"
    index = production.build_source_index(source, root)
    assert index["source_record_count"] == 20003
    assert len(index["shards"]) == 11
    shard = production.materialize_shard(index, 1, root)
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    assert [r["index"] for r in rows] == list(range(2000, 4000))
    assert production.build_source_index(source, root) == index
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="source"):
        production.build_source_index(source, root)


def test_invalid_rows_do_not_shift_shards(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"not json\n" + b"{}\n" * 1999 + b'{"last":true}\n')
    root = tmp_path / "output"
    index = production.build_source_index(source, root)
    assert (
        production.materialize_shard(index, 1, root).read_bytes() == b'{"last":true}\n'
    )


def test_old_10k_index_is_rejected_without_mutation(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")
    root = tmp_path / "output"
    index = production.build_source_index(source, root)
    index["shard_size"] = 10000
    path = root / "manifests/source_index.json"
    production.atomic_json(path, index)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="source index shard size mismatch"):
        production.build_source_index(source, root)
    assert path.read_bytes() == before


class FakeProcessor:
    def __init__(self):
        self.calls = Counter()
        self.fail = set()
        self.interrupt = None

    def identity(self, row):
        return str(row["clip_uid"])

    def process(self, stage, row, temporary, destination):
        key = (row["clip_uid"], stage)
        self.calls[key] += 1
        if key == self.interrupt:
            raise KeyboardInterrupt()
        if key in self.fail:
            raise ValueError("synthetic sample failure")
        (temporary / "result.txt").write_text(str(key))
        task = "t2va" if stage == "t2va" else "ta2va_full_audio"
        return {
            "exports": {task: production.training_row(row["video"], str(key))},
            "model_call_count": 1,
        }


def sample_rows(n=3):
    return [
        {"source_index": i, "clip_uid": f"clip{i}", "video": f"/video/{i}.mp4"}
        for i in range(n)
    ]


def test_failure_isolation_and_stage_resume(tmp_path):
    processor = FakeProcessor()
    processor.fail = {("clip1", "t2va"), ("clip2", "ta2va")}
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    states = production.load_states(tmp_path / "shards" / production.shard_name(0))
    assert states["clip0"]["ta2va_status"] == "ready"
    assert states["clip1"]["t2va_status"] == "failed"
    assert states["clip1"]["ta2va_status"] == "skipped"
    assert states["clip2"]["t2va_status"] == "ready"
    assert states["clip2"]["ta2va_status"] == "failed"
    processor.fail.clear()
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    assert processor.calls["clip0", "t2va"] == 1
    assert processor.calls["clip1", "t2va"] == 2
    assert processor.calls["clip2", "t2va"] == 1
    assert processor.calls["clip2", "ta2va"] == 2


def test_interrupt_and_artifact_recovery(tmp_path):
    processor = FakeProcessor()
    processor.interrupt = ("clip3", "t2va")
    with pytest.raises(KeyboardInterrupt):
        production.process_shard(
            tmp_path, 0, sample_rows(5), processor, request_workers=1
        )
    shard = tmp_path / "shards" / production.shard_name(0)
    (shard / "state.jsonl.partial").write_bytes(b'{"partial":')
    processor.interrupt = None
    production.process_shard(tmp_path, 0, sample_rows(5), processor, request_workers=1)
    assert all(processor.calls[f"clip{i}", "t2va"] == 1 for i in range(3))
    assert processor.calls["clip4", "ta2va"] == 1


def test_skipped_and_lock(tmp_path):
    processor = FakeProcessor()
    rows = sample_rows(1)
    rows[0]["upstream_failure"] = "audio_preprocessing_required"
    production.process_shard(tmp_path, 0, rows, processor)
    production.process_shard(tmp_path, 0, rows, processor)
    assert not processor.calls
    with (
        production.file_lock(tmp_path / "lock"),
        pytest.raises(production.ShardLockedError),
        production.file_lock(tmp_path / "lock"),
    ):
        pass


def test_endpoint_outage_stops_scheduling(tmp_path):
    class Offline(FakeProcessor):
        def process(self, *args):
            raise production.EndpointUnavailable("connection refused")

    with pytest.raises(production.EndpointUnavailable):
        production.process_shard(
            tmp_path, 0, sample_rows(100), Offline(), request_workers=1
        )
    assert not production.load_states(tmp_path / "shards" / production.shard_name(0))


def test_frozen_backend_connection_error_stops_production(job, tmp_path):
    backend = shared.T2VAMimoBackend(
        shared.config(tmp_path),
        client=shared.Client([ConnectionRefusedError("connection refused")]),
    )
    output = tmp_path / "stage"
    output.mkdir()
    with pytest.raises(production.EndpointUnavailable):
        production.t2va_stage(job, backend, output)
    assert json.loads((output / "raw.json").read_text())["model_call_count"] == 1


def test_incremental_exports_and_complete(tmp_path):
    processor = FakeProcessor()
    processor.fail.add(("clip1", "ta2va"))
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    shard = tmp_path / "shards" / production.shard_name(0)
    assert not (shard / "COMPLETE").exists()
    rows = list(production.complete_rows(shard / "exports/t2va.jsonl.partial"))
    assert len(rows) == 3
    assert all(set(r) == {"video", "images", "audios", "caption"} for r in rows)
    assert (
        len(
            list(
                production.complete_rows(
                    shard / "exports/ta2va_full_audio.jsonl.partial"
                )
            )
        )
        == 2
    )
    processor.fail.clear()
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    assert (shard / "COMPLETE").exists()
    assert (shard / "exports/t2va.jsonl").exists()
    assert not (shard / "exports/t2va.jsonl.partial").exists()
    assert len(list(production.complete_rows(shard / "exports/t2va.jsonl"))) == 3


def test_ready_resume_does_not_grow_journal(tmp_path):
    processor = FakeProcessor()
    production.process_shard(tmp_path, 0, sample_rows(1), processor)
    path = tmp_path / "shards" / production.shard_name(0) / "state.jsonl.partial"
    before = path.read_bytes()
    production.process_shard(tmp_path, 0, sample_rows(1), processor)
    assert path.read_bytes() == before


def test_prevalidated_processor_skips_repeated_source_hashes(
    tmp_path, finalized, monkeypatch
):
    from r2v_data_v2.h3 import ta2va_shadow as ta
    from tests.test_h3_ta2va_shadow import ProfileClient

    inventory = shared.build(finalized, tmp_path)
    job = next(item for item in inventory.jobs if not item.upstream_failure)
    backend = shared.T2VAMimoBackend(
        shared.config(tmp_path),
        client=shared.Client([shared.draft_for(job).model_dump_json()]),
    )
    profiles = ta.TA2VAProfileBackend(
        shared.config(tmp_path), client=ProfileClient()
    )
    processor = production.FrozenProductionProcessor(
        {job.clip_uid: (job, inventory)},
        backend,
        profiles,
        allow_unverified=True,
        verify_sources_per_sample=False,
    )

    monkeypatch.setattr(
        ta,
        "_verify",
        lambda *args, **kwargs: pytest.fail(
            "prevalidated processor repeated aggregate source hashes"
        ),
    )
    output = tmp_path / "fast-t2va"
    output.mkdir()
    result = processor.process(
        "t2va",
        {"clip_uid": job.clip_uid},
        output,
        output,
    )
    assert result["model_call_count"] == 2


def test_t2va_shadow_equivalence(tmp_path, finalized):
    from tests import test_h3_t2va_shadow as shared

    inventory = shared.build(finalized, tmp_path)
    drafts = [
        shared.draft_for(j).model_dump_json()
        for j in inventory.jobs
        if not j.upstream_failure
    ]
    old_backend = shared.T2VAMimoBackend(
        shared.config(tmp_path), client=shared.Client(drafts)
    )
    shared.t2va.run_t2va_shadow(inventory, old_backend)
    old_root = shared.t2va.t2va_root(finalized[0], "t2va-test")
    new_backend = shared.T2VAMimoBackend(
        shared.config(tmp_path), client=shared.Client(drafts)
    )
    for job in inventory.jobs:
        if job.upstream_failure:
            continue
        temporary = tmp_path / "new" / job.clip_uid
        temporary.mkdir(parents=True)
        result = production.t2va_stage(job, new_backend, temporary)
        assert json.loads((temporary / "core.json").read_text()) == json.loads(
            (old_root / "core" / f"{job.clip_uid}.json").read_text()
        )
        assert (
            result["exports"]["t2va"]["caption"]
            == (old_root / "prompts" / f"{job.clip_uid}.txt").read_text()
        )
        assert result["model_call_count"] == 2


def test_ta2va_shadow_equivalence(tmp_path, finalized):
    from r2v_data_v2.h3 import ta2va_shadow as ta
    from tests.test_h3_ta2va_shadow import ProfileClient

    inventory = shared.build(finalized, tmp_path)
    drafts = [
        shared.draft_for(j).model_dump_json()
        for j in inventory.jobs
        if not j.upstream_failure
    ]
    shared.t2va.run_t2va_shadow(
        inventory,
        shared.T2VAMimoBackend(shared.config(tmp_path), client=shared.Client(drafts)),
    )
    root = shared.t2va.t2va_root(finalized[0], "t2va-test")
    old = ta.run_ta2va_shadow(
        root,
        "old-ta",
        ta.TA2VAProfileBackend(shared.config(tmp_path), client=ProfileClient()),
        allow_unverified=True,
    )
    expected = ta.read_rows(old / "records.jsonl", ta.TA2VAProduct)
    source, clips, stems, _ = ta.load_source(root)
    for job, core, segments, core_hash in clips:
        destination = tmp_path / "production-ta" / job.clip_uid
        destination.mkdir(parents=True)
        result = production.ta2va_stage(
            job,
            core,
            segments,
            stems[job.clip_uid],
            source.inventory_fingerprint,
            core_hash,
            ta.TA2VAProfileBackend(shared.config(tmp_path), client=ProfileClient()),
            destination,
            destination,
            allow_unverified=True,
        )
        actual = ta.read_rows(destination / "records.jsonl", ta.TA2VAProduct)
        old_products = [p for p in expected if p.clip_uid == job.clip_uid]
        assert [(p.variant, p.prompt, p.warnings) for p in actual] == [
            (p.variant, p.prompt, p.warnings) for p in old_products
        ]
        assert result["model_call_count"] == sum(
            p.model_call_count for p in old_products
        )
        for new, previous in zip(actual, old_products, strict=True):
            assert [a.output_sha256 for a in new.assets] == [
                a.output_sha256 for a in previous.assets
            ]


@pytest.mark.parametrize("change_source", [False, True, "outage"])
def test_ta_profile_failure_and_source_mutation(tmp_path, finalized, change_source):
    from r2v_data_v2.h3 import ta2va_shadow as ta
    from tests.test_h3_ta2va_shadow import ProfileClient

    inventory = shared.build(finalized, tmp_path)
    drafts = [
        shared.draft_for(j).model_dump_json()
        for j in inventory.jobs
        if not j.upstream_failure
    ]
    shared.t2va.run_t2va_shadow(
        inventory,
        shared.T2VAMimoBackend(shared.config(tmp_path), client=shared.Client(drafts)),
    )
    root = shared.t2va.t2va_root(finalized[0], "t2va-test")
    source, clips, stems, _ = ta.load_source(root)
    job, core, segments, core_hash = next(c for c in clips if c[0].speech_facts)
    backend = ta.TA2VAProfileBackend(
        shared.config(tmp_path), client=ProfileClient(fail=not change_source)
    )
    if change_source == "outage":

        class OfflineProfile(ProfileClient):
            def create(self, **request):
                raise ConnectionRefusedError("connection refused")

        backend = ta.TA2VAProfileBackend(
            shared.config(tmp_path), client=OfflineProfile()
        )
    profile = backend.profile

    def mutate(*args):
        result = profile(*args)
        from pathlib import Path

        Path(job.audio_evidence.full_audio_path).write_bytes(b"source changed")
        return result

    if change_source is True:
        backend.profile = mutate
    output = tmp_path / "new-stage"
    output.mkdir()
    if change_source is True:
        with pytest.raises(ValueError, match="source"):
            production.ta2va_stage(
                job,
                core,
                segments,
                stems[job.clip_uid],
                source.inventory_fingerprint,
                core_hash,
                backend,
                output,
                output,
                allow_unverified=True,
            )
    else:
        error = (
            production.EndpointUnavailable
            if change_source == "outage"
            else production.PartialStageFailure
        )
        with pytest.raises(error) as failure:
            production.ta2va_stage(
                job,
                core,
                segments,
                stems[job.clip_uid],
                source.inventory_fingerprint,
                core_hash,
                backend,
                output,
                output,
                allow_unverified=True,
            )
        assert set(failure.value.result["exports"]) == {"ta2va_full_audio"}


def test_outage_retains_completed_partial_variant(tmp_path):
    class Offline(FakeProcessor):
        def process(self, stage, row, temporary, destination):
            result = super().process(stage, row, temporary, destination)
            if stage == "ta2va":
                raise production.EndpointUnavailable("connection refused", result)
            return result

    with pytest.raises(production.EndpointUnavailable):
        production.process_shard(
            tmp_path, 0, sample_rows(5), Offline(), request_workers=1
        )
    snapshot = production.build_snapshot(tmp_path, "outage")
    assert len(list(production.complete_rows(snapshot / "ta2va_full_audio.jsonl"))) == 1


def test_partial_stage_exports_survive_retry(tmp_path):
    class PartialProcessor(FakeProcessor):
        def process(self, stage, row, temporary, destination):
            result = super().process(stage, row, temporary, destination)
            if stage == "ta2va" and self.calls[row["clip_uid"], stage] == 1:
                raise production.PartialStageFailure("profile failed", result)
            return result

    processor = PartialProcessor()
    production.process_shard(tmp_path, 0, sample_rows(1), processor)
    snapshot = production.build_snapshot(tmp_path, "failed-profile")
    assert len(list(production.complete_rows(snapshot / "ta2va_full_audio.jsonl"))) == 1
    production.process_shard(tmp_path, 0, sample_rows(1), processor)
    assert processor.calls["clip0", "t2va"] == 1
    assert processor.calls["clip0", "ta2va"] == 2


def test_runner_dry_run_and_source_adapter(tmp_path, finalized):
    from tools.run_h3_t2va_production import main

    args = [
        "--shot-manifest",
        str(tmp_path / "shots_f03_motion.jsonl"),
        "--clips-root",
        str(tmp_path),
        "--source-videos-root",
        str(tmp_path),
        "--audio-production-root",
        str(finalized[0]),
        "--audio-shadow-run-id",
        finalized[1],
        "--production-root",
        str(tmp_path / "production"),
        "--base-url",
        "http://127.0.0.1:8092/v1",
        "--media-root",
        str(tmp_path),
        "--shards",
        "0",
        "--dry-run",
    ]
    # The dataset fixture is created by the existing Audio fixture.
    report = main(args)
    assert report["dry_run"] is True
    assert report["model_call_count"] == 0
    assert report["shards"] == [0]
    root = tmp_path / "production"
    assert not (root / "shards" / production.shard_name(0) / "artifacts").exists()
    index = production.build_source_index(tmp_path / "shots_f03_motion.jsonl", root)
    rows, contexts = production.prepare_shard(
        root,
        index,
        0,
        audio_production_root=finalized[0],
        audio_shadow_run_id=finalized[1],
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        backend=shared.config(tmp_path).provenance(),
    )
    expected = shared.build(finalized, tmp_path)
    assert [contexts[r["clip_uid"]][0] for r in rows] == expected.jobs
    assert [r["source_index"] for r in rows] == [0, 1, 2]


def test_launcher_dry_run(tmp_path):
    import os
    import shlex
    import subprocess
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/run_h3_t2va_production.sh"
    completed = subprocess.run(
        ["bash", str(script), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PRODUCTION_ROOT": str(tmp_path), "SHARDS": "59,12,87"},
    )
    assert completed.returncode == 0, completed.stderr
    arguments = shlex.split(completed.stdout.splitlines()[1])
    assert arguments[arguments.index("--shards") + 1] == "59,12,87"
    assert "--tp 8 --dp 2" in completed.stdout
    assert "--enable-deterministic-inference" in completed.stdout
    assert not list(tmp_path.iterdir())


def test_launcher_owned_lifecycle_without_server(tmp_path):
    import os
    import shutil
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts/run_h3_t2va_production.sh"
    shutil.copyfile(repo / "scripts/run_h3_t2va_production.sh", script)
    binary = tmp_path / "bin"
    binary.mkdir()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/activate").write_text(f'export PATH="{binary}:$PATH"\n')
    (tmp_path / "server_env.sh").write_text(
        'export http_proxy=http://proxy.invalid:3128\n'
        'export NO_PROXY=existing.internal\n'
        'export no_proxy=stale.invalid\n'
    )
    programs = {
        "python": '#!/bin/bash\n[[ "$NO_PROXY" == "127.0.0.1,localhost,::1,existing.internal" ]] || exit 91\n[[ "$no_proxy" == "$NO_PROXY" ]] || exit 92\n[[ "$http_proxy" == "http://proxy.invalid:3128" ]] || exit 93\nif [[ "$1" == "-c" ]]; then exit 0; fi\nexit 7\n',
        "curl": '#!/bin/bash\nwhile [[ ! -f "$PID_FILE" ]]; do sleep 0.01; done\nexit 0\n',
        "sglang": '#!/bin/bash\necho $$ > "$PID_FILE"\ntrap "exit 0" TERM\nwhile :; do sleep 0.05; done\n',
    }
    for name, contents in programs.items():
        path = binary / name
        path.write_text(contents)
        path.chmod(0o755)
    pid_file = tmp_path / "model.pid"
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env={
            **os.environ,
            "SGLANG_ENV": str(tmp_path),
            "PID_FILE": str(pid_file),
            "AUDIO_PRODUCTION_ROOT": str(tmp_path / "audio"),
            "AUDIO_SHADOW_RUN_ID": "test",
            "PRODUCTION_ROOT": str(tmp_path / "out"),
            "SHARDS": "0",
        },
    )
    assert result.returncode == 7, result.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert list((tmp_path / "out/logs").glob("*/production-*.log"))


def test_concurrent_state_writes(tmp_path):
    processor = FakeProcessor()
    states = production.process_shard(
        tmp_path, 0, sample_rows(40), processor, request_workers=8
    )
    assert len(states) == 40
    assert all(s["ta2va_status"] == "ready" for s in states.values())
    assert all(count == 1 for count in processor.calls.values())


def test_progress_counts_failures_and_skips(tmp_path, capsys):
    processor = FakeProcessor()
    rows = sample_rows(3)
    processor.fail.add(("clip0", "t2va"))
    rows[1]["upstream_failure"] = "missing"
    production.process_shard(
        tmp_path, 0, rows, processor, request_workers=1, progress_every=1
    )
    output = capsys.readouterr().out
    assert "processed=3/3" in output


def test_malformed_source_row_isolated(tmp_path, finalized):
    manifest = tmp_path / "shots_f03_motion.jsonl"
    with manifest.open("a") as handle:
        handle.write("not-json\nnull\n[]\n1\n")
    root = tmp_path / "production"
    index = production.build_source_index(manifest, root)
    rows, contexts = production.prepare_shard(
        root,
        index,
        0,
        audio_production_root=finalized[0],
        audio_shadow_run_id=finalized[1],
        clips_root=tmp_path,
        source_videos_root=tmp_path,
        backend=shared.config(tmp_path).provenance(),
    )
    assert len(rows) == 7
    assert len(contexts) == 3
    assert rows[-1]["preparation_error"]


def test_preparation_strategy_has_stable_identity(tmp_path, finalized, monkeypatch):
    from r2v_data_v2.h3 import t2va_shadow as frozen
    from r2v_data_v2.h3 import ta2va_shadow as ta
    from tests.test_h3_ta2va_shadow import ProfileClient

    root = tmp_path / "production"
    manifest = tmp_path / "shots_f03_motion.jsonl"
    index = production.build_source_index(manifest, root)
    config = shared.config(tmp_path)
    kwargs = {
        "audio_production_root": finalized[0],
        "audio_shadow_run_id": finalized[1],
        "clips_root": tmp_path,
        "source_videos_root": tmp_path,
        "backend": config.provenance(),
    }
    rows, contexts = production.prepare_shard(root, index, 0, **kwargs)
    backend = shared.T2VAMimoBackend(config, client=shared.Client([]))
    profiles = ta.TA2VAProfileBackend(config, client=ProfileClient())
    before = production.FrozenProductionProcessor(contexts, backend, profiles)
    build = frozen.build_t2va_inventory

    def transient(**kw):
        if kw["shot_manifest"].name.startswith("shard-"):
            raise OSError("temporary bulk read failure")
        return build(**kw)

    monkeypatch.setattr(frozen, "build_t2va_inventory", transient)
    other_rows, other_contexts = production.prepare_shard(root, index, 0, **kwargs)
    after = production.FrozenProductionProcessor(other_contexts, backend, profiles)
    assert [before.identity(r) for r in rows] == [after.identity(r) for r in other_rows]


def test_retry_after_preparation_failure(tmp_path):
    class PreparationProcessor(FakeProcessor):
        def identity(self, row):
            return str(bool(row.get("preparation_error")))

        def process(self, stage, row, temporary, destination):
            if row.get("preparation_error"):
                raise ValueError(row["preparation_error"])
            return super().process(stage, row, temporary, destination)

    rows = sample_rows(1)
    rows[0]["preparation_error"] = "temporarily unreadable"
    processor = PreparationProcessor()
    production.process_shard(tmp_path, 0, rows, processor)
    del rows[0]["preparation_error"]
    states = production.process_shard(tmp_path, 0, rows, processor)
    assert states["clip0"]["ta2va_status"] == "ready"


def test_missing_video_keeps_source_identity_for_resume(tmp_path, finalized):
    video = tmp_path / "movie_2.mp4"
    content = video.read_bytes()
    video.unlink()
    root = tmp_path / "production"
    index = production.build_source_index(tmp_path / "shots_f03_motion.jsonl", root)
    kwargs = {
        "audio_production_root": finalized[0],
        "audio_shadow_run_id": finalized[1],
        "clips_root": tmp_path,
        "source_videos_root": tmp_path,
        "backend": shared.config(tmp_path).provenance(),
    }
    rows, _ = production.prepare_shard(root, index, 0, **kwargs)
    assert rows[1]["preparation_error"]
    video.write_bytes(content)
    repaired, _ = production.prepare_shard(root, index, 0, **kwargs)
    assert rows[1]["clip_uid"] == repaired[1]["clip_uid"]
    assert rows[1]["video"] == repaired[1]["video"]


def test_cross_mount_source_identity_and_changed_content(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"{}\n")
    root = tmp_path / "out"
    index = production.build_source_index(source, root)
    index["source_identity"]["device"] = -1
    index["source_identity"]["inode"] = -1
    index["source_identity"]["ctime_ns"] = -1
    production.atomic_json(root / "manifests/source_index.json", index)

    original_sha = production._sha
    calls = []

    def counted_sha(path):
        calls.append(path)
        return original_sha(path)

    monkeypatch.setattr(production, "_sha", counted_sha)
    loaded = production.build_source_index(source, root)
    assert loaded["source_sha256"] == index["source_sha256"]
    assert calls == []

    production.materialize_shard(loaded, 0, root)
    assert calls == []

    source.write_bytes(b"{\"changed\":true}\n")
    with pytest.raises(ValueError, match="source"):
        production.build_source_index(source, root)
    assert calls


def test_runner_end_to_end_cpu(tmp_path, finalized, monkeypatch):
    from r2v_data_v2.h3 import t2va_mimo_backend, ta2va_shadow
    from tests.test_h3_ta2va_shadow import ProfileClient
    from tools.run_h3_t2va_production import main

    inventory = shared.build(finalized, tmp_path)
    client = shared.Client(
        [
            shared.draft_for(j).model_dump_json()
            for j in inventory.jobs
            if not j.upstream_failure
        ]
    )
    profile_client = ProfileClient()
    old_t2va, old_ta = (
        t2va_mimo_backend.T2VAMimoBackend,
        ta2va_shadow.TA2VAProfileBackend,
    )
    monkeypatch.setattr(
        t2va_mimo_backend,
        "T2VAMimoBackend",
        lambda config: old_t2va(config, client=client),
    )
    monkeypatch.setattr(
        ta2va_shadow,
        "TA2VAProfileBackend",
        lambda config: old_ta(config, client=profile_client),
    )
    args = [
        "--shot-manifest",
        str(tmp_path / "shots_f03_motion.jsonl"),
        "--clips-root",
        str(tmp_path),
        "--source-videos-root",
        str(tmp_path),
        "--audio-production-root",
        str(finalized[0]),
        "--audio-shadow-run-id",
        finalized[1],
        "--production-root",
        str(tmp_path / "production"),
        "--media-root",
        str(tmp_path),
        "--shards",
        "0",
        "--request-workers",
        "1",
        "--allow-unverified",
    ]
    report = main(args)
    assert report["shards"][0]["t2va_ready"] == 2
    assert report["shards"][0]["ta2va_ready"] == 2
    assert report["shards"][0]["t2va_skipped"] == 1
    calls = (len(client.calls), len(profile_client.calls))
    assert calls[0] == 4
    main(args)
    assert (len(client.calls), len(profile_client.calls)) == calls
    snapshot = production.build_snapshot(tmp_path / "production", "cpu")
    assert len(list(production.complete_rows(snapshot / "t2va.jsonl"))) == 2
