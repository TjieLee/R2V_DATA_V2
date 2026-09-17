from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_v3_post_mask_production import _write_rows
from tests.test_v3_post_mask_production import case as _case_fixture
from tests.test_v3_post_mask_runtime import _accepted_factory, _ready

case = _case_fixture


def api():
    return importlib.import_module("tools.run_v3_post_mask_visual")


def accepted(config):
    from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND

    return replace(
        config,
        remove=replace(
            config.remove,
            enabled=True,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
        ),
        reference_edit=replace(
            config.reference_edit,
            enabled=True,
            python_executable=config.run_root.parent / "python",
            code_root=config.run_root.parent / "boogu",
            model_path=config.run_root.parent.parent / "models/boogu",
        ),
        reference_integrity=replace(config.reference_integrity, enabled=True),
        qwen=replace(
            config.qwen,
            reference_edit_judge=config.qwen.candidate_judge,
            reference_integrity_judge=config.qwen.candidate_judge,
        ),
        subject_attribute_gme=replace(config.subject_attribute_gme, enabled=False),
    )


def campaign(case, *, strict_closure=False):
    config, root, _shard = case
    config = accepted(config)
    base = config.run_root.parent / "accepted.yaml"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(json.dumps(asdict(config), default=str))
    clips = config.dataset_json.parent / "videos"
    clips.mkdir(exist_ok=True)
    names = [p.name for p in (root / "parts").glob("shard-*.jsonl")]
    if not strict_closure:
        annotations = root.parent / "entity_annotations/parts"
        annotations.mkdir(parents=True, exist_ok=True)
        for name in names:
            (annotations / name).touch()
    with patch.object(
        api(),
        "EXPECTED_PRODUCTION_SHARDS",
        384 if strict_closure else len(names),
        create=True,
    ):
        return api().prepare_campaign(
            config,
            base_config_path=base,
            entity_mask_root=root,
            post_mask_root=config.run_root.parent / "campaign",
            source_jsonl=config.dataset_json,
            clips_root=clips,
        )


@pytest.mark.parametrize(
    "problem", [None, "stage2_missing", "stage1_missing", "replaced"]
)
def test_first_campaign_requires_exact_filename_closure(case, monkeypatch, problem):
    annotations = case[1].parent / "entity_annotations/parts"
    annotations.mkdir(parents=True)
    for i in range(384):
        name = f"shard-{i * 100:09d}-{i * 100 + 99:09d}.jsonl"
        (annotations / name).touch()
        (case[2].parent / name).touch()
    last = "shard-000038300-000038399.jsonl"
    if problem == "stage1_missing":
        (annotations / last).unlink()
    elif problem == "stage2_missing":
        (case[2].parent / last).unlink()
    elif problem == "replaced":
        (case[2].parent / last).rename(
            case[2].parent / "shard-000038400-000038499.jsonl"
        )
    original = Path.open

    def no_body(path, *args, **kwargs):
        assert path.parent not in (annotations, case[2].parent), (
            "canonical content read"
        )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_body)
    if problem is None:
        assert len(campaign(case, strict_closure=True).shards) == 384
    else:
        with pytest.raises(ValueError, match="closure 384/384"):
            campaign(case, strict_closure=True)
        assert not (case[0].run_root.parent / "campaign").exists()
        assert not list(case[0].run_root.parent.parent.rglob("source.yaml"))


@pytest.mark.parametrize("codes", [(0, 0), (17, 0)])
def test_naturally_exited_children_never_signalled(monkeypatch, codes):
    children = [
        subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({code})"], start_new_session=True
        )
        for code in codes
    ]
    for child in children:
        child.wait()
    monkeypatch.setattr(
        api().os, "killpg", lambda *a: pytest.fail("signalled exited process")
    )
    if codes[0]:
        with pytest.raises(RuntimeError, match="17"):
            api().supervise_children(children, poll_seconds=0.01)
    else:
        api().supervise_children(children, poll_seconds=0.01)
    api().terminate_owned_children(children)


def test_384_shards_assignment_complete_disjoint_and_balanced():
    shards = tuple(range(384))
    assignments = [
        api().assigned_shards(shards, rank, 7, slot, 2)
        for rank in range(7)
        for slot in range(2)
    ]
    assert sorted(x for group in assignments for x in group) == list(range(384))
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1
    assert assignments[0][:3] == (0, 14, 28)


@pytest.mark.parametrize("groups", ["", "4:4", "4:5,5:6", "4", "-1:2", "04:5,4:6"])
def test_gpu_groups_reject_overlap_and_invalid_values(groups):
    with pytest.raises(ValueError):
        api().parse_gpu_groups(groups)


@pytest.mark.parametrize(
    "rank,world,slot,count", [(-1, 2, 0, 2), (2, 2, 0, 2), (0, 0, 0, 2), (0, 1, 2, 2)]
)
def test_invalid_worker_topology(rank, world, slot, count):
    with pytest.raises(ValueError):
        api().assigned_shards((0,), rank, world, slot, count)


def test_production_config_rejected_not_silently_normalized(case):
    with pytest.raises(ValueError, match="remove"):
        api().validate_production_config(case[0])
    config = accepted(case[0])
    api().validate_production_config(config)
    with pytest.raises(ValueError, match="GME"):
        api().validate_production_config(
            replace(
                config,
                subject_attribute_gme=replace(
                    config.subject_attribute_gme, enabled=True
                ),
            )
        )


def test_campaign_resume_ignores_runtime_devices_and_new_base_bytes(case):
    _write_rows(case, [_ready(case)])
    first = campaign(case)
    descriptor = first.source_yaml.read_bytes()
    metadata = json.loads(descriptor)
    assert (
        metadata["base_config_fingerprint"]
        == api().load_config(Path(metadata["base_config_path"])).fingerprint()
    )
    config = accepted(case[0])
    config = replace(
        config,
        runtime=replace(config.runtime, cpu_workers=3, qwen_max_inflight=7),
        sam3=replace(config.sam3, device="cuda:7"),
    )
    second = campaign((config, *case[1:]))
    assert first.identity == second.identity
    assert first.paths(case[2]) == second.paths(case[2])
    assert second.source_yaml.read_bytes() == descriptor
    changed = replace(config, pair=replace(config.pair, max_candidates_per_entity=2))
    with pytest.raises(ValueError, match="identity"):
        campaign((changed, *case[1:]))


@pytest.mark.parametrize(
    "section", ["pair", "reference_edit", "reference_integrity", "instruction"]
)
def test_required_production_stages_reject_disabled(case, section):
    config = accepted(case[0])
    config = replace(
        config, **{section: replace(getattr(config, section), enabled=False)}
    )
    with pytest.raises(ValueError, match=section):
        api().validate_production_config(config)


def test_campaign_rejects_source_mismatch_before_writing(case):
    config = accepted(case[0])
    other = config.dataset_json.parent / "other.jsonl"
    other.write_text("")
    clips = config.dataset_json.parent / "videos"
    clips.mkdir(exist_ok=True)
    output = config.run_root.parent / "campaign"
    with pytest.raises(ValueError, match="dataset_json"):
        api().prepare_campaign(
            config,
            base_config_path=Path("unused.yaml"),
            entity_mask_root=case[1],
            post_mask_root=output,
            source_jsonl=other,
            clips_root=clips,
        )
    assert not output.exists()


def test_campaign_and_completed_poll_never_read_canonical_contents(case, monkeypatch):
    _write_rows(case, [])
    second = case[2].with_name("shard-000000100-000000199.jsonl")
    second.write_text("")
    canonical = {case[2], second}
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path not in canonical, "canonical body read at campaign/barrier"
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        c = campaign(case)
    for shard in c.shards:
        _completed(c, shard, count=0)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        assert campaign(case).identity == c.identity
        assert all(api().completed_receipt(c, shard) for shard in c.shards)


def test_only_claimed_shard_hashes_and_mutation_fails_resume(case, monkeypatch):
    from r2v_data_v2.v3 import post_mask_production as production

    _write_rows(case, [])
    second = case[2].with_name("shard-000000100-000000199.jsonl")
    second.write_text("")
    c = campaign(case)
    reads = []
    original = production._digest

    def observe(path):
        if path in c.shards:
            reads.append(path)
        return original(path)

    monkeypatch.setattr(production, "_digest", observe)
    paths = c.paths(case[2])
    config = accepted(case[0])
    production.initialize_shard(config, paths, git_commit="test")
    assert reads == [case[2]]
    case[2].write_text("\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        production.initialize_shard(config, paths, git_commit="test")
    assert set(reads) == {case[2]}


def test_legacy_campaign_receipts_resume_without_rehash_or_provenance_rewrite(
    case, monkeypatch
):
    _write_rows(case, [])
    c = campaign(case)
    legacy = json.loads(json.dumps(c.identity))
    legacy["shards"][case[2].stem]["shard_sha256"] = api().file_digest(case[2])
    receipt = c.root / "campaign.json"
    receipt.write_text(json.dumps(legacy))
    descriptor = json.loads(c.source_yaml.read_text())
    descriptor["campaign_sha256"] = api().digest(legacy)
    c.source_yaml.write_text(json.dumps(descriptor))
    historical = c.source_yaml.read_bytes()
    _completed(c, case[2], count=0)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path != case[2], "legacy resume read canonical body"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    resumed = campaign(case)
    assert resumed.identity == legacy
    assert resumed.source_yaml.read_bytes() == historical
    assert api().completed_receipt(resumed, case[2])


def test_shared_shard_lock_prevents_other_process_mutation(tmp_path):
    lock = tmp_path / "shard.lock"
    with api().exclusive_lock(lock):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl,sys; f=open(sys.argv[1],'a'); "
                    "fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
                ),
                str(lock),
            ],
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
    with api().exclusive_lock(lock):
        pass


def test_unclaimed_legacy_shard_mutation_fails_before_runner(case, monkeypatch):
    _write_rows(case, [])
    c = campaign(case)
    c.identity["shards"][case[2].stem]["shard_sha256"] = api().file_digest(case[2])
    case[2].write_text("\n")
    assert not c.paths(case[2]).identity_path.exists()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="identity mismatch"):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="new",
            event_callback=lambda e: None,
            runner=lambda *a, **kw: pytest.fail("untrusted canonical reached runner"),
        )


def test_worker_skip_completed_and_actual_runtime_passthrough(case, monkeypatch):
    _write_rows(case, [_ready(case)])
    c = campaign(case)
    events, execution_seen = [], []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    runtime = replace(case[0].runtime, qwen_max_inflight=7, cpu_workers=3)
    from r2v_data_v2.v3.post_mask_runtime import run_post_mask_shard

    def runner(config, **kwargs):
        execution_seen.append(kwargs["execution"])
        return run_post_mask_shard(
            config, **kwargs, adapter_factory=_accepted_factory([])
        )

    api().run_worker(
        c,
        accepted(case[0]),
        runtime=runtime,
        rank=0,
        world_size=1,
        slot=0,
        groups=(("4", "5"),),
        git_commit="test",
        runner=runner,
        event_callback=events.append,
        heartbeat_seconds=0.02,
    )
    assert execution_seen[0].runtime == runtime
    assert execution_seen[0].sam_gpu == "4" and execution_seen[0].boogu_gpu == "5"
    api().run_worker(
        c,
        accepted(case[0]),
        runtime=runtime,
        rank=0,
        world_size=1,
        slot=0,
        groups=(("4", "5"),),
        git_commit="new-head",
        runner=runner,
        event_callback=events.append,
    )
    assert len(execution_seen) == 1
    assert events[-1]["event"] == "post_mask_worker_completed"


def test_rank_zero_waits_for_all_markers_and_nonzero_rank_never_compacts(case):
    _write_rows(case, [_ready(case)])
    c = campaign(case)
    calls = []
    for rank in (0, 1):
        with pytest.raises(TimeoutError):
            api().wait_for_global(
                c,
                rank=rank,
                timeout_seconds=0.02,
                poll_seconds=0.005,
                compactor=lambda **kw: calls.append(kw),
                event_callback=lambda e: None,
            )
    assert calls == []
    assert not c.compact_marker.exists()


def test_nonzero_child_waits_for_healthy_sibling_before_reporting_failure(
    tmp_path, monkeypatch
):
    finished = tmp_path / "healthy-finished"
    failed = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(17)"], start_new_session=True
    )
    waiting = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time,pathlib,sys;time.sleep(.15);pathlib.Path(sys.argv[1]).write_text('done')",
            str(finished),
        ],
        start_new_session=True,
    )
    monkeypatch.setattr(
        api().os, "killpg", lambda *a: pytest.fail("normal completion signalled")
    )
    with pytest.raises(RuntimeError, match="17"):
        api().supervise_children([failed, waiting], poll_seconds=0.01)
    assert waiting.returncode == 0
    assert finished.read_text() == "done"


def test_external_abort_still_cleans_owned_children(monkeypatch):
    from types import SimpleNamespace

    children = [
        subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
        )
        for _ in range(2)
    ]

    def abort(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        api(), "time", SimpleNamespace(sleep=abort, monotonic=api().time.monotonic)
    )
    with pytest.raises(KeyboardInterrupt):
        api().supervise_children(children)
    assert all(child.poll() is not None for child in children)


def test_one_click_script_executes_repo_python_and_preserves_exit_status(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".venv/bin").mkdir(parents=True)
    python = repo / ".venv/bin/python"
    python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$PWD" "$RANK/$WORLD_SIZE" "$POST_MASK_GPU_GROUPS" "$OMP_NUM_THREADS" "$PYTHONPATH" "$@"\nexit 19\n'
    )
    python.chmod(0o755)
    import os

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_v3_post_mask_visual_cluster.sh"
    )
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "POST_MASK_REPO": str(repo),
            "RANK": "2",
            "WORLD_SIZE": "3",
            "PYTHONPATH": "retained",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 19
    assert result.stdout.splitlines()[:5] == [
        str(repo),
        "2/3",
        "4:5,6:7",
        "1",
        "/mnt/workspace/litengjie/data/vendor/sam3:retained",
    ]
    assert "tools/run_v3_post_mask_visual.py" in result.stdout
    assert "set -o pipefail" not in script.read_text()
    assert "exit 1" not in script.read_text()


def _completed(c, shard, count=1):
    """Small real-shaped receipts, for barrier tests without image work."""
    from r2v_data_v2.reconciliation import write_json_atomic

    paths = c.paths(shard)
    identity = {
        "shard_identity": {
            **c.identity["shards"][shard.stem],
            "shard_sha256": api().file_digest(shard),
        },
        "git_commit": "historical",
        "clip_count": count,
        "clip_sources_sha256": "a" * 64,
    }
    write_json_atomic(paths.identity_path, identity["shard_identity"])
    write_json_atomic(paths.run_root / "run.json", {"git_commit": "historical"})
    write_json_atomic(paths.state_root / "export_identity.json", identity)
    write_json_atomic(
        paths.export_root / "dataset.json",
        {
            "git_commit": "historical",
            "sample_count": count,
            "config_hash": identity["shard_identity"]["config_hash"],
        },
    )
    (paths.export_root / "samples.jsonl").write_text("")
    write_json_atomic(
        paths.state_root / "completed.json",
        {"identity": identity, "sample_count": count},
    )


def _fake_compactor(c, calls):
    def compact(**kwargs):
        assert kwargs == {
            "shards_root": c.exports_root / "shards",
            "output_root": c.exports_root,
            "source_jsonl": c.source_jsonl,
            "source_yaml": c.source_yaml,
            "runs_root": c.runs_root,
        }
        calls.append(kwargs)
        from r2v_data_v2.reconciliation import write_json_atomic

        (c.exports_root / "samples.jsonl").write_text("fixture\n")
        receipts = {p.stem: api().completed_receipt(c, p) for p in c.shards}
        write_json_atomic(
            c.exports_root / "catalog.json",
            {
                "source_jsonl": str(c.source_jsonl),
                "total_samples": sum(v["sample_count"] for v in receipts.values()),
                "samples_jsonl_sha256": api().file_digest(
                    c.exports_root / "samples.jsonl"
                ),
                "included_shards": [
                    {
                        "shard_id": p.stem,
                        "git_commit": receipts[p.stem]["identity"]["git_commit"],
                        "sample_count": receipts[p.stem]["sample_count"],
                        "config_hash": c.identity["shards"][p.stem]["config_hash"],
                    }
                    for p in c.shards
                ],
            },
        )

    return compact


def test_rank0_barrier_needs_all_384_and_publishes_once(case):
    _write_rows(case, [])
    for i in range(1, 384):
        (case[2].parent / f"shard-{i * 100:09d}-{i * 100 + 99:09d}.jsonl").write_text(
            ""
        )
    c = campaign(case)
    for shard in c.shards[:-1]:
        _completed(c, shard)
    calls = []
    with pytest.raises(TimeoutError):
        api().wait_for_global(
            c,
            rank=0,
            timeout_seconds=0.001,
            compactor=_fake_compactor(c, calls),
            event_callback=lambda e: None,
        )
    assert not calls
    _completed(c, c.shards[-1])
    events = []
    for rank in (0, 0, 1):
        api().wait_for_global(
            c,
            rank=rank,
            compactor=_fake_compactor(c, calls),
            event_callback=events.append,
        )
    assert len(calls) == 1
    assert len([e for e in events if e["event"] == "post_mask_global_completed"]) == 3
    assert c.compact_marker.is_file()


def test_non_rank0_waits_until_rank0_compaction(case):
    import threading

    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    waiting, finished = threading.Event(), threading.Event()
    errors = []

    def follower():
        try:
            api().wait_for_global(
                c,
                rank=1,
                timeout_seconds=2,
                poll_seconds=0.01,
                compactor=lambda **kw: pytest.fail("rank1 compacted"),
                event_callback=lambda e: waiting.set(),
            )
            finished.set()
        except BaseException as exc:  # noqa: BLE001 - convey thread assertions to test
            errors.append(exc)

    thread = threading.Thread(target=follower)
    thread.start()
    assert waiting.wait(1) and not finished.is_set()
    api().wait_for_global(
        c, rank=0, compactor=_fake_compactor(c, []), event_callback=lambda e: None
    )
    thread.join(2)
    assert not errors and finished.is_set()


def test_compaction_failure_never_emits_global_success(case):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    events = []

    def fail(**kw):
        (c.exports_root / "samples.jsonl").write_text("partial")
        raise OSError("injected publication failure")

    with pytest.raises(OSError):
        api().wait_for_global(c, rank=0, compactor=fail, event_callback=events.append)
    assert not c.compact_marker.exists()
    assert "post_mask_global_completed" not in [e["event"] for e in events]
    api().wait_for_global(
        c, rank=0, compactor=_fake_compactor(c, []), event_callback=events.append
    )
    assert events[-1]["event"] == "post_mask_global_completed"


def test_global_marker_rejects_stale_completed_shard_and_partial_publication(case):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    api().wait_for_global(
        c, rank=0, compactor=_fake_compactor(c, []), event_callback=lambda e: None
    )
    (c.exports_root / "samples.jsonl").write_text("partial")
    with pytest.raises(ValueError, match="compacted output"):
        api().wait_for_global(c, rank=1, event_callback=lambda e: None)
    _completed(c, c.shards[0], count=2)
    with pytest.raises(ValueError, match="identity mismatch"):
        api().wait_for_global(c, rank=1, event_callback=lambda e: None)


def test_real_worker_export_global_compaction_h3_reader(case, monkeypatch):
    from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory
    from r2v_data_v2.v3.post_mask_runtime import run_post_mask_shard

    row = _ready(case)
    path = case[1] / row["artifact_root"] / "run/clips/clip-0/clip.json"
    value = json.loads(path.read_text())
    value["source"]["metadata"] = {
        "source_relative_video_path": "movies/show/clip-0.mp4",
        "source_relative_source_video_path": "movies/show/parent.mp4",
    }
    path.write_text(json.dumps(value))
    _write_rows(case, [row])
    before = {str(p): p.read_bytes() for p in case[1].rglob("*") if p.is_file()}
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    def runner(config, **kwargs):
        return run_post_mask_shard(
            config, **kwargs, adapter_factory=_accepted_factory([])
        )

    api().run_worker(
        c,
        accepted(case[0]),
        runtime=case[0].runtime,
        rank=0,
        world_size=1,
        slot=0,
        groups=(("4", "5"),),
        git_commit="test",
        runner=runner,
        event_callback=lambda e: None,
    )
    api().wait_for_global(c, rank=0, event_callback=lambda e: None)
    inventory = load_visual_production_inventory(
        visual_production_root=c.exports_root, visual_runs_root=c.runs_root
    )
    assert inventory.visual_input_mode == "compacted_production"
    assert [clip.identity.clip_uid for clip in inventory.clips] == ["clip-0"]
    assert {str(p): p.read_bytes() for p in case[1].rglob("*") if p.is_file()} == before


def test_worker_heartbeat_keeps_current_shard_stage_while_runner_blocks(
    case, monkeypatch
):
    import threading

    _write_rows(case, [])
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    heartbeats, arrived = [], threading.Event()

    def callback(event):
        if event["event"] == "post_mask_heartbeat":
            heartbeats.append(event)
            arrived.set()

    def runner(config, **kw):
        kw["event_callback"]({"event": "post_mask_stage_started", "stage": "remove"})
        assert arrived.wait(1)
        raise RuntimeError("injected stop")

    with pytest.raises(RuntimeError):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="test",
            runner=runner,
            event_callback=callback,
            heartbeat_seconds=0.01,
        )
    assert heartbeats[0]["current_shard"] == case[2].stem
    assert heartbeats[0]["current_stage"] == "remove"
    assert heartbeats[0]["total_shards"] == 1


def test_launch_requires_explicit_base_and_uses_actual_runtime_overrides(
    case, monkeypatch
):
    monkeypatch.delenv("POST_MASK_BASE_CONFIG", raising=False)
    args = api().parser().parse_args([])
    with pytest.raises(ValueError, match="accepted production YAML"):
        api().load_launch(args)
    _write_rows(case, [])
    base = case[0].run_root.parent / "accepted.yaml"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("fixture")
    clips = case[0].dataset_json.parent / "videos"
    clips.mkdir(exist_ok=True)
    monkeypatch.setattr(api(), "load_config", lambda path: accepted(case[0]))
    args = (
        api()
        .parser()
        .parse_args(
            [
                "--base-config",
                str(base),
                "--entity-mask-root",
                str(case[1]),
                "--source-jsonl",
                str(case[0].dataset_json),
                "--clips-root",
                str(clips),
                "--qwen-max-inflight",
                "7",
                "--cpu-workers",
                "3",
            ]
        )
    )
    annotations = case[1].parent / "entity_annotations/parts"
    annotations.mkdir(parents=True)
    for i in range(384):
        name = f"shard-{i * 100:09d}-{i * 100 + 99:09d}.jsonl"
        (annotations / name).touch()
        (case[2].parent / name).touch()
    config, _campaign, groups = api().load_launch(args)
    assert config.runtime.qwen_max_inflight == 7 and config.runtime.cpu_workers == 3
    assert (
        json.loads(_campaign.source_yaml.read_text())["base_config_fingerprint"]
        == accepted(case[0]).fingerprint()
    )
    assert groups == (("4", "5"), ("6", "7"))


def test_worker_entrypoint_rejects_unisolated_environment_before_library_import():
    import os

    worker = Path(__file__).resolve().parents[1] / "tools/run_v3_post_mask_worker.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import builtins,runpy,sys; original=builtins.__import__; "
                "builtins.__import__=lambda name,*a,**kw: (_ for _ in ()).throw(RuntimeError('EARLY_IMPORT')) "
                "if name.startswith(('r2v_data_v2','tools.')) else original(name,*a,**kw); "
                "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            ),
            str(worker),
            "--worker-slot",
            "0",
            "--sam-gpu",
            "4",
            "--boogu-gpu",
            "5",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "7"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "EARLY_IMPORT" not in result.stderr
    assert "isolated" in result.stderr


def test_successful_compaction_before_marker_crash_is_reused(case):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    compact = _fake_compactor(c, [])
    compact(
        shards_root=c.exports_root / "shards",
        output_root=c.exports_root,
        source_jsonl=c.source_jsonl,
        source_yaml=c.source_yaml,
        runs_root=c.runs_root,
    )
    api().wait_for_global(
        c,
        rank=0,
        compactor=lambda **kw: pytest.fail("repeated complete compaction"),
        event_callback=lambda e: None,
    )
    assert c.compact_marker.exists()


def test_corrupt_complete_marker_fails_closed_before_worker_models(case, monkeypatch):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    marker = c.paths(c.shards[0]).state_root / "completed.json"
    value = json.loads(marker.read_text())
    value["identity"]["shard_identity"]["shard_sha256"] = "wrong"
    marker.write_text(json.dumps(value))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="identity mismatch"):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="new",
            event_callback=lambda e: None,
            runner=lambda *a, **kw: pytest.fail("models called"),
        )


def test_launcher_spawns_isolated_repo_workers_and_no_false_success(
    case, monkeypatch, capsys
):
    _write_rows(case, [])
    c = campaign(case)
    monkeypatch.setattr(
        api(),
        "load_launch",
        lambda args: (accepted(case[0]), c, (("4", "5"), ("6", "7"))),
    )
    real_popen = subprocess.Popen
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return real_popen([sys.executable, "-c", "raise SystemExit(13)"], **kwargs)

    monkeypatch.setattr(api().subprocess, "Popen", popen)
    monkeypatch.setattr(
        api(),
        "wait_for_global",
        lambda *a, **kw: pytest.fail("global wait after failure"),
    )
    assert api().main([]) == 2
    assert [kw["env"]["CUDA_VISIBLE_DEVICES"] for _, kw in commands] == ["4", "6"]
    assert all(kw["start_new_session"] for _, kw in commands)
    assert all(
        cmd[0] == str(api().REPOSITORY_ROOT / ".venv/bin/python") for cmd, _ in commands
    )
    assert commands[1][0][2:8] == [
        "--worker-slot",
        "1",
        "--sam-gpu",
        "6",
        "--boogu-gpu",
        "7",
    ]
    output = capsys.readouterr().out
    assert "post_mask_failed" in output and "post_mask_global_completed" not in output


def test_worker_retryable_result_is_nonzero_and_never_worker_complete(
    case, monkeypatch
):
    from r2v_data_v2.v3.post_mask_runtime import ShardResult

    _write_rows(case, [])
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    events = []
    with pytest.raises(RuntimeError, match="retryable"):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="test",
            event_callback=events.append,
            runner=lambda *a, **kw: ShardResult(False, 1, 0, 0, 0, ("bad",), {}),
        )
    assert "post_mask_worker_completed" not in [e["event"] for e in events]
    assert not c.compact_marker.exists()


def test_incomplete_shard_continues_later_shard_without_retry(case, monkeypatch):
    from r2v_data_v2.v3.post_mask_runtime import ShardResult, run_post_mask_shard

    _write_rows(case, [])
    later = case[2].with_name("shard-000000100-000000199.jsonl")
    later.write_text("")
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    attempted, events = [], []

    def runner(config, **kwargs):
        shard = kwargs["paths"].shard_path
        attempted.append(shard)
        if shard == case[2]:
            return ShardResult(False, 1, 0, 0, 0, ("retryable-clip",), {})
        return run_post_mask_shard(
            config, **kwargs, adapter_factory=_accepted_factory([])
        )

    with pytest.raises(RuntimeError, match="retryable"):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="test",
            runner=runner,
            event_callback=events.append,
        )
    assert attempted == [case[2], later]
    assert api().completed_receipt(c, later) is not None
    assert events[-1]["event"] == "post_mask_worker_incomplete"
    assert events[-1]["unresolved_shards"] == [case[2].stem]
    assert events[-1]["completed_shards"] == 1
    assert not any(e["event"] == "post_mask_worker_completed" for e in events)
    with pytest.raises(TimeoutError):
        api().wait_for_global(
            c,
            rank=0,
            timeout_seconds=0.01,
            poll_seconds=0.005,
            compactor=lambda **kwargs: pytest.fail("unresolved compaction"),
            event_callback=events.append,
        )
    assert not any(e["event"] == "post_mask_global_completed" for e in events)


def test_worker_hydration_counts_visible_during_blocking_phase(case, monkeypatch):
    import threading

    _write_rows(case, [])
    c = campaign(case)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    arrived = threading.Event()
    seen = []

    def callback(event):
        if event["event"] == "post_mask_heartbeat":
            seen.append(event)
            arrived.set()

    def runner(config, **kw):
        kw["event_callback"](
            {
                "event": "post_mask_hydrate_completed",
                "ready": 5,
                "excluded": 3,
                "corrupt": 1,
            }
        )
        assert arrived.wait(1)
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        api().run_worker(
            c,
            accepted(case[0]),
            runtime=case[0].runtime,
            rank=0,
            world_size=1,
            slot=0,
            groups=(("4", "5"),),
            git_commit="test",
            runner=runner,
            event_callback=callback,
            heartbeat_seconds=0.01,
        )
    assert (seen[0]["ready"], seen[0]["excluded"], seen[0]["corrupt"]) == (5, 3, 1)
    assert seen[0]["current_stage"] == "hydrate"


def test_zero_sample_compaction_is_not_h3_ready_success(case):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0], count=0)
    events = []
    with pytest.raises(ValueError, match="zero accepted"):
        api().wait_for_global(
            c, rank=0, compactor=_fake_compactor(c, []), event_callback=events.append
        )
    assert not c.compact_marker.exists()
    assert "post_mask_global_completed" not in [e["event"] for e in events]


def test_successful_compaction_removes_only_owned_abandoned_scratch(case):
    _write_rows(case, [])
    c = campaign(case)
    _completed(c, c.shards[0])
    scratch = [
        c.exports_root / (".samples.jsonl.tmp-" + "a" * 32),
        c.exports_root / ".production-sample-ids-ab_cd123.sqlite3",
        c.exports_root / ".production-sample-ids-ab_cd123.sqlite3-journal",
    ]
    preserved = [
        c.exports_root / "notes.tmp",
        c.exports_root / ".samples.jsonl.tmp-unknown",
    ]
    for path in scratch + preserved:
        path.write_text("owned fixture")
    api().wait_for_global(
        c, rank=0, compactor=_fake_compactor(c, []), event_callback=lambda e: None
    )
    assert not any(path.exists() for path in scratch)
    assert all(path.is_file() for path in preserved)
    assert c.paths(c.shards[0]).run_root.joinpath("run.json").is_file()
