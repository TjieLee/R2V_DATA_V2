from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.v3 import pre_qwen_production as stage2
from tests.test_v3_pre_qwen_production import (
    FakeBackend,
    FakeDecoder,
    _paths,
    _row,
    _write_shard,
)
from tools import entity_mask_recovery_plan as campaign
from tools import run_v3_entity_mask_auto as production
from tools import run_v3_entity_mask_recovery as launcher
from tools import run_v3_entity_mask_recovery_worker as worker


def _fixture(tmp_path, monkeypatch, *, counts=(3, 2, 1)):
    fixture = _paths(tmp_path, monkeypatch)
    paths = [
        _write_shard(
            fixture,
            [_row(fixture, index * 10_000 + offset) for offset in range(count)],
            nominal_end=index * 10_000 + 9999,
        )
        for index, count in enumerate(counts)
    ]
    monkeypatch.setattr(
        production, "validate_entity_mask_production_config", lambda config: None
    )
    stage2.ensure_execution_identity(fixture.output_root, chunk_rows=100)
    stage2.ensure_sam3_session_reuse_identity(fixture.output_root, mode="clip_reset_v1")
    topology = production.expected_static_topology(
        input_root=fixture.input_root,
        shard_paths=paths,
        world_size=5,
        local_gpu_count=8,
    )
    stage2.write_json_atomic(
        production._static_topology_path(fixture.output_root), topology
    )
    return fixture, paths, fixture.writable / "recovery" / "plan-v1.json"


def _obtain(fixture, path, **kwargs):
    return campaign.obtain_plan(
        path,
        fixture.input_root,
        fixture.output_root,
        fixture.identity,
        rank=0,
        **kwargs,
    )


def _item(index, remaining):
    return campaign.RecoveryShard(
        name=f"shard-{index * 10_000:09d}-{index * 10_000 + 9999:09d}.jsonl",
        position=index,
        input_sha256="1" * 64,
        row_count=1000,
        estimated_durable_rows=1000 - remaining,
        estimated_remaining_rows=remaining,
        original_worker_id=4,
        original_rank=0,
        original_local_slot=4,
        original_shard_start=0,
        original_shard_end=5,
    )


def test_weighted_lpt_splits_one_original_owner_across_physical_gpus():
    items = [
        _item(index, remaining) for index, remaining in enumerate([100, 100, 50, 50])
    ]
    assigned = campaign.weighted_assignment(items, world_size=2, local_gpu_count=1)
    assert [[item.position for item in slot] for slot in assigned] == [[0, 2], [1, 3]]
    assert [
        sum(item.estimated_remaining_rows for item in slot) for slot in assigned
    ] == [150, 150]
    assert all(item.original_worker_id == 4 for slot in assigned for item in slot)
    # Independent physical nodes compute the same assignment; input order cannot affect ties.
    assert assigned == campaign.weighted_assignment(
        list(reversed(items)), world_size=2, local_gpu_count=1
    )
    assert len({item.name for slot in assigned for item in slot}) == len(items)
    five = campaign.weighted_assignment(
        [_item(i, 100) for i in range(5)], world_size=1, local_gpu_count=5
    )
    assert [len(slot) for slot in five] == [1] * 5


def test_plan_creation_excludes_canonical_and_reuses_immutable_snapshot(
    tmp_path, monkeypatch
):
    fixture, paths, plan_path = _fixture(tmp_path, monkeypatch)
    stage2.process_shard(
        paths[0],
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    first = _obtain(fixture, plan_path)
    assert [item.name for item in first.shards] == [path.name for path in paths[1:]]
    assert list(first.canonical_shard_sha256s) == [paths[0].name]
    before = plan_path.read_bytes()
    # A planned shard becomes canonical while the same campaign remains frozen.
    stage2.process_shard(
        paths[1],
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(),
        decoder=FakeDecoder(),
    )
    monkeypatch.setattr(
        campaign, "build_plan", lambda *args, **kwargs: pytest.fail("plan regenerated")
    )
    assert _obtain(fixture, plan_path) == first
    assert plan_path.read_bytes() == before
    assert first.frozen_topology["global_worker_count"] == 40


def test_plan_estimates_partial_prefix_without_mutating_torn_tail(
    tmp_path, monkeypatch
):
    fixture, paths, plan_path = _fixture(tmp_path, monkeypatch, counts=(3,))
    shard = stage2.load_annotation_shard(paths[0])
    chunk = stage2.build_execution_chunks(shard, chunk_rows=100)[0]
    result = stage2.process_execution_chunk(
        shard,
        chunk,
        output_root=fixture.output_root,
        config_identity=fixture.identity,
        backend=FakeBackend(failure=RuntimeError("temporary"), fail_on_call=2),
        decoder=FakeDecoder(),
    )
    assert result["retryable"]
    partial = Path(result["path"])
    partial.write_bytes(partial.read_bytes() + b'{"source_index":')
    before = partial.read_bytes()
    plan = _obtain(fixture, plan_path)
    assert plan.shards[0].estimated_durable_rows == 1
    assert plan.shards[0].estimated_remaining_rows == 2
    assert partial.read_bytes() == before


def test_nonzero_physical_rank_waits_for_atomic_plan_publication(tmp_path, monkeypatch):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch)
    sleeps, events = [], []
    current = 0.0

    def sleep(seconds):
        nonlocal current
        sleeps.append(seconds)
        current += seconds
        _obtain(fixture, plan_path)

    result = campaign.obtain_plan(
        plan_path,
        fixture.input_root,
        fixture.output_root,
        fixture.identity,
        rank=2,
        sleep=sleep,
        monotonic=lambda: current,
        timeout_seconds=10,
        emit=lambda **event: events.append(event),
    )
    assert result.plan_sha256 == campaign.plan_digest(result)
    assert sleeps == [5.0]
    assert events[0]["event"] == "recovery_plan_waiting"
    assert events[0]["recovery_rank"] == 2


@pytest.mark.parametrize(
    "corruption",
    ["checksum", "version", "config", "owner", "topology", "source", "inventory"],
)
def test_stale_or_mismatched_plan_fails_closed(tmp_path, monkeypatch, corruption):
    fixture, paths, plan_path = _fixture(tmp_path, monkeypatch)
    _obtain(fixture, plan_path)
    payload = json.loads(plan_path.read_text())
    if corruption == "source":
        paths[0].write_bytes(paths[0].read_bytes() + b"\n")
    elif corruption == "topology":
        marker = production._static_topology_path(fixture.output_root)
        topology = json.loads(marker.read_text())
        topology["world_size"] = 2
        marker.write_text(json.dumps(topology))
    else:
        if corruption == "checksum":
            payload["created_at"] = "changed"
        elif corruption == "version":
            payload["schema_version"] = "unknown"
        elif corruption == "config":
            payload["base_config_sha256"] = "f" * 64
        elif corruption == "owner":
            payload["shards"][0]["original_worker_id"] = 39
        else:
            payload["shards"].pop()
        if corruption != "checksum":
            payload["plan_sha256"] = campaign.plan_digest(
                campaign.RecoveryPlan.model_validate(payload)
            )
        plan_path.write_text(json.dumps(payload))
    before = plan_path.read_bytes()
    with pytest.raises(ValueError):
        _obtain(fixture, plan_path)
    assert plan_path.read_bytes() == before


def _worker_args(
    fixture, plan_path, plan, log_path, *, local_slot=0, gpu="7", local_gpu_count=1
):
    return [
        "--input-root",
        str(fixture.input_root),
        "--output-root",
        str(fixture.output_root),
        "--base-config",
        str(fixture.base_path),
        "--plan",
        str(plan_path),
        "--plan-sha256",
        plan.plan_sha256,
        "--rank",
        "0",
        "--world-size",
        "1",
        "--local-gpu-count",
        str(local_gpu_count),
        "--local-slot",
        str(local_slot),
        "--gpu",
        gpu,
        "--log-file",
        str(log_path),
    ]


def test_physical_worker_loads_one_backend_and_skips_new_canonical_on_restart(
    tmp_path, monkeypatch, capsys
):
    fixture, paths, plan_path = _fixture(tmp_path, monkeypatch)
    plan = _obtain(fixture, plan_path)
    backend, created, processed = FakeBackend(), [], []
    actual_process = stage2.process_shard

    def process(path, **kwargs):
        processed.append(path)
        return actual_process(path, decoder=FakeDecoder(), **kwargs)

    monkeypatch.setattr(stage2, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(
        stage2,
        "default_backend_factory",
        lambda config: created.append(backend) or backend,
    )
    monkeypatch.setattr(
        stage2, "enable_sam3_session_reuse", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(stage2, "process_shard", process)
    monkeypatch.setattr(
        production, "check_stage2_candidate_judge_health", lambda config: {}
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    argv = _worker_args(
        fixture, plan_path, plan, fixture.writable / "logs" / "slot.log"
    )
    first = worker.main(argv)
    assert first["completed_shards"] == 3
    assert created == [backend]
    assert processed == paths
    assert backend.close_calls == 1
    before = {
        path: path.read_bytes()
        for path in fixture.output_root.rglob("*")
        if path.is_file()
    }
    second = worker.main(argv)
    assert second == first
    assert created == [backend] and processed == paths
    assert {path: path.read_bytes() for path in before} == before
    events = [
        json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()
    ]
    assert events.count("shard_started") == 6
    assert events.count("shard_completed") == 6
    assert events.count("physical_worker_completed") == 2


@pytest.mark.parametrize("failure", ["retryable", "exception"])
def test_physical_worker_closes_backend_on_failures(
    tmp_path, monkeypatch, capsys, failure
):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch, counts=(1,))
    plan = _obtain(fixture, plan_path)
    backend = FakeBackend()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(stage2, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(stage2, "default_backend_factory", lambda config: backend)
    monkeypatch.setattr(
        stage2, "enable_sam3_session_reuse", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        production, "check_stage2_candidate_judge_health", lambda config: {}
    )

    def process(*args, **kwargs):
        if failure == "exception":
            raise ValueError("identity mismatch")
        return {"retryable": True, "stage": "sam3"}

    monkeypatch.setattr(stage2, "process_shard", process)
    argv = _worker_args(
        fixture, plan_path, plan, fixture.writable / "logs" / "slot.log", gpu="0"
    )
    if failure == "exception":
        with pytest.raises(ValueError, match="identity mismatch"):
            worker.main(argv)
    else:
        assert worker.main(argv)["retryable"]
    assert backend.close_calls == 1
    events = capsys.readouterr().out
    assert (
        "shard_retryable" if failure == "retryable" else "physical_worker_failed"
    ) in events


def test_cluster_auto_uses_physical_env_and_launches_visible_gpus(
    tmp_path, monkeypatch, capsys
):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch)
    plan = _obtain(fixture, plan_path)
    commands, environments = [], []
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr(launcher, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(launcher, "ALLOWED_WRITABLE_ROOT", fixture.writable)

    def launch(command, **kwargs):
        commands.append(command)
        environments.append(kwargs["env"])
        assert kwargs["stdout"] is None

        def wait():
            assert len(commands) == 2  # Both GPUs launched before any worker wait.
            return 0

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    result = launcher.main(
        [
            "--cluster-auto",
            "--confirm-production-stopped",
            "--input-root",
            str(fixture.input_root),
            "--output-root",
            str(fixture.output_root),
            "--base-config",
            str(fixture.base_path),
            "--plan",
            str(plan_path),
            "--log-root",
            str(fixture.writable / "logs"),
        ]
    )
    assert result["recovery_rank"] == 1 and result["recovery_world_size"] == 2
    assert len(commands) == 2
    assert [env["CUDA_VISIBLE_DEVICES"] for env in environments] == ["3", "7"]
    assert all(command[command.index("--rank") + 1] == "1" for command in commands)
    assert all(
        command[command.index("--local-gpu-count") + 1] == "2" for command in commands
    )
    assert all("--logical-workers" not in command for command in commands)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [
        event["physical_global_slot"]
        for event in events
        if event["event"] == "physical_assignment"
    ] == [2, 3]
    assert list((fixture.writable / "logs" / plan.plan_sha256).glob("*.log"))
    assert (
        json.loads(production._static_topology_path(fixture.output_root).read_text())[
            "world_size"
        ]
        == 5
    )


def test_shards_of_one_original_owner_can_hold_independent_recovery_locks(tmp_path):
    from threading import Barrier

    barrier = Barrier(2)

    def run(index):
        with stage2.shard_lock(tmp_path / f"shard-{index}.lock"):
            barrier.wait(timeout=5)
            return index

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(run, [0, 1])) == [0, 1]


def test_real_frozen_owner_range_is_preserved_while_five_shards_split(
    tmp_path, monkeypatch
):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch, counts=(1,) * 200)
    plan = _obtain(fixture, plan_path)
    owned = [item for item in plan.shards if item.original_worker_id == 4]
    assert [item.position for item in owned] == [20, 21, 22, 23, 24]
    assert all(
        (
            item.original_rank,
            item.original_local_slot,
            item.original_shard_start,
            item.original_shard_end,
        )
        == (0, 4, 20, 25)
        for item in owned
    )
    assigned = campaign.weighted_assignment(owned, world_size=1, local_gpu_count=5)
    assert [len(slot) for slot in assigned] == [1] * 5


def test_single_shard_canary_only_executes_requested_plan_member(tmp_path, monkeypatch):
    fixture, paths, plan_path = _fixture(tmp_path, monkeypatch)
    plan = _obtain(fixture, plan_path)
    visited = []
    backend = FakeBackend()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(stage2, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(stage2, "default_backend_factory", lambda config: backend)
    monkeypatch.setattr(
        stage2, "enable_sam3_session_reuse", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        production, "check_stage2_candidate_judge_health", lambda config: {}
    )
    monkeypatch.setattr(
        stage2,
        "process_shard",
        lambda path, **kwargs: visited.append(path) or {"rows": 2},
    )
    argv = _worker_args(
        fixture, plan_path, plan, fixture.writable / "logs" / "slot.log"
    )
    result = worker.main([*argv, "--single-shard", paths[1].name])
    assert visited == [paths[1]]
    assert result["assigned_shards"] == result["completed_shards"] == 1
    assert backend.close_calls == 1


def test_plan_wait_timeout_never_creates_or_mutates_plan(tmp_path, monkeypatch):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch)
    sleeps = []
    with pytest.raises(TimeoutError):
        campaign.obtain_plan(
            plan_path,
            fixture.input_root,
            fixture.output_root,
            fixture.identity,
            rank=1,
            timeout_seconds=0,
            sleep=sleeps.append,
        )
    assert not plan_path.exists() and sleeps == []


def test_cluster_dry_run_leaves_unpublished_plan_and_artifacts_untouched(
    tmp_path, monkeypatch
):
    fixture, _, plan_path = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(launcher, "load_config_identity", lambda path: fixture.identity)
    monkeypatch.setattr(launcher, "ALLOWED_WRITABLE_ROOT", fixture.writable)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("dry-run spawned worker"),
    )
    before = {
        path: path.read_bytes()
        for path in fixture.output_root.rglob("*")
        if path.is_file()
    }
    result = launcher.main(
        [
            "--cluster-auto",
            "--dry-run",
            "--input-root",
            str(fixture.input_root),
            "--output-root",
            str(fixture.output_root),
            "--base-config",
            str(fixture.base_path),
            "--plan",
            str(plan_path),
            "--log-root",
            str(fixture.writable / "logs"),
        ]
    )
    assert result["recovery_world_size"] == 3
    assert result["local_gpu_count"] == 2
    assert not plan_path.exists()
    assert {path: path.read_bytes() for path in before} == before


def test_original_reconciler_and_normal_launcher_are_unchanged():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    expected = {
        "r2v_data_v2/v3/pre_qwen_production.py": "552dc145ea3f9ba15187a64a81e7ce1bbedceedcc39d75400e9c26d8d45f3a73",
        "tools/run_v3_entity_mask_auto.py": "531b98778b97479c5b1118daf00716c55763a52d7712a3e28542b0b1a2411a7c",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
