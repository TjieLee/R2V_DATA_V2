"""Each production quarter completes and publishes before the next starts."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
from r2v_data_v2.h3.ra2va_group_production import STAGES, GroupCoordinator, read_json
from tests.h3_ra2va_group_http_helpers import (
    claim,
    connect,
    post,
    release,
    result_body,
    serving,
)
from tests.test_h3_ra2va_group_production import complete_stage
from tests.test_h3_ra2va_group_source import publish_group


def production(root, source, **kwargs):
    return GroupCoordinator(root, source, None, mode="cpu_fake_production", group_parts=4, **kwargs)


def test_quarters_keep_source_order_and_only_complete_the_whole_group_at_the_end(tmp_path):
    source, root = tmp_path / "source", tmp_path / "run"
    publish_group(source, count=10)
    publish_group(source, name="group-000001", count=1)
    core = production(root, source)
    first = core.select_current()
    inventory = core.group_root(first.group_id) / "inventory/inventory.json"
    before = inventory.read_bytes()
    assert read_json(inventory)["parts"] == [
        {"start_ordinal": 0, "end_ordinal": 2},
        {"start_ordinal": 2, "end_ordinal": 5},
        {"start_ordinal": 5, "end_ordinal": 7},
        {"start_ordinal": 7, "end_ordinal": 10},
    ]
    # Resuming subsequent quarters never traverses the source JSONL again.
    (source / "group-000000/samples.jsonl").unlink()
    executions = []
    for part, ordinals in enumerate(([0, 1], [2, 3, 4], [5, 6], [7, 8, 9])):
        core = production(root, source)
        for stage in STAGES:
            snapshot = core.select_current()
            assert snapshot.group_id == "group-000000" and snapshot.stage == stage
            stats = core.progress(snapshot)
            assert stats["part_index"] == part and stats["part_count"] == 4
            assert stats["pending"] == len(ordinals)
            actual = []
            while held := core.claim(snapshot, "worker"):
                with held:
                    actual.append(held.task.ordinal)
                    executions.append((held.task.ordinal, stage))
                    core.publish(held, ("ready", "failed", "skipped")[held.task.ordinal % 3], {}, 0)
            assert actual == ordinals
            assert core.progress(snapshot)["pending"] == 0
            assert core.advance(snapshot)
        group = core.group_root("group-000000")
        assert (group / f"parts/part-{part:03d}/COMPLETE").is_file()
        assert (group / "COMPLETE").exists() == (part == 3)
        assert not (group / "PILOT_COMPLETE").exists()
    assert len(executions) == len(set(executions)) == 80
    assert inventory.read_bytes() == before
    assert core.select_current() is None
    assert core.status()["started_groups"] == ["group-000000"]
    assert not core.group_root("group-000001").exists()
    with pytest.raises(ValueError, match="settings"):
        GroupCoordinator(root, source, None, mode="cpu_fake_production", group_parts=1)


def test_quarter_completion_crash_does_not_restart_completed_stages(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_production as module

    source, root = tmp_path / "source", tmp_path / "run"
    publish_group(source, count=8)
    core = production(root, source)
    for _ in range(7):
        snapshot = core.select_current()
        complete_stage(core, snapshot)
        assert core.advance(snapshot)
    snapshot = core.select_current()
    complete_stage(core, snapshot)
    results = {p: p.read_bytes() for p in root.glob("groups/*/stages/*/results/*.json")}
    original = module.atomic_json

    def interrupted(path, value):
        if path == core.control_path and value.get("part_index") == 1:
            raise OSError("crash after quarter publication before control")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", interrupted)
        with pytest.raises(OSError, match="quarter publication"):
            core.advance(snapshot)
    resumed = production(root, source)
    current = resumed.select_current()
    assert current.stage == snapshot.stage and current.generation == snapshot.generation
    assert current.phase == "draining"
    assert resumed.claim(snapshot, "must-not-repeat") is None
    assert resumed.advance(snapshot)
    next_stage = resumed.select_current()
    assert next_stage.stage == "canonical" and next_stage.generation == snapshot.generation + 1
    with resumed.claim(next_stage, "resumed") as held:
        assert held.task.ordinal == 2
    assert all(p.read_bytes() == data for p, data in results.items())


def http_production(root, source, *, native=False):
    return HttpGroupCoordinator(GroupCoordinator(root, source, None, group_parts=4,
        mode="http_native_cpu_production" if native else "cpu_fake_production",
        transport="http", coordinator_host="node-a", coordinator_identity={
            "guard_path": str(root.parent / "local.lock"), "bind_host": "127.0.0.1", "bind_port": 0}))


def test_quarter_boundary_waits_for_release_and_old_receipt_stays_idempotent(tmp_path):
    source = tmp_path / "source"
    publish_group(source, count=4)
    service = http_production(tmp_path / "run", source)
    try:
        last_body = None
        for stage in STAGES:
            session = connect(service)
            task = claim(service, session)
            assert task["task_id"]["ordinal"] == 0 and task["task_id"]["stage"] == stage
            last_body = result_body(task)
            post(service, "/v1/tasks/result", **last_body)
            service.tick()
            assert service.core.status()["phase"] == "draining"
            assert service.core.status().get("part_index", 0) == 0
            release(service, session)
            service.tick()
        assert service.snapshot.stage == "canonical"
        assert service.core.status()["part_index"] == 1
        receipt_path = service.core.group_root("group-000000") / "stages/000008-export/results/000000.json"
        original = receipt_path.read_bytes()
        post(service, "/v1/tasks/result", **last_body)
        assert receipt_path.read_bytes() == original
        assert service.dispatch("POST", "/v1/tasks/result", {**last_body, "payload": {"changed": True}})[0] == 409
        assert service.dispatch("POST", "/v1/tasks/claim", {
            "session_id": session["session_id"], "generation": session["generation"]})[0] == 409
        assert claim(service, connect(service, "next"))["task_id"]["ordinal"] == 1
    finally:
        service.close()


def test_two_http_fake_nodes_share_each_quarter_without_duplicate_terminal_results(tmp_path):
    from r2v_data_v2.h3.ra2va_group_fake import pilot_summary
    from r2v_data_v2.h3.ra2va_group_http_worker import (
        GroupHttpClient,
        run_http_fake_worker,
    )

    source, root = tmp_path / "source", tmp_path / "run"
    publish_group(source, count=20)
    publish_group(source, name="group-000001", count=2)
    service = http_production(root, source)
    with serving(service) as url, ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run_http_fake_worker, GroupHttpClient(url, "test-token"),
            node_id=node, worker_instance=node, stop_event=threading.Event(), fake_delay_seconds=.01)
            for node in ("node-a", "node-b")]
        reports = [future.result(timeout=45) for future in futures]
    rows = [read_json(p) for p in root.glob("groups/*/stages/*/results/*.json")]
    assert sum(r["accepted"] for r in reports) == len(rows) == 160
    assert len({(r["ordinal"], r["stage"]) for r in rows}) == 160
    assert {r["http"]["node_id"] for r in rows} == {"node-a", "node-b"}
    summary = pilot_summary(root)
    assert len(summary["stages"]) == 32 and all(s["pending"] == 0 for s in summary["stages"])
    assert summary["completed_groups"] == 1 and summary["completed_parts"] == 4
    assert summary["model_call_count"] == 0
    assert not (root / "groups/group-000001").exists()


def test_cpu_pipeline_publishes_each_quarter_training_before_next_quarter(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_cpu_fixtures import CPUFixtureFactory
    from r2v_data_v2.h3.ra2va_group_fake import pilot_summary
    from r2v_data_v2.h3.ra2va_group_http_native_worker import run_http_native_worker
    from r2v_data_v2.h3.ra2va_group_http_worker import GroupHttpClient
    from tests.test_h3_ra2va_group_local import fixture_group
    from tests.test_h3_ra2va_group_pipeline import ffmpeg

    source, fixtures = fixture_group(tmp_path, count=8)
    # fixture_group deliberately declares extra rows for Pilot tests; publish this whole source.
    export_marker = source.parent / "state/resource_epochs/group-000000/completed.json"
    value = read_json(export_marker)
    value["sample_count"] = 8
    export_marker.write_text(json.dumps(value))
    root = tmp_path / "run"
    service = http_production(root, source, native=True)
    original, first_outputs = GroupHttpClient.call, {}

    def observe(self, method, route, body=None):
        response = original(self, method, route, body)
        if route == "/v1/tasks/claim" and response.get("task") and response["task_id"]["ordinal"] >= 2:
            bundle = root / "groups/group-000000/parts/part-000/bundle"
            assert (bundle / "summary.json").is_file()
            assert not (root / "groups/group-000000/COMPLETE").exists()
            for path in bundle.rglob("*"):
                if path.is_file():
                    first_outputs.setdefault(path, path.read_bytes())
        return response

    monkeypatch.setattr(GroupHttpClient, "call", observe)
    with serving(service) as url, ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run_http_native_worker, GroupHttpClient(url, "test-token"),
            node_id=node, worker_instance=node, stop_event=threading.Event(),
            backend_factory=CPUFixtureFactory(fixtures, ffmpeg()), ffmpeg=ffmpeg(), execution="cpu_fixture")
            for node in ("node-a", "node-b")]
        reports = [future.result(timeout=90) for future in futures]
    assert sum(r["accepted"] for r in reports) == 64
    assert first_outputs and all(p.read_bytes() == data for p, data in first_outputs.items())
    group = root / "groups/group-000000"
    clips = []
    for part in range(4):
        bundle = group / f"parts/part-{part:03d}/bundle"
        summary = read_json(bundle / "summary.json")
        assert summary["product_ready_count"] == 6 and summary["product_failed_count"] == 0
        assert len(summary["task_counts"]) == 12 and set(summary["task_counts"].values()) == {2}
        assert summary["donor_count"] == 2 and summary["donor_segment_count"] == 4
        contract = read_json(bundle / "source_contract.json")
        clips.extend(j["clip_uid"] for j in contract["jobs"])
        for task in summary["task_counts"]:
            path = bundle / "training" / f"{task}.jsonl"
            for line in path.read_text().splitlines():
                assert set(json.loads(line)) == {"video", "images", "audios", "caption"}
        for donor in read_json(bundle / "voice_donor_reserve.json")["donors"]:
            for segment in donor["segments"]:
                assert (bundle / segment["relative_path"]).is_file()
    assert clips == [f"clip-{i:03d}" for i in range(8)]
    assert (group / "COMPLETE").is_file() and not (group / "PILOT_COMPLETE").exists()
    summary = pilot_summary(root)
    assert len(summary["exports"]) == 4 and summary["completed_parts"] == 4
    assert summary["model_call_count"] == 0 and summary["simulated_mimo_request_count"] == 16


@pytest.mark.parametrize("count, expected", [(0, []), (1, [0]), (2, [0, 1]), (3, [0, 1, 2])])
def test_tiny_full_groups_do_not_create_empty_quarters(tmp_path, count, expected):
    source = tmp_path / "source"
    publish_group(source, count=count)
    core = production(tmp_path / "run", source)
    ordinals = []
    while snapshot := core.select_current():
        while held := core.claim(snapshot, "cpu"):
            with held:
                if snapshot.stage == "canonical":
                    ordinals.append(held.task.ordinal)
                core.publish(held, "ready", {}, 0)
        assert core.advance(snapshot)
    assert ordinals == expected


def test_bash_full_entry_defaults_to_four_parts_but_pilot_stays_one(tmp_path):
    import subprocess

    from tests.test_h3_ra2va_cluster_dynamic import SCRIPT, environment

    for full, expected in (("1", 4), ("0", 1)):
        env = environment(tmp_path, role="coordinator")
        env["R2VA_FULL_GROUP"] = full
        env.pop("R2VA_GROUP_PARTS", None)
        result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=env,
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["group_parts"] == expected
        command = plan["coordinator"]
        assert command[command.index("--group-parts") + 1] == str(expected)


def test_entry_resume_inherits_old_unsplit_run_instead_of_changing_its_scope(tmp_path):
    import subprocess
    from pathlib import Path

    from tests.test_h3_ra2va_cluster_dynamic import SCRIPT, environment

    env = environment(tmp_path, role="coordinator")
    env.update(R2VA_FULL_GROUP="1")
    env.pop("R2VA_GROUP_PARTS", None)
    root = Path(env["R2VA_OUTPUT_ROOT"]) / env["R2VA_RUN_ID"]
    core = GroupCoordinator(root, Path(env["R2VA_SOURCE_ROOT"]), None, mode="cpu_fake_production")
    original = core.control_path.read_bytes()
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["group_parts"] == 1
    assert core.control_path.read_bytes() == original


def test_four_part_run_restarts_second_quarter_with_unexpired_and_expired_claims(tmp_path):
    from tests.h3_ra2va_group_http_helpers import Clock

    source, root, clock = tmp_path / "source", tmp_path / "run", Clock()
    publish_group(source, count=8)

    def open_service():
        core = GroupCoordinator(root, source, None, group_parts=4, mode="cpu_fake_production",
            transport="http", coordinator_host="node-a", coordinator_identity={
                "guard_path": str(tmp_path / "local.lock"), "bind_host": "127.0.0.1", "bind_port": 0})
        return HttpGroupCoordinator(core, wall_clock=clock)

    service = open_service()
    for _ in STAGES:
        session = connect(service)
        for _ in range(2):
            held = claim(service, session)
            post(service, "/v1/tasks/result", **result_body(held))
        service.tick()
        release(service, session)
        service.tick()
    session = connect(service, "interrupted")
    old = claim(service, session)
    assert old["task_id"]["ordinal"] == 2 and old["task_id"]["generation"] == 9
    first_results = {p: p.read_bytes() for p in root.glob("groups/*/stages/*/results/*.json")}
    service.close()
    restored = open_service()
    try:
        assert connect(restored, "interrupted")["claim"] == old
        other = claim(restored, connect(restored, "other"))
        assert other["task_id"]["ordinal"] == 3
        post(restored, "/v1/tasks/result", **result_body(other, "failed"))
        clock.now += 61
        restored.tick()
        recovered = claim(restored, connect(restored, "new"))
        assert recovered["task_id"] == old["task_id"]
        assert recovered["claim_token"] != old["claim_token"]
        assert restored.dispatch("POST", "/v1/tasks/result", result_body(old))[0] == 409
        post(restored, "/v1/tasks/result", **result_body(recovered))
        assert all(p.read_bytes() == data for p, data in first_results.items())
        assert restored.core.progress(restored.snapshot)["failed"] == 1
        assert restored.core.progress(restored.snapshot)["pending"] == 0
    finally:
        restored.close()
