"""Full-group scope stays explicit and separate from historical Pilots."""

import json

import pytest

from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
from r2v_data_v2.h3.ra2va_group_source import (
    discover_published_groups,
    read_inventory_task,
)
from tests.test_h3_ra2va_group_production import complete_group
from tests.test_h3_ra2va_group_source import publish_group


def test_full_inventory_reads_source_once_and_resumes_without_media_scan(tmp_path):
    from r2v_data_v2.h3.ra2va_group_source import freeze_group_inventory

    source = tmp_path / "post_mask"
    publish_group(source, count=205)
    group = discover_published_groups(source)[0]
    inventory = tmp_path / "inventory"
    metadata = freeze_group_inventory(group, inventory, None)
    assert metadata["mode"] == "full_group"
    assert metadata["selected_count"] == 205
    assert read_inventory_task(inventory, 204).ordinal == 204
    group.samples_path.unlink()
    assert freeze_group_inventory(group, inventory, None) == metadata
    with pytest.raises(ValueError, match="settings"):
        freeze_group_inventory(group, inventory, 20)


def test_full_group_complete_and_cumulative_budget_survive_resume(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source, count=3)
    publish_group(source, name="group-000001", count=2)
    root = tmp_path / "run"
    coordinator = GroupCoordinator(root, source, None, mode="cpu_fake_production")
    complete_group(coordinator)
    assert (root / "groups/group-000000/COMPLETE").is_file()
    assert not (root / "groups/group-000000/PILOT_COMPLETE").exists()
    resumed = GroupCoordinator(root, source, None, mode="cpu_fake_production")
    assert resumed.select_current() is None
    assert resumed.status()["started_groups"] == ["group-000000"]
    with pytest.raises(ValueError, match="mode|settings"):
        GroupCoordinator(root, source, 20)


def test_continuous_full_run_discovers_next_published_group(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source, count=2)
    coordinator = GroupCoordinator(tmp_path / "run", source, None, max_groups=0,
                                   mode="cpu_fake_production")
    complete_group(coordinator)
    assert coordinator.select_current() is None
    publish_group(source, name="group-000001", count=1)
    assert coordinator.select_current().group_id == "group-000001"
    complete_group(coordinator)
    assert coordinator.status()["started_groups"] == ["group-000000", "group-000001"]


def test_full_group_rejects_partial_selection_and_unbounded_pilot(tmp_path):
    with pytest.raises(ValueError):
        GroupCoordinator(tmp_path / "one", tmp_path / "source", None, start_row=10)
    with pytest.raises(ValueError):
        GroupCoordinator(tmp_path / "two", tmp_path / "source", 20, max_groups=0)
    with pytest.raises(ValueError):
        GroupCoordinator(tmp_path / "three", tmp_path / "source", 20, mode="cpu_fake_production")


def test_full_node_plan_is_explicit_and_does_not_reuse_default_pilot(tmp_path, capsys):
    from pathlib import Path

    from r2v_data_v2.h3.ra2va_group_launch import launch_node, node_configuration

    repo = Path(__file__).resolve().parents[1]
    launch_node(repo, {"R2VA_ROLE": "coordinator", "R2VA_FULL_GROUP": "1",
                       "R2VA_OUTPUT_ROOT": str(tmp_path)}, dry_run=True)
    plan = json.loads(capsys.readouterr().out)
    assert plan["full_group"] is True and plan["limit"] is None
    assert plan["run_root"].endswith("ra2va-group-production-v1")
    assert "--full-group" in plan["coordinator"] and "--limit" not in plan["coordinator"]
    assert plan["coordinator"][plan["coordinator"].index("--max-groups") + 1] == "0"
    configuration = node_configuration(repo, {})
    assert configuration["max_clip_duration_seconds"] == 20
    assert configuration["asr"]["batch_size"] == 8


@pytest.mark.parametrize("max_groups", [1, 2])
def test_http_full_fake_run_drains_all_stages_without_pilot_marker(tmp_path, max_groups):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    from r2v_data_v2.h3.ra2va_group_http_worker import (
        GroupHttpClient,
        run_http_fake_worker,
    )
    from tests.h3_ra2va_group_http_helpers import serving

    source = tmp_path / "post_mask"
    publish_group(source, count=3)
    if max_groups == 2:
        publish_group(source, name="group-000001", count=2)
    core = GroupCoordinator(tmp_path / "run", source, None, transport="http",
                            coordinator_host="node-a", mode="cpu_fake_production",
                            max_groups=max_groups,
                            coordinator_identity={"guard_path": str(tmp_path / "local.lock"),
                                                  "bind_host": "127.0.0.1", "bind_port": 0})
    service = HttpGroupCoordinator(core)
    with serving(service) as url, ThreadPoolExecutor(2) as pool:
        stop = threading.Event()
        futures = [pool.submit(run_http_fake_worker, GroupHttpClient(url, "test-token"),
                               node_id=node, worker_instance=node, stop_event=stop,
                               fake_delay_seconds=.02) for node in ("a", "b")]
        reports = [future.result(timeout=30) for future in futures]
    expected = 24 if max_groups == 1 else 40
    assert sum(row["accepted"] for row in reports) == expected
    assert all(row["accepted"] for row in reports)
    assert len(list(core.run_root.glob("groups/*/stages/*/results/*.json"))) == expected
    assert len(list(core.run_root.glob("groups/*/COMPLETE"))) == max_groups
    assert not list(core.run_root.glob("groups/*/PILOT_COMPLETE"))


def test_native_full_scope_uses_native_client_and_reported_call_accounting(tmp_path):
    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    from tests.h3_ra2va_group_http_helpers import post

    source = tmp_path / "post_mask"
    publish_group(source, count=1)
    core = GroupCoordinator(tmp_path / "run", source, None, transport="http",
                            coordinator_host="node-a", mode="http_native_production",
                            coordinator_identity={"guard_path": str(tmp_path / "local.lock"),
                                                  "bind_host": "127.0.0.1", "bind_port": 0})
    service = HttpGroupCoordinator(core)
    try:
        session = post(service, "/v1/workers/connect", node_id="a", worker_instance="a",
                       resources_closed=True, execution_mode="native")
        assert session["run_mode"] == "http_native_production"
        assert session["session_id"]
    finally:
        service.close()
    while snapshot := core.select_current():
        while task := core.claim(snapshot, "cpu-test"):
            with task:
                core.publish(task, "ready", {"model_call_count": 2 if snapshot.stage == "mimo" else 0}, 0)
        assert core.advance(snapshot)
    marker = json.loads((core.group_root("group-000000") / "COMPLETE").read_text())
    assert marker["model_call_count"] == 2


@pytest.mark.parametrize("saved", [False, True])
def test_full_native_recovery_keeps_saved_raw_and_never_retries_unresolved_request(tmp_path, saved):
    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    from tests.h3_ra2va_group_http_helpers import (
        Clock,
        claim,
        connect,
        post,
        release,
        result_body,
    )
    from tests.test_h3_ra2va_group_http_mimo import save_response, turn_event

    source = tmp_path / "post_mask"
    publish_group(source, count=1)
    settings = {"transport": "http", "coordinator_host": "node-a", "mode": "http_native_production",
                "coordinator_identity": {"guard_path": str(tmp_path / "local.lock"),
                                         "bind_host": "127.0.0.1", "bind_port": 0}}
    root, clock = tmp_path / "run", Clock()
    service = HttpGroupCoordinator(GroupCoordinator(root, source, None, **settings), wall_clock=clock)
    for _ in range(6):
        session = connect(service, execution_mode="native")
        task = claim(service, session)
        post(service, "/v1/tasks/result", **result_body(task))
        service.tick()
        release(service, session)
        service.tick()
    old = claim(service, connect(service, "old", execution_mode="native"))
    snapshot = service.snapshot
    turn_event(service, old, "visual", "request_started")
    raw_path = save_response(old, "visual") / "visual.response.json" if saved else None
    original = raw_path.read_bytes() if saved else None
    service.close()
    restarted = HttpGroupCoordinator(GroupCoordinator(root, source, None, **settings), wall_clock=clock)
    try:
        assert connect(restarted, "old", execution_mode="native")["claim"] == old
        clock.now += 61
        restarted.tick()
        if saved:
            new = claim(restarted, connect(restarted, "new", execution_mode="native"))
            assert new["replay_roots"] == {"visual": str(raw_path.parent)}
            assert restarted.dispatch("POST", "/v1/tasks/result", result_body(old))[0] == 409
            assert raw_path.read_bytes() == original
        else:
            result = json.loads(restarted.core.result_path(snapshot, 0).read_text())
            assert result["status"] == "failed"
            assert result["payload"]["failure_code"] == "interrupted_request_unresolved"
            assert result["payload"]["execution"] == "native"
            assert result["payload"]["model_call_count"] == 1
    finally:
        restarted.close()
