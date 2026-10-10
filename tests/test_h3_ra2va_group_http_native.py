from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.h3_ra2va_group_http_helpers import (
    claim,
    connect,
    make_service,
    post,
    result_body,
    serving,
)


def test_checkpoint_recovery_belongs_to_selected_clip_only(tmp_path):
    service = make_service(tmp_path, count=2)
    try:
        first = claim(service, connect(service, "a"))
        identity = {k: first[k] for k in ("session_id", "task_id", "claim_token")}
        post(service, "/v1/tasks/event", **identity, event="request_started")
        post(service, "/v1/tasks/event", **identity, event="response_saved", checkpoint={"clip": "first"})
        # Simulate durable revocation before the old local handle is closed.
        with service.core._checked(service.snapshot) as (_, state):
            state["active"]["0"]["http"]["revoked"] = True
            service.core._save_dispatch(service.snapshot, state)
        second = claim(service, connect(service, "b"))
        assert second["task_id"]["ordinal"] == 1
        assert "checkpoint" not in second
        post(service, "/v1/tasks/result", **result_body(second))
        service.claims.pop(first["session_id"]).close()
        recovered = claim(service, connect(service, "c"))
        assert recovered["task_id"] == first["task_id"]
        assert recovered["checkpoint"] == {"clip": "first"}
    finally:
        service.close()


def native_service(root, source, *, mode="http_native_cpu_pilot", limit=3, **kwargs):
    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
    return HttpGroupCoordinator(GroupCoordinator(root, source, limit, transport="http",
        coordinator_host="node-a", mode=mode, coordinator_identity={
            "guard_path": str(root.parent / "native.lock"), "bind_host": "127.0.0.1", "bind_port": 9001}), **kwargs)


def test_native_and_fake_workers_cannot_share_run(tmp_path):
    service = make_service(tmp_path)
    try:
        code, _ = service.dispatch("POST", "/v1/workers/connect", {
            "node_id": "a", "worker_instance": "native", "resources_closed": True,
            "execution_mode": "native"})
        assert code == 409
    finally:
        service.close()
    from tests.test_h3_ra2va_group_source import publish_group
    source = tmp_path / "native-source"
    publish_group(source, count=1)
    native = native_service(tmp_path / "native", source)
    try:
        assert native.dispatch("POST", "/v1/workers/connect", {
            "node_id": "a", "worker_instance": "fake", "resources_closed": True})[0] == 409
    finally:
        native.close()


@pytest.mark.parametrize("count", [3, 20])
def test_http_native_cpu_fixture_eight_stages_and_export(tmp_path, count):
    from r2v_data_v2.h3.ra2va_group_cpu_fixtures import CPUFixtureFactory
    from r2v_data_v2.h3.ra2va_group_http_native_worker import run_http_native_worker
    from r2v_data_v2.h3.ra2va_group_http_worker import GroupHttpClient
    from tests.test_h3_ra2va_group_local import fixture_group
    from tests.test_h3_ra2va_group_pipeline import ffmpeg
    source, fixtures = fixture_group(tmp_path, count=count)
    root = tmp_path / "native"
    service = native_service(root, source, limit=count)
    with serving(service) as url, ThreadPoolExecutor(2) as pool:
        runs = [pool.submit(run_http_native_worker, GroupHttpClient(url, "test-token"),
            node_id="cpu", worker_instance=f"native-{i}", stop_event=threading.Event(),
            backend_factory=CPUFixtureFactory(fixtures, ffmpeg()), ffmpeg=ffmpeg(),
            execution="cpu_fixture") for i in range(2)]
        reports = [future.result(timeout=90) for future in runs]
    assert sum(r["accepted"] for r in reports) == count * 8
    rows = [json.loads(p.read_text()) for p in root.glob("groups/*/stages/*/results/*.json")]
    assert len(rows) == count * 8 and all(r["status"] == "ready" for r in rows)
    assert all(r["payload"]["execution"] == "cpu_fixture" for r in rows)
    paths = [r["payload"]["path"] for r in rows if r["stage"] == "canonical"]
    assert len(set(paths)) == count and all("/attempts/" in p and Path(p).is_file() for p in paths)
    summary = json.loads((root / "groups/group-000000/bundle/summary.json").read_text())
    assert summary["product_ready_count"] == count * 3 and summary["product_failed_count"] == 0
    assert set(summary["task_counts"].values()) == {count}
    assert summary["donor_count"] == count and summary["donor_segment_count"] == count * 2
    assert (root / "groups/group-000000/PILOT_COMPLETE").exists()
    assert not list(root.rglob("COMPLETE"))
    marker = json.loads((root / "groups/group-000000/PILOT_COMPLETE").read_text())
    assert marker["model_call_count"] == 0 and marker["simulated_mimo_request_count"] == count * 2
    mimo_rows = [r for r in rows if r["stage"] == "mimo"]
    assert all(set(r["http"]["events"]) == {f"{turn}.{event}" for turn in ("visual", "joint")
               for event in ("request_started", "response_saved")} for r in mimo_rows)
    # Native completion cannot masquerade as a Fake completion or vice versa.
    with pytest.raises(ValueError, match="mode differs"):
        native_service(root, source, mode="cpu_fake_pilot", limit=count)


def test_native_launcher_lost_event_ack_and_result_ack_are_idempotent(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_http_native_worker import run_http_native_workers
    from r2v_data_v2.h3.ra2va_group_http_worker import GroupHttpClient
    from tests.test_h3_ra2va_group_local import fixture_group
    from tests.test_h3_ra2va_group_pipeline import ffmpeg
    source, fixtures = fixture_group(tmp_path, count=1)
    root = tmp_path / "native"
    service = native_service(root, source, limit=1)
    original, dropped = GroupHttpClient.call, set()

    def lose_once(self, method, route, body=None):
        response = original(self, method, route, body)
        key = (route, body.get("turn")) if body else (route, None)
        if route in ("/v1/tasks/event", "/v1/tasks/result") and key not in dropped:
            dropped.add(key)
            raise OSError("HTTP response lost after durable acceptance")
        return response
    monkeypatch.setattr(GroupHttpClient, "call", lose_once)
    monkeypatch.setenv("R2V_GROUP_COORDINATOR_TOKEN", "test-token")
    with serving(service) as url:
        report = run_http_native_workers(coordinator_url=url, node_id="cpu", workers=2,
                                         cpu_fixtures=fixtures, ffmpeg=ffmpeg())
    assert report["executed"] == report["accepted"] == 8
    record = json.loads(next(root.glob("groups/*/stages/*-mimo/results/*.json")).read_text())
    assert record["payload"]["new_request_count"] == record["payload"]["model_call_count"] == 2
    assert len(list(root.glob("groups/*/stages/*-mimo/attempts/*/*/mimo/*/*.request.json"))) == 2
