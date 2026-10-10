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


def native_service(root, source, *, mode="http_native_cpu_pilot", limit=3):
    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
    return HttpGroupCoordinator(GroupCoordinator(root, source, limit, transport="http",
        coordinator_host="node-a", mode=mode, coordinator_identity={
            "guard_path": str(root.parent / "native.lock"), "bind_host": "127.0.0.1", "bind_port": 9001}))


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


def test_http_native_cpu_fixture_eight_stages_and_export(tmp_path):
    from r2v_data_v2.h3.ra2va_group_cpu_fixtures import CPUFixtureFactory
    from r2v_data_v2.h3.ra2va_group_http_native_worker import run_http_native_worker
    from r2v_data_v2.h3.ra2va_group_http_worker import GroupHttpClient
    from tests.test_h3_ra2va_group_local import fixture_group
    from tests.test_h3_ra2va_group_pipeline import ffmpeg
    source, fixtures = fixture_group(tmp_path, count=3)
    root = tmp_path / "native"
    service = native_service(root, source)
    with serving(service) as url, ThreadPoolExecutor(2) as pool:
        runs = [pool.submit(run_http_native_worker, GroupHttpClient(url, "test-token"),
            node_id="cpu", worker_instance=f"native-{i}", stop_event=threading.Event(),
            backend_factory=CPUFixtureFactory(fixtures, ffmpeg()), ffmpeg=ffmpeg(),
            execution="cpu_fixture") for i in range(2)]
        reports = [future.result(timeout=90) for future in runs]
    assert sum(r["accepted"] for r in reports) == 24
    rows = [json.loads(p.read_text()) for p in root.glob("groups/*/stages/*/results/*.json")]
    assert len(rows) == 24 and all(r["status"] == "ready" for r in rows)
    assert all(r["payload"]["execution"] == "cpu_fixture" for r in rows)
    paths = [r["payload"]["path"] for r in rows if r["stage"] == "canonical"]
    assert len(set(paths)) == 3 and all("/attempts/" in p and Path(p).is_file() for p in paths)
    summary = json.loads((root / "groups/group-000000/bundle/summary.json").read_text())
    assert summary["product_ready_count"] == 9 and summary["product_failed_count"] == 0
    assert set(summary["task_counts"].values()) == {3}
    assert summary["donor_count"] == 3 and summary["donor_segment_count"] == 6
    assert (root / "groups/group-000000/PILOT_COMPLETE").exists()
    assert not list(root.rglob("COMPLETE"))
    # Native completion cannot masquerade as a Fake completion or vice versa.
    with pytest.raises(ValueError, match="mode differs"):
        native_service(root, source, mode="cpu_fake_pilot")
