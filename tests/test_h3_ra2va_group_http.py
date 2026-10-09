from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
from tests.h3_ra2va_group_http_helpers import (
    claim,
    connect,
    make_service,
    post,
    request,
    result_body,
    serving,
)


def test_transport_modes_do_not_migrate_runs(tmp_path):
    local = GroupCoordinator(tmp_path / "local", tmp_path / "source", 20)
    before = local.control_path.read_bytes()
    with pytest.raises(ValueError, match="transport"):
        GroupCoordinator(local.run_root, local.source_root, 20,
                         transport="http", coordinator_host="node-a")
    assert local.control_path.read_bytes() == before
    service = make_service(tmp_path)
    try:
        before = service.core.control_path.read_bytes()
        with pytest.raises(ValueError, match="transport"):
            GroupCoordinator(service.core.run_root, service.core.source_root, 20)
        assert service.core.control_path.read_bytes() == before
    finally:
        service.close()


def test_owner_host_mismatch_does_not_write(tmp_path):
    service = make_service(tmp_path)
    try:
        before = service.core.control_path.read_bytes()
        with pytest.raises(ValueError, match="host"):
            GroupCoordinator(service.core.run_root, service.core.source_root, 20,
                             transport="http", coordinator_host="node-b")
        assert service.core.control_path.read_bytes() == before
    finally:
        service.close()


def test_two_clients_claim_different_clips_and_reconnect(tmp_path):
    service = make_service(tmp_path)
    with serving(service) as url:
        sessions = [request(url, "/v1/workers/connect", {
            "node_id": name, "worker_instance": name, "resources_closed": True,
        })[1] for name in ("a", "b")]
        with ThreadPoolExecutor(2) as pool:
            tasks = list(pool.map(lambda s: request(url, "/v1/tasks/claim", {
                "session_id": s["session_id"], "generation": s["generation"],
            })[1], sessions))
        a, b = tasks
        assert a["task_id"]["generation"] == b["task_id"]["generation"]
        assert a["task_id"]["ordinal"] != b["task_id"]["ordinal"]
        assert a["claim_token"] != b["claim_token"]
        again = request(url, "/v1/tasks/claim", {
            "session_id": sessions[0]["session_id"], "generation": sessions[0]["generation"],
        })[1]
        assert again == a
        assert connect(service, "a")["claim"] == a
        dispatch = json.loads((service.core.stage_root(service.snapshot) / "dispatch.json").read_text())
        assert dispatch["cursor"] == 2
        for task in tasks:
            post(service, "/v1/tasks/result", **result_body(task))
        assert service.core.progress(service.snapshot)["ready"] == 2


def test_auth_failure_does_not_mutate_state(tmp_path):
    service = make_service(tmp_path)
    with serving(service) as url:
        before = service.core.control_path.read_bytes()
        assert request(url, "/v1/workers/connect", {}, token="wrong")[0] == 401
        assert service.core.control_path.read_bytes() == before


def test_local_guard_and_busy_bind_reject_before_factory(tmp_path):
    from r2v_data_v2.h3.ra2va_group_http import (
        build_http_server,
        coordinator_process_guard,
    )
    with (coordinator_process_guard(tmp_path / "local.lock"),
          pytest.raises(RuntimeError, match="Coordinator"),
          coordinator_process_guard(tmp_path / "local.lock")):
        pytest.fail("second owner entered")
    with pytest.raises(ValueError, match="local"), coordinator_process_guard(Path("/mnt/workspace/shared.lock")):
        pytest.fail("shared guard accepted")
    called = []
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        with pytest.raises(OSError):
            build_http_server("127.0.0.1", occupied.getsockname()[1],
                              service_factory=lambda: called.append(True), bearer_token="test")
    assert called == []


def test_claim_token_and_cursor_publish_atomically(tmp_path, monkeypatch):
    import r2v_data_v2.h3.ra2va_group_production as module
    service = make_service(tmp_path)
    try:
        session = connect(service)
        path = service.core.stage_root(service.snapshot) / "dispatch.json"
        before = path.read_bytes()
        original = module.atomic_json

        def interrupted(path, value):
            if path.name == "dispatch.json" and value["active"]:
                row = value["active"]["0"]
                assert row["http"]["claim_token"]
                assert value["cursor"] == 1
                raise OSError("before atomic publication")
            original(path, value)

        with monkeypatch.context() as patch:
            patch.setattr(module, "atomic_json", interrupted)
            with pytest.raises(OSError, match="publication"):
                claim(service, session)
        assert path.read_bytes() == before
        assert claim(service, session)["task_id"]["ordinal"] == 0
    finally:
        service.close()
