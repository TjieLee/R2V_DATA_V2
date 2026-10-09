from __future__ import annotations

import json
import threading
import time

import pytest

from tests.h3_ra2va_group_http_helpers import connect, make_service, serving


def worker_api():
    from r2v_data_v2.h3 import ra2va_group_http_worker
    return ra2va_group_http_worker


def timings(monkeypatch):
    module = worker_api()
    monkeypatch.setattr(module, "HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(module, "WATCHDOG_SECONDS", 0.15)
    return module


class ServiceClient:
    def __init__(self, service):
        self.service = service
        self.renewals = 0

    def call(self, method, path, body=None):
        code, response = self.service.dispatch(method, path, body)
        if code == 409:
            raise worker_api().HttpConflict(response["error"])
        assert code == 200, response
        if path == "/v1/workers/heartbeat":
            self.renewals += 1
        return response


def test_long_task_keeps_independent_heartbeats(tmp_path, monkeypatch):
    module = timings(monkeypatch)
    service = make_service(tmp_path)
    client = ServiceClient(service)
    beat = module.WorkerHeartbeat(client)
    try:
        beat.start(connect(service))
        gate = threading.Event()
        blocked = threading.Thread(target=lambda: gate.wait(2))
        blocked.start()
        deadline = time.monotonic() + 2
        while client.renewals < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.renewals >= 2
        assert blocked.is_alive() and beat.execution_allowed()
        gate.set()
        blocked.join(2)
    finally:
        beat.stop()
        service.close()


def test_delayed_ack_uses_request_send_monotonic_time(tmp_path, monkeypatch):
    module = timings(monkeypatch)
    service = make_service(tmp_path)
    entered, reply = threading.Event(), threading.Event()
    clock = [10.0]

    class Delayed(ServiceClient):
        def call(self, method, path, body=None):
            response = super().call(method, path, body)
            entered.set()
            reply.wait(2)
            return response

    beat = module.WorkerHeartbeat(Delayed(service), monotonic=lambda: clock[0])
    try:
        beat.start(connect(service))
        assert entered.wait(2)
        clock[0] = 11.0
        assert not beat.execution_allowed()
        reply.set()
        time.sleep(0.04)
        assert not beat.execution_allowed()
    finally:
        reply.set()
        beat.stop()
        service.close()


@pytest.mark.parametrize("lost_route", ["connect", "claim", "result"])
def test_lost_connect_and_claim_replies_keep_original_claim(tmp_path, monkeypatch, lost_route):
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=2)
    lost = []

    class Dropped(ServiceClient):
        def call(self, method, path, body=None):
            response = super().call(method, path, body)
            if path.endswith("/" + lost_route) and not lost:
                lost.append(response)
                raise OSError("response lost after acceptance")
            return response

    stop = threading.Event()
    result = {}
    thread = threading.Thread(target=lambda: result.update(module.run_http_fake_worker(
        Dropped(service), node_id="a", worker_instance="a", stop_event=stop)))
    thread.start()
    try:
        deadline = time.monotonic() + 8
        while thread.is_alive() and time.monotonic() < deadline:
            service.tick()
            time.sleep(0.01)
        assert not thread.is_alive()
        assert result["executed"] == result["accepted"] == 16
        results = list(service.core.run_root.glob("groups/*/stages/*/results/*.json"))
        assert len(results) == 16
        assert all(json.loads(p.read_text())["status"] == "ready" for p in results)
        assert lost
    finally:
        stop.set()
        thread.join(3)
        service.close()


def test_short_network_outage_does_not_reexecute(tmp_path, monkeypatch):
    module = timings(monkeypatch)
    service = make_service(tmp_path)

    class Outage(ServiceClient):
        def call(self, method, path, body=None):
            if path.endswith("heartbeat") and self.renewals < 2:
                self.renewals += 1
                raise OSError("temporary disconnect")
            return super().call(method, path, body)

    client = Outage(service)
    beat = module.WorkerHeartbeat(client)
    try:
        session = connect(service)
        beat.start(session)
        time.sleep(0.09)
        assert beat.execution_allowed()
        assert connect(service)["session_id"] == session["session_id"]
    finally:
        beat.stop()
        service.close()


def test_idle_worker_heartbeats_while_other_task_runs(tmp_path, monkeypatch):
    from tests.h3_ra2va_group_http_helpers import claim
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=1)
    busy, idle = connect(service, "busy"), connect(service, "idle")
    claim(service, busy)
    client = ServiceClient(service)
    beat = module.WorkerHeartbeat(client)
    try:
        beat.start(idle)
        assert claim(service, idle)["task"] is None
        deadline = time.monotonic() + 2
        while client.renewals < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.renewals >= 2 and beat.execution_allowed()
        assert len(service.claims) == 1
    finally:
        beat.stop()
        service.close()


def test_watchdog_closes_fake_context_before_rejoin(tmp_path, monkeypatch):
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=1)
    events = []
    stop = threading.Event()
    entered = threading.Event()
    restore = threading.Event()
    original = module.FakeWorker

    class Tracked(original):
        def __enter__(self):
            events.append("open")
            entered.set()
            return self

        def __exit__(self, *args):
            events.append("close")

    class Outage(ServiceClient):
        def call(self, method, path, body=None):
            if path.endswith("heartbeat") and not restore.is_set():
                raise OSError("offline")
            if path.endswith("connect") and "close" in events:
                assert events[-1] == "close"
                events.append("rejoin")
                stop.set()
            return super().call(method, path, body)

    monkeypatch.setattr(module, "FakeWorker", Tracked)
    thread = threading.Thread(target=module.run_http_fake_worker,
        args=(Outage(service),), kwargs={"node_id": "a", "worker_instance": "a",
                                        "stop_event": stop, "fake_delay_seconds": 1})
    thread.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 3
        while "close" not in events and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "close" in events
        assert not list(service.core.run_root.glob("groups/*/stages/*/results/*.json"))
        restore.set()
        thread.join(3)
        assert not thread.is_alive()
        assert events.index("close") < events.index("rejoin")
    finally:
        stop.set()
        restore.set()
        thread.join(3)
        service.close()


def test_release_follows_resource_close(tmp_path, monkeypatch):
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=1)
    events = []
    original = module.FakeWorker

    class Tracked(original):
        def __exit__(self, *args):
            events.append("closed")

    class Client(ServiceClient):
        def call(self, method, path, body=None):
            if path.endswith("release"):
                assert events.pop() == "closed"
            response = super().call(method, path, body)
            service.tick()
            return response

    monkeypatch.setattr(module, "FakeWorker", Tracked)
    try:
        result = module.run_http_fake_worker(Client(service), node_id="a",
                                             worker_instance="a", stop_event=threading.Event())
        assert result["accepted"] == 8
    finally:
        service.close()


def test_http_workers_finish_20_rows_without_shared_state_writes(tmp_path, monkeypatch):
    module = worker_api()
    service = make_service(tmp_path, count=20)
    monkeypatch.setenv("R2V_GROUP_COORDINATOR_TOKEN", "test-token")
    with serving(service) as url:
        result = module.run_http_fake_workers(coordinator_url=url, node_id="local-cpu", workers=3)
        assert result["accepted"] == result["executed"] == 160
        paths = list(service.core.run_root.glob("groups/*/stages/*/results/*.json"))
        assert len(paths) == 160
        assert len(list(service.core.run_root.rglob("PILOT_COMPLETE"))) == 1
        assert not list(service.core.run_root.rglob("COMPLETE"))
        assert result["model_call_count"] == 0
