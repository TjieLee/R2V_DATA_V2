from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import threading
import time
from http.client import IncompleteRead

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


@pytest.mark.parametrize("lost_route", ["connect", "claim", "result"])
def test_truncated_reply_reconnects_without_duplicate_work(tmp_path, monkeypatch, lost_route):
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=1)
    client = module.GroupHttpClient("http://unused", "test-token")
    dropped, errors = [], []
    stop = threading.Event()

    class Reply:
        def __init__(self, response, truncate):
            self.response, self.truncate = response, truncate

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            if self.truncate:
                raise IncompleteRead(b'{"session_id":', 20)
            return json.dumps(self.response).encode()

    class Opener:
        def open(self, req, timeout):
            path = "/" + req.full_url.split("/", 3)[3]
            body = json.loads(req.data)
            code, response = service.dispatch(req.method, path, body)
            assert code == 200, response
            truncate = path.endswith("/" + lost_route) and not dropped
            if truncate:
                dropped.append(response)
            return Reply(response, truncate)

    client.opener = Opener()

    def run():
        try:
            result.update(module.run_http_fake_worker(client, node_id="a",
                worker_instance="a", stop_event=stop))
        except (OSError, RuntimeError, IncompleteRead, AssertionError) as error:
            errors.append(error)

    result = {}
    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while thread.is_alive() and time.monotonic() < deadline:
            service.tick()
            time.sleep(0.01)
        assert not thread.is_alive()
        assert errors == []
        assert dropped and result["executed"] == result["accepted"] == 8
    finally:
        stop.set()
        thread.join(3)
        service.close()


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


@pytest.mark.parametrize("accepted_before_outage", [False, True])
def test_result_outage_closes_context_and_only_reconciles_receipt(
        tmp_path, monkeypatch, accepted_before_outage):
    module = timings(monkeypatch)
    service = make_service(tmp_path, count=1)
    stop, restore, entered, closed = (threading.Event() for _ in range(4))
    beats, errors, result, submissions = [], [], {}, []
    original_beat, original_fake = module.WorkerHeartbeat, module.FakeWorker

    class Beat(original_beat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            beats.append(self)

    class Tracked(original_fake):
        def __exit__(self, *args):
            closed.set()
            return super().__exit__(*args)

    class Outage(ServiceClient):
        def call(self, method, path, body=None):
            if path.endswith("heartbeat") and not restore.is_set():
                raise OSError("offline")
            if path.endswith("result"):
                if not restore.is_set():
                    if accepted_before_outage and not submissions:
                        super().call(method, path, body)
                    submissions.append(body)
                    entered.set()
                    raise OSError("result reply unavailable")
                assert closed.is_set()
                assert body.get("receipt_only") is True
                try:
                    return super().call(method, path, body)
                finally:
                    stop.set()
            return super().call(method, path, body)

    def run():
        try:
            result.update(module.run_http_fake_worker(Outage(service), node_id="a",
                worker_instance="a", stop_event=stop))
        except (OSError, RuntimeError, IncompleteRead, AssertionError) as error:
            errors.append(error)

    monkeypatch.setattr(module, "WorkerHeartbeat", Beat)
    monkeypatch.setattr(module, "FakeWorker", Tracked)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2)
        assert beats[0].expired.wait(2)
        assert closed.wait(1), "result retry held the execution context after watchdog expiry"
        restore.set()
        thread.join(3)
        assert not thread.is_alive() and errors == []
        assert result["accepted"] == int(accepted_before_outage)
        assert service.core.progress(service.snapshot)["ready"] == int(accepted_before_outage)
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


def _pool_waiter(url, node, instance, stop, delay, reports):
    if hasattr(stop, "_cond"):
        reports.put((os.getpid(), True))
    else:
        original = stop.is_set
        announced = False

        def at_wait():
            nonlocal announced
            if not announced:
                reports.put((os.getpid(), False))
                announced = True
            return original()

        stop.is_set = at_wait
    stop.wait(30)


def _launcher_with_killable_waiter(shared_event, reports):
    module = worker_api()
    context = mp.get_context("spawn")

    class Context:
        def Event(self):
            return shared_event

        def Queue(self):
            return reports

        def __getattr__(self, name):
            return getattr(context, name)

    module._child = _pool_waiter
    module.mp.get_context = lambda method: Context()
    try:
        module.run_http_fake_workers(coordinator_url="unused", node_id="a", workers=1)
    except RuntimeError:
        pass  # The launcher owns and has already closed the reports queue.


def test_launcher_cleanup_after_killing_waiting_child_is_bounded():
    context = mp.get_context("spawn")
    shared_event, reports = context.Event(), context.Queue()
    launcher = context.Process(target=_launcher_with_killable_waiter, args=(shared_event, reports))
    launcher.start()
    child_pid = None
    try:
        child_pid, uses_event = reports.get(timeout=8)
        if uses_event:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if shared_event._cond._sleeping_count.acquire(False):
                    shared_event._cond._sleeping_count.release()
                    break
                time.sleep(0.005)
            else:
                pytest.fail("child did not enter the shared Event wait")
        os.kill(child_pid, signal.SIGKILL)
        launcher.join(3)
        assert not launcher.is_alive(), "launcher blocked while stopping a killed Event waiter"
        assert launcher.exitcode == 0
    finally:
        if launcher.is_alive():
            launcher.kill()
        launcher.join(3)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reports.close()
