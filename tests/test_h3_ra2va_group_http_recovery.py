from __future__ import annotations

import json
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.h3_ra2va_group_http_helpers import (
    Clock,
    claim,
    connect,
    make_service,
    post,
    release,
    request,
    result_body,
    serving,
)
from tests.test_h3_ra2va_group_source import publish_group


def complete_stage(service, *, status="ready"):
    session = connect(service, "complete")
    while task := claim(service, session):
        if task.get("task") is None:
            break
        post(service, "/v1/tasks/result", **result_body(task, status))
    service.tick()
    release(service, session)
    service.tick()


def test_accepted_receipt_reack_after_generation_advance(tmp_path):
    service = make_service(tmp_path, count=1)
    try:
        session = connect(service)
        task = claim(service, session)
        body = result_body(task)
        receipt = post(service, "/v1/tasks/result", **body)
        path = service.core.result_path(service.snapshot, 0)
        before = path.read_bytes()
        service.tick()
        release(service, session)
        service.tick()
        current = service.core.control_path.read_bytes()
        for _ in range(3):
            assert post(service, "/v1/tasks/result", **body) == receipt
        assert path.read_bytes() == before
        assert service.core.control_path.read_bytes() == current
        assert service.core.progress(service.snapshot)["ready"] == 0
    finally:
        service.close()


def test_duplicate_concurrent_results_count_once(tmp_path):
    service = make_service(tmp_path, count=1)
    with serving(service) as url:
        session = connect(service)
        body = result_body(claim(service, session))
        with ThreadPoolExecutor(4) as pool:
            responses = list(pool.map(lambda _: request(url, "/v1/tasks/result", body), range(4)))
        assert all(code == 200 for code, _ in responses)
        assert service.core.progress(service.snapshot)["ready"] == 1
        path = service.core.result_path(service.snapshot, 0)
        before = path.read_bytes()
        for field, value in [("claim_token", "wrong"), ("payload", {"different": True}),
                             ("status", "failed"), ("session_id", "wrong")]:
            assert service.dispatch("POST", "/v1/tasks/result", body | {field: value})[0] == 409
        assert path.read_bytes() == before


def test_receipt_only_cannot_publish_unaccepted_result(tmp_path):
    service = make_service(tmp_path, count=1)
    try:
        task = claim(service, connect(service))
        body = result_body(task)
        before = service._dispatch_state()
        assert service.dispatch("POST", "/v1/tasks/result", body | {"receipt_only": True})[0] == 409
        assert service._dispatch_state() == before
        receipt = post(service, "/v1/tasks/result", **body)
        path = service.core.result_path(service.snapshot, 0)
        saved = path.read_bytes()
        assert post(service, "/v1/tasks/result", **(body | {"receipt_only": True})) == receipt
        assert path.read_bytes() == saved
    finally:
        service.close()


def test_restart_restores_unexpired_claim_without_reassignment(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=2, clock=clock)
    session = connect(service)
    held = claim(service, session)
    generation = service.snapshot.generation
    service.close()
    restarted = make_service(tmp_path, clock=clock)
    try:
        other = claim(restarted, connect(restarted, "b"))
        assert other["task_id"]["ordinal"] == 1
        assert connect(restarted)["claim"] == held
        assert restarted.snapshot.generation == generation
        assert restarted.core.progress(restarted.snapshot)["active"] == 2
    finally:
        restarted.close()


def test_expired_token_rejected_and_unpublished_work_reclaimed(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    try:
        old_session = connect(service)
        old = claim(service, old_session)
        clock.now += 61
        service.tick()
        assert service.dispatch("POST", "/v1/tasks/result", result_body(old))[0] == 409
        member = service.core.status()["members"][old_session["session_id"]]
        assert member["http"]["expired"] is True and member["released"] is False
        new = claim(service, connect(service, "b"))
        assert new["task_id"] == old["task_id"]
        assert new["claim_token"] != old["claim_token"]
        post(service, "/v1/tasks/result", **result_body(new))
        assert service.core.progress(service.snapshot)["ready"] == 1
    finally:
        service.close()


def test_restart_restores_draining_members_and_barrier(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    session = connect(service)
    post(service, "/v1/tasks/result", **result_body(claim(service, session)))
    service.tick()
    assert service.core.status()["phase"] == "draining"
    service.close()
    restarted = make_service(tmp_path, clock=clock)
    try:
        restarted.tick()
        assert restarted.core.status()["phase"] == "draining"
        assert restarted.snapshot.generation == 1
        release(restarted, session)
        restarted.tick()
        assert restarted.snapshot.generation == 2
    finally:
        restarted.close()


def test_expired_session_is_revoked_not_marked_released(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    try:
        dead, live = connect(service, "dead"), connect(service, "live")
        post(service, "/v1/tasks/result", **result_body(claim(service, dead)))
        clock.now += 50
        post(service, "/v1/workers/heartbeat", **{
            k: live[k] for k in ("session_id", "generation")})
        clock.now += 11
        service.tick()
        assert service.core.status()["phase"] == "draining"
        member = service.core.status()["members"][dead["session_id"]]
        assert member["http"]["expired"] and not member["released"]
        release(service, live)
        service.tick()
        assert service.snapshot.generation == 2
    finally:
        service.close()


def test_unaccepted_old_generation_result_is_rejected(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    try:
        old = connect(service)
        task = claim(service, old)
        clock.now += 61
        service.tick()
        complete_stage(service)
        before = service.core.control_path.read_bytes()
        old_id = {k: old[k] for k in ("session_id", "generation")}
        for route, body in [
            ("/v1/tasks/result", result_body(task)),
            ("/v1/tasks/event", {k: task[k] for k in ("session_id", "claim_token", "task_id")}
                | {"event": "execution_started"}),
            ("/v1/workers/heartbeat", old_id),
            ("/v1/workers/release", old_id | {"resources_closed": True}),
        ]:
            assert service.dispatch("POST", route, body)[0] == 409
        assert service.core.control_path.read_bytes() == before
    finally:
        service.close()


def test_dynamic_groups_and_global_max_groups_survive_restart(tmp_path):
    service = make_service(tmp_path, count=1, max_groups=2)
    try:
        publish_group(service.core.source_root, "group-000001", count=1)
        for _ in range(8):
            complete_stage(service)
        assert service.snapshot.group_id == "group-000001"
        for _ in range(8):
            complete_stage(service)
        assert service.core.status()["started_groups"] == ["group-000000", "group-000001"]
    finally:
        service.close()
    publish_group(tmp_path / "post_mask", "group-000002", count=1)
    restarted = make_service(tmp_path, max_groups=2)
    try:
        restarted.tick()
        assert restarted.snapshot is None
        assert not (restarted.core.group_root("group-000002") / "inventory").exists()
    finally:
        restarted.close()


def test_terminal_failed_skipped_and_completed_run_never_reexecute(tmp_path):
    service = make_service(tmp_path, count=1)
    try:
        for i in range(8):
            complete_stage(service, status="failed" if i == 0 else "skipped")
        paths = list(service.core.run_root.glob("groups/*/stages/*/results/*.json"))
        before = {str(p): p.read_bytes() for p in paths}
        assert len(paths) == 8
    finally:
        service.close()
    publish_group(tmp_path / "post_mask", "group-000001", count=1)
    restarted = make_service(tmp_path)
    try:
        assert connect(restarted)["finished"]
        restarted.tick()
        assert all(Path(p).read_bytes() == data for p, data in before.items())
        assert len(list(restarted.core.run_root.rglob("PILOT_COMPLETE"))) == 1
        assert not list(restarted.core.run_root.rglob("COMPLETE"))
        assert not (restarted.core.group_root("group-000001") / "inventory").exists()
    finally:
        restarted.close()


def test_dispatch_and_background_tick_share_mutex(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    entered, unblock, ticked = threading.Event(), threading.Event(), threading.Event()
    original = service._connect

    def blocked(body):
        entered.set()
        assert unblock.wait(3)
        return original(body)

    monkeypatch.setattr(service, "_connect", blocked)
    thread = threading.Thread(target=connect, args=(service,))
    tick = threading.Thread(target=lambda: (service.tick(), ticked.set()))
    try:
        thread.start()
        assert entered.wait(2)
        tick.start()
        assert not ticked.wait(0.05)
        unblock.set()
        thread.join(2)
        tick.join(2)
        assert ticked.is_set() and not thread.is_alive()
    finally:
        unblock.set()
        service.close()


@pytest.mark.parametrize("saved", [False, True])
def test_interrupted_request_intent_never_blindly_reissues(tmp_path, saved):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    try:
        session = connect(service)
        task = claim(service, session)
        original_snapshot = service.snapshot
        identity = {k: task[k] for k in ("session_id", "claim_token", "task_id")}
        post(service, "/v1/tasks/event", **identity, event="request_started")
        if saved:
            post(service, "/v1/tasks/event", **identity, event="response_saved",
                 checkpoint={"fake_saved_response": "actual saved fake output"})
        clock.now += 61
        service.tick()
        if saved:
            new = claim(service, connect(service, "b"))
            assert new["checkpoint"] == {"fake_saved_response": "actual saved fake output"}
        else:
            receipt = json.loads(service.core.result_path(original_snapshot, 0).read_text())
            assert receipt["status"] == "failed"
            assert receipt["payload"]["failure_reason"] == "interrupted_request_unresolved"
            assert claim(service, connect(service, "b"))["upstream_status"] == "failed"
    finally:
        service.close()


def crashing_coordinator(tmp_path, point, reached, unblock, address):
    import r2v_data_v2.h3.ra2va_group_production as module
    from r2v_data_v2.h3.ra2va_group_http import (
        build_http_server,
        coordinator_process_guard,
    )
    original = module.atomic_json

    def write(path, value):
        if ((point == "before_dispatch" and path.name == "dispatch.json" and value["active"])
                or (point == "advance" and path.name == "control.json" and value["stage"] == "sam")):
            reached.set()
            unblock.wait(30)
        original(path, value)
        if point == "after_result" and path.parent.name == "results":
            reached.set()
            unblock.wait(30)

    with coordinator_process_guard(tmp_path / "node-a.lock"):
        server = build_http_server("127.0.0.1", 0,
            service_factory=lambda: make_service(tmp_path, count=1), bearer_token="test-token")
        module.atomic_json = write
        if point == "after_claim":
            original_claim = server.service._claim

            def stopped_claim(body):
                result = original_claim(body)
                reached.set()
                unblock.wait(30)
                return result

            server.service._claim = stopped_claim
        address.put(server.server_port)
        server.serve_forever(poll_interval=0.02)


@pytest.mark.parametrize("point", ["before_dispatch", "after_claim", "after_result", "advance"])
def test_event_controlled_coordinator_kill_and_recovery(tmp_path, point):
    context = mp.get_context("spawn")
    reached, unblock, address = context.Event(), context.Event(), context.Queue()
    process = context.Process(target=crashing_coordinator,
                              args=(tmp_path, point, reached, unblock, address))
    process.start()
    futures = ThreadPoolExecutor(1)
    try:
        port = address.get(timeout=10)
        url = f"http://127.0.0.1:{port}"
        session = request(url, "/v1/workers/connect", {
            "node_id": "a", "worker_instance": "a", "resources_closed": True})[1]
        body = {k: session[k] for k in ("session_id", "generation")}
        if point in ("before_dispatch", "after_claim"):
            pending = futures.submit(request, url, "/v1/tasks/claim", body)
        else:
            task = request(url, "/v1/tasks/claim", body)[1]
            if point == "after_result":
                pending = futures.submit(request, url, "/v1/tasks/result", result_body(task))
            else:
                request(url, "/v1/tasks/result", result_body(task))
                pending = futures.submit(request, url, "/v1/workers/release",
                                         body | {"resources_closed": True})
        assert reached.wait(10)
        process.kill()
        process.join(5)
        assert not process.is_alive()
        try:
            pending.result(timeout=4)
        except OSError:
            pass
        from r2v_data_v2.h3.ra2va_group_http import (
            build_http_server,
            coordinator_process_guard,
        )
        with coordinator_process_guard(tmp_path / "node-a.lock"):
            server = build_http_server("127.0.0.1", port,
                service_factory=lambda: make_service(tmp_path, count=1), bearer_token="test-token")
            restored = server.service
            try:
                if point == "before_dispatch":
                    assert claim(restored, connect(restored))["task_id"]["ordinal"] == 0
                elif point == "after_claim":
                    assert connect(restored)["claim"]["task_id"]["ordinal"] == 0
                    assert claim(restored, connect(restored, "b"))["task"] is None
                elif point == "after_result":
                    assert restored.core.progress(restored.snapshot)["ready"] == 1
                    assert post(restored, "/v1/tasks/result", **result_body(task))["accepted"]
                else:
                    restored.tick()
                    assert restored.snapshot.generation == 2
                    assert restored.core.progress(restored.snapshot)["pending"] == 1
            finally:
                restored.close()
                server.server_close()
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        futures.shutdown(wait=True)
        address.close()


def held_worker(url, entered, gate):
    from r2v_data_v2.h3 import ra2va_group_http_worker as module
    original = module.FakeWorker

    class Blocked(original):
        def process(self, task):
            entered.set()
            gate.wait(30)
            return super().process(task)

    module.FakeWorker = Blocked
    module.run_http_fake_worker(module.GroupHttpClient(url, "test-token"),
        node_id="killed-node-b", worker_instance="killed-b", stop_event=threading.Event())


def test_killed_worker_preserves_lease_then_other_worker_recovers(tmp_path):
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    context = mp.get_context("spawn")
    entered, gate = context.Event(), context.Event()
    with serving(service) as url:
        process = context.Process(target=held_worker, args=(url, entered, gate))
        process.start()
        try:
            assert entered.wait(10)
            old_wire = next(service._claim_wire(sid) for sid in service.claims)
            process.kill()
            process.join(5)
            assert not process.is_alive()
            alive = connect(service, "alive-node-a")
            assert claim(service, alive)["task"] is None
            clock.now += 61
            service.tick()
            new = claim(service, connect(service, "recovered-a"))
            assert new["task_id"] == old_wire["task_id"]
            assert new["claim_token"] != old_wire["claim_token"]
            post(service, "/v1/tasks/result", **result_body(new))
            assert service.dispatch("POST", "/v1/tasks/result", result_body(old_wire))[0] == 409
            assert service.core.progress(service.snapshot)["ready"] == 1
        finally:
            if process.is_alive():
                process.kill()
            process.join(5)


def test_recovered_checkpoint_skips_fake_execution(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_http_worker as module
    from tests.test_h3_ra2va_group_http_worker import ServiceClient
    clock = Clock()
    service = make_service(tmp_path, count=1, clock=clock)
    task = claim(service, connect(service))
    identity = {k: task[k] for k in ("session_id", "task_id", "claim_token")}
    post(service, "/v1/tasks/event", **identity, event="request_started")
    post(service, "/v1/tasks/event", **identity, event="response_saved", checkpoint={"saved": True})
    clock.now += 61
    service.tick()
    original = module.FakeWorker
    stop = threading.Event()

    class NoCanonicalExecution(original):
        def process(self, task):
            assert self.stage != "canonical", "saved response executed again"
            return super().process(task)

    class Client(ServiceClient):
        def call(self, method, path, body=None):
            response = super().call(method, path, body)
            if path.endswith("result"):
                stop.set()
            return response

    monkeypatch.setattr(module, "FakeWorker", NoCanonicalExecution)
    try:
        stats = module.run_http_fake_worker(Client(service), node_id="b", worker_instance="b", stop_event=stop)
        assert stats["reused_checkpoints"] == stats["accepted"] == 1
        assert stats["executed"] == stats["model_call_count"] == 0
    finally:
        service.close()


def test_stopped_request_intent_is_not_reclaimed_before_tick(tmp_path):
    service = make_service(tmp_path, count=1)
    try:
        session = connect(service)
        task = claim(service, session)
        identity = {k: task[k] for k in ("session_id", "task_id", "claim_token")}
        post(service, "/v1/tasks/event", **identity, event="request_started")
        post(service, "/v1/workers/heartbeat", **{k: session[k] for k in ("session_id", "generation")},
             execution_stopped=True, resources_closed=True)
        assert claim(service, connect(service, "other"))["task"] is None
        receipt = json.loads(service.core.result_path(service.snapshot, 0).read_text())
        assert receipt["payload"]["failure_reason"] == "interrupted_request_unresolved"
    finally:
        service.close()
