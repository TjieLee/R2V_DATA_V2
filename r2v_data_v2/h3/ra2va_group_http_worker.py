"""HTTP-only CPU Fake workers. No shared scheduling files or model adapters."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import threading
import time
import urllib.error
import urllib.request
import uuid

from r2v_data_v2.h3.ra2va_group_fake import FakeWorker
from r2v_data_v2.h3.ra2va_group_source import GroupTask

HEARTBEAT_SECONDS = 5
WATCHDOG_SECONDS = 30
POLL_SECONDS = 0.05


class HttpConflict(RuntimeError):
    pass


class GroupHttpClient:
    def __init__(self, base_url, bearer_token, *, timeout_seconds=3):
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        # Localhost/private routing must not accidentally use the notebook proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, method, path, body=None):
        request = urllib.request.Request(self.base_url + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.bearer_token}",
                     "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = json.load(error).get("error", str(error))
            if error.code == 409:
                raise HttpConflict(message) from error
            raise RuntimeError(f"Coordinator HTTP {error.code}: {message}") from error


class WorkerHeartbeat:
    def __init__(self, client, *, monotonic=time.monotonic):
        self.client = client
        self.monotonic = monotonic
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.expired = threading.Event()
        self.threads = []

    def start(self, session):
        self.session = {k: session[k] for k in ("session_id", "generation")}
        self.deadline = session.get("sent_at", self.monotonic()) + WATCHDOG_SECONDS
        self.threads = [threading.Thread(target=target, daemon=True)
                        for target in (self._renew, self._watchdog)]
        for thread in self.threads:
            thread.start()

    def execution_allowed(self):
        with self.lock:
            if self.monotonic() >= self.deadline:
                self.expired.set()
            return not self.expired.is_set() and not self.stopped.is_set()

    def _renew(self):
        while not self.stopped.wait(HEARTBEAT_SECONDS) and self.execution_allowed():
            sent = self.monotonic()
            try:
                self.client.call("POST", "/v1/workers/heartbeat", self.session)
            except OSError:
                continue
            except HttpConflict:
                self.expired.set()
                return
            with self.lock:
                # Expiry is sticky, including a late renewal reply.
                if self.monotonic() >= self.deadline:
                    self.expired.set()
                if not self.expired.is_set():
                    self.deadline = sent + WATCHDOG_SECONDS

    def _watchdog(self):
        while not self.stopped.wait(POLL_SECONDS):
            if not self.execution_allowed():
                return

    def stop(self):
        self.stopped.set()
        for thread in self.threads:
            thread.join(5)
            if thread.is_alive():
                raise RuntimeError("heartbeat did not stop within cleanup bound")


def _wait_execution(stop_event, heartbeat, seconds):
    deadline = time.monotonic() + seconds
    while not stop_event.is_set() and heartbeat.execution_allowed():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        stop_event.wait(min(POLL_SECONDS, remaining))
    return False


def run_http_fake_worker(client, *, node_id, worker_instance, stop_event, fake_delay_seconds=0):
    if not 0 <= fake_delay_seconds <= 2:
        raise ValueError("fake delay must be between 0 and 2 seconds")
    counters = {"executed": 0, "accepted": 0, "model_call_count": 0}
    previous = None
    while not stop_event.is_set():
        sent = time.monotonic()
        try:
            if previous is not None:
                # Execution/context and heartbeat are already closed. Ask to revoke
                # only our own stopped permission, never another Worker's lease.
                try:
                    client.call("POST", "/v1/workers/heartbeat", previous | {
                        "execution_stopped": True, "resources_closed": True})
                except HttpConflict:
                    pass
            session = client.call("POST", "/v1/workers/connect", {
                "node_id": node_id, "worker_instance": worker_instance,
                "previous_session_id": None if previous is None else previous["session_id"],
                "resources_closed": True})
        except OSError:
            stop_event.wait(POLL_SECONDS)
            continue
        previous = None
        if session["finished"]:
            return counters
        if session["session_id"] is None:
            stop_event.wait(POLL_SECONDS)
            continue
        session["sent_at"] = sent
        identity = {k: session[k] for k in ("session_id", "generation")}
        heartbeat = WorkerHeartbeat(client)
        heartbeat.start(session)
        normal_drain = False
        try:
            with FakeWorker(session["stage"]) as backend:
                task = session.get("claim")
                while not stop_event.is_set() and heartbeat.execution_allowed():
                    try:
                        task = task or client.call("POST", "/v1/tasks/claim", identity)
                    except OSError:
                        stop_event.wait(POLL_SECONDS)
                        continue
                    if task.get("task") is None:
                        if task.get("phase") in ("draining", "complete"):
                            normal_drain = True
                            break
                        task = None
                        stop_event.wait(POLL_SECONDS)
                        continue
                    started = time.monotonic()
                    if not _wait_execution(stop_event, heartbeat, fake_delay_seconds):
                        break
                    upstream = task["upstream_status"]
                    if upstream is not None and upstream != "ready":
                        status, payload = "skipped", {"fake": True, "reason": "upstream_terminal",
                                                      "upstream_status": upstream}
                    else:
                        status, payload = "ready", backend.process(GroupTask(**task["task"]))
                    counters["executed"] += 1
                    if not heartbeat.execution_allowed() or stop_event.is_set():
                        break
                    body = {k: task[k] for k in ("session_id", "task_id", "claim_token")} | {
                        "status": status, "payload": payload,
                        "elapsed_seconds": time.monotonic() - started}
                    # An uncertain result is reconciled using the same receipt.
                    # It is never a reason to execute or claim a second task.
                    while not stop_event.is_set():
                        try:
                            client.call("POST", "/v1/tasks/result", body)
                            counters["accepted"] += 1
                            break
                        except OSError:
                            stop_event.wait(POLL_SECONDS)
                    task = None
        except HttpConflict:
            pass
        finally:
            heartbeat.stop()
        # Both the context and heartbeat have stopped before release/rejoin.
        if normal_drain:
            try:
                client.call("POST", "/v1/workers/release", identity | {"resources_closed": True})
            except (OSError, HttpConflict):
                previous = identity
        else:
            previous = identity
    return counters


def _child(url, node, instance, stop, delay, reports):
    client = GroupHttpClient(url, os.environ["R2V_GROUP_COORDINATOR_TOKEN"])
    reports.put(run_http_fake_worker(client, node_id=node, worker_instance=instance,
                                     stop_event=stop, fake_delay_seconds=delay))


def run_http_fake_workers(*, coordinator_url, node_id, workers, fake_delay_seconds=0):
    if workers < 1 or not 0 <= fake_delay_seconds <= 2:
        raise ValueError("invalid bounded Fake Worker settings")
    context = mp.get_context("spawn")
    stop, reports = context.Event(), context.Queue()
    identity = uuid.uuid4().hex
    children = [context.Process(target=_child, args=(coordinator_url, node_id,
        f"{identity}-{i}", stop, fake_delay_seconds, reports)) for i in range(workers)]
    handlers = {}

    def request_stop(signum, frame):
        stop.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, request_stop)
    started = time.monotonic()
    try:
        for child in children:
            child.start()
        while any(child.is_alive() for child in children) and not stop.is_set():
            for child in children:
                child.join(0.05)
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("HTTP Fake Worker interrupted; resume original run-id")
        if stop.is_set() or any(child.exitcode != 0 for child in children):
            raise RuntimeError("HTTP Fake Worker interrupted; resume original run-id")
        results = [reports.get(timeout=3) for _ in children]
    finally:
        stop.set()
        for child in children:
            if child.pid is not None:
                child.join(2)
                if child.is_alive():
                    child.terminate()
                    child.join(2)
                if child.is_alive():
                    child.kill()
                    child.join(2)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        reports.close()
    return {key: sum(result[key] for result in results)
            for key in ("executed", "accepted", "model_call_count")} | {
                "invocation_elapsed_seconds": time.monotonic() - started, "node_id": node_id}
