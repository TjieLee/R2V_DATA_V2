"""Single-host writer for bounded Group CPU pilots; no distributed file locks."""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_production import (
    STAGES,
    TERMINAL,
    GroupCoordinator,
    StageSnapshot,
    read_json,
    try_lock,
)
from r2v_data_v2.h3.t2va_production import atomic_json

LEASE_SECONDS = 60


class ClaimConflict(ValueError):
    pass


@contextmanager
def coordinator_process_guard(local_lock_path: Path):
    path = local_lock_path.resolve()
    if path.is_relative_to(Path("/mnt/workspace")):
        raise ValueError("Coordinator process guard must use verified local storage")
    handle = try_lock(path)
    if handle is None:
        raise RuntimeError("Coordinator is already running on this host")
    try:
        yield
    finally:
        handle.close()


class HttpGroupCoordinator:
    def __init__(self, core: GroupCoordinator, *, wall_clock=time.time):
        if core.status().get("transport") != "http":
            raise ValueError("HTTP service requires HTTP transport")
        self.core = core
        self.wall_clock = wall_clock
        self.lock = threading.RLock()
        self.sessions = {}
        self.claims = {}
        self.snapshot = None
        self.closed = False
        # Recover before any HTTP request or background advance can run.
        with self.lock:
            self._restore()

    def _restore(self):
        self.snapshot = self.core.select_current()
        if self.snapshot is None:
            return
        self.core.recover(self.snapshot)
        control = self.core.status()
        for sid, member in control["members"].items():
            meta = member["http"]
            if not member["released"] and not meta.get("expired"):
                self.sessions[sid] = self.core.restore_session(self.snapshot, sid)
        dispatch = self._dispatch_state()
        for ordinal, row in dispatch["active"].items():
            if not row["http"].get("revoked"):
                self.claims[row["worker_id"]] = self.core.restore_claim(
                    self.snapshot, int(ordinal), row["worker_id"])
        self._expire()

    def _dispatch_state(self):
        return read_json(self.core.stage_root(self.snapshot) / "dispatch.json")

    def _close_handles(self):
        for handle in (*self.claims.values(), *self.sessions.values()):
            handle.close()
        self.claims.clear()
        self.sessions.clear()

    def close(self):
        with self.lock:
            self.closed = True
            self._close_handles()

    def _expire(self):
        if self.snapshot is None:
            return
        expired = []
        with self.core._checked(self.snapshot) as (control, dispatch):
            for sid, member in control["members"].items():
                meta = member["http"]
                if (not member["released"] and not meta.get("expired")
                        and meta["lease_deadline"] <= self.wall_clock()):
                    meta["expired"] = True
                    expired.append(sid)
                    for row in dispatch["active"].values():
                        if row["worker_id"] == sid:
                            row["http"]["revoked"] = True
            if expired:
                # Durable revocation precedes dropping local handle ownership.
                self.core._save_dispatch(self.snapshot, dispatch)
                atomic_json(self.core.control_path, control)
        for sid in expired:
            self._retire(sid)
        # A crash can occur after durable revocation but before this policy result.
        # Inspect only active interrupted requests, never completed media/results.
        for ordinal, row in self._dispatch_state()["active"].items():
            meta = row["http"]
            events = meta["events"]
            if meta.get("revoked") and "request_started" in events and "response_saved" not in events:
                sid = row["worker_id"]
                held = self.claims.pop(sid, None)
                if held is None:
                    held = self.core.restore_claim(self.snapshot, int(ordinal), sid)
                try:
                    self.core.publish(held, "failed", {
                        "fake": True, "failure_reason": "interrupted_request_unresolved",
                        "model_call_count": 0,
                    }, 0, receipt_metadata=meta)
                finally:
                    held.close()

    def _retire(self, sid):
        claim = self.claims.pop(sid, None)
        if claim is not None:
            claim.close()
        session = self.sessions.pop(sid, None)
        if session is not None:
            session.close()
        self.core.log("lease_revoked", self.snapshot, session_id=sid)

    def tick(self):
        with self.lock:
            if self.closed:
                return
            if self.snapshot is None:
                self._restore()
            if self.snapshot is None:
                return
            self.core.recover(self.snapshot)
            self._expire()
            if self.core.advance(self.snapshot):
                self._close_handles()
                self._restore()

    def _context(self):
        control = self.core.status()
        return {k: control[k] for k in ("group_id", "stage", "generation", "phase")} | {
            "finished": control["phase"] == "complete"
                and len(control["started_groups"]) >= self.core.max_groups,
            "lease_seconds": LEASE_SECONDS,
        }

    def _member(self, body):
        sid = body["session_id"]
        control = self.core.status()
        if body["generation"] != control["generation"]:
            raise ClaimConflict("old generation")
        member = control.get("members", {}).get(sid)
        if (member is None or member["released"] or member["http"].get("expired")
                or member["http"]["lease_deadline"] <= self.wall_clock()):
            raise ClaimConflict("session expired or released")
        return sid, member

    def _claim_wire(self, sid):
        claim = self.claims.get(sid)
        if claim is None:
            return None
        row = self._dispatch_state()["active"].get(str(claim.task.ordinal))
        if row is None:
            return None
        index = STAGES.index(self.snapshot.stage)
        upstream = None
        if index:
            previous = StageSnapshot(self.snapshot.group_id, STAGES[index - 1],
                                     self.snapshot.generation - 1, "complete")
            upstream = read_json(self.core.result_path(previous, claim.task.ordinal))["status"]
        return {"session_id": sid, "task_id": {
            "group_id": self.snapshot.group_id, "stage": self.snapshot.stage,
            "generation": self.snapshot.generation, "ordinal": claim.task.ordinal,
        }, "claim_token": row["http"]["claim_token"], "task": asdict(claim.task),
            "upstream_status": upstream, "lease_seconds": LEASE_SECONDS,
            **({"checkpoint": row["http"]["checkpoint"]} if "checkpoint" in row["http"] else {})}

    def _connect(self, body):
        node, instance = body["node_id"], body["worker_instance"]
        control = self.core.status()
        for sid, member in control.get("members", {}).items():
            meta = member["http"]
            if meta["node_id"] == node and meta["worker_instance"] == instance:
                if (not member["released"] and not meta.get("expired")
                        and meta["lease_deadline"] > self.wall_clock()):
                    self._heartbeat({"session_id": sid, "generation": control["generation"]})
                    return self._context() | {"session_id": sid, "claim": self._claim_wire(sid)}
                if not body.get("resources_closed"):
                    raise ClaimConflict("local execution must close before a new session")
        if self.snapshot is None or control["phase"] != "running":
            return self._context() | {"session_id": None, "claim": None}
        if not body.get("resources_closed"):
            raise ClaimConflict("new session requires closed previous resources")
        sid = uuid.uuid4().hex
        meta = {"node_id": node, "worker_instance": instance,
                "lease_deadline": self.wall_clock() + LEASE_SECONDS, "expired": False}
        self.sessions[sid] = self.core.join(self.snapshot, sid, member_metadata=meta)
        self.core.log("http_connected", self.snapshot, node_id=node, session_id=sid)
        return self._context() | {"session_id": sid, "claim": None}

    def _heartbeat(self, body):
        sid, _ = self._member(body)
        if body.get("execution_stopped"):
            if not body.get("resources_closed"):
                raise ClaimConflict("stopped execution must close local resources")
            with self.core._checked(self.snapshot) as (control, dispatch):
                control["members"][sid]["http"]["expired"] = True
                for row in dispatch["active"].values():
                    if row["worker_id"] == sid:
                        row["http"]["revoked"] = True
                self.core._save_dispatch(self.snapshot, dispatch)
                atomic_json(self.core.control_path, control)
            self._retire(sid)
            return self._context() | {"revoked": True}
        with self.core._checked(self.snapshot) as (control, _):
            control["members"][sid]["http"]["lease_deadline"] = self.wall_clock() + LEASE_SECONDS
            atomic_json(self.core.control_path, control)
        return self._context() | {"session_id": sid}

    def _claim(self, body):
        self._expire()
        sid, _ = self._member(body)
        existing = self._claim_wire(sid)
        if existing is not None:
            return existing
        if self.core.status()["phase"] != "running":
            return self._context() | {"task": None}
        meta = {"claim_token": uuid.uuid4().hex, "revoked": False, "events": {}}
        self.core.recover(self.snapshot)
        for row in self._dispatch_state()["active"].values():
            if row["http"].get("revoked"):
                events = row["http"]["events"]
                if "response_saved" in events:
                    meta["events"] = events
                    meta["checkpoint"] = events["response_saved"]["checkpoint"]
                break
        claim = self.core.claim(self.snapshot, sid, claim_metadata=meta)
        if claim is None:
            return self._context() | {"task": None}
        self.claims[sid] = claim
        return self._claim_wire(sid)

    def _owned_claim(self, body):
        task_id = body["task_id"]
        sid, _ = self._member({"session_id": body["session_id"],
                               "generation": task_id["generation"]})
        wire = self._claim_wire(sid)
        if (wire is None or wire["task_id"] != task_id
                or wire["claim_token"] != body["claim_token"]):
            raise ClaimConflict("claim token or task differs")
        return sid, self.claims[sid]

    def _receipt(self, body):
        task = body["task_id"]
        # Do not turn a client-supplied group/stage into an arbitrary filesystem path.
        control = self.core.status()
        if task["group_id"] not in control["started_groups"] or task["stage"] not in STAGES:
            raise ClaimConflict("unknown task")
        if type(task["generation"]) is not int or type(task["ordinal"]) is not int:
            raise ClaimConflict("invalid task identity")
        snapshot = StageSnapshot(task["group_id"], task["stage"], task["generation"], "complete")
        path = self.core.result_path(snapshot, task["ordinal"])
        if not path.exists():
            return None
        receipt = read_json(path)
        if (receipt["worker_id"] != body["session_id"]
                or receipt["http"]["claim_token"] != body["claim_token"]
                or receipt["status"] != body["status"] or receipt["payload"] != body["payload"]):
            raise ClaimConflict("terminal receipt differs")
        return {"accepted": True, "receipt": receipt}

    def _result(self, body):
        existing = self._receipt(body)
        if existing is not None:
            return existing
        if body.get("receipt_only"):
            raise ClaimConflict("terminal receipt has not been accepted")
        sid, claim = self._owned_claim(body)
        if body["status"] not in TERMINAL:
            raise ClaimConflict("invalid terminal status")
        meta = self._dispatch_state()["active"][str(claim.task.ordinal)]["http"]
        member = self.core.status()["members"][sid]["http"]
        self.core.publish(claim, body["status"], body["payload"], body["elapsed_seconds"],
                          receipt_metadata=meta | {"node_id": member["node_id"]})
        self.claims.pop(sid).close()
        return self._receipt(body)

    def _event(self, body):
        _, claim = self._owned_claim(body)
        event = body["event"]
        if event not in ("execution_started", "request_started", "response_saved"):
            raise ClaimConflict("unknown execution event")
        with self.core._checked(self.snapshot) as (_, dispatch):
            events = dispatch["active"][str(claim.task.ordinal)]["http"]["events"]
            if event == "response_saved" and ("request_started" not in events or body.get("checkpoint") is None):
                raise ClaimConflict("saved response requires prior intent and a checkpoint")
            if event in events and events[event]["checkpoint"] != body.get("checkpoint"):
                raise ClaimConflict("execution event checkpoint differs")
            events.setdefault(event, {"at": self.wall_clock(), "checkpoint": body.get("checkpoint")})
            self.core._save_dispatch(self.snapshot, dispatch)
        return {"accepted": True}

    def _release(self, body):
        sid, _ = self._member(body)
        if not body.get("resources_closed") or self._claim_wire(sid) is not None:
            raise ClaimConflict("release requires finished work and closed resources")
        self.core.acknowledge_release(self.sessions[sid])
        self.sessions.pop(sid).close()
        return self._context() | {"released": True}

    def dispatch(self, method, path, body=None):
        with self.lock:
            if self.closed:
                return 503, {"error": "Coordinator stopped"}
            try:
                if method == "GET" and path == "/v1/status":
                    return 200, self._context() | {"control": self.core.status(),
                        "progress": None if self.snapshot is None else self.core.progress(self.snapshot)}
                handlers = {
                    "/v1/workers/connect": self._connect,
                    "/v1/workers/heartbeat": self._heartbeat,
                    "/v1/workers/release": self._release,
                    "/v1/tasks/claim": self._claim,
                    "/v1/tasks/event": self._event,
                    "/v1/tasks/result": self._result,
                }
                if method != "POST" or path not in handlers:
                    return 404, {"error": "unknown route"}
                return 200, handlers[path](body or {})
            except ClaimConflict as error:
                return 409, {"error": str(error)}
            except (KeyError, TypeError) as error:
                return 400, {"error": f"invalid request: {error}"}


def build_http_server(host, port, *, service_factory, bearer_token):
    if not bearer_token:
        raise ValueError("Coordinator Bearer token is required")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            if self.headers.get("Authorization") != f"Bearer {bearer_token}":
                code, response = 401, {"error": "unauthorized"}
            else:
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(size)) if size else None
                    code, response = self.server.service.dispatch(self.command, self.path, body)
                except (ValueError, json.JSONDecodeError):
                    code, response = 400, {"error": "invalid JSON request"}
            data = json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Receipt remains durable if the client lost the response.

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def service_actions(self):
            self.service.tick()

    # TCP ownership is acquired BEFORE the factory may touch shared state.
    server = Server((host, port), Handler)
    try:
        server.service = service_factory()
    except BaseException:
        server.server_close()
        raise
    return server
