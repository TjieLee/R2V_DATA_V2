from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
from tests.test_h3_ra2va_group_source import publish_group


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_service(tmp_path, count=2, *, clock=None, max_groups=1):
    from r2v_data_v2.h3.ra2va_group_http import HttpGroupCoordinator
    source = tmp_path / "post_mask"
    if not source.exists():
        publish_group(source, count=count)
    core = GroupCoordinator(tmp_path / "run", source, 20, max_groups=max_groups,
                            transport="http", coordinator_host="node-a")
    return HttpGroupCoordinator(core, wall_clock=clock or Clock())


def post(service, route, **body):
    code, result = service.dispatch("POST", route, body)
    assert code == 200, result
    return result


def connect(service, name="a", **fields):
    return post(service, "/v1/workers/connect", node_id=name,
                worker_instance=name, resources_closed=True, **fields)


def claim(service, session):
    return post(service, "/v1/tasks/claim", **{k: session[k] for k in ("session_id", "generation")})


def result_body(task, status="ready", payload=None):
    return {k: task[k] for k in ("session_id", "task_id", "claim_token")} | {
        "status": status, "payload": {} if payload is None else payload, "elapsed_seconds": 0.01,
    }


def release(service, session):
    return post(service, "/v1/workers/release", session_id=session["session_id"],
                generation=session["generation"], resources_closed=True)


@contextmanager
def serving(service):
    from r2v_data_v2.h3.ra2va_group_http import build_http_server
    server = build_http_server("127.0.0.1", 0, service_factory=lambda: service,
                               bearer_token="test-token")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(3)
        server.server_close()
        service.close()


def request(url, route, body=None, token="test-token"):
    req = urllib.request.Request(url + route,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)
