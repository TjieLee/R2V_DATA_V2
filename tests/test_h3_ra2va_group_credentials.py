"""Fresh-container credentials and private-file isolation, without models."""

import functools
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.test_h3_ra2va_cluster_dynamic import (
    REPO,
    api,
    environment,
    finish,
    launch,
    port,
    wait_for_worker_report,
)
from tests.test_h3_ra2va_group_source import publish_group


def fresh_environment(tmp_path, role="worker", node="node-b"):
    env = environment(tmp_path, role, node)
    env.pop("R2VA_TOKEN_FILE")
    env["HOME"] = str(tmp_path / node / "empty-home")
    return env


@pytest.mark.parametrize("credentials", ["shared", "environment"])
def test_fresh_nodes_start_and_resume_without_a_node_local_token(tmp_path, credentials):
    publish_group(tmp_path / "post_mask", count=20)
    worker_env = fresh_environment(tmp_path)
    worker_env["R2VA_FAKE_DELAY_SECONDS"] = ".04"
    values = api().node_configuration(REPO, {})
    values["mimo"].update(media_root=str(tmp_path), media_base_url=f"http://127.0.0.1:{port()}/")
    configuration = tmp_path / "node.json"
    configuration.write_text(json.dumps(values))
    worker_env["R2VA_CONFIGURATION"] = str(configuration)
    if credentials == "environment":
        worker_env["R2V_GROUP_COORDINATOR_TOKEN"] = "private-test-environment-value"
    coordinator_env = {**worker_env, "R2VA_ROLE": "coordinator", "R2VA_NODE_ID": "node-a",
                       "HOME": str(tmp_path / "node-a/empty-home")}
    values["mimo"]["media_base_url"] = f"http://127.0.0.1:{port()}/"
    coordinator_configuration = tmp_path / "node-a.json"
    coordinator_configuration.write_text(json.dumps(values))
    coordinator_env["R2VA_CONFIGURATION"] = str(coordinator_configuration)
    worker = launch(worker_env, "--fake")
    coordinator = None
    log = tmp_path / "coordinator.log"
    try:
        time.sleep(.2)
        assert worker.poll() is None, worker.communicate()[0]
        assert not (tmp_path / "output").exists()
        with log.open("w") as stream:
            coordinator = launch(coordinator_env, "--fake", output=stream)
        outputs = finish(worker) + wait_for_worker_report(coordinator, log)
        root = tmp_path / "output/discovery20"
        credential = root / ".ra2va-private/coordinator.token"
        if credentials == "shared":
            token = credential.read_text().strip()
            assert len(token) >= 32
            assert credential.stat().st_mode & 0o777 == 0o600
            assert credential.parent.stat().st_mode & 0o777 == 0o700
            saved_credential = credential.read_bytes()
        else:
            token = worker_env["R2V_GROUP_COORDINATOR_TOKEN"]
            assert not credential.exists()
        assert token not in outputs
        assert token not in (root / "control.json").read_text()
        assert token not in (root / "coordinator_endpoint.json").read_text()
        rows = sorted(root.glob("groups/group-000000/stages/*/results/*.json"))
        assert len(rows) == 160
        contents = {str(path): path.read_bytes() for path in rows}
        assert {json.loads(data)["http"]["node_id"] for data in contents.values()} == {"node-a", "node-b"}
        assert all(json.loads(data)["payload"].get("model_call_count", 0) == 0 for data in contents.values())
        assert (root / "groups/group-000000/PILOT_COMPLETE").exists()
        assert not (root / "groups/group-000000/COMPLETE").exists()
        coordinator.terminate()
        coordinator.wait(timeout=5)
        with log.open("a") as stream:
            coordinator = launch(coordinator_env, "--fake", output=stream)
        assert '"executed": 0' in finish(launch(worker_env, "--fake"))
        if credentials == "shared":
            assert credential.read_bytes() == saved_credential
        assert {str(path): path.read_bytes() for path in rows} == contents
    finally:
        for process in (worker, coordinator):
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.communicate(timeout=5)


def test_uncredentialed_worker_times_out_without_writing_shared_state(tmp_path):
    env = fresh_environment(tmp_path)
    env["R2VA_WAIT_SECONDS"] = ".15"
    process = launch(env, "--fake")
    output, _ = process.communicate(timeout=5)
    assert process.returncode != 0
    assert "credential" in output.lower() and "timed out" in output.lower()
    assert not (tmp_path / "output").exists()


def test_bootstrap_requires_tcp_ownership_before_publishing_any_shared_file(tmp_path):
    env = fresh_environment(tmp_path, "coordinator", "node-a")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", int(env["R2VA_COORDINATOR_PORT"])))
        occupied.listen()
        process = launch(env, "--fake")
        output, _ = process.communicate(timeout=5)
    assert process.returncode != 0
    assert "Coordinator exited" in output
    assert not (tmp_path / "output").exists()


def fetch(url, method="GET"):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        response = opener.open(urllib.request.Request(url, method=method), timeout=2)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, response.read()


def test_media_serves_assets_but_never_private_credentials_or_directory_listings(tmp_path):
    from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process

    private = tmp_path / "run/.ra2va-private"
    private.mkdir(parents=True, mode=0o700)
    token = private / "coordinator.token"
    token.write_text("test-secret-not-for-http")
    token.chmod(0o600)
    (tmp_path / "secret-link").symlink_to(token)
    (tmp_path / "directory-link").symlink_to(private, target_is_directory=True)
    (tmp_path / "frame.png").write_bytes(b"test-image-bytes")
    url = f"http://127.0.0.1:{port()}/"
    values = {"mimo": {"media_base_url": url, "media_root": str(tmp_path)}}
    process = api()._start_media(sys.executable, values, os.environ, REPO)
    try:
        assert fetch(url + "frame.png") == (200, b"test-image-bytes")
        for path in ("run/.ra2va-private/coordinator.token", "run/%2Era2va-private/coordinator.token",
                     "secret-link", "directory-link/coordinator.token", "", "run/"):
            for method in ("GET", "HEAD"):
                status, body = fetch(url + path, method)
                assert status == 404, (path, method, status)
                assert b"test-secret-not-for-http" not in body
        assert api()._start_media(sys.executable, values, os.environ, REPO, require_private=True) is None
    finally:
        stop_owned_process(process, set(), grace=1)


def test_bootstrap_does_not_reuse_or_stop_an_unrestricted_media_server(tmp_path):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        values = {"mimo": {"media_base_url": f"http://127.0.0.1:{server.server_port}/",
                           "media_root": str(tmp_path)}}
        try:
            with pytest.raises(RuntimeError, match="private"):
                api()._start_media(sys.executable, values, os.environ, REPO, require_private=True)
            assert fetch(values["mimo"]["media_base_url"])[0] == 200
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("direct", [False, True])
def test_fake_and_direct_coordinator_check_media_before_shared_token_publication(tmp_path, direct):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        env = fresh_environment(tmp_path, "coordinator", "node-a")
        values = api().node_configuration(REPO, {})
        values["mimo"].update(media_root=str(tmp_path), media_base_url=f"http://127.0.0.1:{server.server_port}/")
        config = tmp_path / "node.json"
        config.write_text(json.dumps(values))
        env["R2VA_CONFIGURATION"] = str(config)
        if direct:
            command = [sys.executable, str(REPO / "tools/run_h3_ra2va_group_production.py"), "coordinator",
                       "--output-root", str(tmp_path / "output"), "--run-id", "discovery20", "--limit", "20",
                       "--source-root", str(tmp_path / "post_mask"), "--host", "127.0.0.1",
                       "--port", env["R2VA_COORDINATOR_PORT"], "--local-lock-path", str(tmp_path / "guard.lock"),
                       "--publish-endpoint"]
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        else:
            process = launch(env, "--fake")
        try:
            output, _ = process.communicate(timeout=5)
            assert process.returncode != 0
            assert "private-file isolation" in output
            assert not (tmp_path / "output").exists()
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
            server.shutdown()
            thread.join()


def test_protected_media_reuse_accepts_url_without_a_trailing_slash(tmp_path):
    from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process

    values = {"mimo": {"media_base_url": f"http://127.0.0.1:{port()}/", "media_root": str(tmp_path)}}
    process = api()._start_media(sys.executable, values, os.environ, REPO)
    try:
        values["mimo"]["media_base_url"] = values["mimo"]["media_base_url"].rstrip("/")
        assert api()._start_media(sys.executable, values, os.environ, REPO, require_private=True) is None
    finally:
        stop_owned_process(process, set(), grace=1)


def test_direct_coordinator_sigterm_during_initialization_closes_owned_media(tmp_path):
    from r2v_data_v2.h3.t2va_full_workers import _descendants

    env = fresh_environment(tmp_path, "coordinator", "node-a")
    values = api().node_configuration(REPO, {})
    media_url = f"http://127.0.0.1:{port()}/"
    values["mimo"].update(media_root=str(tmp_path), media_base_url=media_url)
    configuration = tmp_path / "node.json"
    configuration.write_text(json.dumps(values))
    env["R2VA_CONFIGURATION"] = str(configuration)
    gate = tmp_path / "factory-entered"
    tool = REPO / "tools/run_h3_ra2va_group_production.py"
    arguments = [str(tool), "coordinator", "--output-root", str(tmp_path / "output"), "--run-id", "discovery20",
                 "--limit", "20", "--source-root", str(tmp_path / "post_mask"), "--host", "127.0.0.1",
                 "--port", env["R2VA_COORDINATOR_PORT"], "--local-lock-path", str(tmp_path / "guard.lock"),
                 "--publish-endpoint"]
    code = ("import runpy,sys,time; from pathlib import Path; "
            f"sys.path.insert(0,{str(REPO)!r}); "
            "from r2v_data_v2.h3 import ra2va_group_production as core; "
            f"gate=Path({str(gate)!r}); "
            "core.GroupCoordinator=lambda *a,**k:(gate.write_text('entered'),time.sleep(60))[0]; "
            f"sys.argv={arguments!r}; runpy.run_path({str(tool)!r},run_name='__main__')")
    log = tmp_path / "direct.log"
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, text=True)
    children = set()
    try:
        deadline = time.monotonic() + 10
        while not gate.exists() and time.monotonic() < deadline:
            assert process.poll() is None, log.read_text()
            time.sleep(.05)
        assert gate.exists()
        assert fetch(media_url + "_r2va_media_policy")[0] == 204
        children = _descendants({process.pid})
        process.terminate()
        process.communicate(timeout=10)
        with pytest.raises(OSError):
            fetch(media_url + "_r2va_media_policy")
    finally:
        if process.poll() is None:
            process.kill()
        for pid in children:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.communicate(timeout=5)
