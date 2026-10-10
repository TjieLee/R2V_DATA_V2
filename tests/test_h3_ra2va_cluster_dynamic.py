"""Per-node discovery and one-command launch, with real CPU HTTP Workers."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_h3_ra2va_group_source import publish_group

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_h3_ra2va_cluster_dynamic.sh"


def api():
    from r2v_data_v2.h3 import ra2va_group_launch
    return ra2va_group_launch


def port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def environment(tmp_path, role="worker", node="node-b"):
    env = {**os.environ, "R2V_PYTHON": sys.executable,
           "R2VA_OUTPUT_ROOT": str(tmp_path / "output"), "R2VA_RUN_ID": "discovery20",
           "R2VA_SOURCE_ROOT": str(tmp_path / "post_mask"), "R2VA_LIMIT": "20",
           "R2VA_MAX_GROUPS": "1", "R2VA_ROLE": role, "R2VA_NODE_ID": node,
           "R2VA_COORDINATOR_HOST": "127.0.0.1", "R2VA_COORDINATOR_PORT": str(port()),
           "R2VA_WAIT_SECONDS": "10", "R2VA_FAKE_DELAY_SECONDS": ".08",
           "REQUEST_WORKERS": "2", "R2VA_CLEANUP_SECONDS": "3"}
    env.pop("R2V_GROUP_COORDINATOR_TOKEN", None)
    return env


def launch(env, *args, output=None):
    return subprocess.Popen(["bash", str(SCRIPT), *args], env=env,
                            stdout=subprocess.PIPE if output is None else output,
                            stderr=subprocess.STDOUT, text=True)


def finish(process):
    try:
        output, _ = process.communicate(timeout=120)
    except subprocess.TimeoutExpired as error:
        pytest.fail((error.output or b"").decode())
    assert process.returncode == 0, output
    return output


def wait_for_worker_report(process, log):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        contents = log.read_text()
        if '"executed":' in contents:
            time.sleep(.3)
            return contents
        assert process.poll() is None, contents
        time.sleep(.05)
    pytest.fail(log.read_text())


def test_address_is_selected_locally_without_a_node_inventory(monkeypatch):
    class LocalRoute:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect(self, destination):
            assert destination == ("192.0.2.1", 9)

        def getsockname(self):
            return ("10.0.0.23", 40000)

    monkeypatch.setattr(socket, "socket", lambda *args: LocalRoute())
    assert api().local_coordinator_address() == "10.0.0.23"


def test_endpoint_is_atomic_private_and_reuses_the_existing_auth_token(tmp_path):
    root = tmp_path / "run"
    token = api().coordinator_token(root)
    api().publish_endpoint(root, "10.0.0.23", 8780, token)
    endpoint = root / "coordinator_endpoint.json"
    assert json.loads(endpoint.read_text()) == {
        "url": "http://10.0.0.23:8780", "token": token,
    }
    assert endpoint.stat().st_mode & 0o777 == 0o600
    assert api().coordinator_token(root) == token
    assert not list(root.glob("*.tmp*"))


def test_default_bash_plan_needs_no_ips_ssh_or_rank_and_writes_nothing(tmp_path):
    env = environment(tmp_path)
    for key in ("R2VA_COORDINATOR_HOST", "R2VA_NODE_ID", "REQUEST_WORKERS"):
        env.pop(key)
    before = list(tmp_path.rglob("*"))
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["role"] == "worker"
    assert plan["run_root"] == str(tmp_path / "output/discovery20")
    assert plan["workers"] == 8
    assert plan["gpu_ids"] == [str(i) for i in range(8)]
    assert plan["mimo_gpu_groups"] == ["0,1,2,3", "4,5,6,7"]
    assert plan["mimo_ports"] == [8094, 8095]
    assert plan["call_mode"] == "two_step"
    assert plan["coordinator"] is None
    assert not any(flag in result.stdout for flag in ("--rank", "--world-size", "ssh ", "--seed"))
    assert list(tmp_path.rglob("*")) == before


def test_coordinator_dry_run_keeps_pilot_budget_and_uses_auto_address(tmp_path):
    env = environment(tmp_path, role="coordinator")
    env.pop("R2VA_COORDINATOR_HOST")
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    command = plan["coordinator"]
    assert command[command.index("--host") + 1] == "auto"
    assert command[command.index("--limit") + 1] == "20"
    assert command[command.index("--max-groups") + 1] == "1"
    assert command[command.index("--local-lock-path") + 1] == "/tmp/discovery20.coordinator.lock"
    assert "--publish-endpoint" in command
    assert not (tmp_path / "output").exists()


def test_entry_copied_to_parallel_scripts_uses_the_confirmed_server_checkout(tmp_path):
    entry = tmp_path / "parallel_scripts" / SCRIPT.name
    entry.parent.mkdir()
    shutil.copyfile(SCRIPT, entry)
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    env = {**os.environ, "R2V_PYTHON": str(python)}
    env.pop("R2VA_REPO_ROOT", None)
    result = subprocess.run(["bash", str(entry), "--dry-run"], env=env,
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "/mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_h3_ra2va_cluster_node.py", "--dry-run"]


def test_worker_waits_without_creating_global_state_or_becoming_coordinator(tmp_path):
    env = environment(tmp_path)
    env["R2VA_WAIT_SECONDS"] = ".15"
    process = launch(env, "--fake")
    output, _ = process.communicate(timeout=5)
    assert process.returncode != 0
    assert "Coordinator discovery timed out" in output
    assert not (tmp_path / "output").exists()


def test_occupied_coordinator_port_cannot_publish_discovery_or_control(tmp_path):
    env = environment(tmp_path, role="coordinator")
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", int(env["R2VA_COORDINATOR_PORT"])))
        occupied.listen()
        process = launch(env, "--fake")
        output, _ = process.communicate(timeout=5)
    assert process.returncode != 0
    assert "Coordinator exited" in output
    assert not (tmp_path / "output/discovery20/control.json").exists()
    assert not (tmp_path / "output/discovery20/coordinator_endpoint.json").exists()


def test_two_independent_node_launchers_discover_same_run_and_resume_without_reexecution(tmp_path):
    publish_group(tmp_path / "post_mask", count=21)
    worker_env = environment(tmp_path)
    coordinator_env = {**worker_env, "R2VA_ROLE": "coordinator", "R2VA_NODE_ID": "node-a"}
    worker = launch(worker_env, "--fake")
    log = tmp_path / "coordinator.log"
    with log.open("w") as stream:
        coordinator = launch(coordinator_env, "--fake", output=stream)
    try:
        outputs = finish(worker) + wait_for_worker_report(coordinator, log)
        # Keep HTTP available after local completion for delayed remote clients.
        assert coordinator.poll() is None
        late = launch(worker_env, "--fake")
        assert '"executed": 0' in finish(late)
        coordinator.terminate()
        coordinator.wait(timeout=5)
        assert coordinator.returncode == 130
        root = tmp_path / "output/discovery20"
        endpoint = json.loads((root / "coordinator_endpoint.json").read_text())
        assert endpoint["token"] not in outputs
        rows = sorted(root.glob("groups/group-000000/stages/*/results/*.json"))
        assert len(rows) == 160
        contents = {str(path): path.read_bytes() for path in rows}
        # The persisted claim receipt identifies both actual HTTP executors.
        receipts = [json.loads(path.read_text())["http"] for path in rows]
        assert len(receipts) == 160
        assert {row["node_id"] for row in receipts} == {"node-a", "node-b"}
        assert (root / "groups/group-000000/PILOT_COMPLETE").exists()
        assert not (root / "groups/group-000000/COMPLETE").exists()
        assert not (root / "groups/group-000001").exists()
        resume_log = tmp_path / "resumed.log"
        with resume_log.open("w") as stream:
            resumed = launch(coordinator_env, "--fake", output=stream)
        try:
            assert '"executed": 0' in wait_for_worker_report(resumed, resume_log)
        finally:
            resumed.terminate()
            resumed.wait(timeout=5)
        assert {str(path): path.read_bytes() for path in rows} == contents
        assert json.loads((root / "coordinator_endpoint.json").read_text()) == endpoint
        assert all(json.loads(data)["payload"].get("model_call_count", 0) == 0 for data in contents.values())
    finally:
        for process in (worker, coordinator):
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)


def test_hangup_uses_the_same_owned_cleanup_as_interrupt(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: observed.append(sig))
    monkeypatch.setattr(api(), "wait_for_endpoint", lambda *args: (_ for _ in ()).throw(TimeoutError("probe")))
    with pytest.raises(TimeoutError, match="probe"):
        api().launch_node(REPO, environment(tmp_path), fake=True)
    assert signal.SIGHUP in observed


def test_native_configuration_uses_env_settings_without_changing_model_semantics():
    values = api().node_configuration(REPO, {"GPU_IDS": "0,1,2,3", "MIMO_GPU_GROUPS": "0,1,2,3",
                                           "MIMO_PORTS": "8098", "ASR_BATCH_SIZE": "8"})
    assert values["gpu_ids"] == ["0", "1", "2", "3"]
    assert values["mimo"]["gpu_groups"] == ["0,1,2,3"]
    assert values["mimo"]["ports"] == [8098]
    assert values["asr"]["environment"]["QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE"] == "8"
    assert values["mimo"]["media_base_url"] == "http://127.0.0.1:8766/"


def test_node_environment_clears_proxies_and_preserves_per_backend_isolation(tmp_path):
    env = api()._runtime_environment(REPO, tmp_path, {
        "HTTP_PROXY": "http://not-local.invalid:80", "https_proxy": "http://not-local.invalid:80",
        "CUDA_VISIBLE_DEVICES": "7", "SAM_DEPS_ROOT": "/node/audio_deps"})
    assert "HTTP_PROXY" not in env and "https_proxy" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["NO_PROXY"] == env["no_proxy"] == "*"
    assert env["R2VA_RUN_ROOT"] == str(tmp_path)
    assert env["SAM_AUDIO_RUNTIME_PYTHONPATH"].startswith("/node/audio_deps/sam-audio-src:")
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"


def test_custom_native_configuration_is_not_rewritten_by_eight_gpu_defaults(tmp_path):
    expected = json.loads((REPO / "docs/examples/ra2va_group_native_node.json").read_text())
    path = tmp_path / "node.json"
    path.write_text(json.dumps(expected))
    assert api().node_configuration(REPO, {"R2VA_CONFIGURATION": str(path), "GPU_IDS": "7"}) == expected


def test_media_is_started_once_and_existing_server_is_not_owned(tmp_path):
    from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process

    values = {"mimo": {"media_base_url": f"http://127.0.0.1:{port()}/", "media_root": str(tmp_path)}}
    first = api()._start_media(sys.executable, values, os.environ, REPO)
    try:
        assert first.poll() is None
        assert api()._start_media(sys.executable, values, os.environ, REPO) is None
        assert first.poll() is None
    finally:
        stop_owned_process(first, set(), grace=1)
    assert first.poll() is not None
