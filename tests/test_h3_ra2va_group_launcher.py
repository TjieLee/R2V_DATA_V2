"""Run the Bash wrapper with local subprocess doubles, never SSH or models."""

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NAME = "run_h3_ra2va_group_production.sh"
TOKEN = 'private-launch-token-"quoted"-\\value'
OUTPUT = "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA"
DEPS = "/mnt/workspace/litengjie/data/audio_deps"


def invoke(script, env, *args):
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )


@pytest.fixture
def sandbox(tmp_path):
    checkout = tmp_path / "shared checkout"
    script = checkout / "scripts" / NAME
    script.parent.mkdir(parents=True)
    source = REPO / "scripts" / NAME
    if source.exists():
        shutil.copyfile(source, script)
    binary = tmp_path / "bin"
    binary.mkdir()
    common = (
        f"#!{sys.executable}\n"
        + """import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["STUB_ROOT"])
def record(name, data):
    (root / (name + ".json")).write_text(json.dumps(data))
def event(name):
    with (root / "events").open("a") as stream:
        stream.write(name + "\\n")
"""
    )
    programs = {
        "python": """if sys.argv[1] == "-c":
    event("preflight")
    sys.exit(int(os.environ.get("PORT_BUSY", "0")))
args = sys.argv[1:]
kind = args[1]
name = "coordinator" if kind == "coordinator" else "worker-" + args[args.index("--node-id") + 1]
(root / (name + ".pid")).write_text(str(os.getpid()))
environment = {k: os.environ.get(k) for k in (
    "R2VA_RUN_ROOT", "PYTHONPATH", "SAM_AUDIO_RUNTIME_PYTHONPATH", "HOSTNAME",
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NO_PROXY", "no_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    "CUDA_VISIBLE_DEVICES",
)}
data = {"args": args, "cwd": os.getcwd(), "env": environment,
        "token_matches": os.environ.get("R2V_GROUP_COORDINATOR_TOKEN") == os.environ["EXPECTED_TOKEN"]}
if kind == "worker":
    data["config"] = json.loads(Path(args[args.index("--configuration") + 1]).read_text())
record(name, data)
event(name + "-start")
child = None
def stop(*unused):
    if child is not None:
        child.terminate()
        child.wait(timeout=2)
    event(name + "-closed")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
if kind == "coordinator":
    if os.environ.get("COORDINATOR_STATUS"):
        sys.exit(int(os.environ["COORDINATOR_STATUS"]))
    (root / "coordinator-ready").touch()
    while True: time.sleep(.01)
mode = os.environ.get("WORKER_MODE", "success")
if mode == "wait" or (mode == "remote-failure" and name == "worker-local"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / (name + "-child.pid")).write_text(str(child.pid))
    (root / (name + "-ready")).touch()
    while True: time.sleep(.01)
if mode == "remote-failure":
    deadline = time.monotonic() + 3
    while not (root / "worker-local-ready").exists() and time.monotonic() < deadline:
        time.sleep(.01)
time.sleep(.1)
event(name + "-closed")
sys.exit(7 if mode == "remote-failure" else 0)
""",
        "curl": """record("readiness", {"args": sys.argv[1:],
    "token_matches": sys.stdin.read().strip() == "Authorization: Bearer " + os.environ["EXPECTED_TOKEN"]})
sys.exit(0 if (root / "coordinator-ready").exists() and not os.environ.get("NOT_READY") else 1)
""",
        "ssh": """record("ssh", {"args": sys.argv[1:]})
env = dict(os.environ)
for key in ("R2V_GROUP_COORDINATOR_TOKEN", "PYTHONPATH", "R2VA_RUN_ROOT"):
    env.pop(key, None)
os.execve("/bin/bash", ["bash", "-c", sys.argv[-1]], env)
""",
    }
    for name, body in programs.items():
        path = binary / name
        path.write_text(common + body)
        path.chmod(0o755)
    configuration = json.loads(
        (REPO / "docs/examples/ra2va_group_native_node.json").read_text()
    )
    for node, gpu_ids in (
        ("a", ["0", "2", "4", "6"]),
        ("b", [str(i) for i in range(8)]),
    ):
        values = {
            **configuration,
            "gpu_ids": gpu_ids,
            "mimo": {
                **configuration["mimo"],
                "gpu_groups": ["0,1,2,3", "4,5,6,7"],
                "ports": [8094, 8095],
            },
        }
        (checkout / f"node {node}.json").write_text(json.dumps(values))
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "R2V_PYTHON": str(binary / "python"),
        "STUB_ROOT": str(tmp_path),
        "EXPECTED_TOKEN": TOKEN,
        "R2V_GROUP_COORDINATOR_TOKEN": TOKEN,
        "COORDINATOR_STARTUP_POLLS": "10",
        "LAUNCHER_POLL_SECONDS": "0.02",
        "CLEANUP_GRACE_SECONDS": "3",
        "CUDA_VISIBLE_DEVICES": "wrong-inherited-gpu",
    }
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env[key] = "http://wrong-proxy.invalid:3128"
    args = [
        "--run-id",
        "pilot10",
        "--limit",
        "10",
        "--host",
        "node-a.internal",
        "--output-root",
        str(tmp_path / "output"),
        "--node",
        "local",
        "node a.json",
        "4",
        "--node",
        "worker-b",
        "node b.json",
        "8",
    ]
    yield script, env, args, tmp_path
    for path in tmp_path.glob("*.pid"):
        try:
            os.kill(int(path.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def read(root, name):
    return json.loads((root / (name + ".json")).read_text())


def value(args, flag):
    return args[args.index(flag) + 1]


def assert_stopped(root):
    for path in root.glob("*.pid"):
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


def test_dry_run_prints_native_local_remote_plan_without_side_effects(sandbox):
    script, env, args, root = sandbox
    before = sorted(str(path) for path in root.rglob("*"))
    result = invoke(script, env, "--dry-run", *args)
    assert result.returncode == 0, result.stderr
    assert TOKEN not in result.stdout + result.stderr
    plans = [shlex.split(line.split(": ", 1)[1]) for line in result.stdout.splitlines()]
    coordinator, local, remote = plans
    assert coordinator[:3] == [
        env["R2V_PYTHON"],
        "tools/run_h3_ra2va_group_production.py",
        "coordinator",
    ]
    assert value(coordinator, "--execution-mode") == "native"
    assert value(coordinator, "--output-root") == str(root / "output")
    assert value(coordinator, "--local-lock-path") == "/tmp/pilot10.coordinator.lock"
    assert value(coordinator, "--limit") == "10"
    assert value(coordinator, "--max-groups") == "1"
    assert "--native" in local
    assert value(local, "--configuration") == "node a.json"
    assert value(local, "--workers") == "4"
    assert value(local, "--coordinator-url") == "http://node-a.internal:8780"
    assert remote[0] == "ssh" and remote[-2] == "worker-b"
    assert TOKEN not in remote[-1]
    remote_args = shlex.split(remote[-1])
    assert str(script.parent.parent) in remote_args
    assert value(remote_args, "--configuration") == "node b.json"
    assert value(remote_args, "--workers") == "8"
    assert value(remote_args, "--coordinator-url") == "http://node-a.internal:8780"
    assert str(root / "output/pilot10") in remote_args
    assert not any(
        flag in result.stdout
        for flag in ("--rank", "--world-size", "--shard", "--seed")
    )
    assert sorted(str(path) for path in root.rglob("*")) == before


def test_default_root_and_resume_keep_the_explicit_run_identity(sandbox):
    script, env, args, _ = sandbox
    index = args.index("--output-root")
    del args[index : index + 2]
    env.pop("R2V_GROUP_COORDINATOR_TOKEN")
    first = invoke(script, env, "--dry-run", *args)
    second = invoke(script, env, "--dry-run", *args)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    coordinator = shlex.split(first.stdout.splitlines()[0].split(": ", 1)[1])
    assert value(coordinator, "--output-root") == OUTPUT
    assert value(coordinator, "--run-id") == "pilot10"
    assert OUTPUT + "/pilot10" in first.stdout


@pytest.mark.parametrize(
    "flag,replacement",
    [
        ("--run-id", None),
        ("--limit", None),
        ("--host", None),
        ("--run-id", "../escape"),
        ("--limit", "0"),
        ("--limit", "-1"),
        ("--rank", "0"),
        ("--port", "65536"),
        ("--max-groups", "0"),
    ],
)
def test_missing_or_unbounded_selection_cannot_launch(sandbox, flag, replacement):
    script, env, args, root = sandbox
    if flag not in args:
        args += [flag, replacement]
    else:
        index = args.index(flag)
        if replacement is None:
            del args[index : index + 2]
        else:
            args[index + 1] = replacement
    result = invoke(script, env, "--dry-run", *args)
    assert result.returncode != 0
    assert not (root / "events").exists()


@pytest.mark.parametrize(
    "change", ["loopback", "wildcard", "duplicate", "no-local", "zero-workers"]
)
def test_ambiguous_node_topology_cannot_launch(sandbox, change):
    script, env, args, root = sandbox
    if change in {"loopback", "wildcard"}:
        args[args.index("--host") + 1] = (
            "127.0.0.1" if change == "loopback" else "0.0.0.0"
        )
    elif change == "duplicate":
        args += ["--node", "worker-b", "node b.json", "8"]
    elif change == "no-local":
        args[args.index("local")] = "worker-a"
    else:
        args[args.index("4")] = "0"
    assert invoke(script, env, "--dry-run", *args).returncode != 0
    assert not (root / "events").exists()


def test_native_json_and_safe_runtime_environment_reach_both_nodes(sandbox):
    script, env, args, root = sandbox
    result = invoke(script, env, *args)
    assert result.returncode == 0, result.stderr
    assert TOKEN not in result.stdout + result.stderr + json.dumps(read(root, "ssh"))
    for node, workers, gpu_ids in (
        ("local", "4", ["0", "2", "4", "6"]),
        ("worker-b", "8", [str(i) for i in range(8)]),
    ):
        call = read(root, "worker-" + node)
        assert call["cwd"] == str(script.parent.parent)
        assert call["token_matches"]
        assert value(call["args"], "--workers") == workers
        assert value(call["args"], "--coordinator-url") == "http://node-a.internal:8780"
        assert call["config"]["gpu_ids"] == gpu_ids
        assert call["config"]["mimo"]["gpu_groups"] == ["0,1,2,3", "4,5,6,7"]
        assert call["config"]["mimo"]["media_base_url"] == "http://127.0.0.1:8766/"
        child_env = call["env"]
        assert child_env["R2VA_RUN_ROOT"] == str(root / "output/pilot10")
        assert child_env["HF_HUB_OFFLINE"] == child_env["TRANSFORMERS_OFFLINE"] == "1"
        assert child_env["SAM_AUDIO_RUNTIME_PYTHONPATH"] == ":".join(
            DEPS + "/" + part
            for part in (
                "sam-audio-src",
                "perception-models-src",
                "dacvae-src",
                "sam-audio-pydeps",
            )
        )
        assert (
            child_env["PYTHONPATH"]
            == str(script.parent.parent)
            + ":"
            + child_env["SAM_AUDIO_RUNTIME_PYTHONPATH"]
        )
        assert (
            child_env["NO_PROXY"]
            == child_env["no_proxy"]
            == "127.0.0.1,localhost,::1,node-a.internal"
        )
        assert child_env["CUDA_VISIBLE_DEVICES"] is None
        assert all(
            child_env[key] is None
            for key in (
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            )
        )
    assert read(root, "coordinator")["token_matches"]
    assert read(root, "readiness")["token_matches"]
    assert TOKEN not in " ".join(read(root, "readiness")["args"])
    # Ignore ~/.curlrc, which could otherwise enable logging of auth headers.
    assert read(root, "readiness")["args"][0] == "--disable"
    assert_stopped(root)


@pytest.mark.parametrize("failure", ["PORT_BUSY", "COORDINATOR_STATUS", "NOT_READY"])
def test_coordinator_startup_failure_never_launches_workers(sandbox, failure):
    script, env, args, root = sandbox
    result = invoke(script, {**env, failure: "17"}, *args)
    assert result.returncode != 0
    assert not list(root.glob("worker-*.json"))
    assert not (root / "ssh.json").exists()
    assert_stopped(root)


def test_missing_token_never_starts_any_process(sandbox):
    script, env, args, root = sandbox
    env.pop("R2V_GROUP_COORDINATOR_TOKEN")
    assert invoke(script, env, *args).returncode != 0
    assert not (root / "events").exists()


def test_remote_failure_is_nonzero_and_waits_for_local_native_cleanup(sandbox):
    script, env, args, root = sandbox
    result = invoke(script, {**env, "WORKER_MODE": "remote-failure"}, *args)
    assert result.returncode == 7, result.stderr
    events = (root / "events").read_text().splitlines()
    assert events.index("worker-local-closed") < events.index("coordinator-closed")
    assert_stopped(root)


def test_local_only_launch_and_resume_preserve_existing_run_files(sandbox):
    script, env, args, root = sandbox
    del args[-4:]
    args[args.index("--host") + 1] = "127.0.0.1"
    stage = root / "output/pilot10/groups/existing-stage"
    stage.mkdir(parents=True)
    sentinel = stage / "receipt.json"
    sentinel.write_text('{"historical": true}')
    first = invoke(script, env, *args)
    assert first.returncode == 0, first.stderr
    initial = read(root, "coordinator")["args"]
    second = invoke(script, env, *args)
    assert second.returncode == 0, second.stderr
    assert read(root, "coordinator")["args"] == initial
    assert sentinel.read_text() == '{"historical": true}'
    assert not (root / "ssh.json").exists()
    assert_stopped(root)


def test_xtrace_cannot_disclose_token(sandbox):
    script, env, args, _ = sandbox
    result = subprocess.run(
        ["bash", "-x", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert TOKEN not in result.stdout + result.stderr


def test_coordinator_death_stops_workers_and_returns_nonzero(sandbox):
    script, env, args, root = sandbox
    process = subprocess.Popen(
        ["bash", str(script), *args],
        env={**env, "WORKER_MODE": "wait"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 6
        while (
            time.monotonic() < deadline
            and not (root / "worker-worker-b-ready").exists()
        ):
            assert process.poll() is None, process.communicate()
            time.sleep(0.02)
        assert (root / "worker-worker-b-ready").exists()
        os.kill(int((root / "coordinator.pid").read_text()), signal.SIGTERM)
        _, stderr = process.communicate(timeout=8)
        assert process.returncode != 0, stderr
        assert_stopped(root)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)


def test_term_closes_remote_control_channel_and_preserves_unrelated_process(sandbox):
    script, env, args, root = sandbox
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    process = subprocess.Popen(
        ["bash", str(script), *args],
        env={**env, "WORKER_MODE": "wait"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 6
        while (
            time.monotonic() < deadline
            and not (root / "worker-worker-b-ready").exists()
        ):
            assert process.poll() is None, process.communicate()
            time.sleep(0.02)
        assert (root / "worker-worker-b-ready").exists()
        process.terminate()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 143, stderr
        assert TOKEN not in stdout + stderr
        assert foreign.poll() is None
        events = (root / "events").read_text().splitlines()
        assert events.index("worker-worker-b-closed") < events.index(
            "coordinator-closed"
        )
        assert events.index("worker-local-closed") < events.index("coordinator-closed")
        assert_stopped(root)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
        foreign.terminate()
        foreign.wait(timeout=3)
