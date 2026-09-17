"""Exercise the real shell launcher without models, GPUs, or HTTP services."""

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
NAME = "run_h3_t2va_full_production.sh"


def invoke(script, env, *args):
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.fixture
def sandbox(tmp_path):
    script = tmp_path / "scripts" / NAME
    script.parent.mkdir()
    source = REPO / "scripts" / NAME
    if source.exists():
        shutil.copyfile(source, script)
    binary = tmp_path / "bin"
    binary.mkdir()
    venv = tmp_path / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "activate").write_text(f'export PATH="{binary}:$PATH"\n')
    (tmp_path / "server_env.sh").write_text(
        "export http_proxy=http://external.invalid:3128\n"
        "export HTTPS_PROXY=http://secure.invalid:3128\n"
        "export NO_PROXY=existing.internal\n"
        "export no_proxy=lower.internal\n"
    )
    common = (
        f"#!{sys.executable}\n"
        + """import json, os, signal, subprocess, sys, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
def event(value):
    with (root / "events").open("a") as f:
        f.write(value + "\\n")
def pid(name):
    (root / (name + ".pid")).write_text(str(os.getpid()))
    (root / (name + ".session.json")).write_text(json.dumps({
        "pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0),
        "stdin_is_devnull": os.fstat(0).st_rdev == os.stat("/dev/null").st_rdev,
    }))
def stubborn_child(name):
    child_ready = root / (name + "-child-ready")
    worker = subprocess.Popen([sys.executable, "-c",
        "import signal,time,sys; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).touch(); time.sleep(60)", str(child_ready)])
    (root / (name + ".pid")).write_text(str(worker.pid))
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while not child_ready.exists(): time.sleep(.01)
    (root / (name + "-ready")).touch()
    while True: time.sleep(.01)
"""
    )
    programs = {
        "sglang": """pid("mimo")
event("mimo-start")
if os.environ.get("FORCE_STAGE") == "mimo": stubborn_child("mimo-worker")
def stop(*args):
    event("mimo-stop")
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True: time.sleep(.01)
""",
        "curl": """event("curl:" + json.dumps(sys.argv[1:]))
deadline = time.monotonic() + 3
while not (root / "mimo.pid").exists() and time.monotonic() < deadline:
    time.sleep(.01)
mode = os.environ.get("READY_MODE", "ok")
if mode == "dead":
    os.kill(int((root / "mimo.pid").read_text()), signal.SIGTERM)
    time.sleep(.05)
if mode == "timeout": sys.exit(1)
if sys.argv[-1].endswith("/model_info") and not (root / "polled").exists():
    (root / "polled").touch()
    sys.exit(1)
sys.exit(0)
""",
        "python": """if sys.argv[1] == "-c":
    sys.exit(int(os.environ.get("PORT_BUSY", "0")))
pid("runner")
event("runner-start")
(root / "invocation.json").write_text(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}))
if os.environ.get("RUN_MODE") == "wait":
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "worker.pid").write_text(str(worker.pid))
    def stop(*args):
        event("runner-term")
        worker.terminate()
        worker.wait(timeout=3)
        event("workers-stopped")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    (root / "runner-ready").touch()
    while True: time.sleep(.01)
if os.environ.get("FORCE_STAGE") == "runner": stubborn_child("worker")
time.sleep(.1)
event("runner-exit")
sys.exit(int(os.environ.get("RUN_STATUS", "0")))
""",
        # macOS has no util-linux command. This test shim performs the actual
        # POSIX session syscall and exec without forking or changing PID.
        "setsid": "os.setsid()\nos.execvp(sys.argv[1], sys.argv[1:])\n",
    }
    for name, body in programs.items():
        path = binary / name
        if name == "setsid" and (native_setsid := shutil.which(name)):
            path.symlink_to(native_setsid)
            continue
        path.write_text(common + body)
        path.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "GPU_IDS",
            "SHARDS",
            "SHARD_END",
            "REQUEST_WORKERS",
            "CUDA_VISIBLE_DEVICES",
        }
    }
    env.update(
        FAKE_ROOT=str(tmp_path),
        SGLANG_ENV=str(tmp_path),
        PRODUCTION_ROOT=str(tmp_path / "out"),
        SHARDS="59,12,87",
        MIMO_STARTUP_POLLS="3",
        MIMO_POLL_INTERVAL="0.01",
        CLEANUP_GRACE_SECONDS="2",
    )
    yield script, env, tmp_path
    for name in ("runner", "worker", "mimo", "mimo-worker"):
        path = tmp_path / f"{name}.pid"
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def events(root):
    return (root / "events").read_text().splitlines()


def test_no_shell_job_control():
    assert "set -m" not in (REPO / "scripts" / NAME).read_text()


def test_missing_setsid_only_blocks_execution(sandbox):
    script, env, root = sandbox
    binary = root / "bin"
    (binary / "setsid").unlink()
    for name in ("bash", "dirname", "mkdir", "date", "hostname", "sleep"):
        (binary / name).symlink_to(shutil.which(name))
    env = {**env, "PATH": str(binary)}
    assert invoke(script, env, "--dry-run").returncode == 0
    result = invoke(script, env)
    assert result.returncode == 2
    assert "setsid is required" in result.stderr
    assert not (root / "events").exists()


def test_dry_run_exact_accepted_server_and_full_cli(sandbox):
    script, env, root = sandbox
    (root / "server_env.sh").write_text('touch "$FAKE_ROOT/sourced"; exit 99\n')
    result = invoke(script, env, "--dry-run")
    assert result.returncode == 0, result.stderr
    serve, run = map(shlex.split, result.stdout.splitlines())
    original = invoke(REPO / "scripts/run_h3_t2va_production.sh", env, "--dry-run")
    assert serve == shlex.split(original.stdout.splitlines()[0])
    assert run[1] == "tools/run_h3_t2va_full_production.py"
    assert run[run.index("--request-workers") + 1] == "1"
    assert run[run.index("--gpu-ids") + 1] == "0,1,2,3,4,5,6,7"
    assert run[run.index("--shards") + 1] == "59,12,87"
    assert "--audio-production-root" not in run
    assert "--audio-shadow-run-id" not in run
    assert not any((root / name).exists() for name in ("sourced", "out", "events"))


@pytest.mark.parametrize(
    "gpu_ids", ["0,0", "-1", "0,,1", "x", "0,1,2,3,4,5,6,7,8", "01,1", ""]
)
def test_invalid_gpu_ids_rejected_without_start(sandbox, gpu_ids):
    script, env, root = sandbox
    result = invoke(script, {**env, "GPU_IDS": gpu_ids}, "--dry-run")
    assert result.returncode != 0
    assert "GPU_IDS" in result.stderr
    assert not (root / "events").exists()


def test_range_and_optional_arguments(sandbox):
    script, env, _ = sandbox
    env.pop("SHARDS")
    env.update(
        SHARD_START="5",
        SHARD_END="19",
        SHARD_SEED="42",
        RANK="1",
        WORLD_SIZE="2",
        GPU_IDS="7",
        FFMPEG="/custom/ffmpeg",
        ALLOW_UNVERIFIED="1",
        REQUEST_WORKERS="3",
    )
    result = invoke(script, env, "--dry-run")
    assert result.returncode == 0, result.stderr
    run = shlex.split(result.stdout.splitlines()[1])
    for flag, value in {
        "shard-start": "5",
        "shard-end": "19",
        "shard-seed": "42",
        "rank": "1",
        "world-size": "2",
        "gpu-ids": "7",
        "ffmpeg": "/custom/ffmpeg",
        "request-workers": "3",
    }.items():
        assert run[run.index("--" + flag) + 1] == value
    assert "--shards" not in run
    assert "--allow-unverified" in run


@pytest.mark.parametrize("status", [0, 7])
def test_one_server_for_all_shards_and_proxy_readiness(sandbox, status):
    script, env, root = sandbox
    result = invoke(script, {**env, "RUN_STATUS": str(status), "GPU_IDS": "3,7"})
    assert result.returncode == status, result.stderr
    log = events(root)
    assert log.count("mimo-start") == log.count("mimo-stop") == 1
    assert log.count("runner-start") == 1
    assert log.index("runner-exit") < log.index("mimo-stop")
    for name in ("mimo", "runner"):
        session = json.loads((root / f"{name}.session.json").read_text())
        assert session["pid"] == session["pgid"] == session["sid"]
        assert session["sid"] != os.getsid(0)
        assert session["stdin_is_devnull"]
    calls = [json.loads(line[5:]) for line in log if line.startswith("curl:")]
    assert sum(args[-1].endswith("/model_info") for args in calls) == 2
    assert sum(args[-1].endswith("/v1/models") for args in calls) == 2
    for args in calls:
        assert args[args.index("--noproxy") + 1] == "*"
    record = json.loads((root / "invocation.json").read_text())
    for flag, value in {
        "gpu-ids": "3,7",
        "shards": "59,12,87",
        "request-workers": "1",
    }.items():
        assert record["args"][record["args"].index("--" + flag) + 1] == value
    child_env = record["env"]
    assert {
        "existing.internal",
        "lower.internal",
        "localhost",
        "127.0.0.1",
        "::1",
    } <= set(child_env["NO_PROXY"].split(","))
    assert child_env["NO_PROXY"] == child_env["no_proxy"]
    assert child_env["http_proxy"] == "http://external.invalid:3128"
    assert child_env["HTTPS_PROXY"] == "http://secure.invalid:3128"
    assert "CUDA_VISIBLE_DEVICES" not in child_env
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / "mimo.pid").read_text()), 0)


@pytest.mark.parametrize("mode", ["timeout", "dead"])
def test_readiness_failure_never_launches_supervisor(sandbox, mode):
    script, env, root = sandbox
    result = invoke(script, {**env, "READY_MODE": mode})
    assert result.returncode != 0
    assert "runner-start" not in events(root)
    assert events(root).count("mimo-stop") == 1


def test_busy_port_does_not_start_or_stop_foreign_server(sandbox):
    script, env, root = sandbox
    result = invoke(script, {**env, "PORT_BUSY": "1"})
    assert result.returncode != 0
    assert not (root / "events").exists()


@pytest.mark.parametrize(
    "stop_signal, exit_status", [(signal.SIGTERM, 143), (signal.SIGINT, 130)]
)
def test_term_waits_for_supervisor_workers_before_mimo(
    sandbox, stop_signal, exit_status
):
    script, env, root = sandbox
    proc = subprocess.Popen(
        ["bash", str(script)],
        env={**env, "RUN_MODE": "wait"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 8
        while not (root / "runner-ready").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.01)
        assert (root / "runner-ready").exists()
        proc.send_signal(stop_signal)
        _, err = proc.communicate(timeout=8)
        assert proc.returncode == exit_status, err
        log = events(root)
        assert (
            log.index("runner-term")
            < log.index("workers-stopped")
            < log.index("mimo-stop")
        )
        for name in ("runner", "worker", "mimo"):
            with pytest.raises(ProcessLookupError):
                os.kill(int((root / f"{name}.pid").read_text()), 0)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


@pytest.mark.parametrize("stage", ["runner", "mimo"])
def test_forced_cleanup_kills_only_owned_process_groups(sandbox, stage):
    script, env, root = sandbox
    env = {**env, "FORCE_STAGE": stage, "CLEANUP_GRACE_SECONDS": "1"}
    if stage == "mimo":
        env["RUN_MODE"] = "wait"
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    proc = subprocess.Popen(
        ["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        flags = (
            ["worker-ready"]
            if stage == "runner"
            else ["runner-ready", "mimo-worker-ready"]
        )
        deadline = time.monotonic() + 8
        while (
            not all((root / name).exists() for name in flags)
            and time.monotonic() < deadline
        ):
            assert proc.poll() is None
            time.sleep(0.01)
        assert all((root / name).exists() for name in flags)
        proc.terminate()
        _, stderr = proc.communicate(timeout=8)
        assert proc.returncode == 143, stderr
        assert foreign.poll() is None
        for name in ("runner", "worker", "mimo", "mimo-worker"):
            path = root / f"{name}.pid"
            if not path.exists():
                continue
            pid = int(path.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                # An orphan zombie is already dead; PID 1 owns its final reap.
                state = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "stat="],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
                if not state or state.startswith("Z"):
                    break
                time.sleep(0.02)
            else:
                pytest.fail(f"owned {name} process survived forced cleanup")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)
        foreign.terminate()
        foreign.wait(timeout=5)
