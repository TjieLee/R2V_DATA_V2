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
DEPS = "/mnt/workspace/litengjie/data/audio_deps"
DEFAULT_DEPENDENCIES = {
    "AUK_PYTHON": f"{DEPS}/auk/auk-venv/bin/python",
    "AUK_CODE_ROOT": f"{DEPS}/auk/AuK-src",
    "AUK_CHECKPOINT": f"{DEPS}/auk/AuK/auk_base.safetensors",
    "AUK_QWEN_PATH": f"{DEPS}/auk/Qwen2.5-Omni-3B",
    "DIARIZEN_PYTHON": f"{DEPS}/diarizen-venv/bin/python",
    "DIARIZEN_CODE_ROOT": f"{DEPS}/DiariZen",
    "DIARIZEN_MODEL_PATH": f"{DEPS}/diarizen-model-cache",
    "DIARIZEN_MODEL_IDENTIFIER": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
    "DIARIZEN_DEVICE": "cuda:0",
    "DIARIZEN_TIMEOUT_SECONDS": "900",
    "QWEN3_ASR_ENV": f"{DEPS}/qwen3-asr-venv",
    "QWEN3_ASR_MODEL_PATH": "/mnt/workspace/public/pretrained/Qwen/Qwen3-ASR-1.7B",
    "QWEN3_ASR_DEVICE": "cuda:0",
    "QWEN3_ASR_DTYPE": "bfloat16",
    "QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE": "1",
}
REQUIRED_PATHS = {
    "AUK_PYTHON": "executable",
    "AUK_CODE_ROOT": "directory",
    "AUK_CHECKPOINT": "file",
    "AUK_QWEN_PATH": "directory",
    "DIARIZEN_PYTHON": "executable",
    "DIARIZEN_CODE_ROOT": "directory",
    "DIARIZEN_MODEL_PATH": "directory",
    "QWEN3_ASR_ENV": "venv",
    "QWEN3_ASR_MODEL_PATH": "directory",
}


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
        if k not in DEFAULT_DEPENDENCIES
        and k != "AUK_ROOT"
        and k
        not in {
            "GPU_IDS",
            "SHARDS",
            "SHARD_END",
            "REQUEST_WORKERS",
            "CANONICAL_WORKERS",
            "CUDA_VISIBLE_DEVICES",
        }
    }
    for name, kind in REQUIRED_PATHS.items():
        path = tmp_path / "dependencies" / name
        if kind in {"directory", "venv"}:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic local dependency")
            if kind == "executable":
                path.chmod(0o755)
        if kind == "venv":
            (path / "bin").mkdir()
            (path / "bin/python").symlink_to(sys.executable)
        env[name] = str(path)
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


@pytest.mark.parametrize("auk_root", [None, "/custom/auk"])
def test_dependency_defaults_without_server_files(sandbox, auk_root):
    script, env, root = sandbox
    env = {k: v for k, v in env.items() if k not in DEFAULT_DEPENDENCIES}
    if auk_root is not None:
        env["AUK_ROOT"] = auk_root
    # Execute the real setup, stopping before preflight; no /mnt paths are created.
    probe = script.with_name("probe.sh")
    probe.write_text(
        script.read_text().split("command -v setsid")[0]
        + f"\n{shlex.quote(sys.executable)} -c 'import os,json; print(json.dumps(dict(os.environ)))'\n"
    )
    result = invoke(probe, env)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    expected = {
        k: v.replace(f"{DEPS}/auk", auk_root)
        if auk_root and k.startswith("AUK_")
        else v
        for k, v in DEFAULT_DEPENDENCIES.items()
    }
    assert {k: actual.get(k) for k in expected} == expected
    assert not (root / "events").exists()


def test_dependency_overrides_reach_supervisor(sandbox):
    script, env, root = sandbox
    env.update(
        DIARIZEN_DEVICE="cuda:5",
        DIARIZEN_TIMEOUT_SECONDS="123",
        DIARIZEN_MODEL_IDENTIFIER="custom/diar",
        QWEN3_ASR_DEVICE="cuda:6",
        QWEN3_ASR_DTYPE="float32",
        QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE="2",
    )
    result = invoke(script, env)
    assert result.returncode == 0, result.stderr
    actual = json.loads((root / "invocation.json").read_text())["env"]
    for name in DEFAULT_DEPENDENCIES:
        assert actual[name] == env[name]


@pytest.mark.parametrize("name", REQUIRED_PATHS)
def test_missing_dependency_fails_before_mimo_but_not_dry_run(sandbox, name):
    script, env, root = sandbox
    env[name] = str(root / "missing" / name)
    assert invoke(script, env, "--dry-run").returncode == 0
    result = invoke(script, env)
    assert result.returncode == 2
    label = "Qwen3-ASR Python" if name == "QWEN3_ASR_ENV" else name
    assert f"Missing {label}:" in result.stderr
    assert env[name] in result.stderr
    assert not (root / "events").exists()


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
    standalone = shlex.split(original.stdout.splitlines()[0])
    assert serve == standalone
    assert serve[serve.index("--mem-fraction-static") + 1] == "0.65"
    assert run[1] == str(script.parent.parent / "tools/run_h3_t2va_full_production.py")
    assert run[run.index("--request-workers") + 1] == "1"
    assert run[run.index("--canonical-workers") + 1] == "16"
    assert run[run.index("--gpu-ids") + 1] == "0,1,2,3,4,5,6,7"
    assert run[run.index("--mimo-mem-fraction-static") + 1] == "0.65"
    assert run[run.index("--mimo-startup-polls") + 1] == "3"
    assert run[run.index("--mimo-poll-interval") + 1] == "0.01"
    assert run[run.index("--mimo-cleanup-grace-seconds") + 1] == "2"
    assert run[run.index("--shards") + 1] == "59,12,87"
    assert "--audio-production-root" not in run
    assert "--audio-shadow-run-id" not in run
    assert not any((root / name).exists() for name in ("sourced", "out", "events"))


def test_v26_single_call_dry_run_preserves_official_serving(sandbox):
    script, env, _ = sandbox
    result = invoke(
        script,
        {**env, "MIMO_MODEL": "mimo-v2.6-flash-rl", "MIMO_CALL_MODE": "single"},
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    serve, run = map(shlex.split, result.stdout.splitlines())
    assert serve[serve.index("--speculative-algorithm") + 1] == "EAGLE"
    assert run[run.index("--mimo-call-mode") + 1] == "single"


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
        CANONICAL_WORKERS="5",
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
        "canonical-workers": "5",
    }.items():
        assert run[run.index("--" + flag) + 1] == value
    assert "--shards" not in run
    assert "--allow-unverified" in run


@pytest.mark.parametrize("status", [0, 7])
def test_shell_launches_only_supervisor_and_passes_mimo_config(sandbox, status):
    script, env, root = sandbox
    result = invoke(script, {**env, "RUN_STATUS": str(status), "GPU_IDS": "3,7"})
    assert result.returncode == status, result.stderr
    log = events(root)
    assert "mimo-start" not in log
    assert "mimo-stop" not in log
    assert log.count("runner-start") == 1
    assert log.count("runner-exit") == 1
    session = json.loads((root / "runner.session.json").read_text())
    assert session["pid"] == session["pgid"] == session["sid"]
    assert session["sid"] != os.getsid(0)
    assert session["stdin_is_devnull"]
    record = json.loads((root / "invocation.json").read_text())
    for flag, value in {
        "gpu-ids": "3,7",
        "shards": "59,12,87",
        "request-workers": "1",
        "mimo-mem-fraction-static": "0.65",
        "mimo-startup-polls": "3",
        "mimo-poll-interval": "0.01",
        "mimo-cleanup-grace-seconds": "2",
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


@pytest.mark.parametrize(
    "stop_signal, exit_status", [(signal.SIGTERM, 143), (signal.SIGINT, 130)]
)
def test_term_waits_for_supervisor_workers(sandbox, stop_signal, exit_status):
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
        assert log.index("runner-term") < log.index("workers-stopped")
        for name in ("runner", "worker"):
            with pytest.raises(ProcessLookupError):
                os.kill(int((root / f"{name}.pid").read_text()), 0)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=5)


def test_forced_cleanup_kills_only_owned_supervisor_group(sandbox):
    script, env, root = sandbox
    env = {**env, "FORCE_STAGE": "runner", "CLEANUP_GRACE_SECONDS": "1"}
    foreign = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    proc = subprocess.Popen(
        ["bash", str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 8
        while not (root / "worker-ready").exists() and time.monotonic() < deadline:
            assert proc.poll() is None
            time.sleep(0.01)
        assert (root / "worker-ready").exists()
        proc.terminate()
        _, stderr = proc.communicate(timeout=8)
        assert proc.returncode == 143, stderr
        assert foreign.poll() is None
        for name in ("runner", "worker"):
            path = root / f"{name}.pid"
            if not path.exists():
                continue
            pid = int(path.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
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
