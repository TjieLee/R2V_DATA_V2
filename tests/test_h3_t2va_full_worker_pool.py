"""Persistent executor contracts exercised with real CPU-only subprocesses."""

import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

MODULE = "r2v_data_v2.h3.t2va_full_worker_pool"
FAKE = """
import json, os, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path

class Backend:
    def __init__(self, configuration):
        self.config = configuration
        self.root = Path(configuration["events"])
        self.pid = os.getpid()
        self.requests = 0
        self.children = []
        self.event("load")
    def event(self, kind, **extra):
        with (self.root / (str(self.pid) + ".jsonl")).open("a") as f:
            f.write(json.dumps(dict(kind=kind, pid=self.pid, **extra)) + "\\n")
    def __enter__(self):
        self.event("enter")
        if self.config.get("startup_fail") == os.environ["CUDA_VISIBLE_DEVICES"]:
            raise RuntimeError("startup failure")
        time.sleep(self.config.get("startup_sleep", 0))
        return self
    def __exit__(self, *args):
        self.event("close")
        for child in self.children:
            child.wait(timeout=5)
    def bind_request(self, configuration):
        self.config = configuration
        self.requests += 1
        self.event("bind", inventory=configuration.get("inventory"))
        if configuration.get("bind_fail"):
            raise RuntimeError("bind failure")
    @contextmanager
    def batch_jobs(self, jobs):
        self.event("batch_enter", jobs=[j["job_id"] for j in jobs])
        try:
            yield
        finally:
            self.event("batch_close")
    def process(self, job, output):
        print("model stdout is not a protocol: {broken json", flush=True)
        self.event("process", job_id=job["job_id"])
        if job.get("nested"):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                     start_new_session=True)
            self.children.append(child)
            (self.root / "nested.pid").write_text(str(child.pid))
        time.sleep(job.get("sleep", 0))
        if job.get("crash") and not (self.root / "allow").exists():
            os._exit(job.get("exit_code", 23))
        if job.get("fail") and not (self.root / "allow").exists():
            from r2v_data_v2.h3.t2va_full_workers import SampleJobFailure
            raise SampleJobFailure("sample failure", {"diagnostic": True})
        path = output / "artifact.txt"
        path.write_text(job["job_id"])
        return dict(pid=self.pid, gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                    device=self.config["device"], marker=os.environ.get("POOL_MARKER"),
                    inventory=self.config.get("inventory"), requests=self.requests, path=str(path))

def factory(configuration):
    return Backend(configuration)

def plain_factory(configuration):
    backend = Backend(configuration)
    backend.bind_request = None
    backend.batch_jobs = None
    return backend
"""


def api():
    return importlib.import_module(MODULE)


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "fake_pool_backend.py").write_text(FAKE)
    (tmp_path / "events").mkdir()
    return {
        "stage_root": tmp_path / "stage",
        "gpu_ids": ["5", "8"],
        "factory": "fake_pool_backend:factory",
        "configuration": {"events": str(tmp_path / "events"), "model": "v1"},
        "python_path": sys.executable,
        "environment": {
            "PYTHONPATH": os.pathsep.join(
                [str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]
            ),
            "POOL_MARKER": "isolated",
        },
    }


def start(manager, setup, **overrides):
    args = {
        k: setup[k] for k in ("factory", "configuration", "environment", "python_path")
    }
    manager.start("sam", **{**args, **overrides})


def events(setup):
    return [
        json.loads(line)
        for path in Path(setup["configuration"]["events"]).glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def receipt(setup, job_id):
    return (
        Path(setup["stage_root"])
        / "jobs"
        / hashlib.sha256(job_id.encode()).hexdigest()
        / "receipt.json"
    )


def test_three_requests_same_pids_load_close_once_env_and_batch(
    setup, tmp_path, capsys
):
    before = dict(os.environ)
    pids = []
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup, request_keys=("inventory",))
        assert "pool=sam workers=2 startup_seconds=" in capsys.readouterr().out
        assert sum(e["kind"] == "enter" for e in events(setup)) == 2
        for shard in range(3):
            config = {**setup["configuration"], "inventory": str(shard)}
            result = manager.execute_stage(
                **{
                    **setup,
                    "stage_root": tmp_path / str(shard),
                    "configuration": config,
                },
                jobs=[{"job_id": str(i)} for i in range(4)],
            )
            assert list(result) == ["0", "1", "2", "3"]
            pids.append([result[str(i)]["result"]["pid"] for i in range(2)])
            for i, row in enumerate(result.values()):
                assert row["status"] == "ready"
                assert row["result"]["gpu"] == setup["gpu_ids"][i % 2]
                assert row["result"]["device"] == "cuda:0"
                assert row["result"]["marker"] == "isolated"
                assert row["result"]["inventory"] == str(shard)
                assert row["result"]["requests"] == shard + 1
        assert pids[0] == pids[1] == pids[2]
        assert not any(e["kind"] == "close" for e in events(setup))
    assert os.environ == before
    assert "device" not in setup["configuration"]
    for pid in pids[0]:
        kinds = [e["kind"] for e in events(setup) if e["pid"] == pid]
        assert kinds.count("load") == kinds.count("enter") == kinds.count("close") == 1
        assert (
            kinds.count("bind")
            == kinds.count("batch_enter")
            == kinds.count("batch_close")
            == 3
        )
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    assert all(
        "model stdout" in p.read_text() for p in (tmp_path / "pools").glob("*-gpu*.log")
    )


def test_eight_gpu_ready_barrier_and_one_job_per_worker(setup, tmp_path, capsys):
    setup["gpu_ids"] = ["5", "2", "7", "0", "6", "1", "4", "3"]
    before = dict(os.environ)
    root = tmp_path / "pools"
    jobs = [{"job_id": f"job-{rank}"} for rank in range(8)]
    with api().PersistentPoolManager(root, setup["gpu_ids"]) as manager:
        start(manager, setup)
        assert "pool=sam workers=8 startup_seconds=" in capsys.readouterr().out
        ready = list(root.glob("*/sam/worker-*/ready.json"))
        assert len(ready) == 8
        assert all(json.loads(path.read_text()) == {"ready": True} for path in ready)
        loaded = {e["pid"] for e in events(setup) if e["kind"] == "load"}
        entered = {e["pid"] for e in events(setup) if e["kind"] == "enter"}
        assert len(loaded) == 8 and entered == loaded
        assert not any(e["kind"] == "process" for e in events(setup))

        result = manager.execute_stage(**setup, jobs=jobs)
        assert list(result) == [job["job_id"] for job in jobs]
        assert {row["result"]["pid"] for row in result.values()} == loaded
        records = events(setup)
        for job, gpu in zip(jobs, setup["gpu_ids"], strict=True):
            row = result[job["job_id"]]
            assert row["status"] == "ready"
            assert row["result"]["gpu"] == gpu
            assert row["result"]["device"] == "cuda:0"
            assert [
                e["job_id"]
                for e in records
                if e["kind"] == "process" and e["pid"] == row["result"]["pid"]
            ] == [job["job_id"]]
    assert os.environ == before
    assert {e["pid"] for e in events(setup) if e["kind"] == "close"} == loaded


def test_resume_receipts_match_ephemeral_and_hash_validation(setup, tmp_path):
    from r2v_data_v2.h3 import t2va_full_workers as old

    jobs = [
        {"job_id": "good", "inventory_fingerprint": "first"},
        {"job_id": "bad", "fail": True},
    ]
    first = old.execute_stage(**setup, jobs=jobs)
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup, request_keys=("inventory",))
        result = manager.execute_stage(**setup, jobs=jobs)
        assert result["good"] == first["good"]
        assert result["bad"]["result"] == {"diagnostic": True}
        Path(setup["configuration"]["events"], "allow").touch()
        result = manager.execute_stage(**setup, jobs=jobs)
        assert all(row["status"] == "ready" for row in result.values())
        count = len(events(setup))
        jobs[0]["inventory_fingerprint"] = "publication-only"
        assert manager.execute_stage(**setup, jobs=jobs) == result
        assert len(events(setup)) == count
        Path(result["good"]["result"]["path"]).write_text("corrupt")
        repaired = manager.execute_stage(**setup, jobs=jobs)
        assert Path(repaired["good"]["result"]["path"]).read_text() == "good"
        config = {**setup["configuration"], "inventory": "new"}
        changed = manager.execute_stage(**{**setup, "configuration": config}, jobs=jobs)
        assert changed["good"]["result"]["inventory"] == "new"
    assert old.execute_stage(**{**setup, "configuration": config}, jobs=jobs) == changed


@pytest.mark.parametrize("exit_code", [0, 23])
def test_crash_barrier_unhealthy_no_restart_and_resume(setup, tmp_path, exit_code):
    jobs = [
        {"job_id": "crash", "crash": True, "exit_code": exit_code},
        {"job_id": "slow", "sleep": 0.4},
        {"job_id": "pending"},
    ]
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup)
        manager.execute_stage(
            **{**setup, "stage_root": tmp_path / "shard0"}, jobs=[{"job_id": "ok"}]
        )
        with pytest.raises(api().WorkerStageError):
            manager.execute_stage(**setup, jobs=jobs)
        saved = json.loads(receipt(setup, "slow").read_text())
        assert saved["status"] == "ready"
        assert not receipt(setup, "crash").exists()
        assert not receipt(setup, "pending").exists()
        with pytest.raises(api().WorkerStageError, match="unhealthy|closed"):
            manager.execute_stage(**setup, jobs=jobs)
        assert sum(e["kind"] == "load" for e in events(setup)) == 2
    Path(setup["configuration"]["events"], "allow").touch()
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup)
        result = manager.execute_stage(**setup, jobs=jobs)
        assert all(row["status"] == "ready" for row in result.values())
        assert result["slow"]["result"] == saved["result"]


@pytest.mark.parametrize("mode", ["failure", "timeout", "spawn"])
def test_startup_failure_cleans_previously_started_pools(setup, tmp_path, mode):
    with api().PersistentPoolManager(
        tmp_path / "pools", setup["gpu_ids"], startup_timeout=0.5
    ) as manager:
        start(manager, setup)
        config = {
            **setup["configuration"],
            **({"startup_fail": "8"} if mode == "failure" else {"startup_sleep": 120}),
        }
        with pytest.raises(api().WorkerStageError):
            manager.start(
                "auk",
                "fake_pool_backend:plain_factory",
                config,
                environment=setup["environment"],
                python_path="/nonexistent/python"
                if mode == "spawn"
                else sys.executable,
            )
        for event in events(setup):
            with pytest.raises(ProcessLookupError):
                os.kill(event["pid"], 0)


def test_model_config_and_runtime_mismatch_fail_closed(setup, tmp_path):
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup, request_keys=("inventory",))
        for change in (
            {"configuration": {**setup["configuration"], "model": "v2"}},
            {"configuration": {**setup["configuration"], "unknown": 1}},
            {"gpu_ids": ["8", "5"]},
            {"python_path": "/other/python"},
            {"environment": {"POOL_MARKER": "changed"}},
        ):
            with pytest.raises(ValueError, match="mismatch"):
                manager.execute_stage(**{**setup, **change}, jobs=[{"job_id": "x"}])
        assert not any(e["kind"] == "process" for e in events(setup))
        assert manager.execute_stage(**setup, jobs=[]) == {}
        with pytest.raises(ValueError, match="duplicate"):
            manager.execute_stage(**setup, jobs=[{"job_id": "x"}] * 2)


def test_optional_hooks_and_factory_selection(setup, tmp_path):
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup)
        manager.start(
            "auk",
            "fake_pool_backend:plain_factory",
            setup["configuration"],
            environment=setup["environment"],
        )
        result = manager.execute_stage(
            **{**setup, "factory": "fake_pool_backend:plain_factory"},
            jobs=[{"job_id": "x"}],
        )
        assert result["x"]["status"] == "ready"
        assert not any(e["kind"] == "bind" for e in events(setup))


def test_bind_failure_is_infrastructure_not_sample_failure(setup, tmp_path):
    with api().PersistentPoolManager(tmp_path / "pools", setup["gpu_ids"]) as manager:
        start(manager, setup, request_keys=("bind_fail",))
        with pytest.raises(api().WorkerStageError):
            manager.execute_stage(
                **{
                    **setup,
                    "configuration": {**setup["configuration"], "bind_fail": True},
                },
                jobs=[{"job_id": "x"}],
            )
        assert not receipt(setup, "x").exists()


@pytest.mark.parametrize("crash", [False, True])
def test_infrastructure_failure_cleans_detached_children(setup, tmp_path, crash):
    marker = Path(setup["configuration"]["events"]) / "nested.pid"
    nested = None
    try:
        with api().PersistentPoolManager(
            tmp_path / "pools", setup["gpu_ids"]
        ) as manager:
            start(manager, setup, request_keys=("bind_fail",))
            job = {"job_id": "nested", "nested": True, "sleep": 0.5, "crash": crash}
            if crash:
                with pytest.raises(api().WorkerStageError):
                    manager.execute_stage(**setup, jobs=[job])
            else:
                manager.execute_stage(**setup, jobs=[job])
                with pytest.raises(api().WorkerStageError):
                    manager.execute_stage(
                        **{
                            **setup,
                            "configuration": {
                                **setup["configuration"],
                                "bind_fail": True,
                            },
                        },
                        jobs=[{"job_id": "next"}],
                    )
            nested = int(marker.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(nested, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            with pytest.raises(ProcessLookupError):
                os.kill(nested, 0)
    finally:
        if nested is None and marker.exists():
            nested = int(marker.read_text())
        if nested is not None:
            try:
                os.kill(nested, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_cleanup_nested_subprocesses(setup, tmp_path, sig):
    api()
    request = {**setup, "stage_root": str(setup["stage_root"])}
    script = (
        f"import json; from {MODULE} import PersistentPoolManager\n"
        f"s=json.loads({json.dumps(request)!r})\n"
        f"with PersistentPoolManager({str(tmp_path / 'pools')!r}, s['gpu_ids']) as p:\n"
        " p.start('sam', s['factory'], s['configuration'], environment=s['environment'])\n"
        " p.execute_stage(**s, jobs=[{'job_id':'blocked','nested':True,'sleep':120}])\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        env={**os.environ, **setup["environment"]},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        marker = Path(setup["configuration"]["events"]) / "nested.pid"
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        assert marker.exists()
        nested = int(marker.read_text())
        parent.send_signal(sig)
        assert parent.wait(timeout=10) != 0
        for pid in {nested} | {e["pid"] for e in events(setup)}:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert not receipt(setup, "blocked").exists()
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=10)
