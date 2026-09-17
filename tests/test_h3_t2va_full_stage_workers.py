"""CPU-only integration tests: every backend runs in a real subprocess."""

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

MODULE = "r2v_data_v2.h3.t2va_full_workers"
FAKE = """
import json, os, subprocess, sys, time
from pathlib import Path

class Backend:
    def __init__(self, configuration):
        self.config = configuration
        self.root = Path(configuration["events"])
        self.pid = os.getpid()
        self.children = []
        self.event("load")
    def event(self, kind, **extra):
        with (self.root / (str(self.pid) + ".jsonl")).open("a") as f:
            f.write(json.dumps(dict(kind=kind, pid=self.pid, **extra)) + "\\n")
    def __enter__(self):
        self.event("enter")
        return self
    def __exit__(self, *args):
        self.event("close")
        for child in self.children:
            child.terminate()
            child.wait(timeout=5)
    def process(self, job, output_dir):
        self.event("process", job_id=job["job_id"])
        if job.get("nested"):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                     start_new_session=True)
            self.children.append(child)
            (self.root / "nested.pid").write_text(str(child.pid))
        if job.get("crash") and not (self.root / "allow").exists():
            os._exit(job.get("exit_code", 23))
        time.sleep(job.get("sleep", 0))
        if job.get("fail") and not (self.root / "allow").exists():
            (Path(output_dir) / "failed.partial").write_text("unfinished")
            raise ValueError("sample failure")
        if job.get("nonjson"):
            return {"bad": object()}
        if job.get("semantic") and not (self.root / "allow").exists():
            from r2v_data_v2.h3.t2va_full_workers import SampleJobFailure
            raise SampleJobFailure("SAM route failed", result={"calls": ["route-a"], "model_call_count": 1})
        path = Path(output_dir) / "artifact.txt"
        path.write_text(job["job_id"])
        return dict(path=str(path), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    device=self.config["device"], marker=os.environ.get("WORKER_MARKER"),
                    pid=self.pid, inventory_fingerprint=job.get("inventory_fingerprint"))

def factory(configuration):
    return Backend(configuration)
"""


def api():
    return importlib.import_module(MODULE)


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "fake_stage_backend.py").write_text(FAKE)
    events = tmp_path / "events"
    events.mkdir()
    return {
        "stage_root": tmp_path / "stage",
        "gpu_ids": ["7", "2", "9"],
        "factory": "fake_stage_backend:factory",
        "configuration": {"events": str(events)},
        "python_path": sys.executable,
        "environment": {
            "PYTHONPATH": os.pathsep.join(
                [str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]
            ),
            "WORKER_MARKER": "child",
        },
    }


def events(setup):
    return [
        json.loads(line)
        for path in Path(setup["configuration"]["events"]).glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def receipt(setup, job_id):
    key = hashlib.sha256(job_id.encode()).hexdigest()
    return Path(setup["stage_root"]) / "jobs" / key / "receipt.json"


def test_partition_10000_load_once_gpu_mapping_and_order(setup):
    setup["gpu_ids"] = [str(i) for i in range(8)]
    before = dict(os.environ)
    jobs = [{"job_id": f"job-{i}"} for i in range(10000)]
    result = api().execute_stage(jobs=jobs, **setup)
    assert list(result) == [j["job_id"] for j in jobs]
    assert os.environ == before
    assert "device" not in setup["configuration"]
    records = events(setup)
    for rank, gpu in enumerate(setup["gpu_ids"]):
        partition = jobs[rank::8]
        pid = result[partition[0]["job_id"]]["result"]["pid"]
        assert [
            e["job_id"] for e in records if e["kind"] == "process" and e["pid"] == pid
        ] == [j["job_id"] for j in partition]
        for job in partition:
            row = result[job["job_id"]]
            assert row["status"] == "ready" and row["failure_reason"] is None
            assert row["result"]["gpu"] == gpu
            assert row["result"]["device"] == "cuda:0"
            assert row["result"]["marker"] == "child"
            assert Path(row["result"]["path"]).is_file()
    for kind in ("load", "enter", "close"):
        assert sum(e["kind"] == kind for e in records) == 8
    requests = list(Path(setup["stage_root"]).glob("workers/*/worker-*/request.json"))
    assert len(requests) == 8
    assert all(p.with_name("worker.log").is_file() for p in requests)


def test_failure_continues_retries_once_and_ready_skips(setup):
    setup["gpu_ids"] = ["4"]
    setup["log_root"] = Path(setup["stage_root"]).parent / "logs"
    jobs = [{"job_id": "../bad", "fail": True}, {"job_id": "good"}]
    first = api().execute_stage(jobs=jobs, **setup)
    assert (setup["log_root"] / "stage-gpu4.log").is_file()
    assert first["../bad"]["status"] == "failed"
    assert "sample failure" in first["../bad"]["failure_reason"]
    assert first["../bad"]["result"] is None
    assert first["good"]["status"] == "ready"
    second = api().execute_stage(jobs=jobs, **setup)
    assert second["good"] == first["good"]
    assert (
        sum(e["kind"] == "process" and e["job_id"] == "../bad" for e in events(setup))
        == 2
    )
    Path(setup["configuration"]["events"], "allow").touch()
    third = api().execute_stage(jobs=jobs, **setup)
    assert third["../bad"]["status"] == "ready"
    assert (
        not Path(third["../bad"]["result"]["path"]).with_name("failed.partial").exists()
    )
    count = len(events(setup))
    assert api().execute_stage(jobs=jobs, **setup) == third
    assert len(events(setup)) == count
    assert json.loads((Path(setup["stage_root"]) / "invocation.json").read_text()) == {
        "job_count": 2,
        "scheduled_job_count": 0,
        "reused_ready_count": 2,
        "worker_count": 0,
    }
    assert len(list(Path(setup["stage_root"]).glob("jobs/*"))) == 2


def test_full_input_identity_stable_paths_and_owned_cleanup(setup):
    jobs = [{"job_id": "x", "nested": {"b": 2, "a": 1}}]
    # The fake backend reserves the nested key for its process-tree test.
    jobs[0]["payload"] = jobs[0].pop("nested")
    first = api().execute_stage(jobs=jobs, **setup)
    saved = receipt(setup, "x")
    identity = json.loads(saved.read_text())["input_fingerprint"]
    foreign = saved.parent / "foreign.partial"
    foreign.write_text("leave alone")
    reordered = [{"payload": {"a": 1, "b": 2}, "job_id": "x"}]
    assert api().execute_stage(jobs=reordered, **setup) == first
    reordered[0]["payload"]["b"] = 3
    second = api().execute_stage(jobs=reordered, **setup)
    assert second["x"]["result"]["pid"] != first["x"]["result"]["pid"]
    assert second["x"]["result"]["path"] == first["x"]["result"]["path"]
    assert json.loads(saved.read_text())["input_fingerprint"] != identity
    assert foreign.read_text() == "leave alone"
    saved.write_text("{broken")
    assert api().execute_stage(jobs=reordered, **setup)["x"]["status"] == "ready"


def test_inventory_publication_identity_excluded_but_configuration_included(setup):
    jobs = [
        {
            "job_id": "sam",
            "inventory_fingerprint": "full-inventory-v1",
            "source": {"sha256": "source-v1"},
        }
    ]
    first = api().execute_stage(jobs=jobs, **setup)
    assert first["sam"]["result"]["inventory_fingerprint"] == "full-inventory-v1"
    fingerprint = json.loads(receipt(setup, "sam").read_text())["input_fingerprint"]
    jobs[0]["inventory_fingerprint"] = "expanded-full-inventory-v2"
    assert api().execute_stage(jobs=jobs, **setup) == first
    assert (
        json.loads(receipt(setup, "sam").read_text())["input_fingerprint"]
        == fingerprint
    )
    setup["configuration"]["model_revision"] = "new"
    second = api().execute_stage(jobs=jobs, **setup)
    assert second["sam"]["result"]["pid"] != first["sam"]["result"]["pid"]
    assert (
        second["sam"]["result"]["inventory_fingerprint"] == "expanded-full-inventory-v2"
    )


def test_fixed_media_paths_hash_receipts_and_corrupt_output_retry(setup):
    jobs = [{"job_id": "sam"}]
    first = api().execute_stage(jobs=jobs, **setup)
    path = Path(first["sam"]["result"]["path"])
    assert path.parent == receipt(setup, "sam").parent / "media"
    state = json.loads(receipt(setup, "sam").read_text())
    assert state["output_files"] == {
        "artifact.txt": {"size": 3, "sha256": hashlib.sha256(b"sam").hexdigest()}
    }
    original_mtime = path.stat().st_mtime_ns
    assert api().execute_stage(jobs=jobs, **setup) == first
    assert path.stat().st_mtime_ns == original_mtime
    path.write_text("corrupt")
    second = api().execute_stage(jobs=jobs, **setup)
    assert second["sam"]["result"]["pid"] != first["sam"]["result"]["pid"]
    assert second["sam"]["result"]["path"] == str(path)
    assert path.read_text() == "sam"
    path.unlink()
    assert (
        api().execute_stage(jobs=jobs, **setup)["sam"]["result"]["pid"]
        != second["sam"]["result"]["pid"]
    )


def test_semantic_failure_preserves_diagnostics_and_retries(setup):
    setup["gpu_ids"] = ["7"]
    jobs = [{"job_id": "sam", "semantic": True}, {"job_id": "next"}]
    result = api().execute_stage(jobs=jobs, **setup)
    assert result["sam"]["status"] == "failed"
    assert result["sam"]["result"] == {"calls": ["route-a"], "model_call_count": 1}
    assert "SAM route failed" in result["sam"]["failure_reason"]
    assert result["next"]["status"] == "ready"
    Path(setup["configuration"]["events"], "allow").touch()
    retried = api().execute_stage(jobs=jobs, **setup)
    assert retried["sam"]["status"] == "ready"
    assert retried["next"] == result["next"]
    assert issubclass(api().SampleJobFailure, ValueError)
    assert api().SampleJobFailure("failure").result is None


@pytest.mark.parametrize("exit_code", [0, 23])
def test_crash_waits_other_workers_leaves_unfinished_pending_then_resumes(
    setup, exit_code
):
    setup["gpu_ids"] = ["5", "8"]
    jobs = [
        {"job_id": "crash", "crash": True, "exit_code": exit_code},
        {"job_id": "slow", "sleep": 0.4},
        {"job_id": "pending"},
    ]
    with pytest.raises(api().WorkerStageError):
        api().execute_stage(jobs=jobs, **setup)
    slow = json.loads(receipt(setup, "slow").read_text())
    assert slow["status"] == "ready"
    for job_id in ("crash", "pending"):
        path = receipt(setup, job_id)
        assert not path.exists() or json.loads(path.read_text())["status"] == "pending"
    Path(setup["configuration"]["events"], "allow").touch()
    result = api().execute_stage(jobs=jobs, **setup)
    assert all(row["status"] == "ready" for row in result.values())
    assert result["slow"]["result"]["pid"] == slow["result"]["pid"]


def test_empty_validation_and_non_json_sample_failure(setup):
    assert api().execute_stage(jobs=[], **setup) == {}
    assert not events(setup)
    with pytest.raises(ValueError):
        api().execute_stage(jobs=[{"job_id": "x"}] * 2, **setup)
    with pytest.raises(ValueError):
        api().execute_stage(jobs=[{"job_id": "x"}], **{**setup, "gpu_ids": []})
    result = api().execute_stage(
        jobs=[{"job_id": "x", "nonjson": True}, {"job_id": "y"}], **setup
    )
    assert result["x"]["status"] == "failed"
    assert result["y"]["status"] == "ready"
    assert sum(e["kind"] == "load" for e in events(setup)) == 2


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_interrupt_reaps_workers_and_nested_new_session(setup, tmp_path, sig):
    api()  # Fail immediately in the red phase, before creating a supervisor.
    request = {
        **setup,
        "stage_root": str(setup["stage_root"]),
        "jobs": [{"job_id": "blocked", "nested": True, "sleep": 120}],
    }
    script = f"import json; from {MODULE} import execute_stage; execute_stage(**json.loads({json.dumps(request)!r}))"
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
        worker = next(e["pid"] for e in events(setup) if e["kind"] == "load")
        parent.send_signal(sig)
        assert parent.wait(timeout=10) != 0
        for pid in (worker, nested):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        assert not receipt(setup, "blocked").exists()
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=10)
