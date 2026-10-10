from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest


class Permission:
    def __init__(self):
        self.allowed = True

    def execution_allowed(self):
        return self.allowed


class ProcessFixtureFactory:
    def __init__(self, directory, *, wait=False):
        self.directory, self.wait = str(directory), wait

    def __call__(self, stage):
        factory = self

        class Backend:
            def __enter__(self):
                Path(factory.directory, "loaded").write_text(os.environ["CUDA_VISIBLE_DEVICES"])
                return self

            def __exit__(self, *args):
                Path(factory.directory, "closed").touch()

            def materialize_full_audio(self, **kwargs):
                import subprocess
                child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
                Path(factory.directory, "nested.pid").write_text(str(child.pid))
                while factory.wait:
                    time.sleep(.02)
                raise ValueError("fixture business failure")
        return Backend()


def test_native_stage_process_isolated_gpu_and_reused_model(tmp_path):
    from dataclasses import asdict

    from r2v_data_v2.h3.ra2va_group_http_native_worker import NativeStageExecution
    from tests.test_h3_ra2va_group_pipeline import task_at
    task = task_at(tmp_path)
    permission = Permission()
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    executor = NativeStageExecution({"stage": "canonical"}, backend_factory=ProcessFixtureFactory(tmp_path),
        ffmpeg="ffmpeg", execution="native", heartbeat=permission, stop_event=threading.Event(), gpu_id="6")
    with executor:
        pid = executor.process.pid
        for index in range(2):
            status, payload = executor.execute({"task": asdict(task), "upstream_results": {},
                "artifact_root": str(tmp_path / f"claim-{index}")})
            assert status == "failed" and "fixture business failure" in payload["failure_reason"]
            assert executor.process.pid == pid
        assert (tmp_path / "loaded").read_text() == "6"
    assert executor.process.exitcode == 0 and (tmp_path / "closed").exists()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "nested.pid").read_text()), 0)


def test_watchdog_stops_native_inference_and_detached_child(tmp_path):
    from dataclasses import asdict

    from r2v_data_v2.h3.ra2va_group_http_native_worker import NativeStageExecution
    from r2v_data_v2.h3.ra2va_group_http_worker import HttpConflict
    from tests.test_h3_ra2va_group_pipeline import task_at
    permission, stop = Permission(), threading.Event()
    executor = NativeStageExecution({"stage": "canonical"}, backend_factory=ProcessFixtureFactory(tmp_path, wait=True),
        ffmpeg="ffmpeg", execution="native", heartbeat=permission, stop_event=stop, gpu_id="3", cleanup_seconds=.2)
    errors = []

    def run():
        try:
            with executor:
                executor.execute({"task": asdict(task_at(tmp_path)), "upstream_results": {},
                                  "artifact_root": str(tmp_path / "claim")})
        except HttpConflict as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10
    while not (tmp_path / "nested.pid").exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert (tmp_path / "nested.pid").exists()
    time.sleep(.3)  # Supervisor has observed a detached descendant.
    permission.allowed = False
    thread.join(5)
    assert not thread.is_alive() and errors and not executor.process.is_alive()
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "nested.pid").read_text()), 0)
    assert not stop.is_set()


def test_native_watchdog_rejoins_after_cleanup_without_stopping_node(tmp_path, monkeypatch):
    from dataclasses import asdict

    from r2v_data_v2.h3 import ra2va_group_http_worker
    from r2v_data_v2.h3.ra2va_group_http_native_worker import run_http_native_worker
    from tests.test_h3_ra2va_group_pipeline import task_at

    permission, stop = Permission(), threading.Event()
    sessions, errors, reports = [], [], []
    task = {"task": asdict(task_at(tmp_path)), "upstream_results": {},
            "artifact_root": str(tmp_path / "claim"), "session_id": "expired-session",
            "task_id": "canonical/clip", "claim_token": "expired-token"}

    class Heartbeat:
        def __init__(self, client):
            pass

        def start(self, session):
            pass

        def stop(self):
            pass

        def execution_allowed(self):
            return permission.execution_allowed()

    class Client:
        def call(self, method, path, body):
            if path == "/v1/workers/connect":
                sessions.append(body)
                if len(sessions) == 1:
                    return {"finished": False, "session_id": "expired-session", "generation": 1,
                            "stage": "canonical", "claim": task}
                assert body["previous_session_id"] == "expired-session"
                assert body["resources_closed"] is True and not stop.is_set()
                assert (tmp_path / "closed").exists()
                with pytest.raises(ProcessLookupError):
                    os.kill(int((tmp_path / "nested.pid").read_text()), 0)
                return {"finished": True}
            assert path == "/v1/workers/heartbeat"
            assert body["execution_stopped"] is True and body["resources_closed"] is True
            return {}

    monkeypatch.setattr(ra2va_group_http_worker, "WorkerHeartbeat", Heartbeat)

    def run():
        try:
            reports.append(run_http_native_worker(Client(), node_id="node-a", worker_instance="gpu-0",
                stop_event=stop, backend_factory=ProcessFixtureFactory(tmp_path, wait=True), gpu_id="0"))
        except (AssertionError, OSError, RuntimeError) as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "nested.pid").exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert (tmp_path / "nested.pid").exists()
        time.sleep(.3)
        permission.allowed = False
        thread.join(8)
        assert not thread.is_alive() and errors == []
        assert len(sessions) == 2 and not stop.is_set()
        assert reports == [{"executed": 0, "accepted": 0, "model_call_count": 0, "reused_checkpoints": 0}]
    finally:
        permission.allowed = False
        stop.set()
        thread.join(8)


def test_tp4_services_start_once_per_node_generation_and_cleanup(tmp_path):
    from r2v_data_v2.h3.ra2va_group_native_resources import NodeMimoServices
    created, closed = [], []

    class Server:
        def __init__(self, group, port):
            self.group, self.port = group, port

        def start(self, allowed):
            assert allowed()
            created.append((self.group, self.port))

        def close(self):
            closed.append(self.port)
    services = NodeMimoServices({"gpu_groups": ["0,1,2,3", "4,5,6,7"], "ports": [8094, 8095]},
                                log_root=tmp_path, server_factory=Server)
    services.join(7, lambda: True)
    services.join(7, lambda: True)
    assert created == [("0,1,2,3", 8094), ("4,5,6,7", 8095)]
    services.leave(7)
    assert not closed
    services.leave(7)
    assert closed == [8095, 8094]
    services.join(8, lambda: True)
    services.leave(8)
    assert len(created) == len(closed) == 4


def test_tp4_command_and_no_overlapping_gpu_replicas(tmp_path):
    from r2v_data_v2.h3.ra2va_group_native_resources import (
        NodeMimoServices,
        SGLangProcess,
    )
    server = SGLangProcess({"sglang": "/sglang", "checkpoint": "/model"}, "2,3,6,7", 8094, tmp_path)
    assert server.command[server.command.index("--tp") + 1] == "4"
    assert server.command[server.command.index("--moe-runner-backend") + 1] == "marlin"
    assert server.environment["CUDA_VISIBLE_DEVICES"] == "2,3,6,7"
    with pytest.raises(ValueError, match="overlap"):
        NodeMimoServices({"gpu_groups": ["0,1,2,3", "3,4,5,6"], "ports": [8094, 8095]}, log_root=tmp_path)


def test_sglang_does_not_inherit_sam_python_dependencies(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_native_resources import SGLangProcess

    monkeypatch.setenv("PYTHONPATH", "/sam-audio-pydeps")
    server = SGLangProcess({"sglang": "/sglang", "checkpoint": "/model"}, "0,1,2,3", 8094, tmp_path)
    assert "PYTHONPATH" not in server.environment
    assert os.environ["PYTHONPATH"] == "/sam-audio-pydeps"
    assert server.environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_substitute_sglang_process_is_reaped(tmp_path, sig):
    from r2v_data_v2.h3.ra2va_group_native_resources import SGLangProcess
    server = SGLangProcess({"sglang": "unused", "checkpoint": "/model"}, "0,1,2,3", 8094, tmp_path)
    server.command = [sys.executable, "-c", "import time; time.sleep(60)"]
    server.launch()
    server.process.send_signal(sig)
    server.close()
    assert server.process.poll() is not None


def test_group_runtime_configs_do_not_require_legacy_hash_fields(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_runtime import NativeBackendFactory
    from r2v_data_v2.h3.sam_audio_stem_shadow import OfficialSAMAudioBackend
    monkeypatch.setattr(OfficialSAMAudioBackend, "_load", lambda self: None)
    config = {"sam": {"implementation_root": "/sam", "model_path": "/weights/sam",
                       "t5_base_path": "/weights/t5"},
              "auk": {"python_path": sys.executable, "code_root": "/auk",
                      "checkpoint_path": "/auk/auk_base.safetensors", "qwen_path": "/qwen"}}
    factory = NativeBackendFactory(config, mimo_config=None)
    with factory("sam") as backend:
        assert backend.configuration.device == "cuda:0"
        assert not hasattr(backend.configuration, "model_checkpoint_sha256")
        assert backend.configuration.predict_spans is False

    monkeypatch.setattr("r2v_data_v2.h3.ra2va_group_runtime.GroupAukBackend.__enter__", lambda self: self)
    monkeypatch.setattr("r2v_data_v2.h3.ra2va_group_runtime.GroupAukBackend.__exit__", lambda *args: None)
    with factory("auk") as backend:
        assert backend.configuration.device == "cuda:0"
        assert "dependency_files" not in backend.configuration.model_dump()


def test_native_asr_adapter_supplies_ffmpeg_to_existing_worker(monkeypatch):
    from types import SimpleNamespace

    from r2v_data_v2.h3.ra2va_group_runtime import NativeBackendFactory
    closed = []

    class Backend:
        _process = SimpleNamespace(poll=lambda: None)

        def __enter__(self):
            return self

        def close(self, **kwargs):
            closed.append(True)

        def transcribe(self, *, waveform, sample_rate_hz):
            assert waveform == "waveform" and sample_rate_hz == 16000
            return " exact frozen text ", "English"
    monkeypatch.setenv("QWEN3_ASR_DEVICE", "test-device")
    monkeypatch.setattr("tools.run_h3_stem_qwen3_asr_shadow._isolated_backend", Backend)
    factory = NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None, ffmpeg="pilot-ffmpeg")
    with factory("asr") as backend:
        assert backend.transcribe(waveform="waveform", sample_rate_hz=16000) == (
            " exact frozen text ", "English")
    assert closed == [True]


def test_native_child_sigkill_is_infrastructure_interruption(tmp_path):
    from r2v_data_v2.h3.ra2va_group_http_native_worker import NativeStageExecution
    executor = NativeStageExecution({"stage": "canonical"}, backend_factory=ProcessFixtureFactory(tmp_path),
        ffmpeg="ffmpeg", execution="native", heartbeat=Permission(), stop_event=threading.Event(), gpu_id="1")
    with executor:
        executor.process.kill()
        executor.process.join(2)
        with pytest.raises(RuntimeError, match="exited"):
            executor._receive()
    assert executor.process.exitcode == -signal.SIGKILL


def test_sglang_busy_port_is_not_adopted_as_our_model(tmp_path):
    import socket

    from r2v_data_v2.h3.ra2va_group_native_resources import SGLangProcess
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        server = SGLangProcess({"sglang": "unused", "checkpoint": "/model"},
                                "0,1,2,3", occupied.getsockname()[1], tmp_path)
        server.command = [sys.executable, "-c", "import time; time.sleep(60)"]
        try:
            with pytest.raises(OSError):
                server.launch()
        finally:
            server.close()
        assert server.process is None


def test_cleanup_reaps_owned_group_after_leader_has_exited(tmp_path):
    import subprocess

    from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process
    child_pid_path = tmp_path / "orphan.pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent_code = ("import subprocess,sys,time; from pathlib import Path; "
                   f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                   f"Path({str(child_pid_path)!r}).write_text(str(child.pid)); time.sleep(.2)")
    parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    parent.wait(timeout=5)
    child_pid = int(child_pid_path.read_text())
    try:
        stop_owned_process(parent, set(), grace=.1)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.02)
        else:
            pytest.fail("owned child survived after its group leader exited")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_late_event_ack_cannot_authorize_request_after_watchdog_expiry(tmp_path):
    from types import SimpleNamespace

    from r2v_data_v2.h3.ra2va_group_http_native_worker import NativeStageExecution
    from r2v_data_v2.h3.ra2va_group_http_worker import HttpConflict
    permission, sent = Permission(), []

    def call(*args):
        permission.allowed = False
        return {"accepted": True}
    executor = NativeStageExecution({"stage": "mimo"}, backend_factory=None,
        ffmpeg="ffmpeg", execution="native", heartbeat=permission, stop_event=threading.Event(),
        client=SimpleNamespace(call=call))
    executor.pipe = SimpleNamespace(send=sent.append)
    executor._receive = lambda: ("event", ("request_started", "visual"))
    task = {"session_id": "session", "task_id": "task", "claim_token": "token",
            "artifact_root": str(tmp_path), "task": {"clip_uid": "clip"}}
    with pytest.raises(HttpConflict, match="watchdog"):
        executor.execute(task)
    assert sent == [task]
