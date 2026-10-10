"""Native Group stages behind the existing HTTP claim/heartbeat/receipt loop."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_http_worker import HttpConflict, run_http_worker
from r2v_data_v2.h3.ra2va_group_native_resources import stop_owned_process
from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
from r2v_data_v2.h3.ra2va_group_production import read_json
from r2v_data_v2.h3.ra2va_group_source import GroupTask


def _execute(worker, task, execution):
    rows = {stage: read_json(Path(path)) for stage, path in task["upstream_results"].items()}
    if any(row["status"] != "ready" for row in rows.values()):
        return "skipped", {"execution": execution, "reason": "upstream_terminal"}
    worker.root = Path(task["artifact_root"])
    try:
        payload = worker.process(GroupTask(**task["task"]), {k: r["payload"] for k, r in rows.items()})
        status = "failed" if payload.get("product_failed_count", 0) else payload.get("status", "ready")
    except (ValueError, OSError) as error:
        status, payload = "failed", {"failure_reason": f"{type(error).__name__}: {error}"}
    return status, payload | {"execution": execution}


def _stage_child(pipe, stage, factory, ffmpeg, ffprobe, execution, gpu_id):
    os.setsid()
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def stop(*args):
        raise SystemExit("native inference process stopped")
    signal.signal(signal.SIGTERM, stop)
    try:
        with GroupStageWorker(stage, Path("."), backend_factory=factory, ffmpeg=ffmpeg, ffprobe=ffprobe) as worker:
            pipe.send(("ready", os.getpid()))
            while True:
                task = pipe.recv()
                if task is None:
                    return
                pipe.send(("result", _execute(worker, task, execution)))
    except EOFError:
        pass
    except BaseException as error:  # noqa: BLE001 - transport child crashes, including SystemExit
        try:
            pipe.send(("fatal", f"{type(error).__name__}: {error}"))
        except (OSError, EOFError):
            pass
    finally:
        pipe.close()


class _ProcessHandle:
    """Use the same bounded process-group cleanup for Popen and spawn children."""
    def __init__(self, process):
        self.process, self.pid = process, process.pid

    def wait(self, timeout):
        self.process.join(timeout)
        if self.process.is_alive():
            raise subprocess.TimeoutExpired("native stage", timeout)

    def poll(self):
        return self.process.exitcode


class NativeStageExecution:
    def __init__(self, session, *, backend_factory, ffmpeg, execution, heartbeat, stop_event,
                 gpu_id=None, ffprobe="ffprobe", cleanup_seconds=5, services=None, endpoint_index=0):
        self.session, self.factory = session, backend_factory
        self.ffmpeg, self.ffprobe, self.execution = ffmpeg, ffprobe, execution
        self.heartbeat, self.stop_event, self.gpu_id = heartbeat, stop_event, gpu_id
        self.cleanup_seconds, self.services, self.endpoint_index = cleanup_seconds, services, endpoint_index
        self.descendants, self.process, self.pipe = set(), None, None
        self.joined = False

    def allowed(self):
        return not self.stop_event.is_set() and self.heartbeat.execution_allowed()

    def _receive(self):
        from r2v_data_v2.h3.t2va_full_workers import _descendants
        observed = 0
        while self.allowed():
            if time.monotonic() - observed >= .2:
                self.descendants.update(_descendants({self.process.pid}))
                observed = time.monotonic()
            if self.pipe.poll(.05):
                try:
                    kind, value = self.pipe.recv()
                except EOFError as error:
                    raise RuntimeError("Native inference child exited") from error
                if kind == "fatal":
                    raise RuntimeError(value)
                return kind, value
            if not self.process.is_alive():
                raise RuntimeError("Native inference child exited; resume original run-id")
        raise HttpConflict("Native watchdog stopped execution permission")

    def __enter__(self):
        started = time.monotonic()
        try:
            if self.services is not None and self.session["stage"] == "mimo":
                self.services.join(self.session["generation"], self.allowed)
                self.joined = True
            context = mp.get_context("spawn")
            self.pipe, child = context.Pipe()
            self.process = context.Process(target=_stage_child, args=(child, self.session["stage"], self.factory,
                self.ffmpeg, self.ffprobe, self.execution, self.gpu_id))
            self.process.start()
            child.close()
            assert self._receive()[0] == "ready"
            print(json.dumps({"event": "native_stage_loaded", "stage": self.session["stage"],
                "generation": self.session.get("generation"), "pid": self.process.pid, "gpu_id": self.gpu_id,
                "elapsed_seconds": time.monotonic() - started}), flush=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def execute(self, task):
        lease = (self.services.lease(self.endpoint_index, self.allowed) if self.joined else nullcontext())
        with lease:
            if not self.allowed():
                raise HttpConflict("Native execution permission ended")
            self.pipe.send(task)
            kind, value = self._receive()
            assert kind == "result"
            return value

    def __exit__(self, *args):
        started = time.monotonic()
        if not self.allowed():
            self.stop_event.set()
        if self.process is not None and self.process.pid is not None:
            if self.process.is_alive() and self.allowed():
                try:
                    self.pipe.send(None)
                    self.process.join(self.cleanup_seconds)
                except (OSError, EOFError):
                    pass
            stop_owned_process(_ProcessHandle(self.process), self.descendants, grace=self.cleanup_seconds)
        if self.pipe is not None:
            self.pipe.close()
        if self.joined:
            self.services.leave(self.session["generation"])
            self.joined = False
        print(json.dumps({"event": "native_resources_closed", "stage": self.session["stage"],
            "generation": self.session.get("generation"), "elapsed_seconds": time.monotonic() - started}), flush=True)


def run_http_native_worker(client, *, node_id, worker_instance, stop_event, backend_factory,
                           ffmpeg="ffmpeg", ffprobe="ffprobe", execution="native", gpu_id=None,
                           services=None, endpoint_index=0):
    return run_http_worker(client, node_id=node_id, worker_instance=worker_instance, stop_event=stop_event,
        execution_mode=execution, executor_factory=lambda session, heartbeat: NativeStageExecution(
            session, backend_factory=backend_factory, ffmpeg=ffmpeg, ffprobe=ffprobe, execution=execution,
            heartbeat=heartbeat, stop_event=stop_event, gpu_id=gpu_id,
            services=services, endpoint_index=endpoint_index))


def run_http_native_workers(*, coordinator_url, node_id, workers, configuration_path=None,
                            cpu_fixtures=None, ffmpeg="ffmpeg"):
    from r2v_data_v2.h3.ra2va_group_http_worker import GroupHttpClient
    from r2v_data_v2.h3.ra2va_group_native_resources import NodeMimoServices
    from r2v_data_v2.h3.ra2va_group_runtime import load_native_configuration
    if workers < 1:
        raise ValueError("workers must be positive")
    stop, identifier = threading.Event(), uuid.uuid4().hex
    services, factories, gpu_ids = None, [], [None] * workers
    if cpu_fixtures is not None:
        from r2v_data_v2.h3.ra2va_group_cpu_fixtures import CPUFixtureFactory
        factories = [CPUFixtureFactory(cpu_fixtures, ffmpeg) for _ in range(workers)]
        execution, ffprobe = "cpu_fixture", "ffprobe"
    else:
        if configuration_path is None:
            raise ValueError("Native Worker requires --configuration")
        values, _ = load_native_configuration(configuration_path)
        gpu_ids = values["gpu_ids"]
        if len(set(gpu_ids)) != len(gpu_ids) or workers > len(gpu_ids):
            raise ValueError("ordinary GPU Workers require distinct physical GPU IDs")
        services = NodeMimoServices(values["mimo"], log_root=Path(values["log_root"]) / identifier)
        factories = [load_native_configuration(configuration_path, i % len(services.slots))[1]
                     for i in range(workers)]
        execution, ffmpeg, ffprobe = "native", values.get("ffmpeg", "ffmpeg"), values.get("ffprobe", "ffprobe")
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda *args: stop.set())
    started, reports = time.monotonic(), []
    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(run_http_native_worker,
                GroupHttpClient(coordinator_url, os.environ["R2V_GROUP_COORDINATOR_TOKEN"]), node_id=node_id,
                worker_instance=f"{identifier}-{i}", stop_event=stop, backend_factory=factories[i],
                ffmpeg=ffmpeg, ffprobe=ffprobe, execution=execution, gpu_id=gpu_ids[i], services=services,
                endpoint_index=0 if services is None else i % len(services.slots)) for i in range(workers)]
            try:
                for future in as_completed(futures):
                    reports.append(future.result())
            finally:
                stop.set()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return {key: sum(report[key] for report in reports) for key in
            ("executed", "accepted", "model_call_count", "reused_checkpoints")} | {
        "execution": execution, "node_id": node_id, "invocation_elapsed_seconds": time.monotonic() - started}
