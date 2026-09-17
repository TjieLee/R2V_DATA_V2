"""Node-lifetime GPU pools with owned file IPC and ephemeral-compatible receipts.

The manager has one main-thread owner; stage calls are serial. Models stay in
isolated subprocesses. No model/runtime dependency is imported here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

# Also support an isolated interpreter invoking this file directly.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.h3.t2va_full_workers import (
    SampleJobFailure,
    WorkerStageError,
    _atomic_json,
    _descendants,
    _digest,
    _interrupted,
    _json,
    _output_files,
    _read_json,
    _receipt,
    _signal_pid,
    _stop_workers,
)


def _configuration(configuration):
    if not isinstance(configuration, dict):
        raise TypeError("configuration must be a dict")
    config = json.loads(_json(configuration))
    config["device"] = "cuda:0"
    return config


def _model_identity(config, request_keys):
    return _digest(
        {key: value for key, value in config.items() if key not in request_keys}
    )


@dataclass
class _Worker:
    process: subprocess.Popen
    directory: Path


@dataclass
class _Pool:
    identity: str
    request_keys: frozenset[str]
    environment: dict[str, str]
    python_path: str
    workers: list[_Worker] = field(default_factory=list)


class PersistentPoolManager:
    """Own all GPU workers until close, including failures and main-thread signals.

    start() waits for every GPU to enter its backend context. Only explicitly
    listed top-level request_keys may vary between requests; the full normalized
    configuration still participates in receipt identity. startup_timeout is in
    seconds (default 900). A failed manager is never restarted or reused.
    """

    def __init__(self, root, gpu_ids, *, startup_timeout=900.0):
        if (
            not gpu_ids
            or any(
                not isinstance(gpu, str) or not gpu.strip() or "," in gpu or "/" in gpu
                for gpu in gpu_ids
            )
            or len(set(gpu_ids)) != len(gpu_ids)
        ):
            raise ValueError("gpu_ids must contain distinct physical GPU IDs")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("startup_timeout must be finite and positive")
        self.root = Path(root).resolve()
        self.gpu_ids = list(gpu_ids)
        self.startup_timeout = startup_timeout
        self._directory = self.root / uuid.uuid4().hex
        self._pools = {}
        self._names = set()
        self._previous = None
        self._closed = False
        self._unhealthy = False
        self._children = set()
        self._next_observation = 0.0

    def __enter__(self):
        self._check()
        if self._previous is not None:
            raise RuntimeError("manager context already entered")
        self._previous = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        for sig in self._previous:
            signal.signal(sig, self._interrupt)
        return self

    def __exit__(self, *_args):
        self.close()

    def _interrupt(self, signum, frame):
        self.close()
        _interrupted(signum, frame)

    def _check(self):
        if self._unhealthy or self._closed:
            raise WorkerStageError("persistent pool manager is unhealthy or closed")

    def _check_workers(self):
        self._check()
        self._observe_children()
        if any(
            w.process.poll() is not None
            for p in self._pools.values()
            for w in p.workers
        ):
            self._unhealthy = True
            self.close()
            raise WorkerStageError(
                "persistent pool worker exited; manager is unhealthy"
            )

    def _observe_children(self, *, force=False):
        if not force and time.monotonic() < self._next_observation:
            return
        processes = [w.process for p in self._pools.values() for w in p.workers]
        live = {p.pid for p in processes if p.poll() is None}
        descendants = _descendants(live) if live else set()
        # After an abrupt worker exit detached children may already be reparented.
        # Retain the last live snapshot until this unhealthy invocation is closed.
        if len(live) != len(processes):
            self._children.update(descendants)
        else:
            self._children = descendants
        self._next_observation = time.monotonic() + 0.1

    def close(self):
        """Idempotently stop owned process trees and restore prior signal handlers."""
        if self._closed:
            return
        self._closed = True
        previous = self._previous or {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            self._observe_children(force=True)
            for pid in self._children:
                _signal_pid(pid, signal.SIGTERM)
            _stop_workers([w.process for p in self._pools.values() for w in p.workers])
        finally:
            for pid in self._children:
                _signal_pid(pid, signal.SIGKILL)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self._previous = None

    def start(
        self,
        name,
        factory,
        configuration,
        environment=None,
        python_path=None,
        request_keys=(),
    ):
        """Start every GPU concurrently and return only after all READY markers."""
        self._check_workers()
        if self._previous is None:
            self.__enter__()
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise ValueError("pool name must be a single path component")
        if (
            not isinstance(factory, str)
            or len(factory.split(":")) != 2
            or not all(factory.split(":"))
        ):
            raise ValueError("factory must be module:callable")
        if name in self._names or factory in self._pools:
            raise ValueError("duplicate pool name or factory")
        if isinstance(request_keys, str) or any(
            not isinstance(k, str) or not k for k in request_keys
        ):
            raise ValueError("request_keys must contain configuration keys")
        keys = frozenset(request_keys)
        if "device" in keys:
            raise ValueError("device is model configuration, not a request key")
        config = _configuration(configuration)
        pool = _Pool(
            _model_identity(config, keys),
            keys,
            dict(environment or {}),
            str(python_path or sys.executable),
        )
        self._pools[factory] = pool
        self._names.add(name)
        started = time.monotonic()
        deadline = started + self.startup_timeout
        try:
            for rank, gpu in enumerate(self.gpu_ids):
                directory = self._directory / name / f"worker-{rank}"
                manifest = directory / "startup.json"
                _atomic_json(manifest, {"factory": factory, "configuration": config})
                with (self.root / f"{name}-gpu{gpu}.log").open("ab") as log:
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, self._previous)
                    try:
                        process = subprocess.Popen(
                            [
                                pool.python_path,
                                "-u",
                                str(Path(__file__).resolve()),
                                str(manifest),
                            ],
                            env={
                                **os.environ,
                                **pool.environment,
                                "CUDA_VISIBLE_DEVICES": gpu,
                            },
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        pool.workers.append(_Worker(process, directory))
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            while True:
                self._check_workers()
                if all(
                    _read_json(w.directory / "ready.json") == {"ready": True}
                    for w in pool.workers
                ):
                    print(
                        f"pool={name} workers={len(pool.workers)} "
                        f"startup_seconds={time.monotonic() - started:.3f}",
                        flush=True,
                    )
                    return
                if time.monotonic() >= deadline:
                    raise WorkerStageError(f"pool {name} startup timeout")
                time.sleep(0.02)
        except BaseException as exc:
            self._unhealthy = True
            self.close()
            if isinstance(exc, OSError):
                raise WorkerStageError(f"pool {name} could not start: {exc}") from exc
            raise

    def execute_stage(
        self,
        stage_root: str | Path,
        jobs: list[dict],
        gpu_ids: list[str],
        factory: str,
        configuration: dict,
        python_path: str | Path | None = None,
        environment: dict[str, str] | None = None,
        log_root: str | Path | None = None,
    ) -> dict[str, dict]:
        """Execute with the ephemeral call signature, selecting a started factory.

        Logs remain at manager root/<pool>-gpu<ID>.log for the entire model
        lifetime; log_root is accepted for adapter compatibility. Optional runtime
        arguments, when supplied, must match start(). All participating workers
        reach their request barrier before an infrastructure error is raised.
        """
        self._check_workers()
        if factory not in self._pools:
            raise ValueError(f"factory has no started pool: {factory}")
        pool = self._pools[factory]
        config = _configuration(configuration)
        if _model_identity(config, pool.request_keys) != pool.identity:
            raise ValueError("persistent pool model configuration mismatch")
        if list(gpu_ids) != self.gpu_ids:
            raise ValueError("persistent pool GPU mapping mismatch")
        if python_path is not None and str(python_path) != pool.python_path:
            raise ValueError("persistent pool python_path mismatch")
        if environment is not None and environment != pool.environment:
            raise ValueError("persistent pool environment mismatch")
        execution = _digest({"factory": factory, "configuration": config})
        items, seen = [], set()
        for job in jobs:
            if (
                not isinstance(job, dict)
                or not isinstance(job.get("job_id"), str)
                or not job["job_id"]
            ):
                raise ValueError("every job must have a nonempty string job_id")
            job_id = job["job_id"]
            if job_id in seen:
                raise ValueError(f"duplicate job_id: {job_id}")
            seen.add(job_id)
            items.append(
                {
                    "job": json.loads(_json(job)),
                    "key": hashlib.sha256(job_id.encode("utf-8")).hexdigest(),
                    "input_fingerprint": _digest(
                        {k: v for k, v in job.items() if k != "inventory_fingerprint"}
                    ),
                }
            )
        if not items:
            return {}
        root = Path(stage_root).resolve()
        results, pending = {}, []
        for item in items:
            cached = _receipt(root, item, execution)
            if cached and cached["status"] == "ready":
                results[item["job"]["job_id"]] = cached
            else:
                pending.append(item)
        _atomic_json(
            root / "invocation.json",
            {
                "job_count": len(items),
                "scheduled_job_count": len(pending),
                "reused_ready_count": len(results),
                "worker_count": min(len(pending), len(pool.workers)),
            },
        )
        if pending:
            request_id = uuid.uuid4().hex
            waiting, errors = [], []
            try:
                for rank, worker in enumerate(pool.workers):
                    partition = pending[rank :: len(pool.workers)]
                    if not partition:
                        continue
                    _atomic_json(
                        worker.directory / "request.json",
                        {
                            "id": request_id,
                            "stage_root": str(root),
                            "items": partition,
                            "configuration": config,
                            "execution_fingerprint": execution,
                        },
                    )
                    waiting.append(worker)
                while waiting:
                    self._observe_children()
                    for worker in waiting[:]:
                        status = _read_json(worker.directory / "complete.json")
                        if worker.process.poll() is not None:
                            errors.append(
                                f"worker pid={worker.process.pid} exited {worker.process.returncode}"
                            )
                            waiting.remove(worker)
                        elif status == {"id": request_id, "complete": True}:
                            waiting.remove(worker)
                    if waiting:
                        time.sleep(0.02)
                for item in pending:
                    value = _receipt(root, item, execution)
                    if value is None:
                        errors.append(f"unfinished job: {item['job']['job_id']}")
                    else:
                        results[item["job"]["job_id"]] = value
                if errors:
                    raise WorkerStageError("; ".join(errors))
                self._check_workers()
            except BaseException:
                self._unhealthy = True
                self.close()
                raise
        return {item["job"]["job_id"]: results[item["job"]["job_id"]] for item in items}


def _process_batch(backend, request):
    root = Path(request["stage_root"])
    for item in request["items"]:
        directory = root / "jobs" / item["key"]
        output = directory / "media"
        (directory / "receipt.json").unlink(missing_ok=True)
        if output.is_symlink():
            output.unlink()
        elif output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        files = None
        try:
            result = backend.process(item["job"], output)
            _json(result)
            files = _output_files(output, durable=True)
            row = {"status": "ready", "result": result, "failure_reason": None}
        except Exception as exc:  # noqa: BLE001 - match ephemeral sample isolation
            diagnostics = exc.result if isinstance(exc, SampleJobFailure) else None
            try:
                _json(diagnostics)
            except (TypeError, ValueError):
                diagnostics = None
            row = {
                "status": "failed",
                "result": diagnostics,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        _atomic_json(
            directory / "receipt.json",
            {
                "job_id": item["job"]["job_id"],
                "input_fingerprint": item["input_fingerprint"],
                "execution_fingerprint": request["execution_fingerprint"],
                "output_files": files,
                **row,
            },
        )


def _worker(manifest):
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _interrupted)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    startup = json.loads(manifest.read_text(encoding="utf-8"))
    module, attribute = startup["factory"].split(":")
    make_backend = getattr(importlib.import_module(module), attribute)
    directory = manifest.parent
    with make_backend(startup["configuration"]) as backend:
        _atomic_json(directory / "ready.json", {"ready": True})
        previous = None
        while True:
            request = _read_json(directory / "request.json")
            if request is None or request["id"] == previous:
                time.sleep(0.02)
                continue
            bind = getattr(backend, "bind_request", None)
            if bind is not None:
                bind(request["configuration"])
            batch = getattr(backend, "batch_jobs", None)
            with (
                batch([item["job"] for item in request["items"]])
                if batch
                else nullcontext()
            ):
                _process_batch(backend, request)
            previous = request["id"]
            _atomic_json(
                directory / "complete.json", {"id": previous, "complete": True}
            )


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
