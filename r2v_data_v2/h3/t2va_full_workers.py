"""Generic, resumable GPU-partitioned stage execution (stdlib only).

Call from the main thread, with a single owner for each stage_root. Backends
must keep artifacts inside output_dir and return JSON-serializable values.
No model/runtime dependency is imported by the supervisor.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


class WorkerStageError(RuntimeError):
    """A worker failed outside an individual sample, after the stage barrier."""


class SampleJobFailure(ValueError):
    """A retryable sample failure carrying optional JSON diagnostics."""

    def __init__(self, message: str, result=None):
        super().__init__(message)
        self.result = result


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value) -> None:
    payload = _json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        # Never glob/delete another attempt's partial files or backend artifacts.
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _output_files(directory: Path, *, durable: bool = False) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("media must be an owned directory")
    inventory = {}
    directories = [directory]
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"media symlinks are not owned artifacts: {path}")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise ValueError(f"media is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            if durable:
                os.fsync(handle.fileno())
        inventory[path.relative_to(directory).as_posix()] = {
            "sha256": digest.hexdigest(),
            "size": size,
        }
    if durable:
        for path in reversed(directories):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return inventory


def _receipt(root: Path, item: dict, execution: str):
    value = _read_json(root / "jobs" / item["key"] / "receipt.json")
    if not isinstance(value, dict):
        return None
    if (
        value.get("job_id") != item["job"]["job_id"]
        or value.get("input_fingerprint") != item["input_fingerprint"]
        or value.get("execution_fingerprint") != execution
        or value.get("status") not in ("ready", "failed")
        or "result" not in value
        or "failure_reason" not in value
    ):
        return None
    if value["status"] == "ready" and value["failure_reason"] is not None:
        return None
    if value["status"] == "failed" and not isinstance(value["failure_reason"], str):
        return None
    if value["status"] == "ready":
        try:
            if value.get("output_files") != _output_files(
                root / "jobs" / item["key"] / "media"
            ):
                return None
        except (OSError, ValueError):
            return None
    return {key: value[key] for key in ("status", "result", "failure_reason")}


def _interrupted(signum, _frame):
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _descendants(parents: set[int]) -> set[int]:
    """Include nested backends that started a new session/process group."""
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
        pairs = [tuple(map(int, line.split())) for line in output.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()
    found = set(parents)
    while True:
        children = {pid for pid, ppid in pairs if ppid in found}
        if children <= found:
            return found - parents
        found.update(children)


def _signal_pid(pid: int, sig: int, *, group: bool = False) -> None:
    try:
        (os.killpg if group else os.kill)(pid, sig)
    except ProcessLookupError:
        pass


def _stop_workers(processes: list[subprocess.Popen]) -> None:
    live = {p.pid for p in processes if p.poll() is None}
    nested = _descendants(live) if live else set()
    for pid in nested:
        _signal_pid(pid, signal.SIGTERM)
    for process in processes:
        _signal_pid(process.pid, signal.SIGTERM, group=True)
    deadline = time.monotonic() + 2.0
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.02)
    for pid in nested:
        _signal_pid(pid, signal.SIGKILL)
    for process in processes:
        _signal_pid(process.pid, signal.SIGKILL, group=True)
    for process in processes:
        process.wait()


def execute_stage(
    stage_root: str | Path,
    jobs: list[dict],
    gpu_ids: list[str],
    factory: str,
    configuration: dict,
    python_path: str | Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Run pending[rank::len(gpu_ids)] in one fresh child per nonempty slice.

    python_path selects the worker interpreter (defaults to sys.executable).
    environment overlays a COPY of the parent's environment; GPU visibility
    is always set to the assigned physical ID. Factory(configuration) must be
    a context manager yielding a backend with process(job, output_dir: Path).
    The copied configuration always has device='cuda:0'.

    Return insertion-ordered job_id -> {status, result, failure_reason}. Ready
    receipts skip only when the canonical JSON job (excluding the top-level
    inventory_fingerprint publication field), factory/config identities, and
    all media hashes match. Keep global inventory identity out of configuration;
    the parent re-signs publication. Neither job nor result is rewritten.
    Failed jobs get one attempt per invocation; SampleJobFailure preserves its
    optional JSON result diagnostics. Infrastructure
    failures raise WorkerStageError after all launched workers finish; incomplete
    jobs receive no failure receipt. SIGTERM/interrupt cleans up owned workers
    and re-raises SystemExit/KeyboardInterrupt. output_dir is the fixed absolute
    jobs/<sha256(job_id)>/media path, never renamed. Retries clear only their own
    media; validated ready jobs are never touched. Atomic receipts are published
    only after successful output hashing and fsync.
    """
    if (
        not isinstance(factory, str)
        or len(factory.split(":")) != 2
        or not all(factory.split(":"))
    ):
        raise ValueError("factory must be module:callable")
    if not isinstance(configuration, dict):
        raise TypeError("configuration must be a dict")
    config = json.loads(_json(configuration))
    config["device"] = "cuda:0"
    execution = _digest({"factory": factory, "configuration": config})
    items = []
    seen = set()
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
        input_identity = {
            key: value for key, value in job.items() if key != "inventory_fingerprint"
        }
        items.append(
            {
                "job": json.loads(_json(job)),
                "input_fingerprint": _digest(input_identity),
                "key": hashlib.sha256(job_id.encode("utf-8")).hexdigest(),
            }
        )
    if not items:
        return {}
    if not gpu_ids or any(
        not isinstance(gpu, str) or not gpu.strip() or "," in gpu for gpu in gpu_ids
    ):
        raise ValueError("gpu_ids must contain physical GPU IDs")
    root = Path(stage_root).resolve()
    results = {}
    pending = []
    for item in items:
        cached = _receipt(root, item, execution)
        if cached and cached["status"] == "ready":
            results[item["job"]["job_id"]] = cached
        else:
            pending.append(item)
    if not pending:
        return {item["job"]["job_id"]: results[item["job"]["job_id"]] for item in items}

    invocation = root / "workers" / uuid.uuid4().hex
    child_env = os.environ.copy()
    child_env.update(environment or {})
    processes = []
    manifests = []
    errors = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, _interrupted)
        for rank, gpu in enumerate(gpu_ids):
            partition = pending[rank :: len(gpu_ids)]
            if not partition:
                continue
            manifest = invocation / f"worker-{rank}" / "request.json"
            _atomic_json(
                manifest,
                {
                    "stage_root": str(root),
                    "items": partition,
                    "factory": factory,
                    "configuration": config,
                    "execution_fingerprint": execution,
                    "gpu_id": gpu,
                },
            )
            manifests.append(manifest)
            try:
                with manifest.with_name("worker.log").open("wb") as log:
                    # Defer signals until the newly owned process is registered.
                    mask = signal.pthread_sigmask(signal.SIG_BLOCK, previous)
                    try:
                        processes.append(
                            subprocess.Popen(
                                [
                                    str(python_path or sys.executable),
                                    "-u",
                                    str(Path(__file__).resolve()),
                                    str(manifest),
                                ],
                                env={**child_env, "CUDA_VISIBLE_DEVICES": gpu},
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        )
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            except OSError as exc:
                errors.append(f"worker-{rank} could not start: {exc}")
        for process in processes:
            code = process.wait()
            if code != 0:
                errors.append(f"worker pid={process.pid} exited {code}")
        for manifest in manifests:
            if _read_json(manifest.with_name("complete.json")) != {"complete": True}:
                errors.append(f"incomplete worker: {manifest}")
        for item in pending:
            value = _receipt(root, item, execution)
            if value is None:
                errors.append(f"unfinished job: {item['job']['job_id']}")
            else:
                results[item["job"]["job_id"]] = value
        if errors:
            raise WorkerStageError("; ".join(errors))
        return {item["job"]["job_id"]: results[item["job"]["job_id"]] for item in items}
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            _stop_workers(processes)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _worker(manifest: Path) -> None:
    # The supervisor temporarily masks signals around Popen; do not inherit it.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _interrupted)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    request = json.loads(manifest.read_text(encoding="utf-8"))
    module_name, attribute = request["factory"].split(":")
    make_backend = getattr(importlib.import_module(module_name), attribute)
    root = Path(request["stage_root"])
    with make_backend(request["configuration"]) as backend:
        for item in request["items"]:
            directory = root / "jobs" / item["key"]
            output = directory / "media"
            # Invalidate an old failed/mismatched receipt before retrying this job.
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
            except Exception as exc:  # noqa: BLE001 - isolate arbitrary backend sample failures
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
    _atomic_json(manifest.with_name("complete.json"), {"complete": True})


if __name__ == "__main__":
    # Share exception identity with adapters without importing package-level
    # model dependencies just to run a generic worker.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.modules["r2v_data_v2.h3.t2va_full_workers"] = sys.modules[__name__]
    _worker(Path(sys.argv[1]))
