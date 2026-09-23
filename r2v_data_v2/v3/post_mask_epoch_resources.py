"""Owned Qwen / Boogu / SAM resource lifecycles for resource-epoch execution.

Every resource here obeys one rule: it may only ever signal a PID or process
group that *this* launcher created. A resource epoch switches models on all
eight GPUs, so an over-broad ``pkill`` would take down unrelated jobs sharing
the node. Nothing here reimplements a model call, a prompt, a threshold or an
accept/reject rule; each epoch only hands the semantic layer a live handle.

Three epochs are supported:

    Qwen   one locally managed vLLM server, TP1 x DP8 across GPU0-7
    Boogu  eight persistent Qwen-Image worker processes, one per GPU; the
           resource name is retained for durable scheduler compatibility
    SAM    eight persistent SAM3 workers, one per GPU

Exactly one heavy GPU resource is loaded at a time. :class:`ResourceEpochManager`
owns that invariant: entering a resource fully exits the previous one.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Protocol

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    job_order_key,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    BatchJobExecutor,
    JobExecution,
    JobResult,
    ResourceJobExecutor,
)

#: Execution-only CPU worker budget for the Resource Epoch. Deliberately an
#: environment variable rather than a ``V3Config`` field: ``runtime.cpu_workers``
#: is part of the config fingerprint, so changing it would change run identity
#: for a pure execution decision. It never reaches a ModelJob, an input digest, a
#: semantic plan, a receipt or any public schema.
CPU_WORKERS_ENV = "POST_MASK_CPU_WORKERS"
QWEN_MAX_INFLIGHT_ENV = "POST_MASK_QWEN_MAX_INFLIGHT"

#: Kept for compatibility with callers that never set the environment.
DEFAULT_CPU_WORKERS = 8

#: Upper bound for pure disk-hashing fan-out. Beyond this the work is I/O bound
#: and more threads only add contention.
HASH_WORKER_CAP = 16


class EpochResourceError(RuntimeError):
    """A managed resource could not be started or stopped safely."""


def resolve_cpu_workers(config: Any = None) -> int:
    """Resolve the Resource Epoch CPU worker budget. Execution-only.

    Precedence: ``POST_MASK_CPU_WORKERS``, then the config's existing
    ``runtime.cpu_workers`` for old callers, then :data:`DEFAULT_CPU_WORKERS`.
    """
    raw = os.environ.get(CPU_WORKERS_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise EpochResourceError(
                f"{CPU_WORKERS_ENV} must be an integer, got {raw!r}"
            ) from exc
        if value < 1:
            raise EpochResourceError(f"{CPU_WORKERS_ENV} must be >= 1, got {value}")
        return value
    fallback = getattr(getattr(config, "runtime", None), "cpu_workers", None)
    if isinstance(fallback, int) and not isinstance(fallback, bool) and fallback >= 1:
        return fallback
    return DEFAULT_CPU_WORKERS


def resolve_qwen_max_inflight() -> int:
    """Execution-only Qwen request concurrency; absent override keeps DP8 default."""
    raw = os.environ.get(QWEN_MAX_INFLIGHT_ENV, "").strip()
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError as exc:
        raise EpochResourceError(
            f"{QWEN_MAX_INFLIGHT_ENV} must be a positive integer"
        ) from exc
    if value < 1:
        raise EpochResourceError(
            f"{QWEN_MAX_INFLIGHT_ENV} must be a positive integer"
        )
    return value


def resolve_hash_workers(config: Any = None, *, tasks: int | None = None) -> int:
    """Worker budget for read-only disk hashing, capped and never below 1."""
    workers = min(resolve_cpu_workers(config), HASH_WORKER_CAP)
    if tasks is not None:
        workers = min(workers, max(1, tasks))
    return max(1, workers)


def limit_process_native_threads() -> None:
    """Stop OpenCV/pyav from oversubscribing the epoch orchestrator process.

    Execution-only: model work runs in its own subprocesses, whose CPU policy is
    not touched. A missing OpenCV is not an error - some deployments run the
    orchestrator without it.
    """
    try:
        import cv2  # optional: only present in the model image
    except ImportError:
        return
    try:
        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 - never fail a run over a tuning call
        return


@dataclass(frozen=True)
class OwnedProcess:
    """A process this launcher created and is therefore allowed to signal."""

    pid: int
    pgid: int
    argv: tuple[str, ...]
    log_path: Path | None = None


class EpochProcessManager(Protocol):
    """Spawn and stop owned processes. Real and fake implementations exist."""

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        log_path: Path | None = None,
    ) -> OwnedProcess: ...

    def alive(self, process: OwnedProcess) -> bool: ...

    def stop(self, process: OwnedProcess, *, grace_seconds: float) -> bool: ...


class SubprocessEpochProcessManager:
    """Spawn each resource in its own process group and stop only that group.

    The ``Popen`` handle is retained so the child is always waited on and
    reaped: after a normal stop, after a startup failure, and after a SIGTERM
    timeout escalates to SIGKILL.
    """

    def __init__(self) -> None:
        self._children: dict[int, subprocess.Popen] = {}
        self.reaped: list[int] = []
        self._reap_timeout_seconds = 30.0

    @property
    def open_children(self) -> tuple[int, ...]:
        return tuple(sorted(self._children))

    def spawn(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        log_path: Path | None = None,
    ) -> OwnedProcess:
        if not argv:
            raise EpochResourceError("resource command is empty")
        handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
        try:
            child = subprocess.Popen(
                list(argv),
                env=dict(env),
                stdout=handle if handle is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            if handle is not None:
                handle.close()
        self._children[child.pid] = child
        pgid = child.pid
        getpgid = getattr(os, "getpgid", None)
        if getpgid is not None:
            try:
                pgid = getpgid(child.pid)
            except OSError:  # pragma: no cover - race with immediate exit
                pgid = child.pid
        return OwnedProcess(
            pid=child.pid, pgid=pgid, argv=tuple(argv), log_path=log_path
        )

    def alive(self, process: OwnedProcess) -> bool:
        child = self._children.get(process.pid)
        if child is not None:
            return child.poll() is None
        try:
            os.kill(process.pid, 0)
        except OSError:
            return False
        return True

    def _signal_group(self, process: OwnedProcess, signal_number: int) -> bool:
        """Signal only the process group this manager created."""
        killpg = getattr(os, "killpg", None)
        if killpg is None:
            return False
        try:
            killpg(process.pgid, signal_number)
        except (OSError, ValueError):
            return False
        return True

    def stop(self, process: OwnedProcess, *, grace_seconds: float) -> bool:
        import signal as signal_module

        child = self._children.get(process.pid)
        if child is None:
            # Never signal a PID we did not spawn. A misrouted killpg could
            # take down an unrelated job sharing the node.
            return False
        if child.poll() is not None:
            # Already exited (for example a failed health check). Reap only:
            # signalling a dead process group risks hitting a recycled PGID.
            self._reap(process, child)
            return True
        self._signal_group(process, signal_module.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if child.poll() is not None:
                break
            time.sleep(0.05)
        if child.poll() is None:
            # SIGKILL is POSIX-only; fall back to TERM where it is absent.
            fatal = getattr(signal_module, "SIGKILL", signal_module.SIGTERM)
            self._signal_group(process, fatal)
        self._reap(process, child)
        return child.poll() is not None

    def _reap(self, process: OwnedProcess, child: Any) -> None:
        """Always wait on an owned child so it cannot linger as a zombie."""
        try:
            child.wait(timeout=max(self._reap_timeout_seconds, 1.0))
        except subprocess.TimeoutExpired:  # pragma: no cover - hostile child
            child.kill()
            child.wait()
        finally:
            self._children.pop(process.pid, None)
            self.reaped.append(process.pid)


@dataclass
class EpochResource:
    """Load-once / stop-once base with startup and shutdown wall measurements."""

    name: str
    process_manager: EpochProcessManager
    log_root: Path | None = None
    start_count: int = 0
    stop_count: int = 0
    startup_wall_seconds: float = 0.0
    shutdown_wall_seconds: float = 0.0
    service_seconds: float = 0.0
    _started_at: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._owned: list[OwnedProcess] = []
        self._open = False

    @property
    def open(self) -> bool:
        return self._open

    def _log_path(self, stem: str) -> Path | None:
        if self.log_root is None:
            return None
        self.log_root.mkdir(parents=True, exist_ok=True)
        return self.log_root / f"{stem}.log"

    def start(self) -> None:
        if self._open:
            raise EpochResourceError(f"{self.name} epoch is already started")
        started = time.monotonic()
        try:
            self._start()
        except BaseException:
            self.stop()
            raise
        self.startup_wall_seconds += time.monotonic() - started
        self.start_count += 1
        self._started_at = time.monotonic()
        self._open = True

    def stop(self) -> None:
        started = time.monotonic()
        if self._started_at is not None:
            self.service_seconds += time.monotonic() - self._started_at
            self._started_at = None
        self._stop()
        if self._open or self._owned:
            self.stop_count += 1
        self._open = False
        self.shutdown_wall_seconds += time.monotonic() - started

    def _start(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _stop(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def counters(self) -> dict[str, float]:
        return {
            "start_count": self.start_count,
            "stop_count": self.stop_count,
            "startup_wall_seconds": self.startup_wall_seconds,
            "shutdown_wall_seconds": self.shutdown_wall_seconds,
            "service_seconds": self.service_seconds,
        }


# ---------------------------------------------------------------------------
# Qwen epoch: one managed vLLM server on all eight GPUs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenEpochConfig:
    model_path: Path
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    data_parallel_size: int = 8
    max_model_len: int = 49152
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"
    allowed_local_media_path: Path | None = None
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    executable: str = "vllm"
    startup_timeout_seconds: float = 1800.0
    shutdown_grace_seconds: float = 30.0
    health_path: str = "/v1/models"
    #: Startup polling interval. A DP8 cold start can take a long time, so the
    #: default is 1s rather than hammering /v1/models at 20 QPS. Execution-only.
    health_poll_interval_seconds: float = 1.0
    #: Explicit served model id. vLLM reports the model path on this server,
    #: not just its basename, so callers should pass the absolute path.
    served_model_name: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def expected_served_model_id(self) -> str:
        return self.served_model_name or str(self.model_path)

    def argv(self) -> tuple[str, ...]:
        argv = [
            self.executable,
            "serve",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--data-parallel-size",
            str(self.data_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--dtype",
            self.dtype,
            "--served-model-name",
            self.expected_served_model_id,
        ]
        if self.allowed_local_media_path is not None:
            argv += [
                "--allowed-local-media-path",
                str(self.allowed_local_media_path),
            ]
        return tuple(argv)

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in self.gpu_ids)
        return env


def port_in_use(host: str, port: int) -> bool:
    """True when something already listens; used for managed-mode fail-fast."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        return probe.connect_ex((host, port)) == 0
    finally:
        probe.close()


def default_health_probe(base_url: str, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url}/models", method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def served_model_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Every model id advertised by /v1/models, not just the first one."""
    data = payload.get("data")
    if not isinstance(data, list):
        return ()
    return tuple(
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id") is not None
    )


class QwenEpochResource(EpochResource):
    """Start, validate and stop one owned vLLM server for a Qwen epoch."""

    def __init__(
        self,
        config: QwenEpochConfig,
        *,
        process_manager: EpochProcessManager,
        log_root: Path | None = None,
        health_probe: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            RESOURCE_QWEN, process_manager=process_manager, log_root=log_root
        )
        self.config = config
        self.health_probe = health_probe or default_health_probe
        self.process: OwnedProcess | None = None

    def _start(self) -> None:
        config = self.config
        if port_in_use(config.host, config.port):
            # Managed mode must never adopt or kill someone else's server.
            raise EpochResourceError(
                f"port {config.port} on {config.host} is already serving; "
                "refusing to adopt an unmanaged Qwen service"
            )
        executable = config.executable
        if "/" not in executable and "\\" not in executable:
            resolved = shutil.which(executable)
            if resolved is None:
                raise EpochResourceError(
                    f"vLLM executable {executable!r} not found on PATH"
                )
            executable = resolved
        argv = (executable, *config.argv()[1:])
        self.process = self.process_manager.spawn(
            argv,
            env=config.environment(),
            log_path=self._log_path("qwen-vllm"),
        )
        self._await_ready()

    def _await_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        last_error: Exception | None = None
        expected = self.config.expected_served_model_id
        while time.monotonic() < deadline:
            if self.process is not None and not self.process_manager.alive(self.process):
                raise EpochResourceError(
                    "managed Qwen process exited before becoming healthy"
                ) from last_error
            try:
                payload = self.health_probe(self.config.base_url)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
                time.sleep(self.config.health_poll_interval_seconds)
                continue
            if expected in served_model_ids(payload):
                return
            last_error = EpochResourceError(
                f"served models {served_model_ids(payload)} do not include {expected!r}"
            )
            time.sleep(self.config.health_poll_interval_seconds)
        raise EpochResourceError("managed Qwen startup timed out") from last_error

    def _stop(self) -> None:
        if self.process is None:
            return
        self.process_manager.stop(
            self.process, grace_seconds=self.config.shutdown_grace_seconds
        )
        self.process = None


# ---------------------------------------------------------------------------
# Boogu / SAM epochs: eight persistent workers, one per GPU
# ---------------------------------------------------------------------------


def deterministic_slot(job: Any, slot_count: int) -> int:
    """Static hash slot for single-job dispatch; no stealing, no balancing."""
    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    return int(job.job_id(), 16) % slot_count


@dataclass(frozen=True)
class WorkerPoolConfig:
    gpu_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    timeout_seconds: float = 900.0
    shutdown_grace_seconds: float = 30.0


class WorkerEpochResource(EpochResource):
    """Load ``len(gpu_ids)`` persistent workers once and drain them."""

    def __init__(
        self,
        name: str,
        pool: WorkerPoolConfig,
        *,
        process_manager: EpochProcessManager,
        worker_factory: Callable[[int, int], Any],
        log_root: Path | None = None,
    ) -> None:
        super().__init__(name, process_manager=process_manager, log_root=log_root)
        self.pool = pool
        self.worker_factory = worker_factory
        self.workers: list[Any] = []
        self.slot_job_counts: dict[int, int] = {
            slot: 0 for slot in range(len(pool.gpu_ids))
        }

    @property
    def slot_count(self) -> int:
        return len(self.pool.gpu_ids)

    def _start(self) -> None:
        """Load every GPU's worker concurrently, then restore slot order.

        Serial construction would multiply a Boogu epoch's model-load latency
        by the slot count. Results are re-ordered by slot, so completion order
        never changes slot mapping.
        """
        slots = list(enumerate(self.pool.gpu_ids))
        if not slots:
            return
        built: dict[int, Any] = {}
        failures: dict[int, BaseException] = {}
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            futures = {
                pool.submit(self.worker_factory, slot, gpu_id): slot
                for slot, gpu_id in slots
            }
            for future, slot in futures.items():
                try:
                    built[slot] = future.result()
                except Exception as exc:  # noqa: BLE001 - rollback below
                    failures[slot] = exc
        if failures:
            # Never leave a partial set of GPU models resident. Every
            # successfully built worker is asked to close, and one failing
            # close must not abort the rollback: stopping after the first
            # cleanup exception would strand the remaining slots' models in
            # GPU memory for the rest of the process.
            cleanup_errors: list[str] = []
            for slot in sorted(built):
                close = getattr(built[slot], "close", None)
                if not callable(close):
                    continue
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - best-effort rollback
                    cleanup_errors.append(f"slot {slot}: {exc}")
            built.clear()
            self.workers = []
            first = min(failures)
            # The original startup failure stays the primary error; cleanup
            # failures are appended, never substituted for it.
            message = (
                f"{self.name} worker slot {first} failed to start: {failures[first]}"
            )
            if cleanup_errors:
                message += f" (rollback incomplete: {'; '.join(cleanup_errors)})"
            raise EpochResourceError(message)
        self.workers = [built[slot] for slot in sorted(built)]

    def _stop(self) -> None:
        """Close every worker, then report. One failure must not leak the rest."""
        workers, self.workers = self.workers, []
        errors: list[str] = []
        for slot, worker in enumerate(workers):
            close = getattr(worker, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - best-effort shutdown
                errors.append(f"slot {slot}: {exc}")
        if errors:
            raise EpochResourceError(
                f"{self.name} shutdown incomplete: {'; '.join(errors)}"
            )

    def handle_for_slot(self, slot_id: int) -> Any:
        """Return the live worker backend for a slot, never the slot id.

        The semantic runner must not learn about GPU slots or indices; it only
        ever receives the backend handle it should call.
        """
        if not self._open:
            raise EpochResourceError(f"{self.name} epoch is not started")
        if not 0 <= slot_id < self.slot_count:
            raise EpochResourceError(f"invalid {self.name} slot: {slot_id}")
        self.slot_job_counts[slot_id] = self.slot_job_counts.get(slot_id, 0) + 1
        return self.workers[slot_id]

    def handle_for(self, job: Any) -> Any:
        return self.handle_for_slot(deterministic_slot(job, self.slot_count))

    def counters(self) -> dict[str, float]:
        payload = super().counters()
        payload["gpu_slot_job_counts"] = dict(sorted(self.slot_job_counts.items()))
        return payload


def boogu_worker_factory(
    *,
    config: Any,
    temporary_root: Path,
    allowed_server_root: Path,
    pool: WorkerPoolConfig,
) -> Callable[[int, int], Any]:
    """Build one persistent image-edit subprocess pinned to one physical GPU."""

    def factory(slot: int, gpu_id: int) -> Any:
        backend_name = str(getattr(config, "backend", ""))
        if backend_name == "qwen_image_2_1":
            from r2v_data_v2.v3.qwen_image21_backend import (
                QwenImage21SubprocessBackend,
                QwenImage21WorkerConfig,
            )

            worker_config = QwenImage21WorkerConfig(
                python_executable=config.python_executable,
                model_path=config.model_path,
                prompt_enhancer_i2i_path=config.prompt_enhancer_i2i_path,
                num_inference_steps=config.num_inference_steps,
                device="cuda:0",
                cuda_visible_devices=str(gpu_id),
                timeout_seconds=int(
                    min(config.timeout_seconds, pool.timeout_seconds)
                ),
                temporary_root=Path(temporary_root) / f"slot-{slot}",
            )
            backend = QwenImage21SubprocessBackend(worker_config)
            backend.start(
                stderr_log_path=Path(temporary_root) / f"image-edit-slot-{slot}.log"
            )
            return backend

        if backend_name != "boogu_image_0_1_edit_turbo":
            raise EpochResourceError(
                f"unsupported image-edit backend for resource epoch: {backend_name!r}"
            )

        from r2v_data_v2.v3.reference_edit_boogu import (
            BooguSubprocessBackend,
            BooguWorkerConfig,
        )

        worker_config = BooguWorkerConfig(
            python_executable=config.python_executable,
            code_root=config.code_root,
            model_path=config.model_path,
            model_revision=config.model_revision,
            device="cuda:0",
            cuda_visible_devices=str(gpu_id),
            timeout_seconds=int(
                min(config.timeout_seconds, pool.timeout_seconds)
            ),
            temporary_root=Path(temporary_root) / f"slot-{slot}",
            allowed_server_root=allowed_server_root,
        )
        backend = BooguSubprocessBackend(worker_config)
        backend.start(
            stderr_log_path=Path(temporary_root) / f"boogu-slot-{slot}.log"
        )
        return backend

    return factory


def sam_worker_factory(
    *,
    config: Any,
    pool: WorkerPoolConfig,
    backend_builder: Callable[[Any], Any] | None = None,
) -> Callable[[int, int], Any]:
    """Build one persistent SAM3 worker bound to the *physical* GPU id.

    ``gpu_ids`` may be non-contiguous, so the logical device must follow
    ``gpu_id`` rather than the slot index. SAM prompts, mask semantics,
    thresholds and review behaviour are inherited unchanged.
    """

    def factory(slot: int, gpu_id: int) -> Any:
        from dataclasses import replace

        if backend_builder is not None:
            return backend_builder(replace(config, device=f"cuda:{gpu_id}"))
        from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend

        return Sam3SegmentationBackend(replace(config, device=f"cuda:{gpu_id}"))

    return factory


def build_boogu_epoch(
    config: Any,
    *,
    pool: WorkerPoolConfig,
    process_manager: EpochProcessManager,
    temporary_root: Path,
    allowed_server_root: Path,
    log_root: Path | None = None,
) -> WorkerEpochResource:
    return WorkerEpochResource(
        RESOURCE_BOOGU,
        pool,
        process_manager=process_manager,
        worker_factory=boogu_worker_factory(
            config=config,
            temporary_root=temporary_root,
            allowed_server_root=allowed_server_root,
            pool=pool,
        ),
        log_root=log_root,
    )


def build_sam_epoch(
    config: Any,
    *,
    pool: WorkerPoolConfig,
    process_manager: EpochProcessManager,
    log_root: Path | None = None,
    backend_builder: Callable[[Any], Any] | None = None,
) -> WorkerEpochResource:
    return WorkerEpochResource(
        RESOURCE_SAM,
        pool,
        process_manager=process_manager,
        worker_factory=sam_worker_factory(config=config, pool=pool, backend_builder=backend_builder),
        log_root=log_root,
    )


# ---------------------------------------------------------------------------
# Batch executors
# ---------------------------------------------------------------------------


class SerialBatchExecutor:
    """Run one job at a time. Useful for tests and CPU-only paths."""

    def __init__(self, runner: Callable[[Any, Any], JobResult], handle: Any = None):
        self.runner = runner
        self.handle = handle
        self._queued: list[Any] = []

    def capacity(self) -> int:
        return 1

    def submit(self, job: Any) -> None:
        self._queued.append(job)

    def collect(self) -> list[JobExecution]:
        if not self._queued:
            return []
        job, self._queued = self._queued[0], self._queued[1:]
        try:
            return [JobExecution(job, self.runner(job, self.handle), None)]
        except Exception as exc:  # noqa: BLE001 - isolate per job
            return [JobExecution(job, None, exc)]

    def close(self) -> None:
        self._queued = []

    def execute_batch(
        self, jobs: Sequence[Any]
    ) -> Mapping[str, JobExecution]:
        results: dict[str, JobExecution] = {}
        for job in sorted(jobs, key=job_order_key):
            self.submit(job)
            for execution in self.collect():
                results[execution.job.job_id()] = execution
        return results


class QwenConcurrentExecutor:
    """Keep up to ``max_inflight`` requests live on the TP1 x DP8 endpoint.

    A slot that comes back is refilled immediately: the scheduler is told about
    each completion as it lands instead of waiting for the whole batch, so the
    vLLM server is never sitting idle behind one slow request.
    """

    def __init__(
        self,
        runner: Callable[[Any, Any], JobResult],
        *,
        endpoint: Any = None,
        max_inflight: int = 8,
    ):
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        self.runner = runner
        self.endpoint = endpoint
        self.max_inflight = max_inflight
        self._pool: ThreadPoolExecutor | None = None
        self._inflight: dict[Any, Any] = {}

    def capacity(self) -> int:
        return self.max_inflight

    def submit(self, job: Any) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.max_inflight)
        self._inflight[self._pool.submit(self.runner, job, self.endpoint)] = job

    def collect(self) -> list[JobExecution]:
        if not self._inflight:
            return []
        done, _ = wait(list(self._inflight), return_when=FIRST_COMPLETED)
        settled = sorted(
            ((self._inflight.pop(future), future) for future in done),
            key=lambda pair: job_order_key(pair[0]),
        )
        executions: list[JobExecution] = []
        for job, future in settled:
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate per job
                executions.append(JobExecution(job, None, exc))
            else:
                executions.append(JobExecution(job, result, None))
        return executions

    def close(self) -> None:
        """Stop accepting work and drain every request that actually started.

        A control-flow exception can escape collect() while sibling requests are
        still executing. The resource manager must not unload Qwen underneath
        those threads, so queued work is cancelled but running calls are joined
        before the executor is considered closed. Normal fixed-point shutdown has
        no in-flight work, so this adds no steady-state latency.
        """
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        self._inflight.clear()

    def execute_batch(self, jobs: Sequence[Any]) -> Mapping[str, JobExecution]:
        results: dict[str, JobExecution] = {}
        try:
            for job in sorted(jobs, key=job_order_key):
                self.submit(job)
            while self._inflight:
                for execution in self.collect():
                    results[execution.job.job_id()] = execution
            return results
        finally:
            # A BaseException from one future must still drain/cancel siblings.
            self.close()


@dataclass(frozen=True)
class _SlotAbort:
    """A control-flow exception from a slot thread, never a per-job failure."""

    exception: BaseException


class WorkerSlotExecutor:
    """One persistent worker per GPU slot, fed only when that slot is free.

    A job is handed to a slot that is *actually* available at submission time.
    Targeting a fixed round-robin slot instead would queue the job behind
    whatever that slot is still running while other slots sit idle - which is
    exactly the stall the streaming refill exists to remove.

    Placement stays execution-only and never affects ModelJob identity: when
    several slots are free the next one in round-robin order is taken, so work
    still spreads across GPUs, and a slot never runs two jobs at once. The handle
    the runner receives is the live backend of the slot that really executes it.
    """

    def __init__(
        self,
        runner: Callable[[Any, Any], JobResult],
        *,
        slot_count: int = 8,
        resource: Any = None,
    ):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        self.runner = runner
        self.slot_count = slot_count
        self.resource = resource
        self._queues: list[Queue] = [Queue() for _ in range(slot_count)]
        self._events: Queue = Queue()
        self._free: set[int] = set(range(slot_count))
        self._next_slot = 0
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._slot_freed = threading.Condition(self._lock)

    def capacity(self) -> int:
        return self.slot_count

    def submit(self, job: Any) -> None:
        self._start_slots()
        with self._slot_freed:
            while not self._free:
                # Capacity is slot_count, so the scheduler never gets here. A
                # direct over-submission waits for a slot rather than queueing
                # behind one that is already busy.
                self._slot_freed.wait()
            slot_id = self._claim_free_slot()
        self._queues[slot_id].put(job)

    def _claim_free_slot(self) -> int:
        """Reserve the next free slot in round-robin order. Lock is held."""
        start = self._next_slot % self.slot_count
        for offset in range(self.slot_count):
            candidate = (start + offset) % self.slot_count
            if candidate in self._free:
                self._free.discard(candidate)
                self._next_slot = candidate + 1
                return candidate
        raise RuntimeError("no free slot available")  # pragma: no cover

    def _start_slots(self) -> None:
        with self._lock:
            if self._threads:
                return
            for slot_id in range(self.slot_count):
                thread = threading.Thread(
                    target=self._run_slot, args=(slot_id,), daemon=True
                )
                thread.start()
                self._threads.append(thread)

    def _handle_for(self, slot_id: int) -> Any:
        # The runner receives the live backend, never a GPU slot id.
        if self.resource is None:
            return slot_id
        return self.resource.handle_for_slot(slot_id)

    def _run_slot(self, slot_id: int) -> None:
        queue = self._queues[slot_id]
        try:
            while True:
                job = queue.get()
                if job is None:  # shutdown sentinel
                    return
                try:
                    result = self.runner(job, self._handle_for(slot_id))
                except Exception as exc:  # noqa: BLE001 - isolate per job
                    self._events.put(JobExecution(job, None, exc))
                else:
                    self._events.put(JobExecution(job, result, None))
                finally:
                    # The slot is available again the moment its job settles, so
                    # the next submit can use it even if every other slot is busy.
                    with self._slot_freed:
                        self._free.add(slot_id)
                        self._slot_freed.notify_all()
        except BaseException as exc:  # re-raised on the caller's thread
            self._events.put(_SlotAbort(exc))
            raise

    def collect(self) -> list[JobExecution]:
        if not self._threads:
            return []
        executions: list[JobExecution] = []
        fatal: list[BaseException] = []
        # Block for the first event, then take anything else already settled.
        self._absorb(self._events.get(), executions, fatal)
        while True:
            try:
                self._absorb(self._events.get_nowait(), executions, fatal)
            except Empty:
                break
        if fatal:
            raise fatal[0]
        executions.sort(key=lambda execution: job_order_key(execution.job))
        return executions

    @staticmethod
    def _absorb(
        event: Any,
        executions: list[JobExecution],
        fatal: list[BaseException],
    ) -> None:
        if isinstance(event, _SlotAbort):
            fatal.append(event.exception)
        else:
            executions.append(event)

    def close(self) -> None:
        """Drain running slot work before its GPU resource may be unloaded.

        On a control-flow abort one slot can fail while siblings are still inside
        model calls. Sentinels stop any further work, then an unbounded join waits
        for those already-running calls to leave their backends. The old batch
        executor also joined every slot before returning; keeping that guarantee
        is required because ResourceEpochManager may unload the model immediately
        after this method returns.
        """
        with self._lock:
            threads, self._threads = self._threads, []
        if not threads:
            return
        for queue in self._queues:
            queue.put(None)
        for thread in threads:
            thread.join()

        # A slot that aborted via BaseException can leave its shutdown sentinel
        # unread. Recreate the execution-only queues/state so a reused executor
        # cannot inherit a stale sentinel or a dead worker.
        with self._slot_freed:
            self._queues = [Queue() for _ in range(self.slot_count)]
            self._events = Queue()
            self._free = set(range(self.slot_count))
            self._next_slot = 0
            self._slot_freed.notify_all()

    def execute_batch(self, jobs: Sequence[Any]) -> Mapping[str, JobExecution]:
        results: dict[str, JobExecution] = {}
        remaining = sorted(jobs, key=job_order_key)
        outstanding = 0
        try:
            # Keep at most slot_count in flight and refill as slots come back, so
            # a job is never queued behind a slot that is still working.
            while remaining or outstanding:
                while remaining and outstanding < self.slot_count:
                    self.submit(remaining.pop(0))
                    outstanding += 1
                if not outstanding:
                    break
                for execution in self.collect():
                    results[execution.job.job_id()] = execution
                    outstanding -= 1
            return results
        finally:
            # A slot abort still has to drain siblings before GPU teardown.
            self.close()


# ---------------------------------------------------------------------------
# Resource epoch manager
# ---------------------------------------------------------------------------


class ResourceEpochManager:
    """Own at most one heavy GPU resource at a time for the scheduler.

    The scheduler decides *when* to enter and exit; the manager owns loading,
    unloading and the invariant that Qwen, Boogu and SAM are never resident
    together.
    """

    def __init__(
        self,
        factories: Mapping[
            str, Callable[[], tuple[EpochResource, ResourceJobExecutor]]
        ],
    ) -> None:
        unknown = set(factories) - {RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM}
        if unknown:
            raise ValueError(f"unknown resource factories: {sorted(unknown)}")
        self._factories = dict(factories)
        self.timeline: list[tuple[str, str]] = []
        self.max_open_observed = 0
        self._open: str | None = None
        self._resource: EpochResource | None = None
        self._executor: BatchJobExecutor | None = None
        #: Aggregated per-resource lifecycle counters. Unloading a resource
        #: must not discard its startup/shutdown/load measurements, and a
        #: resource entered twice (qwen -> boogu -> qwen) accumulates both.
        self._resource_counters: dict[str, dict[str, Any]] = {}

    @property
    def open_resource(self) -> str | None:
        return self._open

    @property
    def open_count(self) -> int:
        return 1 if self._open is not None else 0

    def enter(self, resource: str) -> BatchJobExecutor:
        if self._open == resource and self._executor is not None:
            return self._executor
        self.exit_current()
        factory = self._factories.get(resource)
        if factory is None:
            raise EpochResourceError(f"no resource factory for {resource!r}")
        # exit_current() raises on a failed unload, so a broken resource can
        # never be left half-unloaded while the next one loads.
        built_resource, executor = factory()
        built_resource.start()
        self._resource = built_resource
        self._executor = executor
        self._open = resource
        self.timeline.append(("start", resource))
        self.max_open_observed = max(self.max_open_observed, 1)
        return executor

    def exit_current(self) -> None:
        if self._open is None:
            return
        resource = self._open
        self._open = None
        executor = self._executor
        owned_resource = self._resource
        executor_error: BaseException | None = None
        resource_error: BaseException | None = None
        try:
            close = getattr(executor, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:  # noqa: BLE001 - cleanup must unload the GPU model
                    executor_error = exc
            if owned_resource is not None:
                try:
                    owned_resource.stop()
                except BaseException as exc:  # noqa: BLE001 - cleanup must finish
                    resource_error = exc
                else:
                    # Snapshot after the stop so shutdown time and stop_count are
                    # included; unloading must not discard the measurements.
                    self._accumulate(resource, owned_resource.counters())
        finally:
            self.timeline.append(("stop", resource))
            self._resource = None
            self._executor = None
        if resource_error is not None:
            raise resource_error
        if executor_error is not None:
            raise executor_error

    _ACCUMULATED = (
        "start_count",
        "stop_count",
        "startup_wall_seconds",
        "shutdown_wall_seconds",
        "service_seconds",
    )

    def _accumulate(self, name: str, snapshot: Mapping[str, Any]) -> None:
        bucket = self._resource_counters.setdefault(
            name,
            {key: 0 for key in self._ACCUMULATED} | {"gpu_slot_job_counts": {}},
        )
        for key in self._ACCUMULATED:
            bucket[key] = bucket.get(key, 0) + snapshot.get(key, 0)
        slots = bucket.setdefault("gpu_slot_job_counts", {})
        for slot, count in (snapshot.get("gpu_slot_job_counts") or {}).items():
            slots[slot] = slots.get(slot, 0) + count

    def close(self) -> None:
        self.exit_current()

    def counters(self) -> dict[str, Any]:
        return {
            "timeline": [f"{action}:{name}" for action, name in self.timeline],
            "max_open_observed": self.max_open_observed,
            "open_resource": self._open,
            "resources": {
                name: dict(sorted(bucket.items()))
                for name, bucket in sorted(self._resource_counters.items())
            },
        }
