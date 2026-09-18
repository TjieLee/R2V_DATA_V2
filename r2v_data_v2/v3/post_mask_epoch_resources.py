"""Owned Qwen / Boogu / SAM resource lifecycles for resource-epoch execution.

Every resource here obeys one rule: it may only ever signal a PID or process
group that *this* launcher created. A resource epoch switches models on all
eight GPUs, so an over-broad ``pkill`` would take down unrelated jobs sharing
the node. Nothing here reimplements a model call, a prompt, a threshold or an
accept/reject rule; each epoch only hands the semantic layer a live handle.

Three epochs are supported:

    Qwen   one locally managed vLLM server, TP1 x DP8 across GPU0-7
    Boogu  eight persistent Boogu worker processes, one per GPU
    SAM    eight persistent SAM3 workers, one per GPU

Each epoch loads once and shuts down once. ``start_count`` / ``stop_count`` are
exposed so the launcher can assert healthy single-load behaviour.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import JobResult


class EpochResourceError(RuntimeError):
    """A managed resource could not be started or stopped safely."""


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
    """Spawn each resource in its own process group and stop only that group."""

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
        try:
            pgid = os.getpgid(child.pid)
        except OSError:  # pragma: no cover - race with immediate exit
            pgid = child.pid
        return OwnedProcess(pid=child.pid, pgid=pgid, argv=tuple(argv), log_path=log_path)

    def alive(self, process: OwnedProcess) -> bool:
        try:
            os.kill(process.pid, 0)
        except OSError:
            return False
        return True

    def stop(self, process: OwnedProcess, *, grace_seconds: float) -> bool:
        import signal

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pgid, sig)
            except OSError:
                return not self.alive(process)
            deadline = time.monotonic() + grace_seconds
            while time.monotonic() < deadline:
                if not self.alive(process):
                    return True
                time.sleep(0.2)
        return not self.alive(process)


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

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def served_model_id(self) -> str:
        return Path(self.model_path).name

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
    request = urllib.request.Request(
        f"{base_url}/models".replace("/v1/v1/", "/v1/"), method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


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
        while time.monotonic() < deadline:
            if self.process is not None and not self.process_manager.alive(self.process):
                raise EpochResourceError(
                    "managed Qwen process exited before becoming healthy"
                ) from last_error
            try:
                payload = self.health_probe(self.config.base_url)
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
                time.sleep(1.0)
                continue
            served = (payload.get("data") or [{}])[0].get("id")
            if served == self.config.served_model_id:
                return
            last_error = EpochResourceError(
                f"served model {served!r} != expected "
                f"{self.config.served_model_id!r}"
            )
            time.sleep(1.0)
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


def deterministic_slot(job: ModelJob, slot_count: int) -> int:
    """Static round-robin slot for a job; no stealing, no dynamic balancing."""
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
        for slot, gpu_id in enumerate(self.pool.gpu_ids):
            self.workers.append(self.worker_factory(slot, gpu_id))

    def _stop(self) -> None:
        for worker in self.workers:
            close = getattr(worker, "close", None)
            if callable(close):
                close()
        self.workers = []

    def handle_for(self, job: ModelJob) -> Any:
        if not self._open:
            raise EpochResourceError(f"{self.name} epoch is not started")
        slot = deterministic_slot(job, self.slot_count)
        self.slot_job_counts[slot] = self.slot_job_counts.get(slot, 0) + 1
        return self.workers[slot]

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
    """Build one persistent Boogu subprocess pinned to a single GPU."""

    def factory(slot: int, gpu_id: int) -> Any:
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
            timeout_seconds=min(config.timeout_seconds, pool.timeout_seconds),
            temporary_root=Path(temporary_root) / f"slot-{slot}",
            allowed_server_root=allowed_server_root,
        )
        backend = BooguSubprocessBackend(worker_config)
        backend.start(stderr_log_path=Path(temporary_root) / f"boogu-slot-{slot}.log")
        return backend

    return factory


def sam_worker_factory(*, config: Any, pool: WorkerPoolConfig) -> Callable[[int, int], Any]:
    """Build one persistent SAM3 worker bound to ``cuda:<slot>``.

    All eight GPUs are visible inside the epoch, so the logical device selects
    the physical GPU. SAM prompts, mask semantics, thresholds and review
    behaviour are inherited unchanged from the frozen backend.
    """

    def factory(slot: int, gpu_id: int) -> Any:
        from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend

        from dataclasses import replace

        return Sam3SegmentationBackend(replace(config, device=f"cuda:{slot}"))

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
) -> WorkerEpochResource:
    return WorkerEpochResource(
        RESOURCE_SAM,
        pool,
        process_manager=process_manager,
        worker_factory=sam_worker_factory(config=config, pool=pool),
        log_root=log_root,
    )


# ---------------------------------------------------------------------------
# Scheduler bridge
# ---------------------------------------------------------------------------


class HandleExecutor:
    """Hand the semantic runner a live resource handle, nothing more.

    The runner owns prompts, inputs, seeds and accept/reject policy. This class
    exists so the scheduler never learns what a model call means.
    """

    def __init__(
        self,
        resource: EpochResource,
        runner: Callable[[ModelJob, Any], JobResult],
    ) -> None:
        self.resource = resource
        self.runner = runner

    def _handle(self, job: ModelJob) -> Any:
        if isinstance(self.resource, WorkerEpochResource):
            return self.resource.handle_for(job)
        return self.resource

    def execute(self, job: ModelJob) -> JobResult:
        return self.runner(job, self._handle(job))
