"""Worker-scoped serialized model resources for Post-Mask execution."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar

from r2v_data_v2.v3.config import Sam3Config, V3Config

if TYPE_CHECKING:
    from r2v_data_v2.v3.post_mask_runtime import ExecutionSettings


_T = TypeVar("_T")


@dataclass
class _ServiceOccupancy:
    """Monotonic perf_counter timestamps, never wall-clock/UTC timestamps.

    Owner holds its metrics lock. Callers already serialize actual service.
    """

    first_service_started_at: float | None = None
    last_service_finished_at: float | None = None
    idle_seconds_between_calls: float = 0.0
    longest_idle_gap_seconds: float = 0.0

    def started(self, now: float) -> None:
        if self.first_service_started_at is None:
            self.first_service_started_at = now
        if self.last_service_finished_at is not None:
            gap = max(0.0, now - self.last_service_finished_at)
            self.idle_seconds_between_calls += gap
            self.longest_idle_gap_seconds = max(self.longest_idle_gap_seconds, gap)

    def summary(self, prefix: str, service_seconds: float) -> dict[str, float | None]:
        span = (
            self.last_service_finished_at - self.first_service_started_at
            if self.last_service_finished_at is not None
            and self.first_service_started_at is not None
            else 0.0
        )
        return {
            f"{prefix}_first_service_started_at": self.first_service_started_at,
            f"{prefix}_last_service_finished_at": self.last_service_finished_at,
            f"{prefix}_idle_seconds_between_calls": self.idle_seconds_between_calls,
            f"{prefix}_longest_idle_gap_seconds": self.longest_idle_gap_seconds,
            f"{prefix}_utilization_over_active_span": (
                service_seconds / span if span > 0 else 0.0
            ),
        }


class _FifoLock:
    """Serve one bounded clip request at a time, in arrival order."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._queue: deque[object] = deque()

    def __enter__(self) -> Self:
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: self._queue[0] is ticket)
            except BaseException:
                self._queue.remove(ticket)
                self._condition.notify_all()
                raise
        return self

    def __exit__(self, *exc: object) -> None:
        with self._condition:
            self._queue.popleft()
            self._condition.notify_all()


class _ObservedQwenGate:
    """Observe the existing node-wide gate, never acquire a second slot."""

    def __init__(self, gate: Any, owner: PostMaskWorkerResources) -> None:
        self._gate = gate
        self._owner = owner

    @property
    def maximum(self) -> int:
        return self._gate.maximum

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        self._owner._require_open()
        with self._gate.acquire() as observation:
            self._owner._require_open()
            with self._owner._metrics_lock:
                if self._owner._qwen_inflight == 0:
                    self._owner._qwen_active_started_at = time.perf_counter()
                self._owner._qwen_call_count += 1
                self._owner._qwen_queue_wait_seconds += observation.queue_wait_seconds
                self._owner._qwen_inflight += 1
                self._owner._qwen_max_inflight_observed = max(
                    self._owner._qwen_max_inflight_observed,
                    self._owner._qwen_inflight,
                )
            try:
                yield observation
            finally:
                with self._owner._metrics_lock:
                    self._owner._qwen_inflight -= 1
                    if self._owner._qwen_inflight == 0:
                        self._owner._qwen_active_seconds += (
                            time.perf_counter() - self._owner._qwen_active_started_at
                        )


class PostMaskWorkerResources:
    """Own lazy Boogu/SAM resources and narrow FIFO call locks.

    ``boogu_start_count`` counts successful starts only. Its first successful
    start is the normal worker start; every later successful start also
    increments ``boogu_restart_count``.
    """

    def __init__(
        self,
        config: V3Config,
        execution: ExecutionSettings,
        *,
        runtime_root: Path,
    ) -> None:
        from r2v_data_v2.v3 import config as config_module

        resolved_root = runtime_root.expanduser().resolve(strict=False)
        allowed_root = config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
        if resolved_root == allowed_root or not resolved_root.is_relative_to(
            allowed_root
        ):
            raise ValueError(
                "Post-Mask worker runtime root must stay below allowed writable root"
            )
        if runtime_root.is_symlink():
            raise ValueError("Post-Mask worker runtime root cannot be a symlink")

        self.config = config
        self.execution = execution
        self.runtime_root = resolved_root
        self._boogu_scratch_root = resolved_root / "boogu-scratch"
        self._boogu_log_path = resolved_root / "boogu.log"

        self._boogu_lock = _FifoLock()
        self._sam_lock = _FifoLock()
        self._lifecycle_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._boogu_backend: Any | None = None
        self._sam_backend: Any | None = None
        self._qwen_gate: _ObservedQwenGate | None = None
        self._closed = False
        self._stopping = False
        self._started_at = time.perf_counter()
        self._finished_at: float | None = None
        self._active = {"shards": 0, "clip_pipelines": 0}
        self._max_active = dict(self._active)
        self._sam_model_start_count = 0
        self._qwen_inflight = 0
        self._qwen_max_inflight_observed = 0
        self._qwen_queue_wait_seconds = 0.0
        self._qwen_call_count = 0
        self._qwen_active_seconds = 0.0
        self._qwen_active_started_at = 0.0
        self._parallel_review_metrics: dict[str, int | float] = {
            "reference_complete_parallel_attempts": 0,
            "reference_complete_parallel_both_success": 0,
            "reference_complete_parallel_qwen_seconds": 0.0,
            "reference_complete_parallel_sam_seconds": 0.0,
            "reference_complete_parallel_wall_seconds": 0.0,
        }
        self._boogu_occupancy = _ServiceOccupancy()
        self._sam_occupancy = _ServiceOccupancy()

        self._boogu_start_count = 0
        self._boogu_restart_count = 0
        self._boogu_call_count = 0
        self._boogu_queue_wait_seconds = 0.0
        self._boogu_service_seconds = 0.0
        self._boogu_inflight = 0
        self._boogu_max_inflight_observed = 0

        self._sam_call_count = 0
        self._sam_queue_wait_seconds = 0.0
        self._sam_service_seconds = 0.0
        self._sam_inflight = 0
        self._sam_max_inflight_observed = 0

    def __enter__(self) -> Self:
        with self._lifecycle_lock:
            if self._closed or self._stopping:
                raise RuntimeError("Post-Mask worker resources are closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_open(self) -> None:
        with self._lifecycle_lock:
            if self._closed or self._stopping:
                raise RuntimeError("Post-Mask worker resources are closed")

    @property
    def stopping(self) -> bool:
        with self._lifecycle_lock:
            return self._stopping or self._closed

    @property
    def qwen_gate(self) -> _ObservedQwenGate:
        with self._lifecycle_lock:
            if self._closed or self._stopping:
                raise RuntimeError("Post-Mask worker resources are closed")
            if self._qwen_gate is None:
                self._qwen_gate = _ObservedQwenGate(self.execution.qwen_gate(), self)
            return self._qwen_gate

    @contextmanager
    def activity(self, kind: str) -> Iterator[None]:
        """Measure active orchestration without locking its work."""
        if kind not in self._active:
            raise ValueError(f"Unknown worker activity: {kind}")
        self._require_open()
        with self._metrics_lock:
            self._active[kind] += 1
            self._max_active[kind] = max(self._max_active[kind], self._active[kind])
        try:
            yield
        finally:
            with self._metrics_lock:
                self._active[kind] -= 1

    def sam_backend(self, config: Sam3Config) -> Any:
        """Return the same lazy backend across phases and shards.

        The resource-local subclass only observes successful predictor builds,
        including a backend-owned compile fallback; all SAM policy is inherited.
        Construction is cheap; callers still serialize inference with sam_call.
        """
        with self._sam_lock:
            self._require_open()
            if self._sam_backend is None:
                from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend

                owner = self

                class WorkerSamBackend(Sam3SegmentationBackend):
                    def _build_predictor(self, *, compile_enabled: bool) -> object:
                        predictor = super()._build_predictor(
                            compile_enabled=compile_enabled
                        )
                        with owner._metrics_lock:
                            owner._sam_model_start_count += 1
                        return predictor

                self._sam_backend = WorkerSamBackend(config)
            elif self._sam_backend.config != config:
                raise ValueError(
                    "Worker SAM configuration must be identical across shards"
                )
            return self._sam_backend

    def _backend_for_request(self) -> Any:
        backend = self._boogu_backend
        if backend is not None and backend.started:
            return backend

        # Import at creation time so tests and launchers can replace the
        # process boundary without importing model dependencies eagerly.
        from r2v_data_v2.v3 import config as config_module
        from r2v_data_v2.v3 import reference_edit_boogu

        reference_config = self.config.reference_edit
        worker_config = reference_edit_boogu.BooguWorkerConfig(
            python_executable=reference_config.python_executable,
            code_root=reference_config.code_root,
            model_path=reference_config.model_path,
            model_revision=reference_config.model_revision,
            device="cuda:0",
            cuda_visible_devices=self.execution.boogu_gpu,
            timeout_seconds=min(
                reference_config.timeout_seconds,
                self.execution.runtime.worker_timeout_seconds,
            ),
            temporary_root=self._boogu_scratch_root,
            allowed_server_root=config_module.ALLOWED_WRITABLE_ROOT,
        )
        candidate = reference_edit_boogu.BooguSubprocessBackend(worker_config)
        try:
            candidate.start(stderr_log_path=self._boogu_log_path)
        except BaseException:
            with suppress(Exception):
                candidate.close()
            raise

        with self._metrics_lock:
            if self._boogu_start_count:
                self._boogu_restart_count += 1
            self._boogu_start_count += 1
        self._boogu_backend = candidate
        return candidate

    def edit(self, **kwargs: Any) -> Any:
        """Run one actual Boogu edit while holding only the Boogu call lock."""

        queued_at = time.perf_counter()
        with self._boogu_lock:
            self._require_open()
            acquired_at = time.perf_counter()
            with self._metrics_lock:
                self._boogu_queue_wait_seconds += acquired_at - queued_at

            backend = self._backend_for_request()
            service_started = time.perf_counter()
            with self._metrics_lock:
                self._boogu_occupancy.started(service_started)
                self._boogu_call_count += 1
                self._boogu_inflight += 1
                self._boogu_max_inflight_observed = max(
                    self._boogu_max_inflight_observed,
                    self._boogu_inflight,
                )
            try:
                return backend.edit(**kwargs)
            finally:
                service_finished = time.perf_counter()
                elapsed = service_finished - service_started
                with self._metrics_lock:
                    self._boogu_occupancy.last_service_finished_at = service_finished
                    self._boogu_inflight -= 1
                    self._boogu_service_seconds += elapsed

    def sam_call(
        self,
        function: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one actual SAM call while holding only the SAM call lock."""

        queued_at = time.perf_counter()
        with self._sam_lock:
            self._require_open()
            acquired_at = time.perf_counter()
            with self._metrics_lock:
                self._sam_queue_wait_seconds += acquired_at - queued_at
                self._sam_call_count += 1
                self._sam_inflight += 1
                self._sam_max_inflight_observed = max(
                    self._sam_max_inflight_observed,
                    self._sam_inflight,
                )
            service_started = time.perf_counter()
            with self._metrics_lock:
                self._sam_occupancy.started(service_started)
            try:
                return function(*args, **kwargs)
            finally:
                service_finished = time.perf_counter()
                elapsed = service_finished - service_started
                with self._metrics_lock:
                    self._sam_occupancy.last_service_finished_at = service_finished
                    self._sam_inflight -= 1
                    self._sam_service_seconds += elapsed

    def record_parallel_review(self, metrics: Mapping[str, int | float]) -> None:
        """Aggregate execution-only review measurements, never model payloads."""
        with self._metrics_lock:
            for name in self._parallel_review_metrics:
                self._parallel_review_metrics[name] += metrics.get(name, 0)

    def summary(self) -> dict[str, int | float | str | None]:
        """Snapshot worker measurements; service timestamps use perf_counter.

        Qwen active seconds measure the union of this worker's acquired-gate
        intervals, including client/network overhead, not remote GPU occupancy.
        Boogu service and occupancy exclude its initial backend startup.
        """

        with self._metrics_lock:
            return {
                "scheduler_mode": getattr(self.execution, "scheduler_mode", "wavefront_v2"),
                **self._parallel_review_metrics,
                **self._boogu_occupancy.summary("boogu", self._boogu_service_seconds),
                **self._sam_occupancy.summary("sam", self._sam_service_seconds),
                "clip_inflight": self.execution.clip_inflight,
                "shard_inflight": self.execution.shard_inflight,
                "max_active_shards_observed": self._max_active["shards"],
                "max_active_clip_pipelines_observed": self._max_active[
                    "clip_pipelines"
                ],
                "worker_wall_seconds": (
                    self._finished_at
                    if self._finished_at is not None
                    else time.perf_counter()
                )
                - self._started_at,
                "boogu_start_count": self._boogu_start_count,
                "boogu_restart_count": self._boogu_restart_count,
                "boogu_call_count": self._boogu_call_count,
                "boogu_queue_wait_seconds": self._boogu_queue_wait_seconds,
                "boogu_service_seconds": self._boogu_service_seconds,
                "boogu_max_inflight_observed": self._boogu_max_inflight_observed,
                "sam_call_count": self._sam_call_count,
                "sam_model_start_count": self._sam_model_start_count,
                "sam_queue_wait_seconds": self._sam_queue_wait_seconds,
                "sam_service_seconds": self._sam_service_seconds,
                "sam_max_inflight_observed": self._sam_max_inflight_observed,
                "qwen_max_inflight_configured": (
                    self.execution.runtime.qwen_max_inflight
                ),
                "qwen_max_inflight_observed": self._qwen_max_inflight_observed,
                "qwen_queue_wait_seconds": self._qwen_queue_wait_seconds,
                "qwen_call_count": self._qwen_call_count,
                "qwen_active_seconds": self._qwen_active_seconds + (
                    time.perf_counter() - self._qwen_active_started_at
                    if self._qwen_inflight else 0.0
                ),
                "qwen_active_seconds_scope": "worker_local_gate_interval_union",
            }

    def close(self) -> None:
        """Drain call locks and close owned Boogu and SAM at most once."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            with self._boogu_lock:
                backend = self._boogu_backend
                self._boogu_backend = None
                if backend is not None and backend.started:
                    backend.close()
        finally:
            try:
                with self._sam_lock:
                    backend = self._sam_backend
                    self._sam_backend = None
                    if backend is not None:
                        backend.close()
            finally:
                with self._metrics_lock:
                    self._finished_at = time.perf_counter()

    def stop_accepting(self) -> None:
        """Prevent resource-queued clips starting new calls during interruption."""
        with self._lifecycle_lock:
            self._stopping = True
