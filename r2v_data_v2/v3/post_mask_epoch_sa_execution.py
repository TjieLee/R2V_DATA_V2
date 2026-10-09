"""Bounded, execution-only CPU/SAM/CPU overlap for Subject Attributes.

Inference callbacks must return independent CPU-owned data before returning.
Only persistence produces a completion eligible for a durable receipt.
"""

from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue
from typing import Any

from .post_mask_epoch_jobs import job_order_key
from .post_mask_epoch_resources import (
    EpochResource,
    EpochResourceError,
    WorkerSlotExecutor,
)
from .post_mask_epoch_scheduler import JobExecution


def resolve_sa_sam_workers_per_gpu() -> int:
    """Execution-only opt-in; one keeps the original shared SAM handles."""
    raw = os.environ.get("POST_MASK_SA_SAM_WORKERS_PER_GPU", "").strip() or "1"
    if raw not in {"1", "2", "4"}:
        raise ValueError("POST_MASK_SA_SAM_WORKERS_PER_GPU must be 1, 2, or 4")
    return int(raw)


class SASamModeResource(EpochResource):
    """Swap only SAM residency at a drained RE/SA boundary in opt-in mode.

    The old parent models are closed before the SA children load, avoiding an
    extra SAM copy per GPU. Qwen/Boogu residency is unaffected.
    """

    def __init__(self, shared_resource: Any, sa_factory: Any, *,
                 pipeline_runner: Any, workers_per_gpu: int) -> None:
        super().__init__(shared_resource.name, shared_resource.process_manager,
                         log_root=shared_resource.log_root)
        self.physical_slot_count = shared_resource.slot_count
        self.workers_per_gpu = workers_per_gpu
        self._shared = shared_resource
        self._sa_factory = sa_factory
        self._provider = pipeline_runner
        self._active: Any = None
        self._sa_mode = False

    @property
    def slot_count(self) -> int:
        return self.physical_slot_count * (self.workers_per_gpu if self._sa_mode else 1)

    def _activate(self, sa_mode: bool) -> None:
        if self._active is not None:
            if self._sa_mode == sa_mode:
                return
            self._active.stop()
            self._active = None
        self._sa_mode = sa_mode
        resource = self._sa_factory() if sa_mode else self._shared
        try:
            resource.start()
        except BaseException:
            resource.stop()
            raise
        self._active = resource

    def _start(self) -> None:
        self._activate(self._provider() is not None)

    def select_mode(self, sa_mode: bool) -> None:
        if self.open:
            self._activate(sa_mode)
        else:
            # Inspecting capacity must not load models before resource.start.
            self._sa_mode = sa_mode

    def handle_for_slot(self, slot: int) -> Any:
        if not self.open or self._active is None:
            raise EpochResourceError("SAM resource is not started")
        return self._active.handle_for_slot(slot)

    def _stop(self) -> None:
        if self._active is not None:
            self._active.stop()
            self._active = None

    def counters(self) -> dict[str, Any]:
        active = {} if self._active is None else self._active.counters()
        return {**super().counters(), "sam_workers_per_gpu": self.workers_per_gpu,
                "sam_processes": self.slot_count, "sam_worker_pool": active}


class SASamPipelineExecutor:
    """One inference thread per SAM handle; bounded logical admission.

    Admission remains occupied until collect, including completed results. Thus
    the entire pipeline (not merely any one queue) has a fixed memory bound.
    close drains stages in upstream order, so downstream workers remain alive
    while an upstream worker can still publish to them.
    """

    def __init__(
        self, runner: Any, *, slot_count: int = 8, resource: Any = None,
        admission_capacity: int | None = None, prepare_workers: int = 2,
        persist_workers: int = 2,
    ) -> None:
        capacity = 3 * slot_count if admission_capacity is None else admission_capacity
        if min(slot_count, capacity, prepare_workers, persist_workers) <= 0:
            raise ValueError("slot and worker counts and admission capacity must be positive")
        self.runner = runner
        self.resource = resource
        self.slot_count = slot_count
        self._capacity = capacity
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._admitted = 0
        self._queues = {stage: Queue(maxsize=capacity)
                        for stage in ("prepare", "infer", "persist")}
        self._events: Queue = Queue(maxsize=capacity)
        self._stats: dict[str, int | float] = {"admitted_peak": 0}
        self._active = {stage: 0 for stage in self._queues}
        for stage in self._queues:
            self._stats.update({f"{stage}_seconds": 0.0, f"{stage}_count": 0,
                                f"{stage}_active_peak": 0, f"{stage}_queue_peak": 0})
        self._stats["infer_persist_overlap_count"] = 0
        self._stats["infer_persist_overlap_seconds"] = 0.0
        self._stats["persist_wait_seconds"] = 0.0
        self._activity_updated = time.monotonic()
        for slot in range(slot_count):
            self._stats[f"infer_slot_{slot}_seconds"] = 0.0
        self._stage_threads: dict[str, list[threading.Thread]] = {}
        self._threads: list[threading.Thread] = []
        for stage, count in (("prepare", prepare_workers), ("infer", slot_count),
                             ("persist", persist_workers)):
            threads = [threading.Thread(target=self._run, args=(stage, index),
                                        name=f"sa-sam-{stage}-{index}")
                       for index in range(count)]
            self._stage_threads[stage] = threads
            self._threads.extend(threads)
        for thread in self._threads:
            thread.start()

    def capacity(self) -> int:
        return self._capacity

    def submit(self, job: Any) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SA executor is closed")
            if self._admitted >= self._capacity:
                raise RuntimeError("SA executor admission capacity exceeded")
            self._admitted += 1
            self._stats["admitted_peak"] = max(self._admitted, self._stats["admitted_peak"])
            # No stage can hold capacity items here: this newly admitted job is
            # part of the same global bound, making this put nonblocking.
            self._queues["prepare"].put_nowait((job, None, time.monotonic()))
            self._record_queue("prepare")

    def _record_queue(self, stage: str) -> None:
        key = f"{stage}_queue_peak"
        self._stats[key] = max(self._stats[key], self._queues[stage].qsize())

    def _run(self, stage: str, slot: int) -> None:
        queue = self._queues[stage]
        while True:
            item = queue.get()
            if item is None:
                return
            job, value, enqueued = item
            started = time.monotonic()
            with self._lock:
                if stage == "persist":
                    self._stats["persist_wait_seconds"] += started - enqueued
                self._record_overlap()
                self._active[stage] += 1
                key = f"{stage}_active_peak"
                self._stats[key] = max(self._stats[key], self._active[stage])
                if ((stage == "infer" and self._active["persist"]) or
                        (stage == "persist" and self._active["infer"])):
                    self._stats["infer_persist_overlap_count"] += 1
            try:
                if stage == "prepare":
                    output = self.runner.prepare_sam_job(job)
                    next_stage = "infer"
                elif stage == "infer":
                    handle = (slot if self.resource is None
                              else self.resource.handle_for_slot(slot))
                    output = self.runner.infer_sam_job(job, value, handle)
                    next_stage = "persist"
                else:
                    output = self.runner.persist_sam_job(job, value)
                    next_stage = None
            except BaseException as exc:  # noqa: BLE001 - transported to collect, including interrupts
                self._events.put((job, None, exc))
            else:
                if next_stage is None:
                    self._events.put((job, output, None))
                else:
                    self._queues[next_stage].put((job, output, time.monotonic()))
                    with self._lock:
                        self._record_queue(next_stage)
            finally:
                with self._lock:
                    self._record_overlap()
                    self._active[stage] -= 1
                    elapsed = time.monotonic() - started
                    self._stats[f"{stage}_seconds"] += elapsed
                    self._stats[f"{stage}_count"] += 1
                    if stage == "infer":
                        self._stats[f"infer_slot_{slot}_seconds"] += elapsed

    def _record_overlap(self) -> None:
        now = time.monotonic()
        if self._active["infer"] and self._active["persist"]:
            self._stats["infer_persist_overlap_seconds"] += now - self._activity_updated
        self._activity_updated = now

    def collect(self) -> list[JobExecution]:
        with self._lock:
            if not self._admitted:
                return []
        events = [self._events.get()]
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        with self._lock:
            self._admitted -= len(events)
        for _job, _result, exception in events:
            if exception is not None and not isinstance(exception, Exception):
                raise exception
        executions = [JobExecution(job, result, exception)
                      for job, result, exception in events]
        executions.sort(key=lambda execution: job_order_key(execution.job))
        return executions

    def diagnostics(self) -> dict[str, int | float]:
        """Execution-only cumulative timings and bounded occupancy peaks."""
        with self._lock:
            self._record_overlap()
            return {**self._stats, "admitted": self._admitted,
                    "physical_slots": getattr(self.resource, "physical_slot_count",
                                              self.slot_count),
                    "sam_processes": self.slot_count,
                    "workers_per_gpu": getattr(self.resource, "workers_per_gpu", 1),
                    "capacity": self._capacity}

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
            for stage in ("prepare", "infer", "persist"):
                for _ in self._stage_threads[stage]:
                    self._queues[stage].put(None)
                for thread in self._stage_threads[stage]:
                    thread.join()


class StageAwareSASamExecutor:
    """Select SA overlap only at a drained stage boundary.

    The provider is queried after dispatch binds a stage. Reference Edit keeps
    the original executor and callback. Opt-in mode swaps only the SAM handles.
    """

    def __init__(self, run_job: Any, *, pipeline_runner: Any = None,
                 slot_count: int = 8, resource: Any = None) -> None:
        self._run_job = run_job
        self._provider = pipeline_runner or (lambda: None)
        self._slot_count = slot_count
        self._resource = resource
        self._executor: Any = None
        self._runner: Any = None
        self._outstanding = 0
        self._closed = False

    def _select(self) -> Any:
        if self._closed:
            raise RuntimeError("SA executor is closed")
        runner = self._provider()
        if self._executor is not None and runner is self._runner:
            return self._executor
        if self._outstanding:
            raise RuntimeError("cannot switch SA runner with in-flight jobs")
        if self._executor is not None:
            self._executor.close()
        select_mode = getattr(self._resource, "select_mode", None)
        if select_mode is not None:
            select_mode(runner is not None)
        self._runner = runner
        if runner is None:
            self._executor = WorkerSlotExecutor(self._run_job,
                slot_count=self._slot_count, resource=self._resource)
        else:
            self._executor = SASamPipelineExecutor(runner,
                slot_count=getattr(self._resource, "slot_count", self._slot_count),
                resource=self._resource)
            runner.sam_execution_diagnostics = self._executor.diagnostics
        return self._executor

    def capacity(self) -> int:
        return self._select().capacity()

    def submit(self, job: Any) -> None:
        self._select().submit(job)
        self._outstanding += 1

    def collect(self) -> list[JobExecution]:
        if not self._outstanding:
            return []
        # Do not reselect here: the caller must be able to drain old work even
        # if its dispatch provider has already changed.
        try:
            results = self._executor.collect()
        except BaseException:
            # Fatal control flow may consume a wave without returning it. Drain
            # threads before resource teardown; uncommitted work resumes cold.
            self.close()
            raise
        self._outstanding -= len(results)
        return results

    def diagnostics(self) -> dict[str, int | float]:
        if isinstance(self._executor, SASamPipelineExecutor):
            return self._executor.diagnostics()
        return {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._executor is not None:
                self._executor.close()
        finally:
            self._outstanding = 0
