"""Fixed-point ready-job scheduler for the Post-Mask resource epoch mode.

The scheduler is allowed to decide only:

    when a job runs
    on which GPU class it runs
    how same-resource jobs are batched and windowed
    when models are loaded/unloaded

It is *never* allowed to decide whether a job exists, what it asks a model, or
how a result is judged. Jobs enter ``pending`` only when the CPU finalizer of an
earlier job creates them, which is what makes conditional second attempts
structurally impossible to speculate.

Execution proceeds by resource epoch:

    RESOURCE ENTER (start/load once)
      -> sorted ready jobs
      -> bounded concurrent windows of model calls
      -> deterministic serial publish + receipt + CPU finalize
      -> finalizers may unlock new same-resource jobs
      -> repeat until this resource reaches a fixed point
    RESOURCE EXIT (stop/unload completely)

Resource ENTER/EXIT is guarded by try/finally, so an executor exception or a
KeyboardInterrupt still unloads the owned resource.

Concurrency is confined to the model calls. Receipt appends, artifact
publication, finalizers and pending-map mutation all happen on the scheduler
thread in deterministic job order, which keeps JSONL append and durable state
free of races.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Protocol

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    RESOURCE_TYPES,
    JobResult,
    ModelJob,
    job_order_key,
)
from r2v_data_v2.v3.post_mask_epoch_state import (
    STATE_MISMATCH,
    GroupLedger,
    JobState,
)

#: Deterministic fallback order when the current resource has no ready work.
DEFAULT_RESOURCE_PRIORITY = (RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM)

#: Conditional second-attempt job types whose counts must be reported.
CONDITIONAL_JOB_TYPES = (
    "background_removal_candidate2",
    "reference_completion_candidate2",
    "attribute_completion_candidate2",
)

#: Default bounded execution window. Execution-only, never part of identity.
DEFAULT_WINDOW_SIZE = 64


@dataclass(frozen=True)
class JobExecution:
    """One model call outcome. An exception must not abort its siblings."""

    job: ModelJob
    result: JobResult | None = None
    exception: BaseException | None = None


class BatchJobExecutor(Protocol):
    """Run a bounded batch of same-resource jobs, possibly concurrently."""

    def execute_batch(
        self, jobs: Sequence[ModelJob]
    ) -> Mapping[str, JobExecution]: ...


class ResourceJobExecutor(Protocol):
    """Run jobs of one resource and hand each completion back as it lands.

    This is the protocol the scheduler actually drives. An executor that submits
    a whole batch and only reports back once every job settled forces every
    dependent job to wait for the slowest sibling, so a resource epoch stalls
    even while the model service is idle. Completion-driven executors instead
    take work up to a capacity and report completions incrementally, so the
    scheduler can publish, commit and finalize one job and immediately refill
    the slot it just freed.
    """

    def capacity(self) -> int:
        """Jobs allowed in flight at once. ``<= 0`` means unbounded."""

    def submit(self, job: ModelJob) -> None:
        """Start one job, without waiting for any other job."""

    def collect(self) -> list[JobExecution]:
        """Block until at least one in-flight job settles, then return them all.

        Returns an empty list when nothing is in flight.
        """

    def close(self) -> None:
        """Release threads or connections this executor owns. Idempotent."""


class BatchExecutorAdapter:
    """Drive a legacy ``execute_batch`` executor through the streaming protocol.

    The adapter has a bounded capacity, so the scheduler hands over at most
    ``window_size`` jobs and then waits for that batch, which is exactly the
    pre-streaming bounded-window behaviour. That is still the right answer for an
    executor with no incremental capability, and it lets a plain callable or test
    fake keep working unchanged while the scheduler only ever speaks one
    protocol.
    """

    def __init__(self, executor: Any, *, window_size: int) -> None:
        self._executor = executor
        self._window_size = max(1, int(window_size))
        self._queued: list[ModelJob] = []

    def capacity(self) -> int:
        return self._window_size

    def submit(self, job: ModelJob) -> None:
        self._queued.append(job)

    def collect(self) -> list[JobExecution]:
        if not self._queued:
            return []
        batch, self._queued = self._queued, []
        executions = self._executor.execute_batch(batch) or {}
        return [
            executions.get(job.job_id()) or JobExecution(job, None, None)
            for job in batch
        ]

    def close(self) -> None:
        self._queued = []


def as_job_executor(executor: Any, *, window_size: int) -> ResourceJobExecutor:
    """View an executor through the streaming protocol.

    A real completion-driven executor is used as-is. Anything else is wrapped
    once: the adapter *is* the bounded-batch behaviour expressed in the streaming
    protocol, so the scheduler never has two execution paths.
    """
    if hasattr(executor, "submit") and hasattr(executor, "collect"):
        return executor
    return BatchExecutorAdapter(executor, window_size=window_size)


@dataclass
class _SchedulerRun:
    """Execution-only state shared by every drain of one scheduler invocation.

    Never persisted, never part of a ModelJob, a receipt or any schema: it is
    only how much of this invocation's work is still pending, done, already
    given a model attempt or already given a finalizer attempt.
    """

    pending: dict[str, ModelJob] = field(default_factory=dict)
    #: Newly observed jobs are kept in resource-local heaps until their phase
    #: has durably planned the whole ready frontier. Neither queue is durable.
    ready_unplanned: dict[str, list[tuple[tuple[str, str, int, str], str]]] = (
        field(default_factory=lambda: {name: [] for name in RESOURCE_TYPES})
    )
    ready_members: set[str] = field(default_factory=set)
    ready_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in RESOURCE_TYPES}
    )
    #: Preserve the old fallback for a custom priority that omits a resource:
    #: the first ready job by job ID chooses that resource.
    ready_by_id: list[tuple[str, str]] = field(default_factory=list)
    classified: dict[str, JobState] = field(default_factory=dict)
    #: Jobs already given one real model attempt in *this* invocation. Post-Mask
    #: has no in-launch automatic retry; restart is the retry.
    attempted: set[str] = field(default_factory=set)
    #: Jobs whose CPU finalizer was already attempted in *this* invocation. A
    #: finalizer failure is also retried on restart, never in-launch, so a
    #: successful sibling must not cause a second attempt here.
    finalize_attempted: set[str] = field(default_factory=set)
    resolved: set[str] = field(default_factory=set)
    #: Jobs handed to this invocation by its caller. Jobs a finalizer unlocks
    #: later are pending work but NOT seeded work, so this stays the count of
    #: what the stage actually seeded.
    seed_job_ids: set[str] = field(default_factory=set)
    current: str | None = None
    round_index: int = 0

    def add(
        self, jobs: Iterable[ModelJob], *, seed: bool, track_by_id: bool
    ) -> list[ModelJob]:
        newly_added: list[ModelJob] = []
        for job in jobs:
            job_id = job.job_id()
            if seed:
                self.seed_job_ids.add(job_id)
            if job_id in self.pending:
                continue
            self.pending[job_id] = job
            heappush(self.ready_unplanned[job.resource], (job_order_key(job), job_id))
            if track_by_id:
                heappush(self.ready_by_id, (job_id, job.resource))
            self.ready_members.add(job_id)
            self.ready_counts[job.resource] += 1
            newly_added.append(job)
        return newly_added


class SchedulerError(RuntimeError):
    """Raised when durable state is unsafe to continue from."""


@dataclass
class EpochDiagnostics:
    """Machine-readable execution counters. No throughput is inferred here."""

    group_count: int = 0
    group_completed: int = 0
    group_incomplete: int = 0
    resource_switches: int = 0
    phase_count: int = 0
    window_count: int = 0
    receipt_appends: int = 0
    receipt_syncs: int = 0
    conditional_attempts: dict[str, int] = field(default_factory=dict)
    resume: dict[str, int] = field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Owned-resource lifecycle counters (start/stop/load times per resource).
    #: Kept out of ``resources`` because that dict holds model-job counters.
    resource_lifecycle: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in CONDITIONAL_JOB_TYPES:
            self.conditional_attempts.setdefault(name, 0)
        for name in (
            "receipts_reused",
            "torn_receipts_repaired",
            "jobs_rerun_after_incomplete_commit",
            "finalizers_replayed",
            "finalizer_failures",
        ):
            self.resume.setdefault(name, 0)
        for resource in RESOURCE_TYPES:
            self.resource(resource)

    def resource(self, name: str) -> dict[str, Any]:
        return self.resources.setdefault(
            name,
            {
                "jobs_planned": 0,
                "jobs_skipped_durable": 0,
                "jobs_executed": 0,
                "jobs_terminal_rejected": 0,
                "jobs_retryable_failed": 0,
                "model_service_seconds": 0.0,
                "queue_wait_seconds": 0.0,
                "epoch_wall_seconds": 0.0,
                "epoch_count": 0,
                "executor_capacity": 0,
                "jobs_per_epoch": [],
                "jobs_submitted": 0,
                "peak_ready_jobs": 0,
                "peak_inflight": 0,
                "refill_count": 0,
                "ready_queue_pushes": 0,
                "ready_queue_pops": 0,
                "ready_queue_peak": 0,
                "classify_calls": 0,
                "plan_write_calls": 0,
                "plan_write_jobs": 0,
                "refill_calls": 0,
                "refill_wall_seconds": 0.0,
                "settle_wall_seconds": 0.0,
                "collect_wait_seconds": 0.0,
                "capacity_starved_refills": 0,
                "gpu_slot_job_counts": {},
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "group_completed": self.group_completed,
            "group_incomplete": self.group_incomplete,
            "phase_count": self.phase_count,
            "window_count": self.window_count,
            "receipt_appends": self.receipt_appends,
            "receipt_syncs": self.receipt_syncs,
            "resource_switches": self.resource_switches,
            "conditional_attempts": dict(sorted(self.conditional_attempts.items())),
            "resume": dict(sorted(self.resume.items())),
            "resources": {k: self.resources[k] for k in sorted(self.resources)},
            "resource_lifecycle": self.resource_lifecycle,
        }


class ResourceEpochScheduler:
    """Drain one group's ready jobs resource-by-resource, durably."""

    def __init__(
        self,
        *,
        ledger: GroupLedger,
        finalize: Callable[[ModelJob, JobResult], Sequence[ModelJob]],
        executors: Mapping[str, BatchJobExecutor] | None = None,
        resource_manager: Any | None = None,
        resource_priority: Sequence[str] = DEFAULT_RESOURCE_PRIORITY,
        window_size: int = DEFAULT_WINDOW_SIZE,
        diagnostics: EpochDiagnostics | None = None,
        max_rounds: int = 100_000,
        close_resource_manager_on_exit: bool = True,
    ) -> None:
        if executors is None and resource_manager is None:
            raise ValueError("scheduler needs executors or a resource manager")
        if executors is not None:
            unknown = set(executors) - set(RESOURCE_TYPES)
            if unknown:
                raise ValueError(f"unknown resource executors: {sorted(unknown)}")
        if any(name not in RESOURCE_TYPES for name in resource_priority):
            raise ValueError("resource_priority contains an unknown resource")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.ledger = ledger
        self.finalize = finalize
        self.executors = dict(executors or {})
        self.resource_manager = resource_manager
        self.resource_priority = tuple(resource_priority)
        self._needs_ready_id_fallback = bool(
            set(RESOURCE_TYPES) - set(self.resource_priority)
        )
        self.window_size = window_size
        self.diagnostics = diagnostics if diagnostics is not None else EpochDiagnostics()
        self.max_rounds = max_rounds
        # Default: this scheduler owns and closes its resource manager. A shared
        # Removal -> Pair session opts out so the outer session closes once.
        self.close_resource_manager_on_exit = close_resource_manager_on_exit

    # -- planning ---------------------------------------------------------
    def _pick_resource(self, run: _SchedulerRun) -> str:
        if run.current is not None and run.ready_counts[run.current]:
            return run.current
        for name in self.resource_priority:
            if run.ready_counts[name]:
                return name
        while run.ready_by_id:
            job_id, resource = run.ready_by_id[0]
            if job_id in run.ready_members:
                return resource
            heappop(run.ready_by_id)
        raise SchedulerError("ready resource index is empty")

    def _add_jobs(
        self, run: _SchedulerRun, jobs: Iterable[ModelJob], *, seed: bool
    ) -> None:
        for job in run.add(
            jobs, seed=seed, track_by_id=self._needs_ready_id_fallback
        ):
            counters = self.diagnostics.resource(job.resource)
            counters["ready_queue_pushes"] += 1
            counters["ready_queue_peak"] = max(
                counters["ready_queue_peak"], run.ready_counts[job.resource]
            )
            counters["peak_ready_jobs"] = max(
                counters["peak_ready_jobs"], run.ready_counts[job.resource]
            )

    def _executor_for(self, resource: str) -> BatchJobExecutor:
        if self.resource_manager is not None:
            return self.resource_manager.enter(resource)
        executor = self.executors.get(resource)
        if executor is None:
            raise SchedulerError(
                f"no executor configured for resource {resource!r}"
            )
        return executor

    # -- execution --------------------------------------------------------
    def run(self, seed_jobs: Sequence[ModelJob]) -> dict[str, Any]:
        started = time.perf_counter()
        run = _SchedulerRun()
        self._add_jobs(run, seed_jobs, seed=True)
        try:
            self._drain_to_fixed_point(run)
        finally:
            self._close_resource_manager()
        return self._run_outcome(run, started)

    def run_batches(self, batches: Iterable[Sequence[ModelJob]]) -> dict[str, Any]:
        """Drain consecutive job batches through ONE resource session.

        Each batch is drained to its own fixed point before the next one is
        requested, so a producer can hold a bounded amount of prepared work and
        release it as soon as its batch is terminal - which is what makes a
        10k-clip Pair stage feasible without unbounded memory. The managed
        resource is entered through the normal ``_executor_for`` path and closed
        exactly once, at the end: batching must never turn one stage into one
        model-server start per batch. Every drain takes the next phase id from a
        single monotonically increasing counter, so no two drains can collide on
        a phase plan, and the phase index is execution-only placement - it is
        never part of ModelJob identity, a semantic input digest or a receipt.

        Jobs a batch could not resolve stay unresolved in the aggregate outcome,
        exactly as they would after a single ``run`` of the same jobs.
        """
        started = time.perf_counter()
        run = _SchedulerRun()
        try:
            for batch in batches:
                self._add_jobs(run, batch, seed=True)
                self._drain_to_fixed_point(run)
        finally:
            self._close_resource_manager()
        return self._run_outcome(run, started)

    def _drain_to_fixed_point(self, run: _SchedulerRun) -> None:
        """Drain every ready job, taking the next phase id for each drain."""
        while run.round_index < self.max_rounds:
            if not run.ready_members:
                return
            resource = self._pick_resource(run)
            if run.current is not None and resource != run.current:
                self.diagnostics.resource_switches += 1
            run.current = resource
            executor = as_job_executor(
                self._executor_for(resource), window_size=self.window_size
            )
            counters = self.diagnostics.resource(resource)
            counters["epoch_count"] += 1
            counters["executor_capacity"] = executor.capacity()
            self.diagnostics.phase_count += 1
            epoch_started = time.perf_counter()
            epoch_jobs_before = counters["jobs_executed"]
            phase_id = f"r{run.round_index:03d}-{resource}"
            phase = self.ledger.phase(phase_id)
            appends_before = self.diagnostics.receipt_appends
            try:
                epochs_progressed = self._drain_resource(
                    phase_id,
                    phase,
                    resource,
                    executor,
                    counters,
                    run,
                )
                if phase.sync_receipts():
                    self.diagnostics.receipt_syncs += 1
                elif self.diagnostics.receipt_appends > appends_before:
                    raise SchedulerError(
                        f"phase {phase_id!r} committed jobs but has no receipts"
                    )
            finally:
                executor.close()
            counters["epoch_wall_seconds"] += time.perf_counter() - epoch_started
            counters["jobs_per_epoch"].append(
                counters["jobs_executed"] - epoch_jobs_before
            )
            run.round_index += 1
            if not epochs_progressed:
                return

    def _close_resource_manager(self) -> None:
        if self.resource_manager is None:
            return
        # Close first so the final exit is part of the recorded timeline, unless
        # an outer session owns that lifetime (4b).
        if self.close_resource_manager_on_exit:
            self.resource_manager.close()
        self.diagnostics.resource_lifecycle = self.resource_manager.counters()

    def _run_outcome(self, run: _SchedulerRun, started: float) -> dict[str, Any]:
        self.diagnostics.resume["torn_receipts_repaired"] = (
            self.ledger.torn_receipts_repaired
        )
        unresolved = sorted(set(run.pending) - run.resolved)
        return {
            "completed": not unresolved,
            "unresolved_job_ids": unresolved,
            #: Every job the scheduler observed, seeded or unlocked.
            "job_count": len(run.pending),
            #: Only the jobs the caller seeded; additive, so ``job_count`` keeps
            #: its existing meaning for every current caller.
            "seed_job_count": len(run.seed_job_ids),
            "resolved_job_count": len(run.resolved),
            "attempted_job_count": len(run.attempted),
            "resource_lifecycle": self.diagnostics.resource_lifecycle,
            "wall_seconds": time.perf_counter() - started,
            "diagnostics": self.diagnostics.summary(),
        }

    def _ready(
        self,
        pending: Mapping[str, ModelJob],
        resolved: set[str],
        attempted: set[str],
        finalize_attempted: set[str],
    ) -> list[ModelJob]:
        return [
            job
            for key, job in sorted(pending.items())
            if key not in resolved
            and key not in attempted
            and key not in finalize_attempted
        ]

    def _drain_resource(
        self,
        phase_id: str,
        phase: Any,
        resource: str,
        executor: ResourceJobExecutor,
        counters: dict[str, Any],
        run: _SchedulerRun,
    ) -> bool:
        """Run one resource to its completion-driven fixed point.

        The resource stays loaded and keeps receiving work for as long as it has
        either an in-flight job or a ready job. Each time a job settles the
        scheduler settles its durable state on this thread - publish artifact,
        publish result, commit receipt, finalize - and only then refills, so a
        job unlocked by that finalizer can enter the very slot that just freed
        up instead of waiting for the slowest sibling in the batch.

        Jobs of any *other* resource unlocked here stay in ``pending``: they do
        not switch the resource. The resource is exited only once it is genuinely
        quiescent, which is what keeps a Qwen -> Boogu -> Qwen pattern from
        thrashing the loaded model.
        """
        inflight: dict[str, ModelJob] = {}
        planned_ready: list[tuple[tuple[str, str, int, str], str]] = []
        progressed = False
        while True:
            before = (len(run.resolved), len(run.attempted), len(run.pending))
            self._refill(
                resource,
                executor,
                counters,
                phase,
                planned_ready,
                inflight,
                run,
            )
            if not inflight:
                # Nothing running and nothing ready: this resource is done. A
                # refill that only replayed durable finalizers still counts as
                # progress, because it can unlock work for the next resource.
                progressed = progressed or (
                    len(run.resolved),
                    len(run.attempted),
                    len(run.pending),
                ) != before
                break
            self.diagnostics.window_count += 1
            collecting_started = time.perf_counter()
            try:
                completions = executor.collect()
            finally:
                counters["collect_wait_seconds"] += (
                    time.perf_counter() - collecting_started
                )
            if not completions and inflight:
                # ``collect`` promises to block until something settles. Anything
                # else would spin here forever on work that can never finish.
                raise SchedulerError(
                    f"{resource} executor returned no completion while "
                    f"{len(inflight)} job(s) are still in flight"
                )
            # Free every finished slot *before* doing any CPU work. Those jobs
            # have already handed their slots back to the executor, so anything
            # that was ready before this collect can be submitted now and the
            # model service starts on it while this thread is still settling.
            for execution in completions:
                inflight.pop(execution.job.job_id(), None)
            submitted_before = counters["jobs_submitted"]
            self._refill(
                resource,
                executor,
                counters,
                phase,
                planned_ready,
                inflight,
                run,
            )
            if counters["jobs_submitted"] > submitted_before:
                counters["refill_count"] += 1
            for execution in completions:
                settling_started = time.perf_counter()
                try:
                    self._settle(phase_id, phase, counters, execution, run)
                finally:
                    counters["settle_wall_seconds"] += (
                        time.perf_counter() - settling_started
                    )
                # A finalizer is CPU work and can be slow. Anything it unlocked
                # for this resource is submitted before the next sibling is
                # settled, so one slow finalizer cannot idle every free slot.
                submitted_before = counters["jobs_submitted"]
                self._refill(
                    resource,
                    executor,
                    counters,
                    phase,
                    planned_ready,
                    inflight,
                    run,
                )
                if counters["jobs_submitted"] > submitted_before:
                    counters["refill_count"] += 1
            after = (len(run.resolved), len(run.attempted), len(run.pending))
            if before != after:
                progressed = True
        return progressed

    def _refill(
        self,
        resource: str,
        executor: ResourceJobExecutor,
        counters: dict[str, Any],
        phase: Any,
        planned_ready: list[tuple[tuple[str, str, int, str], str]],
        inflight: dict[str, ModelJob],
        run: _SchedulerRun,
    ) -> None:
        """Top up in-flight work without rescanning the pending map.

        Newly unlocked jobs join the planned heap before this call takes its
        first job. A durable replay can unlock more jobs while the current
        ready wave is being consumed; as before, finish that snapshotted wave
        before considering those unlocks.
        """
        started = time.perf_counter()
        counters["refill_calls"] += 1
        try:
            capacity = executor.capacity()
            if capacity > 0 and len(inflight) >= capacity:
                return
            self._plan_frontier(resource, phase, counters, planned_ready, run)
            while True:
                if capacity > 0 and len(inflight) >= capacity:
                    return
                if not planned_ready:
                    self._plan_frontier(resource, phase, counters, planned_ready, run)
                if not planned_ready:
                    if inflight:
                        counters["capacity_starved_refills"] += 1
                    return
                _, job_id = heappop(planned_ready)
                run.ready_members.remove(job_id)
                run.ready_counts[resource] -= 1
                counters["ready_queue_pops"] += 1
                job = run.pending[job_id]
                state = run.classified.pop(job_id)
                if state.skippable:
                    counters["jobs_skipped_durable"] += 1
                    self.diagnostics.resume["receipts_reused"] += 1
                    if state.state == "rerun_no_receipt":
                        self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                    self._replay_committed(job, run)
                    continue
                if state.state == "rerun_no_receipt":
                    self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                run.attempted.add(job_id)
                inflight[job_id] = job
                counters["jobs_submitted"] += 1
                counters["peak_inflight"] = max(
                    counters["peak_inflight"], len(inflight)
                )
                executor.submit(job)
        finally:
            counters["refill_wall_seconds"] += time.perf_counter() - started

    def _plan_frontier(
        self,
        resource: str,
        phase: Any,
        counters: dict[str, Any],
        planned_ready: list[tuple[tuple[str, str, int, str], str]],
        run: _SchedulerRun,
    ) -> None:
        """Plan one entire newly ready wave before submitting any of it."""
        unplanned = run.ready_unplanned[resource]
        if not unplanned:
            return
        frontier = [run.pending[heappop(unplanned)[1]] for _ in range(len(unplanned))]
        states: dict[str, JobState] = {}
        for job in frontier:
            job_id = job.job_id()
            state = run.classified.get(job_id)
            if state is None:
                state = self.ledger.classify(job)
                counters["classify_calls"] += 1
            if state.state == STATE_MISMATCH:
                raise SchedulerError(
                    f"{job_id}: {state.detail or 'receipt mismatch'}"
                )
            states[job_id] = state
        phase.write_plan(frontier)
        counters["plan_write_calls"] += 1
        counters["plan_write_jobs"] += len(frontier)
        counters["jobs_planned"] += len(frontier)
        run.classified.update(states)
        for job in frontier:
            heappush(planned_ready, (job_order_key(job), job.job_id()))

    def _settle(
        self,
        phase_id: str,
        phase: Any,
        counters: dict[str, Any],
        execution: JobExecution,
        run: _SchedulerRun,
    ) -> None:
        """Publish, commit and finalize one settled job on the scheduler thread.

        Durable state is never touched by the worker: the executor only reports
        the outcome, and every artifact, receipt, pending-map mutation and
        finalizer call happens here.
        """
        job = execution.job
        counters["jobs_executed"] += 1
        if job.job_type in self.diagnostics.conditional_attempts:
            self.diagnostics.conditional_attempts[job.job_type] += 1
        if execution is None or (
            execution.exception is None and execution.result is None
        ):
            counters["jobs_retryable_failed"] += 1
            return
        if execution.exception is not None:
            counters["jobs_retryable_failed"] += 1
            return
        result = execution.result
        assert result is not None
        if result.committed:
            digests: dict[str, str] = {}
            for name, payload in sorted(result.artifacts.items()):
                digests[name] = phase.publish_artifact(job.job_id(), name, payload)
                self.ledger.note_artifact(phase_id, job.job_id())
            # ``result.json`` is bound by ``result_digest`` alone. Also listing
            # it as an artifact digest made every resume hash the same file twice
            # and forced a directory walk for jobs whose only committed artifact
            # is the result.
            result_digest = phase.publish_result(job, result)
            receipt = phase.commit(
                job,
                outcome=result.outcome,
                artifact_digests=digests,
                result_digest=result_digest,
                external_artifacts=result.external_artifacts,
            )
            self.diagnostics.receipt_appends += 1
            self.ledger.note_commit(phase_id, receipt)
            if result.outcome == OUTCOME_TERMINAL_REJECT:
                counters["jobs_terminal_rejected"] += 1
        else:
            counters["jobs_retryable_failed"] += 1
        if result.committed:
            self._finalize(job, result, run)

    def _replay_committed(
        self,
        job: ModelJob,
        run: _SchedulerRun,
    ) -> None:
        """Re-run the CPU finalizer for an already-committed job.

        A downstream job exists only because some finalizer ran, so a crash
        between receipt and finalize would otherwise strand every dependent job.
        The model call is skipped; the finalizer is not.
        """
        result = self.ledger.load_committed_result(job)
        if result is None:
            raise SchedulerError(
                f"{job.job_id()}: committed job has no replayable durable result"
            )
        self.diagnostics.resume["finalizers_replayed"] += 1
        self._finalize(job, result, run)

    def _finalize(
        self,
        job: ModelJob,
        result: JobResult,
        run: _SchedulerRun,
    ) -> None:
        # Marked before the call so a raising finalizer still counts as this
        # invocation's single attempt.
        run.finalize_attempted.add(job.job_id())
        try:
            unlocked = self.finalize(job, result) or ()
        except Exception:  # noqa: BLE001 - finalizer failure must not be silent
            self.diagnostics.resume["finalizer_failures"] += 1
            # Leave the job unresolved: the model call is durable, the
            # finalizer is replayed on the next run instead of being retried
            # inside this invocation.
            return
        self._add_jobs(run, unlocked, seed=False)
        run.resolved.add(job.job_id())
