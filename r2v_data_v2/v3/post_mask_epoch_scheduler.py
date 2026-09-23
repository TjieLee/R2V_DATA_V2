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

    def add(self, jobs: Iterable[ModelJob]) -> None:
        for job in jobs:
            self.pending.setdefault(job.job_id(), job)
            self.seed_job_ids.add(job.job_id())


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
        self.window_size = window_size
        self.diagnostics = diagnostics if diagnostics is not None else EpochDiagnostics()
        self.max_rounds = max_rounds
        # Default: this scheduler owns and closes its resource manager. A shared
        # Removal -> Pair session opts out so the outer session closes once.
        self.close_resource_manager_on_exit = close_resource_manager_on_exit

    # -- planning ---------------------------------------------------------
    def _pick_resource(self, ready: Sequence[ModelJob], current: str | None) -> str:
        if current is not None and any(job.resource == current for job in ready):
            return current
        for name in self.resource_priority:
            if any(job.resource == name for job in ready):
                return name
        return ready[0].resource

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
        run.add(seed_jobs)
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
                run.add(batch)
                self._drain_to_fixed_point(run)
        finally:
            self._close_resource_manager()
        return self._run_outcome(run, started)

    def _drain_to_fixed_point(self, run: _SchedulerRun) -> None:
        """Drain every ready job, taking the next phase id for each drain."""
        while run.round_index < self.max_rounds:
            ready = self._ready(
                run.pending, run.resolved, run.attempted, run.finalize_attempted
            )
            if not ready:
                return
            resource = self._pick_resource(ready, run.current)
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
            try:
                epochs_progressed = self._drain_resource(
                    phase_id,
                    self.ledger.phase(phase_id),
                    resource,
                    executor,
                    counters,
                    run.pending,
                    run.resolved,
                    run.attempted,
                    run.finalize_attempted,
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
        pending: dict[str, ModelJob],
        resolved: set[str],
        attempted: set[str],
        finalize_attempted: set[str],
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
        planned: dict[str, JobState] = {}
        progressed = False
        while True:
            before = (len(resolved), len(attempted), len(pending))
            self._refill(
                resource,
                executor,
                counters,
                phase,
                planned,
                inflight,
                pending,
                resolved,
                attempted,
                finalize_attempted,
            )
            if not inflight:
                # Nothing running and nothing ready: this resource is done. A
                # refill that only replayed durable finalizers still counts as
                # progress, because it can unlock work for the next resource.
                progressed = progressed or (
                    len(resolved),
                    len(attempted),
                    len(pending),
                ) != before
                break
            self.diagnostics.window_count += 1
            completions = executor.collect()
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
                planned,
                inflight,
                pending,
                resolved,
                attempted,
                finalize_attempted,
            )
            if counters["jobs_submitted"] > submitted_before:
                counters["refill_count"] += 1
            for execution in completions:
                self._settle(
                    phase_id,
                    phase,
                    counters,
                    execution,
                    pending,
                    resolved,
                    attempted,
                    finalize_attempted,
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
                    planned,
                    inflight,
                    pending,
                    resolved,
                    attempted,
                    finalize_attempted,
                )
                if counters["jobs_submitted"] > submitted_before:
                    counters["refill_count"] += 1
            after = (len(resolved), len(attempted), len(pending))
            if before != after:
                progressed = True
        return progressed

    def _refill(
        self,
        resource: str,
        executor: ResourceJobExecutor,
        counters: dict[str, Any],
        phase: Any,
        planned: dict[str, JobState],
        inflight: dict[str, ModelJob],
        pending: dict[str, ModelJob],
        resolved: set[str],
        attempted: set[str],
        finalize_attempted: set[str],
    ) -> None:
        """Top up in-flight work for one resource from its ready set.

        A job with a durable receipt is skipped here but still has its finalizer
        replayed, which can unlock further ready jobs, so the ready set is
        recomputed while capacity remains.
        """
        capacity = executor.capacity()
        while True:
            if capacity > 0 and len(inflight) >= capacity:
                return
            ready = sorted(
                (
                    job
                    for job in self._ready(
                        pending, resolved, attempted, finalize_attempted
                    )
                    if job.resource == resource and job.job_id() not in inflight
                ),
                key=job_order_key,
            )
            if not ready:
                return
            counters["peak_ready_jobs"] = max(
                counters["peak_ready_jobs"], len(ready)
            )
            newly_ready = [
                job for job in ready if job.job_id() not in planned
            ]
            if newly_ready:
                states = {
                    job.job_id(): self.ledger.classify(job)
                    for job in newly_ready
                }
                for job in newly_ready:
                    if states[job.job_id()].state == STATE_MISMATCH:
                        raise SchedulerError(
                            f"{job.job_id()}: "
                            f"{states[job.job_id()].detail or 'receipt mismatch'}"
                        )
                # Register the current ready frontier once, before any worker
                # can finish. Later dependency unlocks extend this same plan.
                phase.write_plan(newly_ready)
                planned.update(states)
                counters["jobs_planned"] += len(newly_ready)
            handled = 0
            for job in ready:
                if capacity > 0 and len(inflight) >= capacity:
                    return
                handled += 1
                state = planned[job.job_id()]
                if state.skippable:
                    counters["jobs_skipped_durable"] += 1
                    self.diagnostics.resume["receipts_reused"] += 1
                    if state.state == "rerun_no_receipt":
                        self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                    self._replay_committed(job, pending, resolved, finalize_attempted)
                    continue
                if state.state == "rerun_no_receipt":
                    self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                attempted.add(job.job_id())
                inflight[job.job_id()] = job
                counters["jobs_submitted"] += 1
                counters["peak_inflight"] = max(
                    counters["peak_inflight"], len(inflight)
                )
                executor.submit(job)
            if not handled:
                return

    def _settle(
        self,
        phase_id: str,
        phase: Any,
        counters: dict[str, Any],
        execution: JobExecution,
        pending: dict[str, ModelJob],
        resolved: set[str],
        attempted: set[str],
        finalize_attempted: set[str],
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
            self.ledger.note_commit(phase_id, receipt)
            if result.outcome == OUTCOME_TERMINAL_REJECT:
                counters["jobs_terminal_rejected"] += 1
        else:
            counters["jobs_retryable_failed"] += 1
        if result.committed:
            self._finalize(job, result, pending, resolved, finalize_attempted)

    def _replay_committed(
        self,
        job: ModelJob,
        pending: dict[str, ModelJob],
        resolved: set[str],
        finalize_attempted: set[str],
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
        self._finalize(job, result, pending, resolved, finalize_attempted)

    def _finalize(
        self,
        job: ModelJob,
        result: JobResult,
        pending: dict[str, ModelJob],
        resolved: set[str],
        finalize_attempted: set[str],
    ) -> None:
        # Marked before the call so a raising finalizer still counts as this
        # invocation's single attempt.
        finalize_attempted.add(job.job_id())
        try:
            unlocked = self.finalize(job, result) or ()
        except Exception:  # noqa: BLE001 - finalizer failure must not be silent
            self.diagnostics.resume["finalizer_failures"] += 1
            # Leave the job unresolved: the model call is durable, the
            # finalizer is replayed on the next run instead of being retried
            # inside this invocation.
            return
        for unlocked_job in unlocked:
            pending.setdefault(unlocked_job.job_id(), unlocked_job)
        resolved.add(job.job_id())
