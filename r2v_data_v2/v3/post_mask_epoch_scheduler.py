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
from collections.abc import Callable, Mapping, Sequence
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
        pending: dict[str, ModelJob] = {job.job_id(): job for job in seed_jobs}
        resolved: set[str] = set()
        #: Jobs already given one real model attempt in *this* invocation.
        #: Post-Mask has no in-launch automatic retry; restart is the retry.
        attempted: set[str] = set()
        #: Jobs whose CPU finalizer was already attempted in *this* invocation.
        #: A finalizer failure is also retried on restart, never in-launch, so
        #: a successful sibling must not cause a second attempt here. This is
        #: invocation-local execution state: never persisted, never in identity.
        finalize_attempted: set[str] = set()
        current: str | None = None
        round_index = 0
        started = time.perf_counter()

        try:
            while round_index < self.max_rounds:
                ready = self._ready(pending, resolved, attempted, finalize_attempted)
                if not ready:
                    break
                resource = self._pick_resource(ready, current)
                if current is not None and resource != current:
                    self.diagnostics.resource_switches += 1
                current = resource
                executor = self._executor_for(resource)
                counters = self.diagnostics.resource(resource)
                counters["epoch_count"] += 1
                epoch_started = time.perf_counter()
                epochs_progressed = False

                # Fixed point: keep this resource loaded while it has work.
                while round_index < self.max_rounds:
                    batch = sorted(
                        (
                            job
                            for job in self._ready(pending, resolved, attempted, finalize_attempted)
                            if job.resource == resource
                        ),
                        key=job_order_key,
                    )
                    if not batch:
                        break
                    phase_id = f"r{round_index:03d}-{resource}"
                    phase = self.ledger.phase(phase_id)
                    phase.write_plan(batch)
                    self.diagnostics.phase_count += 1
                    before = (len(resolved), len(attempted), len(pending))
                    self._drain_phase(
                        phase_id,
                        phase,
                        batch,
                        executor,
                        counters,
                        pending,
                        resolved,
                        attempted,
                        finalize_attempted,
                    )
                    after = (len(resolved), len(attempted), len(pending))
                    round_index += 1
                    if before == after:
                        break
                    epochs_progressed = True

                counters["epoch_wall_seconds"] += time.perf_counter() - epoch_started
                if not epochs_progressed:
                    break
        finally:
            if self.resource_manager is not None:
                # Close first so the final exit is part of the recorded timeline,
                # unless an outer session owns that lifetime (4b).
                if self.close_resource_manager_on_exit:
                    self.resource_manager.close()
                self.diagnostics.resource_lifecycle = (
                    self.resource_manager.counters()
                )

        self.diagnostics.resume["torn_receipts_repaired"] = (
            self.ledger.torn_receipts_repaired
        )
        unresolved = sorted(set(pending) - resolved)
        return {
            "completed": not unresolved,
            "unresolved_job_ids": unresolved,
            "job_count": len(pending),
            "resolved_job_count": len(resolved),
            "attempted_job_count": len(attempted),
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

    def _drain_phase(
        self,
        phase_id: str,
        phase: Any,
        batch: Sequence[ModelJob],
        executor: BatchJobExecutor,
        counters: dict[str, Any],
        pending: dict[str, ModelJob],
        resolved: set[str],
        attempted: set[str],
        finalize_attempted: set[str],
    ) -> None:
        for start in range(0, len(batch), self.window_size):
            chunk = batch[start : start + self.window_size]
            self.diagnostics.window_count += 1
            to_execute: list[ModelJob] = []

            # Skip durable work first, but never skip its finalizer.
            for job in chunk:
                counters["jobs_planned"] += 1
                state: JobState = self.ledger.classify(job)
                if state.state == STATE_MISMATCH:
                    raise SchedulerError(
                        f"{job.job_id()}: {state.detail or 'receipt mismatch'}"
                    )
                if state.skippable:
                    counters["jobs_skipped_durable"] += 1
                    self.diagnostics.resume["receipts_reused"] += 1
                    if state.state == "rerun_no_receipt":
                        self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                    self._replay_committed(job, pending, resolved, finalize_attempted)
                    continue
                if state.state == "rerun_no_receipt":
                    self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                to_execute.append(job)

            executions: Mapping[str, JobExecution] = (
                executor.execute_batch(to_execute) if to_execute else {}
            )

            # Durable publication and finalization stay on this thread.
            for job in to_execute:
                counters["jobs_executed"] += 1
                attempted.add(job.job_id())
                if job.job_type in self.diagnostics.conditional_attempts:
                    self.diagnostics.conditional_attempts[job.job_type] += 1
                execution = executions.get(job.job_id())
                if execution is None or (
                    execution.exception is None and execution.result is None
                ):
                    counters["jobs_retryable_failed"] += 1
                    continue
                if execution.exception is not None:
                    counters["jobs_retryable_failed"] += 1
                    continue
                result = execution.result
                assert result is not None
                if result.committed:
                    digests: dict[str, str] = {}
                    for name, payload in sorted(result.artifacts.items()):
                        digests[name] = phase.publish_artifact(
                            job.job_id(), name, payload
                        )
                        self.ledger.note_artifact(phase_id, job.job_id())
                    # ``result.json`` is bound by ``result_digest`` alone. Also
                    # listing it as an artifact digest made every resume hash the
                    # same file twice and forced a directory walk for jobs whose
                    # only committed artifact is the result.
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
                    self._finalize(
                        job, result, pending, resolved, finalize_attempted
                    )

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
