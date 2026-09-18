"""Fixed-point ready-job scheduler for the Post-Mask resource epoch mode.

The scheduler is allowed to decide only:

    when a job runs
    on which GPU class it runs
    how same-resource jobs are batched
    when models are loaded/unloaded

It is *never* allowed to decide whether a job exists, what it asks a model, or
how a result is judged. Jobs enter ``pending`` only when the CPU finalizer of an
earlier job creates them, which is what makes conditional second attempts
structurally impossible to speculate: a candidate-2 job cannot be in the ready
set before the policy that unlocks it has run.

Execution proceeds by resource epoch. Within one epoch the scheduler drains the
ready set of that resource to a fixed point before switching models:

    load Qwen -> run every Qwen-ready job -> CPU finalize -> run jobs those
    finalizers unlocked -> repeat until no Qwen-ready work remains -> unload

Resource choice is deterministic and deliberately simple: keep the current
resource while it still has ready work, otherwise fall back to a fixed priority
order. That avoids thrashing without a cost model or dynamic prediction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    RESOURCE_TYPES,
    ModelJob,
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


class JobExecutor(Protocol):
    """One model resource. Owns its own lifecycle; the scheduler only calls it."""

    def execute(self, job: ModelJob) -> JobResult:  # noqa: D102 - protocol
        ...


@dataclass(frozen=True)
class JobResult:
    """Outcome of one model call plus the artifacts to publish durably."""

    outcome: str
    artifacts: Mapping[str, bytes] = field(default_factory=dict)
    detail: str = ""

    @property
    def committed(self) -> bool:
        return self.outcome in (OUTCOME_COMPLETED, OUTCOME_TERMINAL_REJECT)


@dataclass
class EpochDiagnostics:
    """Machine-readable execution counters. No throughput is inferred here."""

    group_count: int = 0
    group_completed: int = 0
    group_incomplete: int = 0
    resource_switches: int = 0
    phase_count: int = 0
    conditional_attempts: dict[str, int] = field(default_factory=dict)
    resume: dict[str, int] = field(default_factory=dict)
    resources: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in CONDITIONAL_JOB_TYPES:
            self.conditional_attempts.setdefault(name, 0)
        for name in ("receipts_reused", "torn_receipts_repaired", "jobs_rerun_after_incomplete_commit"):
            self.resume.setdefault(name, 0)
        for resource in RESOURCE_TYPES:
            self.resources.setdefault(
                resource,
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
            "resource_switches": self.resource_switches,
            "conditional_attempts": dict(sorted(self.conditional_attempts.items())),
            "resume": dict(sorted(self.resume.items())),
            "resources": {k: self.resources[k] for k in sorted(self.resources)},
        }


class SchedulerError(RuntimeError):
    """Raised when durable state is unsafe to continue from."""


def _order_key(job: ModelJob) -> tuple[str, str, int, str]:
    return (job.clip_uid, job.job_type, job.attempt_index, job.job_id())


class ResourceEpochScheduler:
    """Drain one group's ready jobs resource-by-resource, durably."""

    def __init__(
        self,
        *,
        ledger: GroupLedger,
        executors: Mapping[str, JobExecutor],
        finalize: Callable[[ModelJob, JobResult], Sequence[ModelJob]],
        resource_priority: Sequence[str] = DEFAULT_RESOURCE_PRIORITY,
        diagnostics: EpochDiagnostics | None = None,
        max_rounds: int = 10_000,
    ) -> None:
        unknown = set(executors) - set(RESOURCE_TYPES)
        if unknown:
            raise ValueError(f"unknown resource executors: {sorted(unknown)}")
        if any(name not in RESOURCE_TYPES for name in resource_priority):
            raise ValueError("resource_priority contains an unknown resource")
        self.ledger = ledger
        self.executors = dict(executors)
        self.finalize = finalize
        self.resource_priority = tuple(resource_priority)
        self.diagnostics = diagnostics if diagnostics is not None else EpochDiagnostics()
        self.max_rounds = max_rounds

    # -- planning ---------------------------------------------------------
    def _pick_resource(self, ready: Sequence[ModelJob], current: str | None) -> str:
        if current is not None and any(job.resource == current for job in ready):
            return current
        for name in self.resource_priority:
            if any(job.resource == name for job in ready):
                return name
        return ready[0].resource

    # -- execution --------------------------------------------------------
    def run(self, seed_jobs: Sequence[ModelJob]) -> dict[str, Any]:
        pending: dict[str, ModelJob] = {job.job_id(): job for job in seed_jobs}
        resolved: set[str] = set()
        current: str | None = None
        round_index = 0
        started = time.perf_counter()

        while round_index < self.max_rounds:
            ready = [
                job for key, job in sorted(pending.items()) if key not in resolved
            ]
            if not ready:
                break
            resource = self._pick_resource(ready, current)
            if current is not None and resource != current:
                self.diagnostics.resource_switches += 1
            current = resource
            batch = sorted(
                (job for job in ready if job.resource == resource), key=_order_key
            )
            if not batch:
                break
            phase_id = f"r{round_index:03d}-{resource}"
            phase = self.ledger.phase(phase_id)
            phase.write_plan(batch)
            self.diagnostics.phase_count += 1
            counters = self.diagnostics.resource(resource)
            counters["epoch_count"] += 1
            epoch_started = time.perf_counter()

            progressed = False
            for job in batch:
                counters["jobs_planned"] += 1
                state: JobState = self.ledger.classify(job)
                if state.state == STATE_MISMATCH:
                    raise SchedulerError(
                        f"{job.job_id()}: {state.detail or 'receipt mismatch'}"
                    )
                if state.skippable:
                    counters["jobs_skipped_durable"] += 1
                    self.diagnostics.resume["receipts_reused"] += 1
                    if state.state == "completed" and state.detail:
                        self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                    resolved.add(job.job_id())
                    progressed = True
                    continue
                if state.state == "rerun_no_receipt":
                    self.diagnostics.resume["jobs_rerun_after_incomplete_commit"] += 1
                executor = self.executors.get(resource)
                if executor is None:
                    raise SchedulerError(
                        f"no executor configured for resource {resource!r} "
                        f"(job {job.job_id()})"
                    )
                result = executor.execute(job)
                counters["jobs_executed"] += 1
                if job.job_type in self.diagnostics.conditional_attempts:
                    self.diagnostics.conditional_attempts[job.job_type] += 1
                digests: dict[str, str] = {}
                if result.committed:
                    for name, payload in sorted(result.artifacts.items()):
                        digests[name] = phase.publish_artifact(
                            job.job_id(), name, payload
                        )
                    phase.commit(job, outcome=result.outcome, artifact_digests=digests)
                    if result.outcome == OUTCOME_TERMINAL_REJECT:
                        counters["jobs_terminal_rejected"] += 1
                else:
                    counters["jobs_retryable_failed"] += 1
                    if result.outcome != OUTCOME_RETRYABLE_FAILED:
                        counters["jobs_retryable_failed"] += 0
                if result.committed:
                    resolved.add(job.job_id())
                    progressed = True
                    for unlocked in self.finalize(job, result):
                        pending.setdefault(unlocked.job_id(), unlocked)
            counters["epoch_wall_seconds"] += time.perf_counter() - epoch_started
            round_index += 1
            if not progressed:
                break

        self.diagnostics.resume["torn_receipts_repaired"] = (
            self.ledger.torn_receipts_repaired
        )
        unresolved = sorted(set(pending) - resolved)
        return {
            "completed": not unresolved,
            "unresolved_job_ids": unresolved,
            "job_count": len(pending),
            "resolved_job_count": len(resolved),
            "wall_seconds": time.perf_counter() - started,
            "diagnostics": self.diagnostics.summary(),
        }
