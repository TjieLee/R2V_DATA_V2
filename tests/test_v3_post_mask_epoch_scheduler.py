"""In-memory regressions for the resource-epoch scheduler hot path."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import pytest

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    JobResult,
    job_order_key,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    JobExecution,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import (
    STATE_COMPLETED,
    STATE_PENDING,
    JobState,
)


@dataclass(frozen=True)
class _Job:
    clip_uid: str
    job_type: str = "attribute_review"
    resource: str = RESOURCE_QWEN
    attempt_index: int = 0

    def job_id(self) -> str:
        return f"{self.clip_uid}:{self.job_type}:{self.attempt_index}"


class _Phase:
    def __init__(self) -> None:
        self.plan_writes: list[list[str]] = []
        self.commits = 0
        self.syncs = 0

    def write_plan(self, jobs: list[_Job]) -> None:
        self.plan_writes.append([job.job_id() for job in jobs])

    def publish_result(self, job: _Job, result: JobResult) -> str:
        return "result"

    def commit(self, job: _Job, **kwargs: Any) -> object:
        self.commits += 1
        return object()

    def sync_receipts(self) -> bool:
        self.syncs += 1
        return self.commits > 0


class _Ledger:
    torn_receipts_repaired = 0

    def __init__(self) -> None:
        self.phases: dict[str, _Phase] = {}
        self.classifications: Counter[str] = Counter()
        self.states: dict[str, JobState] = {}

    def phase(self, phase_id: str) -> _Phase:
        return self.phases.setdefault(phase_id, _Phase())

    def classify(self, job: _Job) -> JobState:
        self.classifications[job.job_id()] += 1
        return self.states.get(job.job_id(), JobState(STATE_PENDING))

    def load_committed_result(self, job: _Job) -> JobResult:
        return JobResult(OUTCOME_COMPLETED)

    def note_commit(self, phase_id: str, receipt: object) -> None:
        pass


class _Executor:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self.inflight: deque[_Job] = deque()
        self.submitted: list[str] = []
        self.collect_loads: list[int] = []

    def capacity(self) -> int:
        return self._capacity

    def submit(self, job: _Job) -> None:
        self.inflight.append(job)
        self.submitted.append(job.job_id())

    def collect(self) -> list[JobExecution]:
        self.collect_loads.append(len(self.inflight))
        job = self.inflight.popleft()
        return [JobExecution(job, JobResult(OUTCOME_COMPLETED), None)]

    def close(self) -> None:
        pass


def _scheduler(ledger: _Ledger, executor: _Executor) -> ResourceEpochScheduler:
    return ResourceEpochScheduler(
        ledger=ledger,
        finalize=lambda job, result: (),
        executors={RESOURCE_QWEN: executor},
    )


def test_ready_hot_path_never_calls_full_pending_scan(monkeypatch: pytest.MonkeyPatch):
    jobs = [_Job(f"clip-{index:04d}") for index in reversed(range(20))]
    ledger = _Ledger()
    executor = _Executor(8)
    scheduler = _scheduler(ledger, executor)

    def forbidden_ready(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("_ready must not scan pending on the hot path")

    monkeypatch.setattr(scheduler, "_ready", forbidden_ready)
    outcome = scheduler.run(jobs)

    assert outcome["completed"] is True
    assert executor.submitted == [job.job_id() for job in sorted(jobs, key=job_order_key)]


@pytest.mark.parametrize("capacity", [8, 32])
def test_each_freed_slot_is_immediately_refilled(capacity: int):
    jobs = [_Job(f"clip-{index:04d}") for index in reversed(range(100))]
    ledger = _Ledger()
    executor = _Executor(capacity)
    outcome = _scheduler(ledger, executor).run(jobs)

    counters = outcome["diagnostics"]["resources"][RESOURCE_QWEN]
    assert outcome["completed"] is True
    assert executor.collect_loads[: 100 - capacity + 1] == [capacity] * (
        100 - capacity + 1
    )
    assert counters["peak_inflight"] == capacity
    assert counters["jobs_executed"] == 100


def test_fifty_thousand_ready_jobs_have_one_plan_wave_and_no_full_scan(
    monkeypatch: pytest.MonkeyPatch,
):
    jobs = [_Job(f"clip-{index:05d}") for index in reversed(range(50_000))]
    ledger = _Ledger()
    executor = _Executor(32)
    scheduler = _scheduler(ledger, executor)
    ready_calls = 0

    def forbidden_ready(*args: Any, **kwargs: Any) -> None:
        nonlocal ready_calls
        ready_calls += 1
        raise AssertionError("_ready was called while draining 50k jobs")

    monkeypatch.setattr(scheduler, "_ready", forbidden_ready)
    outcome = scheduler.run(jobs)
    counters = outcome["diagnostics"]["resources"][RESOURCE_QWEN]

    assert outcome["completed"] is True
    assert ready_calls == 0
    assert executor.submitted == [job.job_id() for job in reversed(jobs)]
    assert len(ledger.classifications) == 50_000
    assert set(ledger.classifications.values()) == {1}
    assert [len(batch) for batch in ledger.phases["r000-qwen"].plan_writes] == [50_000]
    assert counters["ready_queue_pushes"] == 50_000
    assert counters["ready_queue_pops"] == 50_000
    assert counters["ready_queue_peak"] == 50_000
    assert counters["classify_calls"] == 50_000
    assert counters["plan_write_calls"] == 1
    assert counters["plan_write_jobs"] == 50_000
    assert counters["peak_inflight"] == 32


def test_ten_thousand_initial_jobs_are_planned_in_one_write():
    jobs = [_Job(f"clip-{index:05d}") for index in range(10_000)]
    ledger = _Ledger()
    outcome = _scheduler(ledger, _Executor(32)).run(jobs)

    assert outcome["completed"] is True
    assert [len(batch) for batch in ledger.phases["r000-qwen"].plan_writes] == [10_000]


def test_conditional_unlock_is_enqueued_once_and_not_planned_early():
    first = _Job("clip-0001")
    sibling = _Job("clip-0002")
    unlocked = _Job("clip-0003", "reference_completion_candidate2")
    ledger = _Ledger()
    executor = _Executor(2)

    def finalize(job: _Job, result: JobResult) -> tuple[_Job, ...]:
        return (unlocked, unlocked) if job.job_id() == first.job_id() else ()

    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=finalize,
        executors={RESOURCE_QWEN: executor},
    )
    outcome = scheduler.run([sibling, first])
    plan_writes = ledger.phases["r000-qwen"].plan_writes

    assert outcome["completed"] is True
    assert set(plan_writes[0]) == {first.job_id(), sibling.job_id()}
    assert plan_writes[1] == [unlocked.job_id()]
    assert executor.submitted.count(unlocked.job_id()) == 1
    assert ledger.classifications[unlocked.job_id()] == 1
    assert outcome["diagnostics"]["resources"][RESOURCE_QWEN][
        "ready_queue_pushes"
    ] == 3


def test_receipt_replay_finishes_the_existing_ready_wave_before_unlocks():
    replayed = _Job("clip-0001")
    unlocked = _Job("clip-0002")
    sibling = _Job("clip-0003")
    ledger = _Ledger()
    ledger.states[replayed.job_id()] = JobState(STATE_COMPLETED)
    executor = _Executor(2)

    def finalize(job: _Job, result: JobResult) -> tuple[_Job, ...]:
        return (unlocked,) if job.job_id() == replayed.job_id() else ()

    outcome = ResourceEpochScheduler(
        ledger=ledger,
        finalize=finalize,
        executors={RESOURCE_QWEN: executor},
    ).run([sibling, replayed])

    assert outcome["completed"] is True
    assert executor.submitted == [sibling.job_id(), unlocked.job_id()]
    assert [len(batch) for batch in ledger.phases["r000-qwen"].plan_writes] == [2, 1]


def test_custom_priority_fallback_preserves_lowest_ready_job_id():
    sam_job = _Job("clip-0001", resource=RESOURCE_SAM)
    boogu_job = _Job("clip-0002", resource=RESOURCE_BOOGU)
    ledger = _Ledger()
    executor = _Executor(1)
    outcome = ResourceEpochScheduler(
        ledger=ledger,
        finalize=lambda job, result: (),
        executors={RESOURCE_SAM: executor, RESOURCE_BOOGU: executor},
        resource_priority=(RESOURCE_QWEN,),
    ).run([boogu_job, sam_job])

    assert outcome["completed"] is True
    assert executor.submitted == [sam_job.job_id(), boogu_job.job_id()]


def test_shared_resource_executor_survives_resource_switch_until_session_close():
    first = _Job("clip-0001", resource=RESOURCE_QWEN)
    middle = _Job("clip-0001", job_type="sam_probe", resource=RESOURCE_SAM)
    last = _Job("clip-0001", job_type="final_review", resource=RESOURCE_QWEN)

    class LiveExecutor(_Executor):
        def __init__(self) -> None:
            super().__init__(1)
            self.closed = False
            self.close_count = 0

        def submit(self, job: _Job) -> None:
            assert not self.closed, "a shared executor closed before session shutdown"
            super().submit(job)

        def close(self) -> None:
            self.closed = True
            self.close_count += 1

    class SharedManager:
        def __init__(self) -> None:
            self.executors = {
                RESOURCE_QWEN: LiveExecutor(),
                RESOURCE_SAM: LiveExecutor(),
            }
            self.close_count = 0

        def enter(self, resource: str) -> LiveExecutor:
            return self.executors[resource]

        def close(self) -> None:
            self.close_count += 1
            for executor in self.executors.values():
                executor.close()

        def counters(self) -> dict[str, Any]:
            return {}

    manager = SharedManager()

    def finalize(job: _Job, _result: JobResult) -> tuple[_Job, ...]:
        if job == first:
            return (middle,)
        if job == middle:
            return (last,)
        return ()

    outcome = ResourceEpochScheduler(
        ledger=_Ledger(), finalize=finalize, resource_manager=manager
    ).run([first])

    assert outcome["completed"] is True
    assert manager.executors[RESOURCE_QWEN].submitted == [
        first.job_id(), last.job_id()
    ]
    assert manager.close_count == 1
    assert {resource: executor.close_count for resource, executor in manager.executors.items()} == {
        RESOURCE_QWEN: 1,
        RESOURCE_SAM: 1,
    }


def test_opt_in_collect_wave_commits_each_receipt_before_owner_finalization():
    first = _Job("clip-0001", "sam_probe_a1")
    second = _Job("clip-0001", "sam_probe_a2")
    ledger = _Ledger()

    class WaveExecutor(_Executor):
        def collect(self) -> list[JobExecution]:
            jobs = list(self.inflight)
            self.inflight.clear()
            return [JobExecution(job, JobResult(OUTCOME_COMPLETED), None) for job in jobs]

    advances: list[tuple[str, ...]] = []

    def finalize_wave(committed: list[tuple[_Job, JobResult]]) -> dict[str, tuple[_Job, ...]]:
        assert ledger.phases["r000-qwen"].commits == 2
        advances.append(tuple(job.job_id() for job, _ in committed))
        return {job.job_id(): () for job, _ in committed}

    outcome = ResourceEpochScheduler(
        ledger=ledger,
        finalize=lambda job, result: pytest.fail("per-job finalize was called"),
        finalize_wave=finalize_wave,
        executors={RESOURCE_QWEN: WaveExecutor(2)},
    ).run([first, second])

    assert outcome["completed"] is True
    assert ledger.phases["r000-qwen"].commits == 2
    assert advances == [(first.job_id(), second.job_id())]


def test_opt_in_collect_wave_failed_finalizer_preserves_committed_receipt():
    good = _Job("clip-0001", "sam_probe_a1")
    bad = _Job("clip-0001", "sam_probe_a2")
    ledger = _Ledger()

    class WaveExecutor(_Executor):
        def collect(self) -> list[JobExecution]:
            jobs = list(self.inflight)
            self.inflight.clear()
            return [JobExecution(job, JobResult(OUTCOME_COMPLETED), None) for job in jobs]

    def finalize_wave(committed: list[tuple[_Job, JobResult]]) -> dict[str, tuple[_Job, ...] | None]:
        return {good.job_id(): (), bad.job_id(): None}

    outcome = ResourceEpochScheduler(
        ledger=ledger,
        finalize=lambda job, result: pytest.fail("per-job finalize was called"),
        finalize_wave=finalize_wave,
        executors={RESOURCE_QWEN: WaveExecutor(2)},
    ).run([good, bad])

    assert outcome["completed"] is False
    assert outcome["resolved_job_count"] == 1
    assert outcome["unresolved_job_ids"] == [bad.job_id()]
    assert ledger.phases["r000-qwen"].commits == 2
    assert outcome["diagnostics"]["resume"]["finalizer_failures"] == 1


def test_run_batches_keeps_plans_and_receipt_syncs_without_duplicate_enqueue():
    first = _Job("clip-0001")
    unlocked = _Job("clip-0002")
    later = _Job("clip-0003")
    ledger = _Ledger()
    executor = _Executor(2)

    def finalize(job: _Job, result: JobResult) -> tuple[_Job, ...]:
        return (unlocked,) if job.job_id() == first.job_id() else ()

    outcome = ResourceEpochScheduler(
        ledger=ledger,
        finalize=finalize,
        executors={RESOURCE_QWEN: executor},
    ).run_batches([[first], [unlocked, later]])

    assert outcome["completed"] is True
    assert outcome["job_count"] == outcome["seed_job_count"] == 3
    assert executor.submitted == [first.job_id(), unlocked.job_id(), later.job_id()]
    assert [len(batch) for batch in ledger.phases["r000-qwen"].plan_writes] == [1, 1]
    assert [len(batch) for batch in ledger.phases["r001-qwen"].plan_writes] == [1]
    assert all(phase.syncs == 1 for phase in ledger.phases.values())
    assert set(ledger.classifications.values()) == {1}


def test_three_hundred_initial_jobs_drain_each_batch_with_one_resource_session():
    """A later batch is requested only after every current clip reaches its final job."""
    initial = [_Job(f"clip-{index:04d}", "discovery") for index in range(300)]
    completed: set[str] = set()
    batch_sizes: list[int] = []

    class LiveExecutor(_Executor):
        def __init__(self) -> None:
            super().__init__(8)
            self.closed = False

        def submit(self, job: _Job) -> None:
            assert not self.closed, "resource closed across a batch boundary"
            super().submit(job)

        def close(self) -> None:
            self.closed = True

    class SharedManager:
        def __init__(self) -> None:
            self.executors: dict[str, LiveExecutor] = {}
            self.starts: Counter[str] = Counter()
            self.close_count = 0

        def enter(self, resource: str) -> LiveExecutor:
            if resource not in self.executors:
                self.executors[resource] = LiveExecutor()
                self.starts[resource] += 1
            return self.executors[resource]

        def close(self) -> None:
            self.close_count += 1
            for executor in self.executors.values():
                executor.close()

        def counters(self) -> dict[str, Any]:
            return {}

    manager = SharedManager()

    def batches():
        for start in range(0, len(initial), 128):
            assert len(completed) == start
            assert manager.close_count == 0
            batch = initial[start : start + 128]
            batch_sizes.append(len(batch))
            yield batch

    def finalize(job: _Job, _result: JobResult) -> tuple[_Job, ...]:
        if job.job_type == "discovery":
            return (_Job(job.clip_uid, "sam_probe", RESOURCE_SAM),)
        if job.job_type == "sam_probe":
            return (_Job(job.clip_uid, "completion", RESOURCE_BOOGU),)
        if job.job_type == "completion":
            return (_Job(job.clip_uid, "final_review", RESOURCE_QWEN),)
        assert job.job_type == "final_review"
        completed.add(job.clip_uid)
        return ()

    outcome = ResourceEpochScheduler(
        ledger=_Ledger(), finalize=finalize, resource_manager=manager
    ).run_batches(batches())

    assert outcome["completed"] is True
    assert outcome["seed_job_count"] == 300
    assert outcome["job_count"] == 1_200
    assert batch_sizes == [128, 128, 44]
    assert len(completed) == 300
    assert manager.starts == Counter(
        {RESOURCE_QWEN: 1, RESOURCE_SAM: 1, RESOURCE_BOOGU: 1}
    )
    assert manager.close_count == 1
