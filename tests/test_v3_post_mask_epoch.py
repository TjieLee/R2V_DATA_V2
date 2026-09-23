"""Resource-epoch execution substrate: identity, durability and scheduling.

These tests cover the ``resource_epoch_v3`` substrate only. They never
construct a real model, GPU or endpoint, and they do not touch the frozen
semantic modules. Semantic equivalence against the existing scheduler is a
separate, server-side validation and is explicitly *not* claimed here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.post_mask_epoch_groups import (
    DEFAULT_GROUP_SIZE,
    GROUP_COMPLETED,
    GROUP_INCOMPLETE,
    assigned_groups,
    build_groups,
    campaign_identity,
    group_root,
    read_group_outcomes,
    record_group_outcome,
    write_group_descriptor,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    ArtifactReference,
    JobResult,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    DEFAULT_WINDOW_SIZE,
    EpochDiagnostics,
    JobExecution,
    ResourceEpochScheduler,
    SchedulerError,
)
from r2v_data_v2.v3.post_mask_epoch_state import (
    STATE_COMPLETED,
    STATE_MISMATCH,
    STATE_PENDING,
    STATE_RERUN_NO_RECEIPT,
    STATE_TERMINAL_REJECT,
    GroupLedger,
    LedgerError,
    PhaseLedger,
    PlanMismatchError,
    verify_external_artifact,
)

CAMPAIGN = {"config_hash": "cfg-1", "dataset": "jea-motion-v1"}


def _shards(count: int) -> list[str]:
    return [f"shard-{i:09d}-{i + 999:09d}" for i in range(count)]


def _job(
    *,
    job_type: str = "background_removal",
    resource: str = RESOURCE_BOOGU,
    clip_uid: str = "clip-000001",
    attempt_index: int = 0,
    seed: int | None = None,
    semantic: Any = None,
    shard: str = "shard-000000000-000000999",
    model: str = "boogu-image-0.1",
    target: dict | None = None,
) -> ModelJob:
    return ModelJob.create(
        job_type=job_type,
        resource=resource,
        canonical_shard=shard,
        clip_uid=clip_uid,
        semantic_inputs=semantic if semantic is not None else {"prompt": "remove bg"},
        model_identity=model,
        target=target,
        attempt_index=attempt_index,
        seed=seed,
    )


class _FakeBatchExecutor:
    """Deterministic fake resource; records every batch it is asked to run."""

    def __init__(self, outcomes: dict[str, JobResult] | None = None):
        self.calls: list[ModelJob] = []
        self.batch_sizes: list[int] = []
        self.outcomes = outcomes or {}

    def execute_batch(self, jobs):
        self.batch_sizes.append(len(jobs))
        results = {}
        for index, job in enumerate(jobs):
            self.calls.append(job)
            outcome = self.outcomes.get(
                job.job_id(),
                JobResult(
                    OUTCOME_COMPLETED,
                    {"out.png": b"ok"},
                    payload={"index": index},
                ),
            )
            results[job.job_id()] = JobExecution(job, outcome, None)
        return results

    def call_ids(self) -> list[str]:
        return [job.job_id() for job in self.calls]


def _scheduler(
    tmp_path: Path,
    executors: dict,
    finalize,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    ledger: GroupLedger | None = None,
) -> ResourceEpochScheduler:
    return ResourceEpochScheduler(
        ledger=ledger if ledger is not None else GroupLedger(tmp_path / "group"),
        executors=executors,
        finalize=finalize,
        window_size=window_size,
    )


# --------------------------------------------------------------------------
# 1. deterministic 8-shard grouping
# --------------------------------------------------------------------------


def test_grouping_is_contiguous_over_sorted_canonical_shards():
    groups = build_groups(_shards(16), campaign=CAMPAIGN)
    assert len(groups) == 2
    assert groups[0].canonical_shards == tuple(sorted(_shards(16))[:8])
    assert groups[1].canonical_shards == tuple(sorted(_shards(16))[8:])


def test_campaign_384_shards_yield_48_groups():
    groups = build_groups(_shards(384), campaign=CAMPAIGN)
    assert len(groups) == 48
    assert all(len(g.canonical_shards) == DEFAULT_GROUP_SIZE for g in groups)


def test_group_identity_excludes_rank_world_host_gpu_and_pid():
    groups = build_groups(_shards(16), campaign=CAMPAIGN)
    blob = json.dumps(groups[0].identity_payload()).lower()
    for forbidden in ("rank", "world", "host", "gpu", "pid", "cuda"):
        assert forbidden not in blob
    for rank, world in ((0, 1), (1, 4), (3, 7)):
        owned = assigned_groups(groups, rank=rank, world_size=world)
        for group in owned:
            assert group.identity() == groups[group.group_index].identity()


def test_group_identity_changes_with_shards_campaign_or_size():
    base = build_groups(_shards(8), campaign=CAMPAIGN)[0]
    other_shards = build_groups(_shards(9)[1:], campaign=CAMPAIGN)[0]
    other_campaign = build_groups(_shards(8), campaign={"config_hash": "cfg-2"})[0]
    other_size = build_groups(_shards(8), campaign=CAMPAIGN, group_size=4)[0]
    identities = {
        base.identity(),
        other_shards.identity(),
        other_campaign.identity(),
        other_size.identity(),
    }
    assert len(identities) == 4


def test_static_assignment_partitions_groups_without_overlap():
    groups = build_groups(_shards(32), campaign=CAMPAIGN)
    owned = [assigned_groups(groups, rank=r, world_size=4) for r in range(4)]
    assert sorted(g.group_index for group in owned for g in group) == list(range(4))
    for index, group in enumerate(owned):
        assert all(g.group_index % 4 == index for g in group)


def test_assignment_rejects_invalid_topology():
    groups = build_groups(_shards(8), campaign=CAMPAIGN)
    with pytest.raises(ValueError):
        assigned_groups(groups, rank=4, world_size=4)
    with pytest.raises(ValueError):
        assigned_groups(groups, rank=0, world_size=0)


# --------------------------------------------------------------------------
# 2. job identity
# --------------------------------------------------------------------------


def test_job_id_is_the_full_identity_digest():
    job = _job()
    assert job.job_id() == job.identity()
    assert len(job.job_id()) == 64
    assert set(job.job_id()) <= set("0123456789abcdef")


def test_job_identity_is_stable_and_seed_aware():
    first = _job()
    again = _job()
    assert first.identity() == again.identity()
    assert _job(seed=1).identity() != first.identity()
    assert _job(attempt_index=1).identity() != first.identity()


def test_job_identity_binds_semantic_inputs_dependencies_and_model():
    base = _job()
    assert _job(semantic={"prompt": "other"}).identity() != base.identity()
    assert _job(model="other-model").identity() != base.identity()
    assert _job(target={"owner": "ent-1"}).identity() != base.identity()
    with_deps = ModelJob.create(
        job_type="background_removal",
        resource=RESOURCE_BOOGU,
        canonical_shard=base.canonical_shard,
        clip_uid=base.clip_uid,
        semantic_inputs={"prompt": "remove bg"},
        model_identity="boogu-image-0.1",
        dependencies={"candidate1": "deadbeef"},
    )
    assert with_deps.identity() != base.identity()


def test_job_identity_excludes_execution_only_label():
    plain = _job()
    labelled = ModelJob.create(
        job_type=plain.job_type,
        resource=plain.resource,
        canonical_shard=plain.canonical_shard,
        clip_uid=plain.clip_uid,
        semantic_inputs={"prompt": "remove bg"},
        model_identity=plain.model_identity,
        label="slot-3/gpu-7/host-a",
    )
    assert labelled.identity() == plain.identity()


def test_job_rejects_unknown_resource():
    with pytest.raises(ValueError):
        _job(resource="flux")


def test_many_jobs_have_unique_ids():
    jobs = [_job(clip_uid=f"clip-{index:06d}") for index in range(2000)]
    assert len({job.job_id() for job in jobs}) == 2000


# --------------------------------------------------------------------------
# 3. durable ledger
# --------------------------------------------------------------------------


def test_plan_is_monotonic_and_detects_changed_record(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    jobs = [_job(), _job(clip_uid="clip-000002")]
    digest = ledger.write_plan(jobs)
    assert ledger.write_plan(jobs) == digest

    class _Mutant:
        """Same job id, different plan record: the only real violation."""

        def job_id(self) -> str:
            return jobs[0].job_id()

        def plan_record(self) -> dict[str, Any]:
            return {**jobs[0].plan_record(), "mutated": True}

    with pytest.raises(PlanMismatchError):
        ledger.write_plan([_Mutant()])


def test_missing_plan_fails_closed(tmp_path: Path):
    with pytest.raises(LedgerError):
        PhaseLedger(tmp_path / "absent").read_plan()


def test_pending_job_has_no_receipt(tmp_path: Path):
    assert PhaseLedger(tmp_path / "phase").classify(_job()).state == STATE_PENDING


def test_completed_receipt_with_matching_artifacts_skips_model_call(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    _commit_result(ledger, job, JobResult(OUTCOME_COMPLETED, {"out.png": b"payload"}))
    state = ledger.classify(job)
    assert state.state == STATE_COMPLETED
    assert state.skippable


def test_artifact_without_receipt_is_treated_as_incomplete(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    assert ledger.classify(job).state == STATE_RERUN_NO_RECEIPT
    assert not ledger.classify(job).skippable


def test_terminal_reject_is_durable_and_never_paid_again(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="removal_judge")
    _commit_result(
        ledger, job, JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": True})
    )
    state = ledger.classify(job)
    assert state.state == STATE_TERMINAL_REJECT
    assert state.skippable


def test_digest_mismatch_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    ledger.commit(job, outcome=OUTCOME_COMPLETED, artifact_digests={"out.png": digest})
    ledger.publish_artifact(job.job_id(), "out.png", b"tampered")
    state = ledger.classify(job)
    assert state.state == STATE_MISMATCH
    assert not state.skippable


def test_missing_durable_result_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    ledger.commit(
        job,
        outcome=OUTCOME_COMPLETED,
        artifact_digests={"out.png": digest},
        result_digest="deadbeef",
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_durable_result_round_trips(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    result = JobResult(OUTCOME_COMPLETED, payload={"accepted": True, "score": 0.5})
    digest = ledger.publish_result(job, result)
    assert digest
    loaded = ledger.load_result(job)
    assert loaded is not None
    assert loaded.outcome == OUTCOME_COMPLETED
    assert loaded.payload == {"accepted": True, "score": 0.5}


def test_torn_final_receipt_line_is_repaired_and_earlier_lines_kept(tmp_path: Path):
    from r2v_data_v2.v3.post_mask_epoch_state import _load_jsonl

    ledger = PhaseLedger(tmp_path / "phase")
    first = _job(clip_uid="clip-000001")
    _commit_result(ledger, first, JobResult(OUTCOME_COMPLETED, payload={"ok": True}))
    with ledger.receipts_path.open("ab") as handle:
        handle.write(b'{"job_id":"torn","job_ide')
    loaded, was_repaired = _load_jsonl(ledger.receipts_path)
    assert was_repaired is True
    assert [r["job_id"] for r in loaded] == [first.job_id()]
    assert ledger.classify(first).state == STATE_COMPLETED


def test_receipts_are_append_only_across_resume(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    ledger.commit(job, outcome=OUTCOME_TERMINAL_REJECT, artifact_digests={})
    before = ledger.receipts_path.read_bytes()
    ledger.append_receipt(
        ledger.commit(job, outcome=OUTCOME_TERMINAL_REJECT, artifact_digests={})
    )
    assert ledger.receipts_path.read_bytes().startswith(before)


def test_group_ledger_locates_receipt_across_phases(tmp_path: Path):
    group = GroupLedger(tmp_path / "group")
    job = _job()
    _commit_result(
        group.phase("r000-boogu"),
        job,
        JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": True}),
    )
    assert group.classify(job).state == STATE_TERMINAL_REJECT


def test_group_ledger_reports_artifact_without_receipt(tmp_path: Path):
    group = GroupLedger(tmp_path / "group")
    job = _job()
    group.phase("r000-boogu").publish_artifact(job.job_id(), "out.png", b"x")
    assert group.classify(job).state == STATE_RERUN_NO_RECEIPT


def test_group_ledger_classify_does_not_rescan_jsonl_per_job(tmp_path, monkeypatch):
    """200 committed jobs must not cause 200 JSONL parses."""
    import r2v_data_v2.v3.post_mask_epoch_state as state_module

    root = tmp_path / "group"
    ledger = GroupLedger(root)
    jobs = [_job(clip_uid=f"clip-{index:06d}") for index in range(200)]
    phase = ledger.phase("r000-boogu")
    phase.write_plan(jobs)
    for job in jobs:
        artifact = phase.publish_artifact(job.job_id(), "out.png", b"x")
        result = phase.publish_result(
            job, JobResult(OUTCOME_COMPLETED, payload={"ok": True})
        )
        receipt = phase.commit(
            job,
            outcome=OUTCOME_COMPLETED,
            artifact_digests={"out.png": artifact, "result.json": result},
            result_digest=result,
        )
        ledger.note_commit("r000-boogu", receipt)

    calls = {"count": 0}
    real = state_module._load_jsonl

    def counting(path):
        calls["count"] += 1
        return real(path)

    monkeypatch.setattr(state_module, "_load_jsonl", counting)
    fresh = GroupLedger(root)
    for job in jobs:
        assert fresh.classify(job).state == STATE_COMPLETED
    # One refresh over one phase, not one parse per job.
    assert calls["count"] <= 2
    assert fresh.jsonl_load_calls <= 2


# --------------------------------------------------------------------------
# 4. group outcomes and isolation
# --------------------------------------------------------------------------


def test_group_outcomes_are_append_only(tmp_path: Path):
    group = build_groups(_shards(8), campaign=CAMPAIGN)[0]
    record_group_outcome(tmp_path, group, outcome=GROUP_INCOMPLETE, reason="crash")
    record_group_outcome(tmp_path, group, outcome=GROUP_COMPLETED)
    outcomes = read_group_outcomes(tmp_path, group)
    assert [o["outcome"] for o in outcomes] == [GROUP_INCOMPLETE, GROUP_COMPLETED]


def test_group_descriptor_is_immutable(tmp_path: Path):
    group = build_groups(_shards(8), campaign=CAMPAIGN)[0]
    write_group_descriptor(tmp_path, group)
    write_group_descriptor(tmp_path, group)
    other = build_groups(_shards(9)[1:], campaign=CAMPAIGN)[0]
    with pytest.raises(ValueError):
        write_group_descriptor(tmp_path, other)
    assert group_root(tmp_path, group).is_dir()


def test_one_incomplete_group_does_not_alter_others(tmp_path: Path):
    groups = build_groups(_shards(16), campaign=CAMPAIGN)
    record_group_outcome(tmp_path, groups[0], outcome=GROUP_INCOMPLETE)
    assert read_group_outcomes(tmp_path, groups[1]) == []


def test_campaign_identity_is_semantic_only():
    assert campaign_identity({"a": 1}) != campaign_identity({"a": 2})


# --------------------------------------------------------------------------
# 5. scheduler
# --------------------------------------------------------------------------


def test_scheduler_runs_ready_jobs_and_commits_receipts(tmp_path: Path):
    executor = _FakeBatchExecutor()
    scheduler = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda job, result: ())
    job = _job()
    outcome = scheduler.run([job])
    assert outcome["completed"] is True
    assert executor.call_ids() == [job.job_id()]
    assert GroupLedger(tmp_path / "group").classify(job).state == STATE_COMPLETED


def test_scheduler_never_creates_conditional_second_attempt_speculatively(tmp_path: Path):
    boogu = _FakeBatchExecutor()
    qwen = _FakeBatchExecutor()
    second = _job(job_type="background_removal_candidate2", attempt_index=1, seed=2)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        # Existing policy accepted candidate1, so candidate2 is never unlocked.
        return ()

    scheduler = _scheduler(
        tmp_path, {RESOURCE_BOOGU: boogu, RESOURCE_QWEN: qwen}, finalize
    )
    scheduler.run([_job()])
    assert boogu.call_ids() == [_job().job_id()]
    assert qwen.calls == []
    assert second.job_id() not in boogu.call_ids()


def test_conditional_second_attempt_runs_only_when_policy_unlocks_it(tmp_path: Path):
    executor = _FakeBatchExecutor()
    second = _job(job_type="background_removal_candidate2", attempt_index=1, seed=2)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        return (second,) if job.attempt_index == 0 else ()

    scheduler = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize)
    scheduler.run([_job()])
    assert len(executor.calls) == 2
    assert executor.calls[1].job_id() == second.job_id()


def test_fixed_point_keeps_same_resource_before_switching(tmp_path: Path):
    qwen = _FakeBatchExecutor()
    boogu = _FakeBatchExecutor()
    second_qwen = _job(job_type="removal_judge2", resource=RESOURCE_QWEN, attempt_index=1)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        return (second_qwen,) if job.resource == RESOURCE_BOOGU else ()

    scheduler = _scheduler(
        tmp_path, {RESOURCE_QWEN: qwen, RESOURCE_BOOGU: boogu}, finalize
    )
    outcome = scheduler.run([_job(resource=RESOURCE_BOOGU)])
    assert outcome["completed"] is True
    assert len(boogu.calls) == 1
    assert len(qwen.calls) == 1
    assert scheduler.diagnostics.resource_switches == 1


def test_resource_selection_is_deterministic_across_runs(tmp_path: Path):
    def run(root: Path) -> list[str]:
        executors = {
            name: _FakeBatchExecutor()
            for name in (RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM)
        }
        jobs = [
            _job(resource=RESOURCE_SAM, job_type="sam_review", clip_uid="clip-000001"),
            _job(resource=RESOURCE_QWEN, job_type="qwen_judge", clip_uid="clip-000002"),
            _job(resource=RESOURCE_BOOGU, job_type="boogu_edit", clip_uid="clip-000003"),
        ]
        scheduler = _scheduler(root, executors, lambda job, result: ())
        scheduler.run(jobs)
        return list(GroupLedger(root / "group").phase_ids())

    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_completed_jobs_are_skipped_on_second_run(tmp_path: Path):
    executor = _FakeBatchExecutor()
    job = _job()
    _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run([job])
    second = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ())
    outcome = second.run([job])
    assert outcome["completed"] is True
    assert len(executor.calls) == 1
    assert second.diagnostics.resume["receipts_reused"] >= 1


def test_execution_uses_bounded_windows(tmp_path: Path):
    jobs = [_job(clip_uid=f"clip-{index:03d}") for index in range(10)]
    executor = _FakeBatchExecutor()
    scheduler = _scheduler(
        tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: (), window_size=4
    )
    outcome = scheduler.run(jobs)
    assert outcome["completed"] is True
    assert max(executor.batch_sizes) <= 4
    assert sum(executor.batch_sizes) == 10
    assert scheduler.diagnostics.window_count == 3


def test_diagnostics_report_required_counters(tmp_path: Path):
    executor = _FakeBatchExecutor()
    scheduler = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ())
    scheduler.diagnostics.group_count = 48
    outcome = scheduler.run([_job()])
    summary = outcome["diagnostics"]
    for key in (
        "group_count",
        "group_completed",
        "group_incomplete",
        "phase_count",
        "window_count",
        "resource_switches",
        "conditional_attempts",
        "resume",
        "resources",
    ):
        assert key in summary
    boogu = summary["resources"][RESOURCE_BOOGU]
    for key in (
        "jobs_planned",
        "jobs_skipped_durable",
        "jobs_executed",
        "jobs_terminal_rejected",
        "jobs_retryable_failed",
        "model_service_seconds",
        "queue_wait_seconds",
        "epoch_wall_seconds",
        "gpu_slot_job_counts",
    ):
        assert key in boogu


def test_diagnostics_defaults_are_complete():
    summary = EpochDiagnostics().summary()
    assert set(summary["resources"]) == {RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM}


def test_scheduler_rejects_unknown_executor(tmp_path: Path):
    with pytest.raises(ValueError):
        ResourceEpochScheduler(
            ledger=GroupLedger(tmp_path / "g"),
            executors={"flux": _FakeBatchExecutor()},
            finalize=lambda j, r: (),
        )


def test_scheduler_fails_clearly_when_a_ready_resource_has_no_executor(tmp_path: Path):
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "g"),
        executors={RESOURCE_QWEN: _FakeBatchExecutor()},
        finalize=lambda j, r: (),
    )
    with pytest.raises(SchedulerError, match="no executor configured"):
        scheduler.run([_job(resource=RESOURCE_SAM)])


def test_scheduler_fails_closed_on_receipt_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    job = _job()
    ledger = GroupLedger(tmp_path / "group")
    executor = _FakeBatchExecutor()
    _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run([job])
    ledger.phase(ledger.phase_ids()[0]).publish_artifact(
        job.job_id(), "out.png", b"tampered"
    )
    def no_plan_write(self, jobs):
        raise AssertionError("a mismatched receipt must fail before plan mutation")

    monkeypatch.setattr(PhaseLedger, "write_plan", no_plan_write)
    with pytest.raises(SchedulerError):
        _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run([job])


# --------------------------------------------------------------------------
# 6. resume: skip model call != skip finalizer
# --------------------------------------------------------------------------


def test_receipt_before_finalize_crash_replays_finalizer(tmp_path: Path):
    """Case A: crash after receipt, before finalize, must not strand children."""
    first = _job(job_type="removal")
    second = _job(job_type="removal_candidate2", attempt_index=1, seed=2)
    state = {"fail": True, "finalized": []}
    executor = _FakeBatchExecutor()

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        if state["fail"]:
            raise RuntimeError("simulated crash before finalize")
        state["finalized"].append(job.job_id())
        return (second,) if job.job_id() == first.job_id() else ()

    broken = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize)
    outcome = broken.run([first])
    assert outcome["completed"] is False
    assert executor.call_ids() == [first.job_id()]

    state["fail"] = False
    resumed = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize)
    outcome = resumed.run([first])
    assert outcome["completed"] is True
    # The model was called once for `first`; only `second` is new work.
    assert executor.call_ids() == [first.job_id(), second.job_id()]
    assert state["finalized"] == [first.job_id(), second.job_id()]
    assert resumed.diagnostics.resume["finalizers_replayed"] >= 1


def test_terminal_reject_replays_finalizer_without_repeat_call(tmp_path: Path):
    """Case B: terminal reject is durable and its finalizer is replayed."""
    job = _job(job_type="removal_judge", resource=RESOURCE_QWEN)
    executor = _FakeBatchExecutor(
        {job.job_id(): JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": True})}
    )
    state = {"fail": True, "seen": []}

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        if state["fail"]:
            raise RuntimeError("simulated crash before finalize")
        state["seen"].append((job.job_id(), result.outcome, dict(result.payload)))
        return ()

    _scheduler(tmp_path, {RESOURCE_QWEN: executor}, finalize).run([job])
    assert len(executor.calls) == 1
    state["fail"] = False
    outcome = _scheduler(tmp_path, {RESOURCE_QWEN: executor}, finalize).run([job])
    assert outcome["completed"] is True
    assert len(executor.calls) == 1
    assert state["seen"] == [
        (job.job_id(), OUTCOME_TERMINAL_REJECT, {"rejected": True})
    ]


def test_finalizer_failure_is_retried_on_restart_not_replayed_model(tmp_path: Path):
    """Case C: a finalizer crash must not cost another model call."""
    job = _job()
    executor = _FakeBatchExecutor()
    state = {"fail": True, "finalized": 0}

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        if state["fail"]:
            raise RuntimeError("finalizer boom")
        state["finalized"] += 1
        return ()

    failed = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize)
    outcome = failed.run([job])
    assert outcome["completed"] is False
    assert failed.diagnostics.resume["finalizer_failures"] == 1

    state["fail"] = False
    resumed = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize)
    outcome = resumed.run([job])
    assert outcome["completed"] is True
    assert executor.call_ids() == [job.job_id()]
    assert state["finalized"] == 1


def test_finalizer_replay_is_idempotent(tmp_path: Path):
    job = _job()
    executor = _FakeBatchExecutor()
    counter = {"calls": 0}

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        counter["calls"] += 1
        return (_job(job_type="downstream", clip_uid="clip-000009"),)

    for _ in range(3):
        outcome = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize).run([job])
        assert outcome["completed"] is True
    # One model call, one downstream job, replays do not duplicate either.
    assert executor.call_ids() == [job.job_id(), _job(
        job_type="downstream", clip_uid="clip-000009"
    ).job_id()]


# --------------------------------------------------------------------------
# 7. retryable: once per invocation, retry on restart
# --------------------------------------------------------------------------


def test_retryable_job_runs_once_per_invocation(tmp_path: Path):
    retryable = _job()
    healthy = _job(clip_uid="clip-000002")
    executor = _FakeBatchExecutor(
        {retryable.job_id(): JobResult(OUTCOME_RETRYABLE_FAILED)}
    )
    outcome = _scheduler(
        tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()
    ).run([retryable, healthy])
    calls = executor.call_ids()
    assert calls.count(retryable.job_id()) == 1
    assert calls.count(healthy.job_id()) == 1
    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [retryable.job_id()]
    # No receipt was written, so resume must pay for the call again.
    assert GroupLedger(tmp_path / "group").classify(retryable).state == STATE_PENDING


def test_retryable_job_is_retried_after_restart(tmp_path: Path):
    retryable = _job()
    healthy = _job(clip_uid="clip-000002")
    executor = _FakeBatchExecutor(
        {retryable.job_id(): JobResult(OUTCOME_RETRYABLE_FAILED)}
    )
    _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run(
        [retryable, healthy]
    )
    outcome = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run(
        [retryable, healthy]
    )
    calls = executor.call_ids()
    assert calls.count(retryable.job_id()) == 2
    # The healthy sibling stays committed and is never re-paid.
    assert calls.count(healthy.job_id()) == 1
    assert outcome["completed"] is False


def test_terminal_reject_is_not_paid_again_on_resume(tmp_path: Path):
    job = _job(job_type="removal_judge", resource=RESOURCE_QWEN)
    executor = _FakeBatchExecutor(
        {job.job_id(): JobResult(OUTCOME_TERMINAL_REJECT)}
    )
    _scheduler(tmp_path, {RESOURCE_QWEN: executor}, lambda j, r: ()).run([job])
    outcome = _scheduler(tmp_path, {RESOURCE_QWEN: executor}, lambda j, r: ()).run([job])
    assert outcome["completed"] is True
    assert len(executor.calls) == 1


def test_model_exception_does_not_abort_sibling_jobs(tmp_path: Path):
    boom = _job()
    healthy = _job(clip_uid="clip-000002")

    class _FlakyExecutor:
        def __init__(self):
            self.calls: list[ModelJob] = []

        def execute_batch(self, jobs):
            results = {}
            for job in jobs:
                self.calls.append(job)
                if job.job_id() == boom.job_id():
                    results[job.job_id()] = JobExecution(
                        job, None, RuntimeError("model infra failure")
                    )
                else:
                    results[job.job_id()] = JobExecution(
                        job, JobResult(OUTCOME_COMPLETED, {"out.png": b"ok"}), None
                    )
            return results

    executor = _FlakyExecutor()
    outcome = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, lambda j, r: ()).run(
        [boom, healthy]
    )
    assert len(executor.calls) == 2
    assert outcome["unresolved_job_ids"] == [boom.job_id()]
    assert GroupLedger(tmp_path / "group").classify(healthy).state == STATE_COMPLETED


# --------------------------------------------------------------------------
# 8. monotonic phase plan across restart
# --------------------------------------------------------------------------


class _OutcomeExecutor:
    """Executes jobs, optionally forcing some to a retryable outcome."""

    def __init__(self, log: list[str], retryable: set[str] | None = None):
        self.log = log
        self.retryable = retryable or set()

    def execute_batch(self, jobs):
        results = {}
        for job in jobs:
            self.log.append(job.job_id())
            if job.job_id() in self.retryable:
                results[job.job_id()] = JobExecution(
                    job, JobResult(OUTCOME_RETRYABLE_FAILED), None
                )
            else:
                results[job.job_id()] = JobExecution(
                    job,
                    JobResult(
                        OUTCOME_COMPLETED,
                        {"out.png": b"x"},
                        payload={"ok": True},
                    ),
                    None,
                )
        return results


def test_dynamic_ready_set_growth_across_restart(tmp_path: Path):
    """A phase's ready set may grow after a restart; the plan must not break."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    qwen_first = _job(
        job_type="removal_judge", resource=RESOURCE_QWEN, clip_uid="clip-000001"
    )
    qwen_second = _job(
        job_type="removal_judge", resource=RESOURCE_QWEN, clip_uid="clip-000002"
    )
    boogu_calls: list[str] = []
    qwen_calls: list[str] = []

    def finalize(job: ModelJob, result: JobResult):
        if job.job_id() == first.job_id():
            return (qwen_first,)
        if job.job_id() == second.job_id():
            return (qwen_second,)
        return ()

    # Run 1: `second` is retryable, so only qwen_first is unlocked.
    boogu = _OutcomeExecutor(boogu_calls, {second.job_id()})
    qwen = _OutcomeExecutor(qwen_calls)
    first_run = _scheduler(
        tmp_path, {RESOURCE_BOOGU: boogu, RESOURCE_QWEN: qwen}, finalize
    )
    outcome = first_run.run([first, second])
    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [second.job_id()]

    # Run 2: `second` succeeds, so the qwen phase now holds two jobs.
    boogu_resumed = _OutcomeExecutor(boogu_calls)
    second_run = _scheduler(
        tmp_path, {RESOURCE_BOOGU: boogu_resumed, RESOURCE_QWEN: qwen}, finalize
    )
    outcome = second_run.run([first, second])

    assert outcome["completed"] is True
    assert boogu_calls.count(first.job_id()) == 1
    assert boogu_calls.count(second.job_id()) == 2
    assert qwen_calls.count(qwen_first.job_id()) == 1
    assert qwen_calls.count(qwen_second.job_id()) == 1


def test_phase_plan_job_set_only_grows(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    first = _job()
    second = _job(clip_uid="clip-000002")
    ledger.write_plan([first])
    ledger.write_plan([first, second])
    planned = {record["job_id"] for record in ledger.read_plan()}
    assert planned == {first.job_id(), second.job_id()}
    # A subset call must not shrink or reject the plan.
    ledger.write_plan([second])
    planned = {record["job_id"] for record in ledger.read_plan()}
    assert planned == {first.job_id(), second.job_id()}


def test_phase_plan_rejects_changed_per_job_record(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    ledger.write_plan([job])

    class _SameIdDifferentRecord:
        def job_id(self) -> str:
            return job.job_id()

        def plan_record(self) -> dict[str, Any]:
            return {**job.plan_record(), "job_type": "rewritten"}

    with pytest.raises(PlanMismatchError):
        ledger.write_plan([_SameIdDifferentRecord()])


# --------------------------------------------------------------------------
# 9. committed outcome integrity
# --------------------------------------------------------------------------


def _commit_result(
    ledger: PhaseLedger,
    job: ModelJob,
    result: JobResult,
    *,
    outcome: str | None = None,
    result_digest: str | None = None,
    legacy_result_artifact: bool = False,
) -> None:
    """Commit a result exactly like the scheduler does.

    ``result.json`` is bound by ``result_digest``; ``legacy_result_artifact``
    additionally lists it among ``artifact_digests``, which is what receipts
    written before that split look like.
    """
    digests: dict[str, str] = {}
    for name, payload in sorted(result.artifacts.items()):
        digests[name] = ledger.publish_artifact(job.job_id(), name, payload)
    digest = ledger.publish_result(job, result)
    if legacy_result_artifact:
        digests["result.json"] = digest
    ledger.commit(
        job,
        outcome=outcome or result.outcome,
        artifact_digests=digests,
        result_digest=digest if result_digest is None else result_digest,
        external_artifacts=result.external_artifacts,
    )


def test_missing_result_digest_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    result = JobResult(OUTCOME_COMPLETED, payload={"ok": True})
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"x")
    result_digest = ledger.publish_result(job, result)
    ledger.commit(
        job,
        outcome=OUTCOME_COMPLETED,
        artifact_digests={"out.png": digest, "result.json": result_digest},
        result_digest="",
    )
    # No migration path: a receipt without a durable result is not accepted.
    assert ledger.classify(job).state == STATE_MISMATCH


def test_terminal_reject_validates_result_digest(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    result = JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": True})
    _commit_result(ledger, job, result)
    assert ledger.classify(job).state == STATE_TERMINAL_REJECT
    # Tamper with the durable result, keeping it valid JSON.
    target = ledger.artifact_path(job.job_id(), "result.json")
    tampered = JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": False})
    target.write_text(json.dumps(tampered.durable(), sort_keys=True))
    assert ledger.classify(job).state == STATE_MISMATCH


def test_terminal_reject_result_outcome_mismatch_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    _commit_result(ledger, job, JobResult(OUTCOME_TERMINAL_REJECT, payload={}))
    receipt = ledger.receipts()[job.job_id()]
    # Re-commit with the same digests but a different outcome would be a lie.
    assert receipt["outcome"] == OUTCOME_TERMINAL_REJECT
    assert ledger.load_result(job) is not None
    assert ledger.load_result(job).outcome == OUTCOME_TERMINAL_REJECT


def test_external_artifact_is_verified_on_resume(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    payload_dir = tmp_path / "production"
    payload_dir.mkdir()
    external = payload_dir / "candidate.png"
    external.write_bytes(b"external-bytes")
    digest = hashlib.sha256(b"external-bytes").hexdigest()
    reference = ArtifactReference(
        path=str(external), sha256=digest, size=external.stat().st_size
    )
    job = _job()
    _commit_result(
        ledger, job, JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    )
    assert ledger.classify(job).state == STATE_COMPLETED


def test_missing_external_artifact_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    external = tmp_path / "gone.png"
    reference = ArtifactReference(
        path=str(external), sha256="0" * 64, size=None
    )
    job = _job()
    _commit_result(
        ledger, job, JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_external_artifact_size_mismatch_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    external = tmp_path / "candidate.png"
    external.write_bytes(b"bytes")
    reference = ArtifactReference(
        path=str(external), sha256=hashlib.sha256(b"bytes").hexdigest(), size=999
    )
    job = _job()
    _commit_result(
        ledger, job, JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_external_artifact_sha_mismatch_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    external = tmp_path / "candidate.png"
    external.write_bytes(b"bytes")
    reference = ArtifactReference(
        path=str(external), sha256="1" * 64, size=external.stat().st_size
    )
    job = _job()
    _commit_result(
        ledger, job, JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_relative_external_artifact_path_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    reference = ArtifactReference(path="relative/path.png", sha256="0" * 64)
    job = _job()
    _commit_result(
        ledger, job, JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_result_external_refs_must_match_receipt(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    external = tmp_path / "candidate.png"
    external.write_bytes(b"bytes")
    reference = ArtifactReference(
        path=str(external), sha256=hashlib.sha256(b"bytes").hexdigest()
    )
    job = _job()
    result = JobResult(OUTCOME_COMPLETED, external_artifacts=(reference,))
    digests = {
        "result.json": ledger.publish_result(job, result),
    }
    ledger.commit(
        job,
        outcome=OUTCOME_COMPLETED,
        artifact_digests=digests,
        result_digest=digests["result.json"],
        external_artifacts=(),  # receipt disagrees with result.json
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_scheduler_refuses_tampered_terminal_result(tmp_path: Path):
    job = _job(job_type="judge", resource=RESOURCE_QWEN)
    seen: list[dict] = []

    def finalize(j: ModelJob, result: JobResult):
        seen.append(dict(result.payload))
        return ()

    executor = _FakeBatchExecutor(
        {job.job_id(): JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": True})}
    )
    _scheduler(tmp_path, {RESOURCE_QWEN: executor}, finalize).run([job])
    assert seen == [{"rejected": True}]

    ledger = GroupLedger(tmp_path / "group")
    phase_id = ledger.phase_for(job)
    assert phase_id is not None
    target = ledger.phase(phase_id).artifact_path(job.job_id(), "result.json")
    tampered = JobResult(OUTCOME_TERMINAL_REJECT, payload={"rejected": False})
    target.write_text(json.dumps(tampered.durable(), sort_keys=True))

    seen.clear()
    with pytest.raises(SchedulerError):
        _scheduler(tmp_path, {RESOURCE_QWEN: executor}, finalize).run([job])
    # The finalizer must never observe the tampered payload.
    assert seen == []


# --------------------------------------------------------------------------
# 10. ledger lookup cost and torn-receipt counter
# --------------------------------------------------------------------------


def test_new_pending_jobs_do_not_rescan_artifact_dirs(tmp_path, monkeypatch):
    """1000 brand-new pending jobs must not scan phase dirs per job."""
    root = tmp_path / "group"
    ledger = GroupLedger(root)
    ledger.phase("r000-boogu").write_plan([_job()])
    jobs = [_job(clip_uid=f"clip-{i:05d}") for i in range(1000)]

    fresh = GroupLedger(root)
    fresh.refresh()
    scans = {"n": 0}
    real_iterdir = Path.iterdir

    def counting_iterdir(self):
        if self.name == "artifacts" or (self.parent.name == "phases"):
            scans["n"] += 1
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)
    for job in jobs:
        assert fresh.classify(job).state == STATE_PENDING
    assert scans["n"] == 0


def test_torn_receipt_counter_survives_refresh(tmp_path: Path):
    ledger = GroupLedger(tmp_path / "group")
    job = _job()
    ledger.phase("r000-boogu").commit(
        job, outcome=OUTCOME_TERMINAL_REJECT, artifact_digests={}
    )
    receipts = ledger.phase("r000-boogu").receipts_path
    with receipts.open("ab") as handle:
        handle.write(b'{"job_id":"torn","job_ide')

    fresh = GroupLedger(tmp_path / "group")
    fresh.refresh()
    assert fresh.torn_receipts_repaired == 1


# --------------------------------------------------------------------------
# 11. shared committed validation
# --------------------------------------------------------------------------


def test_terminal_reject_internal_artifact_tamper_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    result = JobResult(
        OUTCOME_TERMINAL_REJECT,
        {"diagnostic.json": b"original", "out.bin": b"bytes"},
        payload={"rejected": True},
    )
    _commit_result(ledger, job, result)
    assert ledger.classify(job).state == STATE_TERMINAL_REJECT
    # Tamper an internal artifact, leaving result.json untouched.
    ledger.publish_artifact(job.job_id(), "diagnostic.json", b"tampered")
    assert ledger.classify(job).state == STATE_MISMATCH


def test_terminal_reject_missing_internal_artifact_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    result = JobResult(
        OUTCOME_TERMINAL_REJECT,
        {"diagnostic.json": b"original"},
        payload={"rejected": True},
    )
    _commit_result(ledger, job, result)
    assert ledger.classify(job).state == STATE_TERMINAL_REJECT
    (ledger.artifacts_root / job.job_id() / "diagnostic.json").unlink()
    assert ledger.classify(job).state == STATE_MISMATCH


def test_external_artifact_filesystem_error_fails_closed(tmp_path, monkeypatch):
    """A filesystem race must fail closed, not be reported as a valid artifact."""
    external = tmp_path / "candidate.png"
    external.write_bytes(b"bytes")
    reference = ArtifactReference(
        path=str(external), sha256=hashlib.sha256(b"bytes").hexdigest()
    )
    assert verify_external_artifact(reference) is True

    def exploding_is_file(self):
        raise OSError("transient filesystem error")

    monkeypatch.setattr(Path, "is_file", exploding_is_file)
    assert verify_external_artifact(reference) is False


# --------------------------------------------------------------------------
# 12. finalizer once per invocation
# --------------------------------------------------------------------------


def test_failed_finalizer_runs_once_even_when_a_sibling_progresses(tmp_path: Path):
    """A finalizer failure is retried on restart, not inside the same run."""
    failing = _job(job_type="removal", clip_uid="clip-000001")
    healthy = _job(job_type="removal", clip_uid="clip-000002")
    executor = _FakeBatchExecutor()
    counts: dict[str, int] = {failing.job_id(): 0, healthy.job_id(): 0}

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        counts[job.job_id()] += 1
        if job.job_id() == failing.job_id():
            raise RuntimeError("finalizer boom")
        return ()

    # Commit both jobs with a finalizer that always succeeds.
    def ok_finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        return ()

    _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, ok_finalize).run(
        [failing, healthy]
    )
    assert executor.call_ids() == [failing.job_id(), healthy.job_id()]

    counts[failing.job_id()] = 0
    counts[healthy.job_id()] = 0
    outcome = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize).run(
        [failing, healthy]
    )
    # The healthy sibling resolved, so the scheduler had further rounds; the
    # failing finalizer must still be attempted exactly once.
    assert counts[failing.job_id()] == 1
    assert counts[healthy.job_id()] == 1
    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [failing.job_id()]
    # No model call was repeated for the committed jobs.
    assert executor.call_ids() == [failing.job_id(), healthy.job_id()]

    counts[failing.job_id()] = 0
    counts[healthy.job_id()] = 0
    restarted = _scheduler(tmp_path, {RESOURCE_BOOGU: executor}, finalize).run(
        [failing, healthy]
    )
    # Restart gives the failing finalizer exactly one new attempt. The healthy
    # sibling is committed too, so its finalizer is also replayed once; the
    # point is that neither is retried more than once per invocation.
    assert counts[failing.job_id()] == 1
    assert counts[healthy.job_id()] == 1
    assert restarted["completed"] is False
    assert executor.call_ids() == [failing.job_id(), healthy.job_id()]


# --------------------------------------------------------------------------
# 13. malformed existing phase plans fail closed
# --------------------------------------------------------------------------


def _write_plan_payload(ledger: PhaseLedger, payload: Any) -> None:
    ledger.root.mkdir(parents=True, exist_ok=True)
    ledger.plan_path.write_text(json.dumps(payload, sort_keys=True))


def _valid_plan_payload(job: ModelJob) -> dict[str, Any]:
    jobs = [job.plan_record()]
    return {
        "plan_hash": hashlib.sha256(
            json.dumps(jobs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "job_count": len(jobs),
        "jobs": jobs,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_hash",
        "bad_count",
        "jobs_not_list",
        "duplicate_job_id",
        "non_dict_record",
        "missing_job_id",
    ],
)
def test_malformed_existing_plan_fails_closed(tmp_path: Path, mutation: str):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    payload = _valid_plan_payload(job)
    if mutation == "bad_hash":
        payload["plan_hash"] = "0" * 64
    elif mutation == "bad_count":
        payload["job_count"] = 99
    elif mutation == "jobs_not_list":
        payload["jobs"] = {"not": "a list"}
    elif mutation == "duplicate_job_id":
        payload["jobs"] = [job.plan_record(), job.plan_record()]
        payload["job_count"] = 2
        payload["plan_hash"] = hashlib.sha256(
            json.dumps(
                payload["jobs"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    elif mutation == "non_dict_record":
        payload["jobs"] = ["not-a-dict"]
        payload["job_count"] = 1
    elif mutation == "missing_job_id":
        broken = dict(job.plan_record())
        broken.pop("job_id")
        payload["jobs"] = [broken]
        payload["job_count"] = 1
    _write_plan_payload(ledger, payload)
    with pytest.raises((PlanMismatchError, LedgerError)):
        ledger.write_plan([_job(clip_uid="clip-000002")])


def test_valid_plan_growth_still_passes(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    _write_plan_payload(ledger, _valid_plan_payload(job))
    other = _job(clip_uid="clip-000002")
    ledger.write_plan([other])
    planned = {record["job_id"] for record in ledger.read_plan()}
    assert planned == {job.job_id(), other.job_id()}


# --------------------------------------------------------------------------
# 12. result.json is bound once, by digest
# --------------------------------------------------------------------------


def _scheduled_receipt(tmp_path: Path, result: JobResult) -> dict[str, Any]:
    """Run one real scheduler pass and return the receipt it committed."""
    job = _job()
    ledger = GroupLedger(tmp_path / "group")
    executor = _FakeBatchExecutor({job.job_id(): result})
    _scheduler(
        tmp_path, {RESOURCE_BOOGU: executor}, lambda *a, **k: (), ledger=ledger
    ).run([job])
    return ledger.phase(f"r000-{RESOURCE_BOOGU}").receipts()[job.job_id()]


def test_scheduler_does_not_list_result_json_as_an_artifact(tmp_path: Path):
    """The durable result is bound by its own digest, not by an artifact digest."""
    result = JobResult(OUTCOME_COMPLETED, {"out.png": b"payload"}, payload={"ok": True})
    receipt = _scheduled_receipt(tmp_path, result)
    assert set(receipt["artifact_digests"]) == {"out.png"}
    assert receipt["result_digest"]
    assert "result.json" not in receipt["artifact_digests"]


def test_result_only_receipt_resumes_without_walking_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A job whose only committed artifact is its result never scans a directory."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    _commit_result(ledger, job, JobResult(OUTCOME_COMPLETED, payload={"ok": True}))

    monkeypatch.setattr(
        PhaseLedger,
        "artifact_digests",
        lambda *a, **k: pytest.fail("a result-only receipt walked the artifact dir"),
    )
    state = ledger.classify(job)
    assert state.state == STATE_COMPLETED
    assert state.skippable


def test_legacy_receipt_listing_result_json_still_resumes(tmp_path: Path):
    """Receipts written before the split keep resuming."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    _commit_result(
        ledger,
        job,
        JobResult(OUTCOME_COMPLETED, {"out.png": b"payload"}),
        legacy_result_artifact=True,
    )
    state = ledger.classify(job)
    assert state.state == STATE_COMPLETED
    assert state.skippable


def test_legacy_receipt_with_disagreeing_result_artifact_fails_closed(tmp_path: Path):
    """A legacy receipt that disagrees with its own result digest is corrupt."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    result = JobResult(OUTCOME_COMPLETED, {"out.png": b"payload"})
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    result_digest = ledger.publish_result(job, result)
    ledger.commit(
        job,
        outcome=OUTCOME_COMPLETED,
        artifact_digests={"out.png": digest, "result.json": "0" * 64},
        result_digest=result_digest,
    )
    assert ledger.classify(job).state == STATE_MISMATCH


def test_result_tamper_still_fails_closed(tmp_path: Path):
    """The result digest remains the authority for the result bytes."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    _commit_result(ledger, job, JobResult(OUTCOME_COMPLETED, payload={"ok": True}))
    assert ledger.classify(job).state == STATE_COMPLETED

    path = ledger.artifact_path(job.job_id(), "result.json")
    path.write_text(json.dumps({"outcome": "completed", "payload": {"ok": False}}))
    assert ledger.classify(job).state == STATE_MISMATCH


# --------------------------------------------------------------------------
# 13. multi-node / multi-GPU execution correctness (no model is ever run)
# --------------------------------------------------------------------------


def test_static_ownership_union_covers_every_group_exactly_once():
    """Under world_size 3 and 5 the union is the campaign and overlaps are empty."""
    groups = build_groups(_shards(24 * DEFAULT_GROUP_SIZE), campaign=CAMPAIGN)
    assert len(groups) == 24

    for world_size in (3, 5):
        owners: dict[int, list[int]] = {}
        for rank in range(world_size):
            for group in assigned_groups(groups, rank=rank, world_size=world_size):
                owners.setdefault(group.group_index, []).append(rank)
        # Union: every group is owned. Intersection: by exactly one rank.
        assert sorted(owners) == list(range(24)), world_size
        assert all(len(ranks) == 1 for ranks in owners.values()), owners


def test_job_identity_is_independent_of_slot_rank_and_world_size():
    """Placement is execution-only: one semantic job is always one identity."""
    plain = _job()
    for label in (
        "slot-0/gpu-0",
        "slot-7/gpu-5",
        "rank=0/world_size=3",
        "rank=2/world_size=5",
    ):
        placed = ModelJob.create(
            job_type=plain.job_type,
            resource=plain.resource,
            canonical_shard=plain.canonical_shard,
            clip_uid=plain.clip_uid,
            semantic_inputs={"prompt": "remove bg"},
            model_identity=plain.model_identity,
            label=label,
        )
        assert placed.identity() == plain.identity(), label
        assert placed.job_id() == plain.job_id(), label


def test_partial_group_restart_matches_an_uninterrupted_run(tmp_path: Path):
    """A restart re-pays every committed job zero times and converges identically."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    follow = _job(
        job_type="removal_judge", resource=RESOURCE_QWEN, clip_uid="clip-000001"
    )

    def finalize(job: ModelJob, result: JobResult):
        return (follow,) if job.job_id() == first.job_id() else ()

    def receipts_by_phase(root: Path) -> dict[str, Any]:
        ledger = GroupLedger(root)
        return {
            phase_id: ledger.phase(phase_id).receipts()
            for phase_id in ledger.phase_ids()
        }

    # Uninterrupted serial run: everything succeeds on the first pass.
    clean_root = tmp_path / "clean"
    clean_log: list[str] = []
    clean = _scheduler(
        clean_root,
        {
            RESOURCE_BOOGU: _OutcomeExecutor(clean_log),
            RESOURCE_QWEN: _OutcomeExecutor(clean_log),
        },
        finalize,
        ledger=GroupLedger(clean_root),
    )
    assert clean.run([first, second])["completed"] is True

    # Interrupted run: `second` fails retryably, so the group stays incomplete.
    partial_root = tmp_path / "partial"
    partial_log: list[str] = []
    interrupted = _scheduler(
        partial_root,
        {
            RESOURCE_BOOGU: _OutcomeExecutor(partial_log, {second.job_id()}),
            RESOURCE_QWEN: _OutcomeExecutor(partial_log),
        },
        finalize,
        ledger=GroupLedger(partial_root),
    )
    outcome = interrupted.run([first, second])
    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [second.job_id()]
    assert partial_log.count(second.job_id()) == 1

    # Restart with a brand-new ledger over the same root: only `second` is pending.
    restarted = _scheduler(
        partial_root,
        {
            RESOURCE_BOOGU: _OutcomeExecutor(partial_log),
            RESOURCE_QWEN: _OutcomeExecutor(partial_log),
        },
        finalize,
        ledger=GroupLedger(partial_root),
    )
    outcome = restarted.run([first, second])
    assert outcome["completed"] is True
    assert partial_log.count(first.job_id()) == 1, "a committed job is never re-paid"
    assert partial_log.count(follow.job_id()) == 1, "the finalizer replay is not re-paid"
    assert partial_log.count(second.job_id()) == 2, "only the pending job runs again"

    # The durable outcome converges on the uninterrupted one, job for job.
    assert receipts_by_phase(partial_root) == receipts_by_phase(clean_root)


# --------------------------------------------------------------------------
# 14. resume read cost: one scan at most, one result read at most
# --------------------------------------------------------------------------


def _count_resume_io(monkeypatch, ledger, job):
    """Count artifact_digests() calls and per-file reads for one job's dir."""
    seen: dict[str, list[Any]] = {"digests_calls": [], "reads": []}
    real_digests = PhaseLedger.artifact_digests

    def counting_digests(self, job_id, **kwargs):
        seen["digests_calls"].append(kwargs.get("include_result", True))
        return real_digests(self, job_id, **kwargs)

    monkeypatch.setattr(PhaseLedger, "artifact_digests", counting_digests)
    root = ledger.artifacts_root / job.job_id()
    real_read_bytes = Path.read_bytes

    def counting_read(self, *args, **kwargs):
        if str(self).startswith(str(root)):
            seen["reads"].append(Path(self).name)
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    return seen


def test_result_only_resume_reads_the_result_once_and_never_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A result-only job pays for exactly one read: its own result."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="judge")
    _commit_result(ledger, job, JobResult(OUTCOME_COMPLETED, payload={"ok": True}))

    seen = _count_resume_io(monkeypatch, ledger, job)
    state = ledger.classify(job)

    assert state.state == STATE_COMPLETED
    assert seen["digests_calls"] == [], "a result-only job must not scan its directory"
    assert seen["reads"] == ["result.json"], seen


def test_binary_artifact_resume_scans_once_and_reads_the_result_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """One directory scan, each artifact once, and the result read exactly once."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    result = JobResult(
        OUTCOME_COMPLETED, {"a.bin": b"a", "b.bin": b"b"}, payload={"ok": True}
    )
    _commit_result(ledger, job, result)

    seen = _count_resume_io(monkeypatch, ledger, job)
    state = ledger.classify(job)

    assert state.state == STATE_COMPLETED
    assert seen["digests_calls"] == [False], "exactly one scan, result excluded"
    assert sorted(seen["reads"]) == ["a.bin", "b.bin", "result.json"], seen


def test_unexpected_extra_binary_artifact_fails_closed(tmp_path: Path):
    """The binary artifact set is exact: an unlisted file is corruption."""
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    _commit_result(ledger, job, JobResult(OUTCOME_COMPLETED, {"a.bin": b"a"}))
    assert ledger.classify(job).state == STATE_COMPLETED

    ledger.publish_artifact(job.job_id(), "extra.bin", b"surprise")

    state = ledger.classify(job)
    assert state.state == STATE_MISMATCH
    assert not state.skippable


# --------------------------------------------------------------------------
# 15. completion-driven streaming refill
# --------------------------------------------------------------------------


class _ScriptedExecutor:
    """A completion-driven executor the test drives, with no sleeping.

    ``completions`` is one group of job ids per ``collect`` call; when the script
    runs out, everything still in flight settles. Nothing here depends on
    wall-clock timing, so the assertions say *what* completed when rather than
    racing a timeout.
    """

    def __init__(
        self,
        *,
        capacity: int,
        completions: Sequence[Sequence[str]] = (),
        failures: Sequence[str] = (),
        fatal: BaseException | None = None,
    ) -> None:
        self._capacity = capacity
        self._completions = [list(group) for group in completions]
        self._failures = set(failures)
        self._fatal = fatal
        self.inflight: list[Any] = []
        self.submit_order: list[str] = []
        self.collect_calls = 0
        self.closed = False

    def capacity(self) -> int:
        return self._capacity

    def submit(self, job: Any) -> None:
        self.submit_order.append(job.job_id())
        self.inflight.append(job)

    def collect(self) -> list[JobExecution]:
        self.collect_calls += 1
        group = self._completions.pop(0) if self._completions else None
        wanted = (
            {job.job_id() for job in self.inflight} if group is None else set(group)
        )
        done = [job for job in self.inflight if job.job_id() in wanted]
        self.inflight = [job for job in self.inflight if job.job_id() not in wanted]
        executions = [
            JobExecution(
                job,
                None
                if job.job_id() in self._failures
                else JobResult(OUTCOME_COMPLETED, payload={"ok": True}),
                RuntimeError("model down") if job.job_id() in self._failures else None,
            )
            for job in done
        ]
        if self._fatal is not None:
            raise self._fatal
        return executions

    def close(self) -> None:
        self.closed = True


def _stream_scheduler(tmp_path: Path, executor: Any, finalize: Any) -> Any:
    return ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "group"),
        finalize=finalize,
        executors={RESOURCE_BOOGU: executor, RESOURCE_QWEN: executor},
    )


def test_global_ready_frontier_and_execution_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    jobs = [
        _job(clip_uid=f"clip-{index:06d}", resource=RESOURCE_SAM)
        for index in range(1000)
    ]
    executor = _ScriptedExecutor(
        capacity=8, completions=[[jobs[0].job_id()]]
    )
    writes = []
    original_write_plan = PhaseLedger.write_plan

    def counted_write_plan(self, current_jobs):
        writes.append(len(current_jobs))
        return original_write_plan(self, current_jobs)

    monkeypatch.setattr(PhaseLedger, "write_plan", counted_write_plan)
    original_submit = executor.submit

    def checked_submit(job):
        assert writes == [1000], "the full ready frontier must be planned first"
        original_submit(job)

    monkeypatch.setattr(executor, "submit", checked_submit)
    outcome = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "frontier"),
        finalize=lambda job, result: (),
        executors={RESOURCE_SAM: executor},
    ).run(jobs)

    counters = outcome["diagnostics"]["resources"][RESOURCE_SAM]
    assert outcome["completed"] is True
    assert counters["executor_capacity"] == 8
    assert counters["peak_ready_jobs"] == 1000
    assert counters["peak_inflight"] == 8
    assert counters["refill_count"] >= 1
    assert counters["jobs_per_epoch"] == [1000]
    assert executor.submit_order[8] == jobs[8].job_id()
    assert writes == [1000], "ready work is planned once, not rehashed per refill"


def test_independent_completion_chains_fill_each_resource_without_speculation(
    tmp_path: Path,
):
    clips = [f"clip-{index:06d}" for index in range(32)]
    generations = [
        _job(job_type="attribute_completion_generate", clip_uid=uid)
        for uid in clips
    ]
    postchecks = {
        uid: _job(job_type="attribute_completion_postcheck", resource=RESOURCE_SAM,
                  clip_uid=uid)
        for uid in clips
    }
    reviews = {
        uid: _job(job_type="attribute_completion_review", resource=RESOURCE_QWEN,
                  clip_uid=uid)
        for uid in clips
    }
    boogu = _ScriptedExecutor(capacity=8)
    sam = _ScriptedExecutor(capacity=8)
    qwen = _ScriptedExecutor(capacity=32)

    def finalize(job: ModelJob, result: JobResult):
        if job.resource == RESOURCE_BOOGU:
            return (postchecks[job.clip_uid],)
        if job.resource == RESOURCE_SAM:
            return (reviews[job.clip_uid],)
        return ()

    outcome = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "completion"),
        finalize=finalize,
        executors={
            RESOURCE_BOOGU: boogu,
            RESOURCE_SAM: sam,
            RESOURCE_QWEN: qwen,
        },
    ).run(generations)
    assert outcome["completed"] is True
    assert outcome["job_count"] == 96
    assert len(boogu.submit_order) == len(sam.submit_order) == len(qwen.submit_order) == 32
    for resource, capacity in ((RESOURCE_BOOGU, 8), (RESOURCE_SAM, 8),
                               (RESOURCE_QWEN, 32)):
        counters = outcome["diagnostics"]["resources"][resource]
        assert counters["peak_ready_jobs"] == 32
        assert counters["peak_inflight"] == capacity
        assert counters["jobs_per_epoch"] == [32]


def test_streaming_refills_a_freed_slot_before_slow_siblings_settle(tmp_path: Path):
    """case 1: a job unlocked by a completion takes the slot that just freed."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    unlocked = _job(job_type="removal", clip_uid="clip-000003")

    def finalize(job: ModelJob, result: JobResult):
        return (unlocked,) if job.job_id() == first.job_id() else ()

    executor = _ScriptedExecutor(
        capacity=2, completions=[[first.job_id()], [second.job_id()]]
    )
    outcome = _stream_scheduler(tmp_path, executor, finalize).run([first, second])

    assert outcome["completed"] is True
    assert executor.submit_order[0] == first.job_id()
    assert executor.submit_order[1] == second.job_id()
    # Unlocked by first's finalizer, submitted while second was still in flight.
    assert executor.submit_order[2] == unlocked.job_id()


def test_streaming_does_not_switch_resource_while_current_is_still_busy(tmp_path: Path):
    """case 2: an unlock of another resource waits for the current drain."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    other = _job(
        job_type="sam_review", resource=RESOURCE_QWEN, clip_uid="clip-000003"
    )

    def finalize(job: ModelJob, result: JobResult):
        return (other,) if job.job_id() == first.job_id() else ()

    executor = _ScriptedExecutor(
        capacity=2, completions=[[first.job_id()], [second.job_id()]]
    )
    scheduler = _stream_scheduler(tmp_path, executor, finalize)
    outcome = scheduler.run([first, second])

    assert outcome["completed"] is True
    # second is still in flight when `other` is unlocked, so `other` cannot be
    # submitted yet: the current resource drains to its fixed point first.
    assert executor.submit_order == [
        first.job_id(),
        second.job_id(),
        other.job_id(),
    ]
    assert scheduler.diagnostics.resource_switches == 1


def test_streaming_replays_a_durable_receipt_and_refills_from_its_finalizer(
    tmp_path: Path,
):
    """case 3: committed job skips the model but still unlocks streaming work."""
    committed = _job(job_type="removal", clip_uid="clip-000001")
    pending_job = _job(job_type="removal", clip_uid="clip-000002")
    unlocked = _job(job_type="removal", clip_uid="clip-000003")

    ledger = GroupLedger(tmp_path / "group")
    phase = ledger.phase("r000-boogu")
    phase.write_plan([committed])
    result = JobResult(OUTCOME_COMPLETED, payload={"ok": True})
    digest = phase.publish_result(committed, result)
    phase.commit(
        committed, outcome=OUTCOME_COMPLETED, artifact_digests={}, result_digest=digest
    )

    def finalize(job: ModelJob, result: JobResult):
        return (unlocked,) if job.job_id() == committed.job_id() else ()

    executor = _ScriptedExecutor(capacity=2, completions=[[pending_job.job_id()]])
    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=finalize,
        executors={RESOURCE_BOOGU: executor, RESOURCE_QWEN: executor},
    )
    outcome = scheduler.run([committed, pending_job])

    assert outcome["completed"] is True
    # The committed job never reached the executor...
    assert committed.job_id() not in executor.submit_order
    # ...but its finalizer was replayed, and the job it unlocked was submitted.
    assert scheduler.diagnostics.resume["finalizers_replayed"] >= 1
    assert unlocked.job_id() in executor.submit_order
    assert pending_job.job_id() in executor.submit_order


def test_streaming_isolates_one_model_exception_from_its_siblings(tmp_path: Path):
    """case 4: a raising job fails alone and its sibling still commits."""
    failing = _job(job_type="removal", clip_uid="clip-000001")
    healthy = _job(job_type="removal", clip_uid="clip-000002")

    executor = _ScriptedExecutor(capacity=2, failures=[failing.job_id()])
    scheduler = _stream_scheduler(tmp_path, executor, lambda job, result: ())
    outcome = scheduler.run([failing, healthy])

    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [failing.job_id()]
    counters = scheduler.diagnostics.resource(RESOURCE_BOOGU)
    assert counters["jobs_retryable_failed"] == 1
    assert counters["jobs_executed"] == 2


def test_streaming_propagates_control_flow_exceptions(tmp_path: Path):
    """case 5: KeyboardInterrupt is not swallowed as a per-job failure."""
    job = _job(job_type="removal", clip_uid="clip-000001")
    executor = _ScriptedExecutor(capacity=1, fatal=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _stream_scheduler(tmp_path, executor, lambda job, result: ()).run([job])


def test_streaming_keeps_one_resource_loaded_while_it_keeps_unlocking(tmp_path: Path):
    """case 6: a long chain of same-resource unlocks never thrashes the load."""
    chain = [
        _job(job_type="removal", clip_uid=f"clip-{index:03d}") for index in range(6)
    ]
    by_id = {job.job_id(): job for job in chain}

    def finalize(job: ModelJob, result: JobResult):
        position = [item.job_id() for item in chain].index(job.job_id())
        if position + 1 < len(chain):
            return (chain[position + 1],)
        return ()

    executor = _ScriptedExecutor(capacity=2)
    scheduler = _stream_scheduler(tmp_path, executor, finalize)
    outcome = scheduler.run([chain[0]])

    assert outcome["completed"] is True
    assert len(executor.submit_order) == len(chain)
    assert set(executor.submit_order) == set(by_id)
    # Every job unlocked the next one and nothing else was resident, so the
    # resource was entered exactly once.
    assert scheduler.diagnostics.resource_switches == 0
    assert scheduler.diagnostics.resource(RESOURCE_BOOGU)["epoch_count"] == 1


def _recording_executor(capacity: int, completions: Sequence[Sequence[str]], events: list[str]) -> Any:
    """A scripted executor that logs submits and collects into ``events``."""
    executor = _ScriptedExecutor(capacity=capacity, completions=completions)
    real_submit = executor.submit
    real_collect = executor.collect

    def submit(job: Any) -> None:
        events.append(f"submit:{job.clip_uid}")
        real_submit(job)

    def collect() -> list[JobExecution]:
        events.append("collect")
        return real_collect()

    executor.submit = submit  # type: ignore[method-assign]
    executor.collect = collect  # type: ignore[method-assign]
    return executor


def test_streaming_submits_ready_work_before_the_heavy_finalizer(
    tmp_path: Path,
) -> None:
    """Already-ready work is submitted before this thread runs a finalizer."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    third = _job(job_type="removal", clip_uid="clip-000003")

    events: list[str] = []
    executor = _recording_executor(2, [[first.job_id()]], events)

    def finalize(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        events.append(f"settle:{job.clip_uid}")
        return ()

    outcome = _stream_scheduler(tmp_path, executor, finalize).run(
        [first, second, third]
    )

    assert outcome["completed"] is True
    # A finished slot is refilled from the ready set before any CPU finalizer.
    assert events.index(f"submit:{third.clip_uid}") < events.index(
        f"settle:{first.clip_uid}"
    )


def test_streaming_submits_unlocked_work_before_the_next_sibling_finalizer(
    tmp_path: Path,
) -> None:
    """One finalizer's unlock is submitted before the next sibling is settled."""
    first = _job(job_type="removal", clip_uid="clip-000001")
    second = _job(job_type="removal", clip_uid="clip-000002")
    unlocked = _job(job_type="removal", clip_uid="clip-000003")

    events: list[str] = []
    executor = _recording_executor(
        2, [[first.job_id(), second.job_id()]], events
    )

    def finalize(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        events.append(f"settle:{job.clip_uid}")
        return (unlocked,) if job.job_id() == first.job_id() else ()

    outcome = _stream_scheduler(tmp_path, executor, finalize).run([first, second])

    assert outcome["completed"] is True
    assert events.index(f"submit:{unlocked.clip_uid}") < events.index(
        f"settle:{second.clip_uid}"
    )
