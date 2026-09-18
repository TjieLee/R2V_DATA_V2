"""Resource-epoch execution substrate: identity, durability and scheduling.

These tests cover the new ``resource_epoch_v3`` substrate only. They never
construct a real model, GPU or endpoint, and they do not touch the frozen
semantic modules. Semantic equivalence against the existing scheduler is a
separate, server-side validation and is explicitly *not* claimed here.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    EpochDiagnostics,
    JobResult,
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
    semantic: object = None,
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


class _FakeExecutor:
    """Deterministic fake resource; records every call it is asked to make."""

    def __init__(self, resource: str, outcomes: dict[str, JobResult] | None = None):
        self.resource = resource
        self.calls: list[ModelJob] = []
        self.outcomes = outcomes or {}

    def execute(self, job: ModelJob) -> JobResult:
        self.calls.append(job)
        return self.outcomes.get(job.job_id(), JobResult(OUTCOME_COMPLETED, {"out.png": b"ok"}))


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
    # Static ownership must not leak into identity either.
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


# --------------------------------------------------------------------------
# 3. durable ledger: crash injection and resume
# --------------------------------------------------------------------------


def test_plan_is_immutable_and_detects_change(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    jobs = [_job(), _job(clip_uid="clip-000002")]
    digest = ledger.write_plan(jobs)
    assert ledger.write_plan(jobs) == digest
    with pytest.raises(PlanMismatchError):
        ledger.write_plan([jobs[0]])


def test_missing_plan_fails_closed(tmp_path: Path):
    with pytest.raises(LedgerError):
        PhaseLedger(tmp_path / "absent").read_plan()


def test_pending_job_has_no_receipt(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    assert ledger.classify(_job()).state == STATE_PENDING


def test_completed_receipt_with_matching_artifacts_skips_model_call(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    ledger.commit(job, outcome=OUTCOME_COMPLETED, artifact_digests={"out.png": digest})
    state = ledger.classify(job)
    assert state.state == STATE_COMPLETED
    assert state.skippable


def test_artifact_without_receipt_is_treated_as_incomplete(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    # Crash between artifact rename and receipt append.
    assert ledger.classify(job).state == STATE_RERUN_NO_RECEIPT
    assert not ledger.classify(job).skippable


def test_terminal_reject_is_durable_and_never_paid_again(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job(job_type="removal_judge")
    ledger.commit(job, outcome=OUTCOME_TERMINAL_REJECT, artifact_digests={})
    state = ledger.classify(job)
    assert state.state == STATE_TERMINAL_REJECT
    assert state.skippable


def test_retryable_failure_leaves_no_successful_receipt(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    # A retryable exception must not append a receipt at all.
    assert ledger.classify(job).state == STATE_PENDING


def test_digest_mismatch_fails_closed(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job()
    digest = ledger.publish_artifact(job.job_id(), "out.png", b"payload")
    ledger.commit(job, outcome=OUTCOME_COMPLETED, artifact_digests={"out.png": digest})
    ledger.publish_artifact(job.job_id(), "out.png", b"tampered")
    state = ledger.classify(job)
    assert state.state == STATE_MISMATCH
    assert not state.skippable


def test_torn_final_receipt_line_is_repaired_and_earlier_lines_kept(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    first = _job(clip_uid="clip-000001")
    ledger.commit(first, outcome=OUTCOME_COMPLETED, artifact_digests={})
    ledger.receipts_path.write_bytes(
        ledger.receipts_path.read_bytes()
        + b'{"job_id":"torn","job_ide'
    )
    loaded, was_repaired = _load_with_repair(ledger)
    assert was_repaired is True
    assert [r["job_id"] for r in loaded] == [first.job_id()]
    # The surviving receipt is still usable after repair.
    assert ledger.classify(first).state == STATE_COMPLETED


def _load_with_repair(ledger: PhaseLedger):
    from r2v_data_v2.v3.post_mask_epoch_state import _load_jsonl

    return _load_jsonl(ledger.receipts_path)


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
    group.phase("r000-booqu").commit(
        job, outcome=OUTCOME_TERMINAL_REJECT, artifact_digests={}
    )
    state = group.classify(job)
    assert state.state == STATE_TERMINAL_REJECT


def test_group_ledger_reports_artifact_without_receipt(tmp_path: Path):
    group = GroupLedger(tmp_path / "group")
    job = _job()
    group.phase("r000-boogu").publish_artifact(job.job_id(), "out.png", b"x")
    assert group.classify(job).state == STATE_RERUN_NO_RECEIPT


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
# 5. scheduler: no speculative calls, fixed point, deterministic switching
# --------------------------------------------------------------------------


def _scheduler(tmp_path: Path, executors: dict, finalize):
    return ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "group"),
        executors=executors,
        finalize=finalize,
    )


def test_scheduler_runs_ready_jobs_and_commits_receipts(tmp_path: Path):
    boogu = _FakeExecutor(RESOURCE_BOOGU)
    scheduler = _scheduler(tmp_path, {RESOURCE_BOOGU: boogu}, lambda job, result: ())
    job = _job()
    outcome = scheduler.run([job])
    assert outcome["completed"] is True
    assert [j.job_id() for j in boogu.calls] == [job.job_id()]
    assert GroupLedger(tmp_path / "group").classify(job).state == STATE_COMPLETED


def test_scheduler_never_creates_conditional_second_attempt_speculatively(tmp_path: Path):
    boogu = _FakeExecutor(RESOURCE_BOOGU)
    qwen = _FakeExecutor(RESOURCE_QWEN)
    second = _job(job_type="background_removal_candidate2", attempt_index=1, seed=2)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        # Existing policy accepted candidate1, so candidate2 is never unlocked.
        return ()

    scheduler = _scheduler(
        tmp_path, {RESOURCE_BOOGU: boogu, RESOURCE_QWEN: qwen}, finalize
    )
    scheduler.run([_job()])
    assert [j.job_id() for j in boogu.calls] == [_job().job_id()]
    assert qwen.calls == []
    assert second.job_id() not in {j.job_id() for j in boogu.calls}


def test_conditional_second_attempt_runs_only_when_policy_unlocks_it(tmp_path: Path):
    boogu = _FakeExecutor(RESOURCE_BOOGU)
    second = _job(job_type="background_removal_candidate2", attempt_index=1, seed=2)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        return (second,) if job.attempt_index == 0 else ()

    scheduler = _scheduler(tmp_path, {RESOURCE_BOOGU: boogu}, finalize)
    scheduler.run([_job()])
    assert len(boogu.calls) == 2
    assert boogu.calls[1].job_id() == second.job_id()


def test_fixed_point_keeps_same_resource_before_switching(tmp_path: Path):
    qwen = _FakeExecutor(RESOURCE_QWEN)
    boogu = _FakeExecutor(RESOURCE_BOOGU)
    second_qwen = _job(job_type="removal_judge2", resource=RESOURCE_QWEN, attempt_index=1)

    def finalize(job: ModelJob, result: JobResult) -> tuple[ModelJob, ...]:
        return (second_qwen,) if job.resource == RESOURCE_BOOGU else ()

    scheduler = _scheduler(
        tmp_path, {RESOURCE_QWEN: qwen, RESOURCE_BOOGU: boogu}, finalize
    )
    outcome = scheduler.run([_job(resource=RESOURCE_BOOGU)])
    assert outcome["completed"] is True
    # Boogu first (only ready work), then both Qwen jobs in one epoch.
    assert len(boogu.calls) == 1
    assert len(qwen.calls) == 1
    assert scheduler.diagnostics.resource_switches == 1


def test_resource_selection_is_deterministic_across_runs(tmp_path: Path):
    def run(root: Path) -> list[str]:
        executors = {r: _FakeExecutor(r) for r in (RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM)}
        jobs = [
            _job(resource=RESOURCE_SAM, job_type="sam_review", clip_uid="clip-000001"),
            _job(resource=RESOURCE_QWEN, job_type="qwen_judge", clip_uid="clip-000002"),
            _job(resource=RESOURCE_BOOGU, job_type="boogu_edit", clip_uid="clip-000003"),
        ]
        scheduler = _scheduler(root, executors, lambda job, result: ())
        scheduler.run(jobs)
        return [phase for phase in GroupLedger(root / "group").phase_ids()]

    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_completed_jobs_are_skipped_on_second_run(tmp_path: Path):
    executors = {RESOURCE_BOOGU: _FakeExecutor(RESOURCE_BOOGU)}
    job = _job()
    first = _scheduler(tmp_path, executors, lambda job_, result: ())
    first.run([job])
    second = _scheduler(tmp_path, executors, lambda job_, result: ())
    outcome = second.run([job])
    assert outcome["completed"] is True
    assert len(executors[RESOURCE_BOOGU].calls) == 1
    assert second.diagnostics.resume["receipts_reused"] >= 1


def test_retryable_job_is_retried_on_next_run(tmp_path: Path):
    job = _job()
    executors = {
        RESOURCE_BOOGU: _FakeExecutor(
            RESOURCE_BOOGU, {job.job_id(): JobResult(OUTCOME_RETRYABLE_FAILED)}
        )
    }
    outcome = _scheduler(tmp_path, executors, lambda j, r: ()).run([job])
    assert outcome["completed"] is False
    assert outcome["unresolved_job_ids"] == [job.job_id()]
    # No receipt was written, so resume must pay for the call again.
    assert GroupLedger(tmp_path / "group").classify(job).state == STATE_PENDING


def test_scheduler_fails_closed_on_receipt_mismatch(tmp_path: Path):
    job = _job()
    ledger = GroupLedger(tmp_path / "group")
    executors = {RESOURCE_BOOGU: _FakeExecutor(RESOURCE_BOOGU)}
    _scheduler(tmp_path, executors, lambda j, r: ()).run([job])
    # Simulate a tampered artifact after a durable commit.
    ledger.phase(ledger.phase_ids()[0]).publish_artifact(
        job.job_id(), "out.png", b"tampered"
    )
    with pytest.raises(SchedulerError):
        _scheduler(tmp_path, executors, lambda j, r: ()).run([job])


def test_terminal_reject_is_not_paid_again_on_resume(tmp_path: Path):
    job = _job(job_type="removal_judge", resource=RESOURCE_QWEN)
    executors = {
        RESOURCE_QWEN: _FakeExecutor(
            RESOURCE_QWEN, {job.job_id(): JobResult(OUTCOME_TERMINAL_REJECT)}
        )
    }
    _scheduler(tmp_path, executors, lambda j, r: ()).run([job])
    again = _scheduler(tmp_path, executors, lambda j, r: ())
    outcome = again.run([job])
    assert outcome["completed"] is True
    assert len(executors[RESOURCE_QWEN].calls) == 1


def test_diagnostics_report_required_counters(tmp_path: Path):
    executors = {RESOURCE_BOOGU: _FakeExecutor(RESOURCE_BOOGU)}
    scheduler = _scheduler(tmp_path, executors, lambda j, r: ())
    scheduler.diagnostics.group_count = 48
    outcome = scheduler.run([_job()])
    summary = outcome["diagnostics"]
    for key in (
        "group_count",
        "group_completed",
        "group_incomplete",
        "phase_count",
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
            executors={"flux": _FakeExecutor("flux")},
            finalize=lambda j, r: (),
        )


def test_scheduler_fails_clearly_when_a_ready_resource_has_no_executor(tmp_path: Path):
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "g"),
        executors={RESOURCE_QWEN: _FakeExecutor(RESOURCE_QWEN)},
        finalize=lambda j, r: (),
    )
    with pytest.raises(SchedulerError, match="no executor configured"):
        scheduler.run([_job(resource=RESOURCE_SAM)])
