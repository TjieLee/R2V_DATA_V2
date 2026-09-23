"""Durable primary PairStats reconciliation (4c1).

The evidence here is the *legacy* ``pair_clips()`` PairStats for the same
deterministic fixture, not hand-written magic numbers: the epoch primary pass
is driven on one storage and legacy on an independent one, and the 4c1 fields
must agree.

Scope of this commit:

* reconciled: processed, skipped_existing, skipped_not_ready, failed,
  ready/rejected, entities_ready/rejected, backgrounds_bound, repaired and the
  four primary background-final-guard counters;
* NOT reconciled yet (4c2): ``cross_pair_*`` and ``prefilter_*``;
* completion_* stay 0 because the epoch Pair path requires
  reference_edit.enabled, so the legacy completion fallbacks never run.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.config import PairConfig, V3Config
from r2v_data_v2.v3.pair import PairStats, pair_clips
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from tests.test_v3_pair import (
    _config,
    _decision,
    _install_clean_background,
    _Judge,
    _storage,
)

SHARD = "shard-000000000-000000000"

#: 4c1 fields, i.e. the ones this commit claims to reconcile.
RECONCILED = (
    "processed",
    "skipped_existing",
    "skipped_not_ready",
    "failed",
    "ready",
    "rejected",
    "entities_ready",
    "entities_rejected",
    "backgrounds_bound",
    "repaired",
    "background_final_guard_attempted",
    "background_final_guard_accepted",
    "background_final_guard_rejected",
    "background_final_guard_failed_closed",
)
NOT_RECONCILED_YET = (
    "cross_pair_attempted",
    "cross_pair_ready",
    "cross_pair_repaired",
    "prefilter_candidates_examined",
    "prefilter_candidates_filtered",
)


def _epoch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str, **pair_kwargs: Any) -> V3Config:
    base = _config(tmp_path, monkeypatch, pair=PairConfig(**pair_kwargs), reference_edit_enabled=True)
    return replace(base, run_root=base.run_root.parent / run_name)


def _legacy_pair(config: V3Config, storage: Any) -> PairStats:
    return pair_clips(
        config,
        storage,
        judge=_Judge(),
        cross_pair_judge=None,
        background_final_judge=None,
    )


def _epoch_pair(config: V3Config, storage: Any, ledger_root: Path, judge: Any = None) -> Any:
    """Drive the epoch primary pass to completion with fake Qwen decisions."""
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    runner = PairEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(ledger_root),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    _drain_primary(runner, judge or _Judge())
    return runner


def _drain_primary(runner: Any, judge: Any) -> None:
    """Run every seeded primary job, committing like the scheduler does."""
    from r2v_data_v2.v3.post_mask_epoch_pair import PAIR_PHASE

    pending = list(runner.seed_primary_jobs())
    guard = 0
    while pending:
        guard += 1
        if guard > 32:
            raise AssertionError("primary drain did not terminate")
        job = pending.pop(0)
        result = runner.run(job, judge)
        if not result.committed:
            break
        phase = runner.ledger.phase(PAIR_PHASE)
        digest = phase.publish_result(job, result)
        receipt = phase.commit(
            job,
            outcome=result.outcome,
            artifact_digests={"result.json":digest},
            result_digest=digest,
        )
        runner.ledger.note_commit(PAIR_PHASE, receipt)
        pending.extend(runner.finalize(job, result))


def test_primary_ready_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_config = _epoch_config(tmp_path, monkeypatch, "run-legacy")
    epoch_config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))

    legacy = _legacy_pair(legacy_config, legacy_storage)
    runner = _epoch_pair(epoch_config, epoch_storage, tmp_path / "ledger")
    epoch = runner.reconcile_primary_stats(SHARD)

    for field in RECONCILED:
        assert getattr(epoch, field) == getattr(legacy, field), field
    assert epoch.processed == 1
    assert epoch.ready == 1
    assert epoch.entities_ready == 1


def test_primary_rejected_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Rejecting(_Judge):
        def decide(self, **kwargs: Any) -> Any:
            return super().decide(**kwargs)

    legacy_config = _epoch_config(tmp_path, monkeypatch, "run-legacy")
    epoch_config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    # A rejected entity makes the whole clip pairing rejected.
    for storage in (legacy_storage, epoch_storage):
        _install_clean_background(storage)

    legacy = pair_clips(
        legacy_config, legacy_storage, judge=_Judge(scopes={"e1":"reject"})
    )
    runner = _epoch_pair(
        epoch_config, epoch_storage, tmp_path / "ledger",
        _Judge(scopes={"e1":"reject"}),
    )
    epoch = runner.reconcile_primary_stats(SHARD)
    for field in RECONCILED:
        assert getattr(epoch, field) == getattr(legacy, field), field
    assert epoch.rejected == 1


def test_mixed_entities_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_config = _epoch_config(tmp_path, monkeypatch, "run-legacy")
    epoch_config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    legacy_storage = _storage(legacy_config, entity_types=("subject", "object"))
    epoch_storage = _storage(epoch_config, entity_types=("subject", "object"))
    scopes = {"e1":"full", "e2":"reject"}

    legacy = pair_clips(legacy_config, legacy_storage, judge=_Judge(scopes=scopes))
    runner = _epoch_pair(
        epoch_config, epoch_storage, tmp_path / "ledger", _Judge(scopes=scopes)
    )
    epoch = runner.reconcile_primary_stats(SHARD)
    assert (epoch.entities_ready, epoch.entities_rejected) == (
        legacy.entities_ready, legacy.entities_rejected)
    assert epoch.entities_ready == 1 and epoch.entities_rejected == 1


def test_existing_pairing_is_skipped_not_processed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    # Produce a valid existing pairing first.
    _epoch_pair(config, storage, tmp_path / "ledger")
    assert storage.read_clip("clip-1").pairing is not None

    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    fresh = PairEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger2"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    fresh.seed_primary_jobs()  # freezes the classification this launch saw
    stats = fresh.reconcile_primary_stats(SHARD)
    assert stats.skipped_existing == 1
    assert stats.processed == 0
    assert stats.failed == 0


def test_ineligible_coverage_is_skipped_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Annotation/coverage ineligible: legacy skips, so must the epoch."""
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    storage.clear_coverage("clip-1")
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    runner = PairEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    runner.seed_primary_jobs()  # freezes the classification this launch saw
    stats = runner.reconcile_primary_stats(SHARD)
    assert stats.skipped_not_ready == 1
    assert stats.processed == 0


def test_invalid_pair_inputs_are_skipped_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh target whose pre-Pair inputs are invalid is skipped, not processed."""
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    # Drop the sampled frames: the clip is still classified fresh by the plan,
    # but its Pair inputs no longer validate.
    manifest = storage.frames_manifest_path("clip-1")
    manifest.unlink()
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    runner = PairEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    runner.seed_primary_jobs()  # freezes the classification this launch saw
    stats = runner.reconcile_primary_stats(SHARD)
    assert stats.skipped_not_ready == 1
    assert stats.processed == 0


def test_completion_and_unreconciled_fields_stay_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    stats = runner.reconcile_primary_stats(SHARD)
    assert (stats.completion_attempted, stats.completion_ready, stats.completion_rejected) == (0, 0, 0)
    for field in NOT_RECONCILED_YET:
        assert getattr(stats, field) == 0, field


def test_reconciliation_is_restart_stable_and_pure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    first = runner.reconcile_primary_stats(SHARD).to_dict()
    before = _snapshot(tmp_path / "ledger")

    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    fresh = PairEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    second = fresh.reconcile_primary_stats(SHARD).to_dict()
    assert second == first
    # Pure observation: reconciliation added no receipt, semantic file or
    # failure diagnostic, and called no model.
    assert _snapshot(tmp_path / "ledger") == before


def _snapshot(root: Path) -> tuple[str, ...]:
    return tuple(sorted(str(path) for path in root.rglob("*") if path.is_file()))


def test_repaired_counts_one_per_decision_and_not_the_raw_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """repaired is 1 per repaired decision, and replay must not double count."""
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    first = runner.reconcile_primary_stats(SHARD)
    assert first.repaired == 1, "_Judge reports one repair for e1"
    again = runner.reconcile_primary_stats(SHARD)
    assert again.repaired == 1

    legacy_config = _epoch_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    legacy = _legacy_pair(legacy_config, legacy_storage)
    assert legacy.repaired == first.repaired


def test_primary_guard_counters_match_legacy_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _FinalBackgroundJudge

    legacy_config = _epoch_config(
        tmp_path, monkeypatch, "run-legacy", background_final_guard_mode="qwen_v1"
    )
    epoch_config = _epoch_config(
        tmp_path, monkeypatch, "run-epoch", background_final_guard_mode="qwen_v1"
    )
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    _install_clean_background(legacy_storage)
    _install_clean_background(epoch_storage)

    legacy = pair_clips(
        legacy_config,
        legacy_storage,
        judge=_Judge(),
        background_final_judge=_FinalBackgroundJudge(accepted=True),
    )
    runner = _drain_primary_with_guard(
        epoch_config, epoch_storage, tmp_path / "ledger", _FinalBackgroundJudge(accepted=True)
    )
    epoch = runner.reconcile_primary_stats(SHARD)
    assert legacy.background_final_guard_attempted == 1
    assert epoch.background_final_guard_attempted == legacy.background_final_guard_attempted
    assert epoch.background_final_guard_accepted == legacy.background_final_guard_accepted
    assert epoch.background_final_guard_rejected == legacy.background_final_guard_rejected
    assert epoch.background_final_guard_failed_closed == legacy.background_final_guard_failed_closed
    assert epoch.background_final_guard_accepted == 1


def test_primary_guard_counters_match_legacy_rejected_and_failed_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudgeFailure
    from tests.test_v3_pair import _FinalBackgroundJudge

    cases = (
        (_FinalBackgroundJudge(accepted=False), "background_final_guard_rejected"),
        (
            _FinalBackgroundJudge(
                error=FinalBackgroundJudgeFailure("structured output invalid",
                                                  raw_response="{bad json")
            ),
        "background_final_guard_failed_closed",
        ),
    )
    for index, (judge, expected) in enumerate(cases):
        legacy_config = _epoch_config(
            tmp_path, monkeypatch, "run-legacy", background_final_guard_mode="qwen_v1"
        )
        epoch_config = _epoch_config(
            tmp_path, monkeypatch, "run-epoch", background_final_guard_mode="qwen_v1"
        )
        legacy_storage = _storage(legacy_config, entity_types=("subject",))
        epoch_storage = _storage(epoch_config, entity_types=("subject",))
        _install_clean_background(legacy_storage)
        _install_clean_background(epoch_storage)
        legacy = pair_clips(
            legacy_config, legacy_storage, judge=_Judge(), background_final_judge=judge
        )
        runner = _drain_primary_with_guard(
            epoch_config, epoch_storage, tmp_path / f"ledger-{index}", judge
        )
        epoch = runner.reconcile_primary_stats(SHARD)
        assert getattr(epoch, expected) == 1, expected
        assert epoch.background_final_guard_attempted == legacy.background_final_guard_attempted
        assert getattr(epoch, expected) == getattr(legacy, expected)


def _drain_primary_with_guard(
    config: V3Config, storage: Any, ledger_root: Path, guard_judge: Any
) -> Any:
    """Drive the epoch primary pass including its background-final-guard job."""
    from r2v_data_v2.v3.post_mask_epoch_pair import (
        PAIR_BACKGROUND_GUARD_JOB,
        PAIR_PHASE,
        PairEpochRunner,
    )

    runner = PairEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(ledger_root),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    phase = runner.ledger.phase(PAIR_PHASE)
    pending = list(runner.seed_primary_jobs())
    guard = 0
    while pending:
        guard += 1
        if guard > 32:
            raise AssertionError("primary guard drain did not terminate")
        # The scheduler plans every batch before it runs, which is what makes
        # guard jobs visible to durable reconciliation.
        phase.write_plan(pending)
        job = pending.pop(0)
        handle = guard_judge if job.job_type == PAIR_BACKGROUND_GUARD_JOB else _Judge()
        result = runner.run(job, handle)
        if not result.committed:
            break
        phase = runner.ledger.phase(PAIR_PHASE)
        digest = phase.publish_result(job, result)
        receipt = phase.commit(
            job,
            outcome=result.outcome,
            artifact_digests={"result.json":digest},
            result_digest=digest,
        )
        runner.ledger.note_commit(PAIR_PHASE, receipt)
        pending.extend(runner.finalize(job, result))
    return runner


# ---------------------------------------------------------------------------
# 4c1b: reconcile against the REAL ResourceEpochScheduler phase layout
# ---------------------------------------------------------------------------

class _JobRef:
    """Minimal job reference for GroupLedger lookups by job_id."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id




class _GuardSelectExecutor:
    """Serial executor that routes each Pair job to its real judge handle."""

    def __init__(self, runner: Any, entity_judge: Any, guard_judge: Any) -> None:
        self.runner = runner
        self.entity_judge = entity_judge
        self.guard_judge = guard_judge

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

        outcomes: dict[str, Any] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            handle = (
                self.guard_judge
                if job.job_type == "pair_background_final_guard"
                else self.entity_judge
            )
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner.run(job, handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


def _run_real_scheduler(config: V3Config, storage: Any, ledger_root: Path) -> Any:
    """Run the primary pass through the production scheduler, not by hand."""
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
    from tests.test_v3_pair import _FinalBackgroundJudge

    ledger = GroupLedger(ledger_root)
    runner = PairEpochRunner(
        config,
        {SHARD: storage},
        ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={
            RESOURCE_QWEN: _GuardSelectExecutor(
                runner, _Judge(), _FinalBackgroundJudge(accepted=True)
            )
        },
    )
    outcome = scheduler.run(runner.seed_primary_jobs())
    assert outcome["completed"], outcome
    return runner, ledger, outcome


def test_reconcile_reads_real_scheduler_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt authority is rXXX-* phases, not the hand-fed `pair` phase."""
    from tests.test_v3_pair import _FinalBackgroundJudge, _install_clean_background

    legacy_config = _epoch_config(
        tmp_path, monkeypatch, "run-legacy", background_final_guard_mode="qwen_v1"
    )
    epoch_config = _epoch_config(
        tmp_path, monkeypatch, "run-epoch", background_final_guard_mode="qwen_v1"
    )
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    _install_clean_background(legacy_storage)
    _install_clean_background(epoch_storage)
    legacy = pair_clips(
        legacy_config,
        legacy_storage,
        judge=_Judge(),
        background_final_judge=_FinalBackgroundJudge(accepted=True),
    )

    runner, ledger, outcome = _run_real_scheduler(
        epoch_config, epoch_storage, tmp_path / "ledger"
    )
    # The real scheduler really used rXXX-* phases.
    assert any(phase.startswith("r") for phase in ledger.phase_ids()), ledger.phase_ids()
    pair_jobs = runner._planned_jobs(SHARD)
    by_type = {record["job_type"] for record in pair_jobs}
    assert "pair_entity_reference_judge" in by_type
    assert "pair_background_final_guard" in by_type
    # Every committed Pair job lives in an rXXX-* phase, never in "pair".
    for record in pair_jobs:
        if runner._committed_payload(record["job_id"]) is not None:
            phase_id = ledger.phase_for(_JobRef(record["job_id"]))
            assert phase_id is not None and phase_id != "pair", phase_id
    pair_phase = ledger.phase("pair")
    assert pair_phase.receipts() == {}

    stats = runner.reconcile_primary_stats(SHARD)
    assert stats.repaired == legacy.repaired == 1
    assert stats.background_final_guard_attempted == legacy.background_final_guard_attempted
    assert stats.background_final_guard_accepted == legacy.background_final_guard_accepted
    assert stats.processed == legacy.processed
    assert stats.ready == legacy.ready

    # Restart: a fresh ledger over the same root reads the SAME rXXX receipts.
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    fresh_ledger = GroupLedger(tmp_path / "ledger")
    fresh_runner = PairEpochRunner(
        epoch_config,
        {SHARD: epoch_storage},
        fresh_ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    before = fresh_runner.reconcile_primary_stats(SHARD).to_dict()
    for record in fresh_runner._planned_jobs(SHARD):
        assert fresh_ledger.phase_for(_JobRef(record["job_id"])) not in (None, "pair")
    assert before == stats.to_dict()
    # No Qwen was paid for reconciliation: the scheduler's own execution count
    # is the only one, and both reconciliations ran with zero model calls.
    assert (
        outcome["diagnostics"]["resources"]["qwen"]["jobs_executed"]
        == len(pair_jobs)
    )


def test_reconcile_without_a_frozen_plan_fails_closed_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError, PairEpochRunner

    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    ledger_root = tmp_path / "ledger"
    runner = PairEpochRunner(
        config, {SHARD: storage}, GroupLedger(ledger_root),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    before = tuple(sorted(str(p) for p in ledger_root.rglob("*") if p.is_file()))
    with pytest.raises(PairEpochError, match="frozen primary plan"):
        runner.reconcile_primary_stats(SHARD)
    after = tuple(sorted(str(p) for p in ledger_root.rglob("*") if p.is_file()))
    assert after == before
    assert not (ledger_root / "semantic" / "pair" / "primary" / f"{SHARD}.json").exists()


# ---------------------------------------------------------------------------
# 4c2: full legacy-equivalent PairStats + final stage counts
# ---------------------------------------------------------------------------

class _SubsetJudge(_Judge):
    """Like legacy SubsetJudge: tolerates source_images supersets."""

    def decide(self, **kwargs: Any) -> Any:
        candidates = kwargs["candidates"]
        decision = _decision(reference_type=kwargs["entity"].reference_type).model_copy(
            update={"selected_candidate_id": candidates[0].candidate_id}
        )
        from r2v_data_v2.v3.reference_judge import EntityReferenceDecisionAttempt

        return EntityReferenceDecisionAttempt(
            decision=decision,
            raw_responses=("{}",),
            repair_attempts=0,
        )


PREFILTER_FIELDS = (
    "prefilter_candidates_examined",
    "prefilter_candidates_filtered",
    "prefilter_near_silhouette_filtered",
    "prefilter_relative_blur_v2_filtered",
    "prefilter_entities_3_to_2",
    "prefilter_entities_3_to_1",
    "prefilter_entities_3_to_0",
    "prefilter_entities_2_to_1",
    "prefilter_entities_2_to_0",
    "prefilter_qwen_calls_skipped",
    "prefilter_fail_open_entities",
)


@pytest.mark.parametrize(
    ("before_count", "retained_ids", "transition"),
    [
        (3, ("candidate_1", "candidate_3"), "prefilter_entities_3_to_2"),
        (3, ("candidate_2",), "prefilter_entities_3_to_1"),
        (3, (), "prefilter_entities_3_to_0"),
        (2, ("candidate_2",), "prefilter_entities_2_to_1"),
        (2, (), "prefilter_entities_2_to_0"),
    ],
)
def test_prefilter_matches_legacy_all_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    before_count: int,
    retained_ids: tuple[str, ...],
    transition: str,
) -> None:
    import r2v_data_v2.v3.pair as pair_module
    from tests.test_v3_pair import _prefilter_result

    original_builder = pair_module._build_entity_reference_candidates

    def limited_builder(*args: Any, **kwargs: Any) -> Any:
        candidates, tiny, fragmented = original_builder(*args, **kwargs)
        return candidates[:before_count], tiny, fragmented

    def deterministic_prefilter(entity: Any, candidates: Any, source_images: Any) -> Any:
        return _prefilter_result(candidates, retained_ids)

    monkeypatch.setattr(pair_module, "_build_entity_reference_candidates", limited_builder)
    monkeypatch.setattr(
        pair_module, "prefilter_entity_reference_candidates", deterministic_prefilter
    )

    legacy_config = _epoch_config(
        tmp_path, monkeypatch, "run-legacy", reference_prefilter_mode="conservative_v1"
    )
    epoch_config = _epoch_config(
        tmp_path, monkeypatch, "run-epoch", reference_prefilter_mode="conservative_v1"
    )
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    legacy = _legacy_pair(legacy_config, legacy_storage)
    runner = _epoch_pair(epoch_config, epoch_storage, tmp_path / "ledger", _SubsetJudge())
    epoch = runner.reconcile_stats(SHARD)
    for field in PREFILTER_FIELDS:
        assert getattr(epoch, field) == getattr(legacy, field), field
    assert getattr(epoch, transition) == 1
    assert epoch.prefilter_candidates_examined == before_count
    assert epoch.prefilter_qwen_calls_skipped == int(not retained_ids)


def test_prefilter_fail_open_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import r2v_data_v2.v3.pair as pair_module

    def exploding_prefilter(entity: Any, candidates: Any, source_images: Any) -> Any:
        raise RuntimeError("prefilter unavailable")

    monkeypatch.setattr(
        pair_module, "prefilter_entity_reference_candidates", exploding_prefilter
    )
    legacy_config = _epoch_config(
        tmp_path, monkeypatch, "run-legacy", reference_prefilter_mode="conservative_v1"
    )
    epoch_config = _epoch_config(
        tmp_path, monkeypatch, "run-epoch", reference_prefilter_mode="conservative_v1"
    )
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    legacy = _legacy_pair(legacy_config, legacy_storage)
    runner = _epoch_pair(epoch_config, epoch_storage, tmp_path / "ledger", _SubsetJudge())
    epoch = runner.reconcile_stats(SHARD)
    assert epoch.prefilter_fail_open_entities == legacy.prefilter_fail_open_entities == 1
    assert epoch.prefilter_candidates_examined == legacy.prefilter_candidates_examined
    for field in PREFILTER_FIELDS:
        if field in ("prefilter_fail_open_entities", "prefilter_candidates_examined"):
            continue
        assert getattr(epoch, field) == getattr(legacy, field) == 0, field


def test_prefilter_mode_off_is_zero_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    stats = runner.reconcile_stats(SHARD)
    for field in PREFILTER_FIELDS:
        assert getattr(stats, field) == 0, field


# -- cross fixtures ---------------------------------------------------------

def _install_clean_background_for(storage: Any, clip_uid: str) -> None:
    """_install_clean_background, parameterized by clip."""
    from r2v_data_v2.reconciliation import write_json_atomic
    from r2v_data_v2.v3.schemas import (
        BackgroundAnnotation,
        BackgroundReferenceState,
        ReferencesState,
    )

    clip = storage.read_clip(clip_uid)
    assert clip.annotation is not None
    annotation = clip.annotation.model_copy(
        update={
            "background": BackgroundAnnotation(
                phrase="a stone courtyard",
                grounding_prompt="the stone courtyard and surrounding walls",
            )
        }
    )
    write_json_atomic(
        storage.clip_path(clip_uid),
        clip.model_copy(update={"annotation": annotation}).model_dump(mode="json"),
    )
    storage.write_references(
        clip_uid,
        ReferencesState(
            background=BackgroundReferenceState(
                status="clean_raw",
                source_image_path=f"clips/{clip_uid}/frames/00.jpg",
                output_image_path=f"clips/{clip_uid}/frames/00.jpg",
                source_frame_slot=0,
                source_frame_index=0,
                source_foreground_area_pixels=0,
                source_foreground_area_ratio=0.0,
            )
        ),
    )





def _cross_epoch_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    guard: bool = False,
    run_name: str = "run-epoch",
) -> tuple[Any, Any, Any]:
    from tests.test_v3_post_mask_epoch_pair import (
        _drain_primary,
        _runner,
        _ScopedJudge,
        _three_donor_shard,
    )

    kwargs: dict[str, Any] = {"same_parent_fallback_enabled": True}
    if guard:
        kwargs["background_final_guard_mode"] = "qwen_v1"
    config = _epoch_config(tmp_path, monkeypatch, run_name, **kwargs)
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    if guard:
        for uid in ("clip-1", "donor", "target-b", "target-c"):
            _install_clean_background_for(storage, uid)
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    if guard:
        from tests.test_v3_pair import _FinalBackgroundJudge

        _drain_primary_routed(
            runner,
            _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
            _FinalBackgroundJudge(accepted=True),
        )
    else:
        _drain_primary(
            runner,
            _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
        )
    return config, storage, runner


def _drain_cross(runner: Any, cross_judge: Any) -> None:
    """Run every seeded cross job to its terminal state."""
    from r2v_data_v2.v3.post_mask_epoch_pair import PAIR_PHASE

    phase = runner.ledger.phase(PAIR_PHASE)
    pending = list(runner.freeze_cross_pair_after_primary_quiescence())
    while pending:
        # The scheduler plans every batch before running it; that plan is what
        # durable reconciliation reads.
        phase.write_plan(pending)
        job = pending.pop(0)
        result = runner.run(job, cross_judge)
        if not result.committed:
            break
        phase = runner.ledger.phase(PAIR_PHASE)
        digest = phase.publish_result(job, result)
        receipt = phase.commit(
            job,
            outcome=result.outcome,
            artifact_digests={"result.json": digest},
            result_digest=digest,
        )
        runner.ledger.note_commit(PAIR_PHASE, receipt)
        pending.extend(runner.finalize(job, result))


def _legacy_cross_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cross_judge: Any,
    *,
    guard_judge: Any = None,
    run_name: str = "run-legacy",
) -> Any:
    from tests.test_v3_post_mask_epoch_pair import _three_donor_shard

    kwargs: dict[str, Any] = {"same_parent_fallback_enabled": True}
    if guard_judge is not None:
        kwargs["background_final_guard_mode"] = "qwen_v1"
    config = _epoch_config(tmp_path, monkeypatch, run_name, **kwargs)
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    if guard_judge is not None:
        for uid in ("clip-1", "donor", "target-b", "target-c"):
            _install_clean_background_for(storage, uid)

    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    return pair_clips(
        config,
        storage,
        judge=_ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
        cross_pair_judge=cross_judge,
        background_final_judge=guard_judge,
    )


def test_cross_donor1_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _CrossJudge

    # Both rejected targets accept the first donor: two attempts, two ready.
    legacy = _legacy_cross_pair(tmp_path, monkeypatch, _CrossJudge([True, True]))
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    _drain_cross(runner, _CrossJudge([True, True]))
    stats = runner.reconcile_stats(SHARD)
    assert stats.cross_pair_attempted == legacy.cross_pair_attempted
    assert stats.cross_pair_ready == legacy.cross_pair_ready
    assert stats.cross_pair_repaired == legacy.cross_pair_repaired
    assert stats.cross_pair_attempted >= 1


def test_cross_donor1_reject_then_donor2_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _CrossJudge

    legacy = _legacy_cross_pair(tmp_path, monkeypatch, _CrossJudge([False, True, True]))
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    _drain_cross(runner, _CrossJudge([False, True, True]))
    stats = runner.reconcile_stats(SHARD)
    assert stats.cross_pair_attempted == legacy.cross_pair_attempted
    assert stats.cross_pair_ready == legacy.cross_pair_ready



def test_cross_repaired_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _CrossJudge

    legacy = _legacy_cross_pair(
        tmp_path, monkeypatch, _CrossJudge([True, True], repair_attempts=1)
    )
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    _drain_cross(runner, _CrossJudge([True, True], repair_attempts=1))
    stats = runner.reconcile_stats(SHARD)
    assert stats.cross_pair_repaired == legacy.cross_pair_repaired
    assert stats.cross_pair_repaired >= 1
    # One per repaired decision, never the raw repair_attempts sum.
    assert stats.cross_pair_repaired == 2


def test_cross_judge_failure_counts_failed_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.structured_output import ValidationIssue
    from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure
    from tests.test_v3_pair import _CrossJudge

    class _FailingCross(_CrossJudge):
        def decide(self, **kwargs: Any) -> Any:
            raise CrossPairJudgeFailure(
                raw_responses=["bad"],
                issues=[
                    ValidationIssue(
                        code="schema_invalid", field="verdict", message="missing verdict"
                    )
                ],
                attempt_count=1,
            )

    legacy = _legacy_cross_pair(tmp_path, monkeypatch, _FailingCross())
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    _drain_cross(runner, _FailingCross())
    stats = runner.reconcile_stats(SHARD)
    assert stats.cross_pair_attempted == legacy.cross_pair_attempted
    assert stats.cross_pair_ready == legacy.cross_pair_ready == 0
    assert stats.failed == legacy.failed


def test_primary_rejected_then_cross_ready_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _CrossJudge

    legacy = _legacy_cross_pair(tmp_path, monkeypatch, _CrossJudge([True, True]))
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    _drain_cross(runner, _CrossJudge([True, True]))
    stats = runner.reconcile_stats(SHARD)
    # Primary rejected, cross upgraded: the final net state is ready.
    assert stats.ready == legacy.ready
    assert stats.rejected == legacy.rejected
    assert stats.entities_ready == legacy.entities_ready
    assert stats.entities_rejected == legacy.entities_rejected
    assert stats.backgrounds_bound == legacy.backgrounds_bound


def test_full_stats_with_cross_guard_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _CrossJudge, _FinalBackgroundJudge

    legacy = _legacy_cross_pair(
        tmp_path,
        monkeypatch,
        _CrossJudge([True, True]),
        guard_judge=_FinalBackgroundJudge(accepted=True),
    )
    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch, guard=True)

    class _RoutingJudge:
        """One handle routing entity, cross and guard jobs."""

        def __init__(self) -> None:
            self.entity = _Judge()
            self.cross = _CrossJudge([True, True])
            self.guard = _FinalBackgroundJudge(accepted=True)

        def decide(self, **kwargs: Any) -> Any:
            if "target_clip_uid" in kwargs:
                return self.cross.decide(**kwargs)
            return self.entity.decide(**kwargs)

        def review(self, **kwargs: Any) -> Any:
            return self.guard.review(**kwargs)

    # The fixture already drained the primary guards; finish cross + guards.
    _drain_cross(runner, _RoutingJudge())
    stats = runner.reconcile_stats(SHARD)
    # Guard calls happen at both call sites while the background binds once.
    assert (
        stats.background_final_guard_attempted
        == legacy.background_final_guard_attempted
    )
    assert (
        stats.background_final_guard_accepted
        == legacy.background_final_guard_accepted
    )
    assert stats.backgrounds_bound == legacy.backgrounds_bound
    assert stats.to_dict() == legacy.to_dict()


def _drain_primary_routed(
    runner: Any, entity_judge: Any, guard_judge: Any = None
) -> None:
    """Primary drain whose guard jobs go through the guard handle."""
    from r2v_data_v2.v3.post_mask_epoch_pair import (
        PAIR_BACKGROUND_GUARD_JOB,
        PAIR_PHASE,
    )

    phase = runner.ledger.phase(PAIR_PHASE)
    pending = list(runner.seed_primary_jobs())
    guard = 0
    while pending:
        guard += 1
        if guard > 64:
            raise AssertionError("primary routed drain did not terminate")
        phase.write_plan(pending)
        job = pending.pop(0)
        handle = (
            guard_judge
            if guard_judge is not None and job.job_type == PAIR_BACKGROUND_GUARD_JOB
            else entity_judge
        )
        result = runner.run(job, handle)
        if not result.committed:
            break
        digest = phase.publish_result(job, result)
        receipt = phase.commit(
            job,
            outcome=result.outcome,
            artifact_digests={"result.json": digest},
            result_digest=digest,
        )
        runner.ledger.note_commit(PAIR_PHASE, receipt)
        pending.extend(runner.finalize(job, result))


def test_full_stats_refuse_incomplete_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError
    from tests.test_v3_pair import _CrossJudge

    _config, _storage, runner = _cross_epoch_fixture(tmp_path, monkeypatch)
    # Cross jobs exist but were never run: no terminal, no commit.
    pending = list(runner.freeze_cross_pair_after_primary_quiescence())
    assert pending, "fixture must seed cross work"
    with pytest.raises(PairEpochError, match="not terminal|cross terminal"):
        runner.reconcile_stats(SHARD)
    # A partially drained cross pass also refuses.
    _drain_cross(runner, _CrossJudge([True]))
    try:
        with pytest.raises(PairEpochError, match="not terminal|cross terminal"):
            runner.reconcile_stats(SHARD)
    finally:
        pass


def test_malformed_phase_plan_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    phases = (Path(runner.ledger.root) / "phases").iterdir()
    broken = next(path for path in phases if (path / "plan.json").is_file())
    (broken / "plan.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(PairEpochError, match="invalid Pair phase plan"):
        runner.reconcile_stats(SHARD)


def test_tampered_committed_result_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    config = _epoch_config(tmp_path, monkeypatch, "run-epoch")
    storage = _storage(config, entity_types=("subject",))
    runner = _epoch_pair(config, storage, tmp_path / "ledger")
    record = runner._planned_jobs(SHARD)[0]
    job = runner._job_from_plan_record(record)
    phase_id = runner.ledger.phase_for(_JobRef(job.job_id()))
    assert phase_id is not None
    result_path = Path(runner.ledger.root) / "phases" / phase_id / "artifacts" / job.job_id() / "result.json"
    assert result_path.is_file()
    result_path.write_text('{"outcome":"completed","detail":"","result_payload":{"repair_attempts":99}}', encoding="utf-8")
    with pytest.raises(PairEpochError, match="receipt mismatch|does not rebuild"):
        runner.reconcile_stats(SHARD)


def test_full_stats_restart_stability_and_purity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from tests.test_v3_pair import _CrossJudge

    _config, _storage, runner = _cross_epoch_fixture(
        tmp_path, monkeypatch, guard=True
    )
    _drain_cross(runner, _CrossJudge([True, True]))
    stats1 = runner.reconcile_stats(SHARD).to_dict()
    ledger_root = Path(runner.ledger.root)
    before = tuple(sorted(str(p) for p in ledger_root.rglob("*") if p.is_file()))

    fresh_ledger = GroupLedger(ledger_root)
    fresh_runner = PairEpochRunner(
        _config,
        {SHARD: _storage},
        fresh_ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "donor", "target-b", "target-c"]},
    )
    stats2 = fresh_runner.reconcile_stats(SHARD).to_dict()
    assert stats2 == stats1
    # Zero extra Qwen: reconciliation wrote nothing and read only durable state.
    assert tuple(sorted(str(p) for p in ledger_root.rglob("*") if p.is_file())) == before


def test_primary_and_cross_guard_are_two_paid_calls_one_bound_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The characterized primary+cross guard pair, reconciled durably."""
    from tests.test_v3_pair import _CrossJudge, _FinalBackgroundJudge
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _epoch_config(
        tmp_path, monkeypatch, "run-epoch",
        same_parent_fallback_enabled=True, background_final_guard_mode="qwen_v1",
    )
    storage = _storage(config, entity_types=("subject",))
    from tests.test_v3_post_mask_epoch_pair import _three_donor_shard

    _three_donor_shard(config, storage)
    for uid in ("clip-1", "donor", "target-b", "target-c"):
        _install_clean_background_for(storage, uid)
    from tests.test_v3_post_mask_epoch_pair import _runner

    runner = _runner(
        tmp_path, config, storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    # Primary guard rejects: the background is not bound during primary.
    _drain_primary_routed(
        runner,
        _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
        _FinalBackgroundJudge(accepted=False),
    )

    class _Routing:
        def __init__(self) -> None:
            self.cross = _CrossJudge([True, True])
            self.guard = _FinalBackgroundJudge(accepted=True)

        def decide(self, **kwargs: Any) -> Any:
            return self.cross.decide(**kwargs)

        def review(self, **kwargs: Any) -> Any:
            return self.guard.review(**kwargs)

    routing = _Routing()
    _drain_cross(runner, routing)
    stats = runner.reconcile_stats(SHARD)
    # Both call sites paid one guard call per target, but the background of a
    # cross-published target binds exactly once.
    assert stats.background_final_guard_attempted == 4
    assert stats.background_final_guard_accepted == 2
    assert stats.background_final_guard_rejected == 2
    assert stats.backgrounds_bound == stats.background_final_guard_accepted
