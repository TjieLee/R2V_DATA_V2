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
