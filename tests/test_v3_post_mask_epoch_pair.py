"""Characterization of the frozen ``pair_clips()`` Pair semantics.

These tests run against the *current* production Pair implementation and must
pass unchanged: they record behaviour, they do not propose any. The resource
epoch Pair adapter has to reproduce exactly what is captured here, including
the parts that look unattractive:

* a per-entity judge failure aborts the whole clip, so later entities are
  never called;
* a cross-pair judge failure aborts that target clip's whole fallback, so
  later donors *and* later entities are never called;
* the background final guard is not de-duplicated across the primary pass,
  the cross-pair pass and the trailing loop -- the call count is whatever
  ``pair_clips()`` actually produces.

Nothing here touches prompts, thresholds, ranking, prefilter, donor policy or
reference tokens.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.config import PairConfig, V3Config
from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure
from r2v_data_v2.v3.pair import pair_clips
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceDecisionAttempt,
    EntityReferenceJudgeFailure,
)
from r2v_data_v2.v3.schemas import (
    BackgroundAnnotation,
    BackgroundReferenceState,
    EntityReferenceState,
    PairingState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage
from tests.test_v3_pair import (
    _add_ready_clip,
    _config,
    _cross_decision,
    _CrossJudge,
    _decision,
    _install_clean_background,
    _Judge,
    _same_parent_storage,
    _storage,
)


def _judge_failure(message: str = "structured output invalid") -> Exception:
    return EntityReferenceJudgeFailure(
        raw_responses=["{not json}"],
        issues=[],
        attempt_count=1,
    )


def _cross_failure(message: str = "structured output invalid") -> Exception:
    return CrossPairJudgeFailure(
        raw_responses=["{not json}"],
        issues=[],
        attempt_count=1,
    )


class _FailingEntityJudge:
    """``_Judge`` plus an explicit failure injection for one entity."""

    def __init__(
        self,
        scopes: dict[str, str] | None = None,
        *,
        fail_on: str | None = None,
    ) -> None:
        self.scopes = scopes or {}
        self.fail_on = fail_on
        self.calls: list[tuple[str, list[str]]] = []
        self.close_calls = 0

    def decide(self, *, entity, candidates, source_images):
        self.calls.append(
            (entity.entity_id, [item.candidate_id for item in candidates])
        )
        if entity.entity_id == self.fail_on:
            raise _judge_failure()
        return EntityReferenceDecisionAttempt(
            decision=_decision(
                self.scopes.get(entity.entity_id, "full"),
                entity.reference_type,
            ),
            raw_responses=("{}",),
            repair_attempts=0,
        )

    def close(self) -> None:
        self.close_calls += 1


class _FailingCrossJudge:
    """``_CrossJudge`` plus a cross-pair failure at a chosen donor index."""

    def __init__(
        self,
        decisions: list[bool] | None = None,
        *,
        fail_at: int | None = None,
    ) -> None:
        self.decisions = list(decisions or [])
        self.fail_at = fail_at
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        target_clip_uid,
        target_entity,
        target_evidence_mode,
        target_context_image,
        target_entity_crop,
        donor_clip_uid,
        donor_entity,
        donor_reference_image,
    ):
        index = len(self.calls)
        self.calls.append(
            {
                "target_clip_uid": target_clip_uid,
                "target_entity_id": target_entity.entity_id,
                "target_evidence_mode": target_evidence_mode,
                "donor_clip_uid": donor_clip_uid,
                "donor_entity_id": donor_entity.entity_id,
            }
        )
        if index == self.fail_at:
            raise _cross_failure()
        accept = self.decisions[index] if index < len(self.decisions) else True
        from r2v_data_v2.v3.cross_pair_judge import CrossPairDecisionAttempt

        return CrossPairDecisionAttempt(
            decision=_cross_decision(accept=accept),
            raw_responses=("{}",),
            repair_attempts=0,
        )


class _NeverCalledJudge:
    """Sentinel: any Pair Qwen call is a regression."""

    def __init__(self, label: str = "judge") -> None:
        self.label = label
        self.calls: list[Any] = []

    def decide(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise AssertionError(f"{self.label} must not be called")

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# A. Primary entity judge failure
# ---------------------------------------------------------------------------


def test_characterize_entity_judge_failure_stops_the_entity_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e2 raises: e3 must never be judged and the clip stays unpublished."""
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    judge = _FailingEntityJudge(fail_on="e2")

    stats = pair_clips(config, storage, judge=judge)

    assert [entity_id for entity_id, _ in judge.calls] == ["e1", "e2"], (
        "e3 must not be called after e2 failed"
    )
    clip = storage.read_clip("clip-1")
    assert clip.pairing is None, "a failed entity leaves pairing unpublished"
    assert stats.failed == 1
    assert stats.processed == 1
    assert stats.ready == 0
    assert stats.rejected == 0
    # e1's temporary reference image is rolled back with the clip.
    assert not storage.selected_entity_path("clip-1", "e1").is_file()


def test_characterize_failed_clip_never_enters_cross_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _storage(config, entity_types=("subject", "object"))
    cross = _FailingCrossJudge()

    stats = pair_clips(
        config, storage, judge=_FailingEntityJudge(fail_on="e2"), cross_pair_judge=cross
    )

    assert cross.calls == []
    assert stats.cross_pair_attempted == 0
    assert stats.failed == 1


# ---------------------------------------------------------------------------
# B. Cross-pair ordering and stop-first-accept
# ---------------------------------------------------------------------------


def _three_donor_storage(config: V3Config) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-order-test")
    for uid, suffix in (("donor-10", "10"), ("donor-2", "2"), ("donor-1", "1")):
        _add_ready_clip(
            config, storage, clip_uid=uid, clip_suffix=suffix, entity_types=("subject",)
        )
    _add_ready_clip(
        config, storage, clip_uid="target", clip_suffix="20", entity_types=("subject",)
    )
    return storage


def test_characterize_cross_pair_donor_order_is_natural_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _three_donor_storage(config)
    cross = _CrossJudge([False, False, False])
    from tests.test_v3_pair import _ClipJudge

    pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == [
        "donor-1",
        "donor-2",
        "donor-10",
    ]


def test_characterize_first_donor_accept_stops_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _three_donor_storage(config)
    cross = _CrossJudge([True, True, True])
    from tests.test_v3_pair import _ClipJudge

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == ["donor-1"]
    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 1


def test_characterize_reject_then_accept_stops_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _three_donor_storage(config)
    cross = _CrossJudge([False, True, True])
    from tests.test_v3_pair import _ClipJudge

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert [call["donor_clip_uid"] for call in cross.calls] == ["donor-1", "donor-2"]
    assert stats.cross_pair_attempted == 2
    assert stats.cross_pair_ready == 1


def test_characterize_all_donors_reject_preserves_target_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _three_donor_storage(config)
    cross = _CrossJudge([False, False, False])
    from tests.test_v3_pair import _ClipJudge

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    target = storage.read_clip("target")
    assert target.pairing == PairingState(
        status="rejected",
        reason="no_qualifying_ready_reference",
    )
    assert stats.cross_pair_ready == 0
    assert stats.rejected == 1


# ---------------------------------------------------------------------------
# C. Cross-pair judge failure
# ---------------------------------------------------------------------------


def test_characterize_cross_pair_failure_aborts_target_and_keeps_primary_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _same_parent_storage(config)
    cross = _FailingCrossJudge(decisions=[True], fail_at=0)
    from tests.test_v3_pair import _ClipJudge

    stats = pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert len(cross.calls) == 1, "donor2 must not be called after donor1 failed"
    target = storage.read_clip("target")
    # The primary pairing survives: only the fallback block aborted.
    assert target.pairing is not None
    assert target.pairing.status == "rejected"
    assert stats.failed == 1
    assert stats.cross_pair_ready == 0


def test_characterize_cross_pair_failure_is_not_retried_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target with a durable pairing is skipped, so no re-cross-pair."""
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _same_parent_storage(config)
    from tests.test_v3_pair import _ClipJudge

    pair_clips(
        config,
        storage,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=_FailingCrossJudge(decisions=[True], fail_at=0),
    )

    stats = pair_clips(
        config,
        storage,
        judge=_NeverCalledJudge("entity judge"),
        cross_pair_judge=_NeverCalledJudge("cross judge"),
    )

    assert stats.skipped_existing == 2, "both clips already carry a durable pairing"
    assert stats.processed == 0
    assert stats.cross_pair_attempted == 0
    assert stats.failed == 0


# ---------------------------------------------------------------------------
# D. Donor domain is one storage (one canonical shard)
# ---------------------------------------------------------------------------


def test_characterize_donor_domain_is_one_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-parent donor in another shard's storage is invisible."""
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    target_shard = RunStorage(config)
    target_shard.initialize(git_commit="shard-a")
    _add_ready_clip(
        config,
        target_shard,
        clip_uid="target",
        clip_suffix="20",
        entity_types=("subject",),
    )
    # A perfect same-parent donor, but living in another shard's run root.
    other_config = replace(config, run_root=config.run_root.parent / "shard-b")
    other_shard = RunStorage(other_config)
    other_shard.initialize(git_commit="shard-b")
    _add_ready_clip(
        config,
        other_shard,
        clip_uid="donor",
        clip_suffix="2",
        entity_types=("subject",),
    )
    cross = _CrossJudge([True])
    from tests.test_v3_pair import _ClipJudge

    stats = pair_clips(
        config,
        target_shard,
        judge=_ClipJudge({("target", "e1"): "reject"}),
        cross_pair_judge=cross,
    )

    assert cross.calls == [], "no cross-shard donor may be used"
    assert stats.cross_pair_attempted == 0


# ---------------------------------------------------------------------------
# E. Existing pairing is retained and reused as a donor
# ---------------------------------------------------------------------------


def test_characterize_existing_pairing_is_not_rejudged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    first = pair_clips(config, storage, judge=_Judge())
    assert first.ready == 1

    stats = pair_clips(config, storage, judge=_NeverCalledJudge("entity judge"))

    assert stats.skipped_existing == 1
    assert stats.processed == 0
    assert storage.read_clip("clip-1").pairing is not None


# ---------------------------------------------------------------------------
# F. Background final guard call count
# ---------------------------------------------------------------------------


def test_characterize_background_guard_called_once_for_primary_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(background_final_guard_mode="qwen_v1"),
        debug=True,
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    from tests.test_v3_pair import _FinalBackgroundJudge

    guard = _FinalBackgroundJudge(accepted=True)

    stats = pair_clips(config, storage, judge=_Judge(), background_final_judge=guard)

    # Primary publishes ready -> guard evaluated once in the primary pass, and
    # the trailing loop skips it because it is already evaluated.
    assert stats.background_final_guard_attempted == 1
    assert stats.background_final_guard_accepted == 1
    assert len(guard.calls) == 1
    assert storage.read_clip("clip-1").pairing is not None
    assert storage.read_clip("clip-1").pairing.background_token == "<ref_bg_1>"


def test_characterize_background_guard_mode_off_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(background_final_guard_mode="off")
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    from tests.test_v3_pair import _FinalBackgroundJudge

    guard = _FinalBackgroundJudge(accepted=True)

    stats = pair_clips(config, storage, judge=_Judge(), background_final_judge=guard)

    assert guard.calls == []
    assert stats.background_final_guard_attempted == 0
    assert storage.read_clip("clip-1").pairing.background_token == "<ref_bg_1>"


def test_characterize_background_guard_failure_binds_nothing_but_publishes_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudgeFailure
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(background_final_guard_mode="qwen_v1"),
        debug=True,
    )
    storage = _storage(config, entity_types=("subject",))
    background = _install_clean_background(storage)
    guard = _FinalBackgroundJudge(
        error=FinalBackgroundJudgeFailure(
            "structured output invalid", raw_response="{bad json"
        )
    )

    stats = pair_clips(config, storage, judge=_Judge(), background_final_judge=guard)

    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "ready"
    assert clip.pairing.background_token is None, "guard failure closes background only"
    assert stats.background_final_guard_failed_closed == 1
    assert stats.failed == 0, "a guard failure is not a Pair infrastructure failure"
    assert storage.read_clip("clip-1").references.background == background


# ---------------------------------------------------------------------------
# G. Completion bypass under the production Post-Mask configuration
# ---------------------------------------------------------------------------


def test_characterize_reference_edit_enabled_bypairs_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production Post-Mask runs with reference_edit.enabled=True."""
    config = _config(tmp_path, monkeypatch, reference_edit_enabled=True)
    storage = _storage(config, entity_types=("subject",))

    stats = pair_clips(config, storage, judge=_Judge())

    assert stats.completion_attempted == 0
    assert stats.completion_ready == 0
    assert stats.completion_rejected == 0
    assert stats.ready == 1


# ---------------------------------------------------------------------------
# H. Cross-pair failure stops later target entities
# ---------------------------------------------------------------------------


class _ScopedJudge:
    """Per-clip scopes, plus an explicit "this clip must never be judged" set."""

    def __init__(
        self,
        scopes: dict[tuple[str, str], str] | None = None,
        *,
        forbidden: tuple[str, ...] = (),
    ) -> None:
        self.scopes = scopes or {}
        self.forbidden = set(forbidden)
        self.calls: list[tuple[str, str]] = []
        self.close_calls = 0

    def decide(self, *, entity, candidates, source_images):
        clip_uid = Path(candidates[0].image_path).parts[1]
        self.calls.append((clip_uid, entity.entity_id))
        if clip_uid in self.forbidden:
            raise AssertionError(f"{clip_uid} must not be judged again")
        return EntityReferenceDecisionAttempt(
            decision=_decision(
                self.scopes.get((clip_uid, entity.entity_id), "full"),
                entity.reference_type,
            ),
            raw_responses=("{}",),
            repair_attempts=0,
        )

    def close(self) -> None:
        self.close_calls += 1


def _two_entity_target_storage(config: V3Config) -> RunStorage:
    """One ready full donor plus a two-entity same-parent target."""
    storage = RunStorage(config)
    storage.initialize(git_commit="cross-two-entity-test")
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="2",
        entity_types=("subject",),
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="20",
        entity_types=("subject", "subject"),
    )
    return storage


def test_characterize_cross_pair_failure_stops_later_target_entities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """target/e1 failing must not let target/e2 be cross-paired."""
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)
    cross = _FailingCrossJudge(decisions=[True], fail_at=0)

    stats = pair_clips(
        config,
        storage,
        judge=_ScopedJudge(
            {("target", "e1"): "reject", ("target", "e2"): "reject"}
        ),
        cross_pair_judge=cross,
    )

    observed = [
        (call["target_entity_id"], call["donor_clip_uid"]) for call in cross.calls
    ]
    assert observed == [("e1", "donor")], "e2 must not be cross-paired after e1 failed"
    target = storage.read_clip("target")
    assert target.pairing is not None
    assert target.pairing.status == "rejected"
    assert stats.failed == 1
    assert stats.cross_pair_ready == 0


# ---------------------------------------------------------------------------
# I. Existing completed Pair stays a valid same-shard donor
# ---------------------------------------------------------------------------


def test_characterize_existing_pairing_remains_a_valid_donor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing donor is reused without paying for its primary Qwen again."""
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="existing-donor-test")
    _add_ready_clip(
        config, storage, clip_uid="donor", clip_suffix="2", entity_types=("subject",)
    )

    first = pair_clips(config, storage, judge=_ScopedJudge())
    assert first.ready == 1, "donor publishes a ready full self reference"

    _add_ready_clip(
        config, storage, clip_uid="target", clip_suffix="20", entity_types=("subject",)
    )
    judge = _ScopedJudge({("target", "e1"): "reject"}, forbidden=("donor",))

    stats = pair_clips(
        config, storage, judge=judge, cross_pair_judge=_CrossJudge([True])
    )

    assert ("donor", "e1") not in judge.calls, "existing donor must not be re-judged"
    assert stats.skipped_existing == 1
    assert stats.processed == 1
    assert stats.cross_pair_attempted == 1
    assert stats.cross_pair_ready == 1
    target_state = storage.read_clip("target").references.entities[0]
    assert target_state.source_clip_uid == "donor"
    assert target_state.source_entity_id == "e1"


# ---------------------------------------------------------------------------
# J. Background guard: primary pass AND cross-pair pass
# ---------------------------------------------------------------------------


def _install_background_on(storage: RunStorage, clip_uid: str) -> None:
    from r2v_data_v2.reconciliation import write_json_atomic

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


def test_characterize_background_guard_runs_in_primary_and_again_in_cross_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No de-duplication: the guard is paid for in both passes."""
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(
            same_parent_fallback_enabled=True, background_final_guard_mode="qwen_v1"
        ),
        debug=True,
    )
    storage = _two_entity_target_storage(config)
    _install_background_on(storage, "target")
    guard = _FinalBackgroundJudge(accepted=True)

    stats = pair_clips(
        config,
        storage,
        judge=_ScopedJudge(
            {("target", "e1"): "full", ("target", "e2"): "reject"}
        ),
        cross_pair_judge=_CrossJudge([True]),
        background_final_judge=guard,
    )

    # Primary: e1 is full so the pairing is ready -> guard call 1.
    # Cross-pair upgrades e2 from the donor -> guard call 2. The trailing loop
    # does not add a third because the clip is already in evaluated_clip_uids.
    assert len(guard.calls) == 2
    assert stats.background_final_guard_attempted == 2
    assert stats.background_final_guard_accepted == 2
    # Recorded, not assumed: the pair counter only increments once.
    assert stats.backgrounds_bound == 1
    target = storage.read_clip("target")
    assert target.pairing is not None
    assert target.pairing.status == "ready"
    assert target.pairing.background_token == "<ref_bg_1>"


# ---------------------------------------------------------------------------
# K. Post-Mask runtime treatment of a cross-pair fallback failure
# ---------------------------------------------------------------------------


def _cross_pair_failure_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, target_e1: str
) -> tuple[V3Config, RunStorage]:
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)
    return config, storage


def test_characterize_runtime_does_not_retry_pair_after_cross_pair_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary Pair success + cross-pair failure is a terminal Pair stage."""
    from r2v_data_v2.v3.post_mask_runtime import (
        _record_phase_result,
        phase_needed,
    )

    config, storage = _cross_pair_failure_storage(tmp_path, monkeypatch, target_e1="full")
    target_before = storage.read_clip("target")
    assert phase_needed(storage, target_before, "pair") is True, "no pairing yet"

    stats = pair_clips(
        config,
        storage,
        # e1 is full so the primary pairing is ready; e2 needs the fallback,
        # and the fallback judge fails on it.
        judge=_ScopedJudge({("target", "e1"): "full", ("target", "e2"): "reject"}),
        cross_pair_judge=_FailingCrossJudge(decisions=[True], fail_at=0),
    )

    target = storage.read_clip("target")
    assert stats.failed == 1, "legacy records the failed fallback"
    assert target.pairing is not None
    assert target.pairing.status == "ready", "primary pairing survives the fallback"
    assert phase_needed(storage, target, "pair") is False, (
        "a published pairing ends the Pair stage even when the fallback failed"
    )

    pending: set[str] = set()
    _record_phase_result(
        storage,
        "pair",
        ["target"],
        stats.to_dict(),
        None,
        {},
        pending,
        lambda *args, **kwargs: None,
    )
    assert "target" not in pending, (
        "cross-pair fallback failure must not make the clip retryable"
    )


def test_characterize_runtime_rejected_pairing_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected content outcome is terminal, not retryable."""
    from r2v_data_v2.v3.post_mask_runtime import (
        _record_phase_result,
        phase_needed,
    )

    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)

    stats = pair_clips(
        config,
        storage,
        judge=_ScopedJudge({("target", "e1"): "reject", ("target", "e2"): "reject"}),
        cross_pair_judge=_FailingCrossJudge(decisions=[True], fail_at=0),
    )

    target = storage.read_clip("target")
    assert target.pairing is not None
    assert target.pairing.status == "rejected"
    assert phase_needed(storage, target, "pair") is False

    pending: set[str] = set()
    _record_phase_result(
        storage,
        "pair",
        ["target"],
        stats.to_dict(),
        None,
        {},
        pending,
        lambda *args, **kwargs: None,
    )
    assert "target" not in pending


# ---------------------------------------------------------------------------
# L. Direct helper-boundary tests (guards against Commit 3 misuse)
# ---------------------------------------------------------------------------


def test_helper_prepare_entity_reference_matches_legacy_judge_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare() must yield exactly the candidates the legacy judge saw."""
    from r2v_data_v2.v3.frames import validate_sampled_frames
    from r2v_data_v2.v3.pair import (
        PairStats,
        finalize_entity_reference,
        prepare_entity_reference,
        run_entity_reference_judge,
    )

    config = _config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    capturing = _Judge()
    pair_clips(config, storage, judge=capturing)
    legacy_candidates = [candidates for _, candidates in capturing.calls]

    # A second run root, so "publication stays with pair_clips" is observable.
    helper_config = replace(config, run_root=config.run_root.parent / "helper")
    helper_storage = _storage(helper_config, entity_types=("subject",))
    frames = validate_sampled_frames(helper_storage, "clip-1")
    masks = helper_storage.read_masks("clip-1")
    counters = {field: 0 for field in PairStats.__dataclass_fields__}
    entity = helper_storage.read_clip("clip-1").annotation.entities[0]
    prepared = prepare_entity_reference(
        helper_config,
        helper_storage,
        clip_uid="clip-1",
        entity=entity,
        frames=frames,
        masks=masks,
        counters=counters,
    )

    assert not isinstance(prepared, EntityReferenceState), "e1 needs a judge"
    assert [item.candidate_id for item in prepared.candidates] == legacy_candidates[0]

    attempt = run_entity_reference_judge(prepared, _Judge())
    finalization = finalize_entity_reference(
        helper_config, helper_storage, prepared=prepared, attempt=attempt
    )
    assert finalization.state.status == "ready"
    assert finalization.temporary is not None
    assert finalization.temporary[0].is_file()
    assert not helper_storage.selected_entity_path("clip-1", "e1").is_file()


def test_helper_prepare_cross_pair_evidence_matches_legacy_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.frames import validate_sampled_frames
    from r2v_data_v2.v3.pair import (
        build_entity_reference_candidates,
        prepare_cross_pair_target_evidence,
    )

    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)
    target = storage.read_clip("target")
    frames = validate_sampled_frames(storage, "target")
    masks = storage.read_masks("target")
    # The caller owns the ordering: candidates first, then donors, then this.
    candidates = build_entity_reference_candidates(
        config,
        storage,
        clip_uid="target",
        entity=target.annotation.entities[0],
        frames=frames,
        masks=masks,
    )

    evidence = prepare_cross_pair_target_evidence(
        config,
        storage,
        clip_uid="target",
        entity=target.annotation.entities[0],
        frames=frames,
        target_candidates=candidates,
    )

    # The legacy fake cross judge recorded this exact mode.
    assert evidence.evidence_mode == "masked_candidate"
    assert evidence.entity_crop is not None
    assert evidence.frame_slots == (candidates[0].frame_slot,)


def test_helper_guard_boundary_matches_the_synchronous_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.frames import validate_sampled_frames
    from r2v_data_v2.v3.pair import (
        PairStats,
        _BackgroundFinalGuardRuntime,
    )
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(background_final_guard_mode="qwen_v1"),
        debug=True,
    )
    storage = _storage(config, entity_types=("subject",))
    _install_background_on(storage, "clip-1")
    counters = {field: 0 for field in PairStats.__dataclass_fields__}
    runtime = _BackgroundFinalGuardRuntime(
        config, storage, counters, _FinalBackgroundJudge(accepted=True)
    )
    clip = storage.read_clip("clip-1")
    frames = validate_sampled_frames(storage, "clip-1")

    synchronous = runtime.token_for_ready_pairing(clip=clip, frames=frames)
    step = runtime.prepare_background_final_guard(clip=clip, frames=frames)
    assert step.preparation is not None
    attempt = runtime.run_background_final_guard_judge(step.preparation)
    manual = runtime.finalize_background_final_guard(step.preparation, attempt)

    assert synchronous == manual == "<ref_bg_1>"
    # No de-duplication: each pass through the boundary is paid for.
    assert counters["background_final_guard_attempted"] == 2
    assert counters["background_final_guard_accepted"] == 2


# ---------------------------------------------------------------------------
# M. Frozen cross-pair preparation ordering
# ---------------------------------------------------------------------------


def test_no_donor_means_target_evidence_is_never_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence materialization happens only after a non-empty donor list."""
    import r2v_data_v2.v3.pair as pair_module_local

    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="no-donor-order-test")
    # Same reference_type, but a different parent, so no donor is eligible.
    _add_ready_clip(
        config,
        storage,
        clip_uid="donor",
        clip_suffix="2",
        parent_video_id="unrelated-parent",
        entity_types=("subject",),
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target",
        clip_suffix="20",
        entity_types=("subject",),
    )

    recorded: list[str] = []
    real = pair_module_local.prepare_cross_pair_target_evidence

    def spy(*args: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs.get("clip_uid", "?"))
        return real(*args, **kwargs)

    monkeypatch.setattr(
        pair_module_local, "prepare_cross_pair_target_evidence", spy
    )

    stats = pair_clips(
        config,
        storage,
        judge=_ScopedJudge({("target", "e1"): "reject"}),
        cross_pair_judge=_CrossJudge([True]),
    )

    assert recorded == [], "evidence must not be built when there is no donor"
    assert stats.cross_pair_attempted == 0
    assert stats.failed == 0, "a missing donor is not a Pair failure"


def test_donor_image_load_failure_does_not_count_as_an_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cross_pair_attempted means the donor image was actually loaded.

    Legacy incremented the counter *after* ``Image.open(...).load()``, so a
    donor whose reference image cannot be read is not an attempt.
    """
    import r2v_data_v2.v3.pair as pair_module_local

    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)

    def explode(evidence: Any, donor: Any) -> Any:
        raise OSError("donor reference image is unreadable")

    monkeypatch.setattr(pair_module_local, "prepare_cross_pair_attempt", explode)

    cross = _CrossJudge([True])
    stats = pair_clips(
        config,
        storage,
        judge=_ScopedJudge({("target", "e1"): "reject", ("target", "e2"): "reject"}),
        cross_pair_judge=cross,
    )

    assert cross.calls == [], "no Qwen call without a readable donor image"
    assert stats.cross_pair_attempted == 0
    assert stats.failed == 1


def test_unreadable_donor_reference_is_filtered_from_the_donor_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Characterization: the donor index validates before the loop.

    The storage starts with the donor only, so the second pass really has a
    fresh target-new with no pairing; re-adding an existing clip_uid would
    leave its durable pairing in place and make this vacuous.
    """
    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = RunStorage(config)
    storage.initialize(git_commit="corrupt-donor-test")
    _add_ready_clip(
        config, storage, clip_uid="donor", clip_suffix="2", entity_types=("subject",)
    )

    first = pair_clips(config, storage, judge=_ScopedJudge())
    assert first.ready == 1, "donor publishes a ready full self reference"
    donor_path = storage.selected_entity_path("donor", "e1")
    assert donor_path.is_file()
    donor_path.write_bytes(b"not a png")

    # A genuinely new target: same parent, same reference type, no pairing.
    _add_ready_clip(
        config,
        storage,
        clip_uid="target-new",
        clip_suffix="20",
        entity_types=("subject",),
    )
    assert storage.read_clip("target-new").pairing is None

    cross = _CrossJudge([True])
    judge = _ScopedJudge({("target-new", "e1"): "reject"}, forbidden=("donor",))

    stats = pair_clips(config, storage, judge=judge, cross_pair_judge=cross)

    # The donor keeps its existing pairing and is never re-judged.
    assert ("donor", "e1") not in judge.calls
    # target-new really went through the primary pass.
    assert ("target-new", "e1") in judge.calls
    assert stats.processed == 1
    # _build_same_parent_donor_index validates the donor artifact, so an
    # unreadable reference never reaches the donor loop at all.
    assert cross.calls == []
    assert stats.cross_pair_attempted == 0
    assert stats.cross_pair_ready == 0
    assert storage.read_clip("target-new").pairing == PairingState(
        status="rejected", reason="no_qualifying_ready_reference"
    )
    # Recorded, not assumed: the corrupt donor's own existing-pairing
    # validation is what the current implementation reports as a failure.
    assert stats.skipped_existing == 0
    assert stats.failed == 1


def test_helper_target_evidence_context_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sampled_frames mode: ten frames, a contact sheet, no entity crop."""
    from r2v_data_v2.v3.frames import validate_sampled_frames
    from r2v_data_v2.v3.pair import prepare_cross_pair_target_evidence

    config = _config(
        tmp_path, monkeypatch, pair=PairConfig(same_parent_fallback_enabled=True)
    )
    storage = _two_entity_target_storage(config)
    target = storage.read_clip("target")
    evidence = prepare_cross_pair_target_evidence(
        config,
        storage,
        clip_uid="target",
        entity=target.annotation.entities[0],
        frames=validate_sampled_frames(storage, "target"),
        target_candidates=[],
    )

    assert evidence.evidence_mode == "sampled_frames"
    assert evidence.frame_slots == tuple(range(10))
    assert evidence.entity_crop is None
    assert evidence.context_image.size[0] > 0
