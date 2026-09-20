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

from collections.abc import Sequence
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

SHARD = "shard-000000000-000000000"

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


# ---------------------------------------------------------------------------
# N. Durable primary Pair epoch jobs (3a)
# ---------------------------------------------------------------------------


def _pair_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **pair_kwargs: Any):
    return _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(**pair_kwargs),
        reference_edit_enabled=True,
    )


def _runner(
    tmp_path: Path,
    config: V3Config,
    storage: RunStorage,
    *,
    shard: str = SHARD,
    clip_uids: Sequence[str] = ("clip-1",),
):
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    ledger = GroupLedger(tmp_path / "ledger")
    return PairEpochRunner(
        config,
        {shard: storage},
        ledger,
        eligible_clip_uids_by_shard={shard: list(clip_uids)},
    )


def _commit(runner: Any, job: Any, result: Any) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PAIR_PHASE

    phase = runner.ledger.phase(PAIR_PHASE)
    digest = phase.publish_result(job, result)
    receipt = phase.commit(
        job,
        outcome=result.outcome,
        artifact_digests={"result.json": digest},
        result_digest=digest,
    )
    runner.ledger.note_commit(PAIR_PHASE, receipt)


def _run_one(runner: Any, job: Any, judge: Any) -> Any:
    result = runner.run(job, judge)
    if result.committed:
        _commit(runner, job, result)
    return result


def _drain(runner: Any, judge: Any, *, limit: int = 32) -> list[Any]:
    """Run every seeded job in order, committing and finalizing each."""
    seen: list[Any] = []
    pending = list(runner.seed_primary_jobs())
    while pending and len(seen) < limit:
        job = pending.pop(0)
        seen.append(job)
        result = _run_one(runner, job, judge)
        if not result.committed:
            break
        pending.extend(runner.finalize(job, result))
    return seen


def test_pair_epoch_requires_reference_edit_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError, PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    config = _config(tmp_path, monkeypatch, reference_edit_enabled=False)
    storage = _storage(config, entity_types=("subject",))
    with pytest.raises(PairEpochError, match="reference_edit.enabled"):
        PairEpochRunner(
            config,
            {SHARD: storage},
            GroupLedger(tmp_path / "ledger"),
            eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
        )


def test_seed_primary_jobs_seeds_only_the_first_qwen_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    jobs = runner.seed_primary_jobs()

    assert len(jobs) == 1, "a clip holds at most one primary Qwen job"
    assert dict(jobs[0].target)["entity_id"] == "e1"
    assert jobs[0].job_type == "pair_entity_reference_judge"
    assert jobs[0].resource == "qwen"


def test_finalize_unlocks_one_entity_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)
    judge = _Judge()

    (first,) = runner.seed_primary_jobs()
    result = _run_one(runner, first, judge)
    (second,) = runner.finalize(first, result)

    assert [dict(job.target)["entity_id"] for job in (first, second)] == ["e1", "e2"]
    assert storage.read_clip("clip-1").pairing is None, "not published mid-chain"


def test_entity_judge_failure_leaves_no_receipt_and_no_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_jobs import OUTCOME_RETRYABLE_FAILED

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    (e1,) = runner.seed_primary_jobs()
    assert _run_one(runner, e1, _Judge()).committed
    pending = runner.seed_primary_jobs()
    assert [dict(job.target)["entity_id"] for job in pending] == ["e2"]

    failing = _FailingEntityJudge(fail_on="e2")
    (e2,) = pending
    result = runner.run(e2, failing)

    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert not result.committed, "a judge failure must not leave a receipt"
    assert storage.read_clip("clip-1").pairing is None
    # e3 is never planned while e2 is unresolved.
    assert [dict(job.target)["entity_id"] for job in runner.seed_primary_jobs()] == ["e2"]


def test_primary_replay_does_not_repeat_qwen_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash after the e1 receipt, before the finalizer: no extra Qwen."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)

    (e1,) = runner.seed_primary_jobs()
    _run_one(runner, e1, _Judge())
    # Crash here: no finalize.

    resumed = _runner(tmp_path, config, storage)
    calls: list[str] = []

    class _CountingJudge(_Judge):
        def decide(self, **kwargs: Any) -> Any:
            calls.append(kwargs["entity"].entity_id)
            return super().decide(**kwargs)

    (again,) = resumed.seed_primary_jobs()
    assert dict(again.target)["entity_id"] == "e2", "e1 receipt is reused"
    replayed = _run_one(resumed, again, _CountingJudge())
    resumed.finalize(again, replayed)

    assert calls == ["e2"], "e1 was never re-judged"
    assert storage.read_clip("clip-1").pairing is not None


def test_primary_publication_survives_a_crash_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All entity receipts committed, publication crashed: resume publishes."""
    import hashlib

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    runner = _runner(tmp_path, config, storage)

    (e1,) = runner.seed_primary_jobs()
    _run_one(runner, e1, _Judge())
    # Crash before finalize/publish.

    resumed = _runner(tmp_path, config, storage)
    assert resumed.primary_unresolved_job_ids() == (), "primary work is complete"
    pending = resumed.seed_primary_jobs()
    assert pending == []
    # The finalizer replays from receipts and publishes.
    published = resumed._finalize_primary_clip(e1)
    assert published == ()
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "ready"
    path = storage.selected_entity_path("clip-1", "e1")
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_pairing_is_not_a_fresh_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    assert storage.read_clip("clip-1").pairing is not None

    runner = _runner(tmp_path, config, storage)
    jobs = runner.seed_primary_jobs()

    assert jobs == [], "an existing pairing is never re-judged"


def test_primary_guard_accept_binds_the_background_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    guard = _FinalBackgroundJudge(accepted=True)

    (entity_job,) = runner.seed_primary_jobs()
    entity_result = _run_one(runner, entity_job, _Judge())
    (guard_job,) = runner.finalize(entity_job, entity_result)
    assert guard_job.job_type == "pair_background_final_guard"
    assert dict(guard_job.target)["call_site"] == "primary"

    guard_result = _run_one(runner, guard_job, guard)
    runner.finalize(guard_job, guard_result)

    assert len(guard.calls) == 1, "exactly one primary guard Qwen call"
    clip = storage.read_clip("clip-1")
    assert clip.pairing.status == "ready"
    assert clip.pairing.background_token == "<ref_bg_1>"


def test_primary_guard_failure_is_committed_and_publishes_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudgeFailure
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    guard = _FinalBackgroundJudge(
        error=FinalBackgroundJudgeFailure("structured output invalid")
    )

    (entity_job,) = runner.seed_primary_jobs()
    entity_result = _run_one(runner, entity_job, _Judge())
    (guard_job,) = runner.finalize(entity_job, entity_result)
    result = _run_one(runner, guard_job, guard)

    assert result.committed, "a guard failure is durable, never a scheduler retry"
    assert result.payload["status"] == "failed_closed"
    runner.finalize(guard_job, result)
    clip = storage.read_clip("clip-1")
    assert clip.pairing.status == "ready"
    assert clip.pairing.background_token is None


# ---------------------------------------------------------------------------
# O. Primary resume hardening regression
# ---------------------------------------------------------------------------


def test_frozen_plan_keeps_fresh_classification_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh target stays fresh even after this invocation published it."""
    from r2v_data_v2.v3.post_mask_epoch_pair import CLIP_FRESH_TARGET

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    runner = _runner(tmp_path, config, storage)
    (job,) = runner.seed_primary_jobs()
    result = _run_one(runner, job, _Judge())
    runner.finalize(job, result)
    assert storage.read_clip("clip-1").pairing is not None

    restarted = _runner(tmp_path, config, storage)
    plan = restarted._primary_plan(SHARD)
    assert plan["clips"]["clip-1"]["classification"] == CLIP_FRESH_TARGET
    assert restarted.seed_primary_jobs() == []


def test_deterministic_only_primary_publishes_rejected_with_zero_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",), tracking_status={"e1": "failed"})

    runner = _runner(tmp_path, config, storage)
    jobs = runner.seed_primary_jobs()

    assert jobs == []
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "rejected"


def test_existing_pairing_is_validated_like_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    runner = _runner(tmp_path, config, storage)

    plan = runner._primary_plan(SHARD)
    assert plan["clips"]["clip-1"]["classification"] == "existing_pairing"
    assert runner.stats[SHARD]["failed"] == 0
    assert runner.seed_primary_jobs() == []


def test_corrupt_existing_pairing_is_a_pair_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    storage.selected_entity_path("clip-1", "e1").write_bytes(b"not a png")

    runner = _runner(tmp_path, config, storage)
    runner._primary_plan(SHARD)

    assert runner.stats[SHARD]["failed"] == 1
    assert runner.seed_primary_jobs() == []


def test_guard_input_drift_is_not_committed_as_failed_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    guard = _FinalBackgroundJudge(accepted=True)

    (entity_job,) = runner.seed_primary_jobs()
    entity_result = _run_one(runner, entity_job, _Judge())
    (guard_job,) = runner.finalize(entity_job, entity_result)
    # The planned background input disappears before the job runs.
    frame = storage.root / "clips/clip-1/frames/00.jpg"
    frame.write_bytes(b"corrupted")

    result = runner.run(guard_job, guard)

    assert not result.committed, "drift must not commit under the old job identity"
    assert len(guard.calls) == 0


def test_primary_counts_do_not_multiply_across_seed_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    runner = _runner(tmp_path, config, storage)
    (job,) = runner.seed_primary_jobs()
    result = _run_one(runner, job, _Judge())
    runner.finalize(job, result)
    first = dict(runner.stats[SHARD])

    runner.seed_primary_jobs()
    runner.primary_unresolved_job_ids()
    restarted = _runner(tmp_path, config, storage)
    restarted.seed_primary_jobs()

    assert dict(runner.stats[SHARD]) == first
    assert restarted.stats[SHARD]["entities_ready"] == 0
    assert restarted.stats[SHARD]["ready"] == 0


def test_entity_job_identity_follows_the_canonical_request_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    runner = _runner(tmp_path, config, storage)
    (job,) = runner.seed_primary_jobs()

    assert job.job_id()
    # Same inputs, different execution-only context -> same identity.
    again_runner = _runner(tmp_path / "again", config, storage)
    (same,) = again_runner.seed_primary_jobs()
    assert same.job_id() == job.job_id()


def test_guard_job_identity_includes_image_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from r2v_data_v2.v3.post_mask_epoch_pair import _image_identity

    flat_a = Image.new("RGB", (2, 2), (7, 7, 7))
    flat_b = Image.new("RGB", (1, 4), (7, 7, 7))

    assert flat_a.tobytes() == flat_b.tobytes(), "same raw bytes on purpose"
    assert _image_identity(flat_a) != _image_identity(flat_b)


# ---------------------------------------------------------------------------
# P. Real scheduler crash recovery
# ---------------------------------------------------------------------------


class _PairExecutor:
    """Pair-aware fake Qwen executor driving runner.run() per job."""

    def __init__(self, runner: Any, entity_judge: Any, guard_judge: Any = None) -> None:
        self.runner = runner
        self.entity_judge = entity_judge
        self.guard_judge = guard_judge
        self.calls: list[str] = []

    def execute_batch(self, jobs: Sequence[Any]) -> dict[str, Any]:
        from r2v_data_v2.v3.post_mask_epoch_pair import (
            PAIR_ENTITY_JUDGE_JOB,
        )
        from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

        outcomes: dict[str, Any] = {}
        for job in jobs:
            self.calls.append(job.job_type)
            judge = (
                self.entity_judge
                if job.job_type == PAIR_ENTITY_JUDGE_JOB
                else self.guard_judge
            )
            try:
                result = self.runner.run(job, judge)
            except Exception as exc:  # noqa: BLE001
                outcomes[job.job_id()] = JobExecution(
                    job=job, result=None, exception=exc
                )
            else:
                outcomes[job.job_id()] = JobExecution(
                    job=job, result=result, exception=None
                )
        return outcomes


def _scheduler(ledger: Any, runner: Any, executor: Any) -> Any:
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler

    return ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={RESOURCE_QWEN: executor},
    )


def _fail_publish_once(monkeypatch: pytest.MonkeyPatch, times: int = 1) -> Any:
    """Fault-inject publication. Returns ``(state, restore)``.

    Restoration is explicit rather than ``monkeypatch.undo()``, which would
    also undo the config root patching the fixtures depend on.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    real = pm._publish_pair_result
    state = {"remaining": times, "calls": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        state["calls"] += 1
        if state["remaining"] > 0:
            state["remaining"] -= 1
            raise RuntimeError("publication crashed")
        return real(*args, **kwargs)

    pm._publish_pair_result = flaky

    def restore() -> None:
        pm._publish_pair_result = real

    return state, restore


def test_scheduler_resumes_after_publication_crash_on_last_entity_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entity receipt committed, finalizer crashed: restart reuses the receipt."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    root = tmp_path / "ledger"

    runner = _runner(root, config, storage)
    judge = _Judge()
    executor = _PairExecutor(runner, judge)
    state, restore_publish = _fail_publish_once(monkeypatch)
    scheduler = _scheduler(runner.ledger, runner, executor)
    try:
        scheduler.run(runner.seed_primary_jobs())
    except RuntimeError:
        pass

    assert executor.calls == ["pair_entity_reference_judge"]
    assert state["calls"] == 1, "publication was attempted exactly once"
    assert storage.read_clip("clip-1").pairing is None
    assert not storage.selected_entity_path("clip-1", "e1").is_file()

    # Real restart: new ledger, new runner, new scheduler, no memory reuse.
    restore_publish()
    restarted = _runner(root, config, storage)
    restarted_executor = _PairExecutor(restarted, judge)
    restarted_scheduler = _scheduler(restarted.ledger, restarted, restarted_executor)
    outcome = restarted_scheduler.run(restarted.seed_primary_jobs())

    assert restarted_executor.calls == [], "no extra Qwen: the receipt was reused"
    assert judge.calls and len(judge.calls) == 1
    assert outcome["completed"] is True
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "ready"
    assert storage.selected_entity_path("clip-1", "e1").is_file()


def test_scheduler_resumes_after_publication_crash_on_guard_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard receipt committed, publication crashed: restart replays CPU only."""
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    root = tmp_path / "ledger"

    runner = _runner(root, config, storage)
    judge = _Judge()
    guard = _FinalBackgroundJudge(accepted=True)
    executor = _PairExecutor(runner, judge, guard)
    _, restore_publish = _fail_publish_once(monkeypatch)
    scheduler = _scheduler(runner.ledger, runner, executor)
    try:
        scheduler.run(runner.seed_primary_jobs())
    except RuntimeError:
        pass

    assert sorted(executor.calls) == [
        "pair_background_final_guard",
        "pair_entity_reference_judge",
    ]
    assert len(judge.calls) == 1
    assert len(guard.calls) == 1
    assert storage.read_clip("clip-1").pairing is None

    restore_publish()
    restarted = _runner(root, config, storage)
    restarted_executor = _PairExecutor(restarted, judge, guard)
    restarted_scheduler = _scheduler(restarted.ledger, restarted, restarted_executor)
    outcome = restarted_scheduler.run(restarted.seed_primary_jobs())

    assert restarted_executor.calls == [], "no extra Qwen for entity or guard"
    assert len(judge.calls) == 1
    assert len(guard.calls) == 1
    assert outcome["completed"] is True
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "ready"
    assert clip.pairing.background_token == "<ref_bg_1>"
    assert storage.selected_entity_path("clip-1", "e1").is_file()


def test_frozen_plan_input_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    root = tmp_path / "ledger"
    runner = _runner(root, config, storage)
    runner.seed_primary_jobs()

    # Mutate a frozen pre-Pair input: the annotation semantics.
    clip = storage.read_clip("clip-1")
    assert clip.annotation is not None
    entity = clip.annotation.entities[0]
    drifted = clip.annotation.model_copy(
        update={
            "entities": [
                entity.model_copy(update={"phrase": "drifted phrase"}),
                *clip.annotation.entities[1:],
            ]
        }
    )
    storage.write_annotation("clip-1", drifted)

    restarted = _runner(root, config, storage)
    judge = _Judge()
    with pytest.raises(PairEpochError, match="frozen primary input drifted"):
        restarted.seed_primary_jobs()
    assert judge.calls == []


def test_publication_retry_after_failed_first_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed publication must not leave anything durable behind."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    runner = _runner(tmp_path, config, storage)
    (job,) = runner.seed_primary_jobs()
    result = _run_one(runner, job, _Judge())

    state, restore_publish = _fail_publish_once(monkeypatch)
    with pytest.raises(RuntimeError, match="publication crashed"):
        runner.finalize(job, result)

    assert state["calls"] == 1
    assert storage.read_clip("clip-1").pairing is None
    assert not storage.selected_entity_path("clip-1", "e1").is_file()
    accounted = runner.ledger.root / "semantic/pair/accounted"
    assert not (
        accounted / f"published-{SHARD}-clip-1.json"
    ).exists(), "no durable published marker after a failed publish"

    restore_publish()
    resumed = _runner(tmp_path, config, storage)
    assert resumed.seed_primary_jobs() == []
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None
    assert clip.pairing.status == "ready"
    assert storage.selected_entity_path("clip-1", "e1").is_file()
    # The marker only follows a successful publication; it never decides
    # whether publication runs.
    assert (accounted / f"published-{SHARD}-clip-1.json").exists()


def test_guard_counters_count_one_call_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    guard = _FinalBackgroundJudge(accepted=True)

    (entity_job,) = runner.seed_primary_jobs()
    entity_result = _run_one(runner, entity_job, _Judge())
    (guard_job,) = runner.finalize(entity_job, entity_result)
    guard_result = _run_one(runner, guard_job, guard)
    runner.finalize(guard_job, guard_result)

    counters = runner._guard_counters[SHARD]
    assert counters["background_final_guard_attempted"] == 1
    assert counters["background_final_guard_accepted"] == 1
    assert counters["background_final_guard_failed_closed"] == 0
    assert len(guard.calls) == 1


def test_guard_failure_counters_count_one_failure_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudgeFailure
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    guard = _FinalBackgroundJudge(error=FinalBackgroundJudgeFailure("bad json"))

    (entity_job,) = runner.seed_primary_jobs()
    entity_result = _run_one(runner, entity_job, _Judge())
    (guard_job,) = runner.finalize(entity_job, entity_result)
    guard_result = _run_one(runner, guard_job, guard)
    runner.finalize(guard_job, guard_result)

    counters = runner._guard_counters[SHARD]
    assert counters["background_final_guard_attempted"] == 1
    assert counters["background_final_guard_failed_closed"] == 1
    assert counters["background_final_guard_accepted"] == 0
    clip = storage.read_clip("clip-1")
    assert clip.pairing.status == "ready"
    assert clip.pairing.background_token is None


def test_existing_pairing_is_revalidated_after_plan_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation must run every time; only the diagnostic is de-duplicated."""
    import r2v_data_v2.v3.pair as pair_module

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    root = tmp_path / "ledger"

    runner = _runner(root, config, storage)
    plan = runner._primary_plan(SHARD)
    assert plan["clips"]["clip-1"]["classification"] == "existing_pairing"
    assert runner.stats[SHARD]["failed"] == 0

    real_validate = pair_module._validate_existing_pairing
    seen: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("clip", args[2] if len(args) > 2 else None).clip_uid)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(pair_module, "_validate_existing_pairing", spy)

    # The clip is still healthy: validation runs and passes again.
    revalidated = _runner(root, config, storage)
    revalidated._primary_plan(SHARD)
    assert seen == ["clip-1"], "existing pairing is validated on every load"
    assert revalidated.stats[SHARD]["failed"] == 0

    # Now corrupt the published reference after the plan already exists.
    storage.selected_entity_path("clip-1", "e1").write_bytes(b"not a png")

    after = _runner(root, config, storage)
    after._primary_plan(SHARD)
    assert len(seen) == 2, "validation is not skipped by any success marker"
    assert after.stats[SHARD]["failed"] == 1
    assert after.seed_primary_jobs() == [], "no primary Qwen for an existing pairing"


def test_existing_pairing_failure_diagnostic_does_not_grow_unbounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same corruption is accounted once, not once per status query."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    pair_clips(config, storage, judge=_Judge())
    storage.selected_entity_path("clip-1", "e1").write_bytes(b"not a png")
    root = tmp_path / "ledger"

    runner = _runner(root, config, storage)
    runner._primary_plan(SHARD)
    assert runner.stats[SHARD]["failed"] == 1

    # Repeated loads still validate; only the accounting is de-duplicated.
    runner._primary_plan(SHARD)
    runner._primary_plan(SHARD)
    assert runner.stats[SHARD]["failed"] == 1, "same failure is not counted again"

    # A fresh runner keeps the already-recorded diagnostic from growing.
    restarted = _runner(root, config, storage)
    restarted._primary_plan(SHARD)
    assert restarted.stats[SHARD]["failed"] == 0


# ---------------------------------------------------------------------------
# Q. Frozen per-shard donor snapshot (3b)
# ---------------------------------------------------------------------------


def _three_donor_shard(config: V3Config, storage: RunStorage) -> None:
    """One ready full donor plus two rejected targets, same parent."""
    for uid, suffix in (("donor", "2"),):
        _add_ready_clip(
            config, storage, clip_uid=uid, clip_suffix=suffix, entity_types=("subject",)
        )
    for uid, suffix in (("target-b", "20"), ("target-c", "21")):
        _add_ready_clip(
            config, storage, clip_uid=uid, clip_suffix=suffix, entity_types=("subject",)
        )


def _drain_primary(runner: Any, judge: Any = None) -> None:
    """Run every primary job to its terminal state."""
    judge = judge if judge is not None else _Judge()
    pending = list(runner.seed_primary_jobs())
    guard: Any = None
    while pending:
        job = pending.pop(0)
        result = _run_one(runner, job, judge if guard is None else guard)
        if not result.committed:
            break
        pending.extend(runner.finalize(job, result))


def test_donor_snapshot_order_matches_the_live_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.pair import _build_same_parent_donor_index
    from r2v_data_v2.v3.post_mask_runtime import _ShardStorage

    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    runner.seed_primary_jobs()

    snapshot = runner.freeze_donor_snapshot(SHARD)
    live = _build_same_parent_donor_index(
        config,
        _ShardStorage(
            storage, ("clip-1", "donor", "target-b", "target-c")
        ),
    )
    for group in snapshot["groups"]:
        key = (group["parent_video_id"], group["reference_type"])
        expected = [d.clip.clip_uid for d in live.get(key, ())]
        assert [d["clip_uid"] for d in group["donors"]] == expected
        assert [d["ordinal"] for d in group["donors"]] == list(range(len(expected)))


def test_donor_snapshot_restore_feeds_donors_for_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.pair import _donors_for_target

    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    # Only clip-1 and donor publish a full reference, so they are the donors.
    _drain_primary(
        runner,
        _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
    )
    runner.freeze_donor_snapshot(SHARD)

    index = runner.frozen_donor_index(SHARD)
    target = storage.read_clip("target-b")
    donors = _donors_for_target(
        config, index, target_clip=target, target_entity=target.annotation.entities[0]
    )
    # clip-1 also published a full reference, so it is a donor too; target-b
    # itself is excluded by pair.py's self exclusion.
    assert [d.clip.clip_uid for d in donors] == ["donor", "clip-1"]
    assert "target-b" not in [d.clip.clip_uid for d in donors]


def test_donor_snapshot_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    _drain_primary(runner)
    runner.freeze_donor_snapshot(SHARD)

    storage.selected_entity_path("donor", "e1").write_bytes(b"tampered")
    restarted = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    with pytest.raises(PairEpochError, match="selected PNG changed"):
        restarted.frozen_donor_index(SHARD)


def test_second_freeze_never_rebuilds_the_live_donor_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the first freeze, donor membership is never re-derived.

    A spy on the live builder is the direct proof: if it is not called again,
    a full reference produced later by cross-pair can neither expand the
    snapshot nor make it mismatch.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _three_donor_shard(config, storage)
    uid_list = ("clip-1", "donor", "target-b", "target-c")
    runner = _runner(tmp_path, config, storage, clip_uids=uid_list)
    _drain_primary(
        runner,
        _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
    )

    real_builder = pm._build_same_parent_donor_index
    live_calls: list[int] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        live_calls.append(1)
        return real_builder(*args, **kwargs)

    pm._build_same_parent_donor_index = spy
    try:
        first = runner.freeze_donor_snapshot(SHARD)
        again = runner.freeze_donor_snapshot(SHARD)
        restarted = _runner(tmp_path, config, storage, clip_uids=uid_list)
        third = restarted.freeze_donor_snapshot(SHARD)
    finally:
        pm._build_same_parent_donor_index = real_builder

    assert len(live_calls) == 1, "the live donor index is built exactly once"
    assert again == first, "a second freeze returns the frozen snapshot"
    assert third == first, "a restart reuses the same frozen snapshot"
    donors = [d["clip_uid"] for g in first["groups"] for d in g["donors"]]
    assert "target-b" not in donors and "target-c" not in donors


def test_unresolved_primary_is_excluded_from_snapshot_and_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unresolved clip is excluded; the rest of the shard is not blocked."""
    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="clip-b", clip_suffix="20", entity_types=("subject",)
    )
    _add_ready_clip(
        config, storage, clip_uid="clip-c", clip_suffix="21", entity_types=("subject",)
    )
    runner = _runner(
        tmp_path, config, storage, clip_uids=("clip-1", "clip-b", "clip-c")
    )

    # clip-b raises on its entity judge and stays unresolved.
    pending = runner.seed_primary_jobs()
    while pending:
        job = pending.pop(0)
        if job.clip_uid == "clip-b":
            result = runner.run(job, _FailingEntityJudge(fail_on="e1"))
            assert not result.committed
            continue
        result = _run_one(runner, job, _Judge())
        if result.committed:
            pending.extend(runner.finalize(job, result))

    blocked = runner._unresolved_primary_clip_uids(SHARD, ())
    snapshot = runner.freeze_donor_snapshot(SHARD, unresolved_job_ids=())
    donors = [d["clip_uid"] for g in snapshot["groups"] for d in g["donors"]]

    assert "clip-b" in blocked
    assert "clip-b" not in snapshot["cross_pair_target_clip_uids"]
    assert "clip-b" not in donors
    # The rest of the shard still participates.
    assert "clip-1" in snapshot["cross_pair_target_clip_uids"]
    assert "clip-c" in snapshot["cross_pair_target_clip_uids"]


def test_donor_domain_is_scoped_to_one_canonical_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard-A snapshot never sees a shard-B donor, same parent/type."""
    from dataclasses import replace

    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    shard_a = _storage(config, entity_types=("subject",))
    _add_ready_clip(
        config,
        shard_a,
        clip_uid="donor-a",
        clip_suffix="2",
        entity_types=("subject",),
    )
    _add_ready_clip(
        config,
        shard_a,
        clip_uid="target-a",
        clip_suffix="20",
        entity_types=("subject",),
    )
    shard_b_config = replace(
        config, run_root=config.run_root.parent / "shard-b-run"
    )
    shard_b = RunStorage(shard_b_config)
    shard_b.initialize(git_commit="shard-b")
    _add_ready_clip(
        config,
        shard_b,
        clip_uid="donor-b",
        clip_suffix="3",
        entity_types=("subject",),
    )
    assert shard_b.read_clip("donor-b").source.parent_video_id == "parent"

    runner = PairEpochRunner(
        config,
        {"shard-a": shard_a, "shard-b": shard_b},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={
            "shard-a": ["clip-1", "donor-a", "target-a"],
            "shard-b": ["donor-b"],
        },
    )
    _drain_primary_shard(runner, "shard-a", shard_a)
    _drain_primary_shard(runner, "shard-b", shard_b)
    snapshot = runner.freeze_donor_snapshot("shard-a")
    donors = [d["clip_uid"] for g in snapshot["groups"] for d in g["donors"]]

    assert "donor-a" in donors
    assert "donor-b" not in donors


def _drain_primary_shard(runner: Any, shard: str, storage: RunStorage) -> None:
    pending = list(runner.seed_primary_jobs())
    pending = [job for job in pending if job.canonical_shard == shard]
    while pending:
        job = pending.pop(0)
        result = _run_one(runner, job, _Judge())
        if not result.committed:
            break
        pending.extend(
            job
            for job in runner.finalize(job, result)
            if job.canonical_shard == shard
        )
