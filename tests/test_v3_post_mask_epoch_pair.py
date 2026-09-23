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

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

from r2v_data_v2.v3.config import PairConfig, V3Config
from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure
from r2v_data_v2.v3.pair import pair_clips
from r2v_data_v2.v3.post_mask_epoch_pair import (
    PAIR_BACKGROUND_GUARD_JOB,
    PAIR_ENTITY_JUDGE_JOB,
)
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


def test_qwen_endpoint_handle_rebinds_frozen_dataclass_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Managed Qwen endpoints rebind dataclass services without Pydantic APIs."""
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    config = _pair_config(tmp_path, monkeypatch)
    service = config.qwen.candidate_judge
    assert service is not None
    captured: dict[str, Any] = {}

    def factory(bound_service: Any) -> object:
        captured["service"] = bound_service
        return object()

    endpoint = "http://127.0.0.1:8000/v1"
    resolved = pm._resolve_qwen_judge(endpoint, service, factory)

    rebound = captured["service"]
    assert resolved.owned is True
    assert rebound is not service
    assert rebound.base_url == endpoint
    assert rebound.model == service.model
    assert rebound.api_key == service.api_key


def _runner(
    tmp_path: Path,
    config: V3Config,
    storage: RunStorage,
    *,
    shard: str = SHARD,
    clip_uids: Sequence[str] = ("clip-1",),
    ledger_dir: str = "ledger",
    cpu_workers: int | None = None,
):
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    ledger = GroupLedger(tmp_path / ledger_dir)
    return PairEpochRunner(
        config,
        {shard: storage},
        ledger,
        eligible_clip_uids_by_shard={shard: list(clip_uids)},
        cpu_workers=cpu_workers,
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


def test_seed_primary_jobs_seeds_every_independent_qwen_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One clip's entity judges are independent: all of them are seeded."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    jobs = runner.seed_primary_jobs()

    # A primary entity judge depends only on the clip, its frames and masks,
    # that entity and its own candidates, so nothing here is serialised. The
    # seed is sorted by job id, so the entity order is compared as a set.
    assert sorted(dict(job.target)["entity_id"] for job in jobs) == ["e1", "e2", "e3"]
    assert {job.job_type for job in jobs} == {"pair_entity_reference_judge"}
    assert {job.resource for job in jobs} == {"qwen"}
    assert len({job.job_id() for job in jobs}) == 3


def _temporary_pair_pngs(storage: Any, clip_uid: str = "clip-1") -> list[str]:
    return sorted(
        path.name
        for path in (storage.root / "clips" / clip_uid).rglob("*")
        if path.is_file() and ".tmp-pair-" in path.name
    )


def test_intermediate_entity_completion_does_not_replay_the_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the last primary receipt publishes the clip, and it prepares nothing.

    An intermediate completion has nothing to do: replaying the clip would
    re-run every entity's prepare, crop and PNG encode and then throw the PNGs
    away again, once per sibling. The barrier is answered from the
    execution-only index, so the prepare count must not move at all. The
    terminal pass finalizes from the row this same invocation already prepared,
    so it must not prepare the clip a second time either.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    real_prepare = pm.prepare_entity_reference
    prepared: list[str] = []

    def counting_prepare(*args: Any, **kwargs: Any) -> Any:
        prepared.append(str(kwargs["entity"].entity_id))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(pm, "prepare_entity_reference", counting_prepare)
    seeded = _seeded_by_entity(runner)
    assert sorted(seeded) == ["e1", "e2", "e3"]
    after_seed = len(prepared)
    assert after_seed == 3, "the seed prepares each entity exactly once"

    for entity_id in ("e1", "e2"):
        result = _run_one(runner, seeded[entity_id], _Judge())
        assert result.committed
        # Only the finalizer is under test here: it must not prepare anything.
        before_finalize = len(prepared)
        unlocked = runner.finalize(seeded[entity_id], result)
        assert unlocked == (), f"{entity_id} is intermediate: nothing is unlocked"
        assert len(prepared) == before_finalize, f"{entity_id} must not replay"
        assert _temporary_pair_pngs(storage) == [], "no throwaway PNG is written"
        assert storage.read_clip("clip-1").pairing is None, "not published mid-chain"

    last = _run_one(runner, seeded["e3"], _Judge())
    assert last.committed
    before_finalize = len(prepared)
    runner.finalize(seeded["e3"], last)

    # The terminal pass reuses this invocation's prepared row: zero re-prepare.
    assert len(prepared) == before_finalize, (
        "the hot finalizer must not prepare the clip again"
    )
    assert runner.prepare_counters["primary_hot_finalize_cache_hits"] == 1
    assert runner.prepare_counters["primary_hot_finalize_replay_fallbacks"] == 0
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.pairing.retained_entity_ids == ["e1", "e2", "e3"]
    assert _temporary_pair_pngs(storage) == [], "publication consumed them"
    assert runner.primary_unresolved_job_ids() == ()


def test_entity_judge_failure_leaves_no_receipt_and_no_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_jobs import OUTCOME_RETRYABLE_FAILED

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    seeded = _seeded_by_entity(runner)
    assert _run_one(runner, seeded["e1"], _Judge()).committed
    pending = runner.seed_primary_jobs()
    assert sorted(dict(job.target)["entity_id"] for job in pending) == ["e2", "e3"]

    failing = _FailingEntityJudge(fail_on="e2")
    result = runner.run(seeded["e2"], failing)

    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert not result.committed, "a judge failure must not leave a receipt"
    # One unresolved entity is enough to hold the whole clip un-published.
    assert storage.read_clip("clip-1").pairing is None
    assert sorted(
        dict(job.target)["entity_id"] for job in runner.seed_primary_jobs()
    ) == ["e2", "e3"]


def test_primary_replay_does_not_repeat_qwen_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash after the e1 receipt, before the finalizer: no extra Qwen."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)

    seeded = {
        str(dict(job.target)["entity_id"]): job for job in runner.seed_primary_jobs()
    }
    _run_one(runner, seeded["e1"], _Judge())
    # Crash here: no finalize.

    resumed = _runner(tmp_path, config, storage)
    calls: list[str] = []

    class _CountingJudge(_Judge):
        def decide(self, **kwargs: Any) -> Any:
            calls.append(kwargs["entity"].entity_id)
            return super().decide(**kwargs)

    (again,) = resumed.seed_primary_jobs()
    assert dict(again.target)["entity_id"] == "e2", "e1 receipt is reused"
    assert again.job_id() == seeded["e2"].job_id()
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
    """Pair-aware fake Qwen executor driving runner.run() per job.

    Three-way dispatch: a cross job must never be handed to the guard judge.
    """

    def __init__(
        self,
        runner: Any,
        entity_judge: Any = None,
        guard_judge: Any = None,
        cross_judge: Any = None,
    ) -> None:
        self.runner = runner
        self.entity_judge = entity_judge
        self.guard_judge = guard_judge
        self.cross_judge = cross_judge
        self.calls: list[dict[str, str]] = []

    @property
    def job_types(self) -> list[str]:
        return [call["job_type"] for call in self.calls]

    def execute_batch(self, jobs: Sequence[Any]) -> dict[str, Any]:
        from r2v_data_v2.v3.post_mask_epoch_pair import (
            PAIR_CROSS_JUDGE_JOB,
            PAIR_ENTITY_JUDGE_JOB,
        )
        from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

        outcomes: dict[str, Any] = {}
        for job in jobs:
            self.calls.append(
                {
                    "job_type": job.job_type,
                    "clip_uid": job.clip_uid,
                    "job_id": job.job_id(),
                }
            )
            if job.job_type == PAIR_ENTITY_JUDGE_JOB:
                judge = self.entity_judge
            elif job.job_type == PAIR_CROSS_JUDGE_JOB:
                judge = self.cross_judge
            else:
                judge = self.guard_judge
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

    assert executor.job_types == ["pair_entity_reference_judge"]
    assert state["calls"] == 1, "publication was attempted exactly once"
    assert storage.read_clip("clip-1").pairing is None
    assert not storage.selected_entity_path("clip-1", "e1").is_file()

    # Real restart: new ledger, new runner, new scheduler, no memory reuse.
    restore_publish()
    restarted = _runner(root, config, storage)
    restarted_executor = _PairExecutor(restarted, judge)
    restarted_scheduler = _scheduler(restarted.ledger, restarted, restarted_executor)
    outcome = restarted_scheduler.run(restarted.seed_primary_jobs())

    assert restarted_executor.job_types == [], "no extra Qwen: receipt reused"
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

    assert sorted(executor.job_types) == [
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

    assert restarted_executor.job_types == [], "no extra Qwen for entity/guard"
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


def test_cross_baseline_freezes_the_primary_state_before_cross_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay must start from the frozen PRIMARY state, not the live one."""
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
    snapshot = runner.freeze_donor_snapshot(SHARD)
    assert "target-b" in snapshot["cross_pair_target_clip_uids"]

    baseline = runner.cross_baseline(SHARD, "target-b")
    live = storage.read_clip("target-b")
    assert baseline["primary_pairing"] == live.pairing.model_dump(mode="json")
    assert baseline["primary_entity_references"] == [
        state.model_dump(mode="json") for state in live.references.entities
    ]
    # The rejected primary outcome is what replay must see, not an upgrade.
    assert baseline["primary_entity_references"][0]["status"] == "rejected"

    # Durable and create-or-validate: a restart reads the same primary
    # baseline, so cross replay always starts from the pre-cross state.
    restarted = _runner(tmp_path, config, storage, clip_uids=uid_list)
    assert restarted.cross_baseline(SHARD, "target-b") == baseline
    restarted.freeze_donor_snapshot(SHARD)
    assert restarted.cross_baseline(SHARD, "target-b") == baseline


def test_barrier_seeds_exactly_one_donor1_job_per_frozen_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One chain head per target, donor1 only, never donor2 in advance."""
    from r2v_data_v2.v3.post_mask_epoch_pair import PAIR_CROSS_JUDGE_JOB

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

    jobs = runner.freeze_cross_pair_after_primary_quiescence()

    targets = sorted(job.clip_uid for job in jobs)
    assert targets == ["target-b", "target-c"]
    for job in jobs:
        assert job.job_type == PAIR_CROSS_JUDGE_JOB
        assert job.resource == "qwen"
        assert dict(job.target)["donor_ordinal"] == "0"
        assert dict(job.target)["donor_clip_uid"] in {"donor", "clip-1"}
    # A second barrier call must not create donor2 out of nothing.
    assert runner.freeze_cross_pair_after_primary_quiescence() == tuple(jobs)


def _cross_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
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
    _drain_primary(
        runner,
        _ScopedJudge({("target-b", "e1"): "reject", ("target-c", "e1"): "reject"}),
    )
    return config, storage, runner


def test_cross_pair_judge_failure_is_durable_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structured failure carrier survives and never becomes retryable."""
    import json

    from r2v_data_v2.structured_output import ValidationIssue
    from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    failure = CrossPairJudgeFailure(
        raw_responses=["bad"],
        issues=[
            ValidationIssue(
                code="schema_invalid", field="verdict", message="missing verdict"
            )
        ],
        attempt_count=1,
    )

    judge_calls = {"count": 0}

    class _ExplodingJudge:
        def decide(self, **kwargs: Any) -> Any:
            judge_calls["count"] += 1
            raise failure

    result = runner.run(job, _ExplodingJudge())
    assert result.committed, "a judge failure is committed, not retried"
    assert result.payload["status"] == "cross_pair_failed"
    assert result.payload["failure"]["attempt_count"] == 1
    assert result.payload["failure"]["raw_responses"] == ["bad"]

    primary_before = storage.read_clip("target-b").pairing
    _commit(runner, job, result)
    assert runner.finalize(job, result) == ()
    assert judge_calls["count"] == 1, "the judge is called exactly once"

    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None and terminal["status"] == "failed"
    assert storage.read_clip("target-b").pairing == primary_before, "primary survives"
    # target-b is terminal and is never seeded again; target-c is unaffected.
    remaining = [
        item.clip_uid for item in runner.freeze_cross_pair_after_primary_quiescence()
    ]
    assert "target-b" not in remaining
    assert "target-c" in remaining

    records: list[dict[str, Any]] = []
    for path in sorted(storage.root.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    failures = [item for item in records if item.get("stage") == "pair"]
    assert failures, "a Pair failure diagnostic exists"
    details = failures[-1]["details"]
    assert details["attempt_count"] == 1
    assert details["raw_responses"] == ["bad"]
    assert details["issues"][0]["code"] == "schema_invalid"


def _png_sha(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _chain(runner: Any, job: Any, judge: Any) -> Any:
    """run -> commit the same result -> finalize, like the scheduler does."""
    result = runner.run(job, judge)
    if result.committed:
        _commit(runner, job, result)
    return result, runner.finalize(job, result)


def test_cross_donor1_accept_publishes_frozen_donor_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    donor_uid = dict(job.target)["donor_clip_uid"]
    donor_png = storage.selected_entity_path(donor_uid, "e1").read_bytes()

    judge = _CrossJudge([True])
    result, unlocked = _chain(runner, job, judge)

    assert result.committed
    assert unlocked == (), "donor2 is never planned after an accept"
    assert len(judge.calls) == 1
    assert judge.calls[0]["target_clip_uid"] == "target-b"
    assert judge.calls[0]["donor_clip_uid"] == donor_uid

    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None and terminal["status"] == "completed"

    state = storage.read_clip("target-b").references.entities[0]
    assert state.status == "ready"
    assert state.source_clip_uid == donor_uid
    assert state.source_entity_id == "e1"
    target_png = storage.selected_entity_path("target-b", "e1").read_bytes()
    assert _png_sha(target_png) == _png_sha(donor_png)


def test_cross_donor1_reject_then_donor2_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (first,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    judges = [dict(item.target)["donor_clip_uid"] for item in [first]]
    judge = _CrossJudge([False, True])

    result, unlocked = _chain(runner, first, judge)
    assert result.committed and len(unlocked) == 1
    second = unlocked[0]
    assert dict(second.target)["donor_ordinal"] == "1"
    assert dict(second.target)["donor_clip_uid"] != judges[0]

    result2, unlocked2 = _chain(runner, second, judge)
    assert result2.committed and unlocked2 == ()
    assert len(judge.calls) == 2
    assert [call["donor_clip_uid"] for call in judge.calls] == [
        judges[0],
        dict(second.target)["donor_clip_uid"],
    ]


def test_cross_all_donors_reject_is_terminal_and_survives_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    before = storage.read_clip("target-b")

    pending = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    baseline = runner.cross_baseline(SHARD, "target-b")
    judge = _CrossJudge([False, False, False, False])
    while pending:
        job = pending.pop(0)
        _result, unlocked = _chain(runner, job, judge)
        pending.extend(unlocked)

    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None
    assert terminal["status"] == "completed"
    assert terminal["reason_kind"] == "no_accepted_donor"
    assert storage.read_clip("target-b").pairing == before.pairing
    assert storage.read_clip("target-b").references == before.references

    calls_before_restart = len(judge.calls)
    assert calls_before_restart > 0, "the reject chain really ran"
    restarted = _runner(
        tmp_path,
        _config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    assert restarted.cross_baseline(SHARD, "target-b") == baseline
    assert [
        item.clip_uid
        for item in restarted.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ] == []
    assert len(judge.calls) == calls_before_restart
    calls_before_restart = len(judge.calls)
    assert calls_before_restart > 0, "the reject chain really ran"


def test_frozen_donor_tamper_is_retryable_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen-state corruption is drift, never a legacy target failure."""
    from r2v_data_v2.v3.post_mask_epoch_jobs import OUTCOME_RETRYABLE_FAILED

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    donor_uid = dict(job.target)["donor_clip_uid"]
    storage.selected_entity_path(donor_uid, "e1").write_bytes(b"tampered")

    judge = _CrossJudge([True])
    result = runner.run(job, judge)

    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert not result.committed, "drift must not commit under the old job identity"
    assert judge.calls == [], "Qwen is never called on a frozen-state mismatch"
    assert runner._cross_terminal(SHARD, "target-b") is None


def test_no_frozen_donor_never_materializes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build candidates -> donor lookup -> empty donors -> no evidence."""
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    calls: list[int] = []

    def explode(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        raise AssertionError("evidence must not be materialized")

    real_donors = pm._donors_for_target
    pm.prepare_cross_pair_target_evidence = explode
    pm._donors_for_target = lambda *args, **kwargs: ()
    try:
        _config, _storage, runner = _cross_fixture(tmp_path, monkeypatch)
        jobs = runner.freeze_cross_pair_after_primary_quiescence()
    finally:
        pm.prepare_cross_pair_target_evidence = (
            __import__(
                "r2v_data_v2.v3.pair", fromlist=["prepare_cross_pair_target_evidence"]
            ).prepare_cross_pair_target_evidence
        )
        pm._donors_for_target = real_donors

    assert calls == [], "no evidence materialization when donors is empty"
    assert [job.clip_uid for job in jobs if job.clip_uid == "target-b"] == []
    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None and terminal["status"] == "completed"


def test_multi_entity_fallback_is_strictly_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e1's donor chain must finish before e2's donor1 exists."""
    config = _pair_config(
        tmp_path, monkeypatch, same_parent_fallback_enabled=True
    )
    storage = _storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="donor", clip_suffix="2", entity_types=("subject",)
    )
    _add_ready_clip(
        config,
        storage,
        clip_uid="target-m",
        clip_suffix="20",
        entity_types=("subject", "subject"),
    )
    runner = _runner(
        tmp_path, config, storage, clip_uids=("clip-1", "donor", "target-m")
    )
    _drain_primary(
        runner,
        _ScopedJudge(
            {("target-m", "e1"): "reject", ("target-m", "e2"): "reject"}
        ),
    )
    judge = _CrossJudge([True, True])

    (first,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-m"
    ]
    assert dict(first.target)["target_entity_index"] == "0"

    _result, unlocked = _chain(runner, first, judge)
    assert len(unlocked) == 1
    second = unlocked[0]
    assert dict(second.target)["target_entity_index"] == "1"
    assert dict(second.target)["donor_ordinal"] == "0"

    _result2, unlocked2 = _chain(runner, second, judge)
    assert unlocked2 == ()
    assert [call["target_entity_id"] for call in judge.calls] == ["e1", "e2"]


def test_cross_context_only_evidence_uses_sampled_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No target candidates -> sampled_frames evidence, no entity crop.

    The donor snapshot must be frozen with the REAL candidate builder first:
    patching it earlier would also break validate_entity_reference_artifact
    inside the donor builder, leaving no frozen donors at all.
    """
    import r2v_data_v2.v3.pair as pair_module
    from r2v_data_v2.v3.frames import validate_sampled_frames

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    snapshot = runner.freeze_donor_snapshot(SHARD)
    donor_count = sum(len(g["donors"]) for g in snapshot["groups"])
    assert donor_count > 0, "donors are frozen with the real candidate builder"

    monkeypatch.setattr(
        pair_module,
        "build_entity_reference_candidates",
        lambda *args, **kwargs: [],
    )
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    judge = _CrossJudge([True])
    result, _unlocked = _chain(runner, job, judge)

    assert result.committed
    frames = validate_sampled_frames(storage, "target-b")
    expected_slots = [frame.slot for frame in frames.frames]
    assert judge.calls[0]["target_evidence_mode"] == "sampled_frames"
    assert judge.calls[0]["crop_mode"] is None

    # The rebuilt evidence is what the job identity and the judge really see.
    _prepared, evidence, _donor, _rank, _clip = runner._cross_position(
        SHARD, "target-b", dict(job.target)
    )
    assert list(evidence.frame_slots) == expected_slots
    assert evidence.evidence_mode == "sampled_frames"
    assert evidence.entity_crop is None


def test_cross_publication_is_idempotent_after_marker_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication success then marker loss: restart verifies, never republishes."""
    import r2v_data_v2.v3.pair as pair_module

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    judge = _CrossJudge([True])
    _chain(runner, job, judge)

    published_pairing = storage.read_clip("target-b").pairing
    published_references = storage.read_clip("target-b").references
    target_png = storage.selected_entity_path("target-b", "e1").read_bytes()
    assert _png_sha(target_png)

    # Publication succeeded but the terminal marker never became durable.
    runner._cross_terminal_path(SHARD, "target-b").unlink()

    calls: list[int] = []

    def explode(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        raise AssertionError("must not republish an already-published cross result")

    monkeypatch.setattr(pair_module, "_publish_cross_pair_result", explode)
    restarted = _runner(
        tmp_path,
        _config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    assert [
        item.clip_uid
        for item in restarted.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ] == []

    assert calls == [], "no republish"
    assert len(judge.calls) == 1, "no extra Qwen"
    assert storage.read_clip("target-b").pairing == published_pairing
    assert storage.read_clip("target-b").references == published_references
    assert _png_sha(storage.selected_entity_path("target-b", "e1").read_bytes()) == _png_sha(
        target_png
    )
    terminal = restarted._cross_terminal(SHARD, "target-b")
    assert terminal is not None and terminal["status"] == "completed"


def test_tampered_cross_published_png_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    _chain(runner, job, _CrossJudge([True]))
    runner._cross_terminal_path(SHARD, "target-b").unlink()
    storage.selected_entity_path("target-b", "e1").write_bytes(b"tampered")

    restarted = _runner(
        tmp_path,
        _config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    with pytest.raises(PairEpochError, match="cross-published PNG differs"):
        restarted.freeze_cross_pair_after_primary_quiescence()


def test_cross_cpu_failure_is_durable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary fallback CPU failure terminates the target like legacy."""
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    runner.freeze_donor_snapshot(SHARD)
    before = storage.read_clip("target-b").pairing

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("evidence construction failed")

    monkeypatch.setattr(pm, "prepare_cross_pair_target_evidence", explode)
    judge = _CrossJudge([True])
    jobs = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    assert jobs == [], "no cross job after a CPU terminal failure"
    assert judge.calls == [], "Qwen is never called"

    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None
    assert terminal["status"] == "failed"
    assert terminal["reason_kind"] == "cross_pair_cpu_failure"
    assert storage.read_clip("target-b").pairing == before, "primary survives"

    restarted = _runner(
        tmp_path,
        _config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    assert [
        item.clip_uid
        for item in restarted.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ] == []


def _committed_cross_accept(runner: Any, clip_uid: str) -> Any:
    """Run the donor1 cross job and commit its accept receipt.

    Returns (job, result) with the receipt durable but not yet finalized, so a
    test can patch the next CPU step before the finalizer replays.
    """
    (job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == clip_uid
    ]
    result = runner.run(job, _CrossJudge([True]))
    assert result.committed
    _commit(runner, job, result)
    return job, result


def test_cross_background_cpu_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary cross background CPU failure terminates the target."""
    from r2v_data_v2.v3.post_mask_epoch_pair import CALL_SITE_CROSS_PAIR

    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    job, result = _committed_cross_accept(runner, "target-b")
    before = storage.read_clip("target-b").pairing

    real_background_token = runner._background_token

    def explode(shard: Any, clip: str, frames: Any, call_site: str) -> Any:
        if call_site == CALL_SITE_CROSS_PAIR:
            raise RuntimeError("cross background preparation failed")
        return real_background_token(shard, clip, frames, call_site)

    runner._background_token = explode
    try:
        assert runner.finalize(job, result) == ()
    finally:
        runner._background_token = real_background_token

    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None
    assert terminal["status"] == "failed"
    assert terminal["reason_kind"] == "cross_pair_cpu_failure"
    assert storage.read_clip("target-b").pairing == before, "primary survives"

    restarted = _runner(
        tmp_path,
        _config,
        storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    assert [
        item.clip_uid
        for item in restarted.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ] == []


def test_cross_background_drift_is_not_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen-state drift from _background_token must never become terminal."""
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError

    _config, _storage, runner = _cross_fixture(tmp_path, monkeypatch)
    job, result = _committed_cross_accept(runner, "target-b")

    def drift(*args: Any, **kwargs: Any) -> Any:
        raise PairEpochError("background durable drift")

    runner._background_token = drift
    with pytest.raises(PairEpochError, match="background durable drift"):
        runner.finalize(job, result)
    assert runner._cross_terminal(SHARD, "target-b") is None


def _run_until_quiescent(
    runner: Any, jobs: Sequence[Any], entity_judge: Any, guard_judge: Any, cross_judge: Any
) -> list[Any]:
    """Drive jobs with the right judge per job type, committing each result.

    Returns the background guard jobs it drove, in call order.
    """
    from r2v_data_v2.v3.post_mask_epoch_pair import (
        PAIR_BACKGROUND_GUARD_JOB,
        PAIR_CROSS_JUDGE_JOB,
        PAIR_ENTITY_JUDGE_JOB,
    )

    guard_jobs: list[Any] = []
    pending = list(jobs)
    while pending:
        job = pending.pop(0)
        if job.job_type == PAIR_ENTITY_JUDGE_JOB:
            judge = entity_judge
        elif job.job_type == PAIR_CROSS_JUDGE_JOB:
            judge = cross_judge
        else:
            judge = guard_judge
        if job.job_type == PAIR_BACKGROUND_GUARD_JOB:
            guard_jobs.append(job)
        result = runner.run(job, judge)
        if not result.committed:
            continue
        _commit(runner, job, result)
        pending.extend(runner.finalize(job, result))
    return guard_jobs


def test_primary_and_cross_background_guard_are_two_distinct_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same clip, same background: primary then cross, two separate Qwen calls."""
    from r2v_data_v2.v3.post_mask_epoch_pair import (
        CALL_SITE_CROSS_PAIR,
        CALL_SITE_PRIMARY,
    )
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        tmp_path,
        monkeypatch,
        same_parent_fallback_enabled=True,
        background_final_guard_mode="qwen_v1",
    )
    storage = _storage(config, entity_types=("subject", "subject"))
    _add_ready_clip(
        config, storage, clip_uid="donor", clip_suffix="2", entity_types=("subject",)
    )
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage, clip_uids=("clip-1", "donor"))

    entity_judge = _ScopedJudge({("clip-1", "e2"): "reject"})
    guard_judge = _FinalBackgroundJudge(accepted=True)
    cross_judge = _CrossJudge([True])

    # Primary: e1 full, e2 rejected -> ready pairing -> primary guard.
    primary_guard_jobs = _run_until_quiescent(
        runner, runner.seed_primary_jobs(), entity_judge, guard_judge, cross_judge
    )
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.pairing.background_token == "<ref_bg_1>"
    assert len(guard_judge.calls) == 1, "primary guard ran exactly once"

    # Cross: e1 baseline is already full -> skipped; e2 accepts a frozen donor.
    cross_jobs = runner.freeze_cross_pair_after_primary_quiescence()
    assert [job.clip_uid for job in cross_jobs] == ["clip-1"]
    cross_guard_jobs = _run_until_quiescent(
        runner, cross_jobs, entity_judge, guard_judge, cross_judge
    )

    assert len(cross_judge.calls) == 1
    assert len(guard_judge.calls) == 2, "cross guard is a second paid call"
    counters = runner._guard_counters[SHARD]
    assert counters["background_final_guard_attempted"] == 2
    assert counters["background_final_guard_accepted"] == 2
    assert counters["background_final_guard_failed_closed"] == 0

    final = storage.read_clip("clip-1")
    assert final.pairing.status == "ready"
    assert final.pairing.background_token == "<ref_bg_1>"
    state = next(
        item for item in final.references.entities if item.entity_id == "e2"
    )
    assert state.status == "ready"
    assert state.source_clip_uid == "donor"

    terminal = runner._cross_terminal(SHARD, "clip-1")
    assert terminal is not None and terminal["status"] == "completed"

    # Distinct identities even with identical background image and text.
    (primary_guard_job,) = primary_guard_jobs
    (cross_guard_job,) = cross_guard_jobs
    assert dict(primary_guard_job.target)["call_site"] == CALL_SITE_PRIMARY
    assert dict(cross_guard_job.target)["call_site"] == CALL_SITE_CROSS_PAIR
    assert primary_guard_job.job_id() != cross_guard_job.job_id()


def _guard_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """clip-1 (two subject entities) + a frozen full donor + qwen_v1 guard."""
    from tests.test_v3_pair import _FinalBackgroundJudge

    config = _pair_config(
        root,
        monkeypatch,
        same_parent_fallback_enabled=True,
        background_final_guard_mode="qwen_v1",
    )
    storage = _storage(config, entity_types=("subject", "subject"))
    _add_ready_clip(
        config, storage, clip_uid="donor", clip_suffix="2", entity_types=("subject",)
    )
    _install_clean_background(storage)
    runner = _runner(root / "ledger", config, storage, clip_uids=("clip-1", "donor"))
    judges = {
        "entity": _ScopedJudge({("clip-1", "e2"): "reject"}),
        "guard": _FinalBackgroundJudge(accepted=True),
        "cross": _CrossJudge([True]),
    }
    return config, storage, runner, judges


def test_scheduler_recovers_when_cross_publication_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real scheduler: cross + guard receipts survive a publication crash."""
    # _advance_cross_pair_target imports _publish_cross_pair_result inside the
    # function, so the patch target is the pair module symbol.
    import r2v_data_v2.v3.pair as pm

    root = tmp_path / "run"
    root.mkdir()
    config, storage, runner, judges = _guard_fixture(root, monkeypatch)

    # Primary fully done, then freeze snapshot + cross baselines.
    _run_until_quiescent(
        runner, runner.seed_primary_jobs(),
        judges["entity"], judges["guard"], judges["cross"],
    )
    runner.freeze_donor_snapshot(SHARD)
    baseline = runner.cross_baseline(SHARD, "clip-1")
    donor_png = storage.selected_entity_path("donor", "e1").read_bytes()

    real_publish = pm._publish_cross_pair_result
    publish_calls: list[int] = []

    def explode(*args: Any, **kwargs: Any) -> Any:
        publish_calls.append(1)
        raise RuntimeError("simulated cross publication crash")

    pm._publish_cross_pair_result = explode
    executor = _PairExecutor(
        runner, entity_judge=judges["entity"], guard_judge=judges["guard"],
        cross_judge=judges["cross"],
    )
    scheduler = _scheduler(runner.ledger, runner, executor)
    try:
        outcome = scheduler.run(runner.freeze_cross_pair_after_primary_quiescence())
    finally:
        pm._publish_cross_pair_result = real_publish

    cross_first = len(judges["cross"].calls)
    guard_first = len(judges["guard"].calls)
    assert cross_first == 1, "one paid cross call"
    assert guard_first == 2, "primary guard plus cross guard"
    assert publish_calls == [1], "publication attempted once"
    assert outcome["completed"] is False
    assert runner._cross_terminal(SHARD, "clip-1") is None
    assert runner._live_matches_cross_baseline(storage.read_clip("clip-1"), baseline)

    # Real restart: new ledger, new runner, new executor, new scheduler.
    restarted = _runner(root / "ledger", config, storage, clip_uids=("clip-1", "donor"))
    restarted_executor = _PairExecutor(
        restarted, entity_judge=judges["entity"], guard_judge=judges["guard"],
        cross_judge=judges["cross"],
    )
    restarted_scheduler = _scheduler(restarted.ledger, restarted, restarted_executor)
    restarted_scheduler.run(restarted.freeze_cross_pair_after_primary_quiescence())

    assert len(judges["cross"].calls) == cross_first, "no extra cross Qwen"
    assert len(judges["guard"].calls) == guard_first, "no extra guard Qwen"
    assert restarted_executor.job_types == [], "no restart model job ran"

    final = storage.read_clip("clip-1")
    assert final.pairing.status == "ready"
    assert final.pairing.background_token == "<ref_bg_1>"
    states = {item.entity_id: item for item in final.references.entities}
    assert states["e2"].status == "ready"
    assert states["e2"].source_clip_uid == "donor"
    assert states["e2"].source_entity_id == "e1"
    assert _png_sha(
        storage.selected_entity_path("clip-1", "e2").read_bytes()
    ) == _png_sha(donor_png)
    terminal = restarted._cross_terminal(SHARD, "clip-1")
    assert terminal is not None and terminal["status"] == "completed"


def test_scheduler_recovers_when_terminal_marker_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication succeeds, marker write fails: restart verifies, not republishes."""
    import r2v_data_v2.v3.pair as pm

    root = tmp_path / "run"
    root.mkdir()
    config, storage, runner, judges = _guard_fixture(root, monkeypatch)
    _run_until_quiescent(
        runner, runner.seed_primary_jobs(),
        judges["entity"], judges["guard"], judges["cross"],
    )
    runner.freeze_donor_snapshot(SHARD)
    donor_png = storage.selected_entity_path("donor", "e1").read_bytes()

    real_publish = pm._publish_cross_pair_result
    publish_calls: list[int] = []

    def counting_publish(*args: Any, **kwargs: Any) -> Any:
        publish_calls.append(1)
        return real_publish(*args, **kwargs)

    real_marker = runner._mark_cross_terminal
    marker_calls: list[int] = []

    def explode_marker(
        shard: str, clip_uid: str, *, status: str, reason_kind: str, reason: str
    ) -> None:
        if clip_uid == "clip-1" and status == "completed" and reason_kind == "published":
            marker_calls.append(1)
            if len(marker_calls) == 1:
                raise RuntimeError("simulated terminal marker crash")
        return real_marker(
            shard, clip_uid, status=status, reason_kind=reason_kind, reason=reason
        )

    pm._publish_cross_pair_result = counting_publish
    runner._mark_cross_terminal = explode_marker
    scheduler = _scheduler(
        runner.ledger, runner,
        _PairExecutor(runner, entity_judge=judges["entity"],
                      guard_judge=judges["guard"], cross_judge=judges["cross"]),
    )
    try:
        outcome = scheduler.run(runner.freeze_cross_pair_after_primary_quiescence())
    finally:
        pm._publish_cross_pair_result = real_publish

    assert publish_calls == [1], "publication really succeeded"
    assert marker_calls == [1], "the marker write was the thing that crashed"
    assert outcome["completed"] is False
    assert runner._cross_terminal(SHARD, "clip-1") is None, "marker absent"
    # Live target is already cross-upgraded even though the marker is missing.
    upgraded = storage.read_clip("clip-1")
    states = {item.entity_id: item for item in upgraded.references.entities}
    assert states["e2"].status == "ready"
    assert _png_sha(
        storage.selected_entity_path("clip-1", "e2").read_bytes()
    ) == _png_sha(donor_png)
    published_pairing = upgraded.pairing
    published_references = upgraded.references

    # Restart: Case B must verify the existing publication and rewrite the marker.
    restarted = _runner(root / "ledger", config, storage, clip_uids=("clip-1", "donor"))
    restart_publish_calls: list[int] = []

    def spy_publish(*args: Any, **kwargs: Any) -> Any:
        restart_publish_calls.append(1)
        return real_publish(*args, **kwargs)

    pm._publish_cross_pair_result = spy_publish
    restarted_scheduler = _scheduler(
        restarted.ledger, restarted,
        _PairExecutor(restarted, entity_judge=judges["entity"],
                      guard_judge=judges["guard"], cross_judge=judges["cross"]),
    )
    try:
        restarted_scheduler.run(restarted.freeze_cross_pair_after_primary_quiescence())
    finally:
        pm._publish_cross_pair_result = real_publish

    assert restart_publish_calls == [], "no republish"
    assert len(judges["cross"].calls) == 1, "no extra cross Qwen"
    assert len(judges["guard"].calls) == 2, "no extra guard Qwen"
    final = storage.read_clip("clip-1")
    assert final.pairing == published_pairing
    assert final.references == published_references
    assert _png_sha(
        storage.selected_entity_path("clip-1", "e2").read_bytes()
    ) == _png_sha(donor_png)
    terminal = restarted._cross_terminal(SHARD, "clip-1")
    assert terminal is not None and terminal["status"] == "completed"


def test_scheduler_stops_after_cross_pair_judge_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CrossPairJudgeFailure is terminal for the target, across a restart."""
    from r2v_data_v2.structured_output import ValidationIssue
    from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure

    root = tmp_path / "run"
    root.mkdir()
    config, storage, runner = _cross_fixture(root, monkeypatch)
    primary_before = storage.read_clip("target-b").pairing
    failure = CrossPairJudgeFailure(
        raw_responses=["bad"],
        issues=[
            ValidationIssue(
                code="schema_invalid", field="verdict", message="missing verdict"
            )
        ],
        attempt_count=1,
    )

    class _ExplodingCrossJudge:
        calls: ClassVar[list[int]] = []

        def decide(self, **kwargs: Any) -> Any:
            _ExplodingCrossJudge.calls.append(1)
            raise failure

    scheduler = _scheduler(
        runner.ledger, runner,
        _PairExecutor(runner, cross_judge=_ExplodingCrossJudge()),
    )
    seed = [
        job
        for job in runner.freeze_cross_pair_after_primary_quiescence()
        if job.clip_uid == "target-b"
    ]
    outcome = scheduler.run(seed)

    assert len(_ExplodingCrossJudge.calls) == 1, "one paid cross call"
    # A judge failure is a committed terminal, so the scheduler itself is not
    # left with failed work: the fallback is over for this target.
    assert outcome["completed"] is True
    terminal = runner._cross_terminal(SHARD, "target-b")
    assert terminal is not None and terminal["status"] == "failed"
    assert storage.read_clip("target-b").pairing == primary_before

    restarted = _runner(
        root / "ledger", config, storage,
        clip_uids=("clip-1", "donor", "target-b", "target-c"),
    )
    jobs = [
        item
        for item in restarted.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    assert jobs == [], "later donor and later entity are never planned"
    assert len(_ExplodingCrossJudge.calls) == 1, "no extra cross Qwen"
    assert storage.read_clip("target-b").pairing == primary_before


def test_newly_cross_paired_target_never_becomes_another_donor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross-produced full reference must not enter another target's donors.

    Proven with the actual resource chain call log, not by re-scanning the
    live donor builder: the frozen snapshot is the only donor source.
    """
    _config, storage, runner = _cross_fixture(tmp_path, monkeypatch)
    runner.freeze_donor_snapshot(SHARD)

    frozen_index = runner.frozen_donor_index(SHARD)
    frozen_donor_uids = {
        donor.clip.clip_uid for donors in frozen_index.values() for donor in donors
    }
    assert "target-b" not in frozen_donor_uids, "B starts as a target, not a donor"

    cross_judge = _CrossJudge([True, True])

    # B accepts a frozen donor and becomes ready/full with cross provenance.
    (b_job,) = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-b"
    ]
    _chain(runner, b_job, cross_judge)
    live_b = storage.read_clip("target-b")
    b_state = next(item for item in live_b.references.entities if item.entity_id == "e1")
    assert b_state.status == "ready"
    assert b_state.reference_scope == "full"
    assert b_state.source_clip_uid in frozen_donor_uids

    # Now run C's fallback. B must never appear as a donor for C.
    c_jobs = [
        item
        for item in runner.freeze_cross_pair_after_primary_quiescence()
        if item.clip_uid == "target-c"
    ]
    assert c_jobs, "C still has frozen donors to try"
    pending = c_jobs
    while pending:
        job = pending.pop(0)
        result, unlocked = _chain(runner, job, cross_judge)
        assert result.committed
        pending.extend(unlocked)

    c_calls = [
        call for call in cross_judge.calls if call["target_clip_uid"] == "target-c"
    ]
    assert c_calls, "C's fallback really ran"
    assert all(
        call["donor_clip_uid"] != "target-b" for call in c_calls
    ), "the cross-produced B never became a donor for C"
    assert {call["donor_clip_uid"] for call in c_calls} <= frozen_donor_uids


def test_frozen_donor_index_is_scoped_per_shard_inside_one_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group holds both shards; neither frozen donor index may see the other.

    A resource-epoch group is one execution unit over several canonical shards,
    so a group-level refactor could plausibly build donor snapshots from the
    whole group storage. Legacy donor scope is exactly one canonical shard, and
    this pins both directions on the frozen index the pair runner is fed.
    """
    from dataclasses import replace

    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

    config = _pair_config(tmp_path, monkeypatch, same_parent_fallback_enabled=True)
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

    shard_b_config = replace(config, run_root=config.run_root.parent / "shard-b-run")
    shard_b = RunStorage(shard_b_config)
    shard_b.initialize(git_commit="shard-b")
    _add_ready_clip(
        config,
        shard_b,
        clip_uid="donor-b",
        clip_suffix="3",
        entity_types=("subject",),
    )
    _add_ready_clip(
        config,
        shard_b,
        clip_uid="target-b",
        clip_suffix="30",
        entity_types=("subject",),
    )

    runner = PairEpochRunner(
        config,
        {"shard-a": shard_a, "shard-b": shard_b},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={
            "shard-a": ["clip-1", "donor-a", "target-a"],
            "shard-b": ["donor-b", "target-b"],
        },
    )
    _drain_primary_shard(runner, "shard-a", shard_a)
    _drain_primary_shard(runner, "shard-b", shard_b)
    runner.freeze_donor_snapshot("shard-a")
    runner.freeze_donor_snapshot("shard-b")

    def donor_uids(shard: str) -> set[str]:
        index = runner.frozen_donor_index(shard)
        return {donor.clip.clip_uid for donors in index.values() for donor in donors}

    in_a = donor_uids("shard-a")
    in_b = donor_uids("shard-b")
    assert "donor-a" in in_a
    assert "donor-b" in in_b
    assert "donor-b" not in in_a, "a shard-B donor leaked into shard-A's frozen index"
    assert "donor-a" not in in_b, "a shard-A donor leaked into shard-B's frozen index"
    assert not (in_a & in_b), "the two frozen donor indexes must be disjoint"


# --------------------------------------------------------------------------
# Parallel primary seeding: restart barrier, single publication, legacy parity
# --------------------------------------------------------------------------


def _seeded_by_entity(runner: Any) -> dict[str, Any]:
    return {
        str(dict(job.target)["entity_id"]): job for job in runner.seed_primary_jobs()
    }


def test_restart_seeds_only_the_missing_primary_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e1 and e2 committed, e3 missing: only e3 is seeded and nothing publishes."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)
    seeded = _seeded_by_entity(runner)
    assert sorted(seeded) == ["e1", "e2", "e3"]

    for entity_id in ("e1", "e2"):
        assert _run_one(runner, seeded[entity_id], _Judge()).committed
    # Crash here: no finalize, so nothing is published.

    resumed = _runner(tmp_path, config, storage)
    pending = resumed.seed_primary_jobs()

    assert [str(dict(job.target)["entity_id"]) for job in pending] == ["e3"]
    assert pending[0].job_id() == seeded["e3"].job_id()
    assert storage.read_clip("clip-1").pairing is None, "one missing receipt holds it"


def test_last_primary_receipt_publishes_the_clip_in_annotation_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole clip publishes exactly once, entities in annotation order."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)

    # Seeded and unlocked jobs are deduplicated by job id, as the scheduler does.
    judge = _Judge()
    pending = list(runner.seed_primary_jobs())
    done: dict[str, Any] = {}
    while pending:
        job = pending.pop(0)
        if job.job_id() in done:
            continue
        done[job.job_id()] = job
        result = _run_one(runner, job, judge)
        assert result.committed
        pending.extend(runner.finalize(job, result))

    assert sorted(str(dict(job.target)["entity_id"]) for job in done.values()) == [
        "e1",
        "e2",
        "e3",
    ]

    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.pairing.retained_entity_ids == ["e1", "e2", "e3"]
    # Publication order is annotation order, not completion order.
    assert [item.entity_id for item in clip.references.entities] == ["e1", "e2", "e3"]
    assert runner.primary_unresolved_job_ids() == ()


def test_primary_publication_matches_legacy_pair_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parallel seed produces exactly the legacy Pair state and pixels.

    Each side gets its own independent fixture built from the same inputs. The
    legacy authority runs first and its outcome is captured as plain data,
    because the fixture owns the allowed-root monkeypatches and rebuilding it
    for the epoch would invalidate the legacy config.
    """
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_config = _pair_config(legacy_dir, monkeypatch)
    legacy_storage = _storage(
        legacy_config, entity_types=("subject", "object", "object")
    )
    pair_clips(legacy_config, legacy_storage, judge=_Judge())
    legacy_clip = legacy_storage.read_clip("clip-1")
    legacy_pairing = legacy_clip.pairing
    legacy_references = legacy_clip.references
    legacy_png = {
        entity_id: _png_sha(
            legacy_storage.selected_entity_path("clip-1", entity_id).read_bytes()
        )
        for entity_id in ("e1", "e2", "e3")
    }

    epoch_dir = tmp_path / "epoch"
    epoch_dir.mkdir()
    config = _pair_config(epoch_dir, monkeypatch)
    epoch_storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(epoch_dir / "ledger", config, epoch_storage)
    _drain(runner, _Judge())
    epoch_clip = epoch_storage.read_clip("clip-1")

    assert epoch_clip.pairing == legacy_pairing
    assert epoch_clip.references == legacy_references
    assert {
        entity_id: _png_sha(
            epoch_storage.selected_entity_path("clip-1", entity_id).read_bytes()
        )
        for entity_id in ("e1", "e2", "e3")
    } == legacy_png


def test_restart_after_all_receipts_publishes_once_with_no_extra_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All receipts committed, publication crashed: restart publishes once."""
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))
    runner = _runner(tmp_path, config, storage)
    seeded = _seeded_by_entity(runner)
    for entity_id in ("e1", "e2", "e3"):
        assert _run_one(runner, seeded[entity_id], _Judge()).committed
    assert storage.read_clip("clip-1").pairing is None, "crash before publication"

    real_prepare = pm.prepare_entity_reference
    prepared: list[str] = []

    def counting_prepare(*args: Any, **kwargs: Any) -> Any:
        prepared.append(str(kwargs["entity"].entity_id))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(pm, "prepare_entity_reference", counting_prepare)

    resumed = _runner(tmp_path, config, storage)
    # No entity work is left, so no further Qwen call can be seeded, and the
    # seeding call itself replays the receipts once and completes publication.
    assert resumed.seed_primary_jobs() == []

    # Exactly one reconstruction on the restart, then one publication.
    assert len(prepared) == 3
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.pairing.retained_entity_ids == ["e1", "e2", "e3"]
    assert _temporary_pair_pngs(storage) == []
    assert resumed.primary_unresolved_job_ids() == ()


# --------------------------------------------------------------------------
# Invocation-local caches
# --------------------------------------------------------------------------


def _counting_prepare(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    real = pm.prepare_entity_reference
    calls: list[str] = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(str(kwargs["entity"].entity_id))
        return real(*args, **kwargs)

    monkeypatch.setattr(pm, "prepare_entity_reference", counting)
    return calls


def test_prepared_decision_cache_avoids_a_second_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeding already prepared the entity; the model call reuses it."""
    calls = _counting_prepare(monkeypatch)
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))

    runner = _runner(tmp_path, config, storage)
    (job,) = runner.seed_primary_jobs()
    assert len(calls) == 1

    result = _run_one(runner, job, _Judge())
    assert result.committed
    assert len(calls) == 1, "the prepared decision was reused"


def test_cold_cache_rederives_the_same_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh runner with an empty cache derives the same job and outcome."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))

    warm_runner = _runner(tmp_path / "warm", config, storage)
    (job,) = warm_runner.seed_primary_jobs()
    warm = _run_one(warm_runner, job, _Judge())

    # Nothing is carried over: no cached preparation at all.
    cold_runner = _runner(tmp_path / "cold", config, storage)
    assert cold_runner._prepared == {}
    cold = _run_one(cold_runner, job, _Judge())

    assert cold.committed and warm.committed
    assert dict(cold.payload) == dict(warm.payload), "same decision from a cold cache"


def test_prepared_cache_eviction_does_not_change_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job-cache eviction is covered by the clip row; output is unaffected.

    The job-id cache holds only two decisions here, but the clip's prepared row
    owns all three for the lifetime of the clip, so the model runs still prepare
    nothing. Evicting the clip row itself is what falls back to a real
    re-prepare, which the dedicated eviction test covers.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    monkeypatch.setattr(pm, "_PREPARED_CACHE_LIMIT", 2)
    calls = _counting_prepare(monkeypatch)
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object", "object"))

    runner = _runner(tmp_path, config, storage)
    seeded = _seeded_by_entity(runner)
    after_seed = len(calls)
    assert after_seed == 3

    for entity_id in ("e1", "e2", "e3"):
        result = _run_one(runner, seeded[entity_id], _Judge())
        assert result.committed
    assert len(calls) == after_seed, "the clip row covers job-cache eviction"
    assert runner.prepare_counters["primary_model_prepare_clip_fallback_hits"] >= 1
    assert runner.prepare_counters["primary_model_prepare_cache_misses"] == 0

    runner.finalize(seeded["e3"], runner._committed(seeded["e3"]))
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.pairing.retained_entity_ids == ["e1", "e2", "e3"]
    assert [item.entity_id for item in clip.references.entities] == ["e1", "e2", "e3"]


def test_parallel_prepare_initializes_full_pair_stats_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each parallel entity gets a complete PairStats scratch counter.

    Real reference prefiltering increments named counters directly. An empty
    per-entity dict therefore raises KeyError before any Qwen job can be seeded.
    """

    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)
    context = runner._primary_context(storage, "clip-1")
    assert context is not None
    clip, frames, masks = context

    touched: list[str] = []

    def exercising_prepare(
        config: Any,
        storage: Any,
        *,
        clip_uid: str,
        entity: Any,
        frames: Any,
        masks: Any,
        counters: dict[str, int],
    ) -> str:
        del config, storage, clip_uid, frames, masks
        counters["prefilter_candidates_examined"] += 1
        touched.append(str(entity.entity_id))
        return str(entity.entity_id)

    monkeypatch.setattr(pm, "prepare_entity_reference", exercising_prepare)

    counters = runner._scratch(SHARD)
    prepared = runner._prepare_entities(
        SHARD,
        storage,
        "clip-1",
        list(enumerate(clip.annotation.entities)),
        frames,
        masks,
        counters,
    )

    assert prepared == ["e1", "e2"]
    assert sorted(touched) == ["e1", "e2"]
    assert counters["prefilter_candidates_examined"] == 2


def test_primary_preparation_runs_independent_entities_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two entities are prepared at the same time, not one after the other.

    A barrier proves concurrency: if preparation ever falls back to serial, the
    first entity blocks in the barrier until it times out and the run fails.
    """
    import threading

    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    barrier = threading.Barrier(2, timeout=20)
    real = pm.prepare_entity_reference
    seen: list[str] = []

    def preparing(
        config: Any,
        storage: Any,
        *,
        clip_uid: str,
        entity: Any,
        frames: Any,
        masks: Any,
        counters: Any,
    ) -> Any:
        seen.append(str(entity.entity_id))
        barrier.wait()  # both entities must be inside before either returns
        return real(
            config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            frames=frames,
            masks=masks,
            counters=counters,
        )

    monkeypatch.setattr(pm, "prepare_entity_reference", preparing)

    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)

    jobs = runner.seed_primary_jobs()
    assert len(jobs) == 2
    assert sorted(seen) == ["e1", "e2"]

    # Ordering is still deterministic: each job keeps its annotation index, so
    # completion order of the workers never leaks into the job or state order.
    assert sorted(int(dict(job.target)["entity_index"]) for job in jobs) == [0, 1]
    assert {str(dict(job.target)["entity_id"]) for job in jobs} == {"e1", "e2"}


def _primary_clips_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    clip_uids: Sequence[str],
    *,
    entity_types: tuple[str, ...] = ("subject",),
    cpu_workers: int | None = None,
    config: Any = None,
) -> tuple[Any, Any, Any]:
    """One storage with several ready clips, on its own ledger and run root."""
    base = config if config is not None else _pair_config(tmp_path, monkeypatch)
    run_config = replace(base, run_root=base.run_root.parent / run_name)
    storage = _storage(run_config, entity_types=entity_types)
    for clip_uid in clip_uids[1:]:
        _add_ready_clip(
            run_config, storage, clip_uid=clip_uid, entity_types=entity_types
        )
    runner = _runner(
        tmp_path,
        run_config,
        storage,
        clip_uids=clip_uids,
        ledger_dir=f"ledger-{run_name}",
        cpu_workers=cpu_workers,
    )
    return run_config, storage, runner


def _primary_clip_snapshot(storage: Any, clip_uids: Sequence[str]) -> dict[str, Any]:
    """The durable Primary surface of every clip, canonically ordered."""
    return {
        clip_uid: storage.read_clip(clip_uid).model_dump(mode="json")
        for clip_uid in clip_uids
    }


@pytest.mark.parametrize("cpu_workers", (1, 4))
def test_primary_prepare_failure_is_stage_level_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_workers: int
) -> None:
    """An ordinary preparation failure aborts the stage at its canonical place.

    Pair primary has no per-clip isolation: the exception propagates out of
    seeding, the clips before it are computed and their jobs recorded, and the
    clip that failed and everything after it are left untouched. These judged
    entities publish only after their receipts, so no clip is published here.
    This is the reference behaviour the flat preparation must reproduce at
    every worker count, including that a later clip's speculative preparation
    is never applied ahead of the earlier canonical failure.
    """
    clip_uids = ("clip-1", "clip-2", "clip-3")
    _unused_config, storage, runner = _primary_clips_fixture(
        tmp_path,
        monkeypatch,
        f"run-fail-{cpu_workers}",
        clip_uids,
        cpu_workers=cpu_workers,
    )
    real_prepare = runner._prepare_primary_entity

    def failing(storage_: Any, clip_uid: str, *args: Any, **kwargs: Any) -> Any:
        if clip_uid == "clip-2":
            raise RuntimeError("ordinary preparation failure")
        return real_prepare(storage_, clip_uid, *args, **kwargs)

    monkeypatch.setattr(runner, "_prepare_primary_entity", failing)

    with pytest.raises(RuntimeError, match="ordinary preparation failure"):
        runner.seed_primary_jobs()

    # The clip before the failure was computed and its jobs recorded.
    assert (SHARD, "clip-1") in runner._primary_jobs
    # The failing clip and everything after it were never applied: clip-3 was
    # prepared concurrently here, and its result must not be applied ahead of
    # the earlier canonical failure.
    assert (SHARD, "clip-2") not in runner._primary_jobs
    assert (SHARD, "clip-3") not in runner._primary_jobs
    # No judged clip can publish before its receipts exist.
    for clip_uid in clip_uids:
        assert storage.read_clip(clip_uid).pairing is None


def test_pair_primary_preparation_overlaps_across_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entities of DIFFERENT clips are prepared concurrently.

    Every clip here has one entity, which is the common production shape and
    exactly the case a per-clip pool cannot parallelize. A serial preparation
    cannot satisfy the gate and fails on the bounded timeout.
    """
    clip_uids = ("clip-1", "clip-2", "clip-3", "clip-4")
    _unused_config, _unused_storage, runner = _primary_clips_fixture(
        tmp_path,
        monkeypatch,
        "run-cross-clip",
        clip_uids,
        cpu_workers=4,
    )
    barrier = threading.Barrier(2, timeout=30.0)
    entered: list[tuple[str, int]] = []
    lock = threading.Lock()
    real_prepare = runner._prepare_primary_entity

    def gated(storage_: Any, clip_uid: str, *args: Any, **kwargs: Any) -> Any:
        with lock:
            entered.append((clip_uid, threading.get_ident()))
        barrier.wait()
        return real_prepare(storage_, clip_uid, *args, **kwargs)

    monkeypatch.setattr(runner, "_prepare_primary_entity", gated)
    jobs = runner.seed_primary_jobs()

    assert len(entered) == 4, entered
    assert len({clip_uid for clip_uid, _tid in entered}) >= 2, (
        f"preparations from different clips overlapped: {entered}"
    )
    assert len({tid for _clip_uid, tid in entered}) >= 2
    counters = runner.prepare_counters
    assert counters["primary_prepare_tasks"] == 4
    assert counters["primary_prepare_parallel_tasks"] == 4
    assert counters["primary_prepare_peak_inflight"] >= 2
    assert counters["primary_prepare_peak_buffered_results"] >= 2
    assert counters["primary_prepare_commit_batches"] >= 1
    assert len(jobs) == 4, jobs


def _primary_job_snapshot(jobs: Sequence[Any]) -> list[tuple[str, str, str, str]]:
    return sorted(
        (
            job.job_id(),
            job.input_digest,
            job.clip_uid,
            str(dict(job.target)["entity_id"]),
        )
        for job in jobs
    )


def _primary_selection_hashes(
    storage: Any, clip_uids: Sequence[str], entity_ids: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    import hashlib

    hashes: dict[str, Any] = {}
    for clip_uid in clip_uids:
        for entity_id in entity_ids[clip_uid]:
            path = storage.selected_path(clip_uid, f"{entity_id}.png")
            hashes[f"{clip_uid}/{entity_id}"] = (
                hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            )
    return hashes


def test_parallel_primary_seed_matches_serial_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flat pool reproduces the serial Pair primary result exactly."""
    clip_uids = ("clip-1", "clip-2", "clip-3", "clip-4")
    base = _pair_config(tmp_path, monkeypatch)
    snapshots: dict[str, Any] = {}
    for tag, workers in (("serial", 1), ("parallel", 4)):
        _unused_config, storage, runner = _primary_clips_fixture(
            tmp_path,
            monkeypatch,
            f"run-equiv-{tag}",
            clip_uids,
            cpu_workers=workers,
            config=base,
        )
        seeded = runner.seed_primary_jobs()
        assert seeded, "the fixture has real primary work"
        results = {job.job_id(): _run_one(runner, job, _Judge()) for job in seeded}
        assert all(result.committed for result in results.values())
        for job in seeded:
            runner.finalize(job, results[job.job_id()])

        entity_ids = {
            clip_uid: [
                entity.entity_id
                for entity in storage.read_clip(clip_uid).annotation.entities
            ]
            for clip_uid in clip_uids
        }
        snapshots[tag] = {
            "jobs": _primary_job_snapshot(seeded),
            "primary_index": {
                clip_uid: [job.job_id() for job in runner._primary_jobs[(SHARD, clip_uid)]]
                for clip_uid in clip_uids
                if (SHARD, clip_uid) in runner._primary_jobs
            },
            "references": {
                clip_uid: storage.read_clip(clip_uid).references.model_dump(mode="json")
                for clip_uid in clip_uids
            },
            "pairing": {
                clip_uid: storage.read_clip(clip_uid).pairing.model_dump(mode="json")
                for clip_uid in clip_uids
            },
            "selected": _primary_selection_hashes(storage, clip_uids, entity_ids),
            "stats": runner.reconcile_stats(SHARD).to_dict(),
            "unresolved": runner.primary_unresolved_job_ids(),
        }

    assert snapshots["parallel"] == snapshots["serial"]
    assert snapshots["serial"]["unresolved"] == ()
    assert all(
        payload is not None for payload in snapshots["serial"]["pairing"].values()
    ), "every clip reached a Pair terminal state"


def _counting_primary_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every real entity preparation, whichever path calls it."""
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    prepared: list[str] = []
    real_prepare = pm.prepare_entity_reference

    def counting_prepare(*args: Any, **kwargs: Any) -> Any:
        prepared.append(str(kwargs["entity"].entity_id))
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(pm, "prepare_entity_reference", counting_prepare)
    return prepared


def test_hot_finalizer_prepares_nothing_and_a_cold_runner_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One judged clip: seed prepares once per entity, nothing else prepares."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = _seeded_by_entity(runner)
    seed_calls = len(prepared)
    assert seed_calls == 2, f"one prepare per entity: {prepared}"

    results = {
        entity_id: _run_one(runner, job, _Judge()) for entity_id, job in seeded.items()
    }
    assert all(result.committed for result in results.values())
    assert len(prepared) == seed_calls, "the model run must not prepare again"

    runner.finalize(seeded["e2"], results["e2"])
    assert len(prepared) == seed_calls, (
        "the hot finalizer must finalize from the seed's prepared row"
    )
    assert runner.prepare_counters["primary_hot_finalize_cache_hits"] == 1
    assert runner.prepare_counters["primary_hot_finalize_replay_fallbacks"] == 0
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"

    # A second runner on the same durable state has an empty cache: it must
    # take the strict replay rather than trusting anything in memory.
    cold = _runner(tmp_path, config, storage)
    assert cold._prepared_clips == {}
    before = len(prepared)
    cold.finalize(seeded["e2"], results["e2"])

    assert len(prepared) > before, "a cold runner replays strictly"
    assert cold.prepare_counters["primary_hot_finalize_replay_fallbacks"] >= 1
    assert cold.prepare_counters["primary_hot_finalize_cache_hits"] == 0
    assert storage.read_clip("clip-1").pairing == clip.pairing


def test_hot_finalizer_does_not_reprepare_a_deterministic_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip with one CPU-terminal entity and one judged entity, the common case."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(
        config,
        entity_types=("subject", "object"),
        tracking_status={"e1": "failed"},
    )
    runner = _runner(tmp_path, config, storage)
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = _seeded_by_entity(runner)
    assert "e1" not in seeded, "the failed-tracking entity is CPU terminal"
    assert "e2" in seeded
    after_seed = list(prepared)
    assert sorted(after_seed) == ["e1", "e2"], after_seed

    result = _run_one(runner, seeded["e2"], _Judge())
    assert result.committed
    runner.finalize(seeded["e2"], result)

    # Neither the deterministic sibling nor the judged entity is prepared again.
    assert prepared == after_seed, prepared
    assert runner.prepare_counters["primary_hot_finalize_cache_hits"] == 1
    assert runner.prepare_counters["primary_hot_finalize_replay_fallbacks"] == 0
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None, "the clip was published"

    # The hot path must publish exactly what the strict replay publishes: a cold
    # runner on the same durable state has no row and replays.
    cold = _runner(tmp_path, config, storage)
    before = len(prepared)
    cold.finalize(seeded["e2"], result)
    assert len(prepared) > before, "the cold runner replayed strictly"
    assert storage.read_clip("clip-1").pairing == clip.pairing


def test_evicted_prepared_clip_falls_back_to_the_strict_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real eviction only costs CPU: the strict replay still publishes."""
    clip_uids = ("clip-1", "clip-2")
    _unused_config, storage, runner = _primary_clips_fixture(
        tmp_path, monkeypatch, "run-evict", clip_uids, cpu_workers=1
    )
    monkeypatch.setattr(runner, "_prepared_clip_limit", 1)
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = runner.seed_primary_jobs()
    by_clip = {job.clip_uid: job for job in seeded}
    assert sorted(by_clip) == ["clip-1", "clip-2"]
    after_seed = list(prepared)

    # clip-2's row evicted clip-1's: a real eviction, not a manual clear.
    assert sorted(runner._prepared_clips) == [(SHARD, "clip-2")]

    for clip_uid in clip_uids:
        job = by_clip[clip_uid]
        result = _run_one(runner, job, _Judge())
        assert result.committed
        runner.finalize(job, result)

    assert len(prepared) > len(after_seed), "the evicted clip replayed strictly"
    assert runner.prepare_counters["primary_hot_finalize_replay_fallbacks"] >= 1
    for clip_uid in clip_uids:
        clip = storage.read_clip(clip_uid)
        assert clip.pairing is not None and clip.pairing.status == "ready"
        assert clip.pairing.retained_entity_ids


@pytest.mark.parametrize("mismatch", ("job_ids", "entity_ids"))
def test_mismatched_prepared_clip_row_falls_back_to_the_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    """A row that no longer matches the frozen clip is a miss, never authority."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject", "object"))
    runner = _runner(tmp_path, config, storage)
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = _seeded_by_entity(runner)
    results = {
        entity_id: _run_one(runner, job, _Judge()) for entity_id, job in seeded.items()
    }
    after_seed = len(prepared)

    cached = runner._prepared_clips[(SHARD, "clip-1")]
    runner._prepared_clips[(SHARD, "clip-1")] = replace(
        cached,
        job_ids=("stale-job",) if mismatch == "job_ids" else cached.job_ids,
        # Same length, different order: the row must be rejected on identity,
        # not merely on a length mismatch.
        entity_ids=(
            tuple(f"stale-{index}" for index in range(len(cached.entity_ids)))
            if mismatch == "entity_ids"
            else cached.entity_ids
        ),
    )

    runner.finalize(seeded["e2"], results["e2"])

    assert len(prepared) > after_seed, "a mismatched row must replay strictly"
    assert runner.prepare_counters["primary_hot_finalize_replay_fallbacks"] == 1
    assert runner.prepare_counters["primary_hot_finalize_cache_hits"] == 0
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"


def _many_judged_clips_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    *,
    clip_count: int,
    entity_types: tuple[str, ...],
    cpu_workers: int | None = None,
) -> tuple[Any, Any, Any]:
    """Several clips whose judged entities total well over the job cache."""
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=entity_types)
    clip_uids = tuple(f"clip-{index}" for index in range(1, clip_count + 1))
    for clip_uid in clip_uids[1:]:
        _add_ready_clip(
            config, storage, clip_uid=clip_uid, entity_types=entity_types
        )
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=clip_uids,
        ledger_dir=f"ledger-{run_name}",
        cpu_workers=cpu_workers,
    )
    return config, storage, runner


def test_seeded_jobs_beyond_the_job_cache_never_reprepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eighteen judged clips cross the sixteen-entry job cache.

    A seeding invocation with more judge jobs than the job-id cache can hold
    used to evict decisions before Qwen ever ran them, so model execution
    re-prepared and evicted more. The clip rows must cover all of it.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    assert pm._PREPARED_CACHE_LIMIT == 16, "the fixture must really cross it"
    _unused_config, storage, runner = _many_judged_clips_fixture(
        tmp_path,
        monkeypatch,
        "run-over-job-cache",
        clip_count=18,
        entity_types=("subject",),
        cpu_workers=4,
    )
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = runner.seed_primary_jobs()
    assert len(seeded) == 18
    after_seed = len(prepared)
    assert after_seed == 18, f"one prepare per entity: {prepared}"

    for job in seeded:
        assert _run_one(runner, job, _Judge()).committed
    assert len(prepared) == after_seed, "model execution must not prepare again"

    counters = runner.prepare_counters
    assert counters["primary_model_prepare_cache_misses"] == 0
    assert counters["primary_model_prepare_clip_fallback_hits"] >= 1, (
        "the first jobs were evicted from the job cache and came from the clip row"
    )

    before_finalize = len(prepared)
    for job in seeded:
        runner.finalize(job, runner._committed(job))
    assert len(prepared) == before_finalize, "hot finalizers must not prepare again"
    assert all(
        storage.read_clip(clip_uid).pairing is not None
        for clip_uid in ("clip-1", "clip-18")
    )


def test_model_execution_reuses_clip_rows_whatever_the_job_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job cache holds one decision; execution order must not cause churn.

    The real scheduler sorts jobs by id, which is not the seed insertion order,
    so a cache that only grew on misses would evict work a later job still
    needs. Twenty judged entities against a one-entry job cache prove the clip
    rows carry all of it.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    monkeypatch.setattr(pm, "_PREPARED_CACHE_LIMIT", 1)
    _unused_config, _unused_storage, runner = _many_judged_clips_fixture(
        tmp_path,
        monkeypatch,
        "run-churn",
        clip_count=4,
        entity_types=("subject",) * 5,
        cpu_workers=4,
    )
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = runner.seed_primary_jobs()
    assert len(seeded) == 20
    after_seed = len(prepared)
    assert after_seed == 20, f"one prepare per entity: {prepared}"

    # A different execution order from the insertion order.
    for job in sorted(seeded, key=lambda item: item.job_id(), reverse=True):
        assert _run_one(runner, job, _Judge()).committed

    assert len(prepared) == after_seed, "no churn: model execution prepared nothing"
    counters = runner.prepare_counters
    assert counters["primary_model_prepare_cache_misses"] == 0
    assert counters["primary_model_prepare_clip_fallback_hits"] >= 1
    assert runner._prepared_clips, "the clip rows are the bounded owner"


def test_clip_row_eviction_still_falls_back_to_a_real_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evicting the clip row is what costs CPU; correctness is unaffected.

    Both bounded caches are shrunk: the job cache must not quietly cover the
    clip whose row was evicted, otherwise this would not exercise the fallback.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    monkeypatch.setattr(pm, "_PREPARED_CACHE_LIMIT", 1)
    _unused_config, storage, runner = _many_judged_clips_fixture(
        tmp_path,
        monkeypatch,
        "run-clip-evict",
        clip_count=2,
        entity_types=("subject",),
        cpu_workers=1,
    )
    # The clip limit is an instance attribute, so it is set on the runner.
    monkeypatch.setattr(runner, "_prepared_clip_limit", 1)
    prepared = _counting_primary_prepare(monkeypatch)

    seeded = runner.seed_primary_jobs()
    by_clip = {job.clip_uid: job for job in seeded}
    assert sorted(by_clip) == ["clip-1", "clip-2"]
    after_seed = len(prepared)
    assert sorted(runner._prepared_clips) == [(SHARD, "clip-2")]
    assert len(runner._prepared) == 1, "the job cache kept only the last decision"

    # clip-2 still has both caches; clip-1 was evicted from both.
    assert _run_one(runner, by_clip["clip-2"], _Judge()).committed
    assert len(prepared) == after_seed, "the surviving row still covers clip-2"

    prepared_before_clip1 = len(prepared)
    assert _run_one(runner, by_clip["clip-1"], _Judge()).committed
    assert len(prepared) > prepared_before_clip1, (
        "an evicted clip must really re-prepare"
    )
    assert runner.prepare_counters["primary_model_prepare_cache_misses"] >= 1

    # Correctness is unaffected: both clips still publish.
    for clip_uid in ("clip-1", "clip-2"):
        job = by_clip[clip_uid]
        runner.finalize(job, runner._committed(job))
        assert storage.read_clip(clip_uid).pairing is not None


def _drain_primary_terminal(
    runner: Any, storage: Any, clip_uids: Sequence[str]
) -> dict[str, Any]:
    """Seed, run, commit and finalize every clip to a Pair terminal state."""
    seeded = runner.seed_primary_jobs()
    results: dict[str, Any] = {}
    for job in seeded:
        result = _run_one(runner, job, _Judge())
        assert result.committed
        results[job.job_id()] = result
    for job in seeded:
        runner.finalize(job, results[job.job_id()])
    for clip_uid in clip_uids:
        assert storage.read_clip(clip_uid).pairing is not None
    return results


def _counting_reconcile_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the expensive work a reconcile would have to redo."""
    import r2v_data_v2.v3.pair as pair_module
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    counts = {"plan_validation": 0, "candidates": 0, "prefilter": 0}
    real_validate = pm.PairEpochRunner._validate_primary_plan
    real_candidates = pair_module._build_entity_reference_candidates
    real_prefilter = pair_module.prefilter_entity_reference_candidates

    def validate(*args: Any, **kwargs: Any) -> Any:
        counts["plan_validation"] += 1
        return real_validate(*args, **kwargs)

    def candidates(*args: Any, **kwargs: Any) -> Any:
        counts["candidates"] += 1
        return real_candidates(*args, **kwargs)

    def prefilter(*args: Any, **kwargs: Any) -> Any:
        counts["prefilter"] += 1
        return real_prefilter(*args, **kwargs)

    monkeypatch.setattr(pm.PairEpochRunner, "_validate_primary_plan", validate)
    monkeypatch.setattr(pair_module, "_build_entity_reference_candidates", candidates)
    monkeypatch.setattr(
        pair_module, "prefilter_entity_reference_candidates", prefilter
    )
    return counts


def _four_clip_pair_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str, *, cpu_workers: int
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    clip_uids = ("clip-1", "clip-2", "clip-3", "clip-4")
    # The prefilter accounting only exists in the mode production smoke uses.
    config = _pair_config(
        tmp_path, monkeypatch, reference_prefilter_mode="conservative_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    for clip_uid in clip_uids[1:]:
        _add_ready_clip(config, storage, clip_uid=clip_uid, entity_types=("subject",))
    runner = _runner(
        tmp_path,
        config,
        storage,
        clip_uids=clip_uids,
        ledger_dir=f"ledger-{run_name}",
        cpu_workers=cpu_workers,
    )
    return config, storage, runner, clip_uids


def test_hot_reconcile_reuses_plan_and_prefilter_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal reconcile of this invocation redoes none of the heavy work."""
    _unused_config, _storage_, runner, clip_uids = _four_clip_pair_storage(
        tmp_path, monkeypatch, "run-hot-reconcile", cpu_workers=4
    )
    counts = _counting_reconcile_probe(monkeypatch)
    _drain_primary_terminal(runner, _storage_, clip_uids)

    # Reset: only the reconcile phase is under test.
    counts.update(plan_validation=0, candidates=0, prefilter=0)
    stats = runner.reconcile_stats(SHARD)

    assert counts["plan_validation"] == 0, "the validated plan is reused"
    assert counts["candidates"] == 0, "no candidate rebuild during reconcile"
    assert counts["prefilter"] == 0, "no prefilter replay during reconcile"
    assert runner.prepare_counters["primary_reconcile_cache_hits"] >= 1
    assert runner.prepare_counters["hot_prefilter_cache_hits"] == len(clip_uids)
    assert runner.prepare_counters["strict_prefilter_replay_clips"] == 0
    assert stats.ready + stats.rejected == len(clip_uids)


def test_cold_reconcile_replays_strictly_and_matches_hot_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold runner validates and replays, and produces the identical stats."""
    _config_, storage, runner, clip_uids = _four_clip_pair_storage(
        tmp_path, monkeypatch, "run-cold-reconcile", cpu_workers=1
    )
    counts = _counting_reconcile_probe(monkeypatch)
    _drain_primary_terminal(runner, storage, clip_uids)

    counts.update(plan_validation=0, candidates=0, prefilter=0)
    hot = runner.reconcile_stats(SHARD).to_dict()
    assert counts["plan_validation"] == 0
    assert counts["candidates"] == 0

    # A new runner on the same durable state has an empty invocation cache.
    cold = _runner(
        tmp_path,
        _config_,
        storage,
        clip_uids=clip_uids,
        ledger_dir="ledger-run-cold-reconcile",
        cpu_workers=4,
    )
    assert cold._validated_plans == {}, "a cold runner starts empty"
    counts.update(plan_validation=0, candidates=0, prefilter=0)
    cold_stats = cold.reconcile_stats(SHARD).to_dict()

    assert counts["plan_validation"] >= 1, "the cold runner validates strictly"
    assert counts["candidates"] >= len(clip_uids), "the cold runner replays strictly"
    assert counts["prefilter"] >= len(clip_uids)
    assert cold.prepare_counters["primary_reconcile_cache_fallbacks"] >= 1
    assert cold.prepare_counters["strict_prefilter_replay_clips"] == len(clip_uids)
    assert cold.prepare_counters["hot_prefilter_cache_hits"] == 0

    # The cached and the strict path must agree exactly.
    assert cold_stats == hot


def test_replaced_primary_plan_cannot_be_hidden_by_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed plan file must still be validated strictly and fail closed."""
    _unused_config, storage, runner, clip_uids = _four_clip_pair_storage(
        tmp_path, monkeypatch, "run-plan-drift", cpu_workers=1
    )
    _drain_primary_terminal(runner, storage, clip_uids)

    plan_path = runner._plan_path(SHARD)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["clips"][clip_uids[0]]["digest"] = "drifted-on-purpose"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochError
    with pytest.raises(PairEpochError, match="frozen primary input drifted"):
        runner.reconcile_stats(SHARD)
    assert runner.prepare_counters["primary_plan_validation_cache_misses"] >= 1


def test_stale_prepared_row_is_never_used_for_prefilter_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row whose identity no longer matches must replay, not be trusted."""
    _unused_config, storage, runner, clip_uids = _four_clip_pair_storage(
        tmp_path, monkeypatch, "run-stale-prefilter", cpu_workers=1
    )
    _drain_primary_terminal(runner, storage, clip_uids)

    cached = runner._prepared_clips[(SHARD, clip_uids[0])]
    runner._prepared_clips[(SHARD, clip_uids[0])] = replace(
        cached,
        entity_ids=tuple(
            f"stale-{index}" for index in range(len(cached.entity_ids))
        ),
    )
    counts = _counting_reconcile_probe(monkeypatch)

    stats = runner.reconcile_stats(SHARD)

    assert counts["candidates"] >= 1, "the stale clip must be replayed strictly"
    assert runner.prepare_counters["strict_prefilter_replay_clips"] == 1
    assert stats.ready + stats.rejected == len(clip_uids)


def test_cold_prefilter_replay_overlaps_across_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The strict fallback replays clips concurrently, merged in clip order."""
    _config_, storage, runner, clip_uids = _four_clip_pair_storage(
        tmp_path, monkeypatch, "run-parallel-prefilter", cpu_workers=4
    )
    _drain_primary_terminal(runner, storage, clip_uids)

    cold = _runner(
        tmp_path,
        _config_,
        storage,
        clip_uids=clip_uids,
        ledger_dir="ledger-run-parallel-prefilter",
        cpu_workers=4,
    )
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    barrier = threading.Barrier(2, timeout=30.0)
    entered: list[tuple[str, int]] = []
    lock = threading.Lock()
    real_replay = pm.PairEpochRunner._replay_prefilter_clip
    merged: list[str] = []

    def gated(self: Any, storage_: Any, clip: Any, counts_: Any) -> Any:
        with lock:
            entered.append((clip.clip_uid, threading.get_ident()))
        barrier.wait()
        return real_replay(self, storage_, clip, counts_)

    def recording_merge(self: Any, storage_: Any, pending: Any) -> Any:
        # Record the order the caller will actually merge in, which is the
        # order this returns - not the order it was handed.
        result = real_replay_clips(self, storage_, pending)
        merged.extend(clip_uid for clip_uid, _projection in result)
        return result

    real_replay_clips = pm.PairEpochRunner._replay_prefilter_clips
    monkeypatch.setattr(pm.PairEpochRunner, "_replay_prefilter_clip", gated)
    monkeypatch.setattr(pm.PairEpochRunner, "_replay_prefilter_clips", recording_merge)

    cold.reconcile_stats(SHARD)

    assert len(entered) == len(clip_uids), entered
    assert len({tid for _uid, tid in entered}) >= 2, "the fallback really fanned out"
    # The merge is canonical clip order, whatever the completion order was.
    assert merged == sorted(merged), merged
    assert sorted(merged) == sorted(clip_uids)


def _many_clip_pair_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str, *, clip_count: int,
    cpu_workers: int,
) -> tuple[Any, Any, Any, tuple[str, ...]]:
    config = _pair_config(tmp_path, monkeypatch)
    storage = _storage(config, entity_types=("subject",))
    clip_uids = tuple(f"clip-{index}" for index in range(1, clip_count + 1))
    for clip_uid in clip_uids[1:]:
        _add_ready_clip(config, storage, clip_uid=clip_uid, entity_types=("subject",))
    runner = _runner(
        tmp_path, config, storage, clip_uids=clip_uids,
        ledger_dir=f"ledger-{run_name}", cpu_workers=cpu_workers,
    )
    return config, storage, runner, clip_uids


def _drain_streamed_batches(
    runner: Any, *, judge: Any = None, commit: bool = True
) -> dict[str, Any]:
    """Drain the streaming batches, committing and finalizing each one.

    Sampling the prepared-clip cache size after every batch is what proves the
    prepared lifetime is bounded: the generator may only keep a batch alive
    until the consumer asks for the next one.
    """
    judge = judge or _Judge()
    batches = 0
    jobs = 0
    peak_residency = 0
    for batch in runner.iter_primary_seed_batches():
        batches += 1
        jobs += len(batch)
        for job in sorted(batch, key=lambda item: item.job_id()):
            result = runner.run(job, judge)
            if commit and result.committed:
                _commit(runner, job, result)
            runner.finalize(job, result)
        peak_residency = max(peak_residency, len(runner._prepared_clips))
    return {"batches": batches, "jobs": jobs, "peak_residency": peak_residency}


def test_streamed_primary_preparation_is_bounded_and_prepares_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hundred clips prepare once each while residency stays bounded.

    Every hot cache is far smaller than the fixture, so an unbounded
    seed-everything-first design would evict prepared rows before their model
    run and re-prepare them.
    """
    import r2v_data_v2.v3.post_mask_epoch_pair as pm

    _unused_config, storage, runner, clip_uids = _many_clip_pair_storage(
        tmp_path, monkeypatch, "run-scale", clip_count=100, cpu_workers=4
    )
    monkeypatch.setattr(runner, "_prepared_clip_limit", 4)
    monkeypatch.setattr(pm, "_PREPARED_CACHE_LIMIT", 2)
    prepared = _counting_primary_prepare(monkeypatch)

    seen = _drain_streamed_batches(runner)

    assert len(prepared) == len(clip_uids), (
        f"exactly one preparation per entity: {len(prepared)}"
    )
    counters = runner.prepare_counters
    assert counters["primary_model_prepare_cache_misses"] == 0, "no re-preparation"
    assert counters["primary_hot_finalize_replay_fallbacks"] == 0
    assert seen["jobs"] == len(clip_uids)
    assert seen["batches"] > 1, "the fixture really streamed several batches"
    assert seen["peak_residency"] <= 4, (
        f"heavy prepared residency must stay bounded: {seen['peak_residency']}"
    )
    assert counters["primary_prepare_peak_buffered_results"] <= max(
        4, 2 * 4
    )
    for clip_uid in (clip_uids[0], clip_uids[-1]):
        assert storage.read_clip(clip_uid).pairing is not None


def test_batch_drains_enter_the_resource_once_and_take_increasing_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming must not turn N batches into N model-server startups."""
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import (
        JobExecution,
        ResourceEpochScheduler,
    )

    _unused_config, _unused_storage, runner, _unused_clip_uids = _many_clip_pair_storage(
        tmp_path, monkeypatch, "run-batch-session", clip_count=6, cpu_workers=2
    )
    entered: list[str] = []
    phases: list[str] = []
    real_phase = runner.ledger.phase

    def record_phase(phase_id: str) -> Any:
        phases.append(phase_id)
        return real_phase(phase_id)

    monkeypatch.setattr(runner.ledger, "phase", record_phase)

    class _CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def execute_batch(self, jobs: Any) -> Any:
            self.calls += 1
            outcomes: dict[str, Any] = {}
            for job in sorted(jobs, key=lambda item: item.job_id()):
                result = runner.run(job, _Judge())
                if result.committed:
                    _commit(runner, job, result)
                outcomes[job.job_id()] = JobExecution(job, result, None)
            return outcomes

    executor = _CountingExecutor()

    class _FakeManagedResource:
        """Counts resource session boundaries, which is what a start costs."""

        def __init__(self) -> None:
            self.enters: list[str] = []
            self.closes = 0
            self.open = False

        def enter(self, resource: str) -> Any:
            self.enters.append(resource)
            # A resident resource is reused; only the first entry starts it.
            self.open = True
            return executor

        def close(self) -> None:
            if self.open:
                self.closes += 1
                self.open = False

        def counters(self) -> dict[str, Any]:
            return {"enter_count": len(self.enters), "close_count": self.closes}

    manager = _FakeManagedResource()
    scheduler = ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={RESOURCE_QWEN: executor},
        resource_manager=manager,
    )
    real_executor_for = scheduler._executor_for

    def counting_executor_for(resource: str) -> Any:
        entered.append(resource)
        return real_executor_for(resource)

    monkeypatch.setattr(scheduler, "_executor_for", counting_executor_for)

    outcome = scheduler.run_batches(runner.iter_primary_seed_batches())

    # Only the scheduler's own drains matter here; the runner also asks the
    # ledger for its publication phase.
    scheduler_phases = [phase for phase in phases if phase.endswith("-qwen")]
    assert outcome["completed"] is True
    assert len(scheduler_phases) > 1, (
        f"the fixture really streamed several drains: {scheduler_phases}"
    )
    distinct = sorted(set(scheduler_phases))
    assert distinct == [f"r{index:03d}-qwen" for index in range(len(distinct))], (
        f"each drain takes the next phase id, so they cannot collide: {distinct}"
    )
    assert len(distinct) > 1, distinct
    # Every drain went through the managed resource, and the session was closed
    # exactly once for the whole multi-batch run - not once per batch, which is
    # what would turn N batches into N model-server startups.
    assert entered.count(RESOURCE_QWEN) == len(distinct), entered
    assert manager.closes == 1, manager.counters()


def _pair_judges() -> dict[str, Any]:
    """One judge per Pair job type, so a batch can be driven end to end."""
    from tests.test_v3_pair import _FinalBackgroundJudge

    return {
        PAIR_ENTITY_JUDGE_JOB: _Judge(),
        PAIR_BACKGROUND_GUARD_JOB: _FinalBackgroundJudge(accepted=True),
    }


def _drive_batch(runner: Any, batch: Sequence[Any], judges: Mapping[str, Any]) -> None:
    """Run, commit and finalize one batch, exactly as the scheduler would."""
    for job in sorted(batch, key=lambda item: item.job_id()):
        result = runner.run(job, judges[job.job_type])
        if result.committed:
            _commit(runner, job, result)
        runner.finalize(job, result)


def _pair_publication(storage: Any, runner: Any, clip_uids: Sequence[str]) -> Any:
    import hashlib

    return {
        "references": {
            uid: storage.read_clip(uid).references.model_dump(mode="json")
            for uid in clip_uids
        },
        "pairing": {
            uid: storage.read_clip(uid).pairing.model_dump(mode="json")
            for uid in clip_uids
        },
        "selected": {
            f"{uid}/{entity_id}": hashlib.sha256(
                storage.selected_entity_path(uid, entity_id).read_bytes()
            ).hexdigest()
            for uid in clip_uids
            for entity_id in ("e1",)
        },
        "stats": runner.reconcile_stats(SHARD).to_dict(),
        "unresolved": runner.primary_unresolved_job_ids(),
    }


def _restart_pair_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str, clip_uids: Sequence[str]
) -> tuple[Any, Any]:
    config = _pair_config(tmp_path, monkeypatch)
    run_config = replace(config, run_root=config.run_root.parent / run_name)
    storage = _storage(run_config, entity_types=("subject",))
    for clip_uid in clip_uids[1:]:
        _add_ready_clip(run_config, storage, clip_uid=clip_uid, entity_types=("subject",))
    return run_config, storage


def test_streamed_restart_does_not_repeat_committed_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crashing between batches must not re-run a committed model call."""
    clip_uids = ("clip-1", "clip-2", "clip-3", "clip-4")

    # Control: one uninterrupted streaming run.
    ctrl_config, ctrl_storage = _restart_pair_fixture(
        tmp_path, monkeypatch, "run-restart-control", clip_uids
    )
    ctrl_runner = _runner(
        tmp_path, ctrl_config, ctrl_storage, clip_uids=clip_uids,
        ledger_dir="ledger-restart-control", cpu_workers=2,
    )
    ctrl_runner._prepared_clip_limit = 2
    ctrl_judges = _pair_judges()
    for batch in ctrl_runner.iter_primary_seed_batches():
        _drive_batch(ctrl_runner, batch, ctrl_judges)
    control_calls = len(ctrl_judges[PAIR_ENTITY_JUDGE_JOB].calls)
    control_state = _pair_publication(ctrl_storage, ctrl_runner, clip_uids)
    assert control_calls == len(clip_uids)

    # Interrupted: batch 1 only, then a brand new runner resumes.
    config, storage = _restart_pair_fixture(
        tmp_path, monkeypatch, "run-restart-resume", clip_uids
    )
    first_runner = _runner(
        tmp_path, config, storage, clip_uids=clip_uids,
        ledger_dir="ledger-restart-resume", cpu_workers=2,
    )
    first_runner._prepared_clip_limit = 2
    first_judges = _pair_judges()
    batches = first_runner.iter_primary_seed_batches()
    first_batch = next(batches)
    _drive_batch(first_runner, first_batch, first_judges)
    first_calls = len(first_judges[PAIR_ENTITY_JUDGE_JOB].calls)
    assert first_calls == 2, (
        f"the first batch is bounded to two clips, so it committed two calls: "
        f"{first_calls}"
    )
    # Crash: stop consuming the stream.
    batches.close()

    resumed = _runner(
        tmp_path, config, storage, clip_uids=clip_uids,
        ledger_dir="ledger-restart-resume", cpu_workers=2,
    )
    resumed._prepared_clip_limit = 2
    resumed_judges = _pair_judges()
    for batch in resumed.iter_primary_seed_batches():
        _drive_batch(resumed, batch, resumed_judges)
    resumed_calls = len(resumed_judges[PAIR_ENTITY_JUDGE_JOB].calls)

    # No model call is paid twice, and the resumed run reaches the same state.
    assert first_calls + resumed_calls == control_calls, (
        first_calls, resumed_calls, control_calls
    )
    assert _pair_publication(storage, resumed, clip_uids) == control_state


def test_real_manager_starts_qwen_once_for_many_pair_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many scheduler batches must still be exactly one Qwen lifecycle."""
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager
    from r2v_data_v2.v3.post_mask_epoch_scheduler import (
        JobExecution,
        ResourceEpochScheduler,
    )

    _unused_config, _unused_storage, runner, _unused_clip_uids = _many_clip_pair_storage(
        tmp_path, monkeypatch, "run-real-manager", clip_count=6, cpu_workers=2
    )
    judges = _pair_judges()
    phases: list[str] = []
    real_phase = runner.ledger.phase

    def record_phase(phase_id: str) -> Any:
        phases.append(phase_id)
        return real_phase(phase_id)

    class _RealCounterResource:
        def __init__(self) -> None:
            self.started = 0
            self.stopped = 0

        def start(self) -> None:
            self.started += 1

        def stop(self) -> None:
            self.stopped += 1

        def counters(self) -> dict[str, Any]:
            return {
                "start_count": self.started,
                "stop_count": self.stopped,
                "startup_wall_seconds": 0.0,
                "shutdown_wall_seconds": 0.0,
                "service_seconds": 0.0,
                "gpu_slot_job_counts": {},
            }

    class _Executor:
        def execute_batch(self, jobs: Any) -> Any:
            outcomes: dict[str, Any] = {}
            for job in sorted(jobs, key=lambda item: item.job_id()):
                result = runner.run(job, judges[job.job_type])
                if result.committed:
                    _commit(runner, job, result)
                outcomes[job.job_id()] = JobExecution(job, result, None)
            return outcomes

    manager = ResourceEpochManager({RESOURCE_QWEN: lambda: (_RealCounterResource(), _Executor())})
    scheduler = ResourceEpochScheduler(
        ledger=runner.ledger, finalize=runner.finalize, resource_manager=manager
    )
    monkeypatch.setattr(runner.ledger, "phase", record_phase)
    monkeypatch.setattr(runner, "finalize", scheduler_finalize := runner.finalize)

    outcome = scheduler.run_batches(runner.iter_primary_seed_batches())

    qwen_phases = sorted({phase for phase in phases if phase.endswith("-qwen")})
    assert len(qwen_phases) > 1, f"several scheduler batches really ran: {qwen_phases}"
    assert outcome["completed"] is True

    counters = manager.counters()
    assert counters["resources"]["qwen"]["start_count"] == 1, counters
    assert counters["resources"]["qwen"]["stop_count"] == 1, counters
    assert counters["timeline"].count("start:qwen") == 1, counters["timeline"]
    assert counters["timeline"].count("stop:qwen") == 1, counters["timeline"]
    assert scheduler_finalize is not None


def test_streaming_exposes_pair_prepare_wall_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation wall time is reported separately from Qwen execution."""
    _unused_config, _unused_storage, runner, clip_uids = _many_clip_pair_storage(
        tmp_path, monkeypatch, "run-prepare-wall", clip_count=4, cpu_workers=2
    )
    assert runner.prepare_counters["primary_prepare_wall_seconds"] == 0.0

    judges = _pair_judges()
    for batch in runner.iter_primary_seed_batches():
        _drive_batch(runner, batch, judges)

    wall = runner.prepare_counters["primary_prepare_wall_seconds"]
    assert isinstance(wall, float)
    assert wall > 0.0, "preparation really ran and was measured"
    assert runner.prepare_counters["primary_prepare_tasks"] == len(clip_uids)


def test_scheduler_separates_seeded_jobs_from_unlocked_guard_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log's seeded count must exclude jobs a finalizer unlocked.

    ``pair_primary_job_count`` has always meant "jobs Pair Primary seeded", and
    the scheduler's ``job_count`` also contains dynamically unlocked work (the
    background-final guard here). The two must stay separate.
    """
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import (
        JobExecution,
        ResourceEpochScheduler,
    )

    config = _pair_config(
        tmp_path, monkeypatch, background_final_guard_mode="qwen_v1"
    )
    storage = _storage(config, entity_types=("subject",))
    _install_clean_background(storage)
    runner = _runner(tmp_path, config, storage)
    judges = _pair_judges()
    assert PAIR_BACKGROUND_GUARD_JOB in judges

    class _Executor:
        def execute_batch(self, jobs: Any) -> Any:
            outcomes: dict[str, Any] = {}
            for job in sorted(jobs, key=lambda item: item.job_id()):
                result = runner.run(job, judges[job.job_type])
                if result.committed:
                    _commit(runner, job, result)
                outcomes[job.job_id()] = JobExecution(job, result, None)
            return outcomes

    scheduler = ResourceEpochScheduler(
        ledger=runner.ledger, finalize=runner.finalize,
        executors={RESOURCE_QWEN: _Executor()},
    )
    outcome = scheduler.run_batches(runner.iter_primary_seed_batches())

    # One seeded entity job, plus the guard job its finalizer unlocked.
    assert outcome["seed_job_count"] == 1, outcome
    assert outcome["job_count"] == 2, outcome
    assert outcome["seed_job_count"] < outcome["job_count"]
    assert len(judges[PAIR_BACKGROUND_GUARD_JOB].calls) == 1


def _scheduler_for(runner: Any, judges: Mapping[str, Any]) -> Any:
    """A real scheduler whose Qwen executor drives this runner's jobs."""
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import (
        JobExecution,
        ResourceEpochScheduler,
    )

    class _Executor:
        def execute_batch(self, jobs: Any) -> Any:
            outcomes: dict[str, Any] = {}
            for job in sorted(jobs, key=lambda item: item.job_id()):
                result = runner.run(job, judges[job.job_type])
                if result.committed:
                    _commit(runner, job, result)
                outcomes[job.job_id()] = JobExecution(job, result, None)
            return outcomes

    return ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={RESOURCE_QWEN: _Executor()},
    )


def test_streamed_restart_through_the_real_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new scheduler resumes from the real rXXX-qwen receipts.

    The interrupted run commits its first batch under the real phase machinery,
    the stream is dropped mid-flight, and a completely new runner, ledger and
    scheduler pick the same durable state up through ``run_batches`` again. The
    committed model calls must not be repeated and the phase directories must be
    reusable: the plan is monotonic per job, so re-planning an already committed
    job under the same phase id is safe.
    """
    clip_uids = ("clip-1", "clip-2", "clip-3", "clip-4")

    def build(run_name: str) -> tuple[Any, Any, Any]:
        config, storage = _restart_pair_fixture(
            tmp_path, monkeypatch, run_name, clip_uids
        )
        runner = _runner(
            tmp_path, config, storage, clip_uids=clip_uids,
            ledger_dir=f"ledger-{run_name}", cpu_workers=2,
        )
        runner._prepared_clip_limit = 2
        return config, storage, runner

    # Control: one uninterrupted streaming run through the real scheduler.
    _ctrl_config, ctrl_storage, ctrl_runner = build("run-rs-control")
    ctrl_judges = _pair_judges()
    ctrl_outcome = _scheduler_for(ctrl_runner, ctrl_judges).run_batches(
        ctrl_runner.iter_primary_seed_batches()
    )
    ctrl_calls = len(ctrl_judges[PAIR_ENTITY_JUDGE_JOB].calls)
    ctrl_state = _pair_publication(ctrl_storage, ctrl_runner, clip_uids)
    assert ctrl_outcome["completed"] is True
    assert ctrl_calls == len(clip_uids)

    # Interrupted: only the first batch goes through the real scheduler.
    config, storage, runner_a = build("run-rs-resume")
    judges_a = _pair_judges()
    stream = runner_a.iter_primary_seed_batches()
    first_batch = next(stream)
    _scheduler_for(runner_a, judges_a).run_batches([first_batch])
    first_calls = len(judges_a[PAIR_ENTITY_JUDGE_JOB].calls)
    assert first_calls == 2, f"the first batch is bounded to two clips: {first_calls}"
    first_job = first_batch[0]
    stream.close()

    # A brand new runner, ledger and scheduler on the same durable state.
    resumed = _runner(
        tmp_path, config, storage, clip_uids=clip_uids,
        ledger_dir="ledger-run-rs-resume", cpu_workers=2,
    )
    resumed._prepared_clip_limit = 2
    state = resumed.ledger.classify(first_job)
    assert state.state != "pending", f"batch 1 is durably committed: {state}"

    judges_b = _pair_judges()
    resumed_outcome = _scheduler_for(resumed, judges_b).run_batches(
        resumed.iter_primary_seed_batches()
    )
    resumed_calls = len(judges_b[PAIR_ENTITY_JUDGE_JOB].calls)

    assert resumed_outcome["completed"] is True
    assert first_calls + resumed_calls == ctrl_calls, (
        first_calls, resumed_calls, ctrl_calls
    )
    assert _pair_publication(storage, resumed, clip_uids) == ctrl_state
