"""Removal -> Pair composition for one resource-epoch group (4a).

The composition owns no model semantics. It reuses the hydrated run storages,
the launch eligibility view and the group ledger handed in by the caller, and
drains each stage through the frozen stage runners:

    hydrate
      -> Removal
      -> hard stage barrier
      -> Pair primary
      -> primary quiescence barrier
      -> frozen donor snapshot
      -> Pair cross

Removal is a group-level hard stage barrier: nothing in Pair is seeded, frozen
or called while any removal job is unresolved, even when other clips are
already clean. Removal is complete when the scheduler outcome has no
unresolved work -- deterministic terminals (clean_raw, ready_removed,
rejected) count as complete without any model call.

Pair completion is scheduler-level: semantic terminals (a cross judge failure,
all donors rejected) are committed outcomes, not unresolved work.

The composition now runs reference_edit -> reference_integrity -> instruct.
Instruct is the existing deterministic per-clip production policy called with
``client=None``; subject_attributes and export are still unwired, so the group
itself is never reported completed.

Cross-stage restarts are routed by create-once composition handoff markers
rather than by guessing from live clip state: the Reference Integrity handoff is
written only after every shard reconciled Reference Edit and published its stage
counts, and the Instruct handoff only after every shard reconciled Reference
Integrity. That is what lets a restart skip upstream replays -- whose live-state
verification would otherwise mistake a legitimately rewritten downstream
publication for corruption.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
    ReferenceEditEpochError,
)
from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
    ReferenceIntegrityEpochError,
)
from r2v_data_v2.v3.post_mask_epoch_removal import RemovalEpochRunner
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger, atomic_write_json

#: Scheduler factories are injected so 4a can prove composition correctness
#: without owning model lifecycle. 4b wires one ResourceEpochManager session.
RemovalSchedulerFactory = Callable[[RemovalEpochRunner], Any]
PairSchedulerFactory = Callable[[PairEpochRunner], Any]
RemovalRunnerFactory = Callable[..., RemovalEpochRunner]
PairRunnerFactory = Callable[..., PairEpochRunner]
ReferenceIntegritySchedulerFactory = Callable[[Any], Any]
ReferenceIntegrityRunnerFactory = Callable[..., Any]
SubjectAttributesSchedulerFactory = Callable[[Any], Any]
SubjectAttributesRunnerFactory = Callable[..., Any]

DOWNSTREAM_REASON = "downstream resource-epoch phases are not wired"
EXPORT_PENDING_REASON = "post-mask attribute publication is not wired"
REFERENCE_INTEGRITY_INCOMPLETE_REASON = (
    "reference integrity resource epoch incomplete"
)
INSTRUCT_INCOMPLETE_REASON = "instruct resource epoch incomplete"
SUBJECT_ATTRIBUTES_INCOMPLETE_REASON = (
    "subject attributes resource epoch incomplete"
)

#: Create-once composition handoff markers. They are the restart barrier
#: authority only: they never take part in any model job identity.
COMPOSITION_HANDOFF_SCHEMA = "post_mask_resource_epoch_handoff/1"
REFERENCE_INTEGRITY_STARTED = "reference_integrity_started"
INSTRUCT_STARTED = "instruct_started"
SUBJECT_ATTRIBUTES_STARTED = "subject_attributes_started"
SUBJECT_ATTRIBUTES_COMPLETED = "subject_attributes_completed"
COMPOSITION_HANDOFF_STAGES = (
    REFERENCE_INTEGRITY_STARTED,
    INSTRUCT_STARTED,
    SUBJECT_ATTRIBUTES_STARTED,
    SUBJECT_ATTRIBUTES_COMPLETED,
)


def default_removal_runner_factory(
    config: V3Config,
    storages: Mapping[str, Any],
    ledger: GroupLedger,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    emit: Any = None,
    cpu_workers: int | None = None,
) -> RemovalEpochRunner:
    return RemovalEpochRunner(
        config,
        dict(storages),
        ledger,
        emit=emit,
        cpu_workers=cpu_workers,
        eligible_clip_uids_by_shard={
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        },
    )


def default_pair_runner_factory(
    config: V3Config,
    storages: Mapping[str, Any],
    ledger: GroupLedger,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    emit: Any = None,
    cpu_workers: int | None = None,
) -> PairEpochRunner:
    return PairEpochRunner(
        config,
        dict(storages),
        ledger,
        cpu_workers=cpu_workers,
        eligible_clip_uids_by_shard={
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        },
        emit=emit,
    )


def default_subject_attributes_runner_factory(
    config: V3Config,
    storages: Mapping[str, Any],
    ledger: GroupLedger,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    emit: Any = None,
    cpu_workers: int | None = None,
) -> Any:
    """The production Subject Attributes resource-epoch runner."""
    from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
        SubjectAttributeEpochRunner,
    )

    return SubjectAttributeEpochRunner(
        config,
        dict(storages),
        ledger,
        cpu_workers=cpu_workers,
        eligible_clip_uids_by_shard={
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        },
        emit=emit,
    )


def _emit(emit: Any, event: str, **payload: Any) -> None:
    if emit is not None:
        emit(event, **payload)


def _reference_edit_completion_evidence(
    ledger: GroupLedger, storages: Mapping[str, Any]
) -> bool:
    """True when the Reference Edit stage durably completed for these storages.

    This is a RESTART OPTIMIZATION, never a semantic authority: it only decides
    whether the Pair CPU replay may be skipped, and the Reference Edit stage
    itself still runs its full plan/seed_jobs/reconcile verification right
    after. Malformed or wrongly-shaped durable files therefore report False
    instead of steering the session into the fast path.

    Evidence is a well-formed frozen Reference Edit plan plus a well-formed
    terminal clip outcome marker for every fresh clip it planned. The
    composition barrier guarantees Pair completed before any of that could be
    written, so on a full-session restart the Pair replay -- which compares
    live published state against the Pair-only reconstruction and cannot know
    that Reference Edit legitimately rewrote references afterwards -- must not
    run again.
    """
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        CLIP_FRESH_TARGET,
        REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
        REFERENCE_EDIT_PLAN_SCHEMA,
    )

    semantic_root = Path(ledger.root) / "semantic" / "reference_edit"
    plan_root = semantic_root / "primary"
    outcomes_root = semantic_root / "outcomes"
    if not plan_root.is_dir() or not outcomes_root.is_dir():
        return False
    for shard in storages:
        plan_path = plan_root / f"{shard}.json"
        if not plan_path.is_file():
            return False
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        if not isinstance(plan, dict):
            return False
        if plan.get("schema") != REFERENCE_EDIT_PLAN_SCHEMA:
            return False
        if plan.get("canonical_shard") != shard:
            return False
        clips = plan.get("clips")
        if not isinstance(clips, dict):
            return False
        for clip_uid, entry in clips.items():
            if not isinstance(entry, dict):
                return False
            if entry.get("classification") != CLIP_FRESH_TARGET:
                continue
            marker_path = outcomes_root / shard / f"{clip_uid}.json"
            if not marker_path.is_file():
                return False
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return False
            if not isinstance(marker, dict):
                return False
            if marker.get("schema") != REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA:
                return False
            if marker.get("clip_uid") != clip_uid:
                return False
            if marker.get("terminal") not in {"ready", "failed"}:
                return False
    return True


def _pair_stats_from_stage_counts(storages: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the durable Pair stage counts written by a previous session."""
    stats: dict[str, Any] = {}
    for shard, storage in storages.items():
        counts = storage.read_run().counts
        recovered = {
            key[len("pair.") :]: value
            for key, value in counts.items()
            if key.startswith("pair.")
        }
        if recovered:
            stats[shard] = recovered
    return stats


def _reconcile_and_publish_reference_edit_stats(
    reference_edit: Any,
) -> tuple[bool, dict[str, Any], str | None]:
    """Two-phase Reference Edit completion + stage-count publication.

    Phase 1 reconciles EVERY shard with no writes at all. Phase 2 writes the
    stage counts only when every shard reconciled: one failing shard must not
    leave any other shard with partial ``reference_edit.*`` counts.

    Returns ``(completed, stats, error)``.
    """
    reconciled: dict[str, dict[str, Any]] = {}
    try:
        for shard in sorted(reference_edit.storages):
            stats = reference_edit.reconcile_stats(shard)
            reconciled[shard] = stats.to_dict()
    except ReferenceEditEpochError as exc:
        # Durable corruption/incompleteness: write nothing, report incomplete.
        return False, {}, str(exc)
    for shard, payload in reconciled.items():
        reference_edit.storages[shard].update_stage_counts(
            "reference_edit", payload
        )
    return True, reconciled, None


class StageHandoffError(RuntimeError):
    """Raised when a composition handoff marker is malformed or out of scope."""


def default_reference_integrity_runner_factory(
    config: V3Config,
    storages: Mapping[str, Any],
    ledger: GroupLedger,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    emit: Any = None,
    cpu_workers: int | None = None,
) -> Any:
    from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
        ReferenceIntegrityEpochRunner,
    )

    return ReferenceIntegrityEpochRunner(
        config,
        dict(storages),
        ledger,
        eligible_clip_uids_by_shard={
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        },
        emit=emit,
        cpu_workers=cpu_workers,
    )


def _handoff_path(ledger: GroupLedger, stage: str) -> Path:
    return Path(ledger.root) / "composition" / f"{stage}.json"


def _expected_handoff(
    stage: str, eligible_clip_uids_by_shard: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    return {
        "schema": COMPOSITION_HANDOFF_SCHEMA,
        "stage": stage,
        "canonical_shards": sorted(
            str(shard) for shard in eligible_clip_uids_by_shard
        ),
        "eligible_clip_uids_by_shard": {
            str(shard): [str(uid) for uid in eligible_clip_uids_by_shard[shard]]
            for shard in sorted(eligible_clip_uids_by_shard)
        },
    }


def read_composition_handoff(
    ledger: GroupLedger,
    stage: str,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    """The create-once handoff marker of one stage, or ``None`` if unwritten.

    Missing is a normal first run. Anything else must match the exact expected
    payload, so a malformed marker or a drifted eligible scope fails closed
    instead of silently steering the restart into the wrong branch.
    """
    if stage not in COMPOSITION_HANDOFF_STAGES:
        raise StageHandoffError(f"unknown composition handoff stage {stage!r}")
    path = _handoff_path(ledger, stage)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise StageHandoffError(
            f"invalid composition handoff marker: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise StageHandoffError(
            f"composition handoff marker is not an object: {path}"
        )
    expected = _expected_handoff(stage, eligible_clip_uids_by_shard)
    if payload != expected:
        raise StageHandoffError(
            f"composition handoff marker drifted: {path}"
        )
    return payload


def write_composition_handoff(
    ledger: GroupLedger,
    stage: str,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
) -> None:
    """Create-once handoff write; an existing exact marker is a no-op."""
    path = _handoff_path(ledger, stage)
    payload = _expected_handoff(stage, eligible_clip_uids_by_shard)
    if path.is_file():
        read_composition_handoff(
            ledger,
            stage,
            eligible_clip_uids_by_shard=eligible_clip_uids_by_shard,
        )
        return
    atomic_write_json(path, payload)


def _stage_stats_from_stage_counts(
    storages: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    """Recover one stage's durable counts, or fail closed when a shard lacks them.

    A handoff marker claims the stage completed for the whole group, so every
    shard must have published its counts. Returning a partial mapping would
    report a fake completion, which is exactly what the marker exists to avoid.
    """
    recovered: dict[str, Any] = {}
    for shard, storage in storages.items():
        counts = storage.read_run().counts
        payload = {
            key[len(stage) + 1 :]: value
            for key, value in counts.items()
            if key.startswith(f"{stage}.")
        }
        if not payload:
            raise StageHandoffError(
                f"composition handoff claims {stage!r} completed but shard "
                f"{shard!r} has no {stage} stage counts"
            )
        recovered[shard] = payload
    return recovered


def _reconcile_and_publish_reference_integrity_stats(
    reference_integrity: Any,
) -> tuple[bool, dict[str, Any], str | None]:
    """Two-phase Reference Integrity completion + stage-count publication.

    Phase 1 reconciles EVERY shard with no writes at all. Phase 2 writes the
    stage counts only when every shard reconciled: one failing shard must not
    leave any other shard with partial ``reference_integrity.*`` counts.

    Returns ``(completed, stats, error)``.
    """
    reconciled: dict[str, dict[str, Any]] = {}
    try:
        for shard in sorted(reference_integrity.storages):
            stats = reference_integrity.reconcile_stats(shard)
            reconciled[shard] = stats.to_dict()
    except ReferenceIntegrityEpochError as exc:
        # Durable corruption/incompleteness: write nothing, report incomplete.
        return False, {}, str(exc)
    for shard, payload in reconciled.items():
        reference_integrity.storages[shard].update_stage_counts(
            "reference_integrity", payload
        )
    return True, reconciled, None


def run_deterministic_instruct(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
) -> tuple[bool, dict[str, Any], str | None]:
    """Deterministic per-clip Instruct over every eligible clip.

    The production authority is ``instruct_clips(..., client=None)``, which
    builds the instruction from the annotation alone; there is no instruction
    model job. It is called once per clip through a ``ClipScopedStorage`` --
    exactly like the legacy downstream adapter -- because
    ``ClipScopedStorage.update_stage_counts`` only writes to memory. Every shard
    is computed first and the stage counts are published afterwards, so a shard
    that fails cannot leave partial ``instruct.*`` counts on its siblings.

    A per-clip semantic failure inside ``instruct_clips`` stays a legacy
    terminal result (``status=failed`` plus a failure record) and does not make
    the stage incomplete; only a top-level exception does.

    Returns ``(completed, stats, error)``.
    """
    from r2v_data_v2.v3.instruction import instruct_clips
    from r2v_data_v2.v3.runtime import ClipScopedStorage

    fields = ("processed", "skipped_existing", "skipped_not_ready", "failed", "repaired")
    aggregated: dict[str, dict[str, int]] = {}
    try:
        for shard in sorted(storages):
            storage = storages[shard]
            totals = {field: 0 for field in fields}
            for clip_uid in eligible_clip_uids_by_shard.get(shard, ()):
                scoped = ClipScopedStorage(storage, clip_uid)
                stats = instruct_clips(config, scoped, overwrite=False, client=None)
                payload = stats.to_dict()
                for field in fields:
                    totals[field] += int(payload.get(field, 0))
            aggregated[shard] = totals
    except Exception as exc:  # noqa: BLE001 - top-level stage failure
        return False, {}, str(exc)
    for shard, payload in aggregated.items():
        storages[shard].update_stage_counts("instruct", payload)
    return True, aggregated, None


def _expected_subject_attribute_receipts(
    storages: Mapping[str, Any], eligible: Mapping[str, Sequence[str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Recompute every expected final attribute receipt before writing any.

    A shard that fails derivation must not leave another shard with a published
    receipt, so the whole group is computed first and only then published.
    """
    from r2v_data_v2.v3.post_mask_runtime import _expected_attribute_receipt

    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for shard in sorted(storages):
        for uid in eligible.get(shard, ()):
            expected[(str(shard), str(uid))] = _expected_attribute_receipt(
                storages[shard], str(uid)
            )
    return expected


def publish_subject_attribute_receipts(
    storages: Mapping[str, Any], eligible: Mapping[str, Sequence[str]]
) -> None:
    """Publish the final attribute receipts of every eligible clip, or fail.

    Phase A inspects every expected receipt with zero writes: an exact existing
    receipt passes, a missing one is queued and a malformed or drifted one fails
    immediately. Phase B then writes only the queued ones, so a drift on one clip
    can never leave another clip with a published receipt.
    """
    from r2v_data_v2.v3.post_mask_runtime import (
        _attribute_receipt_state,
        _write_attribute_receipt,
    )

    expected = _expected_subject_attribute_receipts(storages, eligible)
    pending: list[tuple[str, str]] = []
    for (shard, uid), payload in expected.items():
        if not _attribute_receipt_state(storages[shard], uid, payload):
            pending.append((shard, uid))
    for shard, uid in pending:
        _write_attribute_receipt(storages[shard], uid, expected[(shard, uid)])


def verify_subject_attribute_receipts(
    storages: Mapping[str, Any], eligible: Mapping[str, Sequence[str]]
) -> None:
    """Verify every final attribute receipt and its pixels, or fail closed.

    This is the only check a completed handoff is allowed to run: it never seeds
    a job, never reconciles a semantic stage and never rewrites an artifact, so a
    tampered terminal PNG stays tampered and is reported instead of repaired.
    """
    from r2v_data_v2.v3.post_mask_runtime import _expected_attribute_receipt

    for shard in sorted(storages):
        storage = storages[shard]
        for uid in eligible.get(shard, ()):
            path = storage.clip_dir(uid) / ".post_mask_attributes.json"
            if not path.is_file():
                raise StageHandoffError(
                    f"final attribute receipt is missing for {uid!r}"
                )
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise StageHandoffError(
                    f"final attribute receipt is unreadable for {uid!r}"
                ) from exc
            try:
                expected = _expected_attribute_receipt(storage, uid)
            except (OSError, ValueError) as exc:
                raise StageHandoffError(
                    f"final attribute artifact is missing or invalid for {uid!r}: {exc}"
                ) from exc
            if existing != expected:
                raise StageHandoffError(
                    f"final attribute receipt drifted for {uid!r}"
                )


def _reconcile_and_publish_subject_attribute_stats(
    *,
    subject_attributes: Any,
    storages: Mapping[str, Any],
    eligible: Mapping[str, Sequence[str]],
) -> tuple[bool, dict[str, Any], str | None]:
    """Reconcile every shard, publish the final receipts, then the counts.

    Phase 1 reconciles every shard with no writes at all. Phase 2 publishes the
    final attribute receipts, which are the pixel authority of this stage, and
    only then writes the ``subject_attributes.*`` stage counts. A failure in any
    phase leaves the group without counts and without the completed handoff.
    """
    from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
        SubjectAttributeEpochError,
    )

    reconciled: dict[str, dict[str, Any]] = {}
    try:
        for shard in sorted(subject_attributes.storages):
            stats = subject_attributes.reconcile_stats(shard)
            # The run record's durable stage counts are integer counters, so the
            # model durations the stats also carry stay in the receipts instead
            # of being truncated into the counts.
            reconciled[shard] = {
                key: value
                for key, value in stats.to_dict().items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
    except SubjectAttributeEpochError as exc:
        return False, {}, str(exc)
    try:
        publish_subject_attribute_receipts(storages, eligible)
    except (OSError, ValueError) as exc:
        return False, {}, str(exc)
    for shard, payload in reconciled.items():
        subject_attributes.storages[shard].update_stage_counts(
            "subject_attributes", payload
        )
    return True, reconciled, None


def run_subject_attributes_stage(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible: Mapping[str, Sequence[str]],
    ledger: GroupLedger,
    result: dict[str, Any],
    subject_attributes_runner_factory: SubjectAttributesRunnerFactory,
    subject_attributes_scheduler_factory: SubjectAttributesSchedulerFactory,
    emit: Any = None,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    """Subject Attributes, then its receipt publication and completed handoff.

    The started marker is written before the runner may seed or pay for anything,
    so a crash can only ever restart inside this stage. The completed marker is
    written last, after every shard reconciled and every final attribute receipt
    was published, so it is the barrier the export is allowed to trust.
    """
    write_composition_handoff(
        ledger,
        SUBJECT_ATTRIBUTES_STARTED,
        eligible_clip_uids_by_shard=eligible,
    )
    subject_attributes = subject_attributes_runner_factory(
        config=config,
        storages=dict(storages),
        ledger=ledger,
        eligible_clip_uids_by_shard=eligible,
        emit=emit,
        cpu_workers=cpu_workers,
    )
    seeded = subject_attributes.seed_jobs()
    _emit(
        emit,
        "post_mask_epoch_subject_attributes_seeded",
        seeded_jobs=len(seeded),
    )
    outcome = subject_attributes_scheduler_factory(subject_attributes).run(seeded)
    unresolved = tuple(outcome.get("unresolved_job_ids", ()))
    completed = False
    stats: dict[str, Any] = {}
    error: str | None = None
    if not unresolved:
        completed, stats, error = _reconcile_and_publish_subject_attribute_stats(
            subject_attributes=subject_attributes,
            storages=storages,
            eligible=eligible,
        )
    result.update(
        {
            "subject_attributes_job_count": len(seeded),
            "subject_attributes_unresolved": unresolved,
            "subject_attributes_completed": completed,
            "subject_attributes_stats": stats,
            "subject_attributes_reconcile_error": error,
            "reason": (
                EXPORT_PENDING_REASON
                if completed
                else SUBJECT_ATTRIBUTES_INCOMPLETE_REASON
            ),
        }
    )
    _emit(
        emit,
        "post_mask_epoch_subject_attributes_finished",
        completed=completed,
        unresolved=len(unresolved),
    )
    if completed:
        write_composition_handoff(
            ledger,
            SUBJECT_ATTRIBUTES_COMPLETED,
            eligible_clip_uids_by_shard=eligible,
        )
    return result


def _after_reference_edit(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible: Mapping[str, Sequence[str]],
    ledger: GroupLedger,
    result: dict[str, Any],
    reference_edit_completed: bool,
    reference_integrity_runner_factory: Any,
    reference_integrity_scheduler_factory: Any,
    reference_integrity_wired: bool,
    subject_attributes_runner_factory: Any = None,
    subject_attributes_scheduler_factory: Any = None,
    emit: Any = None,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    """Continue into Reference Integrity only when Reference Edit completed.

    Reference Edit not completing is a hard barrier: no Reference Integrity
    handoff, no semantic plan, no seed and no Reference Integrity Qwen call.
    """
    if not reference_edit_completed:
        return result
    if not reference_integrity_wired:
        return result
    return _continue_with_reference_integrity(
        config=config,
        storages=storages,
        eligible=eligible,
        ledger=ledger,
        result=result,
        reference_integrity_runner_factory=reference_integrity_runner_factory,
        reference_integrity_scheduler_factory=reference_integrity_scheduler_factory,
        subject_attributes_runner_factory=subject_attributes_runner_factory,
        subject_attributes_scheduler_factory=subject_attributes_scheduler_factory,
        emit=emit,
        cpu_workers=cpu_workers,
    )


def _continue_with_reference_integrity(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible: Mapping[str, Sequence[str]],
    ledger: GroupLedger,
    result: dict[str, Any],
    reference_integrity_runner_factory: Any,
    reference_integrity_scheduler_factory: Any,
    subject_attributes_runner_factory: Any = None,
    subject_attributes_scheduler_factory: Any = None,
    emit: Any = None,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    """Reference Integrity, deterministic Instruct, then Subject Attributes.

    Both handoff markers are written before the work they guard, so a crash can
    only ever restart ``from`` a barrier, never re-verify an upstream
    publication that a downstream stage legitimately rewrote.
    """
    write_composition_handoff(
        ledger,
        REFERENCE_INTEGRITY_STARTED,
        eligible_clip_uids_by_shard=eligible,
    )
    reference_integrity = reference_integrity_runner_factory(
        config=config,
        storages=dict(storages),
        ledger=ledger,
        eligible_clip_uids_by_shard=eligible,
        emit=emit,
        cpu_workers=cpu_workers,
    )
    reference_integrity_seed = reference_integrity.seed_jobs()
    _emit(
        emit,
        "post_mask_epoch_reference_integrity_seeded",
        seeded_jobs=len(reference_integrity_seed),
    )
    reference_integrity_outcome = reference_integrity_scheduler_factory(
        reference_integrity
    ).run(reference_integrity_seed)
    reference_integrity_unresolved = tuple(
        reference_integrity_outcome.get("unresolved_job_ids", ())
    )
    reference_integrity_stats: dict[str, Any] = {}
    reference_integrity_completed = False
    reference_integrity_error: str | None = None
    if not reference_integrity_unresolved:
        (
            reference_integrity_completed,
            reference_integrity_stats,
            reference_integrity_error,
        ) = _reconcile_and_publish_reference_integrity_stats(reference_integrity)
    result.update(
        {
            "reference_integrity_job_count": len(reference_integrity_seed),
            "reference_integrity_unresolved": reference_integrity_unresolved,
            "reference_integrity_completed": reference_integrity_completed,
            "reference_integrity_stats": reference_integrity_stats,
            "reference_integrity_reconcile_error": reference_integrity_error,
        }
    )
    _emit(
        emit,
        "post_mask_epoch_reference_integrity_finished",
        completed=reference_integrity_completed,
        unresolved=len(reference_integrity_unresolved),
    )
    if not reference_integrity_completed:
        result.update(
            {
                "instruct_completed": False,
                "instruct_stats": {},
                "reason": REFERENCE_INTEGRITY_INCOMPLETE_REASON,
            }
        )
        return result

    write_composition_handoff(
        ledger, INSTRUCT_STARTED, eligible_clip_uids_by_shard=eligible
    )
    return run_instruct_stage(
        config=config,
        storages=storages,
        eligible=eligible,
        result=result,
        ledger=ledger,
        subject_attributes_runner_factory=subject_attributes_runner_factory,
        subject_attributes_scheduler_factory=subject_attributes_scheduler_factory,
        emit=emit,
        cpu_workers=cpu_workers,
    )


def run_instruct_stage(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible: Mapping[str, Sequence[str]],
    result: dict[str, Any],
    ledger: GroupLedger,
    subject_attributes_runner_factory: SubjectAttributesRunnerFactory | None = None,
    subject_attributes_scheduler_factory: SubjectAttributesSchedulerFactory
    | None = None,
    emit: Any = None,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    """Run the deterministic Instruct stage and record it in the result.

    Instruct is cheap and deterministic, so a restart that sees the Instruct
    handoff simply reruns it: a ready instruction becomes the legacy
    ``skipped_existing`` and anything failed or missing is reprocessed. No
    per-clip durable receipt is invented for it.
    """
    instruct_completed, instruct_stats, instruct_error = run_deterministic_instruct(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
    )
    result.update(
        {
            "instruct_completed": instruct_completed,
            "instruct_stats": instruct_stats,
            "instruct_error": instruct_error,
            "reason": (
                DOWNSTREAM_REASON if instruct_completed else INSTRUCT_INCOMPLETE_REASON
            ),
        }
    )
    _emit(
        emit,
        "post_mask_epoch_instruct_finished",
        completed=instruct_completed,
    )
    if not instruct_completed:
        return result
    if (
        subject_attributes_runner_factory is None
        or subject_attributes_scheduler_factory is None
    ):
        return result
    # Subject Attributes is the last semantic stage: its started marker is only
    # written once every instruction is published, and its completed marker is
    # the barrier the per-shard export is allowed to trust.
    return run_subject_attributes_stage(
        config=config,
        storages=storages,
        eligible=eligible,
        ledger=ledger,
        result=result,
        subject_attributes_runner_factory=subject_attributes_runner_factory,
        subject_attributes_scheduler_factory=subject_attributes_scheduler_factory,
        emit=emit,
        cpu_workers=cpu_workers,
    )


def run_removal_pair_epochs(
    *,
    config: V3Config,
    storages: Mapping[str, Any],
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    ledger: GroupLedger,
    removal_scheduler_factory: RemovalSchedulerFactory,
    pair_scheduler_factory: PairSchedulerFactory,
    removal_runner_factory: RemovalRunnerFactory = default_removal_runner_factory,
    pair_runner_factory: PairRunnerFactory = default_pair_runner_factory,
    reference_edit_runner_factory: Callable[..., Any] | None = None,
    reference_edit_scheduler_factory: Callable[[Any], Any] | None = None,
    reference_integrity_runner_factory: Callable[..., Any] | None = None,
    reference_integrity_scheduler_factory: Callable[[Any], Any] | None = None,
    subject_attributes_runner_factory: SubjectAttributesRunnerFactory | None = None,
    subject_attributes_scheduler_factory: SubjectAttributesSchedulerFactory
    | None = None,
    emit: Any = None,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    """Drain Removal, then Pair primary, then the frozen Pair cross pass.

    The same RunStorage objects, the same ledger and the same eligibility view
    are shared by both stages: Pair never re-hydrates or re-enumerates clips.
    """
    eligible = {
        shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
    }
    shared = {
        "config":config,
        "storages":dict(storages),
        "ledger":ledger,
        "eligible_clip_uids_by_shard":eligible,
        "emit":emit,
        "cpu_workers":cpu_workers,
    }

    removal = removal_runner_factory(**shared)
    try:
        removal_seed = removal.seed_jobs()
        _emit(emit,"post_mask_epoch_removal_seeded",seeded_jobs=len(removal_seed))
        removal_outcome = removal_scheduler_factory(removal).run(removal_seed)
    finally:
        removal.close()
    remove_completed = bool(removal_outcome.get("completed"))
    _emit(emit,"post_mask_epoch_removal_finished",remove_completed=remove_completed)

    result: dict[str, Any] = {
        "remove_completed":remove_completed,
        "pair_primary_completed":False,
        "pair_cross_completed":False,
        "pair_completed":False,
        "pair_primary_job_count":0,
        "pair_cross_job_count":0,
        "pair_primary_unresolved":(),
        "pair_cross_unresolved":(),
        "reference_edit_completed":False,
        "reference_edit_job_count":0,
        "reference_edit_unresolved":(),
        "reference_edit_stats":{},
        "reference_edit_reconcile_error":None,
        "reference_integrity_completed":False,
        "reference_integrity_job_count":0,
        "reference_integrity_unresolved":(),
        "reference_integrity_stats":{},
        "reference_integrity_reconcile_error":None,
        "instruct_completed":False,
        "instruct_stats":{},
        "subject_attributes_completed":False,
        "subject_attributes_job_count":0,
        "subject_attributes_unresolved":(),
        "subject_attributes_stats":{},
        "subject_attributes_reconcile_error":None,
        "completed":False,
        "reason":"background removal incomplete",
        "removal_outcome":removal_outcome,
    }
    reference_integrity_wired = (
        reference_integrity_runner_factory is not None
        and reference_integrity_scheduler_factory is not None
    )
    if not remove_completed:
        # Hard stage barrier: no Pair seeding, no donor snapshot, no cross
        # baseline, no Pair Qwen call, no Reference Edit work at all.
        return result

    # A handoff marker is the durable restart barrier: it says, for the whole
    # group, which upstream stages already completed and published their stage
    # counts. The most downstream marker wins, and the upstream stages are then
    # never replayed -- their live-state verification would otherwise mistake a
    # publication that a downstream stage legitimately rewrote for corruption.
    subject_attributes_completed = read_composition_handoff(
        ledger, SUBJECT_ATTRIBUTES_COMPLETED, eligible_clip_uids_by_shard=eligible
    )
    subject_attributes_started = read_composition_handoff(
        ledger, SUBJECT_ATTRIBUTES_STARTED, eligible_clip_uids_by_shard=eligible
    )
    instruct_started = read_composition_handoff(
        ledger, INSTRUCT_STARTED, eligible_clip_uids_by_shard=eligible
    )
    reference_integrity_started = read_composition_handoff(
        ledger, REFERENCE_INTEGRITY_STARTED, eligible_clip_uids_by_shard=eligible
    )
    if subject_attributes_completed is not None:
        # The most downstream marker there is. Subject Attributes semantic is
        # terminal, so the runner may not be constructed at all: re-deriving a
        # processed owner would rerun the legacy materialisers and silently
        # repair a tampered final PNG. Only the receipts are verified.
        if (
            subject_attributes_runner_factory is None
            or subject_attributes_scheduler_factory is None
        ):
            raise StageHandoffError(
                "composition handoff claims Subject Attributes completed but no "
                "Subject Attributes runner is wired"
            )
        result.update(
            {
                "pair_primary_completed": True,
                "pair_cross_completed": True,
                "pair_completed": True,
                "pair_primary_unresolved": (),
                "pair_cross_unresolved": (),
                "pair_stats": _stage_stats_from_stage_counts(storages, "pair"),
                "reference_edit_completed": True,
                "reference_edit_job_count": 0,
                "reference_edit_unresolved": (),
                "reference_edit_stats": _stage_stats_from_stage_counts(
                    storages, "reference_edit"
                ),
                "reference_integrity_completed": True,
                "reference_integrity_job_count": 0,
                "reference_integrity_unresolved": (),
                "reference_integrity_stats": _stage_stats_from_stage_counts(
                    storages, "reference_integrity"
                ),
                "instruct_completed": True,
                "instruct_stats": _stage_stats_from_stage_counts(
                    storages, "instruct"
                ),
                "instruct_error": None,
                "subject_attributes_completed": True,
                "subject_attributes_job_count": 0,
                "subject_attributes_unresolved": (),
                "subject_attributes_stats": _stage_stats_from_stage_counts(
                    storages, "subject_attributes"
                ),
                "subject_attributes_reconcile_error": None,
                "reason": EXPORT_PENDING_REASON,
            }
        )
        verify_subject_attribute_receipts(storages, eligible)
        _emit(
            emit,
            "post_mask_epoch_subject_attributes_verified",
            completed=True,
        )
        return result
    if subject_attributes_started is not None:
        if (
            subject_attributes_runner_factory is None
            or subject_attributes_scheduler_factory is None
        ):
            raise StageHandoffError(
                "composition handoff claims Subject Attributes started but no "
                "Subject Attributes runner is wired"
            )
        result.update(
            {
                "pair_primary_completed": True,
                "pair_cross_completed": True,
                "pair_completed": True,
                "pair_primary_unresolved": (),
                "pair_cross_unresolved": (),
                "pair_stats": _stage_stats_from_stage_counts(storages, "pair"),
                "reference_edit_completed": True,
                "reference_edit_job_count": 0,
                "reference_edit_unresolved": (),
                "reference_edit_stats": _stage_stats_from_stage_counts(
                    storages, "reference_edit"
                ),
                "reference_integrity_completed": True,
                "reference_integrity_job_count": 0,
                "reference_integrity_unresolved": (),
                "reference_integrity_stats": _stage_stats_from_stage_counts(
                    storages, "reference_integrity"
                ),
                "instruct_completed": True,
                "instruct_stats": _stage_stats_from_stage_counts(
                    storages, "instruct"
                ),
                "instruct_error": None,
            }
        )
        return run_subject_attributes_stage(
            config=config,
            storages=storages,
            eligible=eligible,
            ledger=ledger,
            result=result,
            subject_attributes_runner_factory=subject_attributes_runner_factory,
            subject_attributes_scheduler_factory=subject_attributes_scheduler_factory,
            emit=emit,
            cpu_workers=cpu_workers,
        )
    if instruct_started is not None:
        result.update(
            {
                "pair_primary_completed": True,
                "pair_cross_completed": True,
                "pair_completed": True,
                "pair_primary_unresolved": (),
                "pair_cross_unresolved": (),
                "pair_stats": _stage_stats_from_stage_counts(storages, "pair"),
                "reference_edit_completed": True,
                "reference_edit_job_count": 0,
                "reference_edit_unresolved": (),
                "reference_edit_stats": _stage_stats_from_stage_counts(
                    storages, "reference_edit"
                ),
                "reference_integrity_completed": True,
                "reference_integrity_job_count": 0,
                "reference_integrity_unresolved": (),
                "reference_integrity_stats": _stage_stats_from_stage_counts(
                    storages, "reference_integrity"
                ),
            }
        )
        return run_instruct_stage(
            config=config,
            storages=storages,
            eligible=eligible,
            result=result,
            ledger=ledger,
            subject_attributes_runner_factory=subject_attributes_runner_factory,
            subject_attributes_scheduler_factory=(
                subject_attributes_scheduler_factory
            ),
            emit=emit,
            cpu_workers=cpu_workers,
        )
    if reference_integrity_started is not None:
        if not reference_integrity_wired:
            raise StageHandoffError(
                "composition handoff claims Reference Integrity completed but "
                "no Reference Integrity runner is wired"
            )
        result.update(
            {
                "pair_primary_completed": True,
                "pair_cross_completed": True,
                "pair_completed": True,
                "pair_primary_unresolved": (),
                "pair_cross_unresolved": (),
                "pair_stats": _stage_stats_from_stage_counts(storages, "pair"),
                "reference_edit_completed": True,
                "reference_edit_job_count": 0,
                "reference_edit_unresolved": (),
                "reference_edit_stats": _stage_stats_from_stage_counts(
                    storages, "reference_edit"
                ),
            }
        )
        return _continue_with_reference_integrity(
            config=config,
            storages=storages,
            eligible=eligible,
            ledger=ledger,
            result=result,
            reference_integrity_runner_factory=reference_integrity_runner_factory,
            reference_integrity_scheduler_factory=(
                reference_integrity_scheduler_factory
            ),
            emit=emit,
            cpu_workers=cpu_workers,
        )

    if _reference_edit_completion_evidence(ledger, storages):
        # Post-Reference-Edit restart: the barrier already proved Pair
        # completed, and the Pair replay would compare live state that
        # Reference Edit legitimately rewrote. Reuse the durable stage counts.
        pair_stats = _pair_stats_from_stage_counts(storages)
        result.update(
            {
                "pair_primary_completed": True,
                "pair_cross_completed": True,
                "pair_completed": True,
                "pair_primary_unresolved": (),
                "pair_cross_unresolved": (),
                "pair_stats": pair_stats,
                "reason": DOWNSTREAM_REASON,
            }
        )
        if reference_edit_runner_factory is None:
            return result
        reference_edit = reference_edit_runner_factory(**shared)
        reference_edit_seed = reference_edit.seed_jobs()
        result["reference_edit_job_count"] = len(reference_edit_seed)
        reference_edit_outcome = reference_edit_scheduler_factory(
            reference_edit
        ).run(reference_edit_seed)
        reference_edit_unresolved = tuple(
            reference_edit_outcome.get("unresolved_job_ids", ())
        )
        reference_edit_stats: dict[str, Any] = {}
        reference_edit_completed = False
        reconcile_error: str | None = None
        if not reference_edit_unresolved:
            (
                reference_edit_completed,
                reference_edit_stats,
                reconcile_error,
            ) = _reconcile_and_publish_reference_edit_stats(reference_edit)
        result.update(
            {
                "reference_edit_completed": reference_edit_completed,
                "reference_edit_unresolved": reference_edit_unresolved,
                "reference_edit_stats": reference_edit_stats,
                "reference_edit_reconcile_error": reconcile_error,
                "reason": (
                    DOWNSTREAM_REASON
                    if reference_edit_completed
                    else "reference edit resource epoch incomplete"
                ),
            }
        )
        return _after_reference_edit(
            config=config,
            storages=storages,
            eligible=eligible,
            ledger=ledger,
            result=result,
            reference_edit_completed=reference_edit_completed,
            reference_integrity_runner_factory=reference_integrity_runner_factory,
            reference_integrity_scheduler_factory=(
                reference_integrity_scheduler_factory
            ),
            reference_integrity_wired=reference_integrity_wired,
            subject_attributes_runner_factory=subject_attributes_runner_factory,
            subject_attributes_scheduler_factory=(
                subject_attributes_scheduler_factory
            ),
            emit=emit,
            cpu_workers=cpu_workers,
        )

    pair = pair_runner_factory(**shared)
    primary_seed = pair.seed_primary_jobs()
    result["pair_primary_job_count"] = len(primary_seed)
    _emit(emit,"post_mask_epoch_pair_primary_seeded",seeded_jobs=len(primary_seed))
    primary_outcome = pair_scheduler_factory(pair).run(primary_seed)
    primary_unresolved = tuple(primary_outcome.get("unresolved_job_ids", ()))
    _emit(
        emit,
        "post_mask_epoch_pair_primary_finished",
        unresolved=len(primary_unresolved),
    )
    # The formal 3b barrier: blocked clips are excluded, settled fresh targets
    # still enter the frozen cross pass.
    cross_seed = list(
        pair.freeze_cross_pair_after_primary_quiescence(
            unresolved_job_ids=primary_unresolved
        )
    )
    result["pair_cross_job_count"] = len(cross_seed)
    _emit(emit,"post_mask_epoch_pair_cross_seeded",seeded_jobs=len(cross_seed))
    cross_outcome = pair_scheduler_factory(pair).run(cross_seed)
    cross_unresolved = tuple(cross_outcome.get("unresolved_job_ids", ()))

    pair_primary_completed = not primary_unresolved
    pair_cross_completed = not cross_unresolved
    pair_completed = pair_primary_completed and pair_cross_completed
    pair_stats: dict[str, Any] = {}
    if pair_completed:
        # Final durable accounting: only a fully terminal Pair stage may write
        # stage counts, and the write is the same dict on every restart.
        for shard in sorted(pair.storages):
            stats = pair.reconcile_stats(shard)
            payload = stats.to_dict()
            pair.storages[shard].update_stage_counts("pair", payload)
            pair_stats[shard] = payload
    result.update(
        {
            "pair_primary_completed":pair_primary_completed,
            "pair_cross_completed":pair_cross_completed,
            "pair_completed":pair_completed,
            "pair_primary_unresolved":primary_unresolved,
            "pair_cross_unresolved":cross_unresolved,
            "pair_stats":pair_stats,
            "pair_outcome":{
                "primary":primary_outcome,
                "cross":cross_outcome,
            },
            "reason":(
                DOWNSTREAM_REASON
                if pair_completed
                else "pair resource epoch incomplete"
            ),
        }
    )
    if reference_edit_runner_factory is None:
        return result
    if not pair_completed:
        # Hard stage barrier: no Reference Edit plan, seed or model call.
        result["reference_edit_completed"] = False
        result["reference_edit_job_count"] = 0
        result["reference_edit_unresolved"] = ()
        result["reference_edit_stats"] = {}
        return result
    reference_edit = reference_edit_runner_factory(**shared)
    reference_edit_seed = reference_edit.seed_jobs()
    result["reference_edit_job_count"] = len(reference_edit_seed)
    _emit(emit, "post_mask_epoch_reference_edit_seeded",
          seeded_jobs=len(reference_edit_seed))
    reference_edit_outcome = reference_edit_scheduler_factory(
        reference_edit
    ).run(reference_edit_seed)
    reference_edit_unresolved = tuple(
        reference_edit_outcome.get("unresolved_job_ids", ())
    )
    reference_edit_stats: dict[str, Any] = {}
    # Completion is confirmed by the durable reconcile, not by the scheduler
    # alone: every shard must reconcile (which re-verifies each fresh clip's
    # terminal publication) before any stage count is written. The reconcile
    # and the publication are strictly two-phase so a failing shard can never
    # leave partial stage counts behind on its siblings.
    reference_edit_completed = False
    reconcile_error: str | None = None
    if not reference_edit_unresolved:
        (
            reference_edit_completed,
            reference_edit_stats,
            reconcile_error,
        ) = _reconcile_and_publish_reference_edit_stats(reference_edit)
    result.update(
        {
            "reference_edit_completed": reference_edit_completed,
            "reference_edit_unresolved": reference_edit_unresolved,
            "reference_edit_stats": reference_edit_stats,
            "reference_edit_reconcile_error": reconcile_error,
            "reason": (
                DOWNSTREAM_REASON
                if reference_edit_completed
                else "reference edit resource epoch incomplete"
            ),
        }
    )
    return _after_reference_edit(
        config=config,
        storages=storages,
        eligible=eligible,
        ledger=ledger,
        result=result,
        reference_edit_completed=reference_edit_completed,
        reference_integrity_runner_factory=reference_integrity_runner_factory,
        reference_integrity_scheduler_factory=reference_integrity_scheduler_factory,
        reference_integrity_wired=reference_integrity_wired,
        subject_attributes_runner_factory=subject_attributes_runner_factory,
        subject_attributes_scheduler_factory=subject_attributes_scheduler_factory,
        emit=emit,
        cpu_workers=cpu_workers,
    )


# ---------------------------------------------------------------------------
# 4b: one ResourceEpochManager session across the Removal -> Pair barriers
# ---------------------------------------------------------------------------


class StageDispatchError(RuntimeError):
    """Raised when a shared executor is used before a stage is bound."""


class _EpochStageDispatch:
    """Mutable stage binding behind one cached resource executor.

    ``ResourceEpochManager.enter()`` returns the cached executor while its
    resource is open, so the Qwen executor created for Removal would keep
    calling the Removal runner during Pair. This proxy is the only thing that
    changes between stages: it holds no policy of its own.
    """

    def __init__(self) -> None:
        self._runner: Any = None
        self.log: list[str] = []

    def bind(self, runner: Any) -> None:
        self._runner = runner

    @property
    def runner(self) -> Any:
        return self._runner

    def run(self, job: Any, handle: Any) -> Any:
        if self._runner is None:
            raise StageDispatchError("stage dispatch used before bind()")
        self.log.append(f"{type(self._runner).__name__}:{job.job_type}")
        return self._runner.run(job, handle)

    def __call__(self, job: Any, handle: Any) -> Any:
        return self.run(job, handle)


def shared_qwen_model_identities(config: V3Config) -> tuple[str, ...]:
    """Judge services that the shared Qwen session must serve.

    Empty in the config means the service is not used by this launch, so it is
    not part of the compatibility check.
    """
    services = (
        config.qwen.background_remove_judge,
        config.qwen.candidate_judge,
        config.qwen.cross_pair_judge,
        config.qwen.background_final_judge,
        config.qwen.reference_edit_judge
        if getattr(config.reference_edit, "enabled", False)
        else None,
        # The same service also serves the Reference Integrity main and bbox
        # reviews. ``instruction_writer`` is deliberately absent: production
        # Instruct is deterministic with ``client=None``.
        config.qwen.reference_integrity_judge
        if getattr(config.reference_integrity, "enabled", False)
        else None,
    )
    return tuple(
        str(service.model)
        for service in services
        if service is not None and str(service.model)
    )


def validate_shared_qwen_models(
    config: V3Config, *, expected_served_model_id: str
) -> None:
    """Fail before the first model call when a shared server cannot serve all."""
    expected = str(expected_served_model_id)
    incompatible = sorted(
        {model for model in shared_qwen_model_identities(config) if model != expected}
    )
    if incompatible:
        raise StageDispatchError(
            "shared Qwen session cannot serve every stage model: "
            f"expected {expected!r}, configured {incompatible}"
        )


def run_removal_pair_resource_session(
    *,
    manager: Any,
    build_removal_scheduler: Callable[[Any, _EpochStageDispatch], Any],
    build_pair_scheduler: Callable[[Any, _EpochStageDispatch], Any],
    config: V3Config | None = None,
    expected_served_model_id: str | None = None,
    dispatch: _EpochStageDispatch | None = None,
    removal_runner_factory: RemovalRunnerFactory = default_removal_runner_factory,
    pair_runner_factory: PairRunnerFactory = default_pair_runner_factory,
    build_reference_edit_scheduler: Callable[[Any, _EpochStageDispatch], Any]
    | None = None,
    reference_edit_runner_factory: Callable[..., Any] | None = None,
    build_reference_integrity_scheduler: Callable[[Any, _EpochStageDispatch], Any]
    | None = None,
    reference_integrity_runner_factory: ReferenceIntegrityRunnerFactory
    | None = None,
    build_subject_attributes_scheduler: Callable[[Any, _EpochStageDispatch], Any]
    | None = None,
    subject_attributes_runner_factory: SubjectAttributesRunnerFactory | None = None,
    **composition: Any,
) -> dict[str, Any]:
    """Run the 4a composition inside one shared resource session.

    The stage dispatch is rebound before each scheduler so the cached Qwen
    executor always calls the runner that owns the current stage, and the outer
    session -- not the schedulers -- owns the single ``manager.close()``.
    """
    if expected_served_model_id is not None and config is not None:
        validate_shared_qwen_models(config, expected_served_model_id=expected_served_model_id)
    # Production builds its manager factories before this call, so it hands in
    # the dispatch those factories already captured.
    dispatch = dispatch if dispatch is not None else _EpochStageDispatch()

    def removal_scheduler(runner: Any) -> Any:
        dispatch.bind(runner)
        return build_removal_scheduler(runner, dispatch)

    def pair_scheduler(runner: Any) -> Any:
        dispatch.bind(runner)
        return build_pair_scheduler(runner, dispatch)

    def reference_edit_scheduler(runner: Any) -> Any:
        # The cached Qwen executor must reach the Reference Edit runner here,
        # never the Pair or Removal runner.
        dispatch.bind(runner)
        return build_reference_edit_scheduler(runner, dispatch)

    def reference_integrity_scheduler(runner: Any) -> Any:
        # Same one cached Qwen executor, now dispatched to the Reference
        # Integrity runner for both the main and the bbox review jobs.
        dispatch.bind(runner)
        return build_reference_integrity_scheduler(runner, dispatch)

    def subject_attributes_scheduler(runner: Any) -> Any:
        # The same one cached Qwen executor, plus the Boogu and SAM resources
        # Removal already needs, now dispatched to the Subject Attributes runner.
        dispatch.bind(runner)
        return build_subject_attributes_scheduler(runner, dispatch)

    try:
        result = run_removal_pair_epochs(
            config=config,
            removal_scheduler_factory=removal_scheduler,
            pair_scheduler_factory=pair_scheduler,
            removal_runner_factory=removal_runner_factory,
            pair_runner_factory=pair_runner_factory,
            reference_edit_runner_factory=(
                None
                if build_reference_edit_scheduler is None
                else reference_edit_runner_factory
            ),
            reference_edit_scheduler_factory=(
                None
                if build_reference_edit_scheduler is None
                else reference_edit_scheduler
            ),
            reference_integrity_runner_factory=(
                None
                if build_reference_integrity_scheduler is None
                else (
                    reference_integrity_runner_factory
                    or default_reference_integrity_runner_factory
                )
            ),
            reference_integrity_scheduler_factory=(
                None
                if build_reference_integrity_scheduler is None
                else reference_integrity_scheduler
            ),
            subject_attributes_runner_factory=(
                None
                if build_subject_attributes_scheduler is None
                else (
                    subject_attributes_runner_factory
                    or default_subject_attributes_runner_factory
                )
            ),
            subject_attributes_scheduler_factory=(
                None
                if build_subject_attributes_scheduler is None
                else subject_attributes_scheduler
            ),
            **composition,
        )
    finally:
        manager.close()
    result = dict(result)
    # Read the lifecycle only after the outer close, so open_resource is None.
    result["resource_lifecycle"] = manager.counters()
    result["stage_dispatch_log"] = tuple(dispatch.log)
    return result
