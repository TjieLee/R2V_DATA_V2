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

The group itself is never reported completed: the downstream resource-epoch
phases (reference_edit, reference_integrity, instruct, subject_attributes,
export) are not wired yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner
from r2v_data_v2.v3.post_mask_epoch_removal import RemovalEpochRunner
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

#: Scheduler factories are injected so 4a can prove composition correctness
#: without owning model lifecycle. 4b wires one ResourceEpochManager session.
RemovalSchedulerFactory = Callable[[RemovalEpochRunner], Any]
PairSchedulerFactory = Callable[[PairEpochRunner], Any]
RemovalRunnerFactory = Callable[..., RemovalEpochRunner]
PairRunnerFactory = Callable[..., PairEpochRunner]

DOWNSTREAM_REASON = "downstream resource-epoch phases are not wired"


def default_removal_runner_factory(
    config: V3Config,
    storages: Mapping[str, Any],
    ledger: GroupLedger,
    *,
    eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
    emit: Any = None,
) -> RemovalEpochRunner:
    return RemovalEpochRunner(
        config,
        dict(storages),
        ledger,
        emit=emit,
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
) -> PairEpochRunner:
    return PairEpochRunner(
        config,
        dict(storages),
        ledger,
        eligible_clip_uids_by_shard={
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        },
        emit=emit,
    )


def _emit(emit: Any, event: str, **payload: Any) -> None:
    if emit is not None:
        emit(event, **payload)


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
    emit: Any = None,
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
        "completed":False,
        "reason":"background removal incomplete",
        "removal_outcome":removal_outcome,
    }
    if not remove_completed:
        # Hard stage barrier: no Pair seeding, no donor snapshot, no cross
        # baseline, no Pair Qwen call.
        return result

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
    result.update(
        {
            "pair_primary_completed":pair_primary_completed,
            "pair_cross_completed":pair_cross_completed,
            "pair_completed":pair_completed,
            "pair_primary_unresolved":primary_unresolved,
            "pair_cross_unresolved":cross_unresolved,
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
    return result
