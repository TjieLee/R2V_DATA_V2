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
    reference_edit_runner_factory: Callable[..., Any] | None = None,
    reference_edit_scheduler_factory: Callable[[Any], Any] | None = None,
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
    reference_edit_completed = not reference_edit_unresolved
    reference_edit_stats: dict[str, Any] = {}
    if reference_edit_completed:
        for shard in sorted(reference_edit.storages):
            stats = reference_edit.reconcile_stats(shard)
            payload = stats.to_dict()
            reference_edit.storages[shard].update_stage_counts(
                "reference_edit", payload
            )
            reference_edit_stats[shard] = payload
    result.update(
        {
            "reference_edit_completed": reference_edit_completed,
            "reference_edit_unresolved": reference_edit_unresolved,
            "reference_edit_stats": reference_edit_stats,
            "reason": (
                DOWNSTREAM_REASON
                if reference_edit_completed
                else "reference edit resource epoch incomplete"
            ),
        }
    )
    return result


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
            **composition,
        )
    finally:
        manager.close()
    result = dict(result)
    # Read the lifecycle only after the outer close, so open_resource is None.
    result["resource_lifecycle"] = manager.counters()
    result["stage_dispatch_log"] = tuple(dispatch.log)
    return result
