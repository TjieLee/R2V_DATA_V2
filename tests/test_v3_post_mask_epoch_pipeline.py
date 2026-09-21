"""Removal -> Pair composition correctness (4a).

The composition is driven with fake executors and injected scheduler
factories: nothing here decides removal, Pair or donor semantics, and no
model is started. What is proven is the orchestration:

* Removal is a group-level hard stage barrier;
* Pair reuses the same storages, ledger and eligibility view;
* the primary -> cross barrier receives exactly the primary scheduler's
  unresolved job ids;
* the group is never reported completed while the downstream DAG is unwired.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.config import (
    BOOGU_REMOVE_BACKEND,
    PairConfig,
    QwenAnnotationConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    ReferenceScopeConfig,
    V3Config,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
)
from r2v_data_v2.v3.post_mask_epoch_pipeline import (
    DOWNSTREAM_REASON,
    INSTRUCT_STARTED,
    REFERENCE_INTEGRITY_STARTED,
    StageDispatchError,
    default_pair_runner_factory,
    run_removal_pair_epochs,
    run_removal_pair_resource_session,
    shared_qwen_model_identities,
    validate_shared_qwen_models,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    JobExecution,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from tests.test_v3_pair import _decision, _storage
from tests.test_v3_pair import _Judge as _EntityJudge
from tests.test_v3_post_mask_epoch_removal import (
    SHARD,
    _accept,
    _base_config,
    _BooguWorker,
    _Judge,
    _pending_storage,
)

PAIR_SHARD = "shard-000000001-000000000"


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str = "run") -> V3Config:
    """One config that is valid for both the Removal and the Pair stage."""
    base = _base_config(tmp_path, monkeypatch, run_name=run_name)
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    writable = (tmp_path / "workspace" / "data").resolve()
    model = str(pretrained / "Qwen" / "judge")
    config = replace(
        base,
        qwen=QwenServicesConfig(
            annotation=QwenAnnotationConfig(model=model),
            instruction_writer=QwenServiceConfig(model=model),
            candidate_judge=QwenServiceConfig(model=model),
            background_remove_judge=QwenServiceConfig(model=model),
            background_final_judge=QwenServiceConfig(model=model),
            cross_pair_judge=QwenServiceConfig(model=model),
            reference_edit_judge=QwenServiceConfig(model=model),
        ),
        reference_scope=ReferenceScopeConfig(allow_local=True, allow_synthetic_completion=False),
        pair=PairConfig(),
        remove=replace(
            base.remove,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
            adapter_path=None,
        ),
        reference_edit=replace(
            base.reference_edit,
            enabled=True,
            python_executable=writable / "venvs" / "boogu" / "python",
            code_root=writable / "vendor" / "Boogu-Image",
            model_path=writable / "models" / "Boogu-Image",
            min_source_content_area_pixels=1,
            min_source_content_long_side_pixels=1,
        ),
    )
    config.validate()
    return config


class _RecordingExecutor:
    """Serial executor that logs every model call in execution order."""

    def __init__(self, runner: Any, handle: Any, stage: str, log: list[str]) -> None:
        self.runner = runner
        self.handle = handle
        self.stage = stage
        self.log = log

    def execute_batch(self, jobs: Any) -> dict[str, JobExecution]:
        outcomes: dict[str, JobExecution] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            self.log.append(f"{self.stage}:{job.job_type}")
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner.run(job, self.handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


class _NoResultExecutor:
    """Model call that yields nothing: retryable, never a committed attempt."""

    def __init__(self, log: list, stage: str = "removal") -> None:
        self.log = log
        self.stage = stage

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        for job in sorted(jobs, key=lambda item: item.job_id()):
            self.log.append(f"{self.stage}:{job.job_type}")
        return {}


class _FailingEntityJudge(_EntityJudge):
    """Entity judge whose first call fails, so one primary clip stays blocked."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.failed = False

    def decide(self, **kwargs: Any) -> Any:
        self.calls.append("decide")
        if not self.failed:
            self.failed = True
            raise RuntimeError("qwen primary judge unavailable")
        return super().decide(**kwargs)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[V3Config, dict, dict, Any, list]:
    config = _config(tmp_path, monkeypatch, "run-a")
    removal_storage = _pending_storage(config, clip_uids=("clip-1",))
    pair_config = _config(tmp_path, monkeypatch, "run-b")
    pair_storage = _storage(pair_config, entity_types=("subject",))
    storages = {SHARD: removal_storage, PAIR_SHARD: pair_storage}
    eligible = {SHARD: ("clip-1",), PAIR_SHARD: ("clip-1",)}
    return config, storages, eligible, None, []


def _removal_executors(runner: Any, log: list[str], *, worker: Any = None, judge: Any = None) -> dict:
    return {
        RESOURCE_BOOGU: _RecordingExecutor(runner, worker or _BooguWorker(), "removal", log),
        RESOURCE_QWEN: _RecordingExecutor(runner, judge or _Judge([_accept()]), "removal", log),
    }


def test_remove_complete_then_pair_primary_then_cross(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, storages, eligible, _, log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")

    def removal_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors=_removal_executors(runner, log),
        )

    def pair_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors={
                RESOURCE_QWEN: _RecordingExecutor(runner, _EntityJudge(), "pair", log)
            },
        )

    outcome = run_removal_pair_epochs(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        removal_scheduler_factory=removal_factory,
        pair_scheduler_factory=pair_factory,
    )

    assert outcome["remove_completed"] is True
    assert outcome["pair_primary_job_count"] >= 1, "Pair must have real primary work"
    removal_calls = [index for index, entry in enumerate(log) if entry.startswith("removal:")]
    pair_calls = [index for index, entry in enumerate(log) if entry.startswith("pair:")]
    assert removal_calls and pair_calls
    assert max(removal_calls) < min(pair_calls), log
    assert any(entry.endswith("background_removal_generate") for entry in log)
    assert any(entry.endswith("pair_entity_reference_judge") for entry in log)
    assert outcome["pair_completed"] is True
    assert outcome["completed"] is False
    assert outcome["reason"] == DOWNSTREAM_REASON


def test_removal_retryable_blocks_every_pair_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, storages, eligible, _, log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")

    def removal_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors={
                RESOURCE_BOOGU: _NoResultExecutor(log),
                RESOURCE_QWEN: _NoResultExecutor(log),
            },
        )

    def pair_factory(runner: Any) -> Any:
        raise AssertionError("Pair must not be seeded while Removal is unresolved")

    outcome = run_removal_pair_epochs(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        removal_scheduler_factory=removal_factory,
        pair_scheduler_factory=pair_factory,
    )

    assert outcome["remove_completed"] is False
    assert outcome["pair_primary_job_count"] == 0
    assert outcome["pair_cross_job_count"] == 0
    assert outcome["pair_primary_completed"] is False
    assert outcome["pair_cross_completed"] is False
    assert outcome["pair_completed"] is False
    assert outcome["completed"] is False
    assert "removal" in outcome["reason"]
    assert not any(entry.startswith("pair:") for entry in log)


def test_pair_primary_retryable_blocks_only_its_own_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    seen: dict[str, Any] = {}

    def removal_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors=_removal_executors(runner, log),
        )

    class _RecordingRunner:
        """Delegates to the real runner and records the barrier input."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def seed_primary_jobs(self) -> Any:
            return self._inner.seed_primary_jobs()

        def freeze_cross_pair_after_primary_quiescence(self, *, unresolved_job_ids: Any = ()) -> Any:
            seen["unresolved_job_ids"] = tuple(unresolved_job_ids)
            return self._inner.freeze_cross_pair_after_primary_quiescence(
                unresolved_job_ids=unresolved_job_ids
            )

    def pair_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors={
                RESOURCE_QWEN: _RecordingExecutor(runner, _FailingEntityJudge(), "pair", log)
            },
        )

    outcome = run_removal_pair_epochs(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        removal_scheduler_factory=removal_factory,
        pair_scheduler_factory=pair_factory,
        pair_runner_factory=lambda **kwargs: _RecordingRunner(
            default_pair_runner_factory(**kwargs)
        ),
    )

    # The orchestration handed the primary scheduler's unresolved ids straight
    # to the frozen 3b barrier.
    assert seen["unresolved_job_ids"] == outcome["pair_primary_unresolved"]
    assert outcome["pair_primary_unresolved"], "one primary job failed in this fixture"
    assert outcome["remove_completed"] is True
    assert outcome["pair_primary_completed"] is False
    # The blocked clip is absent from the cross pass rather than blocking it.
    assert outcome["pair_cross_completed"] is True


def test_restart_after_removal_receipts_does_not_repeat_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    calls: list[str] = []
    pair_calls: list[str] = []
    worker = _BooguWorker()
    judge = _Judge([_accept()])

    class _CountingPairJudge(_EntityJudge):
        def decide(self, **kwargs: Any) -> Any:
            pair_calls.append("pair-qwen")
            return super().decide(**kwargs)

    class _CountingWorker:
        def edit(self, **kwargs: Any) -> Any:
            calls.append("boogu")
            return worker.edit(**kwargs)

        def close(self) -> None:
            return None

    class _CountingJudge:
        def review(self, **kwargs: Any) -> Any:
            calls.append("qwen")
            return judge.review(**kwargs)

    def build() -> dict[str, Any]:
        log: list[str] = []

        def removal_factory(runner: Any) -> Any:
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                executors=_removal_executors(runner, log, worker=_CountingWorker(), judge=_CountingJudge()),
            )

        def pair_factory(runner: Any) -> Any:
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                executors={RESOURCE_QWEN: _RecordingExecutor(runner, _CountingPairJudge(), "pair", log)},
            )

        return run_removal_pair_epochs(
            config=config,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=ledger,
            removal_scheduler_factory=removal_factory,
            pair_scheduler_factory=pair_factory,
        )

    first = build()
    assert first["remove_completed"] is True
    paid_first = list(calls)
    assert paid_first, "the first run really paid for removal"
    assert first["pair_primary_job_count"] >= 1
    paid_pair = list(pair_calls)
    # Restart: Removal and Pair are durable, so no second model call is paid.
    second = build()
    assert second["remove_completed"] is True
    assert calls == paid_first
    assert pair_calls == paid_pair
    assert second["pair_completed"] is True


def test_restart_after_pair_primary_receipt_does_not_repeat_pair_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    pair_calls: list[str] = []

    class _CountingJudge(_EntityJudge):
        def decide(self, **kwargs: Any) -> Any:
            pair_calls.append("pair-qwen")
            return super().decide(**kwargs)

    def build() -> dict[str, Any]:
        log: list[str] = []

        def removal_factory(runner: Any) -> Any:
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                executors=_removal_executors(runner, log),
            )

        def pair_factory(runner: Any) -> Any:
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                executors={
                    RESOURCE_QWEN: _RecordingExecutor(runner, _CountingJudge(), "pair", log)
                },
            )

        return run_removal_pair_epochs(
            config=config,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=ledger,
            removal_scheduler_factory=removal_factory,
            pair_scheduler_factory=pair_factory,
        )

    first = build()
    assert first["pair_primary_completed"] is True
    paid = list(pair_calls)
    assert paid, "the first run really paid for Pair primary Qwen"
    second = build()
    assert second["pair_primary_completed"] is True
    # CPU replay recovers the receipt chain; no extra Pair Qwen is paid.
    assert pair_calls == paid
    assert second["pair_completed"] is True


def test_pre_launch_existing_pairing_donates_but_is_not_a_cross_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip that already had a pairing before launch can donate, not fall back.

    Proven at the composition level by reading the frozen donor snapshot the
    pipeline produced: the pre-launch clip appears in the frozen donor set and
    is absent from the frozen cross target list.
    """
    import json

    from tests.test_v3_pair import _add_ready_clip, pair_clips
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config, storages, eligible, _, log = _fixture(tmp_path, monkeypatch)
    pair_storage = storages[PAIR_SHARD]
    pair_config = _config(tmp_path, monkeypatch, "run-b")

    # clip-1 gets a pairing BEFORE this launch: it is an existing donor.
    pair_clips(pair_config, pair_storage, judge=_EntityJudge())
    assert pair_storage.read_clip("clip-1").pairing is not None
    # target-x is a fresh clip that will need the fallback.
    _add_ready_clip(
        pair_config,
        pair_storage,
        clip_uid="target-x",
        clip_suffix="20",
        entity_types=("subject",),
    )
    eligible = {
        SHARD: tuple(eligible[SHARD]),
        PAIR_SHARD: ("clip-1", "target-x"),
    }
    storages = {SHARD: storages[SHARD], PAIR_SHARD: pair_storage}
    ledger = GroupLedger(tmp_path / "ledger")

    def removal_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors=_removal_executors(runner, log),
        )

    def pair_factory(runner: Any) -> Any:
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            executors={
                RESOURCE_QWEN: _RecordingExecutor(
                    runner,
                    _ScopedJudge({("target-x", "e1"): "reject"}),
                    "pair",
                    log,
                )
            },
        )

    outcome = run_removal_pair_epochs(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        removal_scheduler_factory=removal_factory,
        pair_scheduler_factory=pair_factory,
    )
    assert outcome["remove_completed"] is True

    snapshot_path = (
        Path(ledger.root) / "semantic" / "pair" / "donor_snapshots" / f"{PAIR_SHARD}.json"
    )
    assert snapshot_path.is_file(), "the pipeline froze a donor snapshot"
    snapshot = json.loads(snapshot_path.read_text())

    targets = list(snapshot["cross_pair_target_clip_uids"])
    donors = {
        entry["clip_uid"] for group in snapshot["groups"] for entry in group["donors"]
    }
    assert "clip-1" not in targets, "a pre-launch pairing never becomes a cross target"
    assert "target-x" in targets, "the fresh clip is the cross target"
    assert "clip-1" in donors, "the pre-launch pairing is a valid frozen donor"
    assert "target-x" not in donors


# ---------------------------------------------------------------------------
# 4b: one ResourceEpochManager session across Removal -> Pair
# ---------------------------------------------------------------------------


class _TrackedResource:
    """Epoch resource that records start/stop on the shared timeline."""

    def __init__(self, name: str, timeline: list[str]) -> None:
        self.name = name
        self.timeline = timeline

    def start(self) -> None:
        self.timeline.append(f"start:{self.name}")

    def stop(self) -> None:
        self.timeline.append(f"stop:{self.name}")

    def counters(self) -> dict[str, Any]:
        return {"start_count":1, "stop_count":1, "startup_wall_seconds":0.0,
                "shutdown_wall_seconds":0.0, "service_seconds":0.0,
                "gpu_slot_job_counts":{}}


class _PassthroughExecutor:
    def __init__(self, runner: Any, handle: Any = None) -> None:
        self.runner = runner
        self.handle = handle

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

        outcomes: dict[str, Any] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner(job, self.handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


class _DispatchHolder:
    """Late-bound dispatch: the session owns the real one and hands it in."""

    def __init__(self) -> None:
        self.dispatch: Any = None

    def __call__(self, job: Any, handle: Any) -> Any:
        if self.dispatch is None:
            raise AssertionError("stage dispatch not bound yet")
        return self.dispatch(job, handle)


class _SharedQwenHandle:
    """One injected handle that both stage runners can use as-is.

    Removal needs ``review()``; Pair accepts any non-string handle and calls
    ``decide()``. A real deployment hands the same endpoint to both.
    """

    def __init__(self) -> None:
        self.reviews = 0
        self.decisions = 0

    def review(self, **kwargs: Any) -> Any:
        self.reviews += 1
        return _accept()

    def decide(self, **kwargs: Any) -> Any:
        self.decisions += 1
        from r2v_data_v2.v3.reference_judge import EntityReferenceDecisionAttempt

        return EntityReferenceDecisionAttempt(
            decision=_decision("full", str(kwargs.get("reference_type", "subject"))),
            raw_responses=("{}",),
            repair_attempts=0,
        )


def _shared_manager(dispatch: Any, timeline: list[str]) -> Any:
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager

    def boogu_factory() -> tuple[Any, Any]:
        return (
            _TrackedResource(RESOURCE_BOOGU, timeline),
            _PassthroughExecutor(dispatch, _BooguWorker()),
        )

    def qwen_factory() -> tuple[Any, Any]:
        return (
            _TrackedResource(RESOURCE_QWEN, timeline),
            _PassthroughExecutor(dispatch, _SharedQwenHandle()),
        )

    return ResourceEpochManager(
        {RESOURCE_BOOGU: boogu_factory, RESOURCE_QWEN: qwen_factory}
    )


def test_scheduler_closes_resource_manager_by_default(tmp_path: Path) -> None:
    timeline: list[str] = []
    manager = _shared_manager(lambda job, handle: None, timeline)
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "ledger"),
        finalize=lambda job, result: (),
        resource_manager=manager,
    )
    scheduler.run([])
    assert scheduler.close_resource_manager_on_exit is True
    # The default owner closes the manager even when nothing ran.
    manager.enter(RESOURCE_QWEN)
    assert manager.open_resource == RESOURCE_QWEN
    scheduler.run([])
    assert manager.open_resource is None
    assert manager.open_count == 0


def test_shared_session_schedulers_do_not_close_the_manager(tmp_path: Path) -> None:
    timeline: list[str] = []
    manager = _shared_manager(lambda job, handle: None, timeline)
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "ledger"),
        finalize=lambda job, result: (),
        resource_manager=manager,
        close_resource_manager_on_exit=False,
    )
    scheduler.run([])
    assert manager.open_resource is None, "an empty run never opens a resource"
    manager.enter(RESOURCE_QWEN)
    assert manager.open_resource == RESOURCE_QWEN
    # A plain scheduler would have closed it in its finally block; this one
    # hands the lifetime to the outer session.
    closing = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "ledger2"),
        finalize=lambda job, result: (),
        resource_manager=manager,
    )
    closing.run([])
    assert manager.open_resource is None, "the default scheduler closed it"
    assert scheduler.close_resource_manager_on_exit is False


def test_removal_to_pair_barrier_does_not_restart_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal judge and Pair primary/cross must share one Qwen start."""
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    timeline: list[str] = []

    def compose(manager: Any, holder: Any) -> dict[str, Any]:
        def removal_scheduler(runner: Any, dispatch: Any) -> Any:
            holder.dispatch = dispatch
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                resource_manager=manager,
                close_resource_manager_on_exit=False,
            )

        def pair_scheduler(runner: Any, dispatch: Any) -> Any:
            holder.dispatch = dispatch
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                resource_manager=manager,
                close_resource_manager_on_exit=False,
            )

        return run_removal_pair_resource_session(
            config=config,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=ledger,
            manager=manager,
            build_removal_scheduler=removal_scheduler,
            build_pair_scheduler=pair_scheduler,
        )

    holder = _DispatchHolder()
    manager = _shared_manager(holder, timeline)
    result = compose(manager, holder)

    assert result["remove_completed"] is True
    assert result["pair_primary_job_count"] >= 1
    assert timeline.count("start:qwen") == 1, timeline
    assert timeline.count("stop:qwen") == 1, timeline
    # The barrier itself never restarts Qwen: start -> ... -> stop, once.
    assert timeline.index("start:qwen") < timeline.index("stop:qwen")
    assert "stop:qwen" not in timeline[: timeline.index("start:qwen") + 1]
    assert result["resource_lifecycle"]["open_resource"] is None
    assert manager.open_count == 0


def test_pair_primary_to_cross_does_not_restart_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    timeline: list[str] = []
    holder = _DispatchHolder()
    manager = _shared_manager(holder, timeline)

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
    )
    assert result["pair_cross_job_count"] >= 0
    assert timeline.count("start:qwen") == 1, timeline
    assert timeline.count("stop:qwen") == 1, timeline


def test_cached_qwen_executor_calls_the_bound_stage_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair jobs must never reach the Removal runner behind the shared Qwen."""
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    manager = _shared_manager(holder, [])

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
    )
    log = list(result["stage_dispatch_log"])
    assert any(entry.startswith("RemovalEpochRunner:") for entry in log), log
    pair_entries = [entry for entry in log if entry.startswith("PairEpochRunner:")]
    assert pair_entries, log
    # Every Pair job went to the Pair runner, never to the Removal runner.
    assert not any(
        entry.startswith("RemovalEpochRunner:pair_") for entry in log
    ), log


def test_removal_incomplete_still_blocks_pair_and_closes_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    timeline: list[str] = []

    class _NoResultExecutor:
        def execute_batch(self, jobs: Any) -> dict[str, Any]:
            return {}

    manager = _shared_manager(holder, timeline)
    manager._factories = {
        RESOURCE_BOOGU: lambda: (_TrackedResource(RESOURCE_BOOGU, timeline), _NoResultExecutor()),
        RESOURCE_QWEN: lambda: (_TrackedResource(RESOURCE_QWEN, timeline), _NoResultExecutor()),
    }

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
    )
    assert result["remove_completed"] is False
    assert result["pair_primary_job_count"] == 0
    assert result["pair_cross_job_count"] == 0
    assert result["stage_dispatch_log"] == ()
    assert manager.open_resource is None
    assert manager.open_count == 0
    assert result["resource_lifecycle"]["open_resource"] is None


def test_shared_session_closes_manager_when_pair_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    timeline: list[str] = []
    manager = _shared_manager(holder, timeline)

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    with pytest.raises(RuntimeError, match="pair exploded"):
        run_removal_pair_resource_session(
            config=config,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=ledger,
            manager=manager,
            build_removal_scheduler=make,
            build_pair_scheduler=lambda runner, _dispatch: (_ for _ in ()).throw(
                RuntimeError("pair exploded")
            ),
        )
    assert manager.open_resource is None
    assert manager.open_count == 0
    assert timeline.count("stop:qwen") >= 1


def test_shared_qwen_model_mismatch_fails_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    holder = _DispatchHolder()
    timeline: list[str] = []
    manager = _shared_manager(holder, timeline)
    mismatched = replace(
        config,
        qwen=replace(config.qwen, cross_pair_judge=QwenServiceConfig(model="/models/other")),
    )
    with pytest.raises(StageDispatchError, match="shared Qwen session"):
        run_removal_pair_resource_session(
            config=mismatched,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=GroupLedger(tmp_path / "ledger"),
            manager=manager,
            expected_served_model_id=str(
                config.qwen.background_remove_judge.model
            ),
            build_removal_scheduler=lambda runner, _dispatch: None,
            build_pair_scheduler=lambda runner, _dispatch: None,
        )
    assert timeline == [], "no resource may start before the compatibility check"


def test_shared_session_restart_does_not_repeat_paid_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new process gets a new manager, but receipts keep the calls paid once."""
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    paid: list[str] = []

    class _CountingQwenHandle(_SharedQwenHandle):
        def review(self, **kwargs: Any) -> Any:
            paid.append("removal-judge")
            return super().review(**kwargs)

        def decide(self, **kwargs: Any) -> Any:
            paid.append("pair-judge")
            return super().decide(**kwargs)

    def build() -> dict[str, Any]:
        holder = _DispatchHolder()
        timeline: list[str] = []
        manager = _shared_manager(holder, timeline)
        manager._factories = {
            RESOURCE_BOOGU: lambda: (
                _TrackedResource(RESOURCE_BOOGU, timeline),
                _PassthroughExecutor(holder, _BooguWorker()),
            ),
            RESOURCE_QWEN: lambda: (
                _TrackedResource(RESOURCE_QWEN, timeline),
                _PassthroughExecutor(holder, _CountingQwenHandle()),
            ),
        }

        def make(runner: Any, dispatch: Any) -> Any:
            holder.dispatch = dispatch
            return ResourceEpochScheduler(
                ledger=ledger,
                finalize=runner.finalize,
                resource_manager=manager,
                close_resource_manager_on_exit=False,
            )

        return run_removal_pair_resource_session(
            config=config,
            storages=storages,
            eligible_clip_uids_by_shard=eligible,
            ledger=ledger,
            manager=manager,
            build_removal_scheduler=make,
            build_pair_scheduler=make,
        )

    first = build()
    assert first["remove_completed"] is True
    assert first["pair_completed"] is True
    paid_first = list(paid)
    assert paid_first, "the first run really paid for Qwen"
    second = build()
    assert second["remove_completed"] is True
    assert second["pair_completed"] is True
    assert paid == paid_first
    assert second["resource_lifecycle"]["open_resource"] is None


def test_production_runner_holds_the_shard_lock_across_removal_and_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4b production wiring: one lock scope spans hydrate -> Removal -> Pair."""
    from r2v_data_v2.v3.post_mask_epoch_groups import resource_epoch_root
    from r2v_data_v2.v3.post_mask_epoch_removal import (
        PreparedRemovalShard,
        build_qwen_epoch_config,
        build_removal_campaign,
        build_removal_epoch_runner,
        removal_shard_paths,
        shard_lock_path,
    )
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager
    from tests.test_v3_post_mask_epoch_removal import (
        _fake_process_manager,
        _group_with,
        _parts_root,
        _pending_storage,
    )

    config = _config(tmp_path, monkeypatch, "run-lock")
    storage = _pending_storage(config, clip_uids=("clip-1",))
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"
    entity_mask_root = _parts_root(tmp_path, (SHARD,))
    lock_path = shard_lock_path(
        removal_shard_paths(
            post_mask_root=post_mask_root,
            entity_mask_root=entity_mask_root,
            shard=SHARD,
        )
    )
    lock_state: list[bool] = []

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.prepare_shard_storage",
        lambda *args, **kwargs: PreparedRemovalShard(
            shard=kwargs["shard"],
            paths=removal_shard_paths(
                post_mask_root=kwargs["post_mask_root"],
                entity_mask_root=kwargs["entity_mask_root"],
                shard=kwargs["shard"],
            ),
            storage=storage,
            clip_uids=("clip-1",),
            ready=1,
            excluded=0,
            corrupt=0,
        ),
    )

    def factories(config: Any, runner: Any, **kwargs: Any) -> dict[str, Any]:
        dispatch = kwargs["job_runner"]

        def boogu_factory() -> tuple[Any, Any]:
            lock_state.append(lock_path.exists())
            return _TrackedResource(RESOURCE_BOOGU, []), _PassthroughExecutor(
                dispatch, _BooguWorker()
            )

        def qwen_factory() -> tuple[Any, Any]:
            lock_state.append(lock_path.exists())
            return _TrackedResource(RESOURCE_QWEN, []), _PassthroughExecutor(
                dispatch, _SharedQwenHandle()
            )

        return {RESOURCE_BOOGU: boogu_factory, RESOURCE_QWEN: qwen_factory}

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_removal_epoch_factories",
        factories,
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.ResourceEpochManager",
        ResourceEpochManager,
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(
        SHARD, campaign=build_removal_campaign(config, entity_mask_root=entity_mask_root)
    )
    outcome = runner_callable(
        group,
        GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
        lambda *a, **k: None,
    )
    assert "pair_primary_completed" in outcome, "production now runs Removal + Pair"
    assert outcome["remove_completed"] is True
    assert outcome["completed"] is False
    assert "downstream" in outcome["reason"]
    assert lock_state, "the shared session really opened resources"
    assert all(lock_state), "every resource open happened under the shard lock"


class _EmptyResultExecutor:
    """Model call that yields nothing: retryable, never a committed attempt."""

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        return {}


def _production_runner_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qwen: Any,
    empty_results: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ``build_removal_epoch_runner`` with fake resources and one handle."""
    from r2v_data_v2.v3.post_mask_epoch_groups import resource_epoch_root
    from r2v_data_v2.v3.post_mask_epoch_removal import (
        PreparedRemovalShard,
        build_qwen_epoch_config,
        build_removal_campaign,
        build_removal_epoch_runner,
        removal_shard_paths,
    )
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager
    from tests.test_v3_post_mask_epoch_removal import (
        _fake_process_manager,
        _group_with,
        _parts_root,
        _pending_storage,
    )

    config = _config(tmp_path, monkeypatch, "run-outcome")
    storage = _pending_storage(config, clip_uids=("clip-1",))
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"
    entity_mask_root = _parts_root(tmp_path, (SHARD,))

    def prepare(*args: Any, **kwargs: Any) -> Any:
        return PreparedRemovalShard(
            shard=kwargs["shard"],
            paths=removal_shard_paths(
                post_mask_root=kwargs["post_mask_root"],
                entity_mask_root=kwargs["entity_mask_root"],
                shard=kwargs["shard"],
            ),
            storage=storage,
            clip_uids=("clip-1",),
            ready=1,
            excluded=0,
            corrupt=0,
        )

    def factories(config: Any, runner: Any, **kwargs: Any) -> dict[str, Any]:
        dispatch = kwargs["job_runner"]
        return {
            RESOURCE_BOOGU: lambda: (
                _TrackedResource(RESOURCE_BOOGU, []),
                _EmptyResultExecutor()
                if empty_results
                else _PassthroughExecutor(dispatch, _BooguWorker()),
            ),
            RESOURCE_QWEN: lambda: (
                _TrackedResource(RESOURCE_QWEN, []),
                _EmptyResultExecutor()
                if empty_results
                else _PassthroughExecutor(dispatch, qwen),
            ),
        }

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.prepare_shard_storage", prepare
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_removal_epoch_factories", factories
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.ResourceEpochManager", ResourceEpochManager
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(
        SHARD, campaign=build_removal_campaign(config, entity_mask_root=entity_mask_root)
    )
    events: list[dict[str, Any]] = []
    outcome = runner_callable(
        group,
        GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
        lambda event, **payload: events.append({"event":event, **payload}),
    )
    return outcome, events


def test_production_wrapper_reports_removal_incomplete_with_unresolved_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal retryable: Pair never starts and unresolved must not read 0."""
    outcome, events = _production_runner_outcome(
        tmp_path, monkeypatch, qwen=_SharedQwenHandle(), empty_results=True
    )
    assert outcome["remove_completed"] is False
    assert outcome["completed"] is False
    assert outcome["reason"] == "background removal incomplete"
    finished = [
        event for event in events if event["event"] == "post_mask_removal_epoch_finished"
    ]
    assert finished
    # Removal's own unresolved jobs are counted, not just Pair's two lists.
    assert finished[-1]["unresolved"] > 0, finished
    assert outcome["pair_primary_job_count"] == 0
    assert outcome["pair_cross_job_count"] == 0


def test_production_wrapper_keeps_pair_incomplete_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal complete + Pair unresolved keeps the composition's reason."""

    class _FailingPairHandle(_SharedQwenHandle):
        """Removal judge accepts; the first Pair decision is retryable."""

        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def decide(self, **kwargs: Any) -> Any:
            if not self.failed:
                self.failed = True
                raise RuntimeError("pair qwen unavailable")
            return super().decide(**kwargs)

    outcome, events = _production_runner_outcome(
        tmp_path, monkeypatch, qwen=_FailingPairHandle()
    )
    assert outcome["remove_completed"] is True
    # Hard assertion: this regression must never silently skip.
    assert outcome["pair_primary_unresolved"], (
        "fixture must produce unresolved Pair primary work"
    )
    assert outcome["pair_completed"] is False
    assert outcome["reason"] == "pair resource epoch incomplete", outcome["reason"]
    assert outcome["completed"] is False
    finished = [
        event
        for event in events
        if event["event"] == "post_mask_removal_epoch_finished"
    ]
    assert finished[-1]["unresolved"] > 0


def _stage_counts_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Any], dict[str, Any], Any, list[str]]:
    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    paid: list[str] = []
    return config, storages, eligible, ledger, paid


def _stage_counts_run(
    config: Any,
    storages: dict[str, Any],
    eligible: dict[str, Any],
    ledger: Any,
    paid: list[str],
) -> dict[str, Any]:
    """Run the shared-session composition over the given durable state."""
    from r2v_data_v2.v3.post_mask_epoch_pipeline import (
        run_removal_pair_resource_session,
    )
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager

    class _CountingQwen(_SharedQwenHandle):
        def review(self, **kwargs: Any) -> Any:
            paid.append("removal-judge")
            return super().review(**kwargs)

        def decide(self, **kwargs: Any) -> Any:
            paid.append("pair-judge")
            return super().decide(**kwargs)

    holder = _DispatchHolder()

    def qwen_factory() -> tuple[Any, Any]:
        return _TrackedResource(RESOURCE_QWEN, []), _PassthroughExecutor(
            holder, _CountingQwen()
        )

    def boogu_factory() -> tuple[Any, Any]:
        return _TrackedResource(RESOURCE_BOOGU, []), _PassthroughExecutor(
            holder, _BooguWorker()
        )

    manager = ResourceEpochManager(
        {RESOURCE_BOOGU: boogu_factory, RESOURCE_QWEN: qwen_factory}
    )

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    return run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
    )


def test_stage_counts_pair_written_only_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pair import PairEpochRunner

    config, storages, eligible, ledger, _paid = _stage_counts_build(tmp_path, monkeypatch)
    outcome = _stage_counts_run(config, storages, eligible, ledger, _paid)
    assert outcome["pair_completed"] is True, outcome["reason"]
    assert outcome["pair_stats"], "a completed Pair must report reconciled stats"
    for shard, storage in storages.items():
        run = storage.read_run()
        runner = PairEpochRunner(
            config,
            {shard: storage},
            GroupLedger(Path(ledger.root)),
            eligible_clip_uids_by_shard={shard: list(eligible[shard])},
        )
        reconciled = runner.reconcile_stats(shard).to_dict()
        for field, value in reconciled.items():
            assert run.counts[f"pair.{field}"] == value, (shard, field)
        assert outcome["pair_stats"][shard] == reconciled


def test_stage_counts_restart_is_idempotent_and_pays_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storages, eligible, ledger, paid = _stage_counts_build(tmp_path, monkeypatch)
    first = _stage_counts_run(config, storages, eligible, ledger, paid)
    assert first["pair_completed"] is True
    paid_first = list(paid)
    before = {
        shard: dict(storage.read_run().counts) for shard, storage in storages.items()
    }
    second = _stage_counts_run(config, storages, eligible, ledger, paid)
    assert second["pair_completed"] is True
    # Restart pays no model call and rewrites exactly the same Pair stage
    # counts. (Removal's own counters legitimately re-classify a finished
    # clip as skipped_existing, which is not this test's subject.)
    assert paid == paid_first
    for shard, storage in storages.items():
        after = {
            key: value
            for key, value in storage.read_run().counts.items()
            if key.startswith("pair.")
        }
        expected = {
            key: value
            for key, value in before[shard].items()
            if key.startswith("pair.")
        }
        assert after == expected, shard
    assert second["pair_stats"] == first["pair_stats"]


def test_stage_counts_not_written_when_pair_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.post_mask_epoch_pipeline import (
        run_removal_pair_resource_session,
    )
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager

    config, storages, eligible, _, _log = _fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")

    class _NoResultExecutor:
        def execute_batch(self, jobs: Any) -> dict[str, Any]:
            return {}

    holder = _DispatchHolder()
    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: lambda: (_TrackedResource(RESOURCE_BOOGU, []), _NoResultExecutor()),
            RESOURCE_QWEN: lambda: (_TrackedResource(RESOURCE_QWEN, []), _NoResultExecutor()),
        }
    )

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    outcome = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
    )
    assert outcome["remove_completed"] is False
    assert outcome["pair_completed"] is False
    assert outcome.get("pair_stats", {}) == {}
    for shard, storage in storages.items():
        counts = storage.read_run().counts
        assert not any(key.startswith("pair.") for key in counts), shard


# ---------------------------------------------------------------------------
# 5c: full composition with a real Reference Edit repairable chain
# ---------------------------------------------------------------------------


class _CompositionQwenHandle:
    """One Qwen handle serving Removal, Pair and Reference Edit stages.

    ``review`` kwargs distinguish the callers: the completion review carries
    ``candidate_rgb``; the removal review carries ``candidate_image``.
    """

    def __init__(self) -> None:
        self.removal_reviews = 0
        self.completion_reviews = 0
        self.pair_decisions = 0
        self.integrity_reviews = 0

    def review(self, **kwargs: Any) -> Any:
        if "final_reference" in kwargs:
            from r2v_data_v2.v3.reference_integrity import (
                ReferenceIntegrityReviewAttempt,
            )
            from tests.test_v3_reference_integrity import _review as _integrity_review

            self.integrity_reviews += 1
            review = _integrity_review(accept=True, reason="clean completion")
            return ReferenceIntegrityReviewAttempt(
                review=review,
                raw_response=review.model_dump_json(),
                raw_responses=(review.model_dump_json(),),
                finish_reasons=("stop",),
            )
        if "candidate_rgb" in kwargs:
            from tests.test_v3_reference_edit_boogu import _completion_review

            self.completion_reviews += 1
            return _completion_review(accept=True)
        from tests.test_v3_post_mask_epoch_removal import _accept as _removal_accept

        self.removal_reviews += 1
        return _removal_accept()

    def decide(self, **kwargs: Any) -> Any:
        from r2v_data_v2.v3.reference_judge import EntityReferenceDecisionAttempt

        self.pair_decisions += 1
        # A repairable decision gives the Reference Edit stage real work.
        return EntityReferenceDecisionAttempt(
            decision=_decision(
                "repairable", str(kwargs.get("reference_type", "subject"))
            ),
            raw_responses=("{}",),
            repair_attempts=0,
        )


class _CompositionSamHandle:
    """The SAM epoch resource handle: a segmenter backend fake."""

    def __init__(self) -> None:
        self.calls = 0

    def track(self, **kwargs: Any) -> Any:
        import numpy as np
        from PIL import Image

        from r2v_data_v2.v3.sam3_backend import (
            BackendMaskObservation,
            EntityTrackResult,
        )

        self.calls += 1
        with Image.open(kwargs["frame_paths"][0]) as opened:
            width, height = opened.size
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 3 : height // 3 + height // 8,
             width // 3 : width // 3 + width // 8] = True
        return EntityTrackResult(
            status="ready",
            observations=(
                BackendMaskObservation(
                    slot=5, mask=mask, confidence=0.9, object_id="target"
                ),
            ),
        )


class _CompositionBooguHandle:
    def __init__(self) -> None:
        self.calls = 0

    def edit(self, **kwargs: Any) -> Any:
        import io

        from PIL import Image

        from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput

        self.calls += 1
        buffer = io.BytesIO()
        Image.new(
            "RGB", (int(kwargs["width"]), int(kwargs["height"])), (191, 22, 43)
        ).save(buffer, format="PNG")
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=buffer.getvalue(),
            original_instruction=instruction,
            rewritten_instruction="rewritten",
            effective_instruction="rewritten",
            worker_metadata={},
        )


def _composition_manager(
    dispatch: Any, timeline: list[str], *, qwen: Any = None
) -> tuple[Any, dict[str, Any]]:
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager

    boogu = _CompositionBooguHandle()
    qwen = qwen if qwen is not None else _CompositionQwenHandle()
    sam = _CompositionSamHandle()

    def boogu_factory() -> tuple[Any, Any]:
        return (
            _TrackedResource(RESOURCE_BOOGU, timeline),
            _PassthroughExecutor(dispatch, boogu),
        )

    def qwen_factory() -> tuple[Any, Any]:
        return (
            _TrackedResource(RESOURCE_QWEN, timeline),
            _PassthroughExecutor(dispatch, qwen),
        )

    def sam_factory() -> tuple[Any, Any]:
        return (
            _TrackedResource(RESOURCE_SAM, timeline),
            _PassthroughExecutor(dispatch, sam),
        )

    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: boogu_factory,
            RESOURCE_QWEN: qwen_factory,
            RESOURCE_SAM: sam_factory,
        }
    )
    return manager, {"boogu": boogu, "qwen": qwen, "sam": sam}


def _reference_edit_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[V3Config, dict, dict]:
    """Removal pending clip + a Pair storage with a repairable reference."""
    from tests.test_v3_pair import pair_clips
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _config(tmp_path, monkeypatch, "run-a")
    removal_storage = _pending_storage(config, clip_uids=("clip-1",))
    pair_config = _config(tmp_path, monkeypatch, "run-b")
    pair_storage = _storage(pair_config, entity_types=("subject",))
    pair_clips(pair_config, pair_storage, judge=_ScopedJudge({("clip-1", "e1"): "repairable"}))
    clip = pair_storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    assert clip.references.entities[0].completeness == "repairable"
    storages = {SHARD: removal_storage, PAIR_SHARD: pair_storage}
    eligible = {SHARD: ("clip-1",), PAIR_SHARD: ("clip-1",)}
    return config, storages, eligible


def _compose_with_reference_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeline: list[str],
    fixture: tuple[V3Config, dict, dict] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        ReferenceEditEpochRunner,
    )

    if fixture is None:
        fixture = _reference_edit_fixture(tmp_path, monkeypatch)
    config, storages, eligible = fixture
    # Same durable ledger root: the restart resumes the same receipts.
    ledger = GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    manager, handles = _composition_manager(holder, timeline)

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
        build_reference_edit_scheduler=make,
        reference_edit_runner_factory=(
            lambda **kwargs: ReferenceEditEpochRunner(
                kwargs["config"],
                kwargs["storages"],
                kwargs["ledger"],
                eligible_clip_uids_by_shard=kwargs[
                    "eligible_clip_uids_by_shard"
                ],
            )
        ),
    )
    return result, storages, handles, ledger


def test_composition_completes_reference_edit_with_three_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline: list[str] = []
    result, storages, handles, ledger = _compose_with_reference_edit(
        tmp_path, monkeypatch, timeline
    )

    assert result["remove_completed"] is True
    assert result["pair_completed"] is True
    assert result["reference_edit_completed"] is True, result.get(
        "reference_edit_reconcile_error"
    )
    assert result["reference_edit_unresolved"] == ()
    assert result["reference_edit_stats"], "reconciled stats must be reported"
    assert result["reason"] == "downstream resource-epoch phases are not wired"
    assert result["completed"] is False

    # The repairable chain really paid all three resources.
    assert handles["boogu"].calls >= 1
    assert handles["qwen"].completion_reviews >= 1
    assert handles["sam"].calls >= 1

    # stage_counts equal the reconciled stats.
    pair_storage = storages[PAIR_SHARD]
    stats = result["reference_edit_stats"][PAIR_SHARD]
    run = pair_storage.read_run()
    for field, value in stats.items():
        assert run.counts[f"reference_edit.{field}"] == value, field

    # The shared dispatch routed every Reference Edit job to its own runner.
    dispatch_log = ",".join(result["stage_dispatch_log"])
    assert "ReferenceEditEpochRunner:reference_edit_boogu_generate" in dispatch_log
    assert "ReferenceEditEpochRunner:reference_edit_qwen_review" in dispatch_log
    assert "ReferenceEditEpochRunner:reference_edit_sam_review" in dispatch_log
    assert "PairEpochRunner:reference_edit_" not in dispatch_log
    assert "RemovalEpochRunner:reference_edit_" not in dispatch_log

    # Resource lifecycle: at most one heavy resource open, all closed at end.
    assert result["resource_lifecycle"]["max_open_observed"] == 1
    assert result["resource_lifecycle"]["open_resource"] is None
    assert timeline.count("start:boogu") >= 1
    assert timeline.count("start:qwen") >= 1
    assert timeline.count("start:sam") >= 1
    # boogu -> qwen -> sam switches actually happened during Reference Edit.
    removal_qwen = timeline.index("start:qwen")
    re_boogu = timeline.index("start:boogu", removal_qwen)
    re_qwen = timeline.index("start:qwen", re_boogu)
    re_sam = timeline.index("start:sam", re_qwen)
    assert re_boogu < re_qwen < re_sam
    del ledger


def test_composition_reference_edit_restart_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _reference_edit_fixture(tmp_path, monkeypatch)
    timeline: list[str] = []
    first, storages, _handles, _ledger = _compose_with_reference_edit(
        tmp_path, monkeypatch, timeline, fixture
    )
    assert first["reference_edit_completed"] is True
    pair_storage = storages[PAIR_SHARD]
    before_counts = dict(pair_storage.read_run().counts)

    # Restart the whole composition over the same durable state.
    restart_timeline: list[str] = []
    second, _storages, restart_handles, _ = _compose_with_reference_edit(
        tmp_path, monkeypatch, restart_timeline, fixture
    )
    assert second["reference_edit_completed"] is True
    assert second["reference_edit_stats"] == first["reference_edit_stats"]
    assert dict(pair_storage.read_run().counts) == before_counts
    # Zero extra Reference Edit model calls on restart.
    assert restart_handles["boogu"].calls == 0
    assert restart_handles["qwen"].completion_reviews == 0
    assert restart_handles["sam"].calls == 0


def test_composition_pair_incomplete_blocks_reference_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pair incomplete: no Reference Edit plan, seed or model call."""
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        ReferenceEditEpochRunner,
    )

    config, storages, eligible = _reference_edit_fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")
    timeline: list[str] = []
    holder = _DispatchHolder()
    manager, handles = _composition_manager(holder, timeline)

    class _NoResultExecutor:
        def execute_batch(self, jobs: Any) -> dict[str, Any]:
            return {}

    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager

    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: lambda: (
                _TrackedResource(RESOURCE_BOOGU, timeline),
                _NoResultExecutor(),
            ),
            RESOURCE_QWEN: lambda: (
                _TrackedResource(RESOURCE_QWEN, timeline),
                _NoResultExecutor(),
            ),
            RESOURCE_SAM: lambda: (
                _TrackedResource(RESOURCE_SAM, timeline),
                _NoResultExecutor(),
            ),
        }
    )

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
        build_reference_edit_scheduler=make,
        reference_edit_runner_factory=(
            lambda **kwargs: ReferenceEditEpochRunner(
                kwargs["config"],
                kwargs["storages"],
                kwargs["ledger"],
                eligible_clip_uids_by_shard=kwargs[
                    "eligible_clip_uids_by_shard"
                ],
            )
        ),
    )
    assert result["remove_completed"] is False
    assert result["pair_completed"] is False
    assert result["reference_edit_completed"] is False
    assert result["reference_edit_job_count"] == 0
    assert result["reference_edit_unresolved"] == ()
    assert result["reference_edit_stats"] == {}
    semantic_root = Path(ledger.root) / "semantic" / "reference_edit"
    assert not semantic_root.exists()
    pair_storage = storages[PAIR_SHARD]
    counts = pair_storage.read_run().counts
    assert not any(key.startswith("reference_edit.") for key in counts)
    # No Reference Edit model call was paid.
    dispatch_log = ",".join(result["stage_dispatch_log"])
    assert "reference_edit_boogu_generate" not in dispatch_log
    assert handles["boogu"].calls == 0


def _two_shard_reference_edit_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[V3Config, dict, dict]:
    """Two independent Pair shards, both with a repairable reference."""
    from tests.test_v3_pair import pair_clips
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _config(tmp_path, monkeypatch, "run-a")
    removal_storage = _pending_storage(config, clip_uids=("clip-1",))
    storages: dict[str, Any] = {SHARD: removal_storage}
    eligible: dict[str, Any] = {SHARD: ("clip-1",)}
    for index, shard in enumerate(("shard-a", "shard-b")):
        shard_config = _config(tmp_path, monkeypatch, f"run-{index + 1}")
        storage = _storage(shard_config, entity_types=("subject",))
        pair_clips(
            shard_config,
            storage,
            judge=_ScopedJudge({("clip-1", "e1"): "repairable"}),
        )
        clip = storage.read_clip("clip-1")
        assert clip.references.entities[0].completeness == "repairable"
        storages[shard] = storage
        eligible[shard] = ("clip-1",)
    return config, storages, eligible


def _run_two_shard_composition(
    config: V3Config,
    storages: dict,
    eligible: dict,
    *,
    after_reference_edit: Any = None,
) -> dict[str, Any]:
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        ReferenceEditEpochRunner,
    )

    ledger = GroupLedger(Path(storages[SHARD].root).parent.parent / "epoch-ledger")
    timeline: list[str] = []
    holder = _DispatchHolder()
    manager, _handles = _composition_manager(holder, timeline)

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    def reference_edit_scheduler(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        scheduler = ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )
        if after_reference_edit is None:
            return scheduler

        class _Hooked:
            def run(self, jobs: Any) -> Any:
                outcome = scheduler.run(jobs)
                after_reference_edit()
                return outcome

        return _Hooked()

    return run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
        build_reference_edit_scheduler=reference_edit_scheduler,
        reference_edit_runner_factory=(
            lambda **kwargs: ReferenceEditEpochRunner(
                kwargs["config"],
                kwargs["storages"],
                kwargs["ledger"],
                eligible_clip_uids_by_shard=kwargs[
                    "eligible_clip_uids_by_shard"
                ],
            )
        ),
    )


def test_two_shard_reconcile_failure_writes_zero_partial_stage_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failing shard must not leave partial counts on its sibling."""
    config, storages, eligible = _two_shard_reference_edit_fixture(
        tmp_path, monkeypatch
    )
    pair_shards = [shard for shard in storages if shard != SHARD]
    snapshot: dict[str, dict[str, int]] = {}

    def corrupt_shard_b() -> None:
        """Snapshot both shards, then corrupt shard B's publication."""
        import r2v_data_v2.v3.storage as storage_module

        for shard in pair_shards:
            snapshot[shard] = dict(storages[shard].read_run().counts)
        storage_b = storages["shard-b"]
        clip = storage_b.read_clip("clip-1")
        assert clip.reference_edit is not None
        tampered = [
            entity.model_copy(update={"metadata_path": "clips/tampered.json"})
            if index == 0
            else entity
            for index, entity in enumerate(clip.reference_edit.entities)
        ]
        storage_module.write_json_atomic(
            storage_b.clip_path("clip-1"),
            clip.model_copy(
                update={
                    "reference_edit": clip.reference_edit.model_copy(
                        update={"entities": tampered}
                    )
                }
            ).model_dump(mode="json"),
        )

    result = _run_two_shard_composition(
        config, storages, eligible, after_reference_edit=corrupt_shard_b
    )

    assert result["reference_edit_completed"] is False
    assert result["reference_edit_stats"] == {}
    assert result["reference_edit_reconcile_error"], "the failure must be reported"
    for shard in pair_shards:
        after = dict(storages[shard].read_run().counts)
        # The snapshot is taken after Pair wrote its counts and before the
        # Reference Edit reconcile, so ANY difference here is partial work.
        assert after == snapshot[shard], f"{shard} was left with partial counts"
        assert not any(
            key.startswith("reference_edit.") for key in after
        ), f"{shard} has partial reference_edit keys"

    # Restart with the shard repaired: both shards are written atomically.
    import r2v_data_v2.v3.storage as storage_module
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        ReferenceEditEpochRunner,
    )

    storage_b = storages["shard-b"]
    stored = storage_b.read_clip("clip-1")
    ledger_root = Path(storages[SHARD].root).parent.parent / "epoch-ledger"
    repair = ReferenceEditEpochRunner(
        config,
        {"shard-b": storage_b},
        GroupLedger(ledger_root),
        eligible_clip_uids_by_shard={"shard-b": ["clip-1"]},
    )
    # Read the frozen plan directly: the strict verifier intentionally refuses
    # the tampered state we are about to repair.
    plan = json.loads(
        Path(repair._plan_path("shard-b")).read_text(encoding="utf-8")
    )
    expected = repair._reconstruct_clip_publication(
        "shard-b", storage_b, "clip-1", plan["clips"]["clip-1"]
    )
    storage_module.write_json_atomic(
        storage_b.clip_path("clip-1"),
        stored.model_copy(
            update={
                "references": expected[0],
                "pairing": expected[1],
                "reference_edit": expected[2],
            }
        ).model_dump(mode="json"),
    )

    second = _run_two_shard_composition(config, storages, eligible)
    assert second["reference_edit_completed"] is True, second.get(
        "reference_edit_reconcile_error"
    )
    for shard in pair_shards:
        counts = storages[shard].read_run().counts
        stats = second["reference_edit_stats"][shard]
        for field, value in stats.items():
            assert counts[f"reference_edit.{field}"] == value, (shard, field)


# ---------------------------------------------------------------------------
# Reference Integrity -> Instruct composition
# ---------------------------------------------------------------------------


def _enable_reference_integrity(config: V3Config) -> V3Config:
    """Turn the Reference Integrity stage on with an already-served judge."""
    model = str(config.qwen.background_remove_judge.model)
    updated = replace(
        config,
        qwen=replace(
            config.qwen,
            reference_integrity_judge=QwenServiceConfig(model=model),
        ),
        reference_integrity=replace(config.reference_integrity, enabled=True),
    )
    updated.validate()
    return updated


def _with_instruction_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the Pair fixture's annotation a deterministic instruction template.

    ``instruct_clips(client=None)`` renders ``annotation.instruction_template``.
    The Pair fixture writes an annotation without one, and rewriting the
    annotation afterwards would invalidate the frames, masks and pairing it also
    writes, so inject the default where the fixture builds its annotation.
    """
    import tests.test_v3_pair as pair_module
    import tests.test_v3_post_mask_epoch_removal as removal_module

    for module in (pair_module, removal_module):
        real = module.AnnotationState

        def state(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            # A ready annotation carries exactly one of the two, so replace the
            # legacy caption with the deterministic template. A paired
            # background must be mentioned exactly once, and an annotation
            # without one must not mention it at all.
            kwargs.pop("t2v_caption", None)
            kwargs.setdefault(
                "instruction_template",
                (
                    "{{entity_1}} moves through {{background}}."
                    if kwargs.get("background") is not None
                    else "{{entity_1}} crosses the frame."
                ),
            )
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, "AnnotationState", state)


def _reference_integrity_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str = "run-a"
) -> tuple[V3Config, dict, dict]:
    """Reference Edit fixture with Reference Integrity enabled and Instruct ready."""
    _with_instruction_template(monkeypatch)
    config, storages, eligible = _reference_edit_fixture(tmp_path, monkeypatch)
    return _enable_reference_integrity(config), storages, eligible


def _compose_with_reference_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeline: list[str],
    *,
    fixture: tuple[V3Config, dict, dict] | None = None,
    qwen: Any = None,
    ledger: GroupLedger | None = None,
    build_reference_edit_scheduler: Any = None,
    factory_log: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
        ReferenceEditEpochRunner,
    )
    from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
        ReferenceIntegrityEpochRunner,
    )

    if fixture is None:
        fixture = _reference_integrity_fixture(tmp_path, monkeypatch)
    config, storages, eligible = fixture
    ledger = ledger if ledger is not None else GroupLedger(tmp_path / "ledger")
    holder = _DispatchHolder()
    manager, handles = _composition_manager(holder, timeline, qwen=qwen)

    def make(runner: Any, dispatch: Any) -> Any:
        holder.dispatch = dispatch
        return ResourceEpochScheduler(
            ledger=ledger,
            finalize=runner.finalize,
            resource_manager=manager,
            close_resource_manager_on_exit=False,
        )

    result = run_removal_pair_resource_session(
        config=config,
        storages=storages,
        eligible_clip_uids_by_shard=eligible,
        ledger=ledger,
        manager=manager,
        build_removal_scheduler=make,
        build_pair_scheduler=make,
        build_reference_edit_scheduler=(
            build_reference_edit_scheduler or make
        ),
        build_reference_integrity_scheduler=make,
        reference_edit_runner_factory=(
            lambda **kwargs: _stage_runner(
                factory_log,
                "reference_edit",
                ReferenceEditEpochRunner(**kwargs),
            )
        ),
        reference_integrity_runner_factory=(
            lambda **kwargs: _stage_runner(
                factory_log,
                "reference_integrity",
                ReferenceIntegrityEpochRunner(**kwargs),
            )
        ),
    )
    return result, storages, handles, ledger


def _stage_runner(log: list[str] | None, stage: str, runner: Any) -> Any:
    """Build a stage runner while recording that the stage was entered."""
    if log is not None:
        log.append(stage)
    return runner


def _stage_positions(log: Any, runner_name: str) -> list[int]:
    return [
        index
        for index, entry in enumerate(log)
        if entry.startswith(f"{runner_name}:")
    ]


def test_composition_runs_reference_integrity_then_instruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference Edit -> Reference Integrity -> deterministic Instruct."""
    timeline: list[str] = []
    result, storages, handles, ledger = _compose_with_reference_integrity(
        tmp_path, monkeypatch, timeline
    )

    assert result["remove_completed"] is True
    assert result["pair_completed"] is True
    assert result["reference_edit_completed"] is True, result.get(
        "reference_edit_reconcile_error"
    )
    assert result["reference_integrity_completed"] is True, result.get(
        "reference_integrity_reconcile_error"
    )
    assert result["reference_integrity_unresolved"] == ()
    assert result["reference_integrity_job_count"] >= 1, "a real review job ran"
    assert result["reference_integrity_stats"], "reconciled stats must be reported"
    assert result["instruct_completed"] is True

    # Every Reference Edit model call precedes the first Reference Integrity one.
    log = list(result["stage_dispatch_log"])
    reference_edit_positions = _stage_positions(log, "ReferenceEditEpochRunner")
    reference_integrity_positions = _stage_positions(
        log, "ReferenceIntegrityEpochRunner"
    )
    assert reference_edit_positions, log
    assert reference_integrity_positions, log
    assert max(reference_edit_positions) < min(reference_integrity_positions), log
    assert len(reference_integrity_positions) == result[
        "reference_integrity_job_count"
    ], "every seeded integrity job really ran"
    # Instruct contributes no model job and no model call at all.
    assert "instruct" not in ",".join(log)
    assert handles["qwen"].integrity_reviews == result["reference_integrity_job_count"]

    # Every eligible clip in every shard reached both new stages.
    for shard in storages:
        clip = storages[shard].read_clip("clip-1")
        assert clip.reference_integrity is not None
        assert clip.reference_integrity.status == "ready"
        assert clip.instruction is not None
        assert clip.instruction.status == "ready"
        counts = storages[shard].read_run().counts
        assert any(key.startswith("reference_integrity.") for key in counts)
        assert any(key.startswith("instruct.") for key in counts)
    assert result["instruct_stats"][PAIR_SHARD]["processed"] == 1
    assert result["instruct_stats"][PAIR_SHARD]["failed"] == 0
    composition_root = Path(ledger.root) / "composition"
    assert (composition_root / f"{REFERENCE_INTEGRITY_STARTED}.json").is_file()
    assert (composition_root / f"{INSTRUCT_STARTED}.json").is_file()

    assert result["completed"] is False
    assert result["reason"] == DOWNSTREAM_REASON


class _InertScheduler:
    """A scheduler that executes nothing and leaves every job unresolved."""

    def __init__(self, jobs: Any = ()) -> None:
        self._jobs = tuple(jobs)

    def run(self, jobs: Any = None) -> dict[str, Any]:
        seeded = tuple(jobs) if jobs is not None else self._jobs
        return {"unresolved_job_ids": tuple(job.job_id() for job in seeded)}


def test_reference_edit_incomplete_blocks_reference_integrity_and_instruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference Edit incomplete: no handoff, no plan, no seed, no Instruct."""
    timeline: list[str] = []
    result, storages, handles, ledger = _compose_with_reference_integrity(
        tmp_path,
        monkeypatch,
        timeline,
        build_reference_edit_scheduler=lambda runner, dispatch: _InertScheduler(),
    )

    assert result["pair_completed"] is True
    assert result["reference_edit_job_count"] >= 1
    assert result["reference_edit_completed"] is False
    assert result["reference_edit_unresolved"], "the seeded jobs stayed unresolved"
    assert result["reason"] == "reference edit resource epoch incomplete"
    assert "reference_edit_boogu_generate" not in ",".join(
        result["stage_dispatch_log"]
    ), "the incomplete stage paid no Reference Edit model call"

    assert result["reference_integrity_completed"] is False
    assert result["reference_integrity_job_count"] == 0
    assert result["reference_integrity_unresolved"] == ()
    assert result["reference_integrity_stats"] == {}
    assert result["instruct_completed"] is False
    assert result["instruct_stats"] == {}
    assert handles["qwen"].integrity_reviews == 0

    composition_root = Path(ledger.root) / "composition"
    assert not (composition_root / f"{REFERENCE_INTEGRITY_STARTED}.json").exists()
    assert not (composition_root / f"{INSTRUCT_STARTED}.json").exists()
    assert not (Path(ledger.root) / "semantic" / "reference_integrity").exists()
    counts = storages[PAIR_SHARD].read_run().counts
    assert not any(key.startswith("reference_integrity.") for key in counts)
    assert not any(key.startswith("instruct.") for key in counts)
    assert storages[PAIR_SHARD].read_clip("clip-1").instruction is None


class _IntegrityFailingQwenHandle(_CompositionQwenHandle):
    """Breaks only the Reference Integrity review call."""

    def review(self, **kwargs: Any) -> Any:
        if "final_reference" in kwargs:
            self.integrity_reviews += 1
            raise RuntimeError("integrity judge exploded")
        return super().review(**kwargs)


def test_reference_integrity_incomplete_blocks_instruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reference Integrity incomplete: no Instruct handoff, no instruction."""
    timeline: list[str] = []
    result, storages, _handles, ledger = _compose_with_reference_integrity(
        tmp_path,
        monkeypatch,
        timeline,
        qwen=_IntegrityFailingQwenHandle(),
    )

    assert result["reference_edit_completed"] is True
    assert result["reference_integrity_completed"] is False
    assert result["reference_integrity_job_count"] >= 1
    assert result["reference_integrity_unresolved"], "the review never committed"
    assert result["reference_integrity_stats"] == {}
    assert result["reason"] == "reference integrity resource epoch incomplete"
    assert result["instruct_completed"] is False
    assert result["instruct_stats"] == {}

    composition_root = Path(ledger.root) / "composition"
    assert (composition_root / f"{REFERENCE_INTEGRITY_STARTED}.json").is_file()
    assert not (composition_root / f"{INSTRUCT_STARTED}.json").exists()
    clip = storages[PAIR_SHARD].read_clip("clip-1")
    assert clip.instruction is None
    counts = storages[PAIR_SHARD].read_run().counts
    assert not any(key.startswith("reference_integrity.") for key in counts)
    assert not any(key.startswith("instruct.") for key in counts)


def test_composition_restart_after_instruct_replays_nothing_paid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both restart barriers resume without replaying any paid upstream work."""
    import r2v_data_v2.v3.post_mask_epoch_pipeline as pipeline_module

    # One fixture and one ledger root: every session resumes the exact durable
    # state the previous one left behind.
    fixture = _reference_integrity_fixture(tmp_path, monkeypatch)
    ledger = GroupLedger(tmp_path / "ledger")

    # 1. Simulate a crash between the Reference Integrity publication and the
    #    Instruct handoff, so only the earlier barrier is durable.
    real_write = pipeline_module.write_composition_handoff
    crashed = {"done": False}

    def crash_before_instruct(
        ledger_arg: Any, stage: str, *, eligible_clip_uids_by_shard: Any
    ) -> None:
        if stage == INSTRUCT_STARTED and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated crash before the Instruct handoff")
        real_write(
            ledger_arg,
            stage,
            eligible_clip_uids_by_shard=eligible_clip_uids_by_shard,
        )

    monkeypatch.setattr(
        pipeline_module, "write_composition_handoff", crash_before_instruct
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        _compose_with_reference_integrity(
            tmp_path, monkeypatch, [], fixture=fixture, ledger=ledger
        )

    composition_root = Path(ledger.root) / "composition"
    assert (composition_root / f"{REFERENCE_INTEGRITY_STARTED}.json").is_file()
    assert not (composition_root / f"{INSTRUCT_STARTED}.json").exists()
    storages = fixture[1]
    clip = storages[PAIR_SHARD].read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    assert clip.instruction is None

    # 2. Restart on the Reference Integrity barrier: Pair and Reference Edit are
    #    recovered from stage counts, Reference Integrity replays its receipts
    #    for free, and Instruct finally runs.
    resume_timeline: list[str] = []
    resume_runners: list[str] = []
    resumed, _resumed_storages, resume_handles, _ledger = (
        _compose_with_reference_integrity(
            tmp_path,
            monkeypatch,
            resume_timeline,
            fixture=fixture,
            ledger=ledger,
            factory_log=resume_runners,
        )
    )
    # Pair and Reference Edit are not replayed: only the Reference Integrity
    # runner is entered, and its receipts make the replay free.
    assert resume_runners == ["reference_integrity"], resume_runners
    assert resume_handles["boogu"].calls == 0
    assert resume_handles["qwen"].completion_reviews == 0
    assert resume_handles["qwen"].removal_reviews == 0
    assert resume_handles["qwen"].pair_decisions == 0
    assert resume_handles["sam"].calls == 0
    assert resumed["pair_completed"] is True
    assert resumed["reference_edit_completed"] is True
    assert resumed["reference_edit_job_count"] == 0
    assert resume_handles["qwen"].integrity_reviews == 0, "receipts are reused"
    assert resumed["reference_integrity_completed"] is True
    assert resumed["instruct_completed"] is True
    instruction = storages[PAIR_SHARD].read_clip("clip-1").instruction
    assert instruction is not None and instruction.status == "ready"
    assert (composition_root / f"{INSTRUCT_STARTED}.json").is_file()

    # 3. Restart on the Instruct barrier: only the cheap deterministic stage
    #    reruns, and a ready instruction is the legacy skipped_existing.
    final_timeline: list[str] = []
    final_runners: list[str] = []
    final, _storages, final_handles, _ledger = _compose_with_reference_integrity(
        tmp_path,
        monkeypatch,
        final_timeline,
        fixture=fixture,
        ledger=ledger,
        factory_log=final_runners,
    )
    # The most downstream barrier short-circuits everything upstream.
    assert final_runners == [], final_runners
    assert final_handles["boogu"].calls == 0
    assert final_handles["qwen"].completion_reviews == 0
    assert final_handles["qwen"].removal_reviews == 0
    assert final_handles["qwen"].pair_decisions == 0
    assert final_handles["qwen"].integrity_reviews == 0
    assert final_handles["sam"].calls == 0
    assert final["pair_completed"] is True
    assert final["reference_edit_completed"] is True
    assert final["reference_edit_job_count"] == 0
    assert final["reference_integrity_completed"] is True
    assert final["reference_integrity_job_count"] == 0
    assert final["instruct_completed"] is True
    assert final["instruct_stats"][PAIR_SHARD]["skipped_existing"] == 1
    assert final["instruct_stats"][PAIR_SHARD]["processed"] == 0
    after = storages[PAIR_SHARD].read_clip("clip-1").instruction
    assert after is not None
    assert result_instruction_unchanged(after, instruction)
    assert final["completed"] is False
    assert final["reason"] == DOWNSTREAM_REASON


def result_instruction_unchanged(current: Any, previous: Any) -> bool:
    return current.model_dump(mode="json") == previous.model_dump(mode="json")


def _production_reference_integrity_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    qwen: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any, Any]:
    """Run the real ``build_removal_epoch_runner`` with Reference Integrity on.

    This is the production ``--job-runner`` wiring path: the same shared
    resource session, the same shard locks and the same stage dispatch, with the
    three resource epochs replaced by fakes. It exists because the composition
    helper being correct is not the same thing as production passing the
    Reference Integrity factories to it.
    """
    from r2v_data_v2.v3.post_mask_epoch_groups import resource_epoch_root
    from r2v_data_v2.v3.post_mask_epoch_removal import (
        PreparedRemovalShard,
        build_qwen_epoch_config,
        build_removal_campaign,
        build_removal_epoch_runner,
        removal_shard_paths,
    )
    from r2v_data_v2.v3.post_mask_epoch_resources import ResourceEpochManager
    from tests.test_v3_post_mask_epoch_removal import (
        _fake_process_manager,
        _group_with,
        _parts_root,
        _pending_storage,
    )

    _with_instruction_template(monkeypatch)
    config = _enable_reference_integrity(_config(tmp_path, monkeypatch, "run-ri"))
    storage = _pending_storage(config, clip_uids=("clip-1",))
    handle = qwen if qwen is not None else _CompositionQwenHandle()
    boogu = _CompositionBooguHandle()
    sam = _CompositionSamHandle()
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"
    entity_mask_root = _parts_root(tmp_path, (SHARD,))

    def prepare(*args: Any, **kwargs: Any) -> Any:
        return PreparedRemovalShard(
            shard=kwargs["shard"],
            paths=removal_shard_paths(
                post_mask_root=kwargs["post_mask_root"],
                entity_mask_root=kwargs["entity_mask_root"],
                shard=kwargs["shard"],
            ),
            storage=storage,
            clip_uids=("clip-1",),
            ready=1,
            excluded=0,
            corrupt=0,
        )

    def factories(config: Any, runner: Any, **kwargs: Any) -> dict[str, Any]:
        dispatch = kwargs["job_runner"]
        return {
            RESOURCE_BOOGU: lambda: (
                _TrackedResource(RESOURCE_BOOGU, []),
                _PassthroughExecutor(dispatch, boogu),
            ),
            RESOURCE_QWEN: lambda: (
                _TrackedResource(RESOURCE_QWEN, []),
                _PassthroughExecutor(dispatch, handle),
            ),
            RESOURCE_SAM: lambda: (
                _TrackedResource(RESOURCE_SAM, []),
                _PassthroughExecutor(dispatch, sam),
            ),
        }

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.prepare_shard_storage", prepare
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_removal_epoch_factories",
        factories,
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.ResourceEpochManager",
        ResourceEpochManager,
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(
        SHARD, campaign=build_removal_campaign(config, entity_mask_root=entity_mask_root)
    )
    ledger = GroupLedger(resource_epoch_root(post_mask_root) / group.group_id)
    events: list[dict[str, Any]] = []
    outcome = runner_callable(
        group,
        ledger,
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    return outcome, events, handle, storage, ledger


def test_production_runner_wires_reference_integrity_and_instruct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real production runner must reach Reference Integrity and Instruct."""
    outcome, events, handle, storage, ledger = _production_reference_integrity_outcome(
        tmp_path, monkeypatch
    )

    assert outcome["remove_completed"] is True
    assert outcome["pair_completed"] is True, outcome.get("pair_primary_unresolved")
    assert outcome["reference_edit_completed"] is True, outcome.get(
        "reference_edit_reconcile_error"
    )
    assert outcome["reference_integrity_completed"] is True, outcome.get(
        "reference_integrity_reconcile_error"
    )
    assert outcome["reference_integrity_unresolved"] == ()
    assert outcome["instruct_completed"] is True
    assert outcome["completed"] is False
    assert outcome["reason"] == DOWNSTREAM_REASON

    # The shared Qwen session really paid the integrity review in production.
    assert handle.integrity_reviews >= 1
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    assert clip.instruction is not None
    assert clip.instruction.status == "ready"

    composition_root = Path(ledger.root) / "composition"
    assert (composition_root / f"{REFERENCE_INTEGRITY_STARTED}.json").is_file()
    assert (composition_root / f"{INSTRUCT_STARTED}.json").is_file()

    finished = [
        event for event in events if event["event"] == "post_mask_removal_epoch_finished"
    ]
    assert finished
    assert finished[-1]["unresolved"] == 0
    assert finished[-1]["remove_completed"] is True


def test_production_wrapper_counts_reference_integrity_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved Reference Integrity job must reach the production counters."""
    outcome, events, _handle, storage, _ledger = (
        _production_reference_integrity_outcome(
            tmp_path, monkeypatch, qwen=_IntegrityFailingQwenHandle()
        )
    )

    assert outcome["remove_completed"] is True
    assert outcome["pair_completed"] is True
    assert outcome["reference_edit_completed"] is True
    assert outcome["reference_integrity_completed"] is False
    assert outcome["reference_integrity_unresolved"], "the review never committed"
    assert outcome["instruct_completed"] is False
    assert outcome["reason"] == "reference integrity resource epoch incomplete"
    assert storage.read_clip("clip-1").instruction is None

    finished = [
        event for event in events if event["event"] == "post_mask_removal_epoch_finished"
    ]
    assert finished
    assert finished[-1]["unresolved"] == len(
        outcome["reference_integrity_unresolved"]
    )
    assert finished[-1]["unresolved"] > 0, "the Reference Integrity job must count"


def test_shared_qwen_identities_include_reference_integrity_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one cached Qwen session must also serve the Reference Integrity judge."""
    config = _enable_reference_integrity(_config(tmp_path, monkeypatch, "run-shared"))
    served = str(config.qwen.background_remove_judge.model)
    assert served in shared_qwen_model_identities(config)

    mismatched = replace(
        config,
        qwen=replace(
            config.qwen,
            reference_integrity_judge=QwenServiceConfig(model="/models/other"),
        ),
    )
    assert "/models/other" in shared_qwen_model_identities(mismatched)
    with pytest.raises(StageDispatchError, match="shared Qwen session"):
        validate_shared_qwen_models(mismatched, expected_served_model_id=served)

    # The instruction writer is deliberately not part of the shared session:
    # production Instruct is deterministic with client=None.
    writer_mismatch = replace(
        config,
        qwen=replace(
            config.qwen, instruction_writer=QwenServiceConfig(model="/models/other")
        ),
    )
    assert "/models/other" not in shared_qwen_model_identities(writer_mismatch)
