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
from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_BOOGU, RESOURCE_QWEN
from r2v_data_v2.v3.post_mask_epoch_pipeline import (
    DOWNSTREAM_REASON,
    StageDispatchError,
    default_pair_runner_factory,
    run_removal_pair_epochs,
    run_removal_pair_resource_session,
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
