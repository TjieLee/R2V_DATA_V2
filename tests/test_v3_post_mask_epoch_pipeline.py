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
    default_pair_runner_factory,
    run_removal_pair_epochs,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    JobExecution,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from tests.test_v3_pair import _Judge as _EntityJudge
from tests.test_v3_pair import _storage
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
