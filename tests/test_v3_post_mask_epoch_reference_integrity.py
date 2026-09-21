"""Reference Integrity resource epoch (6b): durable CPU foundation.

The legacy authority is ``reference_integrity_clips()``: every CPU-only fixture
here proves the epoch reproduces it byte for byte, and the restart/corruption
tests pin the durable contract. 6b has no model job, so nothing in this file
may ever reach a Qwen judge.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.storage as storage_module
import tests.test_v3_reference_integrity as legacy_integrity
from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
    ReferenceIntegrityDurableError,
    ReferenceIntegrityEpochError,
    ReferenceIntegrityEpochRunner,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.reference_integrity import reference_integrity_clips

SHARD = "shard-000000000-000000000"
CLEAN_OBJECT_PHRASE = "a black camera"
HARD_REJECT_PHRASE = "a light-colored dog with long fur"


class _ExplodingIntegrityJudge:
    """Any call proves 6b leaked into a model path."""

    def __init__(self) -> None:
        self.calls = 0

    def review(self, **kwargs: Any) -> Any:
        del kwargs
        self.calls += 1
        raise AssertionError("no Reference Integrity model call is expected")


class _ExplodingBboxJudge:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, **kwargs: Any) -> Any:
        del kwargs
        self.calls += 1
        raise AssertionError("no source bbox fallback call is expected")


def _config_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str) -> Any:
    config = legacy_integrity._config(tmp_path, monkeypatch)
    return replace(config, run_root=config.run_root.parent / run_name)


def _storage_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Build the legacy fixture on its own run root.

    ``_storage_with_ready_pair`` builds its config internally, so the module
    lookup is rebound to the per-test run root instead of duplicating 100 lines
    of fixture setup.
    """
    config = _config_for(tmp_path, monkeypatch, run_name)
    monkeypatch.setattr(legacy_integrity, "_config", lambda *args, **kw: config)
    storage = legacy_integrity._storage_with_ready_pair(
        tmp_path, monkeypatch, **kwargs
    )
    return config, storage


def _runner(config: Any, storage: Any, tmp_path: Path) -> Any:
    return ReferenceIntegrityEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )


def _edit_dump(clip: Any) -> Any:
    return (
        clip.reference_edit.model_dump(mode="json")
        if clip.reference_edit is not None
        else None
    )


def _assert_same_publication(epoch_clip: Any, legacy_clip: Any) -> None:
    assert epoch_clip.references.model_dump(mode="json") == (
        legacy_clip.references.model_dump(mode="json")
    )
    assert epoch_clip.pairing.model_dump(mode="json") == (
        legacy_clip.pairing.model_dump(mode="json")
    )
    assert _edit_dump(epoch_clip) == _edit_dump(legacy_clip)
    assert epoch_clip.reference_integrity.model_dump(mode="json") == (
        legacy_clip.reference_integrity.model_dump(mode="json")
    )
    assert epoch_clip.instruction == legacy_clip.instruction
    assert epoch_clip.export.model_dump(mode="json") == (
        legacy_clip.export.model_dump(mode="json")
    )


def _legacy_run(config: Any, storage: Any) -> tuple[Any, Any, Any]:
    judge = _ExplodingIntegrityJudge()
    bbox_judge = _ExplodingBboxJudge()
    stats = reference_integrity_clips(
        config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )
    assert judge.calls == 0
    assert bbox_judge.calls == 0
    return stats, judge, bbox_judge


def test_semantic_hard_reject_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic hard reject publishes exactly like legacy."""
    legacy_config, legacy_storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-legacy",
        second_scope="full",
        second_phrase=HARD_REJECT_PHRASE,
    )
    legacy_stats, _judge, _bbox_judge = _legacy_run(legacy_config, legacy_storage)

    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=HARD_REJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    assert runner.seed_jobs() == []
    stats = runner.reconcile_stats(SHARD)

    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.semantic_policy_rejected == 1
    assert stats.entities_rejected == 1
    assert stats.entities_skipped_review == 1
    assert stats.entities_reviewed == 0
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    assert epoch_clip.references.entities[1].status == "rejected"
    # The rejected entity keeps its exact legacy policy provenance.
    result = epoch_clip.reference_integrity.entities[1]
    assert result.status == "rejected"
    assert result.reviewed is False
    assert result.review is None
    assert result.semantic_policy_reason == "semantic_policy:living_creature_object"
    assert result.reason == "semantic_policy:living_creature_object"


@pytest.mark.parametrize(
    ("second_scope", "second_phrase"),
    (
        ("full", CLEAN_OBJECT_PHRASE),
        ("full", "a cooked lobster dish"),
    ),
)
def test_clean_reference_without_review_matches_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_scope: str,
    second_phrase: str,
) -> None:
    """Clean full references are skipped, never renamed to accepted."""
    legacy_config, legacy_storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-legacy",
        second_scope=second_scope,
        second_phrase=second_phrase,
    )
    legacy_stats, _judge, _bbox_judge = _legacy_run(legacy_config, legacy_storage)

    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope=second_scope,
        second_phrase=second_phrase,
    )
    runner = _runner(config, storage, tmp_path)
    assert runner.seed_jobs() == []
    stats = runner.reconcile_stats(SHARD)

    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.entities_skipped_review == 2
    assert stats.entities_accepted == 2
    assert stats.entities_reviewed == 0
    assert stats.entities_rejected == 0
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    statuses = [item.status for item in epoch_clip.reference_integrity.entities]
    assert statuses == ["skipped", "skipped"]
    assert [item.reason for item in epoch_clip.reference_integrity.entities] == [
        "clean_real_full_reference",
        "clean_real_full_reference",
    ]


def test_publication_before_marker_crash_backfills_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clip published, marker missing: verify, back-fill, zero model calls."""
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    assert runner.seed_jobs() == []
    expected_stats = runner.reconcile_stats(SHARD).to_dict()
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    assert marker_path.is_file()
    marker_path.unlink()
    failures_path = Path(storage.root) / "failures.jsonl"
    before_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""

    fresh = _runner(config, storage, tmp_path)
    assert fresh.seed_jobs() == []
    assert marker_path.is_file()
    assert fresh.reconcile_stats(SHARD).to_dict() == expected_stats
    after_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    assert after_failures == before_failures


def test_live_publication_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered published state must fail closed, never become a clip failure."""
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    runner.seed_jobs()
    runner.reconcile_stats(SHARD)

    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    tampered_entities = [
        item.model_copy(update={"reason": "tampered"}) if index == 0 else item
        for index, item in enumerate(clip.reference_integrity.entities)
    ]
    storage_module.write_json_atomic(
        storage.clip_path("clip-1"),
        clip.model_copy(
            update={
                "reference_integrity": clip.reference_integrity.model_copy(
                    update={"entities": tampered_entities}
                )
            }
        ).model_dump(mode="json"),
    )

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError, match="does not match"):
        fresh.seed_jobs()


@pytest.mark.parametrize(
    "tamper",
    (
        "reference_artifact",
        "entity_marker_json",
        "entity_marker_schema",
        "clip_marker_not_object",
    ),
)
def test_durable_corruption_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Artifact drift and malformed markers are durable errors, not clip failures."""
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    runner.seed_jobs()
    runner.reconcile_stats(SHARD)
    failures_path = Path(storage.root) / "failures.jsonl"
    before_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""

    if tamper == "reference_artifact":
        # Fragmented alpha changes the recomputed topology diagnostics, which is
        # exactly what the frozen publication is re-derived from.
        alpha = np.zeros((64, 64), dtype=np.uint8)
        for offset in (4, 16, 28, 40, 52):
            alpha[offset : offset + 6, offset : offset + 6] = 255
        rgba = np.zeros((64, 64, 4), dtype=np.uint8)
        rgba[..., :3] = (30, 90, 140)
        rgba[..., 3] = alpha
        Image.fromarray(rgba).save(storage.selected_path("clip-1", "e1.png"))
    elif tamper == "entity_marker_json":
        runner._entity_outcome_path(SHARD, "clip-1", "e1").write_text(
            "{broken json", encoding="utf-8"
        )
    elif tamper == "entity_marker_schema":
        runner._entity_outcome_path(SHARD, "clip-1", "e1").write_text(
            '{"schema": "tampered", "clip_uid": "clip-1", "entity_id": "e1"}',
            encoding="utf-8",
        )
    else:
        runner._clip_outcome_path(SHARD, "clip-1").write_text("[]", encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError):
        fresh.seed_jobs()
    after_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    assert after_failures == before_failures
    # The clip keeps its verified publication, never a semantic failure.
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"


def test_pre_existing_failed_state_is_reprocessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing failed RI state is a fresh target, not skipped_existing.

    Legacy re-runs such a clip; treating the old failure as this epoch's
    publication would silently skip the stage.
    """
    legacy_config, legacy_storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-legacy",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    legacy_storage.write_reference_integrity_failure("clip-1", "old failure")
    legacy_stats, _judge, _bbox_judge = _legacy_run(legacy_config, legacy_storage)
    assert legacy_stats.processed == 1
    assert legacy_stats.skipped_existing == 0
    assert legacy_storage.read_clip("clip-1").reference_integrity.status == "ready"

    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    storage.write_reference_integrity_failure("clip-1", "old failure")
    runner = _runner(config, storage, tmp_path)
    assert runner.seed_jobs() == []
    stats = runner.reconcile_stats(SHARD)

    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.processed == 1
    assert stats.skipped_existing == 0
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_integrity is not None
    assert epoch_clip.reference_integrity.status == "ready"
    _assert_same_publication(epoch_clip, legacy_clip)


def test_entity_delta_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema-valid but wrong entity counter must never reach the stats."""
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    runner.seed_jobs()
    runner.reconcile_stats(SHARD)

    marker_path = runner._entity_outcome_path(SHARD, "clip-1", "e1")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["delta"]["entities_accepted"] = 100
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError, match="entity outcome drifted"):
        fresh.reconcile_stats(SHARD)


@pytest.mark.parametrize(
    ("tamper", "entrypoint"),
    (
        ("delete_clip_entry", "seed"),
        ("delete_clip_entry", "reconcile"),
        ("classification_switch", "seed"),
        ("drop_retained_entity", "seed"),
    ),
)
def test_plan_scope_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    entrypoint: str,
) -> None:
    """A tampered plan may not silently change what this stage executes.

    The seed path and the reconcile path must agree about a valid plan, so the
    scope tamper is checked through both entry points.
    """
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)
    runner._plan(SHARD)  # freeze the plan, exactly like the first seed does
    plan_path = runner._plan_path(SHARD)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if tamper == "delete_clip_entry":
        plan["clips"].pop("clip-1")
    elif tamper == "classification_switch":
        plan["clips"]["clip-1"]["classification"] = "existing"
    else:
        plan["clips"]["clip-1"]["retained_entity_ids"] = ["e1"]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError):
        if entrypoint == "seed":
            fresh.seed_jobs()
        else:
            fresh.reconcile_stats(SHARD)


def test_failed_publication_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed publication only replaces RI; any other drift fails closed."""
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    runner = _runner(config, storage, tmp_path)

    def exploding_publish(shard: str, stor: Any, clip_uid: str) -> None:
        raise ValueError("publication exploded")

    runner._publish_clip_if_terminal = exploding_publish  # type: ignore[method-assign]
    assert runner.seed_jobs() == []
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["terminal"] == "failed"
    assert marker["reason"] == "publication exploded"
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "failed"
    assert clip.reference_integrity.reason == "publication exploded"
    failures_path = Path(storage.root) / "failures.jsonl"
    records = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["reason"] == "publication exploded"
    assert records[0]["details"]["exception_type"] == "ValueError"

    storage_module.write_json_atomic(
        storage.clip_path("clip-1"),
        clip.model_copy(
            update={"export": clip.export.model_copy(update={"accepted": False})}
        ).model_dump(mode="json"),
    )

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(
        ReferenceIntegrityDurableError, match="failed publication drifted"
    ):
        fresh.seed_jobs()
    after = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(after) == 1


def test_pre_existing_failed_same_reason_can_terminal_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run that republishes the SAME failed state is a legal no-op.

    The frozen baseline already carries failed("publication exploded"), and this
    epoch's own CPU failure produces the identical state, so the storage write
    is a no-op and the live state still equals the baseline while a legitimate
    failed clip outcome marker exists. That is terminal, not corruption.
    """
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        "run-epoch",
        second_scope="full",
        second_phrase=CLEAN_OBJECT_PHRASE,
    )
    storage.write_reference_integrity_failure("clip-1", "publication exploded")
    runner = _runner(config, storage, tmp_path)
    runner._plan(SHARD)  # freeze the plan over the pre-existing failed state

    def exploding_publish(shard: str, stor: Any, clip_uid: str) -> None:
        raise ValueError("publication exploded")

    runner._publish_clip_if_terminal = exploding_publish  # type: ignore[method-assign]
    assert runner.seed_jobs() == []

    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "failed"
    assert clip.reference_integrity.reason == "publication exploded"
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["terminal"] == "failed"
    assert marker["reason"] == "publication exploded"
    # The clip delta also carries the counters of the two entities that reached
    # their CPU branch before the publication failed, exactly like legacy.
    assert marker["delta"] == {
        "processed": 1,
        "failed": 1,
        "topology_suspicious": 0,
        "entities_skipped_review": 2,
        "entities_accepted": 2,
    }
    failures_path = Path(storage.root) / "failures.jsonl"
    records = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1

    fresh = _runner(config, storage, tmp_path)
    assert fresh.seed_jobs() == []
    stats = fresh.reconcile_stats(SHARD)
    assert stats.processed == 1
    assert stats.failed == 1
    after = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(after) == 1, "restart must not append another failure record"


# ---------------------------------------------------------------------------
# 6c: main integrity review epoch
# ---------------------------------------------------------------------------


class _SerialQwenExecutor:
    """Serial executor for the real scheduler; one resource, no real model."""

    def __init__(self, runner: ReferenceIntegrityEpochRunner, handle: Any) -> None:
        self.runner = runner
        self.handle = handle
        self.batches = 0

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

        self.batches += 1
        outcomes: dict[str, Any] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner.run(job, self.handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


def _epoch_scheduler(runner: Any, executor: Any, finalize: Any = None) -> Any:
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler

    return ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=finalize or runner.finalize,
        executors={RESOURCE_QWEN: executor},
    )


def _run_epoch(
    config: Any,
    storage: Any,
    tmp_path: Path,
    judge: Any,
    *,
    scheduler_finalize: Any = None,
) -> tuple[Any, Any, Any]:
    """Seed + drain the epoch through the real scheduler."""
    runner = _runner(config, storage, tmp_path)
    executor = _SerialQwenExecutor(runner, judge)
    scheduler = _epoch_scheduler(runner, executor, scheduler_finalize)
    jobs = runner.seed_jobs()
    if jobs:
        scheduler.run(jobs)
    return runner, executor, jobs


def _main_review_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    *,
    second_scope: str = "local",
    second_synthetic: bool = False,
    second_phrase: str = CLEAN_OBJECT_PHRASE,
    completion_state: bool = False,
) -> tuple[Any, Any]:
    config, storage = _storage_variant(
        tmp_path,
        monkeypatch,
        run_name,
        second_scope=second_scope,
        second_synthetic=second_synthetic,
        second_phrase=second_phrase,
    )
    if completion_state:
        legacy_integrity._write_completion_reference_edit_state(storage)
    return config, storage


def _fake_main_judge(*results: Any) -> Any:
    return legacy_integrity.FakeIntegrityJudge(list(results))


def test_main_review_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A main accept terminalizes the entity and publishes exactly like legacy."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_judge = _fake_main_judge(judge_result)
    legacy_stats = legacy_integrity.reference_integrity_clips(
        legacy_config,
        legacy_storage,
        judge=legacy_judge,
        bbox_fallback_judge=legacy_integrity.FakeSourceBboxFallbackJudge([]),
    )
    assert len(legacy_judge.calls) == 1

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(judge_result)
    runner, _executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1
    assert jobs[0].job_type == "reference_integrity_review"
    assert jobs[0].resource == "qwen"
    assert dict(jobs[0].target) == {"entity_id": "e2", "variant": "final"}
    assert len(judge.calls) == 1, "exactly one main review call"
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.entities_reviewed == 1
    assert stats.entities_accepted == 2
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["reviewed"] is True
    assert marker["status"] == "accepted"
    assert marker["judge_failed"] is False
    assert marker["review_job_id"] == jobs[0].job_id()
    assert marker["delta"]["entities_reviewed"] == 1


def test_main_review_reject_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A main reject without alpha/bbox routes is terminal in 6c."""
    judge_result = legacy_integrity._primary_identity_loss_review()
    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_judge = _fake_main_judge(judge_result)
    legacy_stats = legacy_integrity.reference_integrity_clips(
        legacy_config,
        legacy_storage,
        judge=legacy_judge,
        bbox_fallback_judge=legacy_integrity.FakeSourceBboxFallbackJudge([]),
    )
    assert len(legacy_judge.calls) == 1
    assert legacy_storage.read_clip("clip-1").references.entities[1].status == (
        "rejected"
    )

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(judge_result)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(judge.calls) == 1
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.entities_rejected == 1
    assert stats.entities_reviewed == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["status"] == "rejected"
    assert marker["reason"] == judge_result.reason
    assert marker["delta"]["entities_rejected"] == 1


def test_main_review_judge_failure_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge failure is a legacy semantic rejection, not a retry."""
    failure = legacy_integrity.ReferenceIntegrityJudgeFailure(
        "judge exploded", raw_responses=("raw",)
    )
    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_judge = _fake_main_judge(failure)
    legacy_stats = legacy_integrity.reference_integrity_clips(
        legacy_config,
        legacy_storage,
        judge=legacy_judge,
        bbox_fallback_judge=legacy_integrity.FakeSourceBboxFallbackJudge([]),
    )
    assert len(legacy_judge.calls) == 1

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(failure)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(judge.calls) == 1
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.judge_failed == 1
    assert stats.entities_reviewed == 1
    assert stats.entities_rejected == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["judge_failed"] is True
    assert marker["review"] is None
    assert marker["reason"] == "integrity_judge_failed:judge exploded"
    assert marker["delta"]["judge_failed"] == 1


def test_committed_review_receipt_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt committed before the CPU finalizer replays without new calls."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats = legacy_integrity.reference_integrity_clips(
        legacy_config,
        legacy_storage,
        judge=_fake_main_judge(judge_result),
        bbox_fallback_judge=legacy_integrity.FakeSourceBboxFallbackJudge([]),
    )

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(judge_result)
    # Simulate a crash between the receipt commit and the CPU finalizer.
    runner, executor, _jobs = _run_epoch(
        config, storage, tmp_path, judge, scheduler_finalize=lambda job, result: ()
    )
    assert executor.batches == 1
    assert len(judge.calls) == 1
    assert runner._entity_outcome(SHARD, "clip-1", "e2") is None
    from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
    from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler

    fresh = _runner(config, storage, tmp_path)
    fresh_executor = _SerialQwenExecutor(fresh, judge)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={RESOURCE_QWEN: fresh_executor},
    )
    fresh_jobs = fresh.seed_jobs()
    assert len(fresh_jobs) == 1, "the same frozen job is re-seeded"
    fresh_scheduler.run(fresh_jobs)

    assert fresh_executor.batches == 0, "no executor call for a committed receipt"
    assert len(judge.calls) == 1, "restart must not pay another Qwen call"
    stats = fresh.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    _assert_same_publication(clip, legacy_storage.read_clip("clip-1"))


@pytest.mark.parametrize("continuation", ("source_alpha", "artifact_bbox"))
def test_6d_continuations_stay_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, continuation: str
) -> None:
    """A main review that legacy continues into 6d keeps only its receipt."""
    if continuation == "source_alpha":
        config, storage = _main_review_fixture(
            tmp_path,
            monkeypatch,
            "run-epoch",
            second_scope="full",
            second_synthetic=True,
            completion_state=True,
        )
        judge_result = legacy_integrity._severe_reference_artifact_review()
    else:
        config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
        judge_result = legacy_integrity._transient_removal_artifact_review()
    judge = _fake_main_judge(judge_result)
    runner, _executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1
    assert len(judge.calls) == 1, "6c must not run any continuation model call"
    assert runner._entity_outcome(SHARD, "clip-1", "e2") is None
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is None
    with pytest.raises(ReferenceIntegrityEpochError, match="not terminal"):
        runner.reconcile_stats(SHARD)
    # 6d owns the continuation: no bbox judge exists in this epoch at all.
    assert not hasattr(runner, "bbox_fallback_judge")


def test_reviewed_marker_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered reviewed marker must fail closed against its receipt."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(judge_result)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    path = runner._entity_outcome_path(SHARD, "clip-1", "e2")
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["delta"]["entities_accepted"] = 100
    path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError, match="reviewed entity outcome"):
        fresh.reconcile_stats(SHARD)
