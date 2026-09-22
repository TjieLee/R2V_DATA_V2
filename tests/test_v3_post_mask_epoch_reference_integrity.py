"""Reference Integrity resource epoch (6b): durable CPU foundation.

The legacy authority is ``reference_integrity_clips()``: every CPU-only fixture
here proves the epoch reproduces it byte for byte, and the restart/corruption
tests pin the durable contract. 6b has no model job, so nothing in this file
may ever reach a Qwen judge.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.storage as storage_module
import tests.test_v3_reference_integrity as legacy_integrity
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_QWEN,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
    REFERENCE_INTEGRITY_BBOX_REVIEW_JOB,
    REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
    ReferenceIntegrityDurableError,
    ReferenceIntegrityEpochError,
    ReferenceIntegrityEpochRunner,
    _FileSignature,
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


class _FakeEpochJudge:
    """One handle serving both review kinds, dispatching on the call shape.

    The epoch resolves a single job-type blind handle, so a chain that runs an
    integrity review and then a bbox review needs one object that behaves as
    either judge.
    """

    def __init__(self, integrity_results: Any, bbox_results: Any) -> None:
        self.integrity = legacy_integrity.FakeIntegrityJudge(list(integrity_results))
        self.bbox = legacy_integrity.FakeSourceBboxFallbackJudge(list(bbox_results))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.integrity.calls

    @property
    def bbox_calls(self) -> list[dict[str, Any]]:
        return self.bbox.calls

    def review(self, **kwargs: Any) -> Any:
        if "source_bbox_candidate" in kwargs:
            return self.bbox.review(**kwargs)
        return self.integrity.review(**kwargs)


def _bounded_handle(integrity_results: Any, bbox_results: Any) -> Any:
    return _FakeEpochJudge(integrity_results, bbox_results)


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


@pytest.mark.parametrize("variant", ("final", "source_alpha"))
def test_reviewed_marker_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """A tampered reviewed marker must fail closed against its receipts."""
    if variant == "final":
        config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
        results: tuple[Any, ...] = (
            legacy_integrity._review(accept=True, reason="usable reference"),
        )
    else:
        config, storage = _alpha_continuation_fixture(
            tmp_path, monkeypatch, "run-epoch"
        )
        results = (
            legacy_integrity._severe_reference_artifact_review(),
            legacy_integrity._review(accept=True, reason="alpha reference is clean"),
        )
    judge = _fake_main_judge(*results)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    path = runner._entity_outcome_path(SHARD, "clip-1", "e2")
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["delta"]["entities_accepted"] = 100
    path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(ReferenceIntegrityDurableError, match="reviewed entity outcome"):
        fresh.reconcile_stats(SHARD)


# ---------------------------------------------------------------------------
# 6d1: source alpha continuation
# ---------------------------------------------------------------------------


def _frozen_alpha_edit(runner: Any, entity_id: str = "e2") -> Any:
    """The frozen pre-edit entity whose source alpha 6d1 must use."""
    plan = json.loads(runner._plan_path(SHARD).read_text(encoding="utf-8"))
    entities = plan["clips"]["clip-1"]["pre_reference_edit"]["entities"]
    return next(item for item in entities if item["entity_id"] == entity_id)


def _alpha_continuation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str
) -> tuple[Any, Any]:
    return _main_review_fixture(
        tmp_path,
        monkeypatch,
        run_name,
        second_scope="full",
        second_synthetic=True,
        completion_state=True,
    )


def _legacy_alpha_run(
    config: Any, storage: Any, *judge_results: Any
) -> tuple[Any, Any, Any]:
    judge = _fake_main_judge(*judge_results)
    bbox_judge = legacy_integrity.FakeSourceBboxFallbackJudge([])
    stats = legacy_integrity.reference_integrity_clips(
        config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )
    return stats, judge, bbox_judge


def test_source_alpha_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted source alpha publishes the alpha reference exactly like legacy."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._review(accept=True, reason="alpha reference is clean"),
    )
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, legacy_bbox = _legacy_alpha_run(
        legacy_config, legacy_storage, *results
    )
    assert [call["synthetic"] for call in legacy_judge.calls] == [True, False]
    assert legacy_bbox.calls == []

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(*results)
    runner, _executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1, "only the main review is a root job"
    assert dict(jobs[0].target) == {"entity_id": "e2", "variant": "final"}
    assert [call["synthetic"] for call in judge.calls] == [True, False]
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.entities_reviewed == 2
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    frozen = _frozen_alpha_edit(runner)
    final_reference = epoch_clip.references.entities[1]
    assert final_reference.synthetic is False
    assert final_reference.image_path == frozen["source_reference"]["image_path"]
    final_edit = epoch_clip.reference_edit.entities[1]
    assert final_edit.default_variant == "alpha"
    assert final_edit.fallback_policy == "keep_source"
    assert final_edit.reason == "completion_integrity_rejected_fallback_to_alpha"

    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["review_variant"] == "source_alpha"
    assert marker["status"] == "accepted"
    assert marker["reason"] == "completion_integrity_rejected_source_alpha_accepted"
    assert marker["delta"]["entities_reviewed"] == 2
    assert marker["delta"]["entities_accepted"] == 1
    assert marker["judge_failed"] is False
    # The alpha job is conditional on the exact main receipt.
    alpha_job_id = marker["review_job_id"]
    main_marker_absent = (
        runner._entity_outcome_path(SHARD, "clip-1", "e2").is_file()
    )
    assert main_marker_absent
    plan = json.loads(runner._plan_path(SHARD).read_text(encoding="utf-8"))
    assert plan["clips"]["clip-1"]["classification"] == "fresh_target"
    assert alpha_job_id != jobs[0].job_id()


def test_source_alpha_judge_failure_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alpha judge failure is a legacy semantic rejection, not a retry."""
    failure = legacy_integrity.ReferenceIntegrityJudgeFailure(
        "alpha exploded", raw_responses=("raw",)
    )
    results = (legacy_integrity._severe_reference_artifact_review(), failure)
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, _legacy_bbox = _legacy_alpha_run(
        legacy_config, legacy_storage, *results
    )
    assert len(legacy_judge.calls) == 2

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(*results)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(judge.calls) == 2
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.judge_failed == 1
    assert stats.entities_reviewed == 2
    assert stats.entities_rejected == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)
    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["review_variant"] == "source_alpha"
    assert marker["judge_failed"] is True
    assert marker["review"] is None
    assert marker["status"] == "rejected"
    assert marker["reason"] == (
        "completion_rejected_then_alpha_integrity_judge_failed:alpha exploded"
    )
    assert marker["delta"]["entities_reviewed"] == 2
    assert marker["delta"]["entities_rejected"] == 1


def test_source_alpha_reject_without_bbox_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An alpha reject with no bbox route terminalizes as a rejection."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._primary_identity_loss_review(),
    )
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, legacy_bbox = _legacy_alpha_run(
        legacy_config, legacy_storage, *results
    )
    assert len(legacy_judge.calls) == 2
    assert legacy_bbox.calls == []
    assert legacy_stats.source_bbox_fallback_attempted == 0

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(*results)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(judge.calls) == 2
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    frozen = _frozen_alpha_edit(runner)
    # The entity records the alpha reference it was judged on ...
    integrity_entity = next(
        item for item in epoch_clip.reference_integrity.entities
        if item.entity_id == "e2"
    )
    assert integrity_entity.status == "rejected"
    assert integrity_entity.input_reference.model_dump(mode="json") == (
        frozen["source_reference"]
    )
    # ... while the published clip reference is the rejected pre-stage one.
    assert epoch_clip.references.entities[1].status == "rejected"
    assert epoch_clip.references.entities[1].image_path != (
        frozen["source_reference"]["image_path"]
    )
    assert epoch_clip.reference_edit.entities[1].default_variant == "accepted_base"
    marker = json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", "e2").read_text(encoding="utf-8")
    )
    assert marker["review_variant"] == "source_alpha"
    assert marker["status"] == "rejected"
    assert marker["judge_failed"] is False
    assert marker["reason"] == results[1].reason
    assert marker["delta"]["entities_rejected"] == 1


# ---------------------------------------------------------------------------
# 6d2a: artifact source bbox continuation
# ---------------------------------------------------------------------------


def _legacy_bbox_run(
    config: Any,
    storage: Any,
    *,
    integrity_results: Any,
    bbox_results: Any,
) -> tuple[Any, Any, Any]:
    judge = legacy_integrity.FakeIntegrityJudge(list(integrity_results))
    bbox_judge = legacy_integrity.FakeSourceBboxFallbackJudge(list(bbox_results))
    stats = reference_integrity_clips(
        config,
        storage,
        judge=judge,
        bbox_fallback_judge=bbox_judge,
    )
    return stats, judge, bbox_judge


def _resolve(storage: Any, relative: str) -> Path:
    from r2v_data_v2.v3.reference_integrity import _resolve_run_artifact

    return _resolve_run_artifact(storage, relative)


def _frozen_bbox_anchor(runner: Any, entity_id: str = "e2") -> Any:
    return json.loads(
        runner._bbox_input_path(SHARD, "clip-1", entity_id).read_text(encoding="utf-8")
    )


def _expected_marker(runner: Any, entity_id: str = "e2") -> Any:
    return json.loads(
        runner._entity_outcome_path(SHARD, "clip-1", entity_id).read_text(
            encoding="utf-8"
        )
    )


def test_artifact_bbox_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A main artifact reject continued into an accepted bbox fallback."""
    results = (legacy_integrity._transient_removal_artifact_review(),)
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )
    assert len(legacy_judge.calls) == 1
    assert len(legacy_bbox.calls) == 1
    assert legacy_bbox.calls[0]["trigger"] == "artifact_review_reject"

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner, executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1, "only the main review is a root job"
    assert len(judge.calls) == 1
    assert len(judge.bbox_calls) == 1, "exactly one bbox review call"
    assert judge.bbox_calls[0]["trigger"] == "artifact_review_reject"
    assert executor.batches == 2, "one main phase, one bbox phase"
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.source_bbox_fallback_attempted == 1
    assert stats.source_bbox_fallback_accepted == 1
    assert stats.entities_reviewed == 1, "the bbox judge is not an integrity review"
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    anchor = _frozen_bbox_anchor(runner)
    published = epoch_clip.references.entities[1]
    assert published.status == "ready"
    assert published.source_bbox_fallback is True
    assert published.image_path == anchor["candidate_path"]
    assert list(published.source_bbox_xyxy) == anchor["bbox_xyxy"]
    # This fixture has no Reference Edit state, exactly like the legacy side.
    assert epoch_clip.reference_edit is None

    # The legacy metadata advanced from materialized to its final state.
    metadata = json.loads(
        _resolve(storage, anchor["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["status"] == "accepted"
    assert metadata["review_status"] == "accept"
    assert metadata["trigger"] == "artifact_review_reject_v1"
    assert metadata["review"]["verdict"] == "accept"
    assert metadata["finish_reason"] == "stop"

    marker = _expected_marker(runner)
    assert marker["review_variant"] == "final"
    assert marker["parent_review_job_id"] is None
    assert marker["status"] == "accepted"
    assert marker["bbox_judge_failed"] is False
    assert marker["source_bbox_fallback_trigger"] == "artifact_review_reject"
    assert marker["source_bbox_fallback_candidate_path"] == anchor["candidate_path"]
    assert marker["reason"] == bbox_results[0].reason
    assert marker["delta"]["entities_reviewed"] == 1
    assert marker["delta"]["source_bbox_fallback_attempted"] == 1
    assert marker["delta"]["source_bbox_fallback_accepted"] == 1
    assert marker["bbox_review_job_id"] != marker["review_job_id"]


@pytest.mark.parametrize("bbox", ("reject", "judge_failure"))
def test_artifact_bbox_reject_and_judge_failure_match_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bbox: str,
) -> None:
    """A bbox reject and a bbox judge failure are both plain rejections."""
    if bbox == "reject":
        bbox_result: Any = legacy_integrity._bbox_review(accept=False)
    else:
        bbox_result = legacy_integrity.SourceBboxFallbackJudgeFailure(
            "bbox exploded", raw_response="raw"
        )
    results = (legacy_integrity._transient_removal_artifact_review(),)

    legacy_config, legacy_storage = _main_review_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, _legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=(bbox_result,),
    )
    assert len(legacy_bbox.calls) == 1

    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, (bbox_result,))
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.source_bbox_fallback_attempted == 1
    assert stats.source_bbox_fallback_rejected == 1
    assert stats.entities_rejected == 1
    assert stats.entities_reviewed == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    published = epoch_clip.references.entities[1]
    assert published.status == "rejected"
    assert published.source_bbox_fallback is False
    assert epoch_clip.reference_edit is None
    public = next(
        item
        for item in epoch_clip.reference_integrity.entities
        if item.entity_id == "e2"
    )
    assert public.status == "rejected"
    assert public.review is not None, "the parent review is the public provenance"
    marker = _expected_marker(runner)
    if bbox == "judge_failure":
        assert stats.source_bbox_fallback_judge_failed == 1
        assert stats.judge_failed == 0, "the generic counter stays untouched"
        assert marker["bbox_judge_failed"] is True
        assert marker["judge_failed"] is False
        assert marker["source_bbox_fallback_review"] is None
        assert marker["reason"] == (
            "source_bbox_fallback_judge_failed:bbox exploded"
        )
        # legacy publishes no bbox provenance on this branch
        assert public.source_bbox_fallback_trigger is None
        assert public.source_bbox_fallback_candidate_path is None
        assert public.source_bbox_fallback_metadata_path is None
        assert public.source_bbox_xyxy is None
        assert public.source_bbox_fallback_review is None
        assert public.source_bbox_fallback_judge_failed is False
        assert public.judge_failed is False
        assert public.reason == "source_bbox_fallback_judge_failed:bbox exploded"
        metadata = json.loads(
            _resolve(storage, marker["source_bbox_fallback_metadata_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert metadata["status"] == "judge_failed"
        assert metadata["review_status"] == "judge_failed"
        assert metadata["reason"] == "bbox exploded"
        assert metadata["raw_response"] == "raw"
    else:
        assert stats.source_bbox_fallback_judge_failed == 0
        assert marker["bbox_judge_failed"] is False
        assert marker["reason"] == bbox_result.reason
        assert public.source_bbox_fallback_trigger == "artifact_review_reject"
        assert public.source_bbox_fallback_review is not None
        assert public.source_bbox_fallback_review.verdict == "reject"
        metadata = json.loads(
            _resolve(storage, marker["source_bbox_fallback_metadata_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert metadata["status"] == "rejected"
        assert metadata["review_status"] == "reject"


def test_source_alpha_bbox_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three-model chain: main reject, alpha artifact reject, bbox accept."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )
    assert len(legacy_judge.calls) == 2
    assert len(legacy_bbox.calls) == 1

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner, executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1
    assert len(judge.calls) == 2
    assert len(judge.bbox_calls) == 1
    assert executor.batches == 3, "one main phase, one alpha phase, one bbox phase"
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.entities_reviewed == 2, "the bbox judge is not an integrity review"
    assert stats.source_bbox_fallback_accepted == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    anchor = _frozen_bbox_anchor(runner)
    frozen_alpha = _frozen_alpha_edit(runner)
    assert anchor["parent_variant"] == "source_alpha"
    assert anchor["current_reference_path"] == (
        frozen_alpha["source_reference"]["image_path"]
    )
    published = epoch_clip.references.entities[1]
    assert published.image_path == anchor["candidate_path"]
    assert published.source_bbox_fallback is True
    assert epoch_clip.reference_edit.entities[1].default_variant == "bbox"

    marker = _expected_marker(runner)
    assert marker["review_variant"] == "source_alpha"
    assert marker["parent_review_job_id"] is not None
    assert marker["review_job_id"] != marker["parent_review_job_id"]
    assert marker["bbox_review_job_id"] not in {
        marker["review_job_id"],
        marker["parent_review_job_id"],
    }
    assert marker["status"] == "accepted"
    assert marker["delta"]["entities_reviewed"] == 2
    assert marker["delta"]["source_bbox_fallback_accepted"] == 1


def test_source_alpha_bbox_reject_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected alpha-parent bbox rejects the frozen pre-stage reference."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=False),)
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, _legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )
    assert len(legacy_bbox.calls) == 1

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.source_bbox_fallback_rejected == 1
    assert stats.entities_rejected == 1
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    # The published reference is the rejected frozen pre-stage one, never the
    # rejected alpha reference.
    pre = json.loads(
        runner._plan_path(SHARD).read_text(encoding="utf-8")
    )["clips"]["clip-1"]["pre_references"]["entities"][1]
    published = epoch_clip.references.entities[1]
    assert published.status == "rejected"
    assert published.image_path != _frozen_alpha_edit(runner)["source_reference"][
        "image_path"
    ]
    assert epoch_clip.reference_edit.entities[1].default_variant == "accepted_base"
    assert epoch_clip.reference_edit.entities[1].fallback_policy == "not_used"
    public = next(
        item
        for item in epoch_clip.reference_integrity.entities
        if item.entity_id == "e2"
    )
    assert public.input_reference.image_path == (
        _frozen_alpha_edit(runner)["source_reference"]["image_path"]
    )
    assert public.status == "rejected"
    assert pre["entity_id"] == "e2"


def test_bbox_committed_receipt_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the bbox receipt replays the whole three-stage chain free."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, _legacy_judge, _legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner = _runner(config, storage, tmp_path)
    executor = _SerialQwenExecutor(runner, judge)

    def swallow_bbox(job: Any, result: Any) -> Any:
        # Simulate a crash between the bbox receipt and its CPU finalizer.
        if job.job_type == REFERENCE_INTEGRITY_BBOX_REVIEW_JOB:
            return ()
        return runner.finalize(job, result)

    scheduler = _epoch_scheduler(runner, executor, swallow_bbox)
    jobs = runner.seed_jobs()
    assert [dict(job.target)["variant"] for job in jobs] == ["final"]
    scheduler.run(jobs)

    assert len(judge.calls) == 2
    assert len(judge.bbox_calls) == 1
    assert runner._entity_outcome(SHARD, "clip-1", "e2") is None
    assert storage.read_clip("clip-1").reference_integrity is None

    fresh = _runner(config, storage, tmp_path)
    fresh_executor = _SerialQwenExecutor(fresh, judge)
    fresh_scheduler = _epoch_scheduler(fresh, fresh_executor)
    fresh_jobs = fresh.seed_jobs()
    # Neither continuation is a root: only each parent receipt unlocks its child.
    assert [dict(job.target)["variant"] for job in fresh_jobs] == ["final"]
    fresh_scheduler.run(fresh_jobs)

    assert fresh_executor.batches == 0, "all three committed receipts are reused"
    assert len(judge.calls) == 2, "restart must not pay another integrity call"
    assert len(judge.bbox_calls) == 1, "restart must not pay another bbox call"
    stats = fresh.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    _assert_same_publication(clip, legacy_storage.read_clip("clip-1"))


@pytest.mark.parametrize("tamper", ("candidate_drift", "base_metadata_extra_field"))
def test_frozen_bbox_artifact_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Once the bbox input is frozen, neither the candidate nor the frozen
    legacy metadata may change."""
    results = (legacy_integrity._transient_removal_artifact_review(),)
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, ())
    runner = _runner(config, storage, tmp_path)

    unlocked: list[ModelJob] = []

    def hold_bbox(job: Any, result: Any) -> Any:
        unlocked.extend(runner.finalize(job, result))
        return ()  # the bbox job never runs: no bbox receipt exists yet

    scheduler = _epoch_scheduler(runner, _SerialQwenExecutor(runner, judge), hold_bbox)
    scheduler.run(runner.seed_jobs())

    assert len(unlocked) == 1
    assert unlocked[0].job_type == REFERENCE_INTEGRITY_BBOX_REVIEW_JOB
    anchor_path = runner._bbox_input_path(SHARD, "clip-1", "e2")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    assert judge.bbox_calls == []
    if tamper == "candidate_drift":
        candidate = _resolve(storage, anchor["candidate_path"])
        assert candidate.is_file()
        candidate.write_bytes(b"tampered candidate")
        expected = "bbox candidate drifted"
    else:
        # An extra field in the frozen legacy metadata is corruption too.
        assert "epoch_fake_field" not in anchor["base_metadata"]
        anchor["base_metadata"]["epoch_fake_field"] = "bad"
        anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
        expected = "bbox base metadata drifted"

    fresh = _runner(config, storage, tmp_path)
    main_job = fresh.seed_jobs()[0]
    assert dict(main_job.target)["variant"] == "final"
    committed = fresh.ledger.load_committed_result(main_job)
    assert committed is not None
    with pytest.raises(ReferenceIntegrityDurableError, match=expected):
        fresh.finalize(main_job, committed)

    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is None, "must not become a semantic failure"
    assert judge.bbox_calls == [], "no model call is paid for corruption"
    failures_path = Path(storage.root) / "failures.jsonl"
    if failures_path.is_file():
        assert [
            line
            for line in failures_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] == []


@pytest.mark.parametrize("route", ("artifact", "topology"))
def test_finalized_bbox_metadata_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """A durable bbox entity marker implies the metadata already finalized.

    Rewinding the metadata to the frozen materialized base after the outcome is
    durable must fail closed, never be silently re-advanced on restart.
    """
    if route == "artifact":
        results = (legacy_integrity._transient_removal_artifact_review(),)
        config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-epoch")
        entity_id = "e2"
    else:
        results = (legacy_integrity._review(accept=True),)
        config, storage = _topology_upgrade_fixture(
            tmp_path, monkeypatch, "run-epoch"
        )
        entity_id = "e1"
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    judge = _FakeEpochJudge(results, bbox_results)
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    anchor = _frozen_bbox_anchor(runner, entity_id)
    metadata_path = _resolve(storage, anchor["metadata_path"])
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == "accepted"
    failures_path = Path(storage.root) / "failures.jsonl"
    before_failures = (
        failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    )

    metadata_path.write_text(json.dumps(anchor["base_metadata"]), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(
        ReferenceIntegrityDurableError,
        match="final Reference Integrity bbox metadata drifted",
    ):
        fresh.reconcile_stats(SHARD)

    after = storage.read_clip("clip-1")
    assert after.reference_integrity is not None
    assert after.reference_integrity.status == "ready"
    assert len(judge.calls) == 1, "restart pays no integrity call"
    assert len(judge.bbox_calls) == 1, "restart pays no bbox call"
    after_failures = (
        failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    )
    assert after_failures == before_failures, "no semantic failure is recorded"


def test_source_alpha_committed_receipt_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the alpha receipt replays the whole chain for free."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._review(accept=True, reason="alpha reference is clean"),
    )
    legacy_config, legacy_storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, _legacy_judge, _legacy_bbox = _legacy_alpha_run(
        legacy_config, legacy_storage, *results
    )

    config, storage = _alpha_continuation_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _fake_main_judge(*results)
    runner = _runner(config, storage, tmp_path)
    executor = _SerialQwenExecutor(runner, judge)

    def swallow_alpha(job: Any, result: Any) -> Any:
        # Simulate a crash between the alpha receipt and its CPU finalizer.
        if str(dict(job.target).get("variant")) == "source_alpha":
            return ()
        return runner.finalize(job, result)

    scheduler = _epoch_scheduler(runner, executor, swallow_alpha)
    jobs = runner.seed_jobs()
    assert [dict(job.target)["variant"] for job in jobs] == ["final"]
    scheduler.run(jobs)

    assert len(judge.calls) == 2
    assert runner._entity_outcome(SHARD, "clip-1", "e2") is None
    assert storage.read_clip("clip-1").reference_integrity is None

    fresh = _runner(config, storage, tmp_path)
    fresh_executor = _SerialQwenExecutor(fresh, judge)
    fresh_scheduler = _epoch_scheduler(fresh, fresh_executor)
    fresh_jobs = fresh.seed_jobs()
    # The alpha job is never a root: only the main receipt unlocks it.
    assert [dict(job.target)["variant"] for job in fresh_jobs] == ["final"]
    fresh_scheduler.run(fresh_jobs)

    assert fresh_executor.batches == 0, "both committed receipts are reused"
    assert len(judge.calls) == 2, "restart must not pay another Qwen call"
    stats = fresh.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    _assert_same_publication(clip, legacy_storage.read_clip("clip-1"))


# ---------------------------------------------------------------------------
# topology hole upgrade continuation
# ---------------------------------------------------------------------------


def _topology_upgrade_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str
) -> tuple[Any, Any]:
    """A human subject alpha with a real enclosed hole, built by legacy helpers.

    ``_make_human_topology_upgrade_candidate`` makes e1 local and reviewable and
    rewrites its image with an enclosed transparent hole; the helper itself
    asserts the real diagnostics (holes = 1, largest = 2392, ratio >= 0.01,
    suspicious = False). ``_write_not_required_reference_edit`` gives the clip a
    ready Reference Edit state, which the bbox fallback then mutates.
    """
    config, storage = _main_review_fixture(
        tmp_path, monkeypatch, run_name, second_scope="full"
    )
    legacy_integrity._make_human_topology_upgrade_candidate(storage)
    legacy_integrity._write_not_required_reference_edit(storage)
    return config, storage


def test_topology_bbox_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An accepted topology bbox upgrade publishes the candidate reference."""
    results = (legacy_integrity._review(accept=True),)
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    legacy_config, legacy_storage = _topology_upgrade_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )
    assert len(legacy_judge.calls) == 1
    assert len(legacy_bbox.calls) == 1
    assert legacy_bbox.calls[0]["trigger"] == "topology_alpha_hole_upgrade"

    config, storage = _topology_upgrade_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner, executor, jobs = _run_epoch(config, storage, tmp_path, judge)

    assert len(jobs) == 1, "only the main review is a root job"
    assert len(judge.calls) == 1
    assert len(judge.bbox_calls) == 1
    assert judge.bbox_calls[0]["trigger"] == "topology_alpha_hole_upgrade"
    assert executor.batches == 2, "one main phase, one bbox phase"
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.source_bbox_topology_upgrade_attempted == 1
    assert stats.source_bbox_topology_upgrade_accepted == 1
    assert stats.source_bbox_topology_upgrade_kept_original == 0
    assert stats.source_bbox_fallback_attempted == 0
    assert stats.entities_reviewed == 1, "the bbox judge is not an integrity review"
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    anchor = _frozen_bbox_anchor(runner, "e1")
    assert anchor["trigger"] == "topology_alpha_hole_upgrade"
    assert anchor["topology_evidence"] == {
        "alpha_available": True,
        "enclosed_transparent_hole_count": 1,
        "largest_enclosed_hole_area": 2392,
        "enclosed_hole_bbox_ratio": anchor["topology_evidence"][
            "enclosed_hole_bbox_ratio"
        ],
    }
    published = epoch_clip.references.entities[0]
    assert published.image_path == anchor["candidate_path"]
    assert published.source_bbox_fallback is True
    # This fixture installs a variants-less Reference Edit state, exactly like
    # the legacy topology accept test, so only these three fields move.
    edit = epoch_clip.reference_edit.entities[0]
    assert edit.status == "fallback"
    assert edit.fallback_policy == "source_bbox_fallback"
    assert edit.output_image_path == anchor["candidate_path"]

    metadata = json.loads(
        _resolve(storage, anchor["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["trigger"] == "topology_alpha_hole_upgrade_v1"
    assert metadata["status"] == "accepted"
    assert metadata["topology_evidence"] == anchor["topology_evidence"]
    assert "failed_reference_path" not in metadata
    assert "failed_reference_sha256" not in metadata

    marker = _expected_marker(runner, "e1")
    assert marker["review_variant"] == "final"
    assert marker["parent_review_job_id"] is None
    assert marker["source_bbox_fallback_trigger"] == "topology_alpha_hole_upgrade"
    assert marker["status"] == "accepted"
    assert marker["bbox_judge_failed"] is False
    assert marker["reason"] == bbox_results[0].reason
    assert marker["delta"]["entities_reviewed"] == 1
    assert marker["delta"]["source_bbox_topology_upgrade_attempted"] == 1
    assert marker["delta"]["source_bbox_topology_upgrade_accepted"] == 1
    assert marker["delta"]["entities_accepted"] == 1
    assert "source_bbox_fallback_attempted" not in marker["delta"]


@pytest.mark.parametrize("bbox", ("reject", "judge_failure"))
def test_topology_bbox_reject_and_judge_failure_keep_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bbox: str
) -> None:
    """A topology bbox reject or judge failure keeps the accepted original.

    This is where the topology trigger differs most from the artifact one: the
    entity stays accepted, so only an explicit bbox *accept* may replace the
    published reference.
    """
    if bbox == "reject":
        bbox_result: Any = legacy_integrity._bbox_review(accept=False)
    else:
        bbox_result = legacy_integrity.SourceBboxFallbackJudgeFailure(
            "malformed topology comparison"
        )
    results = (legacy_integrity._review(accept=True),)

    legacy_config, legacy_storage = _topology_upgrade_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_original = legacy_storage.read_clip("clip-1").references.entities[0]
    legacy_stats, _legacy_judge, legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=(bbox_result,),
    )
    assert len(legacy_bbox.calls) == 1

    config, storage = _topology_upgrade_fixture(tmp_path, monkeypatch, "run-epoch")
    original = storage.read_clip("clip-1").references.entities[0]
    original_bytes = _resolve(storage, str(original.image_path)).read_bytes()
    judge = _FakeEpochJudge(results, (bbox_result,))
    runner, _executor, _jobs = _run_epoch(config, storage, tmp_path, judge)

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    assert stats.source_bbox_topology_upgrade_attempted == 1
    assert stats.source_bbox_topology_upgrade_kept_original == 1
    assert stats.source_bbox_topology_upgrade_accepted == 0
    assert stats.entities_rejected == 0, "the topology upgrade never rejects"
    assert stats.entities_accepted == 2
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    _assert_same_publication(epoch_clip, legacy_clip)

    # The candidate replacement gate: the marker is an *accepted* outcome, so
    # only the bbox verdict may decide whether the reference is replaced.
    marker = _expected_marker(runner, "e1")
    assert marker["status"] == "accepted"
    assert marker["outcome"] == "accepted"
    anchor = _frozen_bbox_anchor(runner, "e1")
    published = epoch_clip.references.entities[0]
    assert published.image_path == original.image_path
    assert published.image_path != anchor["candidate_path"]
    assert published.image_path == legacy_original.image_path
    assert published.source_bbox_fallback is False
    assert _resolve(storage, str(original.image_path)).read_bytes() == original_bytes
    edit = epoch_clip.reference_edit.entities[0]
    assert edit.status == "not_required"
    assert edit.fallback_policy == "not_used"
    assert edit.default_variant is None
    public = next(
        item
        for item in epoch_clip.reference_integrity.entities
        if item.entity_id == "e1"
    )
    assert public.status == "accepted"
    assert public.final_reference_path == original.image_path
    assert public.source_bbox_fallback_trigger == "topology_alpha_hole_upgrade"
    assert public.source_bbox_fallback_candidate_path == anchor["candidate_path"]

    metadata = json.loads(
        _resolve(storage, anchor["metadata_path"]).read_text(encoding="utf-8")
    )
    if bbox == "judge_failure":
        assert stats.source_bbox_topology_upgrade_judge_failed == 1
        assert stats.source_bbox_fallback_judge_failed == 0
        assert stats.judge_failed == 0
        assert marker["bbox_judge_failed"] is True
        assert marker["judge_failed"] is False
        assert marker["source_bbox_fallback_review"] is None
        assert marker["reason"] == (
            "topology_bbox_upgrade_judge_failed_kept_original:"
            "malformed topology comparison"
        )
        # Unlike the artifact trigger, the topology failure keeps the public
        # bbox provenance and raises the public failure flag.
        assert public.source_bbox_fallback_judge_failed is True
        assert public.source_bbox_fallback_review is None
        assert public.source_bbox_fallback_metadata_path == anchor["metadata_path"]
        assert list(public.source_bbox_xyxy) == anchor["bbox_xyxy"]
        assert metadata["status"] == "judge_failed"
        assert metadata["review_status"] == "judge_failed"
        assert metadata["reason"] == "malformed topology comparison"
        assert metadata["topology_evidence"] == anchor["topology_evidence"]
    else:
        assert marker["bbox_judge_failed"] is False
        assert marker["reason"] == bbox_result.reason
        assert public.source_bbox_fallback_judge_failed is False
        assert public.source_bbox_fallback_review is not None
        assert public.source_bbox_fallback_review.verdict == "reject"
        assert metadata["status"] == "rejected"
        assert metadata["review_status"] == "reject"


def test_topology_bbox_committed_receipt_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after the topology bbox receipt replays the chain for free."""
    results = (legacy_integrity._review(accept=True),)
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    legacy_config, legacy_storage = _topology_upgrade_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_stats, _legacy_judge, _legacy_bbox = _legacy_bbox_run(
        legacy_config,
        legacy_storage,
        integrity_results=results,
        bbox_results=bbox_results,
    )

    config, storage = _topology_upgrade_fixture(tmp_path, monkeypatch, "run-epoch")
    judge = _FakeEpochJudge(results, bbox_results)
    runner = _runner(config, storage, tmp_path)
    executor = _SerialQwenExecutor(runner, judge)

    def swallow_bbox(job: Any, result: Any) -> Any:
        # Simulate a crash between the bbox receipt and its CPU finalizer.
        if job.job_type == REFERENCE_INTEGRITY_BBOX_REVIEW_JOB:
            return ()
        return runner.finalize(job, result)

    scheduler = _epoch_scheduler(runner, executor, swallow_bbox)
    jobs = runner.seed_jobs()
    assert [dict(job.target)["variant"] for job in jobs] == ["final"]
    scheduler.run(jobs)

    assert len(judge.calls) == 1
    assert len(judge.bbox_calls) == 1
    assert runner._entity_outcome(SHARD, "clip-1", "e1") is None
    assert storage.read_clip("clip-1").reference_integrity is None

    fresh = _runner(config, storage, tmp_path)
    fresh_executor = _SerialQwenExecutor(fresh, judge)
    fresh_scheduler = _epoch_scheduler(fresh, fresh_executor)
    fresh_jobs = fresh.seed_jobs()
    # The bbox job is never a root: only the main receipt unlocks it.
    assert [dict(job.target)["variant"] for job in fresh_jobs] == ["final"]
    fresh_scheduler.run(fresh_jobs)

    assert fresh_executor.batches == 0, "both committed receipts are reused"
    assert len(judge.calls) == 1, "restart pays no integrity call"
    assert len(judge.bbox_calls) == 1, "restart pays no bbox call"
    stats = fresh.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_stats.to_dict()
    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    _assert_same_publication(clip, legacy_storage.read_clip("clip-1"))


# ---------------------------------------------------------------------------
# Durable boundary: creation belongs to the main thread, never to a worker
# ---------------------------------------------------------------------------


def _guard_creation(
    monkeypatch: pytest.MonkeyPatch, runner: Any, seen: list[str]
) -> None:
    """Record every frozen-input creation attempt, without changing behaviour.

    Recording rather than raising is deliberate: the scheduler isolates job
    failures, so a raise inside run() would be swallowed instead of failing the
    test.
    """
    import r2v_data_v2.v3.post_mask_epoch_reference_integrity as module

    for name in (
        "_create_or_verify_review_input",
        "_publish_review_context",
        "_create_or_verify_bbox_input",
        "_materialize_bbox_input",
    ):
        real = getattr(runner, name)

        def guard(*args: Any, _name: str = name, _real: Any = real, **kwargs: Any) -> Any:
            seen.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(runner, name, guard)

    real_materialize = module._materialize_source_bbox

    def materialize(*args: Any, **kwargs: Any) -> Any:
        seen.append("_materialize_source_bbox")
        return real_materialize(*args, **kwargs)

    monkeypatch.setattr(module, "_materialize_source_bbox", materialize)


def _runnable_job(runner: Any, variant: str) -> Any:
    jobs = runner.seed_jobs()
    return next(job for job in jobs if dict(job.target)["variant"] == variant)


def test_main_review_worker_cannot_create_its_frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run(job) executes the judge and never freezes or repairs anything."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-worker-guard")
    judge = _fake_main_judge(judge_result)
    runner = _runner(config, storage, tmp_path)
    job = _runnable_job(runner, "final")

    seen: list[str] = []
    _guard_creation(monkeypatch, runner, seen)

    # Only run() is guarded: the main-thread finalizer is a legitimate creation
    # point, the model worker is not.
    result = runner.run(job, judge)

    assert result.payload["status"] == "review"
    assert len(judge.calls) == 1, "the Qwen review really ran"
    assert seen == [], f"the worker tried to create durable input: {seen}"
    # The frozen input it consumed still validates.
    shard, storage_again, _clip, entity, reference, _plan, _pre = (
        runner._job_review_context(job)
    )
    anchor = runner._require_review_input(
        shard, storage_again, job.clip_uid, entity, reference
    )
    assert anchor["semantic_digest"] == job.input_digest


def test_missing_main_review_input_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker must not regenerate a frozen input that vanished."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-worker-missing")
    judge = _fake_main_judge(judge_result)
    runner = _runner(config, storage, tmp_path)
    job = _runnable_job(runner, "final")

    anchor_path = runner._review_input_path(SHARD, "clip-1", "e2", "final")
    assert anchor_path.is_file()
    anchor_path.unlink()

    seen: list[str] = []
    _guard_creation(monkeypatch, runner, seen)
    with pytest.raises(ReferenceIntegrityDurableError):
        runner.run(job, judge)
    assert not anchor_path.exists(), "the worker must not recreate the anchor"
    assert seen in ([], ["_create_or_verify_review_input"]), seen
    assert len(judge.calls) == 0, "no model call may happen without frozen input"


def test_bbox_worker_cannot_materialize_its_frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bbox worker only requires: no candidate, metadata or anchor creation."""
    # A reject on both the main and the alpha review is what routes an entity
    # into the artifact bbox fallback.
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    config, storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-bbox-guard"
    )
    judge = _FakeEpochJudge(results, bbox_results)
    runner = _runner(config, storage, tmp_path)

    # Run the chain once so the bbox job is unlocked by the real main-thread
    # finalizer and its receipt is committed.
    _epoch_scheduler(runner, _SerialQwenExecutor(runner, judge)).run(
        runner.seed_jobs()
    )
    assert len(judge.bbox_calls) == 1, "the chain reached the bbox model call"

    # Same ledger, brand new runner: the whole chain is re-verified read-only,
    # including the bbox input of the already-existing bbox job.
    seen: list[str] = []
    fresh = _runner(config, storage, tmp_path)
    assert fresh.ledger.root == runner.ledger.root
    _guard_creation(monkeypatch, fresh, seen)
    fresh_executor = _SerialQwenExecutor(fresh, judge)
    _epoch_scheduler(fresh, fresh_executor).run(fresh.seed_jobs())

    assert seen == [], f"a worker materialized frozen input: {seen}"
    assert fresh_executor.batches == 0, "every receipt was reused"
    assert len(judge.bbox_calls) == 1, "no repeated bbox call"
    assert fresh.reconcile_stats(SHARD).to_dict() == (
        runner.reconcile_stats(SHARD).to_dict()
    )


def test_bbox_require_refuses_to_materialize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_require_bbox_input fails closed instead of materializing a missing anchor."""
    config, storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-bbox-missing"
    )
    runner = _runner(config, storage, tmp_path)
    anchor_path = runner._bbox_input_path(SHARD, "clip-1", "e2")
    assert not anchor_path.exists()
    entity = next(
        item
        for item in storage.read_clip("clip-1").annotation.entities
        if item.entity_id == "e2"
    )

    # The anchor gate runs first, so the parent chain is never even consulted.
    with pytest.raises(ReferenceIntegrityDurableError):
        runner._require_bbox_input(
            shard=SHARD,
            storage=storage,
            clip_uid="clip-1",
            entity=entity,
            parent=None,
            trigger="artifact",
        )

    assert not anchor_path.exists(), "require must never materialize the anchor"


def test_create_and_require_build_identical_model_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both sides of the boundary derive one and the same ModelJob."""
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-identity")
    runner = _runner(config, storage, tmp_path)
    job = _runnable_job(runner, "final")
    shard, storage_again, _clip, entity, reference, plan_entry, _pre = (
        runner._job_review_context(job)
    )

    _ref_a, _anchor_a, created = runner._create_review_inputs_and_job(
        shard=shard,
        storage=storage_again,
        clip_uid=job.clip_uid,
        plan_entry=plan_entry,
        entity=entity,
        reference=reference,
        variant="final",
    )
    _ref_b, _anchor_b, required = runner._require_review_inputs_and_job(
        shard=shard,
        storage=storage_again,
        clip_uid=job.clip_uid,
        plan_entry=plan_entry,
        entity=entity,
        reference=reference,
        variant="final",
    )

    assert created.job_id() == required.job_id()
    assert created.input_digest == required.input_digest
    assert created.model_identity == required.model_identity
    assert dict(created.target) == dict(required.target)
    assert created.job_id() == job.job_id()


def test_restart_verification_is_require_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-ledger restart reuses receipts and never re-freezes an input."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-restart-guard")
    judge = _fake_main_judge(judge_result)
    first = _runner(config, storage, tmp_path)
    first_executor = _SerialQwenExecutor(first, judge)
    _epoch_scheduler(first, first_executor).run(first.seed_jobs())
    assert len(judge.calls) == 1

    # Same ledger root, brand new runner: every Python cache is empty.
    second = _runner(config, storage, tmp_path)
    assert second.ledger.root == first.ledger.root
    seen: list[str] = []
    _guard_creation(monkeypatch, second, seen)
    second_executor = _SerialQwenExecutor(second, judge)
    _epoch_scheduler(second, second_executor).run(second.seed_jobs())

    assert seen == [], f"restart verification created frozen input: {seen}"
    assert second_executor.batches == 0, "no repeated committed Qwen call"
    assert len(judge.calls) == 1, "the restart paid no extra review"


def test_corrupted_review_context_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context PNG is a model input: different bytes must fail closed."""
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-context-drift")
    judge = _fake_main_judge(judge_result)
    runner = _runner(config, storage, tmp_path)
    job = _runnable_job(runner, "final")

    shard, storage_again, _clip, entity, reference, _plan, _pre = (
        runner._job_review_context(job)
    )
    anchor = runner._require_review_input(
        shard, storage_again, job.clip_uid, entity, reference
    )
    from r2v_data_v2.v3.post_mask_epoch_reference_integrity import (
        _resolve_run_artifact,
    )

    durable_context = _resolve_run_artifact(
        storage, str(anchor["source_context_path"])
    )
    assert durable_context.is_file()
    durable_context.write_bytes(b"not the frozen context")

    # The exact drift message matters: any other failure would mean the bytes
    # were never actually compared against the expected digest.
    with pytest.raises(
        ReferenceIntegrityDurableError, match="integrity context drifted"
    ):
        runner.run(job, judge)
    assert len(judge.calls) == 0, "no model call on drifted context"

    # And the read-only require path rejects it too.
    with pytest.raises(
        ReferenceIntegrityDurableError, match="integrity context drifted"
    ):
        runner._require_review_input(
            shard, storage_again, job.clip_uid, entity, reference
        )


def test_finalize_requires_the_current_committed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed job's input is required by the finalizer, never re-frozen.

    This is the receipt/finalize crash window: the receipt is durable, the
    frozen input is gone, and the finalizer must fail closed instead of
    regenerating the input its committed receipt was judged against.
    """
    judge_result = legacy_integrity._review(accept=True, reason="usable reference")
    config, storage = _main_review_fixture(tmp_path, monkeypatch, "run-finalize-window")
    judge = _fake_main_judge(judge_result)
    runner = _runner(config, storage, tmp_path)
    executor = _SerialQwenExecutor(runner, judge)

    # Commit the receipt but stop before finalisation.
    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    jobs = runner.seed_jobs()
    job = jobs[0]
    _epoch_scheduler(runner, executor, no_finalize).run(jobs)
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None, "the receipt is durable"
    assert not runner._entity_outcome_path(SHARD, "clip-1", "e2").exists()

    anchor_path = runner._review_input_path(SHARD, "clip-1", "e2", "final")
    assert anchor_path.is_file()
    anchor_path.unlink()

    seen: list[str] = []
    _guard_creation(monkeypatch, runner, seen)
    with pytest.raises(ReferenceIntegrityDurableError):
        runner.finalize(job, committed)

    assert not anchor_path.exists(), "the finalizer must not recreate the anchor"
    assert "_create_or_verify_review_input" not in seen, (
        "the current committed job must be reconstructed read-only"
    )
    assert not runner._entity_outcome_path(SHARD, "clip-1", "e2").exists()


def test_finalize_still_creates_the_new_continuation_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opposite direction: a NEW continuation input is still frozen.

    Requiring the current chain read-only must not stop the main-thread
    finalizer from freezing the input of the job it is about to unlock.
    """
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    config, storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-continuation-create"
    )
    judge = _FakeEpochJudge(results, bbox_results)
    runner = _runner(config, storage, tmp_path)

    frozen: list[str] = []
    required: list[str] = []

    real_freeze = runner._create_or_verify_review_input
    # Production reaches the strict require through the *_derived* entry, so
    # that is the only patch target that can observe a real require replay.
    real_require = runner._require_review_input_derived

    def freeze(*args: Any, **kwargs: Any) -> Any:
        frozen.append(str(kwargs.get("variant") or "final"))
        return real_freeze(*args, **kwargs)

    def require(*args: Any, **kwargs: Any) -> Any:
        required.append(str(kwargs.get("variant") or "final"))
        return real_require(*args, **kwargs)

    monkeypatch.setattr(runner, "_create_or_verify_review_input", freeze)
    monkeypatch.setattr(runner, "_require_review_input_derived", require)

    executor = _SerialQwenExecutor(runner, judge)
    jobs = runner.seed_jobs()
    assert frozen == ["final"], "seed freezes the initial main review"

    # Capture what the MAIN finalizer unlocks, then stop the drain so the
    # continuation stays pending and its input stays observable.
    unlocked: list[Any] = []

    def capture_main_unlock(job: Any, result: Any) -> Any:
        if str(dict(job.target).get("variant")) == "source_alpha":
            return ()
        unlocked.extend(runner.finalize(job, result))
        return ()

    frozen.clear()
    required.clear()
    # Exercise the strict path: this test is about the continuation still being
    # frozen when the finalizer cannot reuse run()'s validated context.
    runner._validated_review_contexts.clear()
    _epoch_scheduler(runner, executor, capture_main_unlock).run(jobs)

    # The current committed main review is never re-frozen: it is either reused
    # from the validated context or required read-only.
    assert "final" not in frozen, "the committed main input was re-frozen"
    # The source-alpha continuation input is frozen and durable, and the alpha
    # job is what the next seed issues.
    alpha_input = runner._review_input_path(SHARD, "clip-1", "e2", "source_alpha")
    assert alpha_input.is_file(), "the continuation input was frozen"
    assert alpha_input.exists(), "the continuation input stays durable"


def test_finalize_requires_the_current_source_alpha_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The current alpha job is required too, not just the current main one."""
    results = (
        legacy_integrity._severe_reference_artifact_review(),
        legacy_integrity._severe_reference_artifact_review(),
    )
    bbox_results = (legacy_integrity._bbox_review(accept=True),)
    config, storage = _alpha_continuation_fixture(
        tmp_path, monkeypatch, "run-alpha-window"
    )
    judge = _FakeEpochJudge(results, bbox_results)
    runner = _runner(config, storage, tmp_path)

    alpha_jobs: list[Any] = []

    def capture_main_unlock(job: Any, result: Any) -> Any:
        if str(dict(job.target).get("variant")) == "source_alpha":
            return ()
        unlocked = runner.finalize(job, result)
        alpha_jobs.extend(unlocked)
        return ()

    jobs = runner.seed_jobs()
    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), capture_main_unlock
    ).run(jobs)
    assert len(alpha_jobs) == 1, "the main review unlocked the alpha continuation"
    alpha_job = alpha_jobs[0]

    # Commit the alpha receipt, then stop before its finalisation.

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([alpha_job])
    alpha_result = runner.ledger.load_committed_result(alpha_job)
    assert alpha_result is not None, "the alpha receipt is durable"
    assert alpha_result.payload["status"] == "review"

    alpha_input = runner._review_input_path(SHARD, "clip-1", "e2", "source_alpha")
    assert alpha_input.is_file()
    alpha_input.unlink()

    seen: list[str] = []
    _guard_creation(monkeypatch, runner, seen)
    with pytest.raises(ReferenceIntegrityDurableError):
        runner.finalize(alpha_job, alpha_result)

    assert not alpha_input.exists(), "the alpha input must not be recreated"
    assert "_create_or_verify_review_input" not in seen


# ---------------------------------------------------------------------------
# Frozen-plan validation is per shard, not per clip and not per model job
# ---------------------------------------------------------------------------


def _record_method(
    monkeypatch: pytest.MonkeyPatch, target: Any, name: str, seen: list[Any]
) -> None:
    """Record the clip_uid argument of every call to one bound method."""
    real = getattr(target, name)

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("clip_uid", args[2] if len(args) > 2 else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, recording)


def _seeded_review_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str
) -> tuple[Any, Any, Any, Any]:
    config, storage = _main_review_fixture(tmp_path, monkeypatch, run_name)
    judge = _fake_main_judge(
        legacy_integrity._review(accept=True, reason="usable reference")
    )
    return config, storage, _runner(config, storage, tmp_path), judge


def test_seed_validates_the_shard_plan_once_not_once_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, _judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-seed"
    )
    runner._plan(SHARD)  # freeze the plan first, like a real second invocation

    per_clip: list[Any] = []
    reconciles: list[Any] = []
    _record_method(monkeypatch, runner, "_verify_plan_entry", per_clip)
    _record_method(monkeypatch, runner, "_existing_plan_for_reconcile", reconciles)

    jobs = runner.seed_jobs()

    assert jobs, "the fixture still seeds its review job"
    assert runner.plan_counters["plan_full_validation_count"] == 1
    # Every clip check during seeding belongs to this shard's single clip: the
    # one from the full validation, plus the publication's own hot check.
    assert set(per_clip) == {"clip-1"}, per_clip
    # Neither the seed loop nor _advance_clip re-reads the shard plan.
    assert reconciles == [], reconciles


def test_advance_clip_does_not_revalidate_the_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, _judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-advance"
    )
    plan = runner._plan(SHARD)
    runner._remember_validated_plan(SHARD, plan)
    entry = plan["clips"]["clip-1"]

    reconciles: list[Any] = []
    _record_method(monkeypatch, runner, "_existing_plan_for_reconcile", reconciles)

    jobs = runner._advance_clip(SHARD, storage, "clip-1", entry)

    assert reconciles == [], "an already-validated entry must be passed in"
    assert [dict(job.target)["variant"] for job in jobs] == ["final"]


def test_model_job_validates_only_its_own_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-hot"
    )
    job = _runnable_job(runner, "final")

    per_clip: list[Any] = []
    reconciles: list[Any] = []
    _record_method(monkeypatch, runner, "_verify_plan_entry", per_clip)
    _record_method(monkeypatch, runner, "_existing_plan_for_reconcile", reconciles)

    before = runner.plan_counters["plan_full_validation_count"]
    result = runner.run(job, judge)

    assert result.payload["status"] == "review"
    assert len(judge.calls) == 1, "the model really ran"
    assert reconciles == [], "the hot path must not re-read the shard plan"
    assert runner.plan_counters["plan_full_validation_count"] == before, (
        "the hot path must not run a full-shard validation"
    )
    # Exactly one per-clip validation, for the clip that owns this job.
    assert per_clip == ["clip-1"], per_clip
    assert runner.plan_counters["plan_cache_hit"] >= 1


def test_semantic_plan_tamper_is_not_hidden_by_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-tamper"
    )
    job = _runnable_job(runner, "final")
    assert runner._validated_plan_digests.get(SHARD)

    # A genuinely invalid durable plan change, made after validation.
    plan_path = runner._plan_path(SHARD)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["clips"]["clip-1"]["classification"] = "not_a_classification"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceIntegrityDurableError):
        runner.run(job, judge)
    assert len(judge.calls) == 0, "no model call on a tampered plan"


def test_equivalent_plan_rewrite_keeps_its_semantics() -> None:
    """The identity is canonical semantics, never raw file bytes."""
    compact = {"schema": "s", "clips": {"clip-1": {"a": 1, "b": [1, 2]}}}
    formatted = {"clips": {"clip-1": {"b": [1, 2], "a": 1}}, "schema": "s"}

    assert ReferenceIntegrityEpochRunner._plan_semantic_digest(
        compact
    ) == ReferenceIntegrityEpochRunner._plan_semantic_digest(formatted)
    assert ReferenceIntegrityEpochRunner._plan_semantic_digest(
        compact
    ) != ReferenceIntegrityEpochRunner._plan_semantic_digest(
        {"schema": "s", "clips": {"clip-1": {"a": 2, "b": [1, 2]}}}
    )


def test_current_clip_live_drift_is_still_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is per-plan, never blind trust in the clip's live state."""
    _config, storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-live-drift"
    )
    job = _runnable_job(runner, "final")

    # Drift the live annotation of the very clip this job belongs to.
    clip_path = storage.clip_path("clip-1")
    payload = json.loads(clip_path.read_text(encoding="utf-8"))
    payload["annotation"]["entities"][0]["phrase"] = "a different phrase"
    clip_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReferenceIntegrityDurableError):
        runner.run(job, judge)
    assert len(judge.calls) == 0, "no model call on a drifted clip"


def test_cold_runner_still_fully_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, _seed_runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-plan-cold"
    )
    _run_epoch(config, storage, tmp_path, judge)

    cold = _runner(config, storage, tmp_path)
    assert cold._validated_plan_entries == {}
    assert cold._validated_plan_digests == {}

    cold.seed_jobs()
    assert cold.plan_counters["plan_full_validation_count"] == 1
    assert cold._validated_plan_digests.get(SHARD)

    # And reconcile keeps its own fully strict validation.
    before = cold.plan_counters["plan_full_validation_count"]
    cold.reconcile_stats(SHARD)
    assert cold.plan_counters["plan_full_validation_count"] == before + 1


# ---------------------------------------------------------------------------
# The hot path stats the plan; it does not read, parse or hash it
# ---------------------------------------------------------------------------


def _reset_plan_counters(runner: Any) -> None:
    """Zero the execution-only plan diagnostics before the phase under test."""
    for key in runner.plan_counters:
        runner.plan_counters[key] = 0


def _guard_plan_reads(
    monkeypatch: pytest.MonkeyPatch, runner: Any
) -> dict[str, list[Any]]:
    """Record whole-plan reads and semantic digests, without changing behaviour."""
    import r2v_data_v2.v3.post_mask_epoch_reference_integrity as module

    plan_path = runner._plan_path(SHARD)
    seen: dict[str, list[Any]] = {"reads": [], "digests": [], "reconciles": []}

    real_read = module._read_json

    def reading(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == plan_path:
            seen["reads"].append(path)
        return real_read(path, *args, **kwargs)

    real_digest = ReferenceIntegrityEpochRunner._plan_semantic_digest

    def digest(payload: Any) -> str:
        seen["digests"].append(payload)
        return real_digest(payload)

    real_reconcile = runner._existing_plan_for_reconcile

    def reconcile(shard: str) -> Any:
        seen["reconciles"].append(shard)
        return real_reconcile(shard)

    monkeypatch.setattr(module, "_read_json", reading)
    monkeypatch.setattr(
        ReferenceIntegrityEpochRunner, "_plan_semantic_digest", staticmethod(digest)
    )
    monkeypatch.setattr(runner, "_existing_plan_for_reconcile", reconcile)
    return seen


def test_hot_job_never_reads_the_whole_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-hot"
    )
    job = _runnable_job(runner, "final")

    per_clip: list[Any] = []
    _record_method(monkeypatch, runner, "_verify_plan_entry", per_clip)
    seen = _guard_plan_reads(monkeypatch, runner)
    _reset_plan_counters(runner)

    result = runner.run(job, judge)

    assert result.payload["status"] == "review"
    assert len(judge.calls) == 1, "the model really ran"
    assert seen["reads"] == [], "the hot path must not read the plan"
    assert seen["digests"] == [], "the hot path must not hash the plan"
    assert seen["reconciles"] == [], "the hot path must not full-validate"
    assert per_clip == ["clip-1"], per_clip
    assert runner.plan_counters["plan_stat_hit"] == 1
    assert runner.plan_counters["plan_content_read_count"] == 0
    assert runner.plan_counters["plan_semantic_digest_count"] == 0


def test_hot_finalize_never_reads_the_whole_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-finalize"
    )
    job = _runnable_job(runner, "final")

    # Commit the receipt first, exactly like the scheduler does.
    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None

    per_clip: list[Any] = []
    _record_method(monkeypatch, runner, "_verify_plan_entry", per_clip)
    seen = _guard_plan_reads(monkeypatch, runner)

    runner.finalize(job, committed)

    assert seen["reads"] == [], "finalize must not read the whole plan"
    assert seen["digests"] == [], "finalize must not hash the whole plan"
    assert seen["reconciles"] == [], "finalize must not full-validate"
    # Only this clip is ever verified, never the rest of the shard.
    assert set(per_clip) == {"clip-1"}, per_clip


def test_plan_content_tamper_is_detected_from_the_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-tamper"
    )
    job = _runnable_job(runner, "final")
    plan_path = runner._plan_path(SHARD)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["clips"]["clip-1"]["classification"] = "not_a_classification"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    _reset_plan_counters(runner)

    with pytest.raises(ReferenceIntegrityDurableError):
        runner.run(job, judge)

    assert len(judge.calls) == 0, "no model call on a tampered plan"
    assert runner.plan_counters["plan_stat_miss"] == 1
    assert runner.plan_counters["plan_content_read_count"] == 1
    assert runner.plan_counters["plan_semantic_digest_count"] == 1


def test_equivalent_plan_rewrite_keeps_the_validated_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reformat is not a semantic change, and must not re-validate N clips."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-rewrite"
    )
    job = _runnable_job(runner, "final")
    plan_path = runner._plan_path(SHARD)

    # Same parsed JSON, different key order and whitespace.
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cached_signature = runner._validated_plan_signatures[SHARD]
    assert runner._file_signature(plan_path) != cached_signature

    per_clip: list[Any] = []
    _record_method(monkeypatch, runner, "_verify_plan_entry", per_clip)
    seen = _guard_plan_reads(monkeypatch, runner)

    result = runner.run(job, judge)

    assert result.payload["status"] == "review"
    assert seen["reads"] == [plan_path], "the changed file is read once"
    assert seen["digests"] != [], "the canonical digest is compared"
    assert seen["reconciles"] == [], "an equivalent rewrite is not re-validated"
    assert per_clip == ["clip-1"], "only the current clip is verified"
    # The signature was refreshed, so the next lookup is stat-only again.
    assert runner._validated_plan_signatures[SHARD] == runner._file_signature(
        plan_path
    )
    before_reads = runner.plan_counters["plan_content_read_count"]
    runner._hot_plan_entry(SHARD, _storage, "clip-1")
    assert runner.plan_counters["plan_content_read_count"] == before_reads


def test_same_size_change_with_restored_mtime_is_still_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Size + mtime are not enough; inode/ctime are what make this safe."""
    import os

    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-ctime"
    )
    job = _runnable_job(runner, "final")
    plan_path = runner._plan_path(SHARD)
    before_text = plan_path.read_text(encoding="utf-8")
    before_stat = plan_path.stat()
    cached_signature = runner._validated_plan_signatures[SHARD]

    # Same byte length, different semantics.
    mutated = before_text.replace('"clip-1"', '"clip-2"', 1)
    assert len(mutated) == len(before_text)
    plan_path.write_text(mutated, encoding="utf-8")
    os.utime(plan_path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))

    after_stat = plan_path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert runner._file_signature(plan_path) != cached_signature

    with pytest.raises(ReferenceIntegrityDurableError):
        runner.run(job, judge)
    assert len(judge.calls) == 0


def test_missing_plan_is_never_served_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-missing"
    )
    job = _runnable_job(runner, "final")
    assert runner._validated_plan_entries.get((SHARD, "clip-1"))

    runner._plan_path(SHARD).unlink()

    # The missing-plan error is the pre-existing one from the strict reader; the
    # point is that the cached entry is never used and no model call happens.
    with pytest.raises(
        (ReferenceIntegrityDurableError, ReferenceIntegrityEpochError)
    ):
        runner.run(job, judge)
    assert len(judge.calls) == 0, "a job cannot run without its frozen plan"


def test_cold_runner_captures_a_signature_then_stats_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, _seed_runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-sig-cold"
    )
    job = _runnable_job(_seed_runner, "final")
    _epoch_scheduler(
        _seed_runner, _SerialQwenExecutor(_seed_runner, judge), lambda job, r: ()
    ).run([job])

    cold = _runner(config, storage, tmp_path)
    assert cold._validated_plan_signatures == {}

    cold.seed_jobs()
    assert cold.plan_counters["plan_full_validation_count"] == 1
    assert SHARD in cold._validated_plan_signatures

    # The next lookup is stat-only.
    _reset_plan_counters(cold)
    cold._hot_plan_entry(SHARD, storage, "clip-1")
    assert cold.plan_counters["plan_stat_hit"] == 1
    assert cold.plan_counters["plan_content_read_count"] == 0


# ---------------------------------------------------------------------------
# The finalizer reuses the chain run() strictly validated
# ---------------------------------------------------------------------------


def _review_derivation_guards(
    monkeypatch: pytest.MonkeyPatch, runner: Any
) -> dict[str, list[Any]]:
    """Record expensive review-input derivation, without changing behaviour."""
    import r2v_data_v2.v3.post_mask_epoch_reference_integrity as module

    seen: dict[str, list[Any]] = {"evidence": [], "requires": [], "encodes": []}

    real_evidence = module._source_evidence

    def evidence(*args: Any, **kwargs: Any) -> Any:
        seen["evidence"].append(kwargs.get("clip_uid"))
        return real_evidence(*args, **kwargs)

    # The require path goes through the derived variant, so guarding that one
    # catches every strict review-input reconstruction.
    real_require = runner._require_review_input_derived

    def require(*args: Any, **kwargs: Any) -> Any:
        # Record the caller so a require replay driven by terminal publication
        # can be told apart from the job-context replay this commit removes.
        caller = inspect.currentframe().f_back.f_code.co_name  # type: ignore[union-attr]
        seen["requires"].append((kwargs.get("variant"), caller))
        return real_require(*args, **kwargs)

    real_encode = runner._encode_review_context

    def encode(*args: Any, **kwargs: Any) -> Any:
        seen["encodes"].append(kwargs.get("variant"))
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(module, "_source_evidence", evidence)
    monkeypatch.setattr(runner, "_require_review_input_derived", require)
    monkeypatch.setattr(runner, "_encode_review_context", encode)
    return seen


def test_run_still_strictly_rederives_after_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """seed -> run must stay a strict re-derivation, never a cache reuse."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-strict-again"
    )
    job = _runnable_job(runner, "final")

    seen = _review_derivation_guards(monkeypatch, runner)
    runner.run(job, judge)

    assert len(judge.calls) == 1, "the model really ran"
    assert seen["evidence"], "run() must re-derive the source evidence"
    assert seen["requires"] == [("final", "_review_inputs_and_job")], seen["requires"]
    assert runner.review_context_counters["review_context_cache_store"] == 1


def test_finalize_hot_cache_skips_the_second_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalizer reuses run()'s validated chain: no second derivation."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-hot-reuse"
    )
    job = _runnable_job(runner, "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None

    seen = _review_derivation_guards(monkeypatch, runner)
    runner.finalize(job, committed)

    # The review-input chain is not re-derived. Source evidence may still be
    # derived by the terminal publication, which this commit deliberately does
    # not cache (topology/entity publication is a later change).
    job_context_requires = [
        entry for entry in seen["requires"] if entry[1] == "_review_outcome"
    ]
    assert job_context_requires == [], job_context_requires
    assert runner.review_context_counters["review_context_cache_hit"] == 1
    assert runner._entity_outcome_path(SHARD, "clip-1", "e2").is_file()


def _cached_chain_paths(runner: Any, job: Any, storage: Any) -> dict[str, Any]:
    """Resolve the four signed artifacts of one cached chain, unambiguously.

    Roles are read off the real artifact layout the derivation uses - the anchor
    JSON, the derived ``integrity_context_*.png``, the frame under ``frames/``
    and the remaining reference image - never guessed from a suffix alone.
    """
    cached = runner._validated_review_contexts[job.job_id()]
    signed = [path for path, _signature in cached.file_signatures]
    assert len(signed) == 4, signed

    anchor_path = next(path for path in signed if path.suffix == ".json")
    context_path = next(path for path in signed if "integrity_context" in path.name)
    source_frame_path = next(path for path in signed if path.parent.name == "frames")
    remaining = [
        path
        for path in signed
        if path not in {anchor_path, context_path, source_frame_path}
    ]
    assert len(remaining) == 1, remaining
    reference_path = remaining[0]

    assert os.path.realpath(source_frame_path) != os.path.realpath(reference_path)
    assert os.path.realpath(source_frame_path) != os.path.realpath(context_path)
    assert os.path.realpath(anchor_path) != os.path.realpath(context_path)
    return {
        "anchor": anchor_path,
        "context": context_path,
        "reference": reference_path,
        "source_frame": source_frame_path,
        "cached": cached,
    }


@pytest.mark.parametrize("which", ("source_frame", "context", "reference", "anchor"))
def test_changed_artifact_bypasses_the_cache_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """A changed artifact is not a cache hit, and the strict path decides."""
    _config, storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, f"run-changed-{which}"
    )
    job = _runnable_job(runner, "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None

    paths = _cached_chain_paths(runner, job, storage)
    assert os.path.realpath(paths["source_frame"]) != os.path.realpath(
        paths["reference"]
    )
    assert os.path.realpath(paths["source_frame"]) != os.path.realpath(
        paths["context"]
    )
    target = paths[which]
    assert target.is_file()
    target.write_bytes(target.read_bytes() + b"x")

    seen = _review_derivation_guards(monkeypatch, runner)
    if which == "source_frame":
        # Appending bytes to the JPEG does not change the decoded source pixels,
        # so the strict path legitimately re-derives the same input and accepts
        # it: what matters here is that it ran instead of trusting the cache.
        runner.finalize(job, committed)
        assert seen["requires"], "the strict require path must actually run"
        assert runner._entity_outcome_path(SHARD, "clip-1", "e2").is_file()
    else:
        with pytest.raises(ReferenceIntegrityDurableError):
            runner.finalize(job, committed)
        assert seen["requires"], "the strict require path must actually run"
        assert not runner._entity_outcome_path(SHARD, "clip-1", "e2").exists(), (
            "no semantic publication on a bypassed cache"
        )
    assert runner.review_context_counters["review_context_signature_miss"] >= 1


def test_cached_context_signatures_are_all_real_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored generation may never contain a missing-file placeholder."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-signature-invariant"
    )
    job = _runnable_job(runner, "final")
    runner.run(job, judge)

    cached = runner._validated_review_contexts[job.job_id()]
    assert cached.file_signatures
    for path, signature in cached.file_signatures:
        assert isinstance(signature, _FileSignature), signature
        assert signature is not None
        assert path.is_file(), path

def test_missing_file_at_capture_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that vanished during capture disables the cache, not semantics."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-capture-race"
    )
    job = _runnable_job(runner, "final")
    anchor_path = runner._review_input_path(SHARD, "clip-1", "e2", "final")

    real_build = runner._build_validated_context

    def racing(*args: Any, **kwargs: Any) -> Any:
        # The strict require has already succeeded; the frozen anchor disappears
        # before the signatures are captured.
        anchor_path.unlink()
        return real_build(*args, **kwargs)

    monkeypatch.setattr(runner, "_build_validated_context", racing)

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])

    assert len(judge.calls) == 1, "the model call still proceeded"
    assert job.job_id() not in runner._validated_review_contexts, (
        "an incomplete generation must never be cached"
    )
    assert runner.review_context_counters["review_context_capture_miss"] == 1
    monkeypatch.setattr(runner, "_build_validated_context", real_build)
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None
    seen = _review_derivation_guards(monkeypatch, runner)
    with pytest.raises(
        (ReferenceIntegrityDurableError, ReferenceIntegrityEpochError)
    ):
        runner.finalize(job, committed)
    assert seen["requires"], "the strict require path must actually run"
    assert not anchor_path.exists(), "the missing input must not be recreated"


def test_real_eviction_falls_back_to_the_strict_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real eviction only costs CPU: the strict require path runs instead."""
    _config, _storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-real-evict"
    )
    monkeypatch.setattr(runner, "_review_context_limit", 1)

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    jobs = runner.seed_jobs()
    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run(jobs)
    first = jobs[0]
    assert runner.review_context_counters["review_context_cache_store"] == 1

    # A second, independent job context evicts the first.
    second = ModelJob.create(
        job_type=first.job_type,
        resource=first.resource,
        canonical_shard=first.canonical_shard,
        clip_uid=first.clip_uid,
        semantic_inputs={"probe": "eviction"},
        model_identity=first.model_identity,
        target=dict(first.target),
    )
    cached = runner._validated_review_contexts[first.job_id()]
    runner._remember_validated_context(
        replace(cached, job_id=second.job_id())
    )

    assert runner.review_context_counters["review_context_cache_eviction"] == 1
    assert first.job_id() not in runner._validated_review_contexts

    runner.review_context_counters["review_context_cache_hit"] = 0
    seen = _review_derivation_guards(monkeypatch, runner)
    committed = runner.ledger.load_committed_result(first)
    runner.finalize(first, committed)

    assert runner.review_context_counters["review_context_cache_hit"] == 0
    assert seen["requires"], "an evicted context must replay strictly"
    assert runner._entity_outcome_path(SHARD, "clip-1", "e2").is_file()


def test_cold_runner_never_assumes_a_cached_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new runner on the same ledger starts empty and replays strictly."""
    config, storage, seed_runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-cold-context"
    )
    job = _runnable_job(seed_runner, "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        seed_runner, _SerialQwenExecutor(seed_runner, judge), no_finalize
    ).run([job])
    committed = seed_runner.ledger.load_committed_result(job)

    cold = _runner(config, storage, tmp_path)
    assert cold.ledger.root == seed_runner.ledger.root
    assert cold._validated_review_contexts == {}
    assert cold._fresh_validated_context(job, storage, {}) is None
    assert cold.review_context_counters["review_context_cache_miss"] == 1

    # The strict path still finalises correctly with no cache at all.
    cold.finalize(job, committed)
    assert cold._entity_outcome_path(SHARD, "clip-1", "e2").is_file()


# ---------------------------------------------------------------------------
# Terminal publication reuses the marker this invocation just published
# ---------------------------------------------------------------------------


def _terminal_replay_guards(
    monkeypatch: pytest.MonkeyPatch, runner: Any
) -> dict[str, list[Any]]:
    """Record the expensive terminal-reconstruction work, without changing it."""
    seen: dict[str, list[Any]] = {
        "verified_branch": [],
        "diagnostics": [],
        "requires": [],
        "marker_reads": [],
    }
    for name, bucket in (
        ("_verify_entity_branch", "verified_branch"),
        ("_review_diagnostics", "diagnostics"),
        ("_require_review_input_derived", "requires"),
        ("_entity_outcome", "marker_reads"),
    ):
        real = getattr(runner, name)

        def wrapper(*args: Any, _real: Any = real, _bucket: str = bucket, **kwargs: Any) -> Any:
            seen[_bucket].append(True)
            return _real(*args, **kwargs)

        monkeypatch.setattr(runner, name, wrapper)
    return seen


def _committed_review(runner: Any, judge: Any) -> tuple[Any, Any]:
    """Seed, run and commit one main review job without finalising it."""
    job = _runnable_job(runner, "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None
    return job, committed


@pytest.fixture
def published_except_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, Any, Any, Any]:
    """A finalised entity marker whose terminal clip publication is still ahead."""
    config, storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-marker-pending"
    )
    job, committed = _committed_review(runner, judge)
    real_publish = runner._publish_clip_if_terminal
    monkeypatch.setattr(runner, "_publish_clip_if_terminal", lambda *a, **k: None)
    runner.finalize(job, committed)
    monkeypatch.setattr(runner, "_publish_clip_if_terminal", real_publish)
    assert runner._verified_entity_markers, "the marker was verified and cached"
    return config, storage, runner, real_publish, job


def test_main_reviewed_entity_skips_immediate_terminal_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finalizer publishes the marker the terminal replay then reuses."""
    _config, storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-no-replay"
    )
    job, committed = _committed_review(runner, judge)

    seen = _terminal_replay_guards(monkeypatch, runner)
    runner.finalize(job, committed)

    # The finalizer derives the marker once, and nothing else does.
    assert len(seen["diagnostics"]) == 1, seen["diagnostics"]
    assert seen["marker_reads"], "the durable marker must still be read"
    assert seen["verified_branch"] == [], seen["verified_branch"]
    assert seen["requires"] == [], seen["requires"]
    assert runner.entity_verify_counters["entity_verify_cache_hit"] >= 1
    assert runner.entity_verify_counters["entity_verify_cache_miss"] == 0

    clip = storage.read_clip("clip-1")
    assert clip.reference_integrity is not None
    assert clip.reference_integrity.status == "ready"
    assert runner._clip_outcome_path(SHARD, "clip-1").is_file()


def test_shape_valid_marker_tamper_is_not_hidden(
    published_except_clip: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed-but-valid durable marker bypasses the cache and fails closed."""
    _config, storage, runner, real_publish, _job = published_except_clip

    marker_path = runner._entity_outcome_path(SHARD, "clip-1", "e2")
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["delta"] = {**payload["delta"], "entities_reviewed": 99}
    marker_path.write_text(json.dumps(payload), encoding="utf-8")
    # Still structurally acceptable to the durable reader: only the semantic
    # branch verification can notice.
    assert runner._entity_outcome(SHARD, "clip-1", "e2")["delta"][
        "entities_reviewed"
    ] == 99

    seen = _terminal_replay_guards(monkeypatch, runner)
    with pytest.raises(ReferenceIntegrityDurableError):
        real_publish(SHARD, storage, "clip-1")

    assert seen["verified_branch"], "the strict verifier must run on a mismatch"
    assert runner.entity_verify_counters["entity_verify_marker_mismatch"] >= 1
    assert not runner._clip_outcome_path(SHARD, "clip-1").exists(), (
        "no publication after a mismatch"
    )


def test_missing_marker_is_not_served_from_the_cache(
    published_except_clip: tuple[Any, Any, Any, Any, Any],
) -> None:
    """A removed durable marker fails closed; the cache cannot substitute."""
    _config, storage, runner, real_publish, _job = published_except_clip

    runner._entity_outcome_path(SHARD, "clip-1", "e2").unlink()

    # The durable reader is the only source: with the marker gone the clip is
    # simply not terminal, so nothing is published. The cache must not stand in
    # for the missing marker.
    real_publish(SHARD, storage, "clip-1")

    assert not runner._clip_outcome_path(SHARD, "clip-1").exists()
    assert storage.read_clip("clip-1").reference_integrity is None


def test_publication_readback_must_equal_what_was_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker only ever becomes trusted after an equal durable readback."""
    _config, _storage, runner, _judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-marker-readback"
    )
    body = {
        "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "reviewed": False,
    }

    # The durable readback is what makes the cache trustworthy, so a readback
    # that disagrees with what was just written must not enter it.
    monkeypatch.setattr(
        runner, "_entity_outcome", lambda *a, **k: {**body, "reason": "drifted"}
    )

    with pytest.raises(ReferenceIntegrityDurableError):
        runner._publish_entity_marker(SHARD, "clip-1", "e1", body)
    assert (SHARD, "clip-1", "e1") not in runner._verified_entity_markers


def test_durable_read_cannot_be_replaced_by_the_cache(
    published_except_clip: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-entity durable read is load-bearing: the cache may not stand in.

    ``_publish_clip_if_terminal`` returns early when a marker is already gone, so
    a marker that disappears *between* that existence check and the per-entity
    read is the case where the cache could otherwise substitute for durable
    state. The strict reader must still be the only source.
    """
    _config, storage, runner, real_publish, _job = published_except_clip
    assert runner._verified_entity_markers, "the markers are cached"

    real_reconstruction = runner._reconstruct_clip_publication
    vanished = runner._entity_outcome_path(SHARD, "clip-1", "e1")
    assert vanished.is_file()

    def racing(*args: Any, **kwargs: Any) -> Any:
        # The marker survived the existence check and disappears right before the
        # durable read this reconstruction performs.
        vanished.unlink()
        return real_reconstruction(*args, **kwargs)

    monkeypatch.setattr(runner, "_reconstruct_clip_publication", racing)

    with pytest.raises(ReferenceIntegrityDurableError):
        real_publish(SHARD, storage, "clip-1")
    assert not runner._clip_outcome_path(SHARD, "clip-1").exists()
    assert not vanished.exists(), "the missing marker must not be recreated"


def test_cold_runner_reverifies_the_marker_strictly(
    published_except_clip: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new runner on the same ledger starts empty and replays strictly."""
    config, storage, seed_runner, _real_publish, _job = published_except_clip

    cold = _runner(config, storage, seed_runner.ledger.root.parent)
    assert cold._verified_entity_markers == {}

    seen = _terminal_replay_guards(monkeypatch, cold)
    cold._publish_clip_if_terminal(SHARD, storage, "clip-1")

    assert seen["marker_reads"], "the durable marker is still read"
    assert seen["verified_branch"], "a cold runner must verify strictly"
    assert cold.entity_verify_counters["entity_verify_cache_miss"] >= 1
    assert cold._clip_outcome_path(SHARD, "clip-1").is_file()


def test_real_marker_cache_eviction_falls_back_to_strict_verification(
    published_except_clip: tuple[Any, Any, Any, Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real eviction only costs CPU: the strict verifier runs instead."""
    _config, storage, runner, real_publish, _job = published_except_clip

    monkeypatch.setattr(runner, "_verified_marker_limit", 1)
    other = runner._verified_entity_markers[(SHARD, "clip-1", "e2")]
    runner._remember_verified_marker(SHARD, "clip-1", "e9", other)

    assert runner.entity_verify_counters["entity_verify_cache_eviction"] >= 1
    assert (SHARD, "clip-1", "e2") not in runner._verified_entity_markers

    before = runner.entity_verify_counters["entity_verify_cache_hit"]
    seen = _terminal_replay_guards(monkeypatch, runner)
    real_publish(SHARD, storage, "clip-1")

    assert seen["verified_branch"], "an evicted marker must be verified strictly"
    assert runner.entity_verify_counters["entity_verify_cache_hit"] == before
    assert runner._clip_outcome_path(SHARD, "clip-1").is_file()


def _guarded_terminal_phases(
    monkeypatch: pytest.MonkeyPatch, runner: Any
) -> dict[str, list[Any]]:
    """Record per-entity branch verification and every real topology pass.

    ``topology`` grows by one for each genuine ``reference_topology_diagnostics``
    call, so a caller can snapshot its length around one phase and measure exactly
    what that phase computed.
    """
    import r2v_data_v2.v3.post_mask_epoch_reference_integrity as module

    seen: dict[str, list[Any]] = {"branch_entities": [], "topology": []}

    real_branch = runner._verify_entity_branch

    def branch(*args: Any, **kwargs: Any) -> Any:
        marker = args[7] if len(args) > 7 else kwargs.get("marker")
        seen["branch_entities"].append(str(dict(marker).get("entity_id")))
        return real_branch(*args, **kwargs)

    real_topology = module.reference_topology_diagnostics

    def topology(*args: Any, **kwargs: Any) -> Any:
        seen["topology"].append(True)
        return real_topology(*args, **kwargs)

    monkeypatch.setattr(runner, "_verify_entity_branch", branch)
    monkeypatch.setattr(module, "reference_topology_diagnostics", topology)
    return seen


def _phase_topology(
    monkeypatch: pytest.MonkeyPatch, runner: Any, seen: dict[str, list[Any]], name: str
) -> list[int]:
    """Length of ``seen["topology"]`` right after each ``<name>`` call."""
    measured: list[int] = []
    real = getattr(runner, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        measured.append(len(seen["topology"]))
        return out

    monkeypatch.setattr(runner, name, wrapper)
    return measured


def test_cpu_terminal_entity_does_not_recompute_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU-terminal entity computes topology once and never again.

    The fixture naturally carries two entities: ``e1`` terminates on the CPU
    branch and ``e2`` needs the model. Both topology passes during seeding are
    that entity's own decision - one for the CPU marker, one for deciding that
    ``e2`` requires a review - and terminal publication afterwards adds none.
    """
    _config, storage, runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-cpu-topology"
    )

    seen = _guarded_terminal_phases(monkeypatch, runner)
    jobs = runner.seed_jobs()

    # Seeding derives the CPU-terminal marker itself, through the cached
    # publication point, and never re-verifies it.
    assert seen["topology"] == [True, True], seen["topology"]
    assert seen["branch_entities"] == [], seen["branch_entities"]
    assert runner._entity_outcome(SHARD, "clip-1", "e1") is not None
    assert (SHARD, "clip-1", "e1") in runner._verified_entity_markers

    job = next(item for item in jobs if dict(item.target)["variant"] == "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        runner, _SerialQwenExecutor(runner, judge), no_finalize
    ).run([job])
    committed = runner.ledger.load_committed_result(job)
    assert committed is not None

    # One more topology pass: the reviewed entity's own finalizer diagnostics.
    before = len(seen["topology"])
    reconstruction = _phase_topology(
        monkeypatch, runner, seen, "_reconstruct_clip_publication"
    )
    runner.finalize(job, committed)

    assert len(seen["topology"]) - before == 1, seen["topology"]
    assert reconstruction == [before + 1], reconstruction
    assert "e1" not in seen["branch_entities"], seen["branch_entities"]
    assert runner.entity_verify_counters["entity_verify_cache_hit"] >= 1
    assert storage.read_clip("clip-1").reference_integrity is not None
    assert runner._clip_outcome_path(SHARD, "clip-1").is_file()


def test_inherited_marker_is_verified_before_it_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inherited marker is strictly verified, and only then cached."""
    config, storage, seed_runner, judge = _seeded_review_fixture(
        tmp_path, monkeypatch, "run-inherited"
    )
    job = _runnable_job(seed_runner, "final")

    def no_finalize(job: Any, result: Any) -> Any:
        return ()

    _epoch_scheduler(
        seed_runner, _SerialQwenExecutor(seed_runner, judge), no_finalize
    ).run([job])
    committed = seed_runner.ledger.load_committed_result(job)
    assert committed is not None

    real_publish = seed_runner._publish_clip_if_terminal
    monkeypatch.setattr(
        seed_runner, "_publish_clip_if_terminal", lambda *a, **k: None
    )
    seed_runner.finalize(job, committed)
    monkeypatch.setattr(seed_runner, "_publish_clip_if_terminal", real_publish)

    # Runner B: same storage and ledger root, brand new, empty cache.
    inherited = _runner(config, storage, seed_runner.ledger.root.parent)
    assert inherited.ledger.root == seed_runner.ledger.root
    assert inherited._verified_entity_markers == {}

    seen = _guarded_terminal_phases(monkeypatch, inherited)
    # Seeding sees the durable markers and does not re-issue any entity, but the
    # terminal publication it reaches must still verify both strictly.
    assert inherited.seed_jobs() == []
    assert sorted(seen["branch_entities"]) == ["e1", "e2"], seen["branch_entities"]
    assert inherited.entity_verify_counters["entity_verify_cache_miss"] >= 2
    assert inherited._verified_entity_markers

    # Round two in the same runner: both trusted markers are reused.
    verified_once = list(seen["branch_entities"])
    inherited._publish_clip_if_terminal(SHARD, storage, "clip-1")
    assert seen["branch_entities"] == verified_once, "round two must not re-verify"
    assert inherited.entity_verify_counters["entity_verify_cache_hit"] >= 2
