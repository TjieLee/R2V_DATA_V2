"""Reference Integrity resource epoch (6b): durable CPU foundation.

The legacy authority is ``reference_integrity_clips()``: every CPU-only fixture
here proves the epoch reproduces it byte for byte, and the restart/corruption
tests pin the durable contract. 6b has no model job, so nothing in this file
may ever reach a Qwen judge.
"""

from __future__ import annotations

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
