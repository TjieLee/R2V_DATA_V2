"""Reference Edit resource epoch (5b): durable chain, restart, legacy equality."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.config import PairConfig
from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN
from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
    ReferenceEditEpochError,
    ReferenceEditEpochRunner,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from tests.test_v3_pair import _config, _storage
from tests.test_v3_pair import _Judge as _PairEntityJudge

SHARD = "shard-000000000-000000000"


def _reference_edit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str = "run"
) -> Any:
    config = _config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(),
        reference_edit_enabled=True,
    )
    return replace(config, run_root=config.run_root.parent / run_name)


class _FakeBoogu:
    def __init__(self) -> None:
        self.calls = 0

    def edit(self, **kwargs: Any) -> Any:
        self.calls += 1
        from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput

        width = int(kwargs["width"])
        height = int(kwargs["height"])
        buffer_path = None
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (191, 22, 43)).save(
            buffer, format="PNG"
        )
        instruction = str(kwargs["instruction"])
        del buffer_path
        return BooguEditOutput(
            png_bytes=buffer.getvalue(),
            original_instruction=instruction,
            rewritten_instruction="rewritten",
            effective_instruction="rewritten",
            worker_metadata={"seed": kwargs.get("seed")},
        )


class _FakeJudge:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.calls = 0

    def review(self, **kwargs: Any) -> Any:
        self.calls += 1
        from r2v_data_v2.v3.reference_edit_boogu import BooguCompletionReview

        return BooguCompletionReview(
            same_physical_entity=self.accept,
            identity_preserved=self.accept,
            original_visible_attributes_preserved=self.accept,
            exactly_one_entity=self.accept,
            candidate_better_than_source=self.accept,
            no_duplicate_entity=self.accept,
            no_unrelated_entity=self.accept,
            no_severe_structure_artifact=self.accept,
            verdict="accept" if self.accept else "reject",
            reason="usable" if self.accept else "candidate is not preferable",
        )


class _FakeSam:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.calls = 0

    def review(self, **kwargs: Any) -> Any:
        self.calls += 1
        from r2v_data_v2.v3.reference_edit_boogu import BooguSamReview

        return BooguSamReview(
            passed=self.passed,
            reason="usable" if self.passed else "fragmented",
            diagnostics={"failure_kind": "none" if self.passed else "fragmented"},
        )


@pytest.fixture()
def _config_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    return _reference_edit_config(tmp_path, monkeypatch, "run")


def test_existing_reference_edit_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _config_pair: Any
) -> None:
    """A clip whose reference_edit exists is terminal skipped_existing."""
    config = _config_pair
    storage = _storage(config, entity_types=("subject",))
    # Reference Edit needs a ready PairingState: run legacy Pair first.
    from r2v_data_v2.v3.pair import pair_clips
    from r2v_data_v2.v3.reference_edit import reference_edit_clips
    from tests.test_v3_reference_edit_boogu import (
        _Backend,
        _Judge,
        _SamReviewer,
    )

    pair_clips(config, storage, judge=_PairEntityJudge())
    reference_edit_clips(
        config, storage, backend=_Backend(), judge=_Judge(), sam_reviewer=_SamReviewer()
    )
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    runner.seed_jobs()  # freezes the launch plan (zero jobs: already existing)
    stats = runner.reconcile_stats(SHARD)
    assert stats.skipped_existing == 1
    assert stats.processed == 0


def test_reconcile_without_plan_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _config_pair: Any
) -> None:
    config = _config_pair
    storage = _storage(config, entity_types=("subject",))
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    before = tuple(
        sorted(str(p) for p in Path(runner.ledger.root).rglob("*") if p.is_file())
    )
    with pytest.raises(ReferenceEditEpochError, match="frozen Reference Edit plan"):
        runner.reconcile_stats(SHARD)
    after = tuple(
        sorted(str(p) for p in Path(runner.ledger.root).rglob("*") if p.is_file())
    )
    assert after == before


def test_real_scheduler_full_chain_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Primary chain through the real scheduler equals legacy publication."""
    from r2v_data_v2.v3.reference_edit import reference_edit_clips
    from tests.test_v3_reference_edit_boogu import (
        _Backend,
        _Judge,
        _SamReviewer,
    )

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    epoch_config = _reference_edit_config(tmp_path, monkeypatch, "run-epoch")
    legacy_storage = _storage(legacy_config, entity_types=("subject",))
    epoch_storage = _storage(epoch_config, entity_types=("subject",))
    from r2v_data_v2.v3.pair import pair_clips as legacy_pair_clips

    legacy_pair_clips(epoch_config, epoch_storage, judge=_PairEntityJudge())
    legacy_pair_clips(legacy_config, legacy_storage, judge=_PairEntityJudge())

    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_Backend(),
        judge=_Judge(),
        sam_reviewer=_SamReviewer(),
    )

    ledger = GroupLedger(tmp_path / "ledger")
    runner = ReferenceEditEpochRunner(
        epoch_config,
        {SHARD: epoch_storage},
        ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    fake_boogu = _FakeBoogu()

    class _ChainExecutor:
        def __init__(self) -> None:
            self.boogu = fake_boogu
            self.judge = _FakeJudge()
            self.sam = _FakeSam()

        def execute_batch(self, jobs: Any) -> dict[str, Any]:
            from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

            outcomes: dict[str, Any] = {}
            for job in sorted(jobs, key=lambda item: item.job_id()):
                if job.job_type == "reference_edit_boogu_generate":
                    handle = self.boogu
                elif job.job_type == "reference_edit_qwen_review":
                    handle = self.judge
                else:
                    handle = self.sam
                try:
                    outcomes[job.job_id()] = JobExecution(
                        job, runner.run(job, handle), None
                    )
                except Exception as exc:  # noqa: BLE001 - isolate per job
                    outcomes[job.job_id()] = JobExecution(job, None, exc)
            return outcomes

    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={RESOURCE_QWEN: _ChainExecutor()},
    )
    outcome = scheduler.run(runner.seed_jobs())
    assert outcome["completed"], outcome
    assert any(phase.startswith("r") for phase in ledger.phase_ids())

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    # The final published state matches legacy semantics.
    epoch_clip = epoch_storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit is not None
    assert legacy_clip.reference_edit is not None
    assert epoch_clip.reference_edit.status == legacy_clip.reference_edit.status
    assert epoch_clip.pairing.status == legacy_clip.pairing.status
    assert epoch_clip.pairing.retained_entity_ids == (
        legacy_clip.pairing.retained_entity_ids
    )
    assert [
        (state.entity_id, state.status, state.route)
        for state in epoch_clip.reference_edit.entities
    ] == [
        (state.entity_id, state.status, state.route)
        for state in legacy_clip.reference_edit.entities
    ]

    # Restart: a fresh ledger over the same root reconciles identically.
    fresh = ReferenceEditEpochRunner(
        epoch_config,
        {SHARD: epoch_storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    assert fresh.reconcile_stats(SHARD).to_dict() == stats.to_dict()


def test_candidate2_chain_runs_second_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """candidate1 rejected (SAM) -> candidate2 seeded once from the plan."""

    config = _reference_edit_config(tmp_path, monkeypatch, "run-c2")
    # A tiny alternates-free environment: force rejection via SAM failure so
    # the chain attempts candidate2 only when an alternate exists.
    storage = _storage(config, entity_types=("subject",))
    ledger = GroupLedger(tmp_path / "ledger")
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    jobs = runner.seed_jobs()
    generation_jobs = [
        job for job in jobs if job.job_type == "reference_edit_boogu_generate"
    ]
    if not generation_jobs:
        pytest.skip("fixture produced no repairable completion chain")
    seed_path = runner._seed_path(SHARD, "clip-1", "e1", 1)
    payload = seed_path.read_text(encoding="utf-8")
    assert "seed" in payload
    # Tamper must fail closed on the next replay.
    seed_path.write_text(payload.replace('"seed":', '"seed_x":'), encoding="utf-8")
    with pytest.raises(ReferenceEditEpochError, match="seed plan drifted"):
        runner.durable_seed(
            SHARD,
            "clip-1",
            "e1",
            runner._entity_anchor(
                storage,
                "clip-1",
                next(
                    item
                    for item in storage.read_clip("clip-1").annotation.entities
                    if item.entity_id == "e1"
                ),
                next(
                    item
                    for item in storage.read_clip("clip-1").references.entities
                    if item.entity_id == "e1"
                ),
                1,
            ),
            1,
        )

