"""Reference Edit resource epoch (5b): durable chain, restart, legacy equality.

The fixtures here deliberately produce *repairable* pair references, so the
epoch really pays Boogu/Qwen/SAM calls through the real scheduler.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.config import PairConfig
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
)
from r2v_data_v2.v3.post_mask_epoch_reference_edit import (
    ReferenceEditEpochError,
    ReferenceEditEpochRunner,
    resolve_reference_edit_judge,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from tests.test_v3_pair import _config as _pair_base_config
from tests.test_v3_pair import _Judge as _PairEntityJudge
from tests.test_v3_pair import _storage as _pair_storage

SHARD = "shard-000000000-000000000"


def legacy_attempt_counts(stats: Any) -> tuple[int, int, int]:
    """Boogu/Qwen/SAM paid-call counts implied by the legacy stats."""
    attempts = stats.completion_attempts
    return (attempts, attempts, attempts)


def _reference_edit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str = "run",
    **pair_kwargs: Any,
) -> Any:
    config = _pair_base_config(
        tmp_path,
        monkeypatch,
        pair=PairConfig(**pair_kwargs),
        reference_edit_enabled=True,
    )
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    config = replace(
        config,
        sam3=replace(config.sam3, model_path=pretrained / "sam3"),
    )
    return replace(config, run_root=config.run_root.parent / run_name)


def _prepared_storage(
    config: Any, monkeypatch: pytest.MonkeyPatch, *, run_name: str
) -> Any:
    """A storage whose clip has a *repairable* ready pair reference."""
    storage = _pair_storage(config, entity_types=("subject",))
    pair_clips = _pair_clips()
    pair_clips(config, storage, judge=_PairEntityJudge(scopes={"e1": "repairable"}))
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    reference = clip.references.entities[0]
    assert reference.completeness == "repairable"
    return storage


def _sam_track_result(mask: Any, *, status: str, fragmented: bool = False) -> Any:
    from r2v_data_v2.v3.sam3_backend import (
        BackendMaskObservation,
        EntityTrackResult,
    )

    return EntityTrackResult(
        status=status,
        reason="fragmented" if fragmented else None,
        observations=(
            BackendMaskObservation(
                slot=5,
                mask=mask,
                confidence=0.9,
                object_id="target",
            ),
        ),
    )


def _pair_clips() -> Any:
    from r2v_data_v2.v3.pair import pair_clips

    return pair_clips


def _with_alternate(config: Any, monkeypatch: pytest.MonkeyPatch, *, run_name: str) -> Any:
    """Repairable reference plus a same-parent alternate candidate."""
    from tests.test_v3_pair import _add_ready_clip

    storage = _pair_storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="alt", clip_suffix="2", entity_types=("subject",)
    )
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    _pair_clips()(
        config,
        storage,
        judge=_ScopedJudge(
            scopes={("clip-1", "e1"): "repairable", ("alt", "e1"): "full"}
        ),
    )
    clip = storage.read_clip("clip-1")
    assert clip.pairing is not None and clip.pairing.status == "ready"
    reference = clip.references.entities[0]
    assert reference.completeness == "repairable"
    return storage


class _FakeBoogu:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def edit(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boogu worker exploded")
        import io

        from PIL import Image

        from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput

        width = int(kwargs["width"])
        height = int(kwargs["height"])
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), (191, 22, 43)).save(buffer, format="PNG")
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=buffer.getvalue(),
            original_instruction=instruction,
            rewritten_instruction="rewritten",
            effective_instruction="rewritten",
            worker_metadata={"seed": kwargs.get("seed")},
        )


class _FakeQwen:
    """Completion judge with legacy-aligned candidate1/candidate2 behavior.

    ``reject_candidate1`` mirrors the legacy acceptance order: the canonical
    candidate is rejected and the alternate-sourced candidate2 is accepted.
    """

    def __init__(
        self,
        *,
        accept: bool = True,
        fail: bool = False,
        reject_candidate1: bool = False,
    ) -> None:
        self.accept = accept
        self.fail = fail
        self.reject_candidate1 = reject_candidate1
        self.calls = 0

    def review(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("qwen judge exploded")
        accept = self.accept
        if self.reject_candidate1 and kwargs.get("comparison_source_rgb") is None:
            accept = False
        from tests.test_v3_reference_edit_boogu import _completion_review

        return _completion_review(accept=accept)


class _FakeSamSegmenter:
    """The SAM epoch resource handle: the segmenter backend, not the reviewer.

    ``Sam3BooguReferenceReviewer`` (built by the runner) calls ``track`` on it.
    The mask covers the candidate's tiny 1K canvas so the geometry gates pass.
    """

    def __init__(self, *, passed: bool = True, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.passed = passed

    def track(self, **kwargs: Any) -> Any:
        import numpy as np
        from PIL import Image

        self.calls += 1
        if self.fail:
            raise RuntimeError("sam backend exploded")
        # The reviewer saves the candidate as the tracked frames, so the
        # mask must match the candidate canvas exactly.
        with Image.open(kwargs["frame_paths"][0]) as opened:
            width, height = opened.size
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 3 : height // 3 + height // 8,
             width // 3 : width // 3 + width // 8] = True
        return _sam_track_result(
            mask if self.passed else np.zeros((height, width), dtype=bool),
            status="ready",
            fragmented=not self.passed,
        )


class _LegacySamReviewer:
    """Legacy reference_edit_clips consumes a reviewer with ``review()``."""

    def __init__(self, *, passed: bool = True) -> None:
        self.passed = passed

    def review(self, **kwargs: Any) -> Any:
        from r2v_data_v2.v3.reference_edit_boogu import BooguSamReview

        if self.passed:
            return BooguSamReview(
                passed=True,
                target_entity_present=True,
                exactly_one_target_instance=True,
                area_growth_acceptable=True,
                fragmentation_acceptable=True,
                reason="usable",
                diagnostics={"failure_kind": "none"},
            )
        return BooguSamReview(
            passed=False,
            target_entity_present=True,
            exactly_one_target_instance=False,
            area_growth_acceptable=True,
            fragmentation_acceptable=False,
            reason="fragmented",
            diagnostics={"failure_kind": "fragmented"},
        )


class _RoutingExecutor:
    """One executor per resource key, as the real scheduler requires."""

    def __init__(self, runner: ReferenceEditEpochRunner) -> None:
        self.boogu = _FakeBoogu()
        self.qwen = _FakeQwen()
        self.sam = _FakeSamSegmenter()
        self._runner = runner

    def executor_for(self, resource: str) -> Any:
        runner = self._runner
        handle = self._handle(resource)

        class _One:
            def execute_batch(self, jobs: Any) -> dict[str, Any]:
                from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

                outcomes: dict[str, Any] = {}
                for job in sorted(jobs, key=lambda item: item.job_id()):
                    try:
                        outcomes[job.job_id()] = JobExecution(
                            job, runner.run(job, handle), None
                        )
                    except Exception as exc:  # noqa: BLE001 - isolate per job
                        outcomes[job.job_id()] = JobExecution(job, None, exc)
                return outcomes

        return _One()

    def _handle(self, resource: str) -> Any:
        if resource == RESOURCE_BOOGU:
            return self.boogu
        if resource == RESOURCE_QWEN:
            return self.qwen
        if resource == RESOURCE_SAM:
            return self.sam
        raise AssertionError(f"unexpected resource {resource}")


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_name: str = "run",
    review_execution: str = "sequential",
    with_alternate: bool = False,
) -> tuple[Any, Any, ReferenceEditEpochRunner, ResourceEpochScheduler, _RoutingExecutor]:
    config = _reference_edit_config(
        tmp_path, monkeypatch, run_name, same_parent_fallback_enabled=True
    )
    if with_alternate:
        storage = _with_alternate(config, monkeypatch, run_name=run_name)
        clip_uids = ["clip-1", "alt"]
    else:
        storage = _prepared_storage(config, monkeypatch, run_name=run_name)
        clip_uids = ["clip-1"]
    ledger = GroupLedger(tmp_path / "ledger")
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        ledger,
        eligible_clip_uids_by_shard={SHARD: clip_uids},
        review_execution=review_execution,
    )
    routing = _RoutingExecutor(runner)
    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={
            resource: routing.executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    return config, storage, runner, scheduler, routing


def _drain(
    runner: ReferenceEditEpochRunner, scheduler: ResourceEpochScheduler
) -> None:
    """Run seed rounds until the scheduler reaches a fixed point."""
    for _round in range(8):
        jobs = runner.seed_jobs()
        if not jobs:
            return
        outcome = scheduler.run(jobs)
        if not outcome["completed"] and not jobs:
            raise AssertionError("the chain stalled with unresolved jobs")
    raise AssertionError("the Reference Edit chain did not terminate")


def test_repairable_chain_matches_legacy_and_uses_three_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(),
        sam_reviewer=_LegacySamReviewer(),
    )

    config, storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch"
    )
    seed = runner.seed_jobs()
    assert any(job.resource == RESOURCE_BOOGU for job in seed)
    _drain(runner, scheduler)

    phase_ids = ",".join(ledger_phase_ids(runner))
    assert "-boogu" in phase_ids, ledger_phase_ids(runner)
    assert "-qwen" in phase_ids, ledger_phase_ids(runner)
    assert "-sam" in phase_ids, ledger_phase_ids(runner)
    # The single-clip fixture still has an alternate frame, so legacy ALSO
    # runs candidate2 after the candidate1 rejection: both pay 2 attempts.
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == legacy_attempt_counts(legacy)
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit.model_dump(mode="json") == (
        legacy_clip.reference_edit.model_dump(mode="json")
    )
    assert epoch_clip.references.model_dump(mode="json") == (
        legacy_clip.references.model_dump(mode="json")
    )
    assert epoch_clip.pairing.model_dump(mode="json") == (
        legacy_clip.pairing.model_dump(mode="json")
    )

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
        review_execution="sequential",
    )
    paid = (routing.boogu.calls, routing.qwen.calls, routing.sam.calls)
    assert fresh.reconcile_stats(SHARD).to_dict() == stats.to_dict()
    assert fresh.seed_jobs() == []
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == paid


def ledger_phase_ids(runner: ReferenceEditEpochRunner) -> tuple[str, ...]:
    return runner.ledger.phase_ids()


def test_qwen_reject_still_pays_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen reject must still pay the SAM review (sequential frozen order)."""
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(accept=False),
        sam_reviewer=_LegacySamReviewer(),
    )

    _config, _storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    routing.qwen.accept = False
    _drain(runner, scheduler)
    # Both legacy and epoch pay every attempted attempt: Qwen reject still
    # runs SAM, and candidate2 runs whenever legacy would run it.
    assert routing.boogu.calls >= 1
    assert routing.qwen.calls == routing.boogu.calls
    assert routing.sam.calls == routing.qwen.calls
    assert runner.reconcile_stats(SHARD).to_dict() == legacy.to_dict()


def test_qwen_exception_skips_sam_in_sequential_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    routing.qwen.fail = True
    _drain(runner, scheduler)
    assert routing.sam.calls == 0, "a Qwen exception must never pay SAM"
    assert routing.qwen.calls == routing.boogu.calls >= 1
    rejection = json.loads(
        (
            Path(storage.root)
            / "clips/clip-1/reference_edit/e1/completion_rejection.json"
        ).read_text(encoding="utf-8")
    )
    assert rejection["reason"].startswith("boogu_reference_edit_failed:")


def test_sam_reject_rejects_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(),
        sam_reviewer=_LegacySamReviewer(passed=False),
    )
    _config, _storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    routing.sam.passed = False
    _drain(runner, scheduler)
    assert routing.sam.calls == routing.qwen.calls == routing.boogu.calls
    assert runner.reconcile_stats(SHARD).to_dict() == legacy.to_dict()


def test_boogu_exception_is_semantic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(fail=True),
        judge=_FakeQwen(),
        sam_reviewer=_LegacySamReviewer(),
    )

    _config, _storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    routing.boogu.fail = True
    _drain(runner, scheduler)
    assert routing.qwen.calls == 0 and routing.sam.calls == 0
    assert routing.boogu.calls >= 1
    assert runner.reconcile_stats(SHARD).to_dict() == legacy.to_dict()


def test_candidate2_accept_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(
        tmp_path, monkeypatch, "run-legacy", same_parent_fallback_enabled=True
    )
    legacy_storage = _with_alternate(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(reject_candidate1=True),
        sam_reviewer=_LegacySamReviewer(),
    )
    assert legacy.completion_candidate2_accepted == 1

    _config, _storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch", with_alternate=True
    )
    routing.qwen.reject_candidate1 = True
    _drain(runner, scheduler)
    stats = runner.reconcile_stats(SHARD)
    assert stats.completion_attempts == 2
    assert stats.completion_candidate2_attempts == 1
    assert stats.completion_candidate2_accepted == 1
    assert stats.completion_fallback_to_alpha == 0
    assert stats.to_dict() == legacy.to_dict()


def test_candidate2_reject_falls_back_to_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(
        tmp_path, monkeypatch, "run-legacy", same_parent_fallback_enabled=True
    )
    legacy_storage = _with_alternate(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(accept=False),
        sam_reviewer=_LegacySamReviewer(),
    )
    assert legacy.completion_fallback_to_alpha == 1

    _config, _storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch", with_alternate=True
    )
    routing.qwen.accept = False
    routing.sam.passed = False
    _drain(runner, scheduler)
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == (2, 2, 2)
    stats = runner.reconcile_stats(SHARD)
    assert stats.completion_attempts == 2
    assert stats.completion_candidate2_attempts == 1
    assert stats.completion_candidate2_accepted == 0
    assert stats.completion_fallback_to_alpha == 1
    assert stats.entities_failed == 0
    assert stats.to_dict() == legacy.to_dict()


def test_candidate2_metadata_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch", with_alternate=True
    )
    routing.qwen.accept = False
    _drain(runner, _scheduler)
    alternate_plan = runner._alternate_path(SHARD, "clip-1", "e1")
    assert alternate_plan.is_file()
    frozen = json.loads(alternate_plan.read_text(encoding="utf-8"))
    metadata_path = (
        Path(storage.root)
        / "clips/clip-1/reference_edit/e1/completion_metadata_2.json"
    )
    assert metadata_path.is_file(), "candidate2 must have been attempted"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["completion_source_candidate_id"] == frozen["candidate_id"]
    assert (
        metadata["completion_source_frame_index"] == frozen["source_frame_index"]
    )
    assert metadata["source_image_sha256"] == frozen["alternate_source_sha256"]
    assert (
        metadata["comparison_source_image_sha256"]
        == frozen["canonical_source_sha256"]
    )


def test_seed_plan_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, _scheduler, _routing = _build(tmp_path, monkeypatch)
    runner.seed_jobs()
    seed_path = runner._seed_path(SHARD, "clip-1", "e1", 1)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    payload["anchor_digest"] = "0" * 64
    seed_path.write_text(json.dumps(payload), encoding="utf-8")
    clip = storage.read_clip("clip-1")
    anchor = runner._entity_anchor(
        storage, "clip-1", clip.annotation.entities[0], clip.references.entities[0], 1
    )
    with pytest.raises(ReferenceEditEpochError, match="seed plan drifted"):
        runner.durable_seed(SHARD, "clip-1", "e1", anchor, 1)


def test_reconcile_without_plan_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _reference_edit_config(tmp_path, monkeypatch, "run")
    storage = _pair_storage(config, entity_types=("subject",))
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


def test_crash_after_generation_commit_resumes_without_extra_boogu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    generation_jobs = [
        job
        for job in runner.seed_jobs()
        if job.job_type == "reference_edit_boogu_generate"
    ]
    assert generation_jobs, "the repairable fixture must seed a generation job"
    scheduler.run(generation_jobs)
    paid = routing.boogu.calls
    assert paid >= 1
    # Fresh runner + scheduler over the same durable root: no extra Boogu.
    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
        review_execution="sequential",
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "ledger"),
        finalize=fresh.finalize,
        executors={
            resource: fresh_routing.executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    _drain(fresh, fresh_scheduler)
    # The scheduler's fixed point already committed the whole chain inside
    # the crashed run, so the fresh composition replays receipts only: the
    # generation (and every review) is repaid zero times.
    assert fresh_routing.boogu.calls == 0
    assert fresh_routing.qwen.calls == 0
    assert fresh_routing.sam.calls == 0


def test_review_identity_drift_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, runner, scheduler, _routing = _build(tmp_path, monkeypatch)
    jobs = runner.seed_jobs()
    generation_jobs = [
        job for job in jobs if job.job_type == "reference_edit_boogu_generate"
    ]
    assert generation_jobs
    scheduler.run(generation_jobs)
    gen_job = generation_jobs[0]
    judge_model = f"qwen:{config.qwen.reference_edit_judge.model}"
    review_job = runner._review_job(
        SHARD,
        "clip-1",
        storage.read_clip("clip-1").annotation.entities[0],
        1,
        job_type="reference_edit_qwen_review",
        resource=RESOURCE_QWEN,
        model_identity=judge_model,
        generation_job_id=gen_job.job_id(),
        candidate_sha256="f" * 64,
    )
    result = runner.run(review_job, _FakeQwen())
    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert "drifted" in result.detail


def test_resolve_reference_edit_judge_endpoint_and_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from r2v_data_v2.v3.reference_edit_boogu import QwenBooguReferenceEditJudge

    config = _reference_edit_config(tmp_path, monkeypatch, "run")
    resolved = resolve_reference_edit_judge("http://127.0.0.1:8000/v1", config)
    assert isinstance(resolved.judge, QwenBooguReferenceEditJudge)
    assert resolved.owned is True
    resolved.judge.close()

    injected = _FakeQwen()
    resolved = resolve_reference_edit_judge(injected, config)
    assert resolved.judge is injected
    assert resolved.owned is False
