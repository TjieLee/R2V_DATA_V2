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
    ReferenceEditDurableError,
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


# ---------------------------------------------------------------------------
# parallel_independent: generation unlocks BOTH reviews; attempt finalized once
# ---------------------------------------------------------------------------


class _BoundaryExecutor:
    """Executes only allowed job types; the rest hit a simulated crash.

    The jobs that DO run are committed by the real scheduler into real
    ``rXXX-<resource>`` phases, so the durable layout is production-shaped.
    """

    def __init__(
        self,
        runner: ReferenceEditEpochRunner,
        routing: _RoutingExecutor,
        allowed: tuple[str, ...],
    ) -> None:
        self.runner = runner
        self.routing = routing
        self.allowed = allowed

    def executor_for(self, resource: str) -> Any:
        runner = self.runner
        routing = self.routing
        allowed = self.allowed

        class _One:
            def execute_batch(self, jobs: Any) -> dict[str, Any]:
                from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

                outcomes: dict[str, Any] = {}
                for job in sorted(jobs, key=lambda item: item.job_id()):
                    if job.job_type in allowed:
                        handle = routing._handle(job.resource)
                        outcomes[job.job_id()] = JobExecution(
                            job, runner.run(job, handle), None
                        )
                    else:
                        outcomes[job.job_id()] = JobExecution(
                            job, None, RuntimeError("simulated crash boundary")
                        )
                return outcomes

        del resource
        return _One()


def _boundary_scheduler(
    runner: ReferenceEditEpochRunner,
    routing: _RoutingExecutor,
    allowed: tuple[str, ...],
) -> ResourceEpochScheduler:
    boundary = _BoundaryExecutor(runner, routing, allowed)
    return ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={
            resource: boundary.executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )


def test_parallel_independent_generation_unlocks_both_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generation finalize must plan qwen AND sam against the same generation."""
    _config, _storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", review_execution="parallel_independent"
    )
    boundary = _boundary_scheduler(
        runner, routing, ("reference_edit_boogu_generate",)
    )
    boundary.run(runner.seed_jobs())
    records = runner._plan_record_index()
    qwen_records = [
        r for r in records.values() if r["job_type"] == "reference_edit_qwen_review"
    ]
    sam_records = [
        r for r in records.values() if r["job_type"] == "reference_edit_sam_review"
    ]
    assert qwen_records and sam_records
    assert {dict(r["dependency_digests"])["generation"] for r in qwen_records} == {
        dict(r["dependency_digests"])["generation"] for r in sam_records
    }


def test_parallel_independent_normal_matches_legacy(
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
        review_execution="parallel_independent",
    )

    _config, storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch",
        review_execution="parallel_independent",
    )
    _drain(runner, scheduler)
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == (
        legacy.completion_attempts,
    ) * 3
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit.model_dump(mode="json") == (
        legacy_clip.reference_edit.model_dump(mode="json")
    )
    # Exactly one attempt outcome exists: the CPU finalize ran once.
    assert runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1).is_file()
    assert not runner._attempt_outcome_path(SHARD, "clip-1", "e1", 2).is_file() or (
        stats.completion_attempts == 2
    )


@pytest.mark.parametrize("sam_passed", (True, False))
def test_parallel_independent_qwen_failure_still_pays_sam_but_discards_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sam_passed: bool
) -> None:
    """Qwen exception wins and SAM stays paid but out of publication input."""
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(fail=True),
        sam_reviewer=_LegacySamReviewer(passed=sam_passed),
        review_execution="parallel_independent",
    )

    _config, storage, runner, scheduler, routing = _build(
        tmp_path,
        monkeypatch,
        run_name="run-epoch",
        review_execution="parallel_independent",
    )
    routing.qwen.fail = True
    routing.sam.passed = sam_passed
    _drain(runner, scheduler)

    # Parallel execution still pays SAM even though Qwen's exception is the
    # frozen publication provenance.
    assert routing.sam.calls >= 1
    assert routing.qwen.calls == routing.boogu.calls
    assert routing.sam.calls == routing.qwen.calls

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit.model_dump(mode="json") == (
        legacy_clip.reference_edit.model_dump(mode="json")
    )

    legacy_metadata = json.loads(
        (
            Path(legacy_storage.root)
            / "clips/clip-1/reference_edit/e1/completion_metadata.json"
        ).read_text(encoding="utf-8")
    )
    epoch_metadata = _read_completion_metadata(runner)
    assert epoch_metadata is not None
    assert legacy_metadata["qwen_review"] is None
    assert legacy_metadata["sam_review"] is None
    assert epoch_metadata["qwen_review"] is None
    assert epoch_metadata["sam_review"] is None

    marker = _read_attempt_marker(runner, 1)
    assert marker is not None
    assert marker["status"] == "qwen_failed"
    assert marker["rejection_reason"] == (
        "boogu_reference_edit_failed: qwen judge exploded"
    )


def test_parallel_independent_sam_failure_keeps_qwen_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAM exception preserves the successful Qwen review, matching legacy."""
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    class _FailingLegacySamReviewer:
        def review(self, **kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError("sam backend exploded")

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(),
        sam_reviewer=_FailingLegacySamReviewer(),
        review_execution="parallel_independent",
    )

    _config, storage, runner, scheduler, _routing = _build(
        tmp_path,
        monkeypatch,
        run_name="run-epoch",
        review_execution="parallel_independent",
    )
    sam_fail_calls = 0

    def exploding_epoch_sam_review(*args: Any, **kwargs: Any) -> Any:
        nonlocal sam_fail_calls
        del args, kwargs
        sam_fail_calls += 1
        raise RuntimeError("sam backend exploded")

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_reference_edit.run_boogu_sam_review",
        exploding_epoch_sam_review,
    )
    _drain(runner, scheduler)

    sam_job_id = _committed_job_id(runner, "reference_edit_sam_review")
    _job, _result, sam_payload = runner._validated_committed(sam_job_id)
    assert sam_payload["status"] == "sam_failed"
    assert sam_payload["error"] == "sam backend exploded"
    assert sam_fail_calls >= 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit.model_dump(mode="json") == (
        legacy_clip.reference_edit.model_dump(mode="json")
    )

    legacy_metadata = json.loads(
        (
            Path(legacy_storage.root)
            / "clips/clip-1/reference_edit/e1/completion_metadata.json"
        ).read_text(encoding="utf-8")
    )
    epoch_metadata = _read_completion_metadata(runner)
    assert epoch_metadata is not None
    assert legacy_metadata["qwen_review"] is not None
    assert legacy_metadata["sam_review"] is None
    assert epoch_metadata["qwen_review"] == legacy_metadata["qwen_review"]
    assert epoch_metadata["sam_review"] is None

    # The frozen legacy wrapper never assigns ``sam_review`` when the SAM
    # future raises, and acceptance is the conjunction of the reviews that DID
    # resolve, so this attempt is ACCEPTED with ``sam_review`` unset -- the SAM
    # exception's override never applies to a published attempt. The durable
    # attempt marker must record exactly that.
    marker = _read_attempt_marker(runner, 1)
    assert marker is not None
    assert marker["status"] == "accepted"
    assert marker["accepted"] is True
    assert marker["rejection_reason"] != (
        "boogu_reference_edit_failed: sam backend exploded"
    )


def test_parallel_independent_double_exception_matches_legacy_qwen_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both review failures keep the legacy Qwen-first exception provenance."""
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    class _FailingLegacySamReviewer:
        def review(self, **kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError("sam backend exploded")

    legacy_config = _reference_edit_config(tmp_path, monkeypatch, "run-legacy")
    legacy_storage = _prepared_storage(
        legacy_config, monkeypatch, run_name="run-legacy"
    )
    legacy = reference_edit_clips(
        legacy_config,
        legacy_storage,
        backend=_FakeBoogu(),
        judge=_FakeQwen(fail=True),
        sam_reviewer=_FailingLegacySamReviewer(),
        review_execution="parallel_independent",
    )

    _config, storage, runner, scheduler, routing = _build(
        tmp_path,
        monkeypatch,
        run_name="run-epoch",
        review_execution="parallel_independent",
    )
    routing.qwen.fail = True
    sam_fail_calls = 0

    def exploding_epoch_sam_review(*args: Any, **kwargs: Any) -> Any:
        nonlocal sam_fail_calls
        del args, kwargs
        sam_fail_calls += 1
        raise RuntimeError("sam backend exploded")

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_reference_edit.run_boogu_sam_review",
        exploding_epoch_sam_review,
    )
    _drain(runner, scheduler)

    qwen_job_id = _committed_job_id(runner, "reference_edit_qwen_review")
    _job, _result, qwen_payload = runner._validated_committed(qwen_job_id)
    assert qwen_payload["status"] == "qwen_failed"
    assert qwen_payload["error"] == "qwen judge exploded"
    sam_job_id = _committed_job_id(runner, "reference_edit_sam_review")
    _job, _result, sam_payload = runner._validated_committed(sam_job_id)
    assert sam_payload["status"] == "sam_failed"
    assert sam_payload["error"] == "sam backend exploded"
    assert sam_fail_calls >= 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
    epoch_clip = storage.read_clip("clip-1")
    legacy_clip = legacy_storage.read_clip("clip-1")
    assert epoch_clip.reference_edit.model_dump(mode="json") == (
        legacy_clip.reference_edit.model_dump(mode="json")
    )

    legacy_rejection = json.loads(
        (
            Path(legacy_storage.root)
            / "clips/clip-1/reference_edit/e1/completion_rejection.json"
        ).read_text(encoding="utf-8")
    )
    epoch_rejection = json.loads(
        (
            Path(storage.root)
            / "clips/clip-1/reference_edit/e1/completion_rejection.json"
        ).read_text(encoding="utf-8")
    )
    expected_reason = "boogu_reference_edit_failed: qwen judge exploded"
    assert legacy_rejection["reason"] == expected_reason
    assert epoch_rejection["reason"] == legacy_rejection["reason"]

    marker = _read_attempt_marker(runner, 1)
    assert marker is not None
    assert marker["status"] == "qwen_failed"
    assert marker["rejection_reason"] == expected_reason


def test_parallel_independent_finalize_crash_replays_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review receipts committed, attempt outcome not written: replay CPU-only."""
    config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run",
        review_execution="parallel_independent",
    )
    # Simulate a crash between the review commits and the CPU finalize by
    # swallowing the finalizer for review jobs.
    real_finalize = runner.finalize

    def crashing_finalize(job: Any, result: Any) -> Any:
        if job.job_type in (
            "reference_edit_qwen_review",
            "reference_edit_sam_review",
        ):
            return ()
        return real_finalize(job, result)

    runner.finalize = crashing_finalize
    boundary = ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={
            resource: _BoundaryExecutor(
                runner, routing, ("reference_edit_boogu_generate",)
            ).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    # Let everything run except the review finalizers.
    boundary = _BoundaryExecutor(
        runner, routing, ("reference_edit_boogu_generate",)
    )
    full = ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={
            resource: _FullExecutor(runner, routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    del boundary
    full.run(runner.seed_jobs())
    assert routing.qwen.calls >= 1 and routing.sam.calls >= 1
    assert not runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1).is_file()

    # Fresh runner with the real finalizer: zero extra model calls.
    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
        review_execution="parallel_independent",
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={
            resource: _FullExecutor(fresh, fresh_routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    fresh_scheduler.run(fresh.seed_jobs())
    assert fresh_routing.boogu.calls == 0
    assert fresh_routing.qwen.calls == 0
    assert fresh_routing.sam.calls == 0
    assert runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1).is_file()
    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is not None


def test_parallel_independent_restart_after_attempt_marker_replays_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash after the attempt marker but before _after_attempt completes.

    The marker is written BEFORE ``_after_attempt``, so a restart must not
    treat it as "the continuation already ran": with no entity outcome and no
    candidate2 unlocked yet, the attempt has to be re-finalized (CPU only) so
    candidate2 is re-unlocked and the entity chain can finish.
    """
    from r2v_data_v2.v3.reference_edit import reference_edit_clips

    # Legacy reference for the same parallel candidate1-reject/candidate2-accept
    # chain, so this single test also locks full legacy equality.
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
        review_execution="parallel_independent",
    )

    config, storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch", with_alternate=True,
        review_execution="parallel_independent",
    )
    routing.qwen.reject_candidate1 = True

    # Crash exactly after _write_attempt_outcome: the first _after_attempt call
    # raises, so the marker is durable while the continuation is not.
    real_after_attempt = runner._after_attempt
    crashes = 0

    def crashing_after_attempt(*args: Any, **kwargs: Any) -> Any:
        nonlocal crashes
        crashes += 1
        if crashes == 1:
            raise RuntimeError("crash after the attempt outcome marker")
        return real_after_attempt(*args, **kwargs)

    runner._after_attempt = crashing_after_attempt  # type: ignore[method-assign]
    scheduler.run(runner.seed_jobs())

    # The exact crash window: marker durable, continuation not.
    first_marker = runner._validated_attempt_outcome(SHARD, "clip-1", "e1", 1)
    assert first_marker is not None
    assert first_marker["accepted"] is False
    assert runner._entity_outcome(SHARD, "clip-1", "e1") is None
    assert not runner._attempt_outcome_path(SHARD, "clip-1", "e1", 2).is_file()
    # Candidate1 really paid one Boogu, one Qwen and one SAM call.
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == (1, 1, 1)

    # Restart on the same durable state with the real continuation.
    runner._after_attempt = real_after_attempt  # type: ignore[method-assign]
    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "alt"]},
        review_execution="parallel_independent",
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={
            resource: _FullExecutor(fresh, fresh_routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    _drain(fresh, fresh_scheduler)

    # Only candidate2 paid: attempt1's receipts were reused, its continuation
    # was replayed by CPU, and candidate2 was re-unlocked.
    assert (fresh_routing.boogu.calls, fresh_routing.qwen.calls, fresh_routing.sam.calls) == (
        1,
        1,
        1,
    )

    first_marker = fresh._validated_attempt_outcome(SHARD, "clip-1", "e1", 1)
    second_marker = fresh._validated_attempt_outcome(SHARD, "clip-1", "e1", 2)
    assert first_marker is not None and first_marker["accepted"] is False
    assert second_marker is not None and second_marker["accepted"] is True
    entity_outcome = fresh._entity_outcome(SHARD, "clip-1", "e1")
    assert entity_outcome is not None
    assert entity_outcome["outcome"] == "accepted"
    assert entity_outcome["attempt_index"] == 2
    epoch_clip = storage.read_clip("clip-1")
    assert epoch_clip.reference_edit is not None
    assert epoch_clip.reference_edit.status == "ready"

    # Full legacy equality for the same chain.
    stats = fresh.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy.to_dict()
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


class _FullExecutor:
    """Runs every job through the routing handles."""

    def __init__(self, runner: ReferenceEditEpochRunner, routing: _RoutingExecutor) -> None:
        self.runner = runner
        self.routing = routing

    def executor_for(self, resource: str) -> Any:
        runner = self.runner
        routing = self.routing

        class _One:
            def execute_batch(self, jobs: Any) -> dict[str, Any]:
                from r2v_data_v2.v3.post_mask_epoch_scheduler import JobExecution

                outcomes: dict[str, Any] = {}
                for job in sorted(jobs, key=lambda item: item.job_id()):
                    handle = routing._handle(job.resource)
                    outcomes[job.job_id()] = JobExecution(
                        job, runner.run(job, handle), None
                    )
                return outcomes

        del resource
        return _One()


def _read_attempt_marker(runner: ReferenceEditEpochRunner, attempt: int) -> Any:
    path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", attempt)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_completion_metadata(runner: ReferenceEditEpochRunner) -> Any:
    path = (
        Path(runner._storage_for(SHARD).root)
        / "clips/clip-1/reference_edit/e1/completion_metadata.json"
    )
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# True crash windows (receipts land in real rXXX-<resource> phases)
# ---------------------------------------------------------------------------


def test_crash_after_generation_receipt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation committed, Qwen/SAM never executed."""
    config, storage, runner, _scheduler, routing = _build(tmp_path, monkeypatch)
    boundary = _boundary_scheduler(
        runner, routing, ("reference_edit_boogu_generate",)
    )
    outcome = boundary.run(runner.seed_jobs())
    assert not outcome["completed"]
    phase_ids = ",".join(runner.ledger.phase_ids())
    assert "-boogu" in phase_ids
    assert routing.boogu.calls == 1
    assert routing.qwen.calls == 0 and routing.sam.calls == 0

    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={
            resource: _FullExecutor(fresh, fresh_routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    _drain(fresh, fresh_scheduler)
    assert fresh_routing.boogu.calls == 0
    assert fresh_routing.qwen.calls >= 1
    assert fresh_routing.sam.calls >= 1


def test_crash_after_qwen_receipt_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen committed, SAM never executed (sequential)."""
    config, storage, runner, _scheduler, routing = _build(tmp_path, monkeypatch)
    boundary = _boundary_scheduler(
        runner, routing,
        ("reference_edit_boogu_generate", "reference_edit_qwen_review"),
    )
    outcome = boundary.run(runner.seed_jobs())
    assert not outcome["completed"]
    phase_ids = ",".join(runner.ledger.phase_ids())
    assert "-boogu" in phase_ids and "-qwen" in phase_ids
    assert routing.qwen.calls == 1 and routing.sam.calls == 0

    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={
            resource: _FullExecutor(fresh, fresh_routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    _drain(fresh, fresh_scheduler)
    assert fresh_routing.boogu.calls == 0
    assert fresh_routing.qwen.calls == 0
    assert fresh_routing.sam.calls >= 1


def test_crash_after_sam_receipt_before_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAM committed, attempt CPU outcome not written."""
    config, storage, runner, _scheduler, routing = _build(tmp_path, monkeypatch)
    real_finalize = runner.finalize

    def crashing_finalize(job: Any, result: Any) -> Any:
        if job.job_type == "reference_edit_sam_review":
            return ()
        return real_finalize(job, result)

    runner.finalize = crashing_finalize
    boundary = ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=runner.finalize,
        executors={
            resource: _FullExecutor(runner, routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    # The scheduler reports completed (all jobs committed), but the attempt
    # outcome was never written: exactly the crash window under test.
    boundary.run(runner.seed_jobs())
    assert (routing.boogu.calls, routing.qwen.calls, routing.sam.calls) == (1, 1, 1)
    assert not runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1).is_file()

    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    fresh_routing = _RoutingExecutor(fresh)
    fresh_scheduler = ResourceEpochScheduler(
        ledger=fresh.ledger,
        finalize=fresh.finalize,
        executors={
            resource: _FullExecutor(fresh, fresh_routing).executor_for(resource)
            for resource in (RESOURCE_BOOGU, RESOURCE_QWEN, RESOURCE_SAM)
        },
    )
    _drain(fresh, fresh_scheduler)
    assert (fresh_routing.boogu.calls, fresh_routing.qwen.calls, fresh_routing.sam.calls) == (0, 0, 0)
    assert fresh._attempt_outcome_path(SHARD, "clip-1", "e1", 1).is_file()
    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is not None


def test_publication_before_marker_crash_repairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clip published, outcome marker missing: verify + back-fill, zero calls."""
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    _drain(runner, scheduler)
    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is not None
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    assert marker_path.is_file()
    expected_delta = json.loads(marker_path.read_text(encoding="utf-8"))["delta"]
    marker_path.unlink()

    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    # seed_jobs re-verifies the published clip and repairs the marker.
    fresh.seed_jobs()
    assert marker_path.is_file()
    repaired = json.loads(marker_path.read_text(encoding="utf-8"))
    assert repaired["delta"] == expected_delta
    stats = fresh.reconcile_stats(SHARD)
    assert stats.processed == 1
    del routing


def test_marker_exists_live_state_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marker present but the live published state was tampered: fail closed."""
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    _drain(runner, scheduler)
    del routing
    clip = storage.read_clip("clip-1")
    tampered_entities = [
        entity.model_copy(update={"metadata_path": "clips/tampered.json"})
        if index == 0
        else entity
        for index, entity in enumerate(clip.reference_edit.entities)
    ]
    tampered = clip.reference_edit.model_copy(
        update={"entities": tampered_entities}
    )
    import r2v_data_v2.v3.storage as storage_module

    storage_module.write_json_atomic(
        storage.clip_path("clip-1"),
        clip.model_copy(update={"reference_edit": tampered}).model_dump(mode="json"),
    )
    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    with pytest.raises(ReferenceEditEpochError, match="does not match"):
        fresh.seed_jobs()


# ---------------------------------------------------------------------------
# Alternate plan: create-once and tamper
# ---------------------------------------------------------------------------


def test_alternate_artifact_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", with_alternate=True
    )
    routing.qwen.accept = False
    # Run until candidate1 is rejected and the alternate plan is frozen.
    _drain_run_until_alternate(runner, _scheduler, routing)
    alternate_path = runner._alternate_path(SHARD, "clip-1", "e1")
    assert alternate_path.is_file()
    frozen = json.loads(alternate_path.read_text(encoding="utf-8"))
    artifact = Path(storage.root) / frozen["alternate_source_path"]
    original = artifact.read_bytes()
    artifact.write_bytes(b"tampered")
    clip = storage.read_clip("clip-1")
    with pytest.raises(ReferenceEditEpochError, match="alternate"):
        runner.durable_alternate(
            SHARD, storage, "clip-1", clip.annotation.entities[0],
            clip.references.entities[0],
        )
    # Not regenerated: the tampered bytes stay on disk.
    assert artifact.read_bytes() == b"tampered"
    # Restore for cleanliness.
    artifact.write_bytes(original)


def test_alternate_plan_candidate_id_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen plan whose candidate identity was tampered fails closed."""
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", with_alternate=True
    )
    routing.qwen.accept = False
    _drain_run_until_alternate(runner, _scheduler, routing)
    alternate_path = runner._alternate_path(SHARD, "clip-1", "e1")
    frozen = json.loads(alternate_path.read_text(encoding="utf-8"))
    frozen["candidate_id"] = "candidate_99"
    alternate_path.write_text(json.dumps(frozen), encoding="utf-8")
    clip = storage.read_clip("clip-1")
    with pytest.raises(ReferenceEditEpochError, match="metadata candidate_id"):
        runner.durable_alternate(
            SHARD, storage, "clip-1", clip.annotation.entities[0],
            clip.references.entities[0],
        )


def test_alternate_plan_source_frame_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", with_alternate=True
    )
    routing.qwen.accept = False
    _drain_run_until_alternate(runner, _scheduler, routing)
    alternate_path = runner._alternate_path(SHARD, "clip-1", "e1")
    frozen = json.loads(alternate_path.read_text(encoding="utf-8"))
    frozen["source_frame_index"] = int(frozen["source_frame_index"]) + 1
    alternate_path.write_text(json.dumps(frozen), encoding="utf-8")
    clip = storage.read_clip("clip-1")
    with pytest.raises(
        ReferenceEditEpochError, match="metadata source_frame_index"
    ):
        runner.durable_alternate(
            SHARD, storage, "clip-1", clip.annotation.entities[0],
            clip.references.entities[0],
        )


def test_alternate_metadata_identity_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy materialized metadata is independent evidence of the plan."""
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", with_alternate=True
    )
    routing.qwen.accept = False
    _drain_run_until_alternate(runner, _scheduler, routing)
    alternate_path = runner._alternate_path(SHARD, "clip-1", "e1")
    frozen = json.loads(alternate_path.read_text(encoding="utf-8"))
    metadata_path = (
        Path(storage.root)
        / Path(frozen["alternate_source_path"]).with_name("alternate_source_2.json")
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["candidate_id"] = "candidate_99"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    clip = storage.read_clip("clip-1")
    with pytest.raises(ReferenceEditEpochError, match="metadata candidate_id"):
        runner.durable_alternate(
            SHARD, storage, "clip-1", clip.annotation.entities[0],
            clip.references.entities[0],
        )


def test_alternate_source_image_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original candidate image is re-hashed on every rebuild."""
    _config, storage, runner, _scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run", with_alternate=True
    )
    routing.qwen.accept = False
    _drain_run_until_alternate(runner, _scheduler, routing)
    alternate_path = runner._alternate_path(SHARD, "clip-1", "e1")
    frozen = json.loads(alternate_path.read_text(encoding="utf-8"))
    source_image = Path(storage.root) / frozen["source_image_path"]
    source_image.write_bytes(b"tampered source")
    clip = storage.read_clip("clip-1")
    with pytest.raises(ReferenceEditEpochError, match="source image drifted"):
        runner.durable_alternate(
            SHARD, storage, "clip-1", clip.annotation.entities[0],
            clip.references.entities[0],
        )


def _drain_run_until_alternate(
    runner: ReferenceEditEpochRunner, scheduler: ResourceEpochScheduler,
    routing: _RoutingExecutor,
) -> None:
    """Run until the candidate2 alternate plan exists (candidate1 rejected)."""
    for _round in range(8):
        if runner._alternate_path(SHARD, "clip-1", "e1").is_file():
            return
        jobs = runner.seed_jobs()
        if not jobs:
            return
        scheduler.run(jobs)
    raise AssertionError("alternate plan was never frozen")


# ---------------------------------------------------------------------------
# Corruption matrix (zero paid model calls during detection)
# ---------------------------------------------------------------------------


def _committed_job_id(runner: ReferenceEditEpochRunner, job_type: str) -> str:
    for record in runner._plan_record_index().values():
        if record.get("job_type") == job_type:
            job = runner._job_from_plan_record(record)
            if runner.ledger.classify(job).skippable:
                return job.job_id()
    raise AssertionError(f"no committed {job_type} job")


def _result_path(runner: ReferenceEditEpochRunner, job_id: str) -> Path:
    phase_id = runner._phase_for(job_id)
    return (
        Path(runner.ledger.root)
        / "phases"
        / phase_id
        / "artifacts"
        / job_id
        / "result.json"
    )


def _candidate_path(runner: ReferenceEditEpochRunner, job_id: str) -> Path:
    phase_id = runner._phase_for(job_id)
    return (
        Path(runner.ledger.root)
        / "phases"
        / phase_id
        / "artifacts"
        / job_id
        / "candidate.png"
    )


def test_generation_result_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, _scheduler, routing = _build(tmp_path, monkeypatch)
    # Stop after the generation receipt only.
    boundary = _boundary_scheduler(
        runner, routing, ("reference_edit_boogu_generate",)
    )
    boundary.run(runner.seed_jobs())
    job_id = _committed_job_id(runner, "reference_edit_boogu_generate")
    path = _result_path(runner, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result_payload"]["candidate_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceEditEpochError, match="receipt mismatch|drifted"):
        runner._validated_committed(job_id)


def test_candidate_artifact_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, _storage, runner, _scheduler, routing = _build(tmp_path, monkeypatch)
    boundary = _boundary_scheduler(
        runner, routing, ("reference_edit_boogu_generate",)
    )
    boundary.run(runner.seed_jobs())
    job_id = _committed_job_id(runner, "reference_edit_boogu_generate")
    path = _candidate_path(runner, job_id)
    path.write_bytes(b"tampered")
    with pytest.raises(ReferenceEditEpochError, match="receipt mismatch|drifted"):
        runner._validated_committed(job_id)


def _tamper_review_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    job_type: str,
) -> None:
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    # Commit generation + the target review, then tamper its result. In
    # sequential mode the SAM job only exists after Qwen finalizes, so allow
    # the whole chain; the tamper check reads the receipt directly.
    allowed = (
        "reference_edit_boogu_generate",
        "reference_edit_qwen_review",
        "reference_edit_sam_review",
    )
    boundary = _boundary_scheduler(runner, routing, allowed)
    boundary.run(runner.seed_jobs())
    job_id = _committed_job_id(runner, job_type)
    path = _result_path(runner, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result_payload"]["status"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReferenceEditEpochError, match="receipt mismatch|drifted"):
        runner._validated_committed(job_id)
    del config, storage, scheduler


def test_qwen_result_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tamper_review_result(
        tmp_path, monkeypatch, "reference_edit_qwen_review"
    )


def test_sam_result_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tamper_review_result(
        tmp_path, monkeypatch, "reference_edit_sam_review"
    )


# ---------------------------------------------------------------------------
# Ordinary legacy clip CPU failure isolation
# ---------------------------------------------------------------------------


def test_ordinary_clip_cpu_failure_is_terminal_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One clip's ordinary CPU failure is terminal; the other continues."""
    from tests.test_v3_pair import _add_ready_clip
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _reference_edit_config(
        tmp_path, monkeypatch, "run", same_parent_fallback_enabled=True
    )
    storage = _pair_storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="clip-2", clip_suffix="9", entity_types=("subject",)
    )
    pair_clips = _pair_clips()
    pair_clips(
        config,
        storage,
        judge=_ScopedJudge(
            scopes={("clip-1", "e1"): "full", ("clip-2", "e1"): "full"}
        ),
    )
    ledger = GroupLedger(tmp_path / "ledger")
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        ledger,
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    real_publish = runner._publish_clip_if_terminal

    def exploding_publish(shard: str, stor: Any, clip_uid: str) -> None:
        if clip_uid == "clip-1":
            raise ValueError("publication exploded")
        real_publish(shard, stor, clip_uid)

    runner._publish_clip_if_terminal = exploding_publish
    # seed_jobs processes both not_required clips; clip-1 fails terminally.
    runner.seed_jobs()
    failed_clip = storage.read_clip("clip-1")
    assert failed_clip.reference_edit is not None
    assert failed_clip.reference_edit.status == "failed"
    ok_clip = storage.read_clip("clip-2")
    assert ok_clip.reference_edit is not None
    assert ok_clip.reference_edit.status == "ready"
    failures = (Path(storage.root) / "failures.jsonl").read_text(encoding="utf-8")
    assert "publication exploded" in failures
    stats = runner.reconcile_stats(SHARD)
    assert stats.processed == 2
    assert stats.failed == 1

    # Restart after the crash: the failure publication is verified as-is.
    fresh = ReferenceEditEpochRunner(
        config, {SHARD: storage}, GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    assert fresh.seed_jobs() == []
    assert fresh.reconcile_stats(SHARD).to_dict() == stats.to_dict()


# ---------------------------------------------------------------------------
# Durable authority: seed-plan corruption through the real seed_jobs() path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tamper",
    ("anchor_digest", "seed_not_int", "seed_out_of_range", "schema"),
)
def test_seed_plan_tamper_through_seed_jobs_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """A corrupted seed plan must fail closed, never become a clip failure."""
    config, storage, runner, _scheduler, _routing = _build(tmp_path, monkeypatch)
    runner.seed_jobs()
    seed_path = runner._seed_path(SHARD, "clip-1", "e1", 1)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    if tamper == "anchor_digest":
        payload["anchor_digest"] = "0" * 64
    elif tamper == "seed_not_int":
        payload["seed"] = "not-an-int"
    elif tamper == "seed_out_of_range":
        payload["seed"] = 2**31 + 1
    else:
        payload["schema"] = "tampered-schema"
    seed_path.write_text(json.dumps(payload), encoding="utf-8")

    failures_path = Path(storage.root) / "failures.jsonl"
    before_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    with pytest.raises(ReferenceEditDurableError):
        fresh.seed_jobs()
    # The clip is NOT published as a semantic failure.
    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is None
    after_failures = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    assert after_failures == before_failures, "no new semantic failure diagnostic"
    assert "reference_edit" not in after_failures
    # Zero model calls: seed_jobs is pure CPU and never reached a backend.
    assert clip.pairing is not None


def test_failure_marker_reason_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure marker must bind terminal + reason + delta, not just status."""
    from tests.test_v3_pair import _add_ready_clip
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _reference_edit_config(
        tmp_path, monkeypatch, "run", same_parent_fallback_enabled=True
    )
    storage = _pair_storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="clip-2", clip_suffix="9", entity_types=("subject",)
    )
    _pair_clips()(
        config,
        storage,
        judge=_ScopedJudge(
            scopes={("clip-1", "e1"): "full", ("clip-2", "e1"): "full"}
        ),
    )
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    real_publish = runner._publish_clip_if_terminal

    def exploding_publish(shard: str, stor: Any, clip_uid: str) -> None:
        if clip_uid == "clip-1":
            raise ValueError("publication exploded")
        real_publish(shard, stor, clip_uid)

    runner._publish_clip_if_terminal = exploding_publish
    runner.seed_jobs()
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["terminal"] == "failed"
    marker["reason"] = "tampered"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    with pytest.raises(
        ReferenceEditDurableError, match="failure outcome marker does not match"
    ):
        fresh.seed_jobs()


# ---------------------------------------------------------------------------
# Durable reader: malformed / non-object authority JSON is fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ("{broken json", "[]"))
def test_seed_plan_malformed_json_fails_closed_through_seed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    config, storage, runner, _scheduler, _routing = _build(tmp_path, monkeypatch)
    runner.seed_jobs()
    seed_path = runner._seed_path(SHARD, "clip-1", "e1", 1)
    seed_path.write_text(payload, encoding="utf-8")

    failures_path = Path(storage.root) / "failures.jsonl"
    before = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    with pytest.raises(ReferenceEditDurableError, match="durable Reference Edit JSON"):
        fresh.seed_jobs()
    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is None, "corruption must not publish a failure"
    after = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    assert after == before, "no semantic failure diagnostic for corruption"


@pytest.mark.parametrize(
    ("authority", "reader"),
    (
        ("entity_outcome", "seed"),
        ("clip_outcome", "reconcile"),
    ),
)
def test_malformed_outcome_json_fails_closed_not_semantic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
    reader: str,
) -> None:
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    _drain(runner, scheduler)
    del routing
    if authority == "entity_outcome":
        target = runner._entity_outcome_path(SHARD, "clip-1", "e1")
    else:
        target = runner._clip_outcome_path(SHARD, "clip-1")
    target.write_text("{broken json", encoding="utf-8")

    failures_path = Path(storage.root) / "failures.jsonl"
    before = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    with pytest.raises(ReferenceEditDurableError, match="durable Reference Edit JSON"):
        if reader == "seed":
            fresh.seed_jobs()
        else:
            fresh.reconcile_stats(SHARD)
    after = failures_path.read_text(encoding="utf-8") if failures_path.is_file() else ""
    assert after == before, "corruption must not be recorded as a clip failure"


# ---------------------------------------------------------------------------
# Clip outcome marker is an exact authority
# ---------------------------------------------------------------------------


def _published_clip_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, ReferenceEditEpochRunner, Path]:
    config, storage, runner, scheduler, routing = _build(tmp_path, monkeypatch)
    _drain(runner, scheduler)
    del routing
    marker_path = runner._clip_outcome_path(SHARD, "clip-1")
    assert json.loads(marker_path.read_text(encoding="utf-8"))["terminal"] == "ready"
    return config, storage, runner, marker_path


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("delta", {"processed": 999}),
        ("schema", "tampered-schema"),
        ("clip_uid", "other-clip"),
        ("terminal", "tampered"),
    ),
)
def test_ready_marker_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    config, storage, _runner, marker_path = _published_clip_marker(
        tmp_path, monkeypatch
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1"]},
    )
    with pytest.raises(ReferenceEditDurableError):
        fresh.reconcile_stats(SHARD)


# ---------------------------------------------------------------------------
# Attempt outcome markers are validated accounting authority
# ---------------------------------------------------------------------------


def _candidate2_accept_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, ReferenceEditEpochRunner]:
    """candidate1 rejected -> candidate2 accepted, both markers durable."""
    _config, storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-epoch", with_alternate=True
    )
    routing.qwen.reject_candidate1 = True
    _drain(runner, scheduler)
    assert runner._validated_attempt_outcome(SHARD, "clip-1", "e1", 1) is not None
    assert runner._validated_attempt_outcome(SHARD, "clip-1", "e1", 2) is not None
    return _config, storage, runner


def test_attempt1_accepted_flag_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, runner = _candidate2_accept_run(tmp_path, monkeypatch)
    path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1)
    marker = json.loads(path.read_text(encoding="utf-8"))
    assert marker["accepted"] is False
    marker["accepted"] = True
    marker["status"] = "accepted"
    path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "alt"]},
    )
    with pytest.raises(ReferenceEditDurableError, match="attempt 2 without a rejected"):
        fresh.reconcile_stats(SHARD)


@pytest.mark.parametrize("tamper", ("attempt1_accepted", "attempt2_exists"))
def test_fallback_attempt1_marker_consistency_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """fallback@1 requires rejected attempt1 and no attempt2 marker."""
    _config, _storage, runner, scheduler, routing = _build(
        tmp_path, monkeypatch, run_name="run-fallback-attempt1"
    )
    monkeypatch.setattr(runner, "durable_alternate", lambda *args, **kwargs: None)
    routing.qwen.accept = False
    _drain(runner, scheduler)

    outcome = runner._entity_outcome(SHARD, "clip-1", "e1")
    assert outcome is not None
    assert outcome["outcome"] == "fallback"
    assert outcome["attempt_index"] == 1
    first_path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", 1)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    assert first["accepted"] is False
    second_path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", 2)
    assert not second_path.exists()

    if tamper == "attempt1_accepted":
        first["accepted"] = True
        first["status"] = "accepted"
        first_path.write_text(json.dumps(first), encoding="utf-8")
        match = "fell back on attempt 1.*attempt-1 marker disagrees"
    else:
        second = dict(first)
        second["attempt_index"] = 2
        second_path.parent.mkdir(parents=True, exist_ok=True)
        second_path.write_text(json.dumps(second), encoding="utf-8")
        match = "fell back on attempt 1.*attempt-2 marker exists"

    with pytest.raises(ReferenceEditDurableError, match=match):
        runner.reconcile_stats(SHARD)


def test_attempt2_index_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage, runner = _candidate2_accept_run(tmp_path, monkeypatch)
    path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", 2)
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["attempt_index"] = 1
    path.write_text(json.dumps(marker), encoding="utf-8")

    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "alt"]},
    )
    with pytest.raises(ReferenceEditDurableError, match="index drifted"):
        fresh.reconcile_stats(SHARD)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema", "tampered", "schema drifted"),
        ("status", "unknown_status", "status is unknown"),
        ("accepted", "yes", "not a bool"),
        ("rejection_reason", 5, "not a string"),
    ),
)
def test_attempt_marker_field_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    match: str,
) -> None:
    config, _storage, runner = _candidate2_accept_run(tmp_path, monkeypatch)
    path = runner._attempt_outcome_path(SHARD, "clip-1", "e1", 2)
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker[field] = value
    path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ReferenceEditDurableError, match=match):
        runner.reconcile_stats(SHARD)
    del config


# ---------------------------------------------------------------------------
# Ordinary CPU failure keeps the legacy reason / exception_type
# ---------------------------------------------------------------------------


def test_ordinary_clip_failure_reason_and_exception_type_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip CPU failure keeps the legacy reason AND exception_type."""
    from tests.test_v3_pair import _add_ready_clip
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _reference_edit_config(
        tmp_path, monkeypatch, "run", same_parent_fallback_enabled=True
    )
    storage = _pair_storage(config, entity_types=("subject",))
    _add_ready_clip(
        config, storage, clip_uid="clip-2", clip_suffix="9", entity_types=("subject",)
    )
    _pair_clips()(
        config,
        storage,
        judge=_ScopedJudge(
            scopes={("clip-1", "e1"): "full", ("clip-2", "e1"): "full"}
        ),
    )
    runner = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    real_publish = runner._publish_clip_if_terminal

    def exploding_publish(shard: str, stor: Any, clip_uid: str) -> None:
        if clip_uid == "clip-1":
            raise ValueError("publication exploded")
        real_publish(shard, stor, clip_uid)

    runner._publish_clip_if_terminal = exploding_publish
    runner.seed_jobs()

    clip = storage.read_clip("clip-1")
    assert clip.reference_edit is not None
    assert clip.reference_edit.status == "failed"
    # Legacy provenance: the reason is str(exc), not "TypeError: <message>".
    assert clip.reference_edit.reason == "publication exploded"

    records = [
        json.loads(line)
        for line in (Path(storage.root) / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    reference_edit_records = [
        record for record in records if record.get("stage") == "reference_edit"
    ]
    assert len(reference_edit_records) == 1
    assert reference_edit_records[0]["reason"] == "publication exploded"
    assert reference_edit_records[0]["details"]["exception_type"] == "ValueError"

    marker = json.loads(
        runner._clip_outcome_path(SHARD, "clip-1").read_text(encoding="utf-8")
    )
    assert marker["terminal"] == "failed"
    assert marker["reason"] == "publication exploded"
    assert marker["delta"] == {"processed": 1, "failed": 1}

    # Restart verifies that exact marker and keeps the same stats.
    stats = runner.reconcile_stats(SHARD)
    assert stats.failed == 1
    fresh = ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: ["clip-1", "clip-2"]},
    )
    assert fresh.seed_jobs() == []
    assert fresh.reconcile_stats(SHARD).to_dict() == stats.to_dict()


# ---------------------------------------------------------------------------
# frozen plan validation: once per seed, not once per clip
# ---------------------------------------------------------------------------


def _fresh_target_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int
) -> tuple[Any, Any, list[str]]:
    """One shard of ``count`` clips that all need a Reference Edit repair."""
    from tests.test_v3_pair import _add_ready_clip
    from tests.test_v3_post_mask_epoch_pair import _ScopedJudge

    config = _reference_edit_config(
        tmp_path, monkeypatch, "run-plan-once", same_parent_fallback_enabled=True
    )
    storage = _pair_storage(config, entity_types=("subject",))
    uids = ["clip-1"]
    for index in range(2, count + 1):
        uid = f"clip-{index}"
        _add_ready_clip(
            config,
            storage,
            clip_uid=uid,
            clip_suffix=str(index),
            entity_types=("subject",),
        )
        uids.append(uid)
    _pair_clips()(
        config,
        storage,
        judge=_ScopedJudge(scopes={(uid, "e1"): "repairable" for uid in uids}),
    )
    return config, storage, uids


def _plan_runner(config: Any, storage: Any, tmp_path: Path, uids: list[str]) -> Any:
    return ReferenceEditEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / "ledger"),
        eligible_clip_uids_by_shard={SHARD: uids},
    )


def _count_plan_validations(runner: Any) -> dict[str, int]:
    """Count plan-entry validations, their clip digests, and shard plan reads.

    The patches are on the instance and call through to the real validators, so
    the counters stay exact rather than estimated.
    """
    counts = {"entries": 0, "digests": 0, "shard_reads": 0}
    original_entry = runner._verify_plan_entry
    original_digest = runner._clip_digest
    original_existing = runner._existing_plan_for_reconcile

    def counted_entry(shard: str, storage: Any, clip_uid: str, entry: Any) -> None:
        counts["entries"] += 1
        return original_entry(shard, storage, clip_uid, entry)

    def counted_digest(storage: Any, clip: Any) -> str:
        counts["digests"] += 1
        return original_digest(storage, clip)

    def counted_existing(shard: str) -> Any:
        counts["shard_reads"] += 1
        return original_existing(shard)

    runner._verify_plan_entry = counted_entry
    runner._clip_digest = counted_digest
    runner._existing_plan_for_reconcile = counted_existing
    return counts


def test_seed_validates_the_frozen_plan_once_per_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeding N clips must not verify the shard plan N extra times.

    Advancing clip by clip re-read and re-verified the whole shard plan - every
    clip's digest over its frames and masks manifests - so a shard cost
    O(clips^2) validations while ``_plan()`` had just validated the very same
    payload. The seed now spends exactly one pass, and reconcile keeps its own
    full pass because it verifies the state the stage actually published.
    """
    config, storage, uids = _fresh_target_shard(tmp_path, monkeypatch, count=6)
    # Freeze the plan first, so the seed under test reads it from disk.
    _plan_runner(config, storage, tmp_path, uids)._plan(SHARD)

    runner = _plan_runner(config, storage, tmp_path, uids)
    counts = _count_plan_validations(runner)

    jobs = runner.seed_jobs()

    assert len(jobs) == len(uids), "every clip is a fresh target with one job"
    assert counts["entries"] == len(uids), "one validation pass, not one per clip"
    assert counts["digests"] == len(uids)
    assert counts["shard_reads"] == 0, "the seed must not re-fetch the shard plan"

    # A second seed round validates the shard once more, not once per clip.
    runner.seed_jobs()
    assert counts["entries"] == 2 * len(uids)
    assert counts["shard_reads"] == 0

    # Reconcile is the opposite: it re-reads and re-verifies every entry.
    runner._existing_plan_for_reconcile(SHARD)
    assert counts["entries"] == 3 * len(uids)
    assert counts["shard_reads"] == 1


def test_a_changed_frozen_plan_is_revalidated_and_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed's validated plan must not become reconcile's authority.

    Reconcile verifies the state the stage published, so it re-reads the frozen
    plan and re-runs every entry check: an entry edited after the seed validated
    it is still durable corruption.
    """
    config, storage, uids = _fresh_target_shard(tmp_path, monkeypatch, count=3)
    runner = _plan_runner(config, storage, tmp_path, uids)
    runner.seed_jobs()

    path = runner._plan_path(SHARD)
    before = path.read_bytes()
    text = before.decode("utf-8")
    payload = json.loads(text)
    target = sorted(payload["clips"])[1]
    digest = payload["clips"][target]["digest"]
    # Edit the digest in place, so the byte length cannot be what sees it.
    assert text.count(digest) == 1
    replacement = ("f" if digest[0] != "f" else "0") + digest[1:]
    path.write_text(text.replace(digest, replacement), encoding="utf-8")
    assert len(path.read_bytes()) == len(before)

    with pytest.raises(ReferenceEditDurableError, match="input drifted"):
        runner._existing_plan_for_reconcile(SHARD)
