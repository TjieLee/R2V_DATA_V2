"""Tests for the resource-epoch background-removal adapter.

Two things are covered here:

* Adapter semantics: durable random seeds, global attempt indices, semantic
  failures as committed attempts, digest validation and fail-safe behaviour.
* Legacy equivalence: the same clip, the same fake Boogu worker and the same
  fake judge are driven through ``remove_backgrounds()`` and through the epoch
  scheduler, and the resulting background states must match.

Nothing here patches a prompt, a threshold or an accept/reject rule: if the
adapter grew its own copy of one, these tests would stop describing production.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.remove as remove_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.boogu_remove_backend import BooguBackgroundRemovalBackend
from r2v_data_v2.v3.config import (
    BOOGU_REMOVE_BACKEND,
    BackgroundConfig,
    DebugConfig,
    QwenServiceConfig,
    QwenServicesConfig,
    RemoveConfig,
    SourceConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import encode_binary_mask
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    semantic_input_digest,
)
from r2v_data_v2.v3.post_mask_epoch_removal import (
    ATTEMPT_STATUS_FAILED,
    CANDIDATE_ARTIFACT,
    PAYLOAD_ATTEMPT,
    PAYLOAD_ATTEMPT_STATUS,
    PAYLOAD_SEED,
    REMOVAL_CANDIDATE2_JOB,
    REMOVAL_GENERATE_JOB,
    REMOVAL_JUDGE_JOB,
    SEMANTIC_MISMATCH,
    RemovalEpochError,
    RemovalEpochRunner,
    build_qwen_epoch_config,
    build_removal_epoch_factories,
    build_removal_epoch_runner,
    build_removal_generate_job,
    build_removal_judge_job,
    prepare_shard_storage,
    removal_generation_resource,
    removal_seed_anchor_inputs,
    seed_plan_path,
)
from r2v_data_v2.v3.post_mask_epoch_resources import (
    QwenConcurrentExecutor,
    QwenEpochResource,
    SerialBatchExecutor,
    WorkerEpochResource,
    WorkerPoolConfig,
    WorkerSlotExecutor,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.post_mask_production import (
    ShardPaths,
    hydrate_shard,
    initialize_shard,
)
from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput
from r2v_data_v2.v3.remove import prepare_removal_attempt_context, remove_backgrounds
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundReferenceState,
    BackgroundRemovalReview,
    ClipSource,
    CoverageState,
    EntityVisibilitySummary,
    SampledFrame,
    SampledFramesArtifact,
    TrackedEntityMasks,
    TrackedMaskFrame,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage

WIDTH = 6
HEIGHT = 5
SHARD = "shard-000000000-000000000"


# ---------------------------------------------------------------------------
# Fake models
# ---------------------------------------------------------------------------


def _accept(reason: str = "clean") -> BackgroundRemovalReview:
    return BackgroundRemovalReview(
        verdict="accept",
        foreground_absent=True,
        foreground_not_reconstructed=True,
        no_new_salient_entity=True,
        background_only_in_repaired_region=True,
        background_continuity_ok=True,
        no_visible_artifacts=True,
        reason=reason,
    )


def _reject(reason: str = "visible seam") -> BackgroundRemovalReview:
    return BackgroundRemovalReview(
        verdict="reject",
        foreground_absent=True,
        foreground_not_reconstructed=True,
        no_new_salient_entity=True,
        background_only_in_repaired_region=True,
        background_continuity_ok=True,
        no_visible_artifacts=False,
        reason=reason,
    )


class _BooguWorker:
    """Persistent Boogu worker: exposes ``edit``, not ``remove``."""

    def __init__(self, responses: Sequence[Any] = ()) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def edit(self, **kwargs: Any) -> BooguEditOutput:
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        self.calls.append(dict(kwargs))
        try:
            response = next(self.responses)
        except StopIteration:
            response = (220, 30, 40)
        if isinstance(response, Exception):
            raise response
        buffer = BytesIO()
        Image.new("RGB", (width, height), response).save(buffer, format="PNG")
        instruction = str(kwargs["instruction"])
        return BooguEditOutput(
            png_bytes=buffer.getvalue(),
            original_instruction=instruction,
            rewritten_instruction=None,
            effective_instruction=instruction,
            worker_metadata={"seed": kwargs.get("seed")},
        )

    def close(self) -> None:
        self.closed = True


class _Judge:
    def __init__(self, responses: Sequence[Any] = ()) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def review(self, **kwargs: Any) -> BackgroundRemovalReview:
        self.calls += 1
        try:
            response = next(self.responses)
        except StopIteration:
            response = _accept()
        if isinstance(response, Exception):
            raise response
        return response


class _SeedAllocator:
    """Deterministic stand-in for ``new_boogu_seed`` that counts its calls."""

    def __init__(self, values: Sequence[int] = ()) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> int:
        index = self.calls
        self.calls += 1
        if index < len(self.values):
            return self.values[index]
        return 9000 + index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeds: tuple[int, ...] = (0, 17),
    max_generation_ratio: float = 1.0,
    dilation: int = 0,
    run_name: str = "run",
) -> V3Config:
    writable = (tmp_path / "workspace" / "data").resolve()
    dataset_root = (tmp_path / "public" / "dataset").resolve()
    pretrained = (tmp_path / "public" / "pretrained").resolve()
    user_models = writable / "models"
    for path in (writable, dataset_root, pretrained, user_models):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", dataset_root)
    monkeypatch.setattr(config_module, "ALLOWED_PRETRAINED_ROOT", pretrained)
    monkeypatch.setattr(config_module, "ALLOWED_USER_MODEL_ROOT", user_models)
    dataset = dataset_root / "source.jsonl"
    dataset.write_text("", encoding="utf-8")
    config = V3Config(
        dataset_json=dataset,
        run_root=writable / "runs" / run_name,
        export_root=writable / "datasets" / f"{run_name}-dataset",
        source=SourceConfig(limit=10),
        qwen=QwenServicesConfig(
            candidate_judge=QwenServiceConfig(
                model=str(pretrained / "Qwen" / "judge")
            ),
            background_remove_judge=QwenServiceConfig(
                model=str(pretrained / "Qwen" / "judge")
            ),
        ),
        background=BackgroundConfig(max_pending_remove_area_ratio=0.5),
        remove=RemoveConfig(
            base_model_path=pretrained / "Qwen" / "Qwen-Image-Edit-2511",
            adapter_path=user_models / "object-remover",
            candidate_seeds=seeds,
            generation_mask_dilation_pixels=dilation,
            max_generation_mask_area_ratio=max_generation_ratio,
        ),
        debug=DebugConfig(save_diagnostics=False),
    )
    config.validate()
    return config


def _boogu_config(config: V3Config) -> V3Config:
    writable = config.run_root.parents[1]
    changed = replace(
        config,
        remove=replace(
            config.remove,
            backend=BOOGU_REMOVE_BACKEND,
            inference_profile="boogu_4step_v1",
            adapter_path=None,
        ),
        reference_edit=replace(
            config.reference_edit,
            python_executable=writable / "venvs" / "boogu" / "python",
            code_root=writable / "vendor" / "Boogu-Image",
            model_path=writable / "models" / "Boogu-Image",
            target_area=32 * 32,
        ),
    )
    changed.validate()
    return changed


def _fixture_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str = "run", **kwargs: Any
) -> V3Config:
    return _boogu_config(_base_config(tmp_path, monkeypatch, run_name=run_name, **kwargs))


def _mask() -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=bool)
    value[2, 2] = True
    return value


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _tracked_masks(clip_uid: str) -> TrackedMasksArtifact:
    mask = _mask()
    frames = [
        TrackedMaskFrame(
            slot=slot,
            present=True,
            track_valid=True,
            confidence=0.9,
            backend_confidences=[0.9],
            backend_object_ids=["obj-1"],
            area_pixels=1,
            area_ratio=float(mask.mean()),
            bbox_xyxy=_bbox(mask),
            rle=encode_binary_mask(mask),
        )
        for slot in range(10)
    ]
    return TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=WIDTH,
        height=HEIGHT,
        entities={
            "e1": TrackedEntityMasks(
                status="ready",
                reference_type="subject",
                grounding_prompt="person in red",
                backend_object_ids=["obj-1"],
                frames=frames,
            )
        },
    )


def _coverage() -> CoverageState:
    return CoverageState(
        passed=True,
        qualifying_entity_ids=["e1"],
        entity_visibility_summary={
            "e1": EntityVisibilitySummary(
                status="ready",
                visible_frame_slots=list(range(10)),
                visible_frame_count=10,
                coverage_ratio=1.0,
                qualifies=True,
                per_frame_area_ratio=[1 / (WIDTH * HEIGHT)] * 10,
                per_frame_confidence=[0.9] * 10,
            )
        },
    )


def _write_frames(storage: RunStorage, clip_uid: str) -> None:
    frames = []
    for slot in range(10):
        path = storage.frame_path(clip_uid, slot)
        Image.new("RGB", (WIDTH, HEIGHT), (10 + slot, 20, 30)).save(path, format="JPEG")
        frames.append(
            SampledFrame(
                slot=slot,
                source_frame_index=slot * 10,
                timestamp_seconds=float(slot),
                image_path=f"frames/{slot:02d}.jpg",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    write_json_atomic(
        storage.frames_manifest_path(clip_uid),
        SampledFramesArtifact(
            clip_uid=clip_uid, width=WIDTH, height=HEIGHT, frames=frames
        ).model_dump(mode="json"),
    )


def _pending_storage(
    config: V3Config, *, clip_uids: tuple[str, ...] = ("clip-1",)
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    for clip_uid in clip_uids:
        video = config.dataset_json.parent / f"{clip_uid}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"target")
        storage.create_clip(
            clip_uid=clip_uid,
            source=ClipSource(
                video_path=str(video),
                parent_video_id="parent",
                clip_suffix="1_0",
                source_index=0,
                caption_raw="",
                metadata={},
            ),
        )
        storage.write_annotation(
            clip_uid,
            AnnotationState(
                status="ready",
                t2v_caption="A person crosses an open plaza.",
                entities=[
                    AnnotationEntity(
                        entity_id="e1",
                        reference_type="subject",
                        phrase="person in red",
                        grounding_prompt="person in red",
                    )
                ],
                background=BackgroundAnnotation(
                    phrase="open plaza",
                    grounding_prompt="open plaza",
                ),
            ),
        )
        _write_frames(storage, clip_uid)
        storage.write_masks(clip_uid, _tracked_masks(clip_uid))
        storage.write_coverage(clip_uid, _coverage())
    stats = build_background_candidates(config, storage)
    assert stats.pending_remove == len(clip_uids)
    return storage


def _state(storage: RunStorage, clip_uid: str = "clip-1") -> BackgroundReferenceState:
    background = storage.read_clip(clip_uid).references.background
    assert background is not None
    return background


def _runner(
    config: V3Config,
    storage: RunStorage,
    ledger: GroupLedger,
    *,
    seed_allocator: Any = None,
) -> RemovalEpochRunner:
    if seed_allocator is not None:
        return RemovalEpochRunner(
            config, {SHARD: storage}, ledger, seed_allocator=seed_allocator
        )
    return RemovalEpochRunner(config, {SHARD: storage}, ledger)


def _scheduler(
    ledger: GroupLedger, runner: RemovalEpochRunner, worker: Any, judge: Any
) -> ResourceEpochScheduler:
    return ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={
            RESOURCE_BOOGU: SerialBatchExecutor(runner.run, worker),
            RESOURCE_QWEN: SerialBatchExecutor(runner.run, judge),
        },
    )


# ---------------------------------------------------------------------------
# Planning and the durable seed plan
# ---------------------------------------------------------------------------


def test_seed_jobs_plans_only_the_first_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    allocator = _SeedAllocator([101, 202])
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"), seed_allocator=allocator)

    jobs = runner.seed_jobs()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_type == REMOVAL_GENERATE_JOB
    assert job.resource == removal_generation_resource(config) == RESOURCE_BOOGU
    assert dict(job.target)["cycle_index"] == "0"
    assert job.attempt_index == 0
    assert job.seed == 101
    assert allocator.calls == 1


def test_seed_plan_is_drawn_once_and_reused_across_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    allocator = _SeedAllocator([101])

    first = _runner(config, storage, ledger, seed_allocator=allocator).seed_jobs()
    second = _runner(config, storage, ledger, seed_allocator=allocator).seed_jobs()

    assert allocator.calls == 1, "a restart must never re-randomise a seed"
    assert [job.job_id() for job in first] == [job.job_id() for job in second]
    assert first[0].seed == 101 == second[0].seed


def test_seed_plan_is_durable_state_not_a_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    allocator = _SeedAllocator([101])
    _runner(config, storage, ledger, seed_allocator=allocator).seed_jobs()

    plan = seed_plan_path(
        ledger.root,
        canonical_shard=SHARD,
        clip_uid="clip-1",
        attempt_index=0,
    )
    payload = json.loads(plan.read_text())
    assert payload["seed"] == 101
    assert payload["schema_version"] == "post_mask_epoch_remove_seed/1"
    assert payload["clip_uid"] == "clip-1"
    assert payload["attempt_index"] == 0
    assert payload["cycle_index"] == 0
    assert isinstance(payload["semantic_anchor_digest"], str)


def test_seed_plan_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    allocator = _SeedAllocator([101, 202])
    _runner(config, storage, ledger, seed_allocator=allocator).seed_jobs()
    plan = seed_plan_path(
        ledger.root, canonical_shard=SHARD, clip_uid="clip-1", attempt_index=0
    )
    tampered = json.loads(plan.read_text())
    tampered["cycle_index"] = 7
    plan.write_text(json.dumps(tampered))

    with pytest.raises(RemovalEpochError):
        _runner(config, storage, ledger, seed_allocator=allocator).seed_jobs()

    assert allocator.calls == 1, "a corrupt plan must not silently re-randomise"


def test_seed_anchor_excludes_the_seed_but_binds_the_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    context = prepare_removal_attempt_context(config, storage, "clip-1", _state(storage))

    first = removal_seed_anchor_inputs(config, context, attempt_index=0, cycle_index=0)
    second = removal_seed_anchor_inputs(config, context, attempt_index=1, cycle_index=1)

    assert "seed" not in first
    assert semantic_input_digest(first) != semantic_input_digest(second)


def test_non_boogu_backend_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _base_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)

    with pytest.raises(RemovalEpochError, match="requires Boogu"):
        RemovalEpochRunner(config, {SHARD: storage}, GroupLedger(tmp_path / "group"))


def test_seed_jobs_rejects_an_oversized_generation_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch, max_generation_ratio=0.001)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))

    assert runner.seed_jobs() == []
    assert _state(storage).status == "rejected"
    assert _state(storage).reason == "generation_mask_too_large"


# ---------------------------------------------------------------------------
# Semantic failures are committed attempts, not execution failures
# ---------------------------------------------------------------------------


def test_generation_failure_is_a_committed_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))
    worker = _BooguWorker([RuntimeError("boom")])

    result = runner.run(runner.seed_jobs()[0], worker)

    assert result.outcome == OUTCOME_COMPLETED, "a failed attempt is still durable"
    assert result.committed is True
    assert result.payload[PAYLOAD_ATTEMPT_STATUS] == ATTEMPT_STATUS_FAILED
    assert result.payload[PAYLOAD_ATTEMPT]["status"] == "failed"
    assert result.payload[PAYLOAD_ATTEMPT]["seed"] == result.payload[PAYLOAD_SEED]
    assert result.artifacts == {}


def test_judge_failure_is_a_committed_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    worker = _BooguWorker()
    judge = _Judge([RuntimeError("judge down"), RuntimeError("judge down")])

    outcome = _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())

    # Both cycles execute, each judge failure is a durable failed attempt, and
    # the cycle ends in the legacy retryable publication rather than an
    # unresolved scheduler job.
    assert outcome["completed"] is True
    assert judge.calls == 2
    assert len(worker.calls) == 2
    assert _state(storage).status == "pending_remove"
    assert _state(storage).reason == "removal_infrastructure_failure"
    assert [item.status for item in _state(storage).removal_attempts] == [
        "failed",
        "failed",
    ]


def test_failed_generation_unlocks_the_next_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    worker = _BooguWorker([RuntimeError("boom")])
    judge = _Judge([_accept()])
    unlocked: list[str] = []
    original = runner.finalize

    def tracked(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        created = list(original(job, result))
        unlocked.extend(item.job_type for item in created)
        return created

    runner.finalize = tracked  # type: ignore[method-assign]

    outcome = _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())

    assert outcome["completed"] is True
    assert REMOVAL_CANDIDATE2_JOB in unlocked
    assert len(worker.calls) == 2
    assert judge.calls == 1
    assert _state(storage).status == "ready_removed"


# ---------------------------------------------------------------------------
# Digest validation and candidate integrity
# ---------------------------------------------------------------------------


def _tampered_job(job: ModelJob, **changes: Any) -> ModelJob:
    payload = job.identity_payload()
    payload.update(changes)
    rebuilt = ModelJob.create(
        job_type=job.job_type,
        resource=job.resource,
        canonical_shard=job.canonical_shard,
        clip_uid=job.clip_uid,
        semantic_inputs={"tampered": True},
        model_identity=job.model_identity,
        target=dict(job.target),
        attempt_index=job.attempt_index,
        seed=job.seed,
        dependencies=dict(job.dependency_digests),
    )
    return replace(rebuilt, input_digest="0" * 64)


def test_generation_digest_mismatch_makes_no_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))
    worker = _BooguWorker()
    job = _tampered_job(runner.seed_jobs()[0])

    result = runner.run(job, worker)

    assert worker.calls == []
    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert result.detail == SEMANTIC_MISMATCH


def test_candidate_tamper_makes_no_judge_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    worker = _BooguWorker()
    judge = _Judge([_accept()])
    scheduler = _scheduler(ledger, runner, worker, judge)

    (generate_job,) = runner.seed_jobs()
    generation = runner.run(generate_job, worker)
    phase = ledger.phase("manual")
    digest = phase.publish_artifact(
        generate_job.job_id(),
        CANDIDATE_ARTIFACT,
        generation.artifacts[CANDIDATE_ARTIFACT],
    )
    result_digest = phase.publish_result(generate_job, generation)
    receipt = phase.commit(
        generate_job,
        outcome=generation.outcome,
        artifact_digests={CANDIDATE_ARTIFACT: digest, "result.json": result_digest},
        result_digest=result_digest,
    )
    ledger.note_commit("manual", receipt)
    (judge_job,) = runner.finalize(generate_job, generation)

    candidate = phase.artifact_path(generate_job.job_id(), CANDIDATE_ARTIFACT)
    candidate.write_bytes(candidate.read_bytes() + b"tampered")

    result = runner.run(judge_job, judge)

    assert judge.calls == 0
    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert "candidate artifact" in result.detail
    assert scheduler is not None


def test_judge_job_binds_the_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))
    (generate_job,) = runner.seed_jobs()
    context = prepare_removal_attempt_context(config, storage, "clip-1", _state(storage))

    judge_job = build_removal_judge_job(
        config,
        context,
        canonical_shard=SHARD,
        clip_uid="clip-1",
        cycle_index=0,
        attempt_index=0,
        seed=101,
        candidate_sha256="deadbeef",
        generate_job=generate_job,
    )

    target = dict(judge_job.target)
    assert target["candidate_sha256"] == "deadbeef"
    assert target["generation_job_id"] == generate_job.job_id()
    assert dict(judge_job.dependency_digests) == {"candidate": generate_job.job_id()}


# ---------------------------------------------------------------------------
# Attempt index across retry cycles
# ---------------------------------------------------------------------------


def test_second_cycle_uses_the_next_attempt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    context = prepare_removal_attempt_context(config, storage, "clip-1", _state(storage))

    second = build_removal_generate_job(
        config,
        context,
        canonical_shard=SHARD,
        clip_uid="clip-1",
        cycle_index=1,
        attempt_index=1,
        seed=202,
    )

    assert second.job_type == REMOVAL_CANDIDATE2_JOB
    assert second.attempt_index == 1


def test_retry_cycle_advances_the_attempt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    allocator = _SeedAllocator([101, 202, 303])
    worker = _BooguWorker([RuntimeError("boom"), RuntimeError("boom")])
    judge = _Judge()

    runner = _runner(config, storage, ledger, seed_allocator=allocator)
    _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())
    assert _state(storage).status == "pending_remove"
    assert len(_state(storage).removal_attempts) == 2

    resumed = _runner(config, storage, ledger, seed_allocator=allocator)
    jobs = resumed.seed_jobs()

    assert len(jobs) == 1
    assert jobs[0].attempt_index == 2, "run two must not collide with run one receipts"
    assert jobs[0].seed == 303


# ---------------------------------------------------------------------------
# Resume after a crash
# ---------------------------------------------------------------------------


def test_generation_committed_then_finalizer_crash_resumes_without_a_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    worker = _BooguWorker()
    judge = _Judge([_accept()])

    runner = _runner(config, storage, ledger)
    original = runner.finalize
    crash = {"pending": True}

    def crashing(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        if crash["pending"] and job.job_type == REMOVAL_GENERATE_JOB:
            crash["pending"] = False
            raise RuntimeError("simulated crash after the receipt")
        return original(job, result)

    runner.finalize = crashing  # type: ignore[method-assign]
    first = _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())
    assert first["completed"] is False
    assert len(worker.calls) == 1
    assert judge.calls == 0

    resumed = _runner(config, storage, ledger)
    second = _scheduler(ledger, resumed, worker, judge).run(resumed.seed_jobs())

    assert second["completed"] is True
    assert len(worker.calls) == 1, "a committed generation is never re-requested"
    assert judge.calls == 1
    assert _state(storage).status == "ready_removed"


def test_judge_committed_then_publication_crash_resumes_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    worker = _BooguWorker()
    judge = _Judge([_accept()])

    runner = _runner(config, storage, ledger)
    original = runner.finalize
    crash = {"pending": True}

    def crashing(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        if crash["pending"] and job.job_type == REMOVAL_JUDGE_JOB:
            crash["pending"] = False
            raise RuntimeError("simulated crash before publication")
        return original(job, result)

    runner.finalize = crashing  # type: ignore[method-assign]
    _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())
    assert len(worker.calls) == 1
    assert judge.calls == 1
    assert _state(storage).status == "pending_remove"

    resumed = _runner(config, storage, ledger)
    outcome = _scheduler(ledger, resumed, worker, judge).run(resumed.seed_jobs())

    assert outcome["completed"] is True
    assert len(worker.calls) == 1
    assert judge.calls == 1
    assert _state(storage).status == "ready_removed"
    assert outcome["diagnostics"]["resume"]["finalizers_replayed"] >= 1


# ---------------------------------------------------------------------------
# GPU epoch factories (no GPU is touched)
# ---------------------------------------------------------------------------


def _fake_process_manager() -> Any:
    class _Manager:
        def spawn(self, argv: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("no process may be spawned in tests")

    return _Manager()


def test_qwen_factory_builds_the_managed_judge_server_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    qwen_config = build_qwen_epoch_config(config)
    factories = build_removal_epoch_factories(
        config,
        runner,
        qwen_epoch_config=qwen_config,
        pool=WorkerPoolConfig(),
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
    )

    assert set(factories) == {RESOURCE_BOOGU, RESOURCE_QWEN}
    resource, executor = factories[RESOURCE_QWEN]()

    assert isinstance(resource, QwenEpochResource)
    assert isinstance(executor, QwenConcurrentExecutor)
    assert not hasattr(resource, "workers"), "no remover worker pool may be loaded"
    assert qwen_config.tensor_parallel_size == 1
    assert qwen_config.data_parallel_size == 8
    assert qwen_config.served_model_name == str(
        config.qwen.background_remove_judge.model
    )
    assert executor.max_inflight == 8
    assert executor.endpoint == qwen_config.base_url


def test_boogu_factory_builds_the_worker_slot_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    factories = build_removal_epoch_factories(
        config,
        runner,
        qwen_epoch_config=build_qwen_epoch_config(config),
        pool=WorkerPoolConfig(),
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
    )

    resource, executor = factories[RESOURCE_BOOGU]()

    assert isinstance(resource, WorkerEpochResource)
    assert isinstance(executor, WorkerSlotExecutor)
    assert executor.slot_count == 8


# ---------------------------------------------------------------------------
# Removal-only runner never completes a group
# ---------------------------------------------------------------------------


def test_removal_only_runner_never_reports_group_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.prepare_shard_storage",
        lambda *args, **kwargs: storage,
    )

    class _Group:
        group_id = "group-000000"
        canonical_shards = (SHARD,)

    class _Handles:
        def edit(self, **kwargs: Any) -> BooguEditOutput:
            return _BooguWorker().edit(**kwargs)

    class _FakeResource:
        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def counters(self) -> dict[str, Any]:
            return {}

    handles = _Handles()
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_removal_epoch_factories",
        lambda config, runner, **kwargs: {
            RESOURCE_BOOGU: lambda: (
                _FakeResource(),
                SerialBatchExecutor(runner.run, handles),
            ),
            RESOURCE_QWEN: lambda: (
                _FakeResource(),
                SerialBatchExecutor(runner.run, _Judge([_accept()])),
            ),
        },
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=tmp_path / "campaign",
        entity_mask_root=tmp_path / "entity_mask",
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
        qwen_epoch_config=build_qwen_epoch_config(config),
    )

    outcome = runner_callable(_Group(), GroupLedger(tmp_path / "group"), lambda *a, **k: None)

    assert outcome["remove_completed"] is True
    assert outcome["completed"] is False
    assert "downstream" in outcome["reason"]


# ---------------------------------------------------------------------------
# Fresh-root hydration
# ---------------------------------------------------------------------------


def _stage2_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shard: str = SHARD,
    clip_uid: str = "clip-1",
) -> tuple[Path, str, str]:
    config = _fixture_config(tmp_path, monkeypatch, run_name="stage2")
    entity_mask_root = tmp_path / "public" / "entity_mask"
    parts = entity_mask_root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    storage = _pending_storage(config, clip_uids=(clip_uid,))
    destination = entity_mask_root / "artifacts" / shard / clip_uid / "run"
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.root.rename(destination)
    (parts / f"{shard}.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "r2v.v3.pre_qwen_stage2.1",
                "source_index": 0,
                "clip_uid": clip_uid,
                "status": "ready_background_pending_remove",
                "coverage_passed": True,
                "background_status": "pending_remove",
                "reason": None,
                "artifact_root": f"artifacts/{shard}/{clip_uid}",
            }
        )
        + "\n"
    )
    return entity_mask_root, shard, clip_uid


def test_fresh_run_root_is_hydrated_before_seeding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity_mask_root, shard, clip_uid = _stage2_fixture(tmp_path, monkeypatch)
    config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"

    storage = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )

    clips = list(storage.iter_clips())
    assert [clip.clip_uid for clip in clips] == [clip_uid]
    assert clips[0].references.background is not None
    assert clips[0].references.background.status == "pending_remove"

    runner = RemovalEpochRunner(
        config, {shard: storage}, GroupLedger(tmp_path / "group")
    )
    assert len(runner.seed_jobs()) == 1


def test_hydrated_root_is_reused_without_duplicating_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity_mask_root, shard, clip_uid = _stage2_fixture(tmp_path, monkeypatch)
    config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"

    storage = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )
    assert [clip.clip_uid for clip in storage.iter_clips()] == [clip_uid]

    # Re-entering an initialized shard keeps the historical commit and every
    # hydrated clip instead of duplicating or resetting them.
    paths = ShardPaths.for_shard(
        post_mask_root, entity_mask_root / "parts" / f"{shard}.jsonl"
    )
    reentered = initialize_shard(config, paths, git_commit="new-code")
    assert [clip.clip_uid for clip in reentered.iter_clips()] == [clip_uid]
    assert reentered.read_run().git_commit in {"test", "new-code"}

    if sys.platform != "win32":
        # hydrate_shard indexes manifests with str(Path), which uses OS
        # separators; on Windows a re-hydration lookup raises KeyError inside
        # the frozen production module. Reported, not patched here.
        hydrate_shard(
            reentered,
            entity_mask_root=entity_mask_root,
            shard_path=entity_mask_root / "parts" / f"{shard}.jsonl",
            paths=paths,
        )
        assert [clip.clip_uid for clip in reentered.iter_clips()] == [clip_uid]


# ---------------------------------------------------------------------------
# Legacy-vs-epoch equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scenario:
    name: str
    generation: tuple[Any, ...]
    judge: tuple[Any, ...]
    seeds: tuple[int, ...]
    status: str
    reason: str | None
    boogu_calls: int
    qwen_calls: int
    candidate2: bool
    max_generation_ratio: float = 1.0


_SCENARIOS = (
    _Scenario("A-first-accept", (), (_accept("clean"),), (101,), "ready_removed", None, 1, 1, False),
    _Scenario(
        "B-reject-accept", (), (_reject("seam"), _accept("clean")), (101, 202), "ready_removed", None, 2, 2, True
    ),
    _Scenario(
        "C-reject-reject", (), (_reject("seam"), _reject("artifact")), (101, 202), "rejected", "all_removal_candidates_rejected", 2, 2, True
    ),
    _Scenario(
        "D-generation-failure-accept", (RuntimeError("boom"),), (_accept("clean"),), (101, 202), "ready_removed", None, 2, 1, True
    ),
    _Scenario(
        "E-judge-failure-accept", (), (RuntimeError("judge down"), _accept("clean")), (101, 202), "ready_removed", None, 2, 2, True
    ),
    _Scenario(
        "F-reject-failure", (), (_reject("seam"), RuntimeError("judge down")), (101, 202), "pending_remove", "removal_infrastructure_failure", 2, 2, True
    ),
    _Scenario(
        "G-failure-failure", (RuntimeError("boom"), RuntimeError("boom")), (), (101, 202), "pending_remove", "removal_infrastructure_failure", 2, 0, True
    ),
    _Scenario(
        "H-mask-too-large", (), (), (), "rejected", "generation_mask_too_large", 0, 0, False, 0.001
    ),
)


def _legacy_run(
    config: V3Config,
    storage: RunStorage,
    monkeypatch: pytest.MonkeyPatch,
    scenario: _Scenario,
    allocator: _SeedAllocator,
) -> tuple[Any, Any]:
    worker = _BooguWorker(scenario.generation)
    judge = _Judge(scenario.judge)
    monkeypatch.setattr(remove_module, "new_boogu_seed", allocator)
    backend = BooguBackgroundRemovalBackend(
        worker,
        target_area=config.reference_edit.target_area,
        alignment=config.reference_edit.alignment,
    )
    remove_backgrounds(config, storage, backend=backend, judge=judge)
    return worker, judge


def _epoch_run(
    config: V3Config,
    storage: RunStorage,
    tmp_path: Path,
    scenario: _Scenario,
    allocator: _SeedAllocator,
) -> tuple[Any, Any, list[str]]:
    ledger = GroupLedger(tmp_path / "epoch-group")
    runner = RemovalEpochRunner(
        config, {SHARD: storage}, ledger, seed_allocator=allocator
    )
    worker = _BooguWorker(scenario.generation)
    judge = _Judge(scenario.judge)
    unlocked: list[str] = []
    original = runner.finalize

    def tracked(job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        created = list(original(job, result))
        unlocked.extend(item.job_type for item in created)
        return created

    runner.finalize = tracked  # type: ignore[method-assign]
    _scheduler(ledger, runner, worker, judge).run(runner.seed_jobs())
    return worker, judge, unlocked


def _normalized(state: BackgroundReferenceState) -> dict[str, Any]:
    payload = state.model_dump(mode="json")
    for attempt in payload.get("removal_attempts", []):
        attempt.pop("runtime_seconds", None)
    return payload


def _artifact_bytes(storage: RunStorage, clip_uid: str) -> dict[str, bytes]:
    state = _state(storage, clip_uid)
    payload: dict[str, bytes] = {}
    output = storage.selected_background_output_path(clip_uid)
    payload["selected"] = output.read_bytes() if output.is_file() else b""
    for name, relative in (
        ("output", state.output_image_path),
        ("generation_mask", state.generation_mask_path),
    ):
        payload[name] = (
            (storage.root / relative).read_bytes() if relative else b""
        )
    return payload


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda case: case.name)
def test_legacy_and_epoch_agree(
    scenario: _Scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_config = _fixture_config(
        tmp_path,
        monkeypatch,
        run_name="legacy",
        max_generation_ratio=scenario.max_generation_ratio,
    )
    epoch_config = _fixture_config(
        tmp_path,
        monkeypatch,
        run_name="epoch",
        max_generation_ratio=scenario.max_generation_ratio,
    )
    legacy_storage = _pending_storage(legacy_config)
    epoch_storage = _pending_storage(epoch_config)

    legacy_worker, legacy_judge = _legacy_run(
        legacy_config, legacy_storage, monkeypatch, scenario, _SeedAllocator(scenario.seeds)
    )
    epoch_worker, epoch_judge, unlocked = _epoch_run(
        epoch_config, epoch_storage, tmp_path, scenario, _SeedAllocator(scenario.seeds)
    )

    assert len(legacy_worker.calls) == scenario.boogu_calls
    assert len(epoch_worker.calls) == scenario.boogu_calls
    assert legacy_judge.calls == scenario.qwen_calls
    assert epoch_judge.calls == scenario.qwen_calls

    legacy_state = _state(legacy_storage)
    epoch_state = _state(epoch_storage)
    assert legacy_state.status == scenario.status
    assert epoch_state.status == scenario.status
    assert legacy_state.reason == scenario.reason
    assert epoch_state.reason == scenario.reason
    assert _normalized(legacy_state) == _normalized(epoch_state)
    assert _artifact_bytes(legacy_storage, "clip-1") == _artifact_bytes(
        epoch_storage, "clip-1"
    )
    assert [call["seed"] for call in legacy_worker.calls] == [
        call["seed"] for call in epoch_worker.calls
    ]
    assert (REMOVAL_CANDIDATE2_JOB in unlocked) is scenario.candidate2


def test_first_accept_never_creates_the_second_candidate_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    storage = _pending_storage(config)
    scenario = _Scenario(
        "I-no-candidate2", (), (_accept(),), (101,), "ready_removed", None, 1, 1, False
    )
    _epoch_run(config, storage, tmp_path, scenario, _SeedAllocator(scenario.seeds))
    assert _state(storage).status == "ready_removed"


def test_retry_cycle_matches_legacy_after_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = _Scenario(
        "failures",
        (RuntimeError("boom"), RuntimeError("boom")),
        (),
        (101, 102),
        "pending_remove",
        "removal_infrastructure_failure",
        2,
        0,
        True,
    )
    legacy_config = _fixture_config(tmp_path, monkeypatch, run_name="legacy")
    epoch_config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    legacy_storage = _pending_storage(legacy_config)
    epoch_storage = _pending_storage(epoch_config)

    _legacy_run(legacy_config, legacy_storage, monkeypatch, failure, _SeedAllocator([101, 102]))
    _epoch_run(epoch_config, epoch_storage, tmp_path, failure, _SeedAllocator([101, 102]))
    assert [attempt.seed for attempt in _state(legacy_storage).removal_attempts] == [
        101,
        102,
    ]
    assert [attempt.seed for attempt in _state(epoch_storage).removal_attempts] == [
        101,
        102,
    ]

    # Second launch: the attempt index advances, so a new seed and a new job
    # identity are planned instead of colliding with the first cycle.
    recovery = _Scenario(
        "recovery", (), (_accept("clean"),), (201,), "ready_removed", None, 1, 1, False
    )
    _legacy_run(
        legacy_config,
        legacy_storage,
        monkeypatch,
        recovery,
        _SeedAllocator([201]),
    )
    epoch_worker, _, _ = _epoch_run(
        epoch_config, epoch_storage, tmp_path, recovery, _SeedAllocator([201])
    )

    assert [call["seed"] for call in epoch_worker.calls] == [201]
    assert _state(legacy_storage).status == "ready_removed"
    assert _state(epoch_storage).status == "ready_removed"
    assert [attempt.seed for attempt in _state(legacy_storage).removal_attempts] == [
        101,
        102,
        201,
    ]
    assert [attempt.seed for attempt in _state(epoch_storage).removal_attempts] == [
        101,
        102,
        201,
    ]
    assert _normalized(_state(legacy_storage)) == _normalized(_state(epoch_storage))
