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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
import r2v_data_v2.v3.post_mask_epoch_removal as epoch_removal_module
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
from r2v_data_v2.v3.post_mask_epoch_groups import (
    DEFAULT_GROUP_SIZE,
    build_groups,
    resource_epoch_root,
)
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
    PreparedRemovalShard,
    RemovalEpochError,
    RemovalEpochRunner,
    build_qwen_epoch_config,
    build_removal_campaign,
    build_removal_epoch_factories,
    build_removal_epoch_runner,
    build_removal_generate_job,
    build_removal_judge_job,
    prepare_shard_storage,
    removal_generation_resource,
    removal_seed_anchor_inputs,
    removal_shard_paths,
    resolve_removal_judge,
    seed_plan_path,
    shard_lock_path,
    validate_removal_roots,
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
    save_rejected_candidates: bool = False,
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
            save_rejected_candidates=save_rejected_candidates,
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str = "run",
    **kwargs: Any,
) -> V3Config:
    return _boogu_config(
        _base_config(tmp_path, monkeypatch, run_name=run_name, **kwargs)
    )


def _group_with(
    *shards: str, campaign: Mapping[str, Any] | None = None
) -> Any:
    """A real resource-epoch group, so ``campaign_identity`` is real too.

    Hand-rolled stand-ins have no campaign identity at all, which would let a
    test pass while the real production path never validated anything.
    """
    groups = build_groups(
        list(shards),
        campaign=dict(campaign or {}),
        group_size=max(DEFAULT_GROUP_SIZE, len(shards)),
    )
    assert len(groups) == 1
    return groups[0]


def _parts_root(tmp_path: Path, shards: Sequence[str]) -> Path:
    """A Stage2 entity-mask root whose parts/ lists exactly ``shards``."""
    root = tmp_path / "public" / "entity_mask"
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        (parts / f"{shard}.jsonl").write_text("")
    return root


def _mask() -> np.ndarray:
    value = np.zeros((HEIGHT, WIDTH), dtype=bool)
    value[2, 2] = True
    return value


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _tracked_masks(
    clip_uid: str,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    subject_mask: np.ndarray | None = None,
) -> TrackedMasksArtifact:
    """The fixture's mask artifact.

    ``subject_mask`` selects a larger attribute-ready subject geometry. Every
    frame's area, ratio, bbox and RLE stay derived from the actual mask, which is
    what makes the artifact self-consistent for any geometry. With no override the
    original single-pixel mask is used and every default result is unchanged.
    """
    mask = _mask() if subject_mask is None else np.asarray(subject_mask, dtype=bool)
    if subject_mask is not None and mask.shape != (height, width):
        raise ValueError("subject mask must match the fixture geometry")
    area_pixels = int(mask.sum())
    area_ratio = float(mask.mean())
    bbox = _bbox(mask)
    rle = encode_binary_mask(mask)
    frames = [
        TrackedMaskFrame(
            slot=slot,
            present=True,
            track_valid=True,
            confidence=0.9,
            backend_confidences=[0.9],
            backend_object_ids=["obj-1"],
            area_pixels=area_pixels,
            area_ratio=area_ratio,
            bbox_xyxy=bbox,
            rle=rle,
        )
        for slot in range(10)
    ]
    return TrackedMasksArtifact(
        clip_uid=clip_uid,
        width=width,
        height=height,
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


def _coverage(
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    subject_mask: np.ndarray | None = None,
) -> CoverageState:
    ratio = (
        1 / (WIDTH * HEIGHT)
        if subject_mask is None
        else float(np.asarray(subject_mask, dtype=bool).mean())
    )
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
                per_frame_area_ratio=[ratio] * 10,
                per_frame_confidence=[0.9] * 10,
            )
        },
    )


def _write_frames(
    storage: RunStorage,
    clip_uid: str,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    frames = []
    for slot in range(10):
        path = storage.frame_path(clip_uid, slot)
        Image.new("RGB", (width, height), (10 + slot, 20, 30)).save(
            path, format="JPEG"
        )
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
            clip_uid=clip_uid, width=width, height=height, frames=frames
        ).model_dump(mode="json"),
    )


def _pending_storage(
    config: V3Config,
    *,
    clip_uids: tuple[str, ...] = ("clip-1",),
    source_index_start: int = 0,
    width: int = WIDTH,
    height: int = HEIGHT,
    subject_mask: np.ndarray | None = None,
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    for offset, clip_uid in enumerate(clip_uids):
        index = source_index_start + offset
        video = config.dataset_json.parent / f"{clip_uid}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"target")
        storage.create_clip(
            clip_uid=clip_uid,
            source=ClipSource(
                video_path=str(video),
                parent_video_id="parent",
                clip_suffix=f"{index}_0",
                source_index=index,
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
        _write_frames(storage, clip_uid, width=width, height=height)
        storage.write_masks(
            clip_uid,
            _tracked_masks(
                clip_uid,
                width=width,
                height=height,
                subject_mask=subject_mask,
            ),
        )
        storage.write_coverage(
            clip_uid,
            _coverage(
                width=width,
                height=height,
                subject_mask=subject_mask,
            ),
        )
    stats = build_background_candidates(config, storage)
    assert stats.pending_remove == len(clip_uids)
    return storage


def _state(storage: RunStorage, clip_uid: str = "clip-1") -> BackgroundReferenceState:
    background = storage.read_clip(clip_uid).references.background
    assert background is not None
    return background


def _eligible(storage: RunStorage, clip_uids: Sequence[str] | None = None) -> tuple[str, ...]:
    """The epoch eligibility view: explicit, never ``iter_clips``-derived."""
    if clip_uids is not None:
        return tuple(clip_uids)
    return tuple(clip.clip_uid for clip in storage.iter_clips())


def _runner(
    config: V3Config,
    storage: RunStorage,
    ledger: GroupLedger,
    *,
    seed_allocator: Any = None,
    eligible_clip_uids: Sequence[str] | None = None,
) -> RemovalEpochRunner:
    eligible = {SHARD: _eligible(storage, eligible_clip_uids)}
    if seed_allocator is not None:
        return RemovalEpochRunner(
            config,
            {SHARD: storage},
            ledger,
            seed_allocator=seed_allocator,
            eligible_clip_uids_by_shard=eligible,
        )
    return RemovalEpochRunner(
        config,
        {SHARD: storage},
        ledger,
        eligible_clip_uids_by_shard=eligible,
    )


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
        RemovalEpochRunner(
            config,
            {SHARD: storage},
            GroupLedger(tmp_path / "group"),
            eligible_clip_uids_by_shard={SHARD: _eligible(storage)},
        )


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


def test_boogu_factory_passes_reference_edit_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persistent Boogu worker consumes ReferenceEditConfig, not V3Config."""
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    captured: dict[str, Any] = {}

    class _Resource:
        pass

    resource = _Resource()

    def fake_build_boogu_epoch(*, config: Any, **kwargs: Any) -> Any:
        captured["config"] = config
        captured["kwargs"] = kwargs
        return resource

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_boogu_epoch",
        fake_build_boogu_epoch,
    )
    factories = build_removal_epoch_factories(
        config,
        runner,
        qwen_epoch_config=build_qwen_epoch_config(config),
        pool=WorkerPoolConfig(),
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=tmp_path / "workspace" / "data",
    )

    built_resource, executor = factories[RESOURCE_BOOGU]()

    assert built_resource is resource
    assert isinstance(executor, WorkerSlotExecutor)
    assert captured["config"] is config.reference_edit
    assert captured["config"] is not config
    assert (
        captured["config"].python_executable
        == config.reference_edit.python_executable
    )
    assert captured["config"].code_root == config.reference_edit.code_root
    assert captured["config"].model_path == config.reference_edit.model_path


# ---------------------------------------------------------------------------
# Removal-only runner never completes a group
# ---------------------------------------------------------------------------


def test_removal_only_runner_never_reports_group_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"
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
            clip_uids=_eligible(storage),
            ready=1,
            excluded=0,
            corrupt=0,
        ),
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
    entity_mask_root = _parts_root(tmp_path, (SHARD,))
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
        SHARD,
        campaign=build_removal_campaign(config, entity_mask_root=entity_mask_root),
    )

    outcome = runner_callable(
        group,
        GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
        lambda *a, **k: None,
    )

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

    prepared = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )
    storage = prepared.storage
    clips = list(storage.iter_clips())
    assert [clip.clip_uid for clip in clips] == [clip_uid]
    assert clips[0].references.background is not None
    assert clips[0].references.background.status == "pending_remove"
    assert prepared.clip_uids == (clip_uid,)
    assert prepared.ready == 1
    assert prepared.excluded == 0
    assert prepared.corrupt == 0

    runner = RemovalEpochRunner(
        config,
        {shard: storage},
        GroupLedger(tmp_path / "group"),
        eligible_clip_uids_by_shard={shard: prepared.clip_uids},
    )
    assert len(runner.seed_jobs()) == 1


def test_hydrated_root_is_reused_without_duplicating_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity_mask_root, shard, clip_uid = _stage2_fixture(tmp_path, monkeypatch)
    config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"

    prepared = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )
    assert prepared.clip_uids == (clip_uid,)
    assert [clip.clip_uid for clip in prepared.storage.iter_clips()] == [clip_uid]

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
        config,
        {SHARD: storage},
        ledger,
        seed_allocator=allocator,
        eligible_clip_uids_by_shard={SHARD: _eligible(storage)},
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


# ---------------------------------------------------------------------------
# Shard lock lifetime
# ---------------------------------------------------------------------------


SHARD_A = "shard-000000000-000000000"
SHARD_B = "shard-000000001-000000999"


class _LockRecorder:
    """Stand-in for ``file_lock`` that records the whole lock lifetime.

    The legacy Post-Mask worker holds ``state_root/shard.lock`` from hydration
    through publication. The resource-epoch runner must do the same for every
    canonical shard of the group, so this records when each lock is taken, what
    is still held while a model runs, and when everything is released.
    """

    def __init__(self, *, fail_on: Sequence[str] = ()) -> None:
        self.events: list[str] = []
        self.fail_on = set(fail_on)
        self.held: list[str] = []
        self.held_during_model: list[list[str]] = []

    def __call__(self, path: Path, *, blocking: bool = True) -> Any:
        recorder = self
        name = path.parent.name

        @contextmanager
        def lock() -> Iterator[bool]:
            if name in recorder.fail_on:
                recorder.events.append(f"lock-unavailable {name}")
                yield False
                return
            recorder.events.append(f"acquire {name}")
            recorder.held.append(name)
            try:
                yield True
            finally:
                recorder.held.remove(name)
                recorder.events.append(f"release {name}")

        return lock()


class _TracingWorker:
    """Boogu worker that snapshots the lock state on every model call."""

    def __init__(
        self, recorder: _LockRecorder, responses: Sequence[Any] = ()
    ) -> None:
        self.recorder = recorder
        self.inner = _BooguWorker(responses)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.inner.calls

    def edit(self, **kwargs: Any) -> BooguEditOutput:
        self.recorder.events.append("boogu")
        self.recorder.held_during_model.append(list(self.recorder.held))
        return self.inner.edit(**kwargs)

    def close(self) -> None:
        self.inner.close()


class _TracingJudge:
    def __init__(
        self, recorder: _LockRecorder, responses: Sequence[Any] = ()
    ) -> None:
        self.recorder = recorder
        self.inner = _Judge(responses)

    @property
    def calls(self) -> int:
        return self.inner.calls

    def review(self, **kwargs: Any) -> BackgroundRemovalReview:
        self.recorder.events.append("qwen")
        self.recorder.held_during_model.append(list(self.recorder.held))
        return self.inner.review(**kwargs)


class _FakeEpochResource:
    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def counters(self) -> dict[str, Any]:
        return {}


def _install_fake_factories(
    monkeypatch: pytest.MonkeyPatch, worker: Any, judge: Any
) -> None:
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.build_removal_epoch_factories",
        lambda config, runner, **kwargs: {
            RESOURCE_BOOGU: lambda: (
                _FakeEpochResource(),
                SerialBatchExecutor(runner.run, worker),
            ),
            RESOURCE_QWEN: lambda: (
                _FakeEpochResource(),
                SerialBatchExecutor(runner.run, judge),
            ),
        },
    )


def _install_prepared_shards(
    monkeypatch: pytest.MonkeyPatch,
    storages: Mapping[str, RunStorage],
    recorder: _LockRecorder,
    *,
    clip_uids_by_shard: Mapping[str, Sequence[str]] | None = None,
) -> None:
    def fake_prepare(
        config: V3Config,
        *,
        post_mask_root: Path,
        entity_mask_root: Path,
        shard: str,
        git_commit: str,
    ) -> PreparedRemovalShard:
        recorder.events.append(f"hydrate {shard}")
        storage = storages[shard]
        wanted = (clip_uids_by_shard or {}).get(shard)
        return PreparedRemovalShard(
            shard=shard,
            paths=removal_shard_paths(
                post_mask_root=post_mask_root,
                entity_mask_root=entity_mask_root,
                shard=shard,
            ),
            storage=storage,
            clip_uids=_eligible(storage, wanted),
            ready=1,
            excluded=0,
            corrupt=0,
        )

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.prepare_shard_storage", fake_prepare
    )


def _trace_mutations(monkeypatch: pytest.MonkeyPatch, recorder: _LockRecorder) -> None:
    real_publish = epoch_removal_module._publish_ready

    def traced_publish(*args: Any, **kwargs: Any) -> Any:
        recorder.events.append("publish")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(epoch_removal_module, "_publish_ready", traced_publish)

    real_counts = RunStorage.update_stage_counts

    def traced_counts(self: RunStorage, stage: str, counts: dict[str, int]) -> Any:
        recorder.events.append("stage_counts")
        return real_counts(self, stage, counts)

    monkeypatch.setattr(RunStorage, "update_stage_counts", traced_counts)


def _two_shard_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    recorder: _LockRecorder,
    fail: bool = False,
) -> tuple[dict[str, Any], _TracingWorker, _TracingJudge, RunStorage]:
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    config = _fixture_config(tmp_path, monkeypatch, run_name="shard-a")
    other = _fixture_config(tmp_path, monkeypatch, run_name="shard-b")
    storages = {
        SHARD_A: _pending_storage(config, clip_uids=("clip-1",)),
        SHARD_B: _pending_storage(other, clip_uids=("clip-1",)),
    }
    worker = _TracingWorker(recorder)
    judge = _TracingJudge(recorder, [_accept(), _accept()])
    _install_prepared_shards(monkeypatch, storages, recorder)
    _install_fake_factories(monkeypatch, worker, judge)
    _trace_mutations(monkeypatch, recorder)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    entity_mask_root = _parts_root(tmp_path, (SHARD_A, SHARD_B))
    campaign = build_removal_campaign(config, entity_mask_root=entity_mask_root)
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(SHARD_A, SHARD_B, campaign=campaign)
    ledger = GroupLedger(resource_epoch_root(post_mask_root) / group.group_id)
    outcome = runner_callable(group, ledger, lambda *a, **k: None)
    del fail
    return outcome, worker, judge, storages[SHARD_A]


def test_removal_epoch_holds_every_shard_lock_across_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LockRecorder()
    outcome, worker, judge, storage = _two_shard_run(
        tmp_path, monkeypatch, recorder=recorder
    )

    assert outcome["remove_completed"] is True
    trace = recorder.events
    # Phase 1: every canonical shard lock, in lexical shard order, before any
    # shard is touched.
    assert trace[0] == f"acquire {SHARD_A}"
    assert trace[1] == f"acquire {SHARD_B}"
    assert trace.index(f"hydrate {SHARD_A}") > trace.index(f"acquire {SHARD_B}")
    # Model calls, publication and the stage-count update all happen while the
    # locks are still held.
    assert trace.index("boogu") > trace.index(f"hydrate {SHARD_B}")
    assert trace.index("qwen") > trace.index("boogu")
    assert trace.index("publish") > trace.index("qwen")
    assert trace.index("stage_counts") > trace.index("publish")
    assert trace[-2] == f"release {SHARD_B}"
    assert trace[-1] == f"release {SHARD_A}"
    assert recorder.held_during_model, "no model call was observed"
    assert all(
        held == [SHARD_A, SHARD_B] for held in recorder.held_during_model
    ), "a model call ran without every shard lock held"
    assert len(worker.calls) == 2
    assert judge.calls == 2
    assert _state(storage).status == "ready_removed"


def test_second_shard_lock_unavailable_mutates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _LockRecorder(fail_on=(SHARD_B,))
    config = _fixture_config(tmp_path, monkeypatch, run_name="shard-a")
    other = _fixture_config(tmp_path, monkeypatch, run_name="shard-b")
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    storages = {
        SHARD_A: _pending_storage(config, clip_uids=("clip-1",)),
        SHARD_B: _pending_storage(other, clip_uids=("clip-1",)),
    }
    worker = _TracingWorker(recorder)
    judge = _TracingJudge(recorder, [_accept()])
    _install_prepared_shards(monkeypatch, storages, recorder)
    _install_fake_factories(monkeypatch, worker, judge)
    _trace_mutations(monkeypatch, recorder)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    entity_mask_root = _parts_root(tmp_path, (SHARD_A, SHARD_B))
    campaign = build_removal_campaign(config, entity_mask_root=entity_mask_root)
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(SHARD_A, SHARD_B, campaign=campaign)

    with pytest.raises(RemovalEpochError, match="locked elsewhere"):
        runner_callable(
            group,
            GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
            lambda *a, **k: None,
        )

    assert recorder.events == [
        f"acquire {SHARD_A}",
        f"lock-unavailable {SHARD_B}",
        f"release {SHARD_A}",
    ]
    assert not any("hydrate" in event for event in recorder.events)
    assert worker.calls == [], "no generation may run without every shard lock"
    assert judge.calls == 0, "no judge may run without every shard lock"
    assert "publish" not in recorder.events
    assert "stage_counts" not in recorder.events
    assert recorder.held == [], "the acquired lock must be released again"
    assert _state(storages[SHARD_A]).status == "pending_remove"


def test_epoch_shard_lock_is_the_legacy_worker_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    entity_mask_root = writable / "stage2" / "entity_mask"
    legacy_paths = ShardPaths.for_shard(
        post_mask_root, entity_mask_root / "parts" / f"{SHARD}.jsonl"
    )
    epoch_paths = removal_shard_paths(
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=SHARD,
    )

    assert shard_lock_path(epoch_paths) == legacy_paths.state_root / "shard.lock"
    assert shard_lock_path(epoch_paths) == epoch_paths.state_root / "shard.lock"
    assert "resource_epoch" not in str(shard_lock_path(epoch_paths))
    assert config is not None


# ---------------------------------------------------------------------------
# Eligible clip view
# ---------------------------------------------------------------------------


def test_seed_jobs_only_reads_the_explicit_eligible_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config, clip_uids=("clip-A", "clip-B"))
    runner = _runner(
        config,
        storage,
        GroupLedger(tmp_path / "group"),
        eligible_clip_uids=("clip-A",),
    )

    jobs = runner.seed_jobs()

    assert {job.clip_uid for job in jobs} == {"clip-A"}
    assert {clip.clip_uid for clip in storage.iter_clips()} == {"clip-A", "clip-B"}
    assert _state(storage, "clip-B").status == "pending_remove"


def test_corrupt_historical_clip_is_never_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":  # pragma: no cover - known production module bug
        pytest.skip("hydrate_shard re-entry is not Windows-safe")
    entity_mask_root, shard = _stage2_two_clip_fixture(tmp_path, monkeypatch)
    config = _fixture_config(tmp_path, monkeypatch, run_name="epoch")
    post_mask_root = tmp_path / "workspace" / "data" / "campaign"

    first = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )
    assert first.clip_uids == ("clip-A", "clip-B"), (
        first.paths.input_failures_path.read_text()
    )
    assert first.corrupt == 0

    # The second launch sees clip-B as corrupt: its frozen Stage2 manifest is
    # gone, so hydration admits only clip-A this time.
    #
    # The destination mirror is damaged as well, on purpose. A restart answers
    # "already hydrated" from the destination alone and never reopens Stage2, so
    # a source-only change is invisible by design; a destination that can no
    # longer prove itself falls back to the full path, and that path is what
    # discovers the missing frozen manifest.
    corrupt_source = (
        entity_mask_root
        / "artifacts"
        / shard
        / "clip-B"
        / "run"
        / "clips"
        / "clip-B"
        / "frames"
        / "frames.json"
    )
    corrupt_source.unlink()
    (first.storage.root / "clips" / "clip-B" / "frames" / "frames.json").unlink()

    second = prepare_shard_storage(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        shard=shard,
        git_commit="test",
    )

    assert second.clip_uids == ("clip-A",)
    assert second.corrupt == 1
    # clip-B is still on disk from the first launch ...
    assert (second.storage.root / "clips" / "clip-B").is_dir()
    assert {clip.clip_uid for clip in second.storage.iter_clips()} == {
        "clip-A",
        "clip-B",
    }
    # ... but this launch's hydration result is the only eligibility source.
    runner = RemovalEpochRunner(
        config,
        {shard: second.storage},
        GroupLedger(tmp_path / "group"),
        eligible_clip_uids_by_shard={shard: second.clip_uids},
    )
    jobs = runner.seed_jobs()
    assert {job.clip_uid for job in jobs} == {"clip-A"}


def _stage2_two_clip_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shard: str = SHARD,
    clip_uids: tuple[str, ...] = ("clip-A", "clip-B"),
) -> tuple[Path, str]:
    """One canonical Stage2 shard whose parts list two ready clips."""
    entity_mask_root = tmp_path / "public" / "entity_mask"
    parts = entity_mask_root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index, uid in enumerate(clip_uids):
        source_config = _fixture_config(tmp_path, monkeypatch, run_name=f"stage2-{uid}")
        storage = _pending_storage(
            source_config, clip_uids=(uid,), source_index_start=index
        )
        destination = entity_mask_root / "artifacts" / shard / uid / "run"
        destination.parent.mkdir(parents=True, exist_ok=True)
        storage.root.rename(destination)
    rows = [
        json.dumps(
            {
                "schema_version": "r2v.v3.pre_qwen_stage2.1",
                "source_index": index,
                "clip_uid": uid,
                "status": "ready_background_pending_remove",
                "coverage_passed": True,
                "background_status": "pending_remove",
                "reason": None,
                "artifact_root": f"artifacts/{shard}/{uid}",
            }
        )
        for index, uid in enumerate(clip_uids)
    ]
    (parts / f"{shard}.jsonl").write_text("\n".join(rows) + "\n")
    return entity_mask_root, shard


# ---------------------------------------------------------------------------
# Qwen judge client ownership
# ---------------------------------------------------------------------------


class _SharedJudge:
    """A judge handle the runner does not own, even though it can be closed."""

    def __init__(self, response: Any = None) -> None:
        self.response = response
        self.review_calls = 0
        self.close_calls = 0

    def review(self, **kwargs: Any) -> BackgroundRemovalReview:
        self.review_calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return _accept()

    def close(self) -> None:
        self.close_calls += 1


def _install_endpoint_judge(monkeypatch: pytest.MonkeyPatch, response: Any) -> list[Any]:
    created: list[Any] = []

    class _FakeEndpointJudge:
        def __init__(self, service: Any) -> None:
            self.service = service
            self.review_calls = 0
            self.close_calls = 0
            created.append(self)

        def review(self, **kwargs: Any) -> BackgroundRemovalReview:
            self.review_calls += 1
            if isinstance(response, Exception):
                raise response
            return _accept()

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(
        epoch_removal_module, "QwenBackgroundRemovalJudge", _FakeEndpointJudge
    )
    return created


def _judge_job_for(
    config: V3Config,
    storage: RunStorage,
    ledger: GroupLedger,
) -> tuple[RemovalEpochRunner, ModelJob]:
    runner = _runner(config, storage, ledger)
    (generate_job,) = runner.seed_jobs()
    worker = _BooguWorker()
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
    return runner, judge_job


def test_endpoint_judge_client_is_closed_after_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    created = _install_endpoint_judge(monkeypatch, None)
    runner, judge_job = _judge_job_for(config, storage, ledger)

    result = runner.run(judge_job, "http://127.0.0.1:8000/v1")

    assert result.outcome == OUTCOME_COMPLETED
    assert len(created) == 1
    assert created[0].review_calls == 1
    assert created[0].close_calls == 1, "an endpoint client must always be closed"


def test_endpoint_judge_client_is_closed_when_review_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    created = _install_endpoint_judge(monkeypatch, RuntimeError("judge down"))
    runner, judge_job = _judge_job_for(config, storage, ledger)

    result = runner.run(judge_job, "http://127.0.0.1:8000/v1")

    assert result.outcome == OUTCOME_COMPLETED
    assert len(created) == 1
    assert created[0].review_calls == 1
    assert created[0].close_calls == 1, "close must survive a review exception"


def test_shared_judge_handle_is_never_closed_by_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    _install_endpoint_judge(monkeypatch, None)
    runner, judge_job = _judge_job_for(config, storage, ledger)
    shared = _SharedJudge()

    runner.run(judge_job, shared)

    assert shared.review_calls == 1
    assert shared.close_calls == 0, "a shared handle belongs to its owner"
    resolved = resolve_removal_judge(shared, config)
    assert resolved.judge is shared
    assert resolved.owned is False


# ---------------------------------------------------------------------------
# Launcher / runner root consistency
# ---------------------------------------------------------------------------


def test_runner_refuses_a_ledger_root_outside_the_launcher_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    writable = tmp_path / "workspace" / "data"
    campaign_a = writable / "campaign-A"
    campaign_b = writable / "campaign-B"
    entity_mask_root = _parts_root(tmp_path, (SHARD,))
    recorder = _LockRecorder()
    _install_prepared_shards(monkeypatch, {SHARD: storage}, recorder)
    _install_fake_factories(monkeypatch, _BooguWorker(), _Judge([_accept()]))
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=campaign_b,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    # The campaign matches, so the ledger root is what must fail.
    group = _group_with(
        SHARD, campaign=build_removal_campaign(config, entity_mask_root=entity_mask_root)
    )

    with pytest.raises(RemovalEpochError, match="Post-Mask root"):
        runner_callable(
            group,
            GroupLedger(resource_epoch_root(campaign_a) / "group-X"),
            lambda *a, **k: None,
        )

    assert not any("hydrate" in event for event in recorder.events)
    assert recorder.events == [], "root validation runs before any lock"
    assert _state(storage).status == "pending_remove"


def test_runner_refuses_an_entity_mask_root_that_is_not_the_campaign(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign-A"
    ledger = GroupLedger(resource_epoch_root(root) / "group-X")

    with pytest.raises(RemovalEpochError, match="entity mask root"):
        validate_removal_roots(
            ledger,
            group_id="group-X",
            post_mask_root=root,
            entity_mask_root=tmp_path / "stage2-A",
            campaign={"entity_mask_root": str(tmp_path / "stage2-B")},
        )


def test_matching_roots_validate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign-A"
    ledger = GroupLedger(resource_epoch_root(root) / "group-X")
    entity_mask_root = tmp_path / "stage2-A"

    validate_removal_roots(
        ledger,
        group_id="group-X",
        post_mask_root=root,
        entity_mask_root=entity_mask_root,
        campaign={"entity_mask_root": str(entity_mask_root)},
    )


# ---------------------------------------------------------------------------
# Campaign identity (real runner path, not just the helper)
# ---------------------------------------------------------------------------


def _load_launcher_module() -> Any:
    """Import tools/run_v3_post_mask_resource_epoch.py by path."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_v3_post_mask_resource_epoch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_v3_post_mask_resource_epoch", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_and_launcher_build_the_same_campaign_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    entity_mask_root = _parts_root(tmp_path, (SHARD_A, SHARD_B))
    launcher = _load_launcher_module()

    from_launcher = launcher.build_campaign(
        config,
        entity_mask_root=entity_mask_root,
        canonical_shard_count=2,
    )
    from_runner = build_removal_campaign(config, entity_mask_root=entity_mask_root)

    assert from_runner == from_launcher


def _campaign_drift_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    group_config: V3Config | None = None,
    group_entity_mask_root: Path | None = None,
) -> tuple[_LockRecorder, _TracingWorker, _TracingJudge, Any]:
    """Run the real group runner with a live campaign the group never saw."""
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    config = _fixture_config(tmp_path, monkeypatch, run_name="runner")
    entity_mask_root = _parts_root(tmp_path, (SHARD,))
    storage = _pending_storage(config, clip_uids=("clip-1",))
    recorder = _LockRecorder()
    worker = _TracingWorker(recorder)
    judge = _TracingJudge(recorder, [_accept()])
    _install_prepared_shards(monkeypatch, {SHARD: storage}, recorder)
    _install_fake_factories(monkeypatch, worker, judge)
    _trace_mutations(monkeypatch, recorder)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    group = _group_with(
        SHARD,
        campaign=build_removal_campaign(
            group_config or config,
            entity_mask_root=group_entity_mask_root or entity_mask_root,
        ),
    )
    with pytest.raises(RemovalEpochError, match="campaign identity"):
        runner_callable(
            group,
            GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
            lambda *a, **k: None,
        )
    return recorder, worker, judge, storage


def test_runner_rejects_entity_mask_root_drift_before_any_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_root = _parts_root(tmp_path / "elsewhere", (SHARD,))
    recorder, worker, judge, storage = _campaign_drift_case(
        tmp_path, monkeypatch, group_entity_mask_root=other_root
    )

    assert recorder.events == [], "campaign validation runs before the first lock"
    assert worker.calls == []
    assert judge.calls == 0
    assert _state(storage).status == "pending_remove"


def test_runner_rejects_config_fingerprint_drift_before_any_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = _fixture_config(
        tmp_path, monkeypatch, run_name="drifted", max_generation_ratio=0.5
    )
    recorder, worker, judge, storage = _campaign_drift_case(
        tmp_path, monkeypatch, group_config=drifted
    )

    assert recorder.events == []
    assert worker.calls == []
    assert judge.calls == 0
    assert _state(storage).status == "pending_remove"


def test_runner_rejects_canonical_shard_count_drift_before_any_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _parts_root(tmp_path, (SHARD,))
    config = _fixture_config(tmp_path, monkeypatch)
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    storage = _pending_storage(config, clip_uids=("clip-1",))
    recorder = _LockRecorder()
    worker = _TracingWorker(recorder)
    judge = _TracingJudge(recorder, [_accept()])
    _install_prepared_shards(monkeypatch, {SHARD: storage}, recorder)
    _install_fake_factories(monkeypatch, worker, judge)
    _trace_mutations(monkeypatch, recorder)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
        qwen_epoch_config=build_qwen_epoch_config(config),
    )
    # The group was built when the campaign had exactly one canonical shard.
    group = _group_with(
        SHARD,
        campaign=build_removal_campaign(config, entity_mask_root=root),
    )
    # A second canonical shard appears in the same Stage2 root afterwards.
    (root / "parts" / f"{SHARD_B}.jsonl").write_text("")

    with pytest.raises(RemovalEpochError, match="campaign identity"):
        runner_callable(
            group,
            GroupLedger(resource_epoch_root(post_mask_root) / group.group_id),
            lambda *a, **k: None,
        )

    assert recorder.events == []
    assert worker.calls == []
    assert judge.calls == 0


def test_matching_campaign_reaches_the_first_shard_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path, monkeypatch)
    entity_mask_root = _parts_root(tmp_path, (SHARD,))
    writable = tmp_path / "workspace" / "data"
    post_mask_root = writable / "campaign"
    storage = _pending_storage(config, clip_uids=("clip-1",))
    recorder = _LockRecorder()
    worker = _TracingWorker(recorder)
    judge = _TracingJudge(recorder, [_accept()])
    _install_prepared_shards(monkeypatch, {SHARD: storage}, recorder)
    _install_fake_factories(monkeypatch, worker, judge)
    _trace_mutations(monkeypatch, recorder)
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_removal.file_lock", recorder
    )
    runner_callable = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=_fake_process_manager(),
        temporary_root=tmp_path / "tmp",
        allowed_server_root=writable,
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

    assert outcome["remove_completed"] is True
    assert recorder.events[0] == f"acquire {SHARD}"


# ---------------------------------------------------------------------------
# Debug semantic equivalence
# ---------------------------------------------------------------------------


def _debug_tree(storage: RunStorage, clip_uid: str) -> dict[str, Any]:
    directory = storage.remove_debug_dir(clip_uid)
    payload: dict[str, Any] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".json":
            payload[path.name] = json.loads(path.read_text())
        else:
            payload[path.name] = path.read_bytes()
    return payload


def test_rejected_candidate_debug_artifacts_match_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _Scenario(
        "B-reject-accept",
        (),
        (_reject("seam"), _accept("clean")),
        (101, 202),
        "ready_removed",
        None,
        2,
        2,
        True,
    )
    legacy_config = _fixture_config(
        tmp_path,
        monkeypatch,
        run_name="legacy",
        save_rejected_candidates=True,
    )
    epoch_config = _fixture_config(
        tmp_path,
        monkeypatch,
        run_name="epoch",
        save_rejected_candidates=True,
    )
    legacy_storage = _pending_storage(legacy_config)
    epoch_storage = _pending_storage(epoch_config)

    _legacy_run(
        legacy_config,
        legacy_storage,
        monkeypatch,
        scenario,
        _SeedAllocator(scenario.seeds),
    )
    _epoch_run(
        epoch_config,
        epoch_storage,
        tmp_path,
        scenario,
        _SeedAllocator(scenario.seeds),
    )

    legacy_debug = _debug_tree(legacy_storage, "clip-1")
    epoch_debug = _debug_tree(epoch_storage, "clip-1")

    assert set(legacy_debug) == set(epoch_debug)
    assert "candidate_seed_101.png" in legacy_debug, "the rejected candidate debug"
    assert "candidate_seed_202.png" not in legacy_debug, "accepted candidates do not"
    for name in sorted(legacy_debug):
        assert legacy_debug[name] == epoch_debug[name], f"debug mismatch: {name}"
