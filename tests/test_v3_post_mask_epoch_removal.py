"""Tests for the resource-epoch background-removal adapter.

These tests drive the real scheduler, ledger and shared removal semantics with
fake model handles. Nothing here patches a prompt, a threshold or an
accept/reject rule: if the adapter grew its own copy of one, these tests would
stop describing production.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.config as config_module
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
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
)
from r2v_data_v2.v3.post_mask_epoch_removal import (
    CANDIDATE_ARTIFACT,
    PAYLOAD_ATTEMPT,
    PAYLOAD_CANDIDATE_INDEX,
    REMOVAL_CANDIDATE2_JOB,
    REMOVAL_GENERATE_JOB,
    REMOVAL_JUDGE_JOB,
    RemovalEpochError,
    RemovalEpochHandles,
    RemovalEpochRunner,
    build_removal_generate_job,
    removal_generation_resource,
    removal_seed,
    resolve_removal_backend,
    resolve_removal_judge,
)
from r2v_data_v2.v3.post_mask_epoch_resources import SerialBatchExecutor
from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.reference_edit_boogu import BooguEditOutput
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    AnnotationState,
    BackgroundAnnotation,
    BackgroundRemovalReview,
    ClipSource,
    CoverageState,
    EntityReferenceState,
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


# ---------------------------------------------------------------------------
# Fixtures
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


class _Backend:
    def __init__(self, responses: list[Image.Image | Exception] | None = None) -> None:
        self.responses = iter(responses or [])
        self.seeds: list[int] = []
        self.closed = False

    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        self.seeds.append(seed)
        try:
            response = next(self.responses)
        except StopIteration:
            response = Image.new("RGB", image.size, (220, 30, 40))
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class _Judge:
    def __init__(
        self, responses: list[BackgroundRemovalReview | Exception] | None = None
    ) -> None:
        self.responses = iter(responses or [])
        self.calls = 0
        self.closed = False

    def review(self, **kwargs: object) -> BackgroundRemovalReview:
        self.calls += 1
        try:
            response = next(self.responses)
        except StopIteration:
            response = _accept()
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class _BooguWorker:
    """Duck-typed persistent Boogu worker: exposes ``edit``, not ``remove``."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def edit(self, **kwargs: object) -> BooguEditOutput:
        self.calls.append(dict(kwargs))
        width = int(kwargs["width"])
        height = int(kwargs["height"])
        buffer = BytesIO()
        Image.new("RGB", (width, height), (220, 30, 40)).save(buffer, format="PNG")
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


def _config(
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
        Image.new("RGB", (WIDTH, HEIGHT), (10 + slot, 20, 30)).save(
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
            clip_uid=clip_uid, width=WIDTH, height=HEIGHT, frames=frames
        ).model_dump(mode="json"),
    )


def _entity_reference() -> EntityReferenceState:
    return EntityReferenceState(
        entity_id="e1",
        status="ready",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="complete",
        image_path="selected/entity_e1.png",
        source_frame_index=0,
    )


def _pending_storage(
    config: V3Config,
    *,
    clip_uids: tuple[str, ...] = ("clip-1",),
) -> RunStorage:
    storage = RunStorage(config)
    storage.initialize(git_commit="abc123")
    for clip_uid in clip_uids:
        storage.create_clip(
            clip_uid=clip_uid,
            source=ClipSource(
                video_path=str(config.dataset_json.parent / f"{clip_uid}.mp4"),
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


def _state(storage: RunStorage, clip_uid: str = "clip-1"):
    background = storage.read_clip(clip_uid).references.background
    assert background is not None
    return background


def _runner(
    config: V3Config,
    storage: RunStorage,
    ledger: GroupLedger,
    shard: str = "shard-00000",
) -> RemovalEpochRunner:
    return RemovalEpochRunner(config, {shard: storage}, ledger)


def _scheduler(
    ledger: GroupLedger, runner: RemovalEpochRunner, handles: object
) -> ResourceEpochScheduler:
    return ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={
            RESOURCE_QWEN: SerialBatchExecutor(runner.run, handles),
            RESOURCE_BOOGU: SerialBatchExecutor(runner.run, handles),
        },
    )


# ---------------------------------------------------------------------------
# Job planning
# ---------------------------------------------------------------------------


def test_seed_jobs_plans_only_the_unconditional_first_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))

    jobs = runner.seed_jobs()

    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_type == REMOVAL_GENERATE_JOB
    assert job.resource == removal_generation_resource(config)
    assert job.canonical_shard == "shard-00000"
    assert job.clip_uid == "clip-1"
    assert job.attempt_index == 0
    assert job.seed == config.remove.candidate_seeds[0]


def test_seed_jobs_identity_is_stable_across_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)

    first = _runner(config, storage, GroupLedger(tmp_path / "g1")).seed_jobs()
    second = _runner(config, storage, GroupLedger(tmp_path / "g2")).seed_jobs()

    assert [job.job_id() for job in first] == [job.job_id() for job in second]


def test_boogu_generation_uses_the_boogu_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))
    storage = _pending_storage(config)

    jobs = _runner(config, storage, GroupLedger(tmp_path / "group")).seed_jobs()

    assert jobs[0].resource == RESOURCE_BOOGU
    assert jobs[0].model_identity == str(config.reference_edit.model_path)


def test_boogu_seed_is_deterministic_and_clip_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))

    first = removal_seed(config, "clip-1", 0)
    assert first == removal_seed(config, "clip-1", 0)
    assert first != removal_seed(config, "clip-2", 0)
    assert first != removal_seed(config, "clip-1", 1)
    assert 0 <= first < 2**31
    assert removal_seed(_config(tmp_path, monkeypatch), "clip-1", 1) == 17


def test_seed_jobs_rejects_an_oversized_generation_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch, max_generation_ratio=0.001)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))

    assert runner.seed_jobs() == []
    assert _state(storage).status == "rejected"
    assert _state(storage).reason == "generation_mask_too_large"
    assert runner.stats["shard-00000"]["rejected"] == 1


def test_seed_jobs_skips_clips_that_are_not_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))

    assert len(runner.seed_jobs()) == 1
    # A clip already published by the legacy path is never re-planned.
    references = storage.read_clip("clip-1").references
    storage.write_references(
        "clip-1",
        references.model_copy(
            update={
                "background": _state(storage).model_copy(
                    update={"status": "rejected", "reason": "published elsewhere"}
                )
            }
        ),
    )
    assert runner.seed_jobs() == []


# ---------------------------------------------------------------------------
# Handle resolution
# ---------------------------------------------------------------------------


def test_resolve_removal_backend_wraps_a_persistent_boogu_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _boogu_config(_config(tmp_path, monkeypatch))
    worker = _BooguWorker()

    backend = resolve_removal_backend(worker, config)

    assert isinstance(backend, BooguBackgroundRemovalBackend)
    assert backend.target_area == config.reference_edit.target_area
    assert backend.alignment == config.reference_edit.alignment


def test_resolve_removal_backend_and_judge_read_a_handles_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    backend = _Backend()
    judge = _Judge()
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    assert resolve_removal_backend(handles, config) is backend
    assert resolve_removal_judge(handles, config) is judge


def test_resolve_removal_backend_rejects_an_unusable_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)

    with pytest.raises(RemovalEpochError):
        resolve_removal_backend(object(), config)
    with pytest.raises(RemovalEpochError):
        resolve_removal_judge(object(), config)


# ---------------------------------------------------------------------------
# End-to-end through the scheduler
# ---------------------------------------------------------------------------


def test_accepted_candidate_publishes_ready_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    backend = _Backend()
    judge = _Judge([_accept()])
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    outcome = _scheduler(ledger, runner, handles).run(runner.seed_jobs())

    assert outcome["completed"] is True
    assert backend.seeds == [0]
    assert judge.calls == 1
    published = _state(storage)
    assert published.status == "ready_removed"
    assert published.output_image_path is not None
    assert published.output_sha256 is not None
    assert len(published.removal_attempts) == 1
    assert published.removal_attempts[0].status == "accepted"
    assert published.removal_attempts[0].review is not None
    assert published.removal_attempts[0].runtime_seconds > 0.0
    assert storage.selected_background_output_path("clip-1").is_file()
    assert runner.stats["shard-00000"]["ready_removed"] == 1
    assert runner.stats["shard-00000"]["candidates_generated"] == 1


def test_rejected_first_candidate_unlocks_the_second_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    backend = _Backend()
    judge = _Judge([_reject(), _accept("second try is clean")])
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    outcome = _scheduler(ledger, runner, handles).run(runner.seed_jobs())

    assert outcome["completed"] is True
    assert backend.seeds == [0, 17]
    assert judge.calls == 2
    published = _state(storage)
    assert published.status == "ready_removed"
    assert [attempt.status for attempt in published.removal_attempts] == [
        "rejected",
        "accepted",
    ]
    assert runner.stats["shard-00000"]["candidates_rejected"] == 1


def test_every_candidate_rejected_publishes_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    backend = _Backend()
    judge = _Judge([_reject("seam"), _reject("artifact")])
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    outcome = _scheduler(ledger, runner, handles).run(runner.seed_jobs())

    assert outcome["completed"] is True
    assert backend.seeds == [0, 17]
    published = _state(storage)
    assert published.status == "rejected"
    assert published.reason == "all_removal_candidates_rejected"
    assert [attempt.status for attempt in published.removal_attempts] == [
        "rejected",
        "rejected",
    ]
    assert runner.stats["shard-00000"]["rejected"] == 1


def test_boogu_generation_and_qwen_judge_run_as_two_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Boogu worker epoch and the Qwen judge epoch are separate epochs."""
    config = _boogu_config(_config(tmp_path, monkeypatch))
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    worker = _BooguWorker()
    judge = _Judge([_accept()])
    scheduler = ResourceEpochScheduler(
        ledger=ledger,
        finalize=runner.finalize,
        executors={
            RESOURCE_BOOGU: SerialBatchExecutor(runner.run, worker),
            RESOURCE_QWEN: SerialBatchExecutor(runner.run, judge),
        },
    )

    outcome = scheduler.run(runner.seed_jobs())

    assert outcome["completed"] is True
    assert len(worker.calls) == 1
    assert worker.calls[0]["thinking_enabled"] is False
    assert judge.calls == 1
    published = _state(storage)
    assert published.status == "ready_removed"
    assert published.removal_backend == BOOGU_REMOVE_BACKEND
    assert published.removal_seed == removal_seed(config, "clip-1", 0)


def test_generation_failure_is_retryable_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    backend = _Backend([RuntimeError("boom")])
    judge = _Judge()
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    outcome = _scheduler(ledger, runner, handles).run(runner.seed_jobs())

    assert outcome["completed"] is False
    assert outcome["resolved_job_count"] == 0
    assert judge.calls == 0
    assert _state(storage).status == "pending_remove"
    assert runner.stats["shard-00000"]["ready_removed"] == 0


def test_generation_failure_returns_a_retryable_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    runner = _runner(config, storage, GroupLedger(tmp_path / "group"))
    handles = RemovalEpochHandles(removal_backend=_Backend([RuntimeError("boom")]))

    result = runner.run(runner.seed_jobs()[0], handles)

    assert result.outcome == OUTCOME_RETRYABLE_FAILED
    assert result.committed is False
    assert result.detail == "boom"
    assert result.artifacts == {}


def test_judge_failure_is_retryable_and_unlocks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    backend = _Backend()
    judge = _Judge([RuntimeError("judge unavailable")])
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)

    outcome = _scheduler(ledger, runner, handles).run(runner.seed_jobs())

    assert outcome["completed"] is False
    assert backend.seeds == [0]
    assert judge.calls == 1
    assert _state(storage).status == "pending_remove"


def test_generate_then_judge_commits_with_expected_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    handles = RemovalEpochHandles(
        removal_backend=_Backend(), judge=_Judge([_reject("seam")])
    )
    generate = runner.seed_jobs()[0]

    generation = runner.run(generate, handles)

    assert generation.outcome == OUTCOME_COMPLETED
    assert CANDIDATE_ARTIFACT in generation.artifacts
    phase = ledger.phase("manual")
    digest = phase.publish_artifact(
        generate.job_id(), CANDIDATE_ARTIFACT, generation.artifacts[CANDIDATE_ARTIFACT]
    )
    result_digest = phase.publish_result(generate, generation)
    receipt = phase.commit(
        generate,
        outcome=generation.outcome,
        artifact_digests={
            CANDIDATE_ARTIFACT: digest,
            "result.json": result_digest,
        },
        result_digest=result_digest,
    )
    ledger.note_commit("manual", receipt)

    (judge_job,) = runner.finalize(generate, generation)
    assert judge_job.job_type == REMOVAL_JUDGE_JOB
    assert judge_job.resource == RESOURCE_QWEN
    assert dict(judge_job.dependency_digests) == {"candidate": generate.job_id()}

    reviewed = runner.run(judge_job, handles)

    assert reviewed.outcome == OUTCOME_TERMINAL_REJECT
    assert reviewed.payload[PAYLOAD_ATTEMPT]["status"] == "rejected"
    assert reviewed.payload[PAYLOAD_CANDIDATE_INDEX] == 0


def test_second_candidate_job_type_is_the_conditional_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    from r2v_data_v2.v3.remove import prepare_removal_attempt_context

    context = prepare_removal_attempt_context(
        config, storage, "clip-1", _state(storage)
    )

    second = build_removal_generate_job(
        config,
        context,
        canonical_shard="shard-00000",
        clip_uid="clip-1",
        candidate_index=1,
    )

    assert second.job_type == REMOVAL_CANDIDATE2_JOB
    assert second.attempt_index == 1
    assert second.seed == 17
    assert runner is not None and ledger is not None


def test_resume_replays_the_finalizer_without_a_second_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    backend = _Backend()
    judge = _Judge([_accept()])

    runner = _runner(config, storage, ledger)
    handles = RemovalEpochHandles(removal_backend=backend, judge=judge)
    original_finalize = runner.finalize
    crash = {"pending": True}

    def crashing_finalize(job, result):
        if crash["pending"] and job.job_type == REMOVAL_GENERATE_JOB:
            crash["pending"] = False
            raise RuntimeError("simulated crash after the receipt")
        return original_finalize(job, result)

    runner.finalize = crashing_finalize  # type: ignore[method-assign]
    first = _scheduler(ledger, runner, handles).run(runner.seed_jobs())
    assert first["completed"] is False
    assert backend.seeds == [0]
    assert judge.calls == 0

    resumed = _runner(config, storage, ledger)
    second = _scheduler(ledger, resumed, handles).run(resumed.seed_jobs())

    assert second["completed"] is True
    assert backend.seeds == [0], "a committed generation must never be re-requested"
    assert judge.calls == 1
    assert _state(storage).status == "ready_removed"
    assert second["diagnostics"]["resume"]["finalizers_replayed"] == 1


def test_finalizer_after_publication_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    storage = _pending_storage(config)
    ledger = GroupLedger(tmp_path / "group")
    runner = _runner(config, storage, ledger)
    handles = RemovalEpochHandles(removal_backend=_Backend(), judge=_Judge([_accept()]))

    _scheduler(ledger, runner, handles).run(runner.seed_jobs())
    assert _state(storage).status == "ready_removed"

    # A replayed finalizer on an already-published clip unlocks nothing and
    # does not raise, which is what keeps a crashed publish from stranding the
    # group forever.
    from r2v_data_v2.v3.post_mask_epoch_jobs import JobResult

    judge_job = _judge_job_for_first_candidate(config, storage, runner)
    replayed = runner.finalize(
        judge_job,
        JobResult(
            outcome=OUTCOME_COMPLETED,
            payload={
                PAYLOAD_ATTEMPT: _state(storage).removal_attempts[0].model_dump(
                    mode="json"
                )
            },
        ),
    )
    assert tuple(replayed) == ()


# ---------------------------------------------------------------------------
# Small helpers used by the resume/idempotency tests
# ---------------------------------------------------------------------------


def _judge_job_for_first_candidate(config, storage, runner):
    from r2v_data_v2.v3.remove import prepare_removal_attempt_context

    context = prepare_removal_attempt_context(
        config, storage, "clip-1", _state(storage)
    )
    from r2v_data_v2.v3.post_mask_epoch_removal import build_removal_judge_job

    class _Ref:
        def job_id(self) -> str:
            return "generate-job-id"

    return build_removal_judge_job(
        config,
        context,
        canonical_shard="shard-00000",
        clip_uid="clip-1",
        candidate_index=0,
        generate_job=_Ref(),  # type: ignore[arg-type]
    )
