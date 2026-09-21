"""Subject Attributes resource epoch: durable owner foundation and the 8b chain.

Every fixture here compares the epoch against the legacy authority
``subject_attributes.process_subject_attribute_clip`` on the same real clip: the
legacy owner artifact bytes, the legacy clip counts, the legacy SAM probe order
and the legacy rank-0 routes. 8a covers discovery and the durable owner
foundation; 8b covers the SAM probe chain, the duplicate-mask CPU pass and the
one owner-batched rank-0 raw review. Completion, candidate 2, bbox and the final
human owner artifact stay out of scope.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.test_v3_reference_integrity as legacy_integrity
from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN, RESOURCE_SAM
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    JobExecution,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
    SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
    SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
    SUBJECT_ATTRIBUTE_SAM_PROBE_JOB,
    SubjectAttributeDurableError,
    SubjectAttributeEpochError,
    SubjectAttributeEpochRunner,
    _image_png_sha256,
    _ReplaySegmentationBackend,
)
from r2v_data_v2.v3.subject_attributes import (
    DiscoveredSubjectAttribute,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    SubjectAttributeDiscovery,
    SubjectAttributeReview,
    SubjectAttributeReviewBatch,
    _owner_artifact_path,
    process_subject_attribute_clip,
)

SHARD = "shard-000000000-000000000"
CLIP_UID = "clip-1"
OWNER = "e1"
OUTPUT_DIRNAME = "subject_attributes"


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


def _config_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str) -> Any:
    config = legacy_integrity._config(tmp_path, monkeypatch)
    config = replace(
        config,
        run_root=config.run_root.parent / run_name,
        reference_edit=replace(
            config.reference_edit,
            enabled=False,
            # The fixture clip is small; keep the legacy candidate pool usable.
            min_source_content_area_pixels=1,
            min_source_content_long_side_pixels=1,
        ),
        reference_integrity=replace(config.reference_integrity, enabled=False),
    )
    config.validate()
    return config


def _storage_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """The legacy ready-pair clip on its own run root."""
    config = _config_for(tmp_path, monkeypatch, run_name)
    monkeypatch.setattr(legacy_integrity, "_config", lambda *args, **kw: config)
    storage = legacy_integrity._storage_with_ready_pair(tmp_path, monkeypatch, **kwargs)
    return config, storage


def _runner(
    config: Any, storage: Any, tmp_path: Path, *, ledger_name: str = "ledger"
) -> Any:
    """One runner over its own ledger directory.

    Several storage variants inside one test must NOT share a ledger: the frozen
    plan and outcome paths are keyed by shard/clip/owner only, so two variants
    would collide and appear as durable drift.
    """
    return SubjectAttributeEpochRunner(
        config,
        {SHARD: storage},
        GroupLedger(tmp_path / ledger_name),
        eligible_clip_uids_by_shard={SHARD: [CLIP_UID]},
    )


def _output_root(storage: Any) -> Path:
    return Path(storage.root) / OUTPUT_DIRNAME


def _owner_artifact_file(storage: Any, owner: str = OWNER) -> Path:
    return _owner_artifact_path(
        _output_root(storage), sample_id=CLIP_UID, owner_entity_id=owner
    )


class _DiscoveryClient:
    """The legacy ``SubjectAttributeDiscoveryClient`` protocol, scripted."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def discover(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.requests.append(kwargs)
        result = self.results[self.calls - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _raw_review(attribute_id: str, **overrides: Any) -> Any:
    payload = {
        "attribute_id": attribute_id,
        "matches_attribute": True,
        "owner_binding_correct": True,
        "recognizable": True,
        "characteristic_appearance_visible": True,
        "usable_as_attribute_condition": True,
        "sufficient_source_evidence": True,
        "structure_complete": True,
        "completion_recommended": False,
        "reason": "raw crop is usable",
        **overrides,
    }
    return SubjectAttributeReview.model_validate(payload)


class _SamBackend:
    """The SAM epoch resource handle: one scripted ``segment_frame`` per probe.

    Two script modes. ``*results`` is consumed one entry per call, which is what
    a conditional probe chain needs. ``by_prompt`` keys the returned masks by the
    grounding prompt, so several attributes of one owner can be driven
    independently of the order the scheduler happens to run their probes in.
    """

    def __init__(
        self,
        *results: Any,
        timeline: list[str] | None = None,
        by_prompt: Mapping[str, Any] | None = None,
    ) -> None:
        self.results = list(results)
        self.by_prompt = dict(by_prompt or {})
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.timeline = timeline

    def segment_frame(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.requests.append(kwargs)
        if self.timeline is not None:
            self.timeline.append(f"sam:{kwargs['frame_slot']}")
        if self.by_prompt:
            result = self.by_prompt.get(str(kwargs["grounding_prompt"]), [])
        else:
            result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return list(result)


class _QwenClient:
    """One handle serving both discovery and the owner-batched raw review."""

    def __init__(
        self,
        *,
        discoveries: Any = (),
        reviews: Any = (),
        timeline: list[str] | None = None,
    ) -> None:
        self.discoveries = list(discoveries)
        self.reviews = list(reviews)
        self.discovery_calls = 0
        self.review_calls = 0
        self.review_requests: list[list[str]] = []
        self.timeline = timeline

    def discover(self, **kwargs: Any) -> Any:
        self.discovery_calls += 1
        if self.timeline is not None:
            self.timeline.append("discovery")
        result = self.discoveries[
            min(self.discovery_calls, len(self.discoveries)) - 1
        ]
        if isinstance(result, Exception):
            raise result
        return result

    def review(self, *, owner: Any, candidates: Any) -> Any:
        del owner
        self.review_calls += 1
        self.review_requests.append(
            [candidate.attribute_id for candidate in candidates]
        )
        if self.timeline is not None:
            self.timeline.append(
                f"review:{','.join(candidate.attribute_id for candidate in candidates)}"
            )
        result = self.reviews[min(self.review_calls, len(self.reviews)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def _attribute_mask() -> Any:
    """A plausible attribute mask inside the fixture's owner mask."""
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:14, 10:20] = True
    return mask


class _ForbiddenClient:
    """Any call proves 8a leaked into SAM / review / completion."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        def _call(**kwargs: Any) -> Any:
            del kwargs
            self.calls += 1
            raise AssertionError(f"{self.label} must never be called in 8a")

        return _call


class _SerialQwenExecutor:
    """Serial executor for the real scheduler; one resource, no real model."""

    def __init__(self, runner: Any, handle: Any) -> None:
        self.runner = runner
        self.handle = handle
        self.batches = 0
        self.executed: list[Any] = []

    def execute_batch(self, jobs: Any) -> dict[str, Any]:
        self.batches += 1
        outcomes: dict[str, Any] = {}
        for job in sorted(jobs, key=lambda item: item.job_id()):
            self.executed.append(job)
            try:
                outcomes[job.job_id()] = JobExecution(
                    job, self.runner.run(job, self.handle), None
                )
            except Exception as exc:  # noqa: BLE001 - isolate per job
                outcomes[job.job_id()] = JobExecution(job, None, exc)
        return outcomes


def _scheduler(
    runner: Any, executor: Any, finalize: Any = None, sam: Any = None
) -> Any:
    executors: dict[str, Any] = {RESOURCE_QWEN: executor}
    if sam is not None:
        executors[RESOURCE_SAM] = sam
    return ResourceEpochScheduler(
        ledger=runner.ledger,
        finalize=finalize or runner.finalize,
        executors=executors,
    )


def _drain(
    runner: Any,
    client: Any,
    *,
    finalize: Any = None,
    sam: Any = None,
) -> tuple[Any, list[Any]]:
    executor = _SerialQwenExecutor(runner, client)
    sam_executor = None if sam is None else _SerialQwenExecutor(runner, sam)
    jobs = runner.seed_jobs()
    if jobs:
        _scheduler(runner, executor, finalize, sam_executor).run(jobs)
    return executor, jobs


def _legacy_run(
    config: Any,
    storage: Any,
    client: Any,
    *,
    review: Any = None,
    segmentation: Any = None,
) -> tuple[Any, Any, Any]:
    """The legacy authority on the same clip.

    By default SAM, review and completion surfaces are forbidden, which is what
    8a's discovery-only tests prove. 8b's tests hand in the same scripted models
    the epoch was given, so the two implementations can be compared byte for
    byte on the same model behaviour.
    """
    review = review if review is not None else _ForbiddenClient("review")
    segmentation = (
        segmentation if segmentation is not None else _ForbiddenClient("segmentation")
    )
    result = process_subject_attribute_clip(
        config,
        storage=storage,
        output_root=_output_root(storage),
        clip=storage.read_clip(CLIP_UID),
        discovery_client=client,
        review_client=review,
        segmentation_backend=segmentation,
    )
    return result, review, segmentation


#: One ``accessory`` is safe in this tiny fixture; ``headwear`` carries a
#: legacy 128px long-side rule that no 32x24 clip can satisfy.
DEFAULT_HUMAN_ATTRIBUTES = (
    ("headwear", "a red cap", "red cap on the head"),
)


def _human_discovery(
    owner: str = OWNER,
    attributes: tuple[tuple[str, str, str], ...] | None = None,
) -> SubjectAttributeDiscovery:
    specs = attributes if attributes is not None else DEFAULT_HUMAN_ATTRIBUTES
    return SubjectAttributeDiscovery(
        owner_entity_id=owner,
        owner_is_human=True,
        attributes=[
            DiscoveredSubjectAttribute(
                attribute_type=kind,
                phrase=phrase,
                grounding_prompt=prompt,
            )
            for kind, phrase, prompt in specs
        ],
    )


def _nonhuman_discovery(owner: str = OWNER) -> SubjectAttributeDiscovery:
    return SubjectAttributeDiscovery(
        owner_entity_id=owner, owner_is_human=False, attributes=[]
    )


def _read_artifact(storage: Any, owner: str = OWNER) -> Any:
    return OwnerEnrichmentArtifact.model_validate_json(
        _owner_artifact_file(storage, owner).read_text(encoding="utf-8")
    )


def _add_owner_frame(storage: Any, slot: int = 1) -> None:
    """Make one more owner candidate frame present, changing nothing else.

    Legacy probes the owner candidates in ``prefer_attribute_candidate_frames``
    order; a second present frame is what makes the conditional SAM probe chain
    real (probe 1 can be unusable while probe 2 is usable).
    """
    from r2v_data_v2.reconciliation import write_json_atomic

    path = Path(storage.root) / "clips" / CLIP_UID / "masks.rle.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    track = payload["entities"][OWNER]
    source = next(item for item in track["frames"] if item["present"])
    for frame in track["frames"]:
        if frame["slot"] != slot:
            continue
        frame.update(
            {
                "present": True,
                "track_valid": True,
                "confidence": 0.9,
                "backend_confidences": [0.9],
                "backend_object_ids": list(source["backend_object_ids"]),
                "area_pixels": source["area_pixels"],
                "area_ratio": source["area_ratio"],
                "bbox_xyxy": list(source["bbox_xyxy"]),
                "rle": source["rle"],
            }
        )
    write_json_atomic(path, payload)


def _owner_mask(storage: Any, slot: int = 0) -> Any:
    from r2v_data_v2.v3.subject_attributes import _decode_owner_mask

    decoded = _decode_owner_mask(
        storage.read_masks(CLIP_UID), entity_id=OWNER, slot=slot
    )
    assert decoded is not None, f"owner has no mask on slot {slot}"
    return np.asarray(decoded).astype(bool)


def _attribute_mask(storage: Any, *, slot: int = 0, band: int = 0) -> Any:
    """A small block strictly inside the owner mask: a *usable* SAM mask.

    Legacy rejects an attribute mask that covers its owner and, for clothing,
    rejects strip-like or owner-like shapes, so the block is a compact interior
    rectangle rather than the owner mask itself.
    """
    owner = _owner_mask(storage, slot)
    rows, columns = np.nonzero(owner)
    top = int(rows.min())
    left = int(columns.min())
    height = int(rows.max()) + 1 - top
    width = int(columns.max()) + 1 - left
    block = np.zeros_like(owner)
    offset = 2 if band == 0 else 9
    block[
        top + offset : top + offset + 4,
        left + 5 : left + 13,
    ] = True
    usable = np.logical_and(block, owner)
    assert int(usable.sum()) >= 16, "fixture attribute mask is too small"
    assert max(height, width) > 0
    return usable


def _completion_config(config: Any) -> Any:
    """Completion on, with the crop-quality thresholds neutralised.

    The fixture clip is a tiny synthetic frame: it has no texture, so the legacy
    ``minimum_sharpness_score`` would reject every crop before the review ever
    runs. These are ordinary config thresholds and the epoch binds them into its
    frozen policy, so configuring them does not weaken the test.
    """
    completion = replace(
        config.subject_attributes.completion,
        enabled=True,
        minimum_mean_luminance=0.0,
        minimum_sharpness_score=0.0,
    )
    return replace(
        config,
        subject_attributes=replace(
            config.subject_attributes, completion=completion
        ),
    )


def _clip_plan_path(runner: Any) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "plans"
        / SHARD
        / f"{CLIP_UID}.json"
    )


def _owner_plan_path(runner: Any) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "plan.json"
    )


def _owner_outcome_path(runner: Any) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "outcome.json"
    )


def _sample_path(storage: Any) -> Path:
    return _output_root(storage) / "samples" / f"{CLIP_UID}.json"


def _swallow(job_type: str, runner: Any) -> Any:
    """Swallow one job type's finalizer: a receipt-committed crash window."""

    def finalize(job: Any, result: Any) -> Any:
        if job.job_type == job_type:
            return ()
        return runner.finalize(job, result)

    return finalize


def _planned_owner_plan(runner: Any) -> dict[str, Any]:
    return json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "plan.json"
        ).read_text(encoding="utf-8")
    )


def _selection_marker_file(runner: Any, attribute_id: str) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "attributes"
        / attribute_id
        / "selection.json"
    )


def _selection_marker(runner: Any, attribute_id: str) -> dict[str, Any]:
    return json.loads(
        _selection_marker_file(runner, attribute_id).read_text(encoding="utf-8")
    )


def _attribute_plan_file(runner: Any, attribute_id: str) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "attributes"
        / attribute_id
        / "plan.json"
    )


def _raw_state_file(runner: Any) -> Path:
    return (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "raw_state.json"
    )


def _raw_state_marker(runner: Any) -> dict[str, Any]:
    return json.loads(_raw_state_file(runner).read_text(encoding="utf-8"))


def _preexisting_nonhuman_artifact(storage: Any) -> OwnerEnrichmentArtifact:
    clip = storage.read_clip(CLIP_UID)
    owner = next(item for item in clip.annotation.entities if item.entity_id == OWNER)
    artifact = OwnerEnrichmentArtifact(
        sample_id=CLIP_UID,
        owner_entity_id=OWNER,
        owner_is_human=False,
        attribute_id_start=1,
        owner_phrase=owner.phrase,
        owner_grounding_prompt=owner.grounding_prompt,
        records=[],
        metrics=OwnerEnrichmentMetrics(discovery_calls=1),
        gme_screen_mode=None,
        completion_mode=None,
    )
    path = _owner_artifact_file(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return artifact


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_preexisting_owner_artifact_is_reused_without_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid legacy owner artifact is terminal with zero model calls.

    8a also refuses the GME production path outright: GME is not implemented.
    """
    gme_config, gme_storage = _storage_variant(tmp_path, monkeypatch, "run-gme")
    with pytest.raises(SubjectAttributeEpochError, match="GME disabled"):
        SubjectAttributeEpochRunner(
            replace(
                gme_config,
                subject_attribute_gme=replace(
                    gme_config.subject_attribute_gme, enabled=True
                ),
            ),
            {SHARD: gme_storage},
            GroupLedger(tmp_path / "ledger-gme"),
            eligible_clip_uids_by_shard={SHARD: [CLIP_UID]},
        )

    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    _preexisting_nonhuman_artifact(legacy_storage)
    legacy_client = _DiscoveryClient()
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 0, "the cached path pays nothing"
    assert legacy_result.totals.skipped_existing_owners == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    artifact = _preexisting_nonhuman_artifact(storage)
    artifact_bytes = _owner_artifact_file(storage).read_bytes()
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient()
    executor, jobs = _drain(runner, client)

    assert jobs == [], "a valid cache needs no discovery job"
    assert executor.batches == 0
    assert client.calls == 0
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["skipped_existing_owners"] == 1
    assert stats.to_dict()["discovery_calls"] == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1
    assert stats.terminal_clips == 1

    marker = json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "outcome.json"
        ).read_text(encoding="utf-8")
    )
    assert marker["source"] == "preexisting"
    assert marker["discovery_job_id"] is None
    assert marker["record_count"] == 0
    # The reused artifact is never rewritten.
    assert _owner_artifact_file(storage).read_bytes() == artifact_bytes
    assert _read_artifact(storage) == artifact


def test_discovery_nonhuman_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed non-human owner publishes exactly the legacy artifact."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    legacy_client = _DiscoveryClient(_nonhuman_discovery())
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert jobs[0].job_type == SUBJECT_ATTRIBUTE_DISCOVERY_JOB
    assert jobs[0].resource == RESOURCE_QWEN
    assert dict(jobs[0].target) == {"owner_entity_id": OWNER}
    assert jobs[0].model_identity == f"qwen:{config.qwen.candidate_judge.model}"
    assert executor.batches == 1
    assert client.calls == 1, "exactly one discovery call"

    assert _owner_artifact_file(storage).read_bytes() == (
        _owner_artifact_file(legacy_storage).read_bytes()
    ), "the epoch reuses the legacy OwnerEnrichmentArtifact schema and bytes"
    epoch_artifact = _read_artifact(storage)
    assert epoch_artifact.owner_is_human is False
    assert epoch_artifact.failure_reason is None
    assert epoch_artifact.metrics.discovery_calls == 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["discovery_calls"] == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1
    assert stats.to_dict()["failures"] == 0
    assert not (_output_root(storage) / "samples" / f"{CLIP_UID}.json").exists()


def test_discovery_failure_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A discovery exception is the legacy terminal failure, not a retry."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    legacy_config, legacy_storage = _storage_variant(tmp_path, monkeypatch, "run-legacy")
    legacy_client = _DiscoveryClient(ValueError("discovery exploded"))
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, legacy_client
    )
    assert legacy_client.calls == 1

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(ValueError("discovery exploded"))
    _executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert client.calls == 1
    assert _owner_artifact_file(storage).read_bytes() == (
        _owner_artifact_file(legacy_storage).read_bytes()
    )
    artifact = _read_artifact(storage)
    assert artifact.owner_is_human is None
    assert artifact.records == []
    assert artifact.failure_reason == "discovery_failed:ValueError:discovery exploded"
    assert artifact.metrics.failures == 1
    assert artifact.metrics.discovery_calls == 1

    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["failures"] == 1
    assert stats.to_dict()["eligible_human_owners"] == 0


def test_discovery_receipt_before_finalizer_restart_pays_no_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt committed before the CPU finalizer replays without new calls."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    _executor, jobs = _drain(runner, client, finalize=lambda job, result: ())

    assert len(jobs) == 1
    assert client.calls == 1
    assert not _owner_artifact_file(storage).exists()
    assert not (_output_root(storage) / "samples").exists()

    fresh = _runner(config, storage, tmp_path)
    fresh_executor, fresh_jobs = _drain(fresh, client)

    assert len(fresh_jobs) == 1, "the same frozen job is re-seeded"
    assert fresh_jobs[0].job_id() == jobs[0].job_id()
    assert fresh_executor.batches == 0, "the committed receipt is reused"
    assert client.calls == 1, "restart must not pay another discovery call"
    assert _owner_artifact_file(storage).is_file()
    stats = fresh.reconcile_stats(SHARD)
    assert stats.terminal_clips == 1
    assert stats.to_dict()["screened_nonhuman_subjects"] == 1

    # The marker is never its own authority: a tampered outcome must fail closed
    # against the committed receipt and the artifact it claims.
    outcome_path = (
        Path(fresh.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "outcome.json"
    )
    marker = json.loads(outcome_path.read_text(encoding="utf-8"))
    marker["record_count"] = 7
    outcome_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(SubjectAttributeDurableError, match="owner outcome"):
        _runner(config, storage, tmp_path).reconcile_stats(SHARD)


def test_human_discovery_terminal_and_attribute_owner_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two human discovery shapes: a complete owner, and the 8b boundary."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    zero = SubjectAttributeDiscovery(
        owner_entity_id=OWNER, owner_is_human=True, attributes=[]
    )
    legacy_config, legacy_storage = _storage_variant(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_result, _review, _segmentation = _legacy_run(
        legacy_config, legacy_storage, _DiscoveryClient(zero)
    )

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path, ledger_name="ledger-zero")
    client = _DiscoveryClient(zero)
    executor, jobs = _drain(runner, client)

    assert len(jobs) == 1
    assert client.calls == 1
    assert executor.batches == 1
    # human + zero attributes is a complete legacy terminal owner.
    artifact = _read_artifact(storage)
    assert artifact.owner_is_human is True
    assert artifact.records == []
    assert artifact.model_dump(mode="json") == _read_artifact(
        legacy_storage
    ).model_dump(mode="json")
    stats = runner.reconcile_stats(SHARD)
    assert stats.to_dict() == legacy_result.to_counts()
    assert stats.to_dict()["eligible_human_owners"] == 1
    sample = _output_root(storage) / "samples" / f"{CLIP_UID}.json"
    assert sample.is_file(), "legacy publishes a sample for any human owner"

    # human + attributes keeps only its receipt and continues into 8b.
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-attributes")
    runner = _runner(config, storage, tmp_path, ledger_name="ledger-attrs")
    qwen = _QwenClient(
        discoveries=[_human_discovery()],
        reviews=[SubjectAttributeReviewBatch(owner_entity_id=OWNER, reviews=[
            _raw_review("a1")
        ])],
    )
    sam = _SamBackend(ValueError("sam exploded"))
    executor, jobs = _drain(runner, qwen, sam=sam)

    assert len(jobs) == 1
    assert [job.job_type for job in jobs] == ["subject_attribute_discovery"]
    assert qwen.discovery_calls == 1
    assert sam.calls == 1, "the first SAM probe is the next job"
    assert qwen.review_calls == 0, "a rejected selection pays no raw review"
    # 8b never publishes the human owner artifact: 8c owns that.
    assert not _owner_artifact_file(storage).exists()
    outcome_path = (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "outcome.json"
    )
    assert not outcome_path.exists()
    marker = _raw_state_marker(runner)
    assert [item["route_after_rank0"] for item in marker["attributes"]] == [
        "selection_rejected"
    ]
    assert marker["rank0_review_job_id"] is None
    assert marker["attributes"][0]["selection_status"] == "rejected"
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        runner.reconcile_stats(SHARD)

    fresh = _runner(config, storage, tmp_path, ledger_name="ledger-attrs")
    fresh_qwen = _QwenClient()
    fresh_executor, fresh_jobs = _drain(
        fresh, fresh_qwen, sam=_SamBackend()
    )
    assert fresh_jobs == [], "6b boundary: nothing left to seed"
    assert fresh_executor.batches == 0
    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0
    assert not _owner_artifact_file(storage).exists()
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        fresh.reconcile_stats(SHARD)


def test_frozen_discovery_input_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered frozen candidate source image is durable corruption."""
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    jobs = runner.seed_jobs()

    assert len(jobs) == 1
    assert client.calls == 0
    owner_plan = json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "plan.json"
        ).read_text(encoding="utf-8")
    )
    # The replay backend is the last line of defence: a probe whose input no
    # longer matches the frozen candidate can never reach the legacy CPU policy.
    reference = runner._owner_reference(storage, CLIP_UID, OWNER)
    ordered = runner._ordered_owner_candidates(
        SHARD, storage, CLIP_UID, owner_plan, reference
    )
    replay = _ReplaySegmentationBackend(
        epoch=runner,
        ordered=ordered,
        receipts={},
        storage=storage,
        grounding_prompt="the frozen grounding prompt",
    )
    with pytest.raises(SubjectAttributeDurableError, match="input drifted"):
        replay.segment_frame(
            frame_path=Path("/tmp/not-the-frozen-frame.png"),
            frame_slot=int(ordered[0].frame_slot),
            grounding_prompt="the frozen grounding prompt",
        )
    with pytest.raises(SubjectAttributeDurableError, match="not a frozen owner"):
        replay.segment_frame(
            frame_path=Path(storage.root) / ordered[0].image_path,
            frame_slot=9999,
            grounding_prompt="the frozen grounding prompt",
        )

    candidate = owner_plan["candidates"][0]
    source = Path(storage.root) / candidate["image_path"]
    assert source.is_file()
    original = source.read_bytes()
    # Trailing bytes keep the JPEG decodable, so every derived metric (and the
    # discovery context the model sees) is unchanged: only the frozen source
    # image digest can catch this.
    source.write_bytes(original + b"trailing-tamper")

    fresh = _runner(config, storage, tmp_path)
    with pytest.raises(
        SubjectAttributeDurableError, match="clip plan"
    ):
        fresh.seed_jobs()

    assert client.calls == 0, "no model call may be paid for corruption"
    assert not _owner_artifact_file(storage).exists(), (
        "corruption must never be written as a legacy owner failure"
    )
    assert not (_output_root(storage) / "samples").exists()
    failures_path = Path(storage.root) / "failures.jsonl"
    assert not failures_path.is_file() or not failures_path.read_text(
        encoding="utf-8"
    ).strip()

# ---------------------------------------------------------------------------
# 8b: strict durable validation, SAM probes and the rank-0 raw review
# ---------------------------------------------------------------------------


ACCESSORY = ("accessory", "a leather belt", "belt around the waist")
SCARF = ("upper_clothing", "a green scarf", "scarf around the neck")


def _usable_sam(storage: Any, *, slot: int) -> list[Any]:
    return [_attribute_mask(storage, slot=slot, band=0)]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("clip_plan_extra_field", "clip plan drifted"),
        ("owner_plan_extra_field", "owner plan keyset drifted"),
        ("owner_plan_bad_cache_digest", "artifact cache digest is invalid"),
        ("owner_plan_cache_digest_for_miss", "must not carry a digest"),
        ("owner_plan_missing_cache_key", "artifact cache is malformed"),
    ],
)
def test_durable_plans_reject_any_shape_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, expected: str
) -> None:
    """An extra, missing or reshaped plan field must fail closed with no model."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-strict")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_nonhuman_discovery())
    _drain(runner, client)
    assert client.calls == 1

    if mutation == "clip_plan_extra_field":
        path = _clip_plan_path(runner)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["unexpected"] = 1
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = _owner_plan_path(runner)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "owner_plan_extra_field":
            payload["unexpected"] = 1
        elif mutation == "owner_plan_bad_cache_digest":
            payload["preexisting_owner_artifact"] = {
                "valid": True,
                "sha256": None,
            }
        elif mutation == "owner_plan_cache_digest_for_miss":
            payload["preexisting_owner_artifact"] = {
                "valid": False,
                "sha256": "0" * 64,
            }
        else:
            # A missing key is shape drift, not a KeyError at read time.
            payload["preexisting_owner_artifact"] = {"valid": False}
        path.write_text(json.dumps(payload), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    fresh_client = _DiscoveryClient(_nonhuman_discovery())
    with pytest.raises(SubjectAttributeDurableError, match=expected):
        fresh.seed_jobs()
    assert fresh_client.calls == 0


def test_terminal_clip_rejects_a_tampered_enriched_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal clip's sample artifact is authority, not a rebuildable cache."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-sample")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(_human_discovery(attributes=()))
    _drain(runner, client)
    assert client.calls == 1
    sample = _sample_path(storage)
    assert sample.is_file(), "a human owner publishes a sample"
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    sample.write_text("{\"tampered\": true}", encoding="utf-8")
    with pytest.raises(SubjectAttributeDurableError, match="enriched sample"):
        _runner(config, storage, tmp_path).reconcile_stats(SHARD)


def test_sam_probes_follow_legacy_frame_order_and_are_conditional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe 1 without evidence must be followed by probe 2, exactly like legacy."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)

    legacy_config, legacy_storage = _storage_variant(
        tmp_path, monkeypatch, "run-legacy"
    )
    _add_owner_frame(legacy_storage, slot=1)
    legacy_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    legacy_sam = _SamBackend(
        [],
        _usable_sam(legacy_storage, slot=0),
    )
    _legacy_run(
        legacy_config,
        legacy_storage,
        legacy_qwen,
        review=legacy_qwen,
        segmentation=legacy_sam,
    )

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    _add_owner_frame(storage, slot=1)
    timeline: list[str] = []
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
        timeline=timeline,
    )
    sam = _SamBackend([], _usable_sam(storage, slot=0), timeline=timeline)
    runner = _runner(config, storage, tmp_path)
    _drain(runner, qwen, sam=sam)

    legacy_slots = [int(call["frame_slot"]) for call in legacy_sam.requests]
    epoch_slots = [int(call["frame_slot"]) for call in sam.requests]
    assert legacy_slots == [1, 0], "legacy probes the preferred frame first"
    assert epoch_slots == legacy_slots
    # Probe 2 is only reachable after probe 1 committed and was finalized; the
    # review is only reachable after the selection is terminal.
    assert timeline == ["discovery", "sam:1", "sam:0", "review:a1"]

    plan = runner._clip_plan(SHARD, CLIP_UID)
    owner_plan = runner._owner_plan_for(SHARD, plan, OWNER)
    reference = runner._owner_reference(storage, CLIP_UID, OWNER)
    ordered = runner._ordered_owner_candidates(
        SHARD, storage, CLIP_UID, owner_plan, reference
    )
    discovery_job = runner._expected_discovery_job(SHARD, CLIP_UID, owner_plan)
    attribute_plan = runner._attribute_plan(
        SHARD,
        storage,
        CLIP_UID,
        owner_plan,
        reference,
        _human_discovery(attributes=(ACCESSORY,)),
        0,
        "a1",
        discovery_job.job_id(),
    )
    chain = runner._sam_probe_chain(
        SHARD, CLIP_UID, owner_plan, attribute_plan, ordered
    )
    assert len(chain) == 2
    assert chain[0].dependency_digests == (
        (SUBJECT_ATTRIBUTE_DISCOVERY_JOB, discovery_job.job_id()),
    )
    assert chain[1].dependency_digests == (("a1", chain[0].job_id()),)
    assert [job.attempt_index for job in chain] == [0, 1]

    marker = _selection_marker(runner, "a1")
    assert marker["status"] == "candidates"
    assert len(marker["options"]) == 1, "probe 1 produced no evidence"
    option = marker["options"][0]
    assert option["owner_candidate_id"] == "candidate_2"
    assert option["sam_job_id"] == chain[1].job_id()
    assert option["sam_mask_index"] == 0
    # crop_sha256 is the real RGBA crop rebuilt from the SAM receipt pixels.
    from r2v_data_v2.v3.pair import build_reference_crop
    from r2v_data_v2.v3.subject_attributes import _source_image

    candidate = next(c for c in ordered if c.candidate_id == "candidate_2")
    probe_payload = runner._committed_payload_or_none(chain[1])
    masks = runner._load_sam_masks(probe_payload)
    assert option["sam_mask_sha256"] == probe_payload["masks"][0]["sha256"]
    source = _source_image(storage, candidate)
    crop, _ = build_reference_crop(
        source, masks[0], crop_padding_ratio=float(config.pair.crop_padding_ratio)
    )
    assert option["crop_sha256"] == _image_png_sha256(crop.convert("RGBA"))
    assert option["source_frame_slot"] == 0
    assert qwen.review_calls == 1


def test_sam_receipt_before_finalizer_restart_pays_no_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed SAM receipt with no finalizer replays on CPU, with no SAM call."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-sam-restart")
    runner = _runner(config, storage, tmp_path)
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    sam = _SamBackend(_usable_sam(storage, slot=0))
    sam_executor = _SerialQwenExecutor(runner, sam)

    # Crash window 1: the discovery receipt is committed, its finalizer is lost.
    _scheduler(
        runner,
        _SerialQwenExecutor(runner, qwen),
        _swallow(SUBJECT_ATTRIBUTE_DISCOVERY_JOB, runner),
        sam_executor,
    ).run(runner.seed_jobs())
    assert qwen.discovery_calls == 1
    assert sam.calls == 0
    # A fresh seed continues the committed human discovery into the SAM chain.
    assert [
        job.job_type
        for job in _runner(config, storage, tmp_path).seed_jobs()
    ] == [SUBJECT_ATTRIBUTE_SAM_PROBE_JOB]

    # Crash window 2: the SAM receipt is committed, its finalizer is lost.
    _scheduler(
        runner,
        _SerialQwenExecutor(runner, qwen),
        _swallow(SUBJECT_ATTRIBUTE_SAM_PROBE_JOB, runner),
        sam_executor,
    ).run(runner.seed_jobs())

    assert sam.calls == 1
    assert qwen.discovery_calls == 1, "the discovery receipt is reused too"
    probe_job = next(
        job
        for job in sam_executor.executed
        if job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB
    )
    assert not _raw_state_file(runner).exists()
    assert not _selection_marker_file(runner, "a1").exists()

    fresh = _runner(config, storage, tmp_path)
    fresh_qwen = _QwenClient()
    fresh_sam = _SamBackend()
    fresh_executor, jobs = _drain(fresh, fresh_qwen, sam=fresh_sam)

    assert fresh_sam.calls == 0, "the SAM receipt is durable authority"
    assert fresh_qwen.discovery_calls == 0, "discovery is never re-paid"
    # Only the brand-new rank-0 review is executed; probe 1 is durable.
    assert [job.job_type for job in fresh_executor.executed] == [
        SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB
    ]
    assert [job.job_type for job in jobs] == [SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB]
    # The same probe identity is what makes the committed receipt reusable.
    marker = _selection_marker(fresh, "a1")
    assert marker["status"] == "candidates"
    assert marker["options"][0]["sam_job_id"] == probe_job.job_id()


def test_owner_batched_rank0_review_matches_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rank-0 candidate of one owner travels in a single legacy-ordered call."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    attributes = (ACCESSORY, SCARF)

    def script(storage: Any) -> dict[str, Any]:
        return {
            ACCESSORY[2]: _usable_sam(storage, slot=0),
            SCARF[2]: [_attribute_mask(storage, slot=0, band=1)],
        }

    legacy_config, legacy_storage = _storage_variant(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    legacy_sam = _SamBackend(by_prompt=script(legacy_storage))
    _, _, _ = _legacy_run(
        legacy_config,
        legacy_storage,
        legacy_qwen,
        review=legacy_qwen,
        segmentation=legacy_sam,
    )

    config, storage = _storage_variant(tmp_path, monkeypatch, "run-epoch")
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    sam = _SamBackend(by_prompt=script(storage))
    runner = _runner(config, storage, tmp_path)
    executor, jobs = _drain(runner, qwen, sam=sam)

    review_jobs = [
        job for job in executor.executed if job.job_type == SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB
    ]
    assert len(review_jobs) == 1, "one owner-batched review, not one per attribute"
    assert qwen.review_calls == 1
    assert qwen.review_requests == legacy_qwen.review_requests
    assert qwen.review_requests == [["a1", "a2"]]

    marker = _raw_state_marker(runner)
    assert marker["schema"] == "post_mask_epoch_subject_attribute_raw_state/1"
    assert marker["rank0_review_job_id"] == review_jobs[0].job_id()
    assert [item["route_after_rank0"] for item in marker["attributes"]] == [
        "raw_accept",
        "raw_accept",
    ]
    assert all(item["duplicate_conflict"] is False for item in marker["attributes"])
    # Legacy agrees: two accepted records, in the same order.
    legacy_artifact = _read_artifact(legacy_storage)
    assert [record.attribute_id for record in legacy_artifact.records] == [
        "a1",
        "a2",
    ]
    # 8b never publishes the human owner artifact or its clip outcome.
    assert not _owner_artifact_file(storage).exists()
    assert not _owner_outcome_path(runner).exists()
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        runner.reconcile_stats(SHARD)
    assert jobs

    # The attribute plan is create-once and re-derived, exactly like the clip and
    # owner plans: an extra field must fail closed.
    plan_file = _attribute_plan_file(runner, "a1")
    payload = json.loads(plan_file.read_text(encoding="utf-8"))
    payload["unexpected"] = 1
    plan_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SubjectAttributeDurableError, match="attribute plan drifted"):
        _runner(config, storage, tmp_path).seed_jobs()

    # Two different attributes claiming the same pixels are a legacy duplicate
    # conflict: neither is reviewed, and no review job exists at all.
    legacy_duplicate_config, legacy_duplicate_storage = _storage_variant(
        tmp_path, monkeypatch, "run-duplicate-legacy"
    )
    shared = _attribute_mask(legacy_duplicate_storage, slot=0, band=0)
    duplicate_legacy_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    _legacy_run(
        legacy_duplicate_config,
        legacy_duplicate_storage,
        duplicate_legacy_qwen,
        review=duplicate_legacy_qwen,
        segmentation=_SamBackend(
            by_prompt={ACCESSORY[2]: [shared], SCARF[2]: [shared]}
        ),
    )
    assert duplicate_legacy_qwen.review_calls == 0, "legacy skips a conflict"

    # Its own storage root: a legacy run would otherwise leave a valid owner
    # artifact behind and the epoch would legitimately reuse it.
    duplicate_config, duplicate_storage = _storage_variant(
        tmp_path, monkeypatch, "run-duplicate"
    )
    shared = _attribute_mask(duplicate_storage, slot=0, band=0)
    duplicate_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    duplicate_sam = _SamBackend(
        by_prompt={ACCESSORY[2]: [shared], SCARF[2]: [shared]}
    )
    duplicate_runner = _runner(
        duplicate_config, duplicate_storage, tmp_path, ledger_name="ledger-duplicate"
    )
    _, duplicate_jobs = _drain(duplicate_runner, duplicate_qwen, sam=duplicate_sam)

    assert duplicate_qwen.review_calls == 0
    assert SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB not in {
        job.job_type for job in duplicate_jobs
    }
    duplicate_marker = _raw_state_marker(duplicate_runner)
    assert duplicate_marker["rank0_review_job_id"] is None
    assert all(item["duplicate_conflict"] for item in duplicate_marker["attributes"])
    assert [item["route_after_rank0"] for item in duplicate_marker["attributes"]] == [
        "duplicate_conflict",
        "duplicate_conflict",
    ]
    assert duplicate_runner.seed_jobs() == []


def test_review_round_failure_routes_without_seeding_candidate2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rank-0 review round is a legacy route, not a scheduler retry."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-review-failure")
    _add_owner_frame(storage, slot=1)
    runner = _runner(config, storage, tmp_path)
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[ValueError("review exploded")],
    )
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    sam = _SamBackend(by_prompt={ACCESSORY[2]: masks})
    executor, jobs = _drain(runner, qwen, sam=sam)

    assert qwen.review_calls == 1
    assert sam.calls == 2
    marker = _raw_state_marker(runner)
    attribute = marker["attributes"][0]
    assert attribute["rank0_review"] is None
    assert attribute["rank0_review_failure"] == "review_failed:ValueError:review exploded"
    # Two options exist, so legacy defers to candidate 2 -- but 8b never seeds it.
    assert attribute["route_after_rank0"] == "candidate2_ready"
    assert len(_selection_marker(runner, "a1")["options"]) == 2
    assert [job.job_type for job in executor.executed] == [
        SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
        SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
    ]
    assert [job.job_type for job in jobs] == [SUBJECT_ATTRIBUTE_DISCOVERY_JOB]
    assert runner.seed_jobs() == [], "8b must not speculatively seed candidate 2"
    assert not _owner_artifact_file(storage).exists()
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        runner.reconcile_stats(SHARD)


def test_raw_review_receipt_before_finalizer_restart_pays_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw-state boundary is pure CPU replay once the receipt is committed."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-raw-restart")
    runner = _runner(config, storage, tmp_path)
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    sam = _SamBackend(by_prompt={ACCESSORY[2]: _usable_sam(storage, slot=0)})
    _drain(
        runner,
        qwen,
        finalize=_swallow(SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB, runner),
        sam=sam,
    )

    assert qwen.review_calls == 1
    assert sam.calls == 1
    assert not _raw_state_file(runner).exists()
    assert _selection_marker(runner, "a1")["status"] == "candidates"

    fresh = _runner(config, storage, tmp_path)
    fresh_qwen = _QwenClient()
    fresh_sam = _SamBackend()
    executor, jobs = _drain(fresh, fresh_qwen, sam=fresh_sam)

    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0
    assert fresh_sam.calls == 0
    # The dry run re-offers the same rank-0 job; its committed receipt is reused,
    # so no executor ever ran and only the CPU finalizer replayed.
    assert [job.job_type for job in jobs] == [SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB]
    assert executor.executed == []
    marker = _raw_state_marker(fresh)
    assert [item["route_after_rank0"] for item in marker["attributes"]] == [
        "raw_accept"
    ]
    assert marker["attributes"][0]["rank0_review"] is not None


def test_completion_required_blocks_candidate2_seeding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completion-recommended rank-0 accept must stop before any rank-1 review.

    Legacy tries completion first and only falls back to candidate 2 when that
    fails, so 8b may not speculatively seed a second-candidate review. This is
    the architectural boundary between 8b and 8c.
    """
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    completion_attribute = (
        "upper_clothing",
        "a denim jacket",
        "jacket over the torso",
    )
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-completion")
    _add_owner_frame(storage, slot=1)
    config = _completion_config(config)
    runner = _runner(config, storage, tmp_path)
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(completion_attribute,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1",
                        structure_complete=False,
                        completion_recommended=True,
                    )
                ],
            )
        ],
    )
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    sam = _SamBackend(by_prompt={completion_attribute[2]: masks})
    _drain(runner, qwen, sam=sam)

    marker = _raw_state_marker(runner)
    assert len(marker["attributes"]) == 1
    attribute = marker["attributes"][0]
    assert attribute["selection_status"] == "candidates"
    assert attribute["route_after_rank0"] == "completion_required"
    assert attribute["rank0_review"]["completion_recommended"] is True

    plan = runner._clip_plan(SHARD, CLIP_UID)
    owner_plan = runner._owner_plan_for(SHARD, plan, OWNER)
    reference = runner._owner_reference(storage, CLIP_UID, OWNER)
    discovery_job = runner._expected_discovery_job(SHARD, CLIP_UID, owner_plan)
    attribute_plan = runner._attribute_plan(
        SHARD,
        storage,
        CLIP_UID,
        owner_plan,
        reference,
        _human_discovery(attributes=(completion_attribute,)),
        0,
        "a1",
        discovery_job.job_id(),
    )
    ordered = runner._ordered_owner_candidates(
        SHARD, storage, CLIP_UID, owner_plan, reference
    )
    chain = runner._sam_probe_chain(
        SHARD, CLIP_UID, owner_plan, attribute_plan, ordered
    )
    marker_options = _selection_marker(runner, "a1")["options"]
    assert len(marker_options) == 2, "candidate 2 exists but must stay unseeded"
    assert [option["sam_job_id"] for option in marker_options] == [
        chain[0].job_id(),
        chain[1].job_id(),
    ]

    # No rank-1 review job, no completion job, and the owner stays unresolved.
    assert len(chain) == 2
    assert sam.calls == 2
    assert qwen.review_calls == 1
    assert not _owner_artifact_file(storage).exists()
    assert not _owner_outcome_path(runner).exists()
    fresh = _runner(config, storage, tmp_path)
    assert fresh.seed_jobs() == [], "8b has nothing further to unlock"
    with pytest.raises(SubjectAttributeEpochError, match="not terminal"):
        fresh.reconcile_stats(SHARD)
    assert _owner_artifact_file(storage).exists() is False
