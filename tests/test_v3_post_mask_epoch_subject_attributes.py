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

import io
import itertools
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

import tests.test_v3_reference_integrity as legacy_integrity
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
)
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
    _sha256_bytes,
)
from r2v_data_v2.v3.subject_attributes import (
    ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT,
    DiscoveredSubjectAttribute,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    SubjectAttributeBboxReview,
    SubjectAttributeCompletionReview,
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


def _completion_review(verdict: str = "accept", **overrides: Any) -> Any:
    passed = verdict == "accept"
    payload = {
        "same_physical_attribute": True,
        "original_visible_details_preserved": True,
        "no_wrong_new_instance": True,
        "no_duplicate_component": True,
        "no_unrelated_content": True,
        "no_structural_distortion": True,
        "target_clear_and_prominent": True,
        "candidate_better_than_alpha": passed,
        "certain": True,
        "reason": "completion looks like the same attribute",
        "verdict": verdict,
        **overrides,
    }
    return SubjectAttributeCompletionReview.model_validate(payload)


def _bbox_review(verdict: str = "accept", **overrides: Any) -> Any:
    passed = verdict == "accept"
    payload = {
        "correct_attribute": True,
        "owner_binding_correct": True,
        "target_is_dominant_and_identifiable": True,
        "no_large_competing_attribute_or_entity": True,
        "context_is_limited_and_supportive": True,
        "no_strong_owner_pose_or_scene_leakage": True,
        "no_severe_blur_or_artifact": True,
        "usable_as_attribute_condition": passed,
        "certain": True,
        "reason": "bbox crop is usable",
        "verdict": verdict,
        **overrides,
    }
    return SubjectAttributeBboxReview.model_validate(payload)


def _generated_png(*, colour: tuple[int, int, int] = (210, 70, 70)) -> bytes:
    """A valid generated candidate the tiny fixture can postcheck."""
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), colour).save(buffer, format="PNG")
    return buffer.getvalue()


class _NoOutput:
    """Scripted Boogu result: the backend returns but never writes a file."""


NO_OUTPUT = _NoOutput()


class _BooguBackend:
    """The Boogu completion handle: one scripted generation per call.

    A scripted entry is either PNG bytes, ``NO_OUTPUT`` for a backend that
    reports success without producing a file, or an exception to raise.
    """

    def __init__(
        self,
        *results: Any,
        timeline: list[str] | None = None,
        reported: Any = 0.0,
    ) -> None:
        self.results = list(results)
        self.reported = reported
        self.calls = 0
        self.seeds: list[int] = []
        self.requests: list[dict[str, Any]] = []
        self.timeline = timeline

    def attribute_completion(
        self, *, source_path: Path, output_path: Path, instruction: str, seed: int
    ) -> dict[str, Any]:
        self.calls += 1
        self.seeds.append(int(seed))
        self.requests.append(
            {
                "source_path": Path(source_path),
                "output_path": Path(output_path),
                "instruction": instruction,
            }
        )
        if self.timeline is not None:
            self.timeline.append("boogu")
        result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        reported = (
            self.reported
            if not isinstance(self.reported, list)
            else self.reported[min(self.calls, len(self.reported)) - 1]
        )
        if result is not NO_OUTPUT:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(result)
        return {
            "width": 32,
            "height": 24,
            "model_call_time_seconds": reported,
        }


class _SamBackend:
    """The SAM epoch resource handle: one scripted ``segment_frame`` per probe.

    Three script modes. ``*results`` is consumed one entry per call, which is what
    a conditional probe chain needs. ``by_prompt`` keys the returned masks by the
    grounding prompt, so several attributes of one owner can be driven
    independently of the order the scheduler happens to run their probes in.
    ``by_slot`` keys them by the frame slot instead, which is what proves that a
    probe is answered by the receipt of the frame it actually asked for.
    """

    def __init__(
        self,
        *results: Any,
        timeline: list[str] | None = None,
        by_prompt: Mapping[str, Any] | None = None,
        by_slot: Mapping[int, Any] | None = None,
        generated: Any = (),
    ) -> None:
        self.results = list(results)
        self.by_prompt = dict(by_prompt or {})
        self.by_slot = {int(slot): value for slot, value in (by_slot or {}).items()}
        self.generated_script = list(generated)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.generated: list[Any] = list(self.generated_script)
        self.generated_calls = 0
        self.generated_requests: list[dict[str, Any]] = []
        self.timeline = timeline

    def segment_frame(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.requests.append(kwargs)
        if self.timeline is not None:
            self.timeline.append(f"sam:{kwargs['frame_slot']}")
        if self.by_slot:
            result = self.by_slot.get(int(kwargs["frame_slot"]), [])
        elif self.by_prompt:
            result = self.by_prompt.get(str(kwargs["grounding_prompt"]), [])
        else:
            result = self.results[min(self.calls, len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return list(result)

    def segment_generated_frame(self, **kwargs: Any) -> Any:
        """The completion SAM call: one scripted result per generated frame."""
        self.generated_calls += 1
        self.generated_requests.append(kwargs)
        if self.timeline is not None:
            self.timeline.append("sam:generated")
        result = self.generated[
            min(self.generated_calls, len(self.generated)) - 1
        ]
        if isinstance(result, Exception):
            raise result
        return list(result)


class _QwenClient:
    """One handle serving discovery, the raw reviews, the completion
    reviews and the last-resort bbox review."""

    def __init__(
        self,
        *,
        discoveries: Any = (),
        reviews: Any = (),
        completion_reviews: Any = (),
        bbox_reviews: Any = (),
        timeline: list[str] | None = None,
    ) -> None:
        self.discoveries = list(discoveries)
        self.reviews = list(reviews)
        self.completion_review_results = list(completion_reviews)
        self.bbox_review_results = list(bbox_reviews)
        self.discovery_calls = 0
        self.review_calls = 0
        self.completion_review_calls = 0
        self.bbox_calls = 0
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

    def review(self, **kwargs: Any) -> Any:
        if "candidates" in kwargs:
            return self._raw_review(list(kwargs["candidates"]))
        return self._completion_review()

    def _raw_review(self, candidates: list[Any]) -> Any:
        self.review_calls += 1
        ids = [candidate.attribute_id for candidate in candidates]
        self.review_requests.append(ids)
        if self.timeline is not None:
            self.timeline.append(f"review:{','.join(ids)}")
        result = self.reviews[min(self.review_calls, len(self.reviews)) - 1]
        if isinstance(result, Exception):
            raise result
        return result

    def _completion_review(self) -> Any:
        self.completion_review_calls += 1
        if self.timeline is not None:
            self.timeline.append("completion_review")
        result = self.completion_review_results[
            min(self.completion_review_calls, len(self.completion_review_results)) - 1
        ]
        if isinstance(result, Exception):
            raise result
        return result

    def review_attribute_bbox(self, **kwargs: Any) -> Any:
        self.bbox_calls += 1
        if self.timeline is not None:
            self.timeline.append("bbox")
        result = self.bbox_review_results[
            min(self.bbox_calls, len(self.bbox_review_results)) - 1
        ]
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
    runner: Any,
    executor: Any,
    finalize: Any = None,
    sam: Any = None,
    boogu: Any = None,
) -> Any:
    executors: dict[str, Any] = {RESOURCE_QWEN: executor}
    if sam is not None:
        executors[RESOURCE_SAM] = sam
    if boogu is not None:
        executors[RESOURCE_BOOGU] = boogu
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
    boogu: Any = None,
) -> tuple[Any, list[Any]]:
    executor = _SerialQwenExecutor(runner, client)
    sam_executor = None if sam is None else _SerialQwenExecutor(runner, sam)
    boogu_executor = None if boogu is None else _SerialQwenExecutor(runner, boogu)
    jobs = runner.seed_jobs()
    if jobs:
        _scheduler(runner, executor, finalize, sam_executor, boogu_executor).run(jobs)
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

    assert fresh_jobs == [], "a committed terminal discovery re-seeds nothing"
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
    marker = _raw_state_marker(runner)
    assert [item["route_after_rank0"] for item in marker["attributes"]] == [
        "selection_rejected"
    ]
    assert marker["rank0_review_job_id"] is None
    assert marker["attributes"][0]["selection_status"] == "rejected"
    # A rejected selection is a terminal owner record: 8c publishes it.
    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    assert artifact.metrics.discovery_calls == 1
    assert artifact.metrics.sam3_attempts == 1
    assert artifact.metrics.review_calls == 0
    outcome = json.loads(_owner_outcome_path(runner).read_text(encoding="utf-8"))
    assert outcome["source"] == "processed"
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    fresh = _runner(config, storage, tmp_path, ledger_name="ledger-attrs")
    fresh_qwen = _QwenClient()
    fresh_executor, fresh_jobs = _drain(
        fresh, fresh_qwen, sam=_SamBackend()
    )
    assert fresh_jobs == [], "6b boundary: nothing left to seed"
    assert fresh_executor.batches == 0
    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0
    assert _read_artifact(storage).model_dump(mode="json") == artifact.model_dump(
        mode="json"
    )
    assert fresh.reconcile_stats(SHARD).terminal_clips == 1


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


def test_owner_candidates_are_validated_once_per_runner_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-owner-cache")
    runner = _runner(config, storage, tmp_path)
    plan = runner._clip_plan(SHARD, CLIP_UID)
    owner = runner._eligible_owners(plan)[0]
    owner_plan = runner._owner_plan(SHARD, CLIP_UID, owner, 0, 1)

    original = runner._owner_candidates
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_owner_candidates", counted)
    first = runner._owner_candidate_objects(SHARD, CLIP_UID, owner_plan)
    second = runner._owner_candidate_objects(SHARD, CLIP_UID, owner_plan)
    third = runner._owner_candidate_objects(SHARD, CLIP_UID, owner_plan)

    assert calls == 1
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ] == [item.candidate_id for item in third]


def test_clip_plan_is_rederived_once_per_runner_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated owner-chain replay must not rebuild frozen clip evidence."""
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-plan-cache")
    runner = _runner(config, storage, tmp_path)
    original = runner._derive_clip_plan
    calls = 0

    def counted(storage_arg: Any, clip_uid_arg: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(storage_arg, clip_uid_arg)

    monkeypatch.setattr(runner, "_derive_clip_plan", counted)

    first = runner._clip_plan(SHARD, CLIP_UID)
    second = runner._clip_plan(SHARD, CLIP_UID)
    third = runner._clip_plan(SHARD, CLIP_UID)

    assert calls == 1
    assert first is second is third

    # Restart correctness remains unchanged: a fresh runner has no memory cache
    # and must revalidate the durable plan from live upstream state.
    fresh = _runner(config, storage, tmp_path)
    fresh_calls = 0
    fresh_original = fresh._derive_clip_plan

    def fresh_counted(storage_arg: Any, clip_uid_arg: str) -> dict[str, Any]:
        nonlocal fresh_calls
        fresh_calls += 1
        return fresh_original(storage_arg, clip_uid_arg)

    monkeypatch.setattr(fresh, "_derive_clip_plan", fresh_counted)
    fresh._clip_plan(SHARD, CLIP_UID)
    assert fresh_calls == 1



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
    legacy_result, _, _ = _legacy_run(
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
    # Both attributes are accepted raw, so the owner is terminal and the epoch
    # publishes exactly the legacy artifact and the legacy clip counts.
    assert _read_artifact(storage).model_dump(mode="json") == _read_artifact(
        legacy_storage
    ).model_dump(mode="json")
    outcome = json.loads(_owner_outcome_path(runner).read_text(encoding="utf-8"))
    assert outcome["source"] == "processed"
    assert runner.reconcile_stats(SHARD).to_dict() == legacy_result.to_counts()
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


def test_review_round_failure_defers_to_candidate2_then_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed review round is a legacy route, never a scheduler retry.

    Rank 0 fails and defers to candidate 2, rank 1 fails too, and because there
    is no third candidate the attribute ends in the legacy terminal reject.
    """
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

    assert sam.calls == 2
    marker = _raw_state_marker(runner)
    attribute = marker["attributes"][0]
    assert attribute["rank0_review"] is None
    assert attribute["rank0_review_failure"] == "review_failed:ValueError:review exploded"
    # Two options exist, so legacy defers to candidate 2 and rank 1 is the frozen
    # next round; there is no third candidate, so rank 1 ends the attribute.
    assert attribute["route_after_rank0"] == "candidate2_ready"
    assert len(_selection_marker(runner, "a1")["options"]) == 2
    assert qwen.review_calls == 2
    assert [job.job_type for job in executor.executed] == [
        SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
        SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
        SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
    ]
    assert [job.job_type for job in jobs] == [SUBJECT_ATTRIBUTE_DISCOVERY_JOB]
    assert runner.seed_jobs() == []
    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    assert artifact.records[0].reason == "review_failed:ValueError:review exploded"
    assert artifact.metrics.review_calls == 2
    assert artifact.metrics.attribute_second_candidate_attempts == 1
    assert artifact.failure_reason == "review_failed:ValueError:review exploded"
    assert runner.reconcile_stats(SHARD).terminal_clips == 1


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
    assert jobs == [], "the committed receipt is enough to finish on CPU"
    assert executor.executed == []
    marker = _raw_state_marker(fresh)
    assert [item["route_after_rank0"] for item in marker["attributes"]] == [
        "raw_accept"
    ]
    assert marker["attributes"][0]["rank0_review"] is not None
    # The raw accept is terminal, so the owner artifact is backfilled on CPU too.
    assert _read_artifact(storage).records[0].status == "accepted"
    assert json.loads(
        _owner_outcome_path(fresh).read_text(encoding="utf-8")
    )["source"] == "processed"


def test_completion_accepts_before_any_candidate2_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completion-recommended rank-0 accept never reaches candidate 2.

    Legacy tries completion first and only falls back to candidate 2 when the
    completion chain fails, so the rank-1 review batch must stay empty while the
    rank-0 completion succeeds. The whole completion chain is one Boogu call,
    one generated-frame SAM call and one comparative review, all durable.
    """
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    attribute = ("upper_clothing", "a denim jacket", "jacket over the torso")
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-completion")
    _add_owner_frame(storage, slot=1)
    config = _completion_config(config)
    runner = _runner(config, storage, tmp_path)
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(attribute,))],
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
        completion_reviews=[_completion_review("accept")],
    )
    sam = _SamBackend(
        by_prompt={attribute[2]: masks},
        generated=[masks[:1]],
    )
    boogu = _BooguBackend(_generated_png())
    timeline: list[str] = []
    for surface in (qwen, sam, boogu):
        surface.timeline = timeline
    _drain(runner, qwen, sam=sam, boogu=boogu)

    marker = _raw_state_marker(runner)
    attribute_state = marker["attributes"][0]
    assert attribute_state["selection_status"] == "candidates"
    assert attribute_state["route_after_rank0"] == "completion_required"
    assert attribute_state["rank0_review"]["completion_recommended"] is True

    # Exactly one Boogu call, one generated-frame SAM call, one completion
    # review, and no rank-1 raw review at all.
    assert boogu.calls == 1
    assert sam.calls == 2, "the two rank-0 selection probes"
    assert sam.generated_calls == 1
    assert qwen.review_calls == 1, "rank 1 is never reviewed"
    assert qwen.completion_review_calls == 1
    assert qwen.bbox_calls == 0
    assert timeline == [
        "discovery",
        "sam:1",
        "sam:0",
        "review:a1",
        "boogu",
        "sam:generated",
        "completion_review",
    ]
    assert runner.seed_jobs() == []

    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["accepted"]
    record = artifact.records[0]
    assert record.final_selection == "completed"
    assert record.completion_attempted is True
    assert record.completion_outcome == "selected_completed"
    assert record.completion_review.verdict == "accept"
    assert record.completion_seed == boogu.seeds[0]
    assert artifact.metrics.completion_attempts == 1
    assert artifact.metrics.completion_accepted == 1
    assert artifact.metrics.completion_selected_completed == 1
    assert artifact.metrics.attribute_completion_candidate1_accepted == 1
    assert artifact.metrics.attribute_completion_candidate2_accepted == 0
    assert artifact.metrics.attribute_second_candidate_attempts == 0
    assert artifact.metrics.attribute_bbox_fallback_attempts == 0

    # The durable seed and the completion outcome marker exist and agree.
    seed_path = (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "attributes"
        / "a1"
        / "completion"
        / "rank-0-seed.json"
    )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    assert seed["seed"] == boogu.seeds[0]
    assert seed["owner_candidate_id"] == "candidate_1"
    outcome_path = seed_path.with_name("rank-0-outcome.json")
    completion_outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert completion_outcome["status"] == "accepted"
    assert completion_outcome["reason"] == "completion_identity_accepted"
    assert completion_outcome["candidate_rank"] == 0
    assert completion_outcome["completed_crop_sha256"] is not None
    assert completion_outcome["completion_review"]["verdict"] == "accept"
    assert runner.reconcile_stats(SHARD).terminal_clips == 1


# ---------------------------------------------------------------------------
# 8c: the completion chain, the rank-1 fallback, bbox and the terminal owner
# ---------------------------------------------------------------------------

COMPLETION_ATTRIBUTE = ("upper_clothing", "a denim jacket", "jacket over the torso")
FACE_ATTRIBUTE = ("face", "a calm face", "face of the person")


def _completion_config_for(config: Any) -> Any:
    return _completion_config(config)


def _completion_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_name: str,
    attribute: tuple[str, str, str],
    reviews: list[Any],
    completion_reviews: list[Any] = (),
    bbox_reviews: list[Any] = (),
    boogu: Any = (),
    reported: Any = 0.0,
) -> tuple[Any, Any, Any, _QwenClient, _SamBackend, _BooguBackend, Any]:
    """One human attribute with two owner candidate frames and completion on."""
    config, storage = _storage_variant(tmp_path, monkeypatch, run_name)
    _add_owner_frame(storage, slot=1)
    config = _completion_config(config)
    runner = _runner(config, storage, tmp_path, ledger_name=f"ledger-{run_name}")
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(attribute,))],
        reviews=reviews,
        completion_reviews=list(completion_reviews),
        bbox_reviews=list(bbox_reviews),
    )
    sam = _SamBackend(
        by_prompt={attribute[2]: masks},
        # One generated-frame SAM result per completion chain a test may run.
        generated=[masks[:1], masks[1:]],
    )
    backend = _BooguBackend(*boogu, reported=reported)
    timeline: list[str] = []
    for surface in (qwen, sam, backend):
        surface.timeline = timeline
    return config, storage, runner, qwen, sam, backend, timeline


def test_raw_state_tamper_fails_closed_without_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rank-0 boundary marker is derived, never authority."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-raw-tamper")
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
    _drain(runner, qwen, sam=sam)

    path = _raw_state_file(runner)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["attributes"][0]["route_after_rank0"] = "raw_accept_tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    fresh = _runner(config, storage, tmp_path)
    fresh_qwen = _QwenClient()
    with pytest.raises(SubjectAttributeDurableError, match="raw state drifted"):
        _drain(fresh, fresh_qwen, sam=_SamBackend())
    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0


def test_discovery_failure_receipt_restart_backfills_the_legacy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed discovery failure replays on CPU with no Qwen call."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-failure")
    runner = _runner(config, storage, tmp_path)
    client = _DiscoveryClient(ValueError("discovery exploded"))
    _drain(runner, client, finalize=lambda job, result: ())

    assert client.calls == 1
    assert not _owner_artifact_file(storage).exists()

    fresh = _runner(config, storage, tmp_path)
    fresh_client = _DiscoveryClient(ValueError("discovery exploded"))
    fresh_executor, fresh_jobs = _drain(fresh, fresh_client)

    assert fresh_jobs == [], "a committed terminal failure re-seeds nothing"
    assert fresh_executor.batches == 0
    assert fresh_client.calls == 0, "the failure receipt is reused"
    artifact = _read_artifact(storage)
    assert artifact.owner_is_human is None
    assert artifact.failure_reason == (
        "discovery_failed:ValueError:discovery exploded"
    )
    assert artifact.metrics.failures == 1
    assert fresh.reconcile_stats(SHARD).terminal_clips == 1


def test_raw_review_accepts_an_endpoint_string_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real runner hands a base URL, so the epoch must build the client."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-endpoint")
    runner = _runner(config, storage, tmp_path)
    calls: list[tuple[str, list[str]]] = []

    class _EndpointClient:
        def __init__(self, service: Any) -> None:
            self.service = service
            self.closed = False

        def review(self, *, owner: Any, candidates: list[Any]) -> Any:
            del owner
            calls.append(
                (
                    str(getattr(self.service, "base_url", "")),
                    [candidate.attribute_id for candidate in candidates],
                )
            )
            return SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_epoch_subject_attributes."
        "QwenSubjectAttributeClient",
        _EndpointClient,
    )
    # Commit the discovery receipt first, then drive the review surface with a
    # plain base URL the way the production resource handle does.
    _drain(
        runner,
        _DiscoveryClient(_human_discovery(attributes=(ACCESSORY,))),
        finalize=_swallow(SUBJECT_ATTRIBUTE_DISCOVERY_JOB, runner),
    )
    sam = _SamBackend(by_prompt={ACCESSORY[2]: _usable_sam(storage, slot=0)})
    _drain(runner, "http://127.0.0.1:8123/v1", sam=sam)

    assert calls == [("http://127.0.0.1:8123/v1", ["a1"])]
    assert _read_artifact(storage).records[0].status == "accepted"


def test_attribute_ids_are_relative_to_the_owner_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second human owner starts at a3 and a3 must map to its own first attribute."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-offset")
    runner = _runner(config, storage, tmp_path)
    discovery = _human_discovery(attributes=(ACCESSORY, SCARF))
    assert runner._attribute_index(discovery, "a1", 1) == 0
    assert runner._attribute_index(discovery, "a2", 1) == 1
    assert runner._attribute_index(discovery, "a3", 3) == 0
    assert runner._attribute_index(discovery, "a4", 3) == 1
    with pytest.raises(SubjectAttributeDurableError, match="outside the frozen"):
        runner._attribute_index(discovery, "a1", 3)
    with pytest.raises(SubjectAttributeDurableError, match="outside the frozen"):
        runner._attribute_index(discovery, "a3", 1)

    # And the whole owner flow honours a second owner's numbering end to end.
    monkeypatch.setattr(
        SubjectAttributeEpochRunner,
        "_attribute_id_start",
        lambda self, shard, clip_uid, owners, index: 3,
    )
    runner = _runner(config, storage, tmp_path, ledger_name="ledger-offset-run")
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY, SCARF))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a3"), _raw_review("a4")],
            )
        ],
    )
    sam = _SamBackend(
        by_prompt={
            ACCESSORY[2]: _usable_sam(storage, slot=0),
            SCARF[2]: [_attribute_mask(storage, slot=0, band=1)],
        }
    )
    _drain(runner, qwen, sam=sam)

    artifact = _read_artifact(storage)
    assert artifact.attribute_id_start == 3
    assert [record.attribute_id for record in artifact.records] == ["a3", "a4"]
    assert [record.status for record in artifact.records] == ["accepted", "accepted"]
    assert runner.reconcile_stats(SHARD).terminal_clips == 1


def test_rank0_completion_fails_then_rank1_completion_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completion runs before candidate 2, and candidate 2 gets its own chain."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    _config, storage, runner, qwen, sam, boogu, timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-c2-accept",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        completion_reviews=[
            _completion_review("reject"),
            _completion_review("accept"),
        ],
        boogu=[_generated_png(), _generated_png(colour=(40, 90, 200))],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    assert timeline == [
        "discovery",
        "sam:1",
        "sam:0",
        "review:a1",
        "boogu",
        "sam:generated",
        "completion_review",
        "review:a1",
        "boogu",
        "sam:generated",
        "completion_review",
    ]
    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["accepted"]
    record = artifact.records[0]
    assert record.final_selection == "completed"
    assert artifact.metrics.review_calls == 2
    assert artifact.metrics.completion_attempts == 2
    assert artifact.metrics.completion_qwen_review_rejects == 1
    assert artifact.metrics.completion_accepted == 1
    assert artifact.metrics.attribute_second_candidate_attempts == 1
    assert artifact.metrics.attribute_completion_candidate1_accepted == 0
    assert artifact.metrics.attribute_completion_candidate2_accepted == 1
    assert record.completion_seed == boogu.seeds[1]
    assert boogu.calls == 2
    assert qwen.bbox_calls == 0


def test_two_completion_rejects_never_fall_back_to_raw_or_bbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two failing completion chains end the attribute, exactly like legacy."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    _config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-two-rejects",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        completion_reviews=[
            _completion_review("reject"),
            _completion_review("reject"),
        ],
        boogu=[_generated_png(), _generated_png()],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    assert artifact.records[0].reason == "completion_qwen_review_reject"
    assert artifact.metrics.completion_attempts == 2
    assert artifact.metrics.completion_qwen_review_rejects == 2
    assert artifact.metrics.completion_rejected == 1
    assert artifact.metrics.completion_fallback_to_raw == 0
    assert artifact.metrics.attribute_bbox_fallback_attempts == 0
    assert boogu.calls == 2
    assert qwen.bbox_calls == 0
    assert runner.reconcile_stats(SHARD).terminal_clips == 1


def test_rank1_insufficient_evidence_falls_back_to_the_bbox_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last-resort bbox review is one Qwen job per attribute."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-bbox")
    _add_owner_frame(storage, slot=1)
    runner = _runner(config, storage, tmp_path)
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
        ],
        bbox_reviews=[_bbox_review("accept")],
    )
    sam = _SamBackend(by_prompt={ACCESSORY[2]: masks})
    _drain(runner, qwen, sam=sam)

    assert qwen.bbox_calls == 1
    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["accepted"]
    assert artifact.records[0].final_selection == "bbox"
    assert artifact.metrics.attribute_bbox_fallback_attempts == 1
    assert artifact.metrics.attribute_bbox_fallback_accepted == 1
    assert artifact.metrics.attribute_second_candidate_attempts == 1
    assert artifact.metrics.review_calls == 2


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("reject", "attribute_bbox_fallback_rejected:bbox crop is usable"),
        ("failed", "attribute_bbox_fallback_rejected:attribute_bbox_judge_failed"),
    ],
)
def test_bbox_review_reject_and_judge_failure_are_terminal_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str, expected: str
) -> None:
    """A rejected or failed bbox review is one legacy terminal reject."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(
        tmp_path, monkeypatch, f"run-bbox-{verdict}"
    )
    _add_owner_frame(storage, slot=1)
    runner = _runner(config, storage, tmp_path)
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    bbox_result: Any = (
        ValueError("bbox exploded") if verdict == "failed" else _bbox_review("reject")
    )
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
        ],
        bbox_reviews=[bbox_result],
    )
    sam = _SamBackend(by_prompt={ACCESSORY[2]: masks})
    _drain(runner, qwen, sam=sam)

    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    assert artifact.records[0].reason.startswith(expected)
    if verdict == "failed":
        assert artifact.records[0].reason.endswith(
            "attribute_bbox_judge_failed:ValueError:bbox exploded"
        )
    assert artifact.metrics.attribute_bbox_fallback_attempts == 1
    assert artifact.metrics.attribute_bbox_fallback_accepted == 0
    assert qwen.bbox_calls == 1


def test_face_accept_never_pays_completion_or_the_bbox_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A face is accepted from the raw review with the CPU bbox variant."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    _config, storage, runner, qwen, sam, boogu, timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-face",
        attribute=FACE_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    assert boogu.calls == 0
    assert qwen.bbox_calls == 0
    assert qwen.completion_review_calls == 0
    artifact = _read_artifact(storage)
    record = artifact.records[0]
    assert record.status == "accepted"
    assert record.attribute_type == "face"
    assert record.final_selection == "bbox"
    assert record.variants.bbox.review_status == "accepted_via_raw_face_review"
    assert artifact.metrics.attribute_bbox_fallback_attempts == 0
    assert artifact.metrics.attribute_bbox_fallback_accepted == 0
    assert artifact.metrics.attribute_source_candidates_considered == 2
    assert timeline == ["discovery", "sam:1", "sam:0", "review:a1"]


def test_longest_chain_restart_pays_no_model_call_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The longest chain replays entirely on CPU once every receipt is committed."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-longest",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        completion_reviews=[
            _completion_review("reject"),
            _completion_review("accept"),
        ],
        boogu=[_generated_png(), _generated_png(colour=(40, 90, 200))],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    artifact = _read_artifact(storage)
    counts = runner.reconcile_stats(SHARD).to_dict()
    outcome = json.loads(_owner_outcome_path(runner).read_text(encoding="utf-8"))
    assert outcome["source"] == "processed"
    assert boogu.calls == 2
    assert sam.calls == 2
    assert sam.generated_calls == 2
    assert qwen.review_calls == 2
    assert qwen.completion_review_calls == 2

    # Crash between the last model receipt and the processed owner outcome: the
    # artifact is on disk but the outcome is not.
    _owner_outcome_path(runner).unlink()

    fresh = _runner(config, storage, tmp_path, ledger_name="ledger-run-longest")
    fresh_qwen = _QwenClient()
    fresh_sam = _SamBackend()
    fresh_boogu = _BooguBackend()
    executor, jobs = _drain(fresh, fresh_qwen, sam=fresh_sam, boogu=fresh_boogu)

    assert jobs == [], "every call is durable, so nothing is re-seeded"
    assert executor.executed == []
    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0
    assert fresh_qwen.completion_review_calls == 0
    assert fresh_qwen.bbox_calls == 0
    assert fresh_sam.calls == 0
    assert fresh_sam.generated_calls == 0
    assert fresh_boogu.calls == 0
    assert _read_artifact(storage).model_dump(mode="json") == artifact.model_dump(
        mode="json"
    )
    assert json.loads(
        _owner_outcome_path(fresh).read_text(encoding="utf-8")
    ) == outcome
    assert fresh.reconcile_stats(SHARD).to_dict() == counts


# ---------------------------------------------------------------------------
# 8c seal: durable model timing, failure scope, policy identity and markers
# ---------------------------------------------------------------------------


def _monotonic_clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A strictly increasing deterministic clock, so every call reports time."""
    counter = itertools.count(1)
    monkeypatch.setattr(time, "perf_counter", lambda: float(next(counter)))


def _committed_qwen_seconds(runner: Any, statuses: set[str]) -> float:
    """Sum the Qwen durations of committed receipts with these statuses."""
    total = 0.0
    for path in sorted(Path(runner.ledger.root).rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        body = payload.get("result_payload")
        if not isinstance(body, Mapping) or str(body.get("status")) not in statuses:
            continue
        value = body.get("qwen_model_call_time_seconds", 0.0)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += max(0.0, float(value))
    return total


def test_qwen_model_time_is_the_sum_of_the_receipts_legacy_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Qwen total covers every review round, not just the discovery call."""
    _monotonic_clock(monkeypatch)
    _config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-qwen-time",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        completion_reviews=[
            _completion_review("reject"),
            _completion_review("accept"),
        ],
        boogu=[_generated_png(), _generated_png(colour=(40, 90, 200))],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    # Exactly discovery, the two raw review rounds and the two completion
    # reviews carry Qwen time; the bbox judge never ran here.
    expected = _committed_qwen_seconds(runner, {"discovery", "review"})
    assert expected > 0.0
    artifact = _read_artifact(storage)
    assert artifact.metrics.qwen_model_call_time_seconds == pytest.approx(expected)

    discovery_only = float(
        runner._discovery_payload(SHARD, CLIP_UID, _owner_plan_of(runner)).get(
            "qwen_model_call_time_seconds", 0.0
        )
    )
    assert artifact.metrics.qwen_model_call_time_seconds > discovery_only
    assert qwen.review_calls == 2
    assert qwen.completion_review_calls == 2


def test_bbox_review_time_is_not_part_of_the_qwen_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy never adds the bbox fallback call to the owner Qwen total."""
    _monotonic_clock(monkeypatch)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-bbox-time")
    _add_owner_frame(storage, slot=1)
    runner = _runner(config, storage, tmp_path)
    masks = [
        _attribute_mask(storage, slot=1, band=0),
        _attribute_mask(storage, slot=0, band=0),
    ]
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=(ACCESSORY,))],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1", sufficient_source_evidence=False)],
            ),
        ],
        bbox_reviews=[_bbox_review("accept")],
    )
    sam = _SamBackend(by_prompt={ACCESSORY[2]: masks})
    _drain(runner, qwen, sam=sam)

    assert qwen.bbox_calls == 1
    expected = _committed_qwen_seconds(runner, {"discovery", "review"})
    bbox_seconds = _committed_qwen_seconds(runner, {"bbox", "bbox_failed"})
    assert bbox_seconds > 0.0
    artifact = _read_artifact(storage)
    assert artifact.metrics.qwen_model_call_time_seconds == pytest.approx(expected)
    assert artifact.metrics.qwen_model_call_time_seconds != pytest.approx(
        expected + bbox_seconds
    )


def _owner_plan_of(runner: Any) -> dict[str, Any]:
    return runner._owner_plan_for(
        SHARD, runner._clip_plan(SHARD, CLIP_UID), OWNER
    )


@pytest.mark.parametrize(
    ("scripted", "reported", "expected"),
    [
        (NO_OUTPUT, 0.0, "FileNotFoundError"),
        (b"definitely not a PNG", 0.0, "UnidentifiedImageError"),
        (_generated_png(), float("nan"), None),
    ],
)
def test_completion_generation_failure_scope_matches_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scripted: Any,
    reported: Any,
    expected: str | None,
) -> None:
    """A missing, corrupt or mis-reported generation is a terminal completion failure."""
    _monotonic_clock(monkeypatch)
    _config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name=f"run-gen-{abs(hash(str(expected)))}",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        boogu=[scripted, scripted],
        reported=reported,
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    # Both generations failed, so the completion SAM never ran at all.
    assert boogu.calls == 2
    assert sam.generated_calls == 0
    assert runner.seed_jobs() == []
    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    reason = artifact.records[0].reason
    assert reason.startswith("completion_failed:")
    if expected is not None:
        assert expected in reason
    else:
        assert reason == "completion_failed:ValueError:completion model-call time is invalid"
    assert artifact.metrics.completion_failures == 2
    assert artifact.metrics.completion_backend_failures == 2
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    completion_outcome = json.loads(
        (
            Path(runner.ledger.root)
            / "semantic"
            / "subject_attributes"
            / "owners"
            / SHARD
            / CLIP_UID
            / OWNER
            / "attributes"
            / "a1"
            / "completion"
            / "rank-0-outcome.json"
        ).read_text(encoding="utf-8")
    )
    assert completion_outcome["status"] == "generation_failed"
    assert completion_outcome["review_job_id"] is None
    assert completion_outcome["completed_crop_sha256"] is None


def test_policy_identities_bind_prompts_schemas_and_every_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The semantic identities carry the real prompts, schemas and thresholds."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage = _storage_variant(tmp_path, monkeypatch, "run-policy")
    runner = _runner(config, storage, tmp_path)

    completion_review = runner._completion_review_policy_identity()
    assert completion_review["policy_version"] == "subject_attribute_completion_review/2"
    assert completion_review["temperature"] == 0.0
    assert completion_review["top_p"] == 1.0
    assert completion_review["presence_penalty"] == 0.0
    assert completion_review["repair_retries"] == 1
    assert len(completion_review["system_prompt_sha256"]) == 64
    assert len(completion_review["review_schema_sha256"]) == 64
    assert completion_review["system_prompt_sha256"] == _sha256_bytes(
        ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT.encode("utf-8")
    )
    assert "endpoint" not in completion_review
    assert "timeout" not in completion_review

    bbox = runner._bbox_review_policy_identity()
    assert bbox["policy_version"] == "subject_attribute_bbox_review/2"
    assert len(bbox["bbox_system_prompt_sha256"]) == 64
    assert len(bbox["face_bbox_system_prompt_sha256"]) == 64
    assert len(bbox["bbox_schema_sha256"]) == 64
    assert bbox["bbox_system_prompt_sha256"] != bbox["face_bbox_system_prompt_sha256"]

    postcheck = runner._completion_postcheck_policy_identity()
    assert (
        postcheck["policy_version"]
        == "subject_attribute_completion_postcheck/1"
    )
    quality = config.subject_attributes.completion.quality_filter
    for key, value in (
        ("quality_filter_secondary_component_ratio_max", quality.secondary_component_ratio_max),
        ("quality_filter_second_component_ratio_max", quality.second_component_ratio_max),
        ("quality_filter_soft_alpha_ratio_max", quality.soft_alpha_ratio_max),
        ("quality_filter_weak_alpha_ratio_max", quality.weak_alpha_ratio_max),
        ("quality_filter_outer_weak_alpha_ratio_max", quality.outer_weak_alpha_ratio_max),
        ("quality_filter_foreground_fill_min", quality.foreground_fill_min),
        ("quality_filter_cleanup_component_ratio_max", quality.cleanup_component_ratio_max),
        ("quality_filter_cleanup_component_pixels_max", quality.cleanup_component_pixels_max),
    ):
        assert postcheck[key] == value, key

    # A single quality-filter threshold change must move the completion SAM job,
    # because that threshold decides the postcheck and therefore the crop the
    # completion review is shown.
    plan = runner._clip_plan(SHARD, CLIP_UID)
    owner_plan = runner._owner_plan_for(SHARD, plan, OWNER)
    reference = runner._owner_reference(storage, CLIP_UID, OWNER)
    discovery = _human_discovery(attributes=(COMPLETION_ATTRIBUTE,))
    discovery_job_id = runner._expected_discovery_job(
        SHARD, CLIP_UID, owner_plan
    ).job_id()
    attribute_plan = runner._attribute_plan(
        SHARD,
        storage,
        CLIP_UID,
        owner_plan,
        reference,
        discovery,
        0,
        "a1",
        discovery_job_id,
    )
    stub = SimpleNamespace(
        crop=Image.new("RGBA", (8, 8), (10, 20, 30, 255)),
        owner_candidate=SimpleNamespace(candidate_id="candidate_1"),
    )

    def sam_job_id(active: Any) -> str:
        generate = active._expected_completion_generate_job(
            SHARD, CLIP_UID, owner_plan, attribute_plan, stub, 0, 7
        )
        return active._expected_completion_sam_job(
            SHARD,
            CLIP_UID,
            owner_plan,
            attribute_plan,
            stub,
            0,
            generate,
            "0" * 64,
            "candidate.png",
        ).job_id()

    mutated = _runner(
        replace(
            config,
            subject_attributes=replace(
                config.subject_attributes,
                completion=replace(
                    config.subject_attributes.completion,
                    quality_filter=replace(
                        quality, foreground_fill_min=0.99
                    ),
                ),
            ),
        ),
        storage,
        tmp_path,
        ledger_name="ledger-run-policy-mutated",
    )
    assert sam_job_id(runner) != sam_job_id(mutated)


def test_rank1_completion_outcome_marker_is_published_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rank-1 completion outcome is durable authority, never its own."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-rank1-marker",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        completion_reviews=[
            _completion_review("reject"),
            _completion_review("accept"),
        ],
        boogu=[_generated_png(), _generated_png(colour=(40, 90, 200))],
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    completion_dir = (
        Path(runner.ledger.root)
        / "semantic"
        / "subject_attributes"
        / "owners"
        / SHARD
        / CLIP_UID
        / OWNER
        / "attributes"
        / "a1"
        / "completion"
    )
    rank0 = json.loads(
        (completion_dir / "rank-0-outcome.json").read_text(encoding="utf-8")
    )
    rank1_path = completion_dir / "rank-1-outcome.json"
    rank1 = json.loads(rank1_path.read_text(encoding="utf-8"))
    assert rank0["candidate_rank"] == 0
    assert rank0["status"] == "review_rejected"
    assert rank1["candidate_rank"] == 1
    assert rank1["status"] == "accepted"
    assert rank1["review_job_id"] is not None
    assert rank1["reason"] == "completion_identity_accepted"
    assert rank1["completed_crop_sha256"] is not None
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    for key, value in (
        ("status", "generation_failed"),
        ("seed", int(rank1["seed"]) + 1),
        ("review_job_id", None),
    ):
        tampered = dict(rank1)
        tampered[key] = value
        rank1_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(
            SubjectAttributeDurableError, match="completion outcome drifted"
        ):
            _runner(
                config, storage, tmp_path, ledger_name="ledger-run-rank1-marker"
            ).reconcile_stats(SHARD)
        # Verification is pure CPU: no handle is ever touched, so a tampered
        # marker can never cost a model call.
        assert (qwen.discovery_calls, qwen.review_calls, sam.calls, boogu.calls) == (
            1,
            2,
            2,
            2,
        )
    rank1_path.write_text(json.dumps(rank1), encoding="utf-8")
    assert runner.reconcile_stats(SHARD).terminal_clips == 1


def _committed_completion_receipts(
    runner: Any, status: str
) -> list[dict[str, Any]]:
    """Every committed completion generation receipt with this status."""
    found: list[dict[str, Any]] = []
    for path in sorted(Path(runner.ledger.root).rglob("result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = payload.get("result_payload") if isinstance(payload, dict) else None
        if isinstance(body, Mapping) and str(body.get("status")) == status:
            found.append(dict(body))
    return found


def test_completion_model_time_survives_a_corrupt_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy measures the duration before it opens the generated file."""
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    _config, storage, runner, qwen, sam, boogu, _timeline = _completion_fixture(
        tmp_path,
        monkeypatch,
        run_name="run-completion-time",
        attribute=COMPLETION_ATTRIBUTE,
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[
                    _raw_review(
                        "a1", structure_complete=False, completion_recommended=True
                    )
                ],
            ),
        ],
        boogu=[b"corrupt", b"corrupt"],
        reported=3.25,
    )
    _drain(runner, qwen, sam=sam, boogu=boogu)

    artifact = _read_artifact(storage)
    assert [record.status for record in artifact.records] == ["rejected"]
    # Two generations reported 3.25 each before their output failed to load.
    assert artifact.metrics.completion_model_call_time_seconds == pytest.approx(6.5)
    assert artifact.metrics.completion_attempts == 2
    assert artifact.metrics.completion_failures == 2
    assert sam.generated_calls == 0
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    # The committed semantic is the legacy one: the generation failed inside the
    # same scope that measures the duration, so the receipt is a terminal failure
    # that still carries the reported duration rather than a success.
    failures = _committed_completion_receipts(runner, "completion_failed")
    assert len(failures) == 2
    assert all(
        payload["model_call_time_seconds"] == pytest.approx(3.25)
        for payload in failures
    )
    assert _committed_completion_receipts(runner, "completion") == []


# ---------------------------------------------------------------------------
# The legacy probe index is the only replay job index authority
# ---------------------------------------------------------------------------

#: The frozen owner-plan order this section's fixture produces, and the legacy
#: probe order legacy derives from it. ``_SLOT_PRIORITY`` ranks slot 5 before
#: slot 4 before slot 6, and the owner reference sits on slot 5, so
#: ``prefer_attribute_candidate_frames`` moves that frame to the end and the two
#: orders genuinely disagree. A frozen index of 0 is then never probed at all.
FROZEN_PROBE_SLOTS = (5, 4, 6)
LEGACY_PROBE_SLOTS = (4, 6, 5)
#: One grounding prompt, two different attribute types: the schema forbids
#: repeating a type, so this is the only way two attributes of one owner can
#: share a prompt.
SHARED_PROMPT = "the strap over the left shoulder"
SHARED_PROMPT_ATTRIBUTES = (
    ("accessory", "a leather belt", SHARED_PROMPT),
    ("bag", "a small satchel", SHARED_PROMPT),
)


def _point_owner_reference_at(storage: Any, slot: int) -> None:
    """Move the owner's published reference onto one frame slot.

    The reference frame is what ``prefer_attribute_candidate_frames`` sorts last,
    so a reference frame that is also a probe candidate is exactly what makes the
    legacy probe order differ from the frozen owner-plan order.
    """
    from r2v_data_v2.reconciliation import write_json_atomic

    path = Path(storage.root) / "clips" / CLIP_UID / "clip.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for reference in payload["references"]["entities"]:
        if reference["entity_id"] == OWNER:
            reference["source_frame_index"] = int(slot)
    write_json_atomic(path, payload)


def _probe_block(
    storage: Any,
    *,
    slot: int,
    top_offset: int,
    rows: int,
    left_offset: int = 5,
    width: int = 8,
) -> Any:
    """A usable SAM attribute mask of a *chosen shape* inside the owner.

    The fixture frame is one flat colour, so two probes only produce different
    crops when their mask *shapes* differ. A per-slot script that returned the
    same block everywhere would let a probe be answered by the wrong receipt
    without changing a single recorded byte.
    """
    owner = _owner_mask(storage, slot)
    row_indices, column_indices = np.nonzero(owner)
    top = int(row_indices.min())
    left = int(column_indices.min())
    block = np.zeros_like(owner)
    block[
        top + top_offset : top + top_offset + rows,
        left + left_offset : left + left_offset + width,
    ] = True
    usable = np.logical_and(block, owner)
    assert int(usable.sum()) >= 16, "fixture probe mask is too small"
    return usable


def _probe_order_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_name: str
) -> tuple[Any, Any]:
    """A ready-pair clip whose frozen and legacy probe orders disagree."""
    config, storage = _storage_variant(tmp_path, monkeypatch, run_name)
    for slot in (4, 5, 6):
        _add_owner_frame(storage, slot=slot)
    _point_owner_reference_at(storage, 5)
    return config, storage


def _probe_orders(runner: Any, attribute_id: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The frozen owner-plan order and the frozen legacy probe order, from disk.

    Both are durable: the owner plan freezes the candidate pool and the attribute
    plan freezes the legacy order's candidate IDs, so this reads what the epoch
    actually committed rather than re-deriving it.
    """
    plan = _planned_owner_plan(runner)
    slot_of = {
        str(item["candidate_id"]): int(item["frame_slot"])
        for item in plan["candidates"]
    }
    frozen = tuple(int(item["frame_slot"]) for item in plan["candidates"])
    attribute_plan = json.loads(
        _attribute_plan_file(runner, attribute_id).read_text(encoding="utf-8")
    )
    legacy = tuple(
        slot_of[str(candidate_id)]
        for candidate_id in attribute_plan["ordered_owner_candidate_ids"]
    )
    return frozen, legacy


def test_sam_receipt_follows_the_legacy_probe_order_not_the_frozen_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe must be answered by the receipt of the frame it actually asked for.

    With the frozen and legacy orders disagreeing, the first legacy probe sits at
    frozen index 1. Reading the chain by that frozen index answers probe 0 with
    probe 1's receipt; reading it by the legacy probe index answers it with its
    own, which is the whole reason the replay indexes jobs and receipts by the
    legacy probe order.
    """
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    attributes = (ACCESSORY,)

    legacy_config, legacy_storage = _probe_order_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_sam = _SamBackend(
        by_slot={
            4: [_probe_block(legacy_storage, slot=4, top_offset=2, rows=4)],
            6: [_probe_block(legacy_storage, slot=6, top_offset=9, rows=6)],
        }
    )
    legacy_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    _legacy_run(
        legacy_config,
        legacy_storage,
        legacy_qwen,
        review=legacy_qwen,
        segmentation=legacy_sam,
    )
    legacy_artifact = _read_artifact(legacy_storage)
    assert [record.owner_candidate_id for record in legacy_artifact.records] == [
        "candidate_2"
    ], "legacy's first probe is the frozen order's *second* candidate"
    assert legacy_sam.calls == 2, "legacy stops once it collected its quota"

    config, storage = _probe_order_fixture(tmp_path, monkeypatch, "run-epoch")
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER, reviews=[_raw_review("a1")]
            )
        ],
    )
    sam = _SamBackend(
        by_slot={
            4: [_probe_block(storage, slot=4, top_offset=2, rows=4)],
            6: [_probe_block(storage, slot=6, top_offset=9, rows=6)],
        }
    )
    runner = _runner(config, storage, tmp_path)
    _drain(runner, qwen, sam=sam)

    # The premise, read back from the frozen plans rather than assumed.
    frozen, legacy = _probe_orders(runner, "a1")
    assert frozen == FROZEN_PROBE_SLOTS, frozen
    assert legacy == LEGACY_PROBE_SLOTS, legacy
    assert legacy[0] != frozen[0]
    assert frozen.index(legacy[0]) == 1, "the first legacy probe is not frozen index 0"

    # Probe 0 and probe 1 were answered by two different receipts, so the order
    # they were read in is observable rather than incidental.
    marker = _selection_marker(runner, "a1")
    assert marker["status"] == "candidates"
    assert [option["source_frame_slot"] for option in marker["options"]] == [4, 6]
    cropped = [option["crop_sha256"] for option in marker["options"]]
    assert len(set(cropped)) == 2, "per-slot probes must not be interchangeable"
    assert sam.calls == 2, "the epoch probes exactly what legacy probes"

    # The replay is the legacy authority, candidate for candidate.
    assert _read_artifact(storage).model_dump(mode="json") == legacy_artifact.model_dump(
        mode="json"
    )
    record = _read_artifact(storage).records[0]
    assert record.status == "accepted"
    assert record.owner_candidate_id == "candidate_2"
    assert record.source_frame_slot == 4
    # Its pixels are probe 0's crop, not probe 1's.
    published = _sha256_bytes(
        (_output_root(storage) / str(record.image_path)).read_bytes()
    )
    assert published == marker["options"][0]["crop_sha256"]
    assert published != marker["options"][1]["crop_sha256"]


def test_same_prompt_attributes_keep_their_own_probe_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two attributes that share a prompt must each restart at the first probe.

    ``_collect_attribute_candidates`` may leave an attribute early once it has
    collected ``MAX_ATTRIBUTE_SOURCE_CANDIDATES`` candidates, so the tail of the
    legacy probe order is legitimately never probed. The next attribute then
    starts again from the first probe, even though its prompt is the same. A
    probe *count* cannot tell those apart - the unprobed tail never arrives - so
    it hands the second attribute's first probe to the first attribute and pays
    for a probe legacy never makes.
    """
    monkeypatch.setattr(time, "perf_counter", lambda: 1.0)
    attributes = SHARED_PROMPT_ATTRIBUTES

    legacy_config, legacy_storage = _probe_order_fixture(
        tmp_path, monkeypatch, "run-legacy"
    )
    legacy_script = {
        4: [_probe_block(legacy_storage, slot=4, top_offset=2, rows=4)],
        6: [_probe_block(legacy_storage, slot=6, top_offset=9, rows=6)],
    }
    legacy_timeline: list[str] = []
    legacy_qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    _legacy_run(
        legacy_config,
        legacy_storage,
        legacy_qwen,
        review=legacy_qwen,
        segmentation=_SamBackend(by_slot=legacy_script, timeline=legacy_timeline),
    )
    legacy_artifact = _read_artifact(legacy_storage)
    # The premise: legacy probes C0 and C1 for a1, then C0 and C1 again for a2.
    assert legacy_timeline == [
        f"sam:{LEGACY_PROBE_SLOTS[0]}",
        f"sam:{LEGACY_PROBE_SLOTS[1]}",
        f"sam:{LEGACY_PROBE_SLOTS[0]}",
        f"sam:{LEGACY_PROBE_SLOTS[1]}",
    ], legacy_timeline
    assert [record.attribute_id for record in legacy_artifact.records] == ["a1", "a2"]

    config, storage = _probe_order_fixture(tmp_path, monkeypatch, "run-epoch")
    shared_script = {
        4: [_probe_block(storage, slot=4, top_offset=2, rows=4)],
        6: [_probe_block(storage, slot=6, top_offset=9, rows=6)],
    }
    qwen = _QwenClient(
        discoveries=[_human_discovery(attributes=attributes)],
        reviews=[
            SubjectAttributeReviewBatch(
                owner_entity_id=OWNER,
                reviews=[_raw_review("a1"), _raw_review("a2")],
            )
        ],
    )
    sam = _SamBackend(by_slot=shared_script)
    runner = _runner(config, storage, tmp_path)
    sam_executor = _SerialQwenExecutor(runner, sam)
    _scheduler(
        runner,
        _SerialQwenExecutor(runner, qwen),
        runner.finalize,
        sam_executor,
        None,
    ).run(runner.seed_jobs())

    frozen, legacy = _probe_orders(runner, "a1")
    assert frozen == FROZEN_PROBE_SLOTS, frozen
    assert legacy == LEGACY_PROBE_SLOTS, legacy
    assert _probe_orders(runner, "a2")[1] == LEGACY_PROBE_SLOTS

    # Every attribute probes the legacy order's first two slots, and only those:
    # probe index 2 is never requested, because legacy never requests it.
    probes: dict[str, list[int]] = {}
    job_attribute: dict[str, str] = {}
    for job in sam_executor.executed:
        if job.job_type != SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            continue
        target = dict(job.target)
        attribute_id = str(target["attribute_id"])
        job_attribute[job.job_id()] = attribute_id
        probes.setdefault(attribute_id, []).append(int(target["probe_index"]))
    assert set(probes) == {"a1", "a2"}, probes
    for attribute_id in ("a1", "a2"):
        assert sorted(probes[attribute_id]) == [0, 1], probes
    assert sam.calls == 4, "two attributes, two probes each - never a third"

    # No cross-binding: each attribute's selection points at its own SAM jobs,
    # at the same two legacy probe slots, and never at its sibling's.
    for attribute_id in ("a1", "a2"):
        marker = _selection_marker(runner, attribute_id)
        assert marker["status"] == "candidates", marker
        options = marker["options"]
        assert [option["source_frame_slot"] for option in options] == [4, 6]
        assert {job_attribute[option["sam_job_id"]] for option in options} == {
            attribute_id
        }
    assert qwen.discovery_calls == 1
    # Both attributes claim the same pixels, so this is a legacy duplicate
    # conflict: legacy reviews neither of them, and neither is reviewed here.
    assert qwen.review_calls == 0

    # The epoch replays the legacy authority exactly, duplicate conflict included:
    # the two attributes claim the same pixels, which is a normal legacy outcome.
    assert _read_artifact(storage).model_dump(mode="json") == legacy_artifact.model_dump(
        mode="json"
    )
    raw_state = _raw_state_marker(runner)
    assert [item["route_after_rank0"] for item in raw_state["attributes"]] == [
        "duplicate_conflict",
        "duplicate_conflict",
    ]
    assert runner.reconcile_stats(SHARD).terminal_clips == 1

    # A fresh runner re-walks the same sequence on CPU: no SAM, no Qwen, and the
    # artifact and counts are exactly the ones already durable.
    fresh = _runner(config, storage, tmp_path)
    assert fresh.seed_jobs() == []
    fresh_qwen = _QwenClient()
    fresh_sam = _SamBackend(by_slot=shared_script)
    _drain(fresh, fresh_qwen, sam=fresh_sam)
    assert fresh_sam.calls == 0
    assert fresh_qwen.discovery_calls == 0
    assert fresh_qwen.review_calls == 0
    assert _read_artifact(storage).model_dump(mode="json") == legacy_artifact.model_dump(
        mode="json"
    )
    assert fresh.reconcile_stats(SHARD).to_dict() == runner.reconcile_stats(
        SHARD
    ).to_dict()
