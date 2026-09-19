"""Resource-epoch adapter for the frozen background-removal semantics.

Semantic contract
-----------------

This module reuses the shared helpers in :mod:`r2v_data_v2.v3.remove`
(``prepare_removal_attempt_context``, ``generate_removal_candidate``,
``review_removal_candidate``, ``removal_attempts_are_all_rejected``) and the
existing publication helpers. Nothing here re-describes a prompt, a candidate
preparation rule, a judge call or an accept/reject rule.

v1 supports exactly one topology:

    BOOGU epoch   GPU0-7, eight persistent Boogu workers, one generation each
    QWEN epoch    one managed Qwen3-VL server, TP1 x DP8, judging over HTTP

``config.remove.backend`` must be the Boogu backend. The legacy
``remove_backgrounds()`` still supports the Qwen Image Edit remover unchanged;
this adapter refuses anything else instead of loading a second heavy remover.

Attempt semantics are legacy-equivalent
---------------------------------------

* A Boogu seed is drawn **once** with ``new_boogu_seed()`` and then persisted in
  a durable seed plan, so a restart reuses the same seed and the same job
  identity instead of re-randomising it.
* ``attempt_index`` is global and monotonic:
  ``len(state.removal_attempts) + cycle_index``. A second launch therefore plans
  new jobs instead of colliding with the previous cycle's receipts.
* A generation or judge *semantic* failure is a durable committed attempt with
  ``status="failed"``, exactly like legacy, and the CPU finalizer walks to the
  next configured candidate. Only the inability to form a legal semantic result
  (missing artifact, digest mismatch, unreadable committed result) is retryable
  at the scheduler level.
* Terminal policy is legacy: all-rejected publishes ``rejected`` with only this
  cycle's attempts; any failure publishes ``retryable`` with the carried-over
  attempts and leaves the clip ``pending_remove`` for the next launch.

Nothing here may decide that a *group* is complete: the remove stage is one
phase of a DAG that is not fully wired, so the runner always reports
``completed=False``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.v3.background import validate_background_reference
from r2v_data_v2.v3.boogu_remove_backend import BooguBackgroundRemovalBackend
from r2v_data_v2.v3.boogu_seed import new_boogu_seed
from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND, V3Config
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    semantic_input_digest,
)
from r2v_data_v2.v3.post_mask_epoch_resources import (
    QwenConcurrentExecutor,
    QwenEpochConfig,
    QwenEpochResource,
    ResourceEpochManager,
    WorkerPoolConfig,
    WorkerSlotExecutor,
    build_boogu_epoch,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    DEFAULT_WINDOW_SIZE,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import (
    GroupLedger,
    atomic_write_json,
    file_lock,
)
from r2v_data_v2.v3.post_mask_production import (
    ShardPaths,
    hydrate_shard,
    initialize_shard,
)
from r2v_data_v2.v3.removal_judge import QwenBackgroundRemovalJudge
from r2v_data_v2.v3.remove import (
    CANDIDATE_READY,
    RemovalAttemptContext,
    _debug_enabled,
    _exception_reason,
    _publish_ready,
    _publish_rejected,
    _publish_retryable,
    _write_candidate_debug,
    generate_removal_candidate,
    prepare_removal_attempt_context,
    removal_attempts_are_all_rejected,
    review_removal_candidate,
)
from r2v_data_v2.v3.schemas import BackgroundReferenceState, BackgroundRemovalAttempt
from r2v_data_v2.v3.storage import RunStorage

REMOVAL_GENERATE_JOB = "background_removal_generate"
#: Conditional later candidate of the same removal cycle.
REMOVAL_CANDIDATE2_JOB = "background_removal_candidate2"
REMOVAL_JUDGE_JOB = "background_removal_judge"

#: The judge always runs on the managed Qwen server epoch.
REMOVAL_JUDGE_RESOURCE = RESOURCE_QWEN

CANDIDATE_ARTIFACT = "candidate.png"

SEED_PLAN_SCHEMA_VERSION = "post_mask_epoch_remove_seed/1"
SEED_PLAN_ROOT = Path("semantic") / "background_remove"

PAYLOAD_ATTEMPT_STATUS = "attempt_status"
PAYLOAD_ATTEMPT = "attempt"
PAYLOAD_SEED = "seed"
PAYLOAD_ATTEMPT_INDEX = "attempt_index"
PAYLOAD_CYCLE_INDEX = "cycle_index"
PAYLOAD_CANDIDATE_SHA256 = "candidate_sha256"
PAYLOAD_GENERATION_SECONDS = "generation_seconds"
PAYLOAD_JUDGE_SECONDS = "judge_seconds"
PAYLOAD_GENERATED = "generated"
PAYLOAD_REASON = "reason"

ATTEMPT_STATUS_READY = "candidate_ready"
ATTEMPT_STATUS_FAILED = "failed"

REJECT_REASON_ALL = "all_removal_candidates_rejected"
REJECT_REASON_MASK = "generation_mask_too_large"
RETRY_REASON_INFRASTRUCTURE = "removal_infrastructure_failure"

#: Fail-safe detail for a job whose recomputed identity disagrees with its plan.
SEMANTIC_MISMATCH = "semantic identity mismatch"

#: Statuses that are never re-entered by the remove stage.
_TERMINAL_STATUSES = frozenset({"none", "clean_raw", "rejected"})

_COUNTER_NAMES = (
    "processed",
    "skipped_existing",
    "skipped_not_pending",
    "skipped_disabled",
    "failed",
    "ready_removed",
    "rejected",
    "retryable_pending",
    "candidates_generated",
    "candidates_rejected",
    "candidates_failed",
)

DEFAULT_QWEN_JUDGE_MODEL = Path(
    "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
)


def new_counters() -> dict[str, int]:
    return {name: 0 for name in _COUNTER_NAMES}


class RemovalEpochError(RuntimeError):
    """The removal adapter could not run a job safely; fail closed."""


def require_boogu_backend(config: V3Config) -> None:
    """resource-epoch remove v1 supports the Boogu remover only."""
    if config.remove.backend != BOOGU_REMOVE_BACKEND:
        raise RemovalEpochError(
            "resource_epoch_v3 background remove requires Boogu remove backend"
        )


def removal_generation_resource(config: V3Config) -> str:
    """Which epoch owns one background-removal generation."""
    require_boogu_backend(config)
    return RESOURCE_BOOGU


# ---------------------------------------------------------------------------
# Durable random seed plan
# ---------------------------------------------------------------------------


def seed_plan_path(
    ledger_root: Path,
    *,
    canonical_shard: str,
    clip_uid: str,
    attempt_index: int,
) -> Path:
    return (
        Path(ledger_root)
        / SEED_PLAN_ROOT
        / canonical_shard
        / clip_uid
        / f"attempt-{attempt_index:06d}.json"
    )


def removal_seed_anchor_inputs(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    attempt_index: int,
    cycle_index: int,
) -> dict[str, Any]:
    """Semantic anchor for one removal attempt. Never contains the seed.

    Everything that makes two attempts the *same* attempt is here; the random
    seed is what makes them different runs, so it is validated separately.
    """
    inputs = context.inputs
    return {
        "backend": config.remove.backend,
        "candidate_mode": context.candidate_mode,
        "prompt": context.prompt,
        "source_image": _image_digest(inputs.source_image),
        "source_mask": _mask_digest(inputs.source_mask),
        "generation_mask": _mask_digest(inputs.generation_mask),
        "removal_phrases": list(inputs.removal_phrases),
        "background_phrase": inputs.background_phrase,
        "generation_width": context.generation_width,
        "generation_height": context.generation_height,
        "generation_mask_dilation_pixels": (
            config.remove.generation_mask_dilation_pixels
        ),
        "target_area": config.reference_edit.target_area,
        "alignment": config.reference_edit.alignment,
        "model_identity": context.profile_model,
        "attempt_index": attempt_index,
        "cycle_index": cycle_index,
    }


def _validate_seed_plan(
    payload: Any,
    *,
    canonical_shard: str,
    clip_uid: str,
    attempt_index: int,
    cycle_index: int,
    anchor_digest: str,
    path: Path,
) -> int:
    if not isinstance(payload, dict):
        raise RemovalEpochError(f"seed plan {path} is not an object")
    if payload.get("schema_version") != SEED_PLAN_SCHEMA_VERSION:
        raise RemovalEpochError(f"seed plan {path} has an unknown schema version")
    for key, expected in (
        ("canonical_shard", canonical_shard),
        ("clip_uid", clip_uid),
        ("attempt_index", attempt_index),
        ("cycle_index", cycle_index),
        ("semantic_anchor_digest", anchor_digest),
    ):
        if payload.get(key) != expected:
            raise RemovalEpochError(
                f"seed plan {path} disagrees on {key}; refusing to reuse or rewrite"
            )
    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise RemovalEpochError(f"seed plan {path} holds an invalid seed")
    return seed


def load_boogu_seed(
    *,
    ledger_root: Path,
    canonical_shard: str,
    clip_uid: str,
    attempt_index: int,
    cycle_index: int,
    anchor_digest: str,
) -> int | None:
    """Return an already-persisted seed, or None when this attempt never ran."""
    path = seed_plan_path(
        ledger_root,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        attempt_index=attempt_index,
    )
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        raise RemovalEpochError(f"seed plan {path} is not parseable") from exc
    return _validate_seed_plan(
        payload,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        attempt_index=attempt_index,
        cycle_index=cycle_index,
        anchor_digest=anchor_digest,
        path=path,
    )


def load_or_create_boogu_seed(
    *,
    ledger_root: Path,
    canonical_shard: str,
    clip_uid: str,
    attempt_index: int,
    cycle_index: int,
    anchor_digest: str,
    allocator: Callable[[], int] = new_boogu_seed,
) -> int:
    """Draw a Boogu seed exactly once per semantic attempt, then reuse it.

    The seed must exist *before* a ModelJob identity is computed, so it cannot
    live in the job result: it is written to its own durable plan under the
    group ledger. A restart reads the same seed instead of re-randomising, which
    is what keeps job identity -- and therefore resume -- stable.
    """
    existing = load_boogu_seed(
        ledger_root=ledger_root,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        attempt_index=attempt_index,
        cycle_index=cycle_index,
        anchor_digest=anchor_digest,
    )
    if existing is not None:
        return existing
    seed = allocator()
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise RemovalEpochError("Boogu seed allocator returned an invalid seed")
    payload = {
        "schema_version": SEED_PLAN_SCHEMA_VERSION,
        "canonical_shard": canonical_shard,
        "clip_uid": clip_uid,
        "attempt_index": attempt_index,
        "cycle_index": cycle_index,
        "semantic_anchor_digest": anchor_digest,
        "seed": seed,
    }
    path = seed_plan_path(
        ledger_root,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        attempt_index=attempt_index,
    )
    atomic_write_json(path, payload)
    return seed


# ---------------------------------------------------------------------------
# Semantic inputs and job builders
# ---------------------------------------------------------------------------


def _image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    prefix = f"{rgb.width}x{rgb.height}:".encode()
    return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def _mask_digest(mask: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(mask))
    prefix = f"{array.shape}|{array.dtype.str}|".encode()
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def generation_semantic_inputs(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    seed: int,
    attempt_index: int,
    cycle_index: int,
) -> dict[str, Any]:
    """Everything that changes *what* one generation asks a model to do."""
    inputs = context.inputs
    return {
        "kind": "generation",
        "backend": config.remove.backend,
        "candidate_mode": context.candidate_mode,
        "prompt": context.prompt,
        "seed": seed,
        "attempt_index": attempt_index,
        "cycle_index": cycle_index,
        "removal_phrases": list(inputs.removal_phrases),
        "background_phrase": inputs.background_phrase,
        "source_image": _image_digest(inputs.source_image),
        "source_mask": _mask_digest(inputs.source_mask),
        "generation_mask": _mask_digest(inputs.generation_mask),
        "generation_width": context.generation_width,
        "generation_height": context.generation_height,
        "generation_mask_dilation_pixels": (
            config.remove.generation_mask_dilation_pixels
        ),
        "target_area": config.reference_edit.target_area,
        "alignment": config.reference_edit.alignment,
        "inference_profile": config.remove.inference_profile,
    }


def judge_semantic_inputs(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    seed: int,
    attempt_index: int,
    cycle_index: int,
    candidate_sha256: str,
    generation_job_id: str,
) -> dict[str, Any]:
    """Everything that changes *what* one judge call asks a model to do.

    The candidate is bound twice -- by digest here and by job dependency -- so a
    swapped or truncated artifact can never reach the judge.
    """
    service = config.qwen.background_remove_judge
    if service is None:
        raise RemovalEpochError("remove is enabled without a background remove judge")
    inputs = context.inputs
    return {
        "kind": "judge",
        "candidate_mode": context.candidate_mode,
        "seed": seed,
        "attempt_index": attempt_index,
        "cycle_index": cycle_index,
        "removal_phrases": list(inputs.removal_phrases),
        "background_phrase": inputs.background_phrase,
        "source_image": _image_digest(inputs.source_image),
        "source_mask": _mask_digest(inputs.source_mask),
        "generation_mask": _mask_digest(inputs.generation_mask),
        "candidate_sha256": candidate_sha256,
        "generation_job_id": generation_job_id,
        "judge_model": str(service.model),
    }


def build_removal_generate_job(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    canonical_shard: str,
    clip_uid: str,
    cycle_index: int,
    attempt_index: int,
    seed: int,
) -> ModelJob:
    """One Boogu generation call for one clip, cycle and attempt."""
    require_boogu_backend(config)
    return ModelJob.create(
        job_type=(
            REMOVAL_GENERATE_JOB if cycle_index == 0 else REMOVAL_CANDIDATE2_JOB
        ),
        resource=removal_generation_resource(config),
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        semantic_inputs=generation_semantic_inputs(
            config,
            context,
            seed=seed,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
        ),
        model_identity=context.profile_model,
        target={"cycle_index": str(cycle_index)},
        attempt_index=attempt_index,
        seed=seed,
        label=f"remove-generate-{cycle_index}",
    )


def build_removal_judge_job(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    canonical_shard: str,
    clip_uid: str,
    cycle_index: int,
    attempt_index: int,
    seed: int,
    candidate_sha256: str,
    generate_job: ModelJob,
) -> ModelJob:
    """One Qwen judge call for one committed candidate."""
    service = config.qwen.background_remove_judge
    if service is None:
        raise RemovalEpochError("remove is enabled without a background remove judge")
    return ModelJob.create(
        job_type=REMOVAL_JUDGE_JOB,
        resource=REMOVAL_JUDGE_RESOURCE,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        semantic_inputs=judge_semantic_inputs(
            config,
            context,
            seed=seed,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
            candidate_sha256=candidate_sha256,
            generation_job_id=generate_job.job_id(),
        ),
        model_identity=str(service.model),
        target={
            "cycle_index": str(cycle_index),
            "candidate_sha256": candidate_sha256,
            "generation_job_id": generate_job.job_id(),
        },
        attempt_index=attempt_index,
        seed=seed,
        dependencies={"candidate": generate_job.job_id()},
        label=f"remove-judge-{cycle_index}",
    )


class _JobRef:
    """Job-id-only view, so the ledger can be queried without a full ModelJob."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


def resolve_removal_backend(handle: Any, config: V3Config) -> Any:
    """Return a ``BackgroundRemovalBackend`` for one generation job.

    A persistent Boogu worker is wrapped, never adapted twice, so the shared
    prompt and generation-size rules stay the only implementation.
    """
    if handle is None:
        raise RemovalEpochError("generation epoch supplied no backend handle")
    if hasattr(handle, "removal_backend"):
        return resolve_removal_backend(handle.removal_backend, config)
    if hasattr(handle, "remove"):
        return handle
    if hasattr(handle, "edit"):
        return BooguBackgroundRemovalBackend(
            handle,
            target_area=config.reference_edit.target_area,
            alignment=config.reference_edit.alignment,
        )
    raise RemovalEpochError(
        "generation epoch handle implements neither remove() nor edit()"
    )


def resolve_removal_judge(handle: Any, config: V3Config) -> Any:
    """Return a ``BackgroundRemovalJudge`` for one judge job.

    The Qwen epoch hands the runner its endpoint, so the judge is rebuilt
    against that endpoint and nothing else about the service changes.
    """
    if handle is None:
        raise RemovalEpochError("judge epoch supplied no handle")
    if hasattr(handle, "judge"):
        return resolve_removal_judge(handle.judge, config)
    if hasattr(handle, "review"):
        return handle
    service = config.qwen.background_remove_judge
    if isinstance(handle, str):
        if service is None:
            raise RemovalEpochError("no configured background remove judge")
        return QwenBackgroundRemovalJudge(replace(service, base_url=handle))
    if service is not None and hasattr(handle, "base_url") and hasattr(handle, "model"):
        return QwenBackgroundRemovalJudge(handle)
    raise RemovalEpochError("judge epoch handle implements no review()")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RemovalEpochRunner:
    """CPU side of the removal epoch: seeding, finalizing and stage counts.

    The GPU side is :meth:`run`, called by the scheduler's executors with the
    live backend handle. Neither method may hold GPU, slot, rank or epoch
    state: every decision here is a function of the clip, the durable seed plan
    and one committed result, which is what makes a finalizer replayable.
    """

    def __init__(
        self,
        config: V3Config,
        storages: Mapping[str, RunStorage],
        ledger: GroupLedger,
        *,
        emit: Callable[..., None] | None = None,
        seed_allocator: Callable[[], int] = new_boogu_seed,
    ) -> None:
        require_boogu_backend(config)
        self.config = config
        self.storages: dict[str, RunStorage] = dict(storages)
        self.ledger = ledger
        self.emit = emit or (lambda *args, **kwargs: None)
        self.seed_allocator = seed_allocator
        self.stats: dict[str, dict[str, int]] = {
            shard: new_counters() for shard in self.storages
        }

    # -- CPU seeding ------------------------------------------------------
    def seed_jobs(self) -> list[ModelJob]:
        """Plan the unconditional first candidate of every pending clip.

        Only cycle 0 is seeded. A later candidate exists solely because a judge
        rejected its sibling or an attempt failed, so it can never be
        speculated here.
        """
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            storage = self.storages[shard]
            counters = self.stats[shard]
            for listed in storage.iter_clips():
                clip_uid = listed.clip_uid
                state = listed.references.background
                if state is None or state.status in _TERMINAL_STATUSES:
                    counters["skipped_not_pending"] += 1
                    continue
                if state.status == "ready_removed":
                    try:
                        validate_background_reference(storage, clip_uid, state)
                    except Exception as exc:  # noqa: BLE001 - report, never crash
                        storage.append_failure(
                            clip_uid=clip_uid,
                            stage="remove",
                            reason=_exception_reason(exc),
                        )
                        counters["failed"] += 1
                    else:
                        counters["skipped_existing"] += 1
                    continue
                if not self.config.remove.enabled:
                    counters["skipped_disabled"] += 1
                    continue
                try:
                    context = prepare_removal_attempt_context(
                        self.config, storage, clip_uid, state
                    )
                except Exception as exc:  # noqa: BLE001 - report, never crash
                    storage.append_failure(
                        clip_uid=clip_uid,
                        stage="remove",
                        reason=_exception_reason(exc),
                    )
                    counters["failed"] += 1
                    continue
                mask = context.inputs.generation_mask
                ratio = float(np.count_nonzero(mask)) / float(mask.size)
                if ratio > self.config.remove.max_generation_mask_area_ratio:
                    _publish_rejected(
                        storage,
                        clip_uid=clip_uid,
                        original=state,
                        reason=REJECT_REASON_MASK,
                    )
                    counters["processed"] += 1
                    counters["rejected"] += 1
                    continue
                jobs.append(
                    self._generate_job(
                        shard,
                        clip_uid,
                        state,
                        context,
                        cycle_index=0,
                    )
                )
        return jobs

    # -- model jobs -------------------------------------------------------
    def run(self, job: ModelJob, handle: Any) -> JobResult:
        """Execute one generation or judge model call under a live resource."""
        if job.job_type == REMOVAL_JUDGE_JOB:
            return self._run_judge(job, handle)
        return self._run_generate(job, handle)

    def _run_generate(self, job: ModelJob, handle: Any) -> JobResult:
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        cycle_index = _cycle_index(job)
        attempt_index = job.attempt_index
        storage = self._storage_for(job)
        state = self._pending_state(storage, clip_uid)
        context = prepare_removal_attempt_context(
            self.config, storage, clip_uid, state
        )
        seed = self._seed_for(
            shard, clip_uid, context, attempt_index, cycle_index
        )
        payload = {
            PAYLOAD_SEED: seed,
            PAYLOAD_ATTEMPT_INDEX: attempt_index,
            PAYLOAD_CYCLE_INDEX: cycle_index,
        }
        actual = generation_semantic_inputs(
            self.config,
            context,
            seed=seed,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
        )
        if semantic_input_digest(actual) != job.input_digest:
            # Never call a model with inputs that are not the planned inputs.
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail=SEMANTIC_MISMATCH,
            )
        backend = resolve_removal_backend(handle, self.config)
        generation = generate_removal_candidate(
            context=context,
            backend=backend,
            seed=seed,
            retry_index=cycle_index,
        )
        payload[PAYLOAD_GENERATION_SECONDS] = generation.generation_seconds
        payload[PAYLOAD_GENERATED] = generation.generated
        if generation.status != CANDIDATE_READY:
            attempt = BackgroundRemovalAttempt(
                seed=seed,
                status="failed",
                runtime_seconds=generation.generation_seconds,
                candidate_sha256=None,
                reason=generation.reason,
                review=None,
            )
            payload[PAYLOAD_ATTEMPT_STATUS] = ATTEMPT_STATUS_FAILED
            payload[PAYLOAD_ATTEMPT] = attempt.model_dump(mode="json")
            payload[PAYLOAD_REASON] = generation.reason
            # A failed generation is a durable semantic attempt, not an
            # execution failure: legacy records it and walks to the next
            # candidate without leaving the clip unresolved.
            return JobResult(
                outcome=OUTCOME_COMPLETED,
                payload=payload,
                detail=generation.reason or "generation failed",
            )
        payload[PAYLOAD_ATTEMPT_STATUS] = ATTEMPT_STATUS_READY
        payload[PAYLOAD_CANDIDATE_SHA256] = generation.candidate_sha256
        return JobResult(
            outcome=OUTCOME_COMPLETED,
            artifacts={CANDIDATE_ARTIFACT: generation.candidate_bytes},
            payload=payload,
        )

    def _run_judge(self, job: ModelJob, handle: Any) -> JobResult:
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        cycle_index = _cycle_index(job)
        attempt_index = job.attempt_index
        target = dict(job.target)
        expected_sha = target.get("candidate_sha256")
        generate_job_id = dict(job.dependency_digests).get("candidate")
        payload = {
            PAYLOAD_SEED: job.seed,
            PAYLOAD_ATTEMPT_INDEX: attempt_index,
            PAYLOAD_CYCLE_INDEX: cycle_index,
        }
        if not expected_sha or not generate_job_id:
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail="judge job does not bind its candidate",
            )
        generation = self._load_result(generate_job_id)
        if generation is None or generation.payload.get(
            PAYLOAD_CANDIDATE_SHA256
        ) != expected_sha:
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail="committed generation result is missing or changed",
            )
        candidate_bytes = self._read_artifact(generate_job_id, CANDIDATE_ARTIFACT)
        actual_sha = hashlib.sha256(candidate_bytes).hexdigest()
        if actual_sha != expected_sha:
            # A tampered or truncated artifact must never reach the judge.
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail="candidate artifact does not match its committed digest",
            )
        storage = self._storage_for(job)
        state = self._pending_state(storage, clip_uid)
        context = prepare_removal_attempt_context(
            self.config, storage, clip_uid, state
        )
        seed = self._seed_for(shard, clip_uid, context, attempt_index, cycle_index)
        actual = judge_semantic_inputs(
            self.config,
            context,
            seed=seed,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
            candidate_sha256=actual_sha,
            generation_job_id=generate_job_id,
        )
        if semantic_input_digest(actual) != job.input_digest:
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail=SEMANTIC_MISMATCH,
            )
        judge = resolve_removal_judge(handle, self.config)
        generation_seconds = float(
            generation.payload.get(PAYLOAD_GENERATION_SECONDS, 0.0)
        )
        outcome = review_removal_candidate(
            context=context,
            candidate=self._candidate_image(candidate_bytes),
            candidate_bytes=candidate_bytes,
            candidate_sha256=actual_sha,
            judge=judge,
            seed=seed,
            generation_seconds=generation_seconds,
        )
        payload[PAYLOAD_JUDGE_SECONDS] = outcome.judge_seconds
        payload[PAYLOAD_GENERATION_SECONDS] = generation_seconds
        payload[PAYLOAD_ATTEMPT] = outcome.attempt.model_dump(mode="json")
        if outcome.status == "accepted":
            payload[PAYLOAD_ATTEMPT_STATUS] = "accepted"
            return JobResult(outcome=OUTCOME_COMPLETED, payload=payload)
        if outcome.status == "rejected":
            payload[PAYLOAD_ATTEMPT_STATUS] = "rejected"
            return JobResult(
                outcome=OUTCOME_TERMINAL_REJECT,
                payload=payload,
                detail=outcome.attempt.reason or "candidate rejected",
            )
        payload[PAYLOAD_ATTEMPT_STATUS] = ATTEMPT_STATUS_FAILED
        payload[PAYLOAD_REASON] = outcome.attempt.reason
        # A judge failure is a durable semantic attempt, exactly like legacy.
        return JobResult(
            outcome=OUTCOME_COMPLETED,
            payload=payload,
            detail=outcome.attempt.reason or "judge failed",
        )

    # -- CPU finalizers ---------------------------------------------------
    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """Apply one committed outcome. Returns the jobs it unlocks."""
        if job.job_type == REMOVAL_JUDGE_JOB:
            return self._finalize_judge(job, result)
        return self._finalize_generate(job, result)

    def _finalize_generate(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        cycle_index = _cycle_index(job)
        counters = self.stats[shard]
        if result.payload.get(PAYLOAD_GENERATED):
            counters["candidates_generated"] += 1
        storage = self._storage_for(job)
        state = self._current_state(storage, clip_uid)
        if state.status != "pending_remove":
            # Published by an earlier invocation that crashed after publishing.
            return ()
        if result.payload.get(PAYLOAD_ATTEMPT_STATUS) == ATTEMPT_STATUS_READY:
            context = prepare_removal_attempt_context(
                self.config, storage, clip_uid, state
            )
            seed = self._seed_for(
                shard, clip_uid, context, job.attempt_index, cycle_index
            )
            return (
                build_removal_judge_job(
                    self.config,
                    context,
                    canonical_shard=shard,
                    clip_uid=clip_uid,
                    cycle_index=cycle_index,
                    attempt_index=job.attempt_index,
                    seed=seed,
                    candidate_sha256=str(result.payload[PAYLOAD_CANDIDATE_SHA256]),
                    generate_job=job,
                ),
            )
        counters["candidates_failed"] += 1
        return self._advance(
            job,
            storage=storage,
            state=state,
            cycle_index=cycle_index,
            accepted=False,
        )

    def _finalize_judge(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        cycle_index = _cycle_index(job)
        counters = self.stats[shard]
        storage = self._storage_for(job)
        state = self._current_state(storage, clip_uid)
        if state.status != "pending_remove":
            # Already published by an earlier invocation; replaying nothing is
            # the only idempotent outcome.
            return ()
        status = result.payload.get(PAYLOAD_ATTEMPT_STATUS)
        if status == "accepted":
            context = prepare_removal_attempt_context(
                self.config, storage, clip_uid, state
            )
            attempts = self.collect_current_cycle_attempts(
                shard=shard,
                clip_uid=clip_uid,
                base_attempt_index=job.attempt_index - cycle_index,
            )
            candidate_bytes = self._read_artifact(
                str(dict(job.dependency_digests)["candidate"]), CANDIDATE_ARTIFACT
            )
            _publish_ready(
                self.config,
                storage,
                clip_uid=clip_uid,
                original=state,
                inputs=context.inputs,
                candidate=self._candidate_image(candidate_bytes),
                candidate_bytes=candidate_bytes,
                candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
                seed=int(result.payload[PAYLOAD_SEED]),
                attempts=[*state.removal_attempts, *attempts],
                candidate_mode=context.candidate_mode,
            )
            counters["processed"] += 1
            counters["ready_removed"] += 1
            return ()
        if _debug_enabled(self.config):
            context = prepare_removal_attempt_context(
                self.config, storage, clip_uid, state
            )
            attempt = BackgroundRemovalAttempt.model_validate(
                result.payload[PAYLOAD_ATTEMPT]
            )
            _write_candidate_debug(
                storage,
                clip_uid=clip_uid,
                seed=attempt.seed,
                candidate=self._candidate_image(
                    self._read_artifact(
                        str(dict(job.dependency_digests)["candidate"]),
                        CANDIDATE_ARTIFACT,
                    )
                ),
                review=attempt.review,
                inputs=context.inputs,
            )
        if status == "rejected":
            counters["candidates_rejected"] += 1
        else:
            counters["candidates_failed"] += 1
        return self._advance(
            job,
            storage=storage,
            state=state,
            cycle_index=cycle_index,
            accepted=False,
        )

    def _advance(
        self,
        job: ModelJob,
        *,
        storage: RunStorage,
        state: BackgroundReferenceState,
        cycle_index: int,
        accepted: bool,
    ) -> Sequence[ModelJob]:
        """Walk to the next candidate, or publish this cycle's terminal state."""
        del accepted
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        counters = self.stats[shard]
        next_cycle = cycle_index + 1
        if next_cycle < len(tuple(self.config.remove.candidate_seeds)):
            context = prepare_removal_attempt_context(
                self.config, storage, clip_uid, state
            )
            return (
                self._generate_job(
                    shard,
                    clip_uid,
                    state,
                    context,
                    cycle_index=next_cycle,
                ),
            )
        attempts = self.collect_current_cycle_attempts(
            shard=shard,
            clip_uid=clip_uid,
            base_attempt_index=job.attempt_index - cycle_index,
        )
        if removal_attempts_are_all_rejected(attempts):
            _publish_rejected(
                storage,
                clip_uid=clip_uid,
                original=state,
                reason=REJECT_REASON_ALL,
                removal_backend=self.config.remove.backend,
                attempts=list(attempts),
            )
            counters["rejected"] += 1
        else:
            _publish_retryable(
                storage,
                clip_uid=clip_uid,
                original=state,
                attempts=[*state.removal_attempts, *attempts],
                reason=RETRY_REASON_INFRASTRUCTURE,
                removal_backend=self.config.remove.backend,
            )
            counters["retryable_pending"] += 1
        counters["processed"] += 1
        return ()

    def collect_current_cycle_attempts(
        self,
        *,
        shard: str,
        clip_uid: str,
        base_attempt_index: int,
    ) -> list[BackgroundRemovalAttempt]:
        """Rebuild this cycle's attempts from committed results only.

        There is no in-memory attempt database on purpose: a restart must be
        able to reconstruct exactly what legacy would have accumulated.
        """
        storage = self.storages[shard]
        state = self._current_state(storage, clip_uid)
        context = prepare_removal_attempt_context(
            self.config, storage, clip_uid, state
        )
        attempts: list[BackgroundRemovalAttempt] = []
        for cycle_index in range(len(tuple(self.config.remove.candidate_seeds))):
            attempt_index = base_attempt_index + cycle_index
            seed = self._peek_seed(shard, clip_uid, context, attempt_index, cycle_index)
            if seed is None:
                break
            generate_job = build_removal_generate_job(
                self.config,
                context,
                canonical_shard=shard,
                clip_uid=clip_uid,
                cycle_index=cycle_index,
                attempt_index=attempt_index,
                seed=seed,
            )
            generation = self._committed_result(generate_job)
            if generation is None:
                break
            if generation.payload.get(PAYLOAD_ATTEMPT_STATUS) == ATTEMPT_STATUS_FAILED:
                attempts.append(
                    BackgroundRemovalAttempt.model_validate(
                        generation.payload[PAYLOAD_ATTEMPT]
                    )
                )
                continue
            candidate_sha = str(generation.payload[PAYLOAD_CANDIDATE_SHA256])
            judge_job = build_removal_judge_job(
                self.config,
                context,
                canonical_shard=shard,
                clip_uid=clip_uid,
                cycle_index=cycle_index,
                attempt_index=attempt_index,
                seed=seed,
                candidate_sha256=candidate_sha,
                generate_job=generate_job,
            )
            judged = self._committed_result(judge_job)
            if judged is None:
                # The candidate exists but was never judged: incomplete attempt.
                break
            attempt = BackgroundRemovalAttempt.model_validate(
                judged.payload[PAYLOAD_ATTEMPT]
            )
            attempts.append(attempt)
            if attempt.status == "accepted":
                break
        return attempts

    def close(self) -> None:
        """Publish per-shard stage counts. Safe to call more than once."""
        for shard in sorted(self.stats):
            counters = self.stats[shard]
            if any(counters.values()):
                self.storages[shard].update_stage_counts("remove", dict(counters))

    # -- helpers ----------------------------------------------------------
    def _generate_job(
        self,
        shard: str,
        clip_uid: str,
        state: BackgroundReferenceState,
        context: RemovalAttemptContext,
        *,
        cycle_index: int,
    ) -> ModelJob:
        base = len(tuple(state.removal_attempts))
        attempt_index = base + cycle_index
        seed = self._seed_for(shard, clip_uid, context, attempt_index, cycle_index)
        return build_removal_generate_job(
            self.config,
            context,
            canonical_shard=shard,
            clip_uid=clip_uid,
            cycle_index=cycle_index,
            attempt_index=attempt_index,
            seed=seed,
        )

    def _seed_for(
        self,
        shard: str,
        clip_uid: str,
        context: RemovalAttemptContext,
        attempt_index: int,
        cycle_index: int,
    ) -> int:
        anchor = removal_seed_anchor_inputs(
            self.config,
            context,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
        )
        return load_or_create_boogu_seed(
            ledger_root=self.ledger.root,
            canonical_shard=shard,
            clip_uid=clip_uid,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
            anchor_digest=semantic_input_digest(anchor),
            allocator=self.seed_allocator,
        )

    def _peek_seed(
        self,
        shard: str,
        clip_uid: str,
        context: RemovalAttemptContext,
        attempt_index: int,
        cycle_index: int,
    ) -> int | None:
        anchor = removal_seed_anchor_inputs(
            self.config,
            context,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
        )
        return load_boogu_seed(
            ledger_root=self.ledger.root,
            canonical_shard=shard,
            clip_uid=clip_uid,
            attempt_index=attempt_index,
            cycle_index=cycle_index,
            anchor_digest=semantic_input_digest(anchor),
        )

    def _load_result(self, job_id: str) -> JobResult | None:
        """Load a committed result by job id alone.

        The fully rebuilt ModelJob is only available to the CPU side; a worker
        thread only knows the job id it depends on, so the ledger is queried
        through a job-id-only view.
        """
        phase_id = self.ledger.phase_for(_JobRef(job_id))
        if phase_id is None:
            return None
        return self.ledger.phase(phase_id).load_result(_JobRef(job_id))

    def _committed_result(self, job: Any) -> JobResult | None:
        state = self.ledger.classify(job)
        if not state.skippable:
            return None
        return self.ledger.load_committed_result(job)

    def _storage_for(self, job: ModelJob) -> RunStorage:
        storage = self.storages.get(job.canonical_shard)
        if storage is None:
            raise RemovalEpochError(
                f"no run storage for canonical shard {job.canonical_shard!r}"
            )
        return storage

    @staticmethod
    def _current_state(
        storage: RunStorage, clip_uid: str
    ) -> BackgroundReferenceState:
        state = storage.read_clip(clip_uid).references.background
        if state is None:
            raise RemovalEpochError(f"clip {clip_uid} has no background state")
        return state

    def _pending_state(
        self, storage: RunStorage, clip_uid: str
    ) -> BackgroundReferenceState:
        state = self._current_state(storage, clip_uid)
        if state.status != "pending_remove":
            raise RemovalEpochError(
                f"clip {clip_uid} is {state.status}, not pending_remove"
            )
        return state

    def _read_artifact(self, job_id: str, name: str) -> bytes:
        phase_id = self.ledger.phase_for(_JobRef(job_id))
        if phase_id is None:
            raise RemovalEpochError(
                f"no committed phase holds artifact {name} for {job_id}"
            )
        path = self.ledger.phase(phase_id).artifact_path(job_id, name)
        if not path.is_file():
            raise RemovalEpochError(f"missing artifact {name} for {job_id}")
        return path.read_bytes()

    @staticmethod
    def _candidate_image(payload: bytes) -> Image.Image:
        """Reopen a published RGB PNG.

        PIL's PNG round-trip is byte-stable, so the image reloaded from the
        ledger is the exact candidate the generation job published.
        """
        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            return opened.convert("RGB")


def _cycle_index(job: ModelJob) -> int:
    """Cycle index is authoritative for candidate2; attempt_index is not."""
    return int(dict(job.target)["cycle_index"])


# ---------------------------------------------------------------------------
# Production wiring
# ---------------------------------------------------------------------------


def build_removal_epoch_factories(
    config: V3Config,
    runner: RemovalEpochRunner,
    *,
    qwen_epoch_config: QwenEpochConfig,
    pool: WorkerPoolConfig,
    process_manager: Any,
    temporary_root: Path,
    allowed_server_root: Path,
    qwen_max_inflight: int = 8,
    log_root: Path | None = None,
) -> dict[str, Callable[[], tuple[Any, Any]]]:
    """Resource-epoch factories for the Boogu and Qwen epochs.

    The Qwen epoch is the managed TP1 x DP8 judge server. No image-edit remover
    is ever loaded here: generation belongs to the Boogu worker epoch.
    """
    require_boogu_backend(config)
    slot_count = len(pool.gpu_ids)

    def boogu_factory() -> tuple[Any, Any]:
        epoch = build_boogu_epoch(
            config=config,
            pool=pool,
            process_manager=process_manager,
            temporary_root=temporary_root,
            allowed_server_root=allowed_server_root,
            log_root=log_root,
        )
        return epoch, WorkerSlotExecutor(
            runner.run, slot_count=slot_count, resource=epoch
        )

    def qwen_factory() -> tuple[Any, Any]:
        epoch = QwenEpochResource(
            qwen_epoch_config,
            process_manager=process_manager,
            log_root=log_root,
        )
        return epoch, QwenConcurrentExecutor(
            runner.run,
            endpoint=qwen_epoch_config.base_url,
            max_inflight=qwen_max_inflight,
        )

    return {RESOURCE_BOOGU: boogu_factory, RESOURCE_QWEN: qwen_factory}


def prepare_shard_storage(
    config: V3Config,
    *,
    post_mask_root: Path,
    entity_mask_root: Path,
    shard: str,
    git_commit: str,
) -> RunStorage:
    """Initialize and hydrate one canonical shard under the production lock.

    Hydration is reused from the existing Post-Mask production flow, never
    reimplemented: a fresh run root must produce real Stage2 clips instead of
    silently planning zero removal jobs.
    """
    shard_path = Path(entity_mask_root) / "parts" / f"{shard}.jsonl"
    paths = ShardPaths.for_shard(Path(post_mask_root), shard_path)
    with file_lock(paths.state_root / "shard.lock", blocking=False) as held:
        if not held:
            raise RemovalEpochError(f"canonical shard {shard} is locked elsewhere")
        storage = initialize_shard(config, paths, git_commit=git_commit)
        hydrate_shard(
            storage,
            entity_mask_root=Path(entity_mask_root),
            shard_path=shard_path,
            paths=paths,
        )
    return storage


def _current_git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_removal_epoch_runner(
    config: V3Config,
    *,
    post_mask_root: Path,
    entity_mask_root: Path,
    process_manager: Any,
    temporary_root: Path,
    allowed_server_root: Path,
    qwen_epoch_config: QwenEpochConfig,
    pool: WorkerPoolConfig | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    qwen_max_inflight: int = 8,
    log_root: Path | None = None,
    repo_root: Path | None = None,
) -> Callable[[Any, GroupLedger, Any], dict[str, Any]]:
    """Build the ``--job-runner`` callable for one removal resource epoch.

    The returned callable hydrates each canonical shard, seeds the unconditional
    candidate jobs and drains them through the resource-epoch scheduler. It
    never touches a prompt, a threshold or an accept/reject rule.

    It also never reports a completed *group*: the downstream DAG is not wired,
    so ``completed`` stays False even when the remove stage finished.
    """
    require_boogu_backend(config)
    worker_pool = pool or WorkerPoolConfig()

    def runner(group: Any, ledger: GroupLedger, emit: Any) -> dict[str, Any]:
        storages: dict[str, RunStorage] = {}
        git_commit = _current_git_commit(repo_root or Path.cwd())
        for shard in sorted(group.canonical_shards):
            storages[shard] = prepare_shard_storage(
                config,
                post_mask_root=post_mask_root,
                entity_mask_root=entity_mask_root,
                shard=shard,
                git_commit=git_commit,
            )
        removal = RemovalEpochRunner(config, storages, ledger, emit=emit)
        seed_jobs = removal.seed_jobs()
        emit(
            "post_mask_removal_epoch_planned",
            group_id=getattr(group, "group_id", ""),
            seeded_jobs=len(seed_jobs),
        )
        manager = ResourceEpochManager(
            factories=build_removal_epoch_factories(
                config,
                removal,
                qwen_epoch_config=qwen_epoch_config,
                pool=worker_pool,
                process_manager=process_manager,
                temporary_root=temporary_root,
                allowed_server_root=allowed_server_root,
                qwen_max_inflight=qwen_max_inflight,
                log_root=log_root,
            )
        )
        scheduler = ResourceEpochScheduler(
            ledger=ledger,
            finalize=removal.finalize,
            resource_manager=manager,
            window_size=window_size,
        )
        try:
            outcome = scheduler.run(seed_jobs)
        finally:
            removal.close()
        remove_completed = bool(outcome.get("completed"))
        summary = {
            shard: {
                "ready_removed": counters["ready_removed"],
                "rejected": counters["rejected"],
                "retryable_pending": counters["retryable_pending"],
                "failed": counters["failed"],
                "candidates_generated": counters["candidates_generated"],
            }
            for shard, counters in sorted(removal.stats.items())
        }
        if remove_completed:
            emit(
                "post_mask_removal_epoch_completed",
                group_id=getattr(group, "group_id", ""),
                shards=summary,
            )
        emit(
            "post_mask_removal_epoch_finished",
            group_id=getattr(group, "group_id", ""),
            remove_completed=remove_completed,
            unresolved=len(outcome.get("unresolved_job_ids", ())),
            shards=summary,
        )
        return {
            **outcome,
            "remove_completed": remove_completed,
            "completed": False,
            "reason": (
                "downstream resource-epoch phases are not wired"
                if remove_completed
                else "background removal incomplete"
            ),
        }

    return runner


DEFAULT_ENTITY_MASK_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_mask"
)
DEFAULT_POST_MASK_TAG = "post-mask-v1"


def build_qwen_epoch_config(
    config: V3Config,
    *,
    model_path: Path | None = None,
    allowed_local_media_path: Path | None = None,
) -> QwenEpochConfig:
    """Managed judge server config, using the foundation defaults.

    The served model name is the judge request model, so ``/v1/models`` and the
    OpenAI request agree on one identity.
    """
    import os

    import r2v_data_v2.v3.config as config_module

    service = config.qwen.background_remove_judge
    if service is None:
        raise RemovalEpochError("remove is enabled without a background remove judge")
    resolved_model = Path(
        os.environ.get("POST_MASK_QWEN_MODEL_PATH") or model_path or DEFAULT_QWEN_JUDGE_MODEL
    )
    return QwenEpochConfig(
        model_path=resolved_model,
        served_model_name=str(service.model),
        allowed_local_media_path=allowed_local_media_path
        or config_module.ALLOWED_DATASET_ROOT,
    )


def run_removal_epoch(
    group: Any, ledger: GroupLedger, emit: Any
) -> dict[str, Any]:
    """``--job-runner`` entrypoint: wire the removal epoch from launcher env.

    The launcher calls ``runner(group, ledger, emit)`` with no campaign
    arguments, so the roots come from the same environment the shell wrapper
    already uses.

        POST_MASK_JOB_RUNNER=r2v_data_v2.v3.post_mask_epoch_removal:run_removal_epoch
    """
    import os

    import r2v_data_v2.v3.config as config_module
    from r2v_data_v2.v3.config import load_config
    from r2v_data_v2.v3.post_mask_epoch_resources import (
        SubprocessEpochProcessManager,
    )

    base_config = os.environ.get("POST_MASK_BASE_CONFIG", "").strip()
    if not base_config:
        raise RemovalEpochError(
            "run_removal_epoch needs POST_MASK_BASE_CONFIG in the environment"
        )
    config = load_config(Path(base_config))
    require_boogu_backend(config)
    entity_mask_root = Path(
        os.environ.get("POST_MASK_ENTITY_MASK_ROOT") or DEFAULT_ENTITY_MASK_ROOT
    )
    tag = os.environ.get("POST_MASK_TAG") or DEFAULT_POST_MASK_TAG
    post_mask_root = Path(
        os.environ.get("POST_MASK_ROOT")
        or config_module.ALLOWED_WRITABLE_ROOT
        / "r2v_v3_post_mask"
        / "jea_motion_v1"
        / tag
    )
    temporary_root = Path(
        os.environ.get("POST_MASK_EPOCH_TEMP_ROOT")
        or post_mask_root / "tmp" / "resource_epoch"
    )
    repo_root = Path(os.environ.get("POST_MASK_REPO") or Path.cwd())
    runner = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=SubprocessEpochProcessManager(),
        temporary_root=temporary_root,
        allowed_server_root=config_module.ALLOWED_WRITABLE_ROOT,
        qwen_epoch_config=build_qwen_epoch_config(config),
        log_root=temporary_root / "logs",
        repo_root=repo_root,
    )
    return runner(group, ledger, emit)
