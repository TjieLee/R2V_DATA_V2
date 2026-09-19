"""Resource-epoch adapter for the frozen background-removal semantics.

This module is the *only* place where ``resource_epoch_v3`` meets the frozen
``remove.py`` semantics, and it reuses that semantics rather than re-describing
it. Every prompt, candidate-preparation rule, judge call, attempt record,
runtime measurement and publication helper is the shared implementation in
:mod:`r2v_data_v2.v3.remove`; nothing here re-implements or relaxes one.

The split is deliberately three-sided:

    CPU seeding   decide *which* model jobs exist (one per candidate seed)
    model job     one GPU call: either one generation or one judge call
    CPU finalize  decide what the committed outcome *means* for the clip

Only the middle step runs inside a resource epoch. That is what makes a
conditional second candidate structurally impossible to speculate: candidate 2
is not a job until the judge of candidate 1 has committed a rejection.

Job graph for one clip with ``candidate_seeds = (s0, s1)``:

    generate(s0) -> judge(s0) -> accepted  -> publish ready_removed
                             \\-> rejected  -> generate(s1) -> judge(s1) -> ...

    all candidates rejected -> publish rejected

A generation or judge failure is *retryable*, never terminal: the job stays
unresolved, the group is reported incomplete and the next launch retries it.
Restart is the retry; there is no in-launch re-request of a model call.

Two deliberate semantic differences from the legacy serial loop:

* the Boogu seed is derived deterministically from ``(clip_uid, candidate
  index, configured seed)`` instead of being drawn at random, because a random
  seed would make job identity -- and therefore resume -- impossible;
* per-attempt ``runtime_seconds`` measures generation active seconds plus judge
  active seconds, exactly like the shared helpers. The legacy loop measured one
  wall interval spanning both plus candidate preparation; queue wait,
  resource-switch and epoch wait were never included and still are not.

Artifacts live in the epoch ledger (``artifacts/<job-id>/candidate.png``),
never in the production run tree, so production cleanup can never invalidate a
receipt's external artifact reference.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.v3.background import validate_background_reference
from r2v_data_v2.v3.boogu_remove_backend import BooguBackgroundRemovalBackend
from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND, V3Config
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    OUTCOME_TERMINAL_REJECT,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_resources import (
    EpochResourceError,
    ResourceEpochManager,
    WorkerEpochResource,
    WorkerPoolConfig,
    WorkerSlotExecutor,
    build_boogu_epoch,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import (
    DEFAULT_WINDOW_SIZE,
    ResourceEpochScheduler,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.post_mask_production import ShardPaths, prepare_shard_config
from r2v_data_v2.v3.removal_judge import QwenBackgroundRemovalJudge
from r2v_data_v2.v3.remove import (
    CANDIDATE_READY,
    RemovalAttemptContext,
    _debug_enabled,
    _exception_reason,
    _publish_ready,
    _publish_rejected,
    _write_candidate_debug,
    generate_removal_candidate,
    prepare_removal_attempt_context,
    removal_attempts_are_all_rejected,
    review_removal_candidate,
)
from r2v_data_v2.v3.schemas import BackgroundReferenceState, BackgroundRemovalAttempt
from r2v_data_v2.v3.storage import RunStorage

REMOVAL_GENERATE_JOB = "background_removal_generate"
#: Conditional later candidate. Diagnostics count it separately on purpose: it
#: only ever exists because an earlier judge rejected its sibling.
REMOVAL_CANDIDATE2_JOB = "background_removal_candidate2"
REMOVAL_JUDGE_JOB = "background_removal_judge"

#: The judge is always the Qwen service, whichever backend generates.
REMOVAL_JUDGE_RESOURCE = RESOURCE_QWEN

CANDIDATE_ARTIFACT = "candidate.png"

PAYLOAD_SEED = "seed"
PAYLOAD_CANDIDATE_INDEX = "candidate_index"
PAYLOAD_CANDIDATE_SHA256 = "candidate_sha256"
PAYLOAD_GENERATION_SECONDS = "generation_seconds"
PAYLOAD_JUDGE_SECONDS = "judge_seconds"
PAYLOAD_GENERATED = "generated"
PAYLOAD_VERDICT = "verdict"
PAYLOAD_ATTEMPT = "attempt"
PAYLOAD_REASON = "reason"

#: Statuses that are never re-entered by the remove stage.
_TERMINAL_STATUSES = frozenset({"none", "clean_raw", "rejected"})

_REJECTED_ALL = "all_removal_candidates_rejected"
_REJECTED_MASK_AREA = "generation_mask_too_large"

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


def new_counters() -> dict[str, int]:
    return {name: 0 for name in _COUNTER_NAMES}


class RemovalEpochError(RuntimeError):
    """The removal adapter could not run a job safely; fail closed."""


def removal_generation_resource(config: V3Config) -> str:
    """Which resource epoch owns one background-removal generation."""
    if config.remove.backend == BOOGU_REMOVE_BACKEND:
        return RESOURCE_BOOGU
    return RESOURCE_QWEN


def removal_seed(
    config: V3Config,
    clip_uid: str,
    candidate_index: int,
) -> int:
    """Deterministic seed for one candidate of one clip.

    The legacy loop draws a fresh random seed for every Boogu request. A random
    seed cannot be part of a job identity, so the epoch derives one from the
    clip, the candidate index and the configured seed instead. The seed still
    differs per clip and per candidate, and the configured (non-Boogu) seeds
    are used unchanged.
    """
    seeds = tuple(config.remove.candidate_seeds)
    if not seeds:
        raise RemovalEpochError("remove.candidate_seeds is empty")
    if not 0 <= candidate_index < len(seeds):
        raise RemovalEpochError(
            f"candidate index {candidate_index} is outside remove.candidate_seeds"
        )
    configured = int(seeds[candidate_index])
    if config.remove.backend != BOOGU_REMOVE_BACKEND:
        return configured
    digest = hashlib.sha256(
        f"{clip_uid}|{candidate_index}|{configured}".encode()
    ).hexdigest()
    return int(digest[:8], 16) % (2**31)


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
) -> dict[str, Any]:
    """Everything that changes *what* one generation asks a model to do."""
    inputs = context.inputs
    payload: dict[str, Any] = {
        "kind": "generation",
        "backend": config.remove.backend,
        "candidate_mode": context.candidate_mode,
        "prompt": context.prompt,
        "seed": seed,
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
    }
    if config.remove.backend == BOOGU_REMOVE_BACKEND:
        payload["target_area"] = config.reference_edit.target_area
        payload["alignment"] = config.reference_edit.alignment
    else:
        payload["inference_profile"] = config.remove.inference_profile
        payload["num_inference_steps"] = config.remove.num_inference_steps
        payload["true_cfg_scale"] = config.remove.true_cfg_scale
        payload["guidance_scale"] = config.remove.guidance_scale
        payload["negative_prompt"] = config.remove.negative_prompt
    return payload


def judge_semantic_inputs(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    seed: int,
) -> dict[str, Any]:
    """Everything that changes *what* one judge call asks a model to do.

    The candidate itself is bound through the job's dependency digest, never by
    embedding the generated image here.
    """
    service = config.qwen.background_remove_judge
    if service is None:
        raise RemovalEpochError("remove is enabled without a background remove judge")
    inputs = context.inputs
    return {
        "kind": "judge",
        "candidate_mode": context.candidate_mode,
        "seed": seed,
        "removal_phrases": list(inputs.removal_phrases),
        "background_phrase": inputs.background_phrase,
        "source_image": _image_digest(inputs.source_image),
        "source_mask": _mask_digest(inputs.source_mask),
        "generation_mask": _mask_digest(inputs.generation_mask),
        "judge_model": str(service.model),
    }


def build_removal_generate_job(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    canonical_shard: str,
    clip_uid: str,
    candidate_index: int,
) -> ModelJob:
    """One generation model call for one clip and one candidate seed."""
    seed = removal_seed(config, clip_uid, candidate_index)
    return ModelJob.create(
        job_type=(
            REMOVAL_GENERATE_JOB if candidate_index == 0 else REMOVAL_CANDIDATE2_JOB
        ),
        resource=removal_generation_resource(config),
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        semantic_inputs=generation_semantic_inputs(
            config, context, seed=seed
        ),
        model_identity=context.profile_model,
        target={"candidate": str(candidate_index)},
        attempt_index=candidate_index,
        seed=seed,
        label=f"remove-generate-{candidate_index}",
    )


def build_removal_judge_job(
    config: V3Config,
    context: RemovalAttemptContext,
    *,
    canonical_shard: str,
    clip_uid: str,
    candidate_index: int,
    generate_job: ModelJob,
) -> ModelJob:
    """One judge model call, unlocked by a committed generation."""
    service = config.qwen.background_remove_judge
    if service is None:
        raise RemovalEpochError("remove is enabled without a background remove judge")
    seed = removal_seed(config, clip_uid, candidate_index)
    return ModelJob.create(
        job_type=REMOVAL_JUDGE_JOB,
        resource=REMOVAL_JUDGE_RESOURCE,
        canonical_shard=canonical_shard,
        clip_uid=clip_uid,
        semantic_inputs=judge_semantic_inputs(config, context, seed=seed),
        model_identity=str(service.model),
        target={"candidate": str(candidate_index)},
        attempt_index=candidate_index,
        seed=seed,
        dependencies={"candidate": generate_job.job_id()},
        label=f"remove-judge-{candidate_index}",
    )


class _JobRef:
    """Job-id-only view, so the ledger can be queried without a full ModelJob."""

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


@dataclass(frozen=True)
class RemovalEpochHandles:
    """What one resource epoch hands to the removal semantics.

    Both handles are optional so a generation-only epoch (Boogu workers) and a
    judge-only epoch never have to construct the other. A Qwen epoch that owns
    both a local remover and the judge service supplies both, and the adapter
    picks the handle a job actually needs.
    """

    removal_backend: Any | None = None
    judge: Any | None = None

    def close(self) -> None:
        errors: list[str] = []
        for handle in (self.removal_backend, self.judge):
            close = getattr(handle, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not leak
                errors.append(str(exc))
        if errors:
            raise EpochResourceError(
                f"removal epoch handles did not close: {'; '.join(errors)}"
            )


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
    """Return a ``BackgroundRemovalJudge`` for one judge job."""
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


class RemovalEpochRunner:
    """CPU side of the removal epoch: seeding, finalizing and stage counts.

    The GPU side is :meth:`run`, called by the scheduler's executors with the
    live backend handle. Neither method may hold GPU, slot, rank or epoch
    state: every decision here is a function of the clip and of one committed
    result, which is what makes a finalizer safely replayable after a crash.
    """

    def __init__(
        self,
        config: V3Config,
        storages: Mapping[str, RunStorage],
        ledger: GroupLedger,
        *,
        emit: Callable[..., None] | None = None,
    ) -> None:
        self.config = config
        self.storages: dict[str, RunStorage] = dict(storages)
        self.ledger = ledger
        self.emit = emit or (lambda *args, **kwargs: None)
        self.stats: dict[str, dict[str, int]] = {
            shard: new_counters() for shard in self.storages
        }
        self._attempts: dict[tuple[str, str], list[BackgroundRemovalAttempt]] = {}

    # -- CPU seeding ------------------------------------------------------
    def seed_jobs(self) -> list[ModelJob]:
        """Plan the unconditional first candidate of every pending clip.

        Only candidate 0 is seeded. Later candidates exist solely because a
        judge rejected their sibling, so they can never be speculated here.
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
                        reason=_REJECTED_MASK_AREA,
                    )
                    counters["processed"] += 1
                    counters["rejected"] += 1
                    continue
                jobs.append(
                    build_removal_generate_job(
                        self.config,
                        context,
                        canonical_shard=shard,
                        clip_uid=clip_uid,
                        candidate_index=0,
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
        storage = self._storage_for(job)
        state = self._pending_state(storage, job.clip_uid)
        context = prepare_removal_attempt_context(
            self.config, storage, job.clip_uid, state
        )
        backend = resolve_removal_backend(handle, self.config)
        seed = job.seed if job.seed is not None else 0
        generation = generate_removal_candidate(
            context=context,
            backend=backend,
            seed=seed,
            retry_index=job.attempt_index,
        )
        payload: dict[str, Any] = {
            PAYLOAD_SEED: seed,
            PAYLOAD_CANDIDATE_INDEX: job.attempt_index,
            PAYLOAD_GENERATION_SECONDS: generation.generation_seconds,
            PAYLOAD_GENERATED: generation.generated,
        }
        if generation.status != CANDIDATE_READY:
            payload[PAYLOAD_REASON] = generation.reason
            return JobResult(
                outcome=OUTCOME_RETRYABLE_FAILED,
                payload=payload,
                detail=generation.reason or "generation failed",
            )
        payload[PAYLOAD_CANDIDATE_SHA256] = generation.candidate_sha256
        return JobResult(
            outcome=OUTCOME_COMPLETED,
            artifacts={CANDIDATE_ARTIFACT: generation.candidate_bytes},
            payload=payload,
        )

    def _run_judge(self, job: ModelJob, handle: Any) -> JobResult:
        storage = self._storage_for(job)
        state = self._pending_state(storage, job.clip_uid)
        context = prepare_removal_attempt_context(
            self.config, storage, job.clip_uid, state
        )
        generate_job_id = dict(job.dependency_digests).get("candidate")
        if not generate_job_id:
            raise RemovalEpochError(
                f"judge job {job.job_id()} has no candidate dependency"
            )
        candidate_bytes = self._read_artifact(generate_job_id, CANDIDATE_ARTIFACT)
        judge = resolve_removal_judge(handle, self.config)
        seed = job.seed if job.seed is not None else 0
        prior = self.ledger.load_committed_result(_JobRef(generate_job_id))
        generation_seconds = 0.0
        if prior is not None:
            generation_seconds = float(
                prior.payload.get(PAYLOAD_GENERATION_SECONDS, 0.0)
            )
        outcome = review_removal_candidate(
            context=context,
            candidate=self._candidate_image(candidate_bytes),
            candidate_bytes=candidate_bytes,
            candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
            judge=judge,
            seed=seed,
            generation_seconds=generation_seconds,
        )
        payload: dict[str, Any] = {
            PAYLOAD_SEED: seed,
            PAYLOAD_CANDIDATE_INDEX: job.attempt_index,
            PAYLOAD_GENERATION_SECONDS: generation_seconds,
            PAYLOAD_JUDGE_SECONDS: outcome.judge_seconds,
            PAYLOAD_VERDICT: outcome.attempt.review.verdict
            if outcome.attempt.review is not None
            else None,
            PAYLOAD_ATTEMPT: outcome.attempt.model_dump(mode="json"),
        }
        if outcome.status == "accepted":
            return JobResult(outcome=OUTCOME_COMPLETED, payload=payload)
        if outcome.status == "rejected":
            return JobResult(
                outcome=OUTCOME_TERMINAL_REJECT,
                payload=payload,
                detail=outcome.attempt.reason or "candidate rejected",
            )
        payload[PAYLOAD_REASON] = outcome.attempt.reason
        return JobResult(
            outcome=OUTCOME_RETRYABLE_FAILED,
            payload=payload,
            detail=outcome.attempt.reason or "judge failed",
        )

    # -- CPU finalizers ---------------------------------------------------
    def finalize(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        """Apply one committed outcome. Returns the jobs it unlocks."""
        if job.job_type == REMOVAL_JUDGE_JOB:
            return self._finalize_judge(job, result)
        return self._finalize_generate(job, result)

    def _finalize_generate(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        counters = self.stats[job.canonical_shard]
        if result.payload.get(PAYLOAD_GENERATED):
            counters["candidates_generated"] += 1
        storage = self._storage_for(job)
        state = self._current_state(storage, job.clip_uid)
        if state.status != "pending_remove":
            # Published by an earlier invocation that crashed after publishing.
            return ()
        context = prepare_removal_attempt_context(
            self.config, storage, job.clip_uid, state
        )
        return (
            build_removal_judge_job(
                self.config,
                context,
                canonical_shard=job.canonical_shard,
                clip_uid=job.clip_uid,
                candidate_index=int(job.attempt_index),
                generate_job=job,
            ),
        )

    def _finalize_judge(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        counters = self.stats[job.canonical_shard]
        storage = self._storage_for(job)
        state = self._current_state(storage, job.clip_uid)
        if state.status != "pending_remove":
            # Already published by an earlier invocation; replaying nothing is
            # the only idempotent outcome.
            return ()
        context = prepare_removal_attempt_context(
            self.config, storage, job.clip_uid, state
        )
        attempt = BackgroundRemovalAttempt.model_validate(
            result.payload[PAYLOAD_ATTEMPT]
        )
        generate_job_id = dict(job.dependency_digests).get("candidate")
        if not generate_job_id:
            raise RemovalEpochError(
                f"judge job {job.job_id()} has no candidate dependency"
            )
        key = (job.canonical_shard, job.clip_uid)
        attempts = self._attempts.setdefault(key, [])
        attempts.append(attempt)
        candidate_index = int(job.attempt_index)

        if attempt.status == "accepted":
            candidate_bytes = self._read_artifact(generate_job_id, CANDIDATE_ARTIFACT)
            _publish_ready(
                self.config,
                storage,
                clip_uid=job.clip_uid,
                original=state,
                inputs=context.inputs,
                candidate=self._candidate_image(candidate_bytes),
                candidate_bytes=candidate_bytes,
                candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
                seed=attempt.seed,
                attempts=[*state.removal_attempts, *attempts],
                candidate_mode=context.candidate_mode,
            )
            counters["processed"] += 1
            counters["ready_removed"] += 1
            return ()

        counters["candidates_rejected"] += 1
        if _debug_enabled(self.config):
            _write_candidate_debug(
                storage,
                clip_uid=job.clip_uid,
                seed=attempt.seed,
                candidate=self._candidate_image(
                    self._read_artifact(generate_job_id, CANDIDATE_ARTIFACT)
                ),
                review=attempt.review,
                inputs=context.inputs,
            )

        next_index = candidate_index + 1
        if next_index < len(tuple(self.config.remove.candidate_seeds)):
            return (
                build_removal_generate_job(
                    self.config,
                    context,
                    canonical_shard=job.canonical_shard,
                    clip_uid=job.clip_uid,
                    candidate_index=next_index,
                ),
            )
        if not removal_attempts_are_all_rejected(attempts):
            raise RemovalEpochError(
                f"clip {job.clip_uid} exhausted its candidates without a verdict"
            )
        _publish_rejected(
            storage,
            clip_uid=job.clip_uid,
            original=state,
            reason=_REJECTED_ALL,
            removal_backend=self.config.remove.backend,
            attempts=list(attempts),
        )
        counters["processed"] += 1
        counters["rejected"] += 1
        return ()

    def close(self) -> None:
        """Publish per-shard stage counts. Safe to call more than once."""
        for shard in sorted(self.stats):
            counters = self.stats[shard]
            if any(counters.values()):
                self.storages[shard].update_stage_counts("remove", dict(counters))

    # -- helpers ----------------------------------------------------------
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


def qwen_removal_worker_factory(
    config: V3Config,
    *,
    pool: WorkerPoolConfig,
    backend_builder: Callable[[int, int], Any] | None = None,
) -> Callable[[int, int], RemovalEpochHandles]:
    """Build one Qwen removal bundle per GPU slot.

    The remover is a local pipeline pinned to the physical GPU id; the judge is
    an HTTP client and therefore needs no GPU of its own. Both are handed to the
    adapter, which picks the handle each job needs.
    """

    def factory(slot: int, gpu_id: int) -> RemovalEpochHandles:
        if backend_builder is not None:
            backend = backend_builder(slot, gpu_id)
        else:
            from r2v_data_v2.v3.qwen_image_edit_backend import (
                QwenImageEditRemovalBackend,
            )

            backend = QwenImageEditRemovalBackend(
                replace(config.remove, device=f"cuda:{gpu_id}")
            )
        service = config.qwen.background_remove_judge
        judge = QwenBackgroundRemovalJudge(service) if service is not None else None
        return RemovalEpochHandles(removal_backend=backend, judge=judge)

    return factory


def build_removal_epoch_factories(
    config: V3Config,
    runner: RemovalEpochRunner,
    *,
    pool: WorkerPoolConfig,
    process_manager: Any,
    temporary_root: Path,
    allowed_server_root: Path,
    log_root: Path | None = None,
    qwen_backend_builder: Callable[[int, int], Any] | None = None,
) -> dict[str, Callable[[], tuple[Any, Any]]]:
    """Resource-epoch factories for the two resources removal can need."""
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
        epoch = WorkerEpochResource(
            RESOURCE_QWEN,
            pool,
            process_manager=process_manager,
            worker_factory=qwen_removal_worker_factory(
                config, pool=pool, backend_builder=qwen_backend_builder
            ),
            log_root=log_root,
        )
        return epoch, WorkerSlotExecutor(
            runner.run, slot_count=slot_count, resource=epoch
        )

    return {RESOURCE_BOOGU: boogu_factory, RESOURCE_QWEN: qwen_factory}


DEFAULT_ENTITY_MASK_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_mask"
)
DEFAULT_POST_MASK_TAG = "post-mask-v1"


def run_removal_epoch(
    group: Any, ledger: GroupLedger, emit: Any
) -> dict[str, Any]:
    """``--job-runner`` entrypoint: wire the removal epoch from launcher env.

    The launcher calls ``runner(group, ledger, emit)`` with no campaign
    arguments, so the roots come from the same environment the shell wrapper
    already uses.

        POST_MASK_JOB_RUNNER=r2v_data_v2.v3.post_mask_epoch_removal:run_removal_epoch

    Background removal is the only phase wired here; reference edit and subject
    attributes still need their own runner.
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
    runner = build_removal_epoch_runner(
        config,
        post_mask_root=post_mask_root,
        entity_mask_root=entity_mask_root,
        process_manager=SubprocessEpochProcessManager(),
        temporary_root=temporary_root,
        allowed_server_root=config_module.ALLOWED_WRITABLE_ROOT,
        log_root=temporary_root / "logs",
    )
    return runner(group, ledger, emit)


def build_removal_epoch_runner(
    config: V3Config,
    *,
    post_mask_root: Path,
    entity_mask_root: Path,
    process_manager: Any,
    temporary_root: Path,
    allowed_server_root: Path,
    pool: WorkerPoolConfig | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    log_root: Path | None = None,
    qwen_backend_builder: Callable[[int, int], Any] | None = None,
) -> Callable[[Any, GroupLedger, Any], dict[str, Any]]:
    """Build the ``--job-runner`` callable for one removal resource epoch.

    The returned callable owns per-shard run storage, seeds the unconditional
    candidate jobs, drains them through the resource-epoch scheduler and reports
    per-shard stage counts. It never touches a prompt, a threshold or an
    accept/reject rule: those live in the shared removal semantics.
    """
    worker_pool = pool or WorkerPoolConfig()

    def runner(group: Any, ledger: GroupLedger, emit: Any) -> dict[str, Any]:
        storages: dict[str, RunStorage] = {}
        for shard in sorted(group.canonical_shards):
            shard_path = Path(entity_mask_root) / "parts" / f"{shard}.jsonl"
            paths = ShardPaths.for_shard(Path(post_mask_root), shard_path)
            storages[shard] = RunStorage(prepare_shard_config(config, paths))

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
                pool=worker_pool,
                process_manager=process_manager,
                temporary_root=temporary_root,
                allowed_server_root=allowed_server_root,
                log_root=log_root,
                qwen_backend_builder=qwen_backend_builder,
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
        summary = {
            shard: {
                "ready_removed": counters["ready_removed"],
                "rejected": counters["rejected"],
                "failed": counters["failed"],
                "candidates_generated": counters["candidates_generated"],
            }
            for shard, counters in sorted(removal.stats.items())
        }
        emit(
            "post_mask_removal_epoch_finished",
            group_id=getattr(group, "group_id", ""),
            completed=bool(outcome.get("completed")),
            unresolved=len(outcome.get("unresolved_job_ids", ())),
            shards=summary,
        )
        return outcome

    return runner
