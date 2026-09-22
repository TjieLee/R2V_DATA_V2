"""Durable primary Pair resource-epoch jobs.

Scope: the primary Pair pass only -- the per-entity Qwen judge chain and the
primary background final guard. Cross-pair, the donor snapshot and the
whole-shard barrier are a later commit.

This module owns the *durable* boundary: ModelJob identity, committed results,
replayable CPU finalization and the conditional unlock of the next job. It owns
no Pair policy. Every semantic decision still goes through the frozen helpers in
:mod:`r2v_data_v2.v3.pair`:

    prepare_entity_reference / run_entity_reference_judge /
    finalize_entity_reference
    _BackgroundFinalGuardRuntime.prepare / run / finalize

and publication still goes through ``_publish_pair_result``, so PNG validation,
backup, atomic replacement, clip write, rollback and cleanup stay exactly as
legacy left them.

Composing this with the removal runner is a later commit: nothing here touches
the scheduler, the resource manager or the launcher.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pair import (
    EntityReferenceState,
    PreparedEntityReferenceDecision,
    _BackgroundFinalGuardRuntime,
    _build_same_parent_donor_index,
    _donors_for_target,
    _GuardFailClosed,
    _publish_pair_result,
    finalize_cross_pair_judge,
    finalize_entity_reference,
    prepare_cross_pair_attempt,
    prepare_cross_pair_target_evidence,
    prepare_entity_reference,
    run_cross_pair_judge,
    run_entity_reference_judge,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    JOB_SCHEMA_VERSION,
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    semantic_input_digest,
)
from r2v_data_v2.v3.post_mask_epoch_resources import resolve_cpu_workers
from r2v_data_v2.v3.post_mask_epoch_state import (
    GroupLedger,
    atomic_write_json,
)
from r2v_data_v2.v3.reference_judge import build_entity_reference_request_payload
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    PairingState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage

PAIR_ENTITY_JUDGE_JOB = "pair_entity_reference_judge"
PAIR_BACKGROUND_GUARD_JOB = "pair_background_final_guard"

PAIR_PRIMARY_PLAN_SCHEMA = "post_mask_epoch_pair_primary/1"

CALL_SITE_PRIMARY = "primary"

PAIR_PHASE = "pair"

CLIP_FRESH_TARGET = "fresh_pair_target"
CLIP_EXISTING_PAIRING = "existing_pairing"
CLIP_INELIGIBLE = "ineligible"

#: Bounds for the invocation-local, execution-only caches. Both hold decoded
#: masks and PIL images, so neither may grow with the size of a campaign; an
#: eviction or a cold cache is always safe because everything cached is
#: deterministically re-derivable from the frozen inputs and durable receipts.
_PRIMARY_INDEX_LIMIT = 32
_PRIMARY_CONTEXT_LIMIT = 32
_PREPARED_CACHE_LIMIT = 16


class PairEpochError(RuntimeError):
    """Raised when the Pair epoch cannot proceed without changing semantics."""


# ---------------------------------------------------------------------------
# Durable semantic files
# ---------------------------------------------------------------------------


def _semantic_root(ledger: GroupLedger) -> Path:
    return Path(ledger.root) / "semantic" / "pair"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create-or-validate. An existing frozen file is never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise PairEpochError(
                f"{path.name} already exists with different semantics; "
                "refusing to rewrite a frozen Pair plan"
            )
        return
    # The epoch durable-state contract: tmp -> flush/fsync -> atomic rename.
    atomic_write_json(path, dict(payload))


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _image_identity(image: Image.Image) -> dict[str, Any]:
    """Geometry plus pixel digest.

    Raw ``tobytes()`` alone is ambiguous: a uniformly coloured 2x2 and a 1x4
    image can share a byte sequence while the model receives different
    geometry.
    """
    return {
        "mode": image.mode,
        "size": list(image.size),
        "pixel_sha256": _sha256_bytes(image.tobytes()),
    }


# ---------------------------------------------------------------------------
# Semantic identity helpers
# ---------------------------------------------------------------------------


def _entity_semantics(entity: AnnotationEntity, index: int) -> dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "entity_index": index,
        "reference_type": str(entity.reference_type),
        "phrase": entity.phrase,
        "grounding_prompt": entity.grounding_prompt,
    }


def _candidate_semantics(
    storage: RunStorage, candidates: Sequence[Any]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        path = storage.root / candidate.image_path
        mask = np.ascontiguousarray(candidate.mask)
        items.append(
            {
                "candidate_id": candidate.candidate_id,
                "frame_slot": candidate.frame_slot,
                "source_frame_index": candidate.source_frame_index,
                "image_path": candidate.image_path,
                "image_sha256": (
                    _sha256_bytes(path.read_bytes()) if path.is_file() else ""
                ),
                "mask_sha256": _sha256_bytes(mask.tobytes()),
                "mask_shape": list(mask.shape),
            }
        )
    return items


def _pair_policy_semantics(config: V3Config) -> dict[str, Any]:
    pair = config.pair
    return {
        "crop_padding_ratio": pair.crop_padding_ratio,
        "repair_retries": pair.repair_retries,
        "reference_prefilter_mode": pair.reference_prefilter_mode,
        "background_final_guard_mode": pair.background_final_guard_mode,
        "reference_scope_allow_local": config.reference_scope.allow_local,
    }


def _model_identity(service: Any) -> str:
    return str(getattr(service, "model", "") or "")


# ---------------------------------------------------------------------------
# Judge handle resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedJudge:
    judge: Any
    owned: bool


def _resolve_qwen_judge(handle: Any, service: Any, factory: Any) -> _ResolvedJudge:
    """A shared injected judge is used as-is; an endpoint is owned and closed."""
    if handle is not None and not isinstance(handle, str):
        return _ResolvedJudge(handle, owned=False)
    if isinstance(handle, str):
        return _ResolvedJudge(
            factory(replace(service, base_url=handle)), owned=True
        )
    # No endpoint and no injected judge: this call built a client, so it owns it.
    return _ResolvedJudge(factory(service), owned=True)


def _entity_judge_factory(service: Any, config: V3Config) -> Any:
    from r2v_data_v2.v3.reference_judge import QwenEntityReferenceJudge

    return QwenEntityReferenceJudge(
        service,
        repair_retries=config.pair.repair_retries,
        crop_padding_ratio=config.pair.crop_padding_ratio,
    )


def _guard_judge_factory(service: Any) -> Any:
    from r2v_data_v2.v3.background_final_guard import QwenFinalBackgroundJudge

    return QwenFinalBackgroundJudge(service)


# ---------------------------------------------------------------------------
# Result shims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReviewAttemptShim:
    review: Any
    raw_response: str | None = None


def _attempt_from_result(result: JobResult) -> Any:
    from r2v_data_v2.v3.reference_judge import (
        EntityReferenceDecisionAttempt,
        RawEntityReferenceDecision,
    )

    return EntityReferenceDecisionAttempt(
        decision=RawEntityReferenceDecision.model_validate(result.payload["decision"]),
        raw_responses=tuple(result.payload.get("raw_responses", ())),
        repair_attempts=int(result.payload.get("repair_attempts", 0)),
    )


def _final_background_review(payload: Any) -> Any:
    from r2v_data_v2.v3.background_final_guard import FinalBackgroundReview

    return FinalBackgroundReview.model_validate(payload)


def _validate_frames(storage: RunStorage, clip_uid: str) -> Any:
    from r2v_data_v2.v3.frames import validate_sampled_frames

    return validate_sampled_frames(storage, clip_uid)


def _tokens_for_retained(retained: Sequence[str], entity_by_id: Any) -> dict[str, str]:
    from r2v_data_v2.v3.pair import _tokens_for_retained as legacy_tokens

    return legacy_tokens(list(retained), entity_by_id)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class _JobIdRef:
    """Minimal job reference: the GroupLedger index only needs ``job_id()``."""

    __slots__ = ("_job_id",)

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


class _MutableStats:
    """Attribute access over a plain counter dict while reconciling."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def __getattr__(self, name: str) -> int:
        try:
            return self._counts[name]
        except KeyError as exc:  # pragma: no cover - guards typos
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: int) -> None:
        if name == "_counts":
            object.__setattr__(self, name, value)
            return
        self._counts[name] = value


class PairEpochRunner:
    """Durable primary Pair jobs for one resource-epoch group.

    One clip holds at most one outstanding Qwen job: the entity chain inside a
    clip is a conditional chain, while different clips and different shards may
    be seeded in parallel.
    """

    def __init__(
        self,
        config: V3Config,
        storages: Mapping[str, RunStorage],
        ledger: GroupLedger,
        *,
        eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
        emit: Any = None,
        cpu_workers: int | None = None,
    ) -> None:
        config.validate()
        if not config.pair.enabled:
            raise PairEpochError("V3 pair stage is disabled")
        # Production Post-Mask runs with reference_edit enabled, so the legacy
        # completion fallbacks stay bypassed: the epoch Pair path is Qwen + CPU
        # only, never image-edit completion or SAM completion.
        if not config.reference_edit.enabled:
            raise PairEpochError(
                "Pair resource epoch requires reference_edit.enabled: the legacy "
                "completion fallbacks are not part of the epoch Pair path"
            )
        self.config = config
        # Execution-only CPU budget; never part of any identity or schema.
        self.cpu_workers = (
            resolve_cpu_workers(config) if cpu_workers is None else int(cpu_workers)
        )
        self.storages = dict(storages)
        self.ledger = ledger
        self.eligible = {
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        }
        self.emit = emit or (lambda *args, **kwargs: None)
        self.phase = ledger.phase(PAIR_PHASE)
        self.stats: dict[str, dict[str, int]] = {
            shard: self._empty_stats() for shard in self.storages
        }
        self._guard_counters: dict[str, dict[str, int]] = {
            shard: self._empty_stats() for shard in self.storages
        }
        #: Execution-only index: ``(shard, clip_uid)`` -> every primary entity
        #: judge job this clip needs, in annotation order. It exists so the
        #: finalizer can answer "does this clip still owe an entity receipt?"
        #: without re-running ``prepare_entity_reference``. It is never part of
        #: ModelJob identity, never persisted and never read by any semantic
        #: decision: a restart with an empty index simply rebuilds it on the
        #: next replay, and a clip missing from it falls back to that replay.
        self._primary_jobs: dict[tuple[str, str], tuple[ModelJob, ...]] = {}
        self._primary_jobs_lock = threading.Lock()
        #: Invocation-local caches. Both are execution-only: every entry is
        #: deterministically re-derivable from the frozen inputs and the durable
        #: receipts, so a cold cache, an eviction or no cache at all changes only
        #: how much CPU is spent, never what is decided or published.
        self._frozen_inputs: dict[tuple[str, str], tuple[Any, Any]] = {}
        self._prepared: dict[str, Any] = {}
        self._prepared_lock = threading.Lock()

    def _remember_primary_jobs(
        self,
        shard: str,
        clip_uid: str,
        jobs: Sequence[ModelJob],
    ) -> None:
        """Record a clip's primary entity jobs. Execution-only, bounded."""
        if not jobs:
            return
        with self._primary_jobs_lock:
            if len(self._primary_jobs) >= _PRIMARY_INDEX_LIMIT and (
                shard,
                clip_uid,
            ) not in self._primary_jobs:
                # Bounded: drop the oldest insertion. Losing an entry is always
                # safe - the caller then falls back to one replay.
                self._primary_jobs.pop(next(iter(self._primary_jobs)), None)
            self._primary_jobs[(shard, clip_uid)] = tuple(jobs)

    def _prepared_for_job(self, job: ModelJob) -> Any:
        """The prepared decision this job was planned with, if still cached.

        Seeding already prepared every entity; running the same job immediately
        after would prepare it a second time. The cache is keyed by job id and
        only ever shortcuts CPU work: the job is still rebuilt from the prepared
        decision and its input digest compared below, so a cached entry that no
        longer matches the planned semantic inputs fails closed exactly like the
        uncached path.
        """
        with self._prepared_lock:
            return self._prepared.get(job.job_id())

    def _remember_prepared(self, job: ModelJob, prepared: Any) -> None:
        if not isinstance(prepared, PreparedEntityReferenceDecision):
            return
        with self._prepared_lock:
            if len(self._prepared) >= _PREPARED_CACHE_LIMIT and (
                job.job_id() not in self._prepared
            ):
                # Bounded LRU: a prepared decision holds decoded masks and PIL
                # images, so it must never accumulate across a campaign.
                self._prepared.pop(next(iter(self._prepared)), None)
            self._prepared[job.job_id()] = prepared

    def _clip_primary_settled(self, shard: str, clip_uid: str) -> bool | None:
        """Whether every recorded primary entity job for a clip is committed.

        ``None`` means "no index entry", so the caller must do the replay to
        find out. A committed job here means its durable receipt exists, which
        is the only thing the publication barrier needs.
        """
        with self._primary_jobs_lock:
            jobs = self._primary_jobs.get((shard, clip_uid))
        if jobs is None:
            return None
        return all(self._committed(job) is not None for job in jobs)

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        from r2v_data_v2.v3.pair import PairStats

        return {field: 0 for field in PairStats.__dataclass_fields__}

    def _scratch(self, shard: str) -> dict[str, int]:
        """Counters for deterministic replay that must not touch stage stats.

        Seeding, status queries and receipt replay all re-run the deterministic
        prefix, so prefilter and repair counters would otherwise multiply.
        """
        return self._empty_stats()

    def _marker_path(self, key: str) -> Path:
        return _semantic_root(self.ledger) / "accounted" / f"{key}.json"

    def _marker_exists(self, key: str) -> bool:
        return self._marker_path(key).is_file()

    def _mark_once(self, key: str) -> bool:
        """Durable, idempotent 'account this exactly once' marker."""
        path = self._marker_path(key)
        if path.is_file():
            return False
        _write_json_once(path, {"key": key})
        return True

    def _storage_for(self, shard: str) -> RunStorage:
        storage = self.storages.get(shard)
        if storage is None:
            raise PairEpochError(f"no run storage for canonical shard {shard!r}")
        return storage

    def _eligible_for(self, shard: str) -> tuple[str, ...]:
        return self.eligible.get(shard, ())

    # -- durable primary plan -------------------------------------------

    def _plan_path(self, shard: str) -> Path:
        return _semantic_root(self.ledger) / "primary" / f"{shard}.json"

    def _clip_classification(self, storage: RunStorage, clip_uid: str) -> str:
        clip = storage.read_clip(clip_uid)
        if clip.pairing is not None:
            return CLIP_EXISTING_PAIRING
        if (
            clip.annotation is None
            or clip.annotation.status != "ready"
            or clip.coverage is None
            or not clip.coverage.passed
        ):
            return CLIP_INELIGIBLE
        return CLIP_FRESH_TARGET

    def _frozen_input_digest(self, storage: RunStorage, clip_uid: str) -> str:
        """Digest of every pre-Pair input the primary pass depends on.

        Pairing, Pair-produced entity states, instruction and reference_edit
        are outputs of this invocation and are deliberately NOT bound: a fresh
        target that has since published its pairing is legal, not a mismatch.
        """
        clip = storage.read_clip(clip_uid)
        try:
            frames = _validate_frames(storage, clip_uid)
            frames_payload = frames.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - ineligible inputs are frozen as such
            frames_payload = None
        try:
            masks_payload = storage.read_masks(clip_uid).model_dump(mode="json")
        except Exception:  # noqa: BLE001
            masks_payload = None
        projection = {
            "annotation": (
                clip.annotation.model_dump(mode="json")
                if clip.annotation is not None
                else None
            ),
            "coverage": (
                clip.coverage.model_dump(mode="json")
                if clip.coverage is not None
                else None
            ),
            "sampled_frames": frames_payload,
            "tracked_masks": masks_payload,
            "background": (
                clip.references.background.model_dump(mode="json")
                if clip.references.background is not None
                else None
            ),
        }
        return semantic_input_digest(projection)

    def _primary_plan(self, shard: str) -> dict[str, Any]:
        """Load the frozen plan, or create it once from the launch state.

        An existing plan is never re-classified and never rewritten: a clip
        marked ``fresh_pair_target`` stays fresh for the whole invocation even
        after this invocation published its pairing. Re-classifying on restart
        would turn it into ``existing_pairing`` and destroy the plan's whole
        purpose.
        """
        existing = _read_json(self._plan_path(shard))
        if existing is not None:
            payload = self._validate_primary_plan(shard, existing)
            storage = self._storage_for(shard)
            for clip_uid, entry in payload.get("clips", {}).items():
                if entry.get("classification") == CLIP_EXISTING_PAIRING:
                    # Legacy validates an existing pairing instead of
                    # regenerating it; a corrupt reference is a Pair failure.
                    self._validate_existing_pairing(shard, storage, clip_uid)
            return payload

        storage = self._storage_for(shard)
        eligible = self._eligible_for(shard)
        clips: dict[str, dict[str, Any]] = {}
        for clip_uid in eligible:
            classification = self._clip_classification(storage, clip_uid)
            clips[clip_uid] = {
                "classification": classification,
                "digest": self._frozen_input_digest(storage, clip_uid),
            }
            if classification == CLIP_EXISTING_PAIRING:
                # Legacy validates an existing pairing instead of regenerating
                # it; a corrupt reference is a Pair failure, not a silent skip.
                self._validate_existing_pairing(shard, storage, clip_uid)
        payload = {
            "schema": PAIR_PRIMARY_PLAN_SCHEMA,
            "canonical_shard": shard,
            "config_fingerprint": self.config.fingerprint(),
            "pair_policy": _pair_policy_semantics(self.config),
            "eligible_clip_uids": list(eligible),
            "clips": clips,
        }
        _write_json_once(self._plan_path(shard), payload)
        return payload

    def _validate_primary_plan(self, shard: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate an existing frozen plan. Read-only; never writes it back."""
        if payload.get("schema") != PAIR_PRIMARY_PLAN_SCHEMA:
            raise PairEpochError(f"unsupported primary plan schema for {shard!r}")
        expected = {
            "canonical_shard": shard,
            "config_fingerprint": self.config.fingerprint(),
            "pair_policy": _pair_policy_semantics(self.config),
            "eligible_clip_uids": list(self._eligible_for(shard)),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise PairEpochError(
                    f"frozen primary plan {key} drifted for {shard!r}"
                )
        # The stored per-clip input digests are the whole point of the
        # plan: re-derive them and fail closed on any pre-Pair drift.
        storage = self._storage_for(shard)
        for clip_uid, entry in payload.get("clips", {}).items():
            actual = self._frozen_input_digest(storage, clip_uid)
            if actual != entry.get("digest"):
                raise PairEpochError(
                    f"frozen primary input drifted for {clip_uid!r} in {shard!r}"
                )
        return dict(payload)

    def _existing_primary_plan_for_reconcile(self, shard: str) -> dict[str, Any]:
        """Load the frozen plan for reconciliation without ever creating it."""
        payload = _read_json(self._plan_path(shard))
        if payload is None:
            raise PairEpochError(
                f"reconcile_primary_stats needs a frozen primary plan for {shard!r}"
            )
        return self._validate_primary_plan(shard, payload)

    def _validate_existing_pairing(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> None:
        from r2v_data_v2.v3.pair import (
            _validate_existing_pairing as legacy_validate,
        )
        from r2v_data_v2.v3.pair import (
            _validate_pair_inputs as legacy_inputs,
        )

        # Validation itself is never gated by a marker: a clip that was valid
        # when the plan was created can lose or corrupt its reference PNG
        # later, and 3b must not inherit a donor that was never re-checked.
        # Only the failure diagnostic is de-duplicated, and the marker for it
        # is written after append_failure, so a crash between the two can
        # duplicate one diagnostic but can never drop one.
        clip = storage.read_clip(clip_uid)
        try:
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
            legacy_inputs(clip, frames, masks)
            legacy_validate(self.config, storage, clip, frames=frames, masks=masks)
        except Exception as exc:  # noqa: BLE001 - legacy failure semantics
            fingerprint = semantic_input_digest(
                {
                    "shard": shard,
                    "clip_uid": clip_uid,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            accounted = self._marker_exists(f"existing-invalid-{fingerprint}")
            if not accounted:
                storage.append_failure(
                    stage="pair", clip_uid=clip_uid, reason=str(exc), details={}
                )
                self.stats[shard]["failed"] += 1
                # Written after the diagnostic, never before it.
                self._mark_once(f"existing-invalid-{fingerprint}")

    # -- primary job identity --------------------------------------------

    def _entity_job(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: AnnotationEntity,
        index: int,
        prepared: PreparedEntityReferenceDecision,
    ) -> ModelJob:
        service = self.config.qwen.candidate_judge
        if service is None:
            raise PairEpochError("candidate judge is not configured")
        # The canonical request payload is the authority for which candidate
        # fields the judge actually sees, so a future change in
        # reference_judge.py cannot silently drop fields from job identity.
        semantic_inputs = {
            "clip_uid": clip_uid,
            "entity": _entity_semantics(entity, index),
            "candidates": _candidate_semantics(storage, prepared.candidates),
            "request": build_entity_reference_request_payload(
                entity, list(prepared.candidates)
            ),
            "policy": _pair_policy_semantics(self.config),
        }
        return ModelJob.create(
            job_type=PAIR_ENTITY_JUDGE_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs=semantic_inputs,
            model_identity=_model_identity(service),
            target={"entity_id": entity.entity_id, "entity_index": str(index)},
        )

    def _guard_job(
        self, shard: str, clip_uid: str, call_site: str, preparation: Any
    ) -> ModelJob:
        service = self.config.qwen.background_final_judge
        if service is None:
            raise PairEpochError("final background judge is not configured")
        semantic_inputs = {
            "clip_uid": clip_uid,
            "call_site": call_site,
            "background_status": preparation.background_status,
            "background_phrase": preparation.phrase,
            "background_grounding_prompt": preparation.grounding_prompt,
            "image": _image_identity(preparation.image),
        }
        return ModelJob.create(
            job_type=PAIR_BACKGROUND_GUARD_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs=semantic_inputs,
            model_identity=_model_identity(service),
            target={"call_site": call_site},
        )

    def _committed(self, job: ModelJob) -> JobResult | None:
        state = self.ledger.classify(job)
        if not state.skippable:
            return None
        return self.ledger.load_committed_result(job)

    # -- primary replay ---------------------------------------------------

    def _frozen_pair_inputs(self, storage: RunStorage, clip_uid: str) -> Any:
        """The clip's frames and masks, cached per invocation.

        Both are frozen by the time Pair runs: the epoch Pair pass only ever
        writes references, pairing and selected reference images, never frames
        or masks. The clip itself is deliberately *not* cached - publication
        rewrites clip.json, and the published clip must always be the current
        one read back from storage.
        """
        key = (str(storage.root), clip_uid)
        cached = self._frozen_inputs.get(key)
        if cached is not None:
            return cached
        try:
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
        except Exception:  # noqa: BLE001 - not eligible for Pair
            return None
        if len(self._frozen_inputs) >= _PRIMARY_CONTEXT_LIMIT:
            # Bounded: drop the oldest entry. A miss only costs a re-read.
            self._frozen_inputs.pop(next(iter(self._frozen_inputs)), None)
        self._frozen_inputs[key] = (frames, masks)
        return self._frozen_inputs[key]

    def _primary_context(self, storage: RunStorage, clip_uid: str) -> Any:
        from r2v_data_v2.v3.pair import _validate_pair_inputs

        frozen = self._frozen_pair_inputs(storage, clip_uid)
        if frozen is None:
            return None
        frames, masks = frozen
        clip = storage.read_clip(clip_uid)
        try:
            _validate_pair_inputs(clip, frames, masks)
        except Exception:  # noqa: BLE001 - not eligible for Pair
            return None
        return (clip, frames, masks)

    @staticmethod
    def _discard_temporary(
        temporary_images: Mapping[str, tuple[Path, Image.Image]]
    ) -> None:
        """Replay artifacts are not progress: only committed receipts are."""
        for temporary, _ in temporary_images.values():
            temporary.unlink(missing_ok=True)

    def _prepare_entities(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entities: Sequence[tuple[int, Any]],
        frames: Any,
        masks: Any,
        counters: dict[str, int],
    ) -> list[Any]:
        """Prepare every annotated entity, concurrently when the runtime allows.

        An entity's preparation depends only on the clip, its frames and masks,
        that entity and its own candidates, so independent entities can be
        prepared at the same time. ThreadPoolExecutor is used rather than a
        process pool because the work is PIL, NumPy, RLE and filesystem heavy
        inside C extensions, and a process pool would have to pickle every
        decoded image and mask across the boundary.

        Only the CPU is parallel here. Nothing a worker does touches durable
        state: no ledger commit, no receipt, no phase plan, no pending map and
        no publication - all of that happens afterwards, on this thread, in
        annotation order. Debug diagnostics stay safe because each entity writes
        to its own ``pair_debug_dir(clip_uid, entity_id)`` through an atomic
        replace.

        Results are returned in annotation order, never in completion order.
        """
        workers = int(getattr(self, "cpu_workers", 1) or 1)
        if workers <= 1 or len(entities) < 2:
            return [
                prepare_entity_reference(
                    self.config,
                    storage,
                    clip_uid=clip_uid,
                    entity=entity,
                    frames=frames,
                    masks=masks,
                    counters=counters,
                )
                for _index, entity in entities
            ]
        per_entity: list[dict[str, int]] = [self._empty_stats() for _ in entities]

        def prepare(position: int) -> Any:
            return prepare_entity_reference(
                self.config,
                storage,
                clip_uid=clip_uid,
                entity=entities[position][1],
                frames=frames,
                masks=masks,
                counters=per_entity[position],
            )

        with ThreadPoolExecutor(max_workers=min(workers, len(entities))) as pool:
            results = list(pool.map(prepare, range(len(entities))))
        # Merge in annotation order so the scratch counters stay deterministic.
        for collected in per_entity:
            for key, value in collected.items():
                counters[key] = counters.get(key, 0) + value
        return list(results)

    def _replay_primary_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        context: Any,
        *,
        stop_at_unresolved: bool,
    ) -> tuple[
        list[EntityReferenceState],
        dict[str, tuple[Path, Image.Image]],
        list[ModelJob],
    ]:
        """Rebuild a clip's primary prefix from committed results only.

        Deterministic entities are recomputed; entity judge outcomes are read
        back from their receipts. Nothing lives only in memory, so a crash
        before publication costs zero extra Qwen calls.

        CPU preparation covers *every* annotated entity before anything is
        handed back. A primary entity judge depends only on the clip, its
        frames and masks, that entity and its own candidates - never on another
        entity's Qwen outcome - so the whole clip's independent judges can be
        seeded at once instead of one entity at a time. The per-entity results
        keep annotation order, and the clip is still published as one unit.
        """
        clip, frames, masks = context
        counters = self._scratch(shard)
        entities = list(enumerate(clip.annotation.entities))
        prepared_by_entity = self._prepare_entities(
            shard, storage, clip_uid, entities, frames, masks, counters
        )
        entity_states: list[EntityReferenceState] = []
        temporary_images: dict[str, tuple[Path, Image.Image]] = {}
        pending: list[ModelJob] = []
        entity_jobs: list[ModelJob] = []
        for (index, entity), prepared in zip(entities, prepared_by_entity):
            if isinstance(prepared, EntityReferenceState):
                entity_states.append(prepared)
                continue
            job = self._entity_job(shard, storage, clip_uid, entity, index, prepared)
            entity_jobs.append(job)
            # Seeding has already prepared this entity, so the model call can
            # reuse it instead of preparing a second time.
            self._remember_prepared(job, prepared)
            result = self._committed(job)
            if result is None:
                if not stop_at_unresolved:
                    raise PairEpochError(
                        f"primary entity {entity.entity_id} has no committed result"
                    )
                pending.append(job)
                continue
            finalization = finalize_entity_reference(
                self.config,
                storage,
                prepared=prepared,
                attempt=_attempt_from_result(result),
            )
            entity_states.append(finalization.state)
            if finalization.temporary is not None:
                temporary_images[entity.entity_id] = finalization.temporary
        # Every entity that needs a judge is now known, so the clip's receipt
        # barrier can be answered without preparing anything again.
        self._remember_primary_jobs(shard, clip_uid, entity_jobs)
        if pending:
            # Rebuilt temporaries are not progress: only receipts are, and the
            # clip cannot be published until every entity has one.
            self._discard_temporary(temporary_images)
            return entity_states, {}, pending
        return entity_states, temporary_images, []

    # -- background guard -------------------------------------------------

    def _guard_runtime(self, shard: str, *, scratch: bool = False) -> Any:
        """A guard runtime.

        Every use that merely re-derives semantic inputs uses ``scratch``, so
        planning, digest verification and resume replay cannot move the stage
        counters. Only :meth:`_apply_guard_result` accounts for real.
        """
        counters = self._scratch(shard) if scratch else self._guard_counters[shard]
        return _BackgroundFinalGuardRuntime(
            self.config, self._storage_for(shard), counters, None
        )

    def _background_token(
        self, shard: str, clip_uid: str, frames: Any, call_site: str
    ) -> tuple[str | None, ModelJob | None]:
        """Return ``(token, pending_guard_job)`` for one guard call site.

        Preparation here never touches stage counters: the identity of a guard
        job has to be derivable on every replay, and the ``attempted`` counter
        is incremented exactly once per guard job by :meth:`_account_guard`.
        """
        from r2v_data_v2.v3.background import validate_background_reference
        from r2v_data_v2.v3.background_final_guard import load_final_background_image
        from r2v_data_v2.v3.pair import BackgroundFinalGuardPreparation

        storage = self._storage_for(shard)
        clip = storage.read_clip(clip_uid)
        background = clip.references.background
        if background is None or background.status not in {"clean_raw", "ready_removed"}:
            return None, None
        if self.config.pair.background_final_guard_mode == "off":
            # Legacy: mode off validates and binds without any Qwen call, and a
            # broken background propagates so the primary Pair fails.
            validate_background_reference(storage, clip_uid, background, frames=frames)
            return "<ref_bg_1>", None
        runtime = self._guard_runtime(shard, scratch=True)
        deterministic_key = f"guard_deterministic-{shard}-{call_site}-{clip_uid}"
        try:
            validate_background_reference(storage, clip_uid, background, frames=frames)
            annotation_background = (
                clip.annotation.background if clip.annotation is not None else None
            )
            if annotation_background is None:
                raise ValueError("ready background has no annotation semantics")
            image = load_final_background_image(
                storage, clip_uid=clip_uid, background=background
            )
        except Exception as exc:  # noqa: BLE001 - background-only fail closed
            raw_response = getattr(exc, "raw_response", None)
            runtime._fail_closed(
                clip=clip,
                background=background,
                raw_response=raw_response,
                error=exc,
            )
            # Accounting only, and only after the semantic side effect.
            if self._mark_once(deterministic_key):
                counters = self._guard_counters[shard]
                counters["background_final_guard_attempted"] += 1
                counters["background_final_guard_failed_closed"] += 1
            return None, None
        preparation = BackgroundFinalGuardPreparation(
            clip=clip,
            background=background,
            image=image,
            phrase=annotation_background.phrase,
            grounding_prompt=annotation_background.grounding_prompt,
            background_status=background.status,
        )
        job = self._guard_job(shard, clip_uid, call_site, preparation)
        self._account_guard_attempt(shard, job)
        result = self._committed(job)
        if result is None:
            return None, job
        return self._apply_guard_result(
            runtime, preparation, job, shard, result
        ), None

    def _account_guard_attempt(self, shard: str, job: ModelJob) -> None:
        """One ``attempted`` per guard job, across restarts and replays."""
        if self._mark_once(f"guard_attempted-{job.job_id()}"):
            self._guard_counters[shard]["background_final_guard_attempted"] += 1

    def _apply_guard_result(
        self, runtime: Any, preparation: Any, job: ModelJob, shard: str, result: JobResult
    ) -> str | None:
        """Apply one committed guard outcome. Accounting happens after it."""
        if str(result.payload.get("status", "")) == "failed_closed":
            runtime._fail_closed(
                clip=preparation.clip,
                background=preparation.background,
                raw_response=result.payload.get("raw_response"),
                error=RuntimeError(str(result.payload.get("error") or "guard failed")),
            )
            if self._mark_once(f"guard_applied-{job.job_id()}"):
                self._guard_counters[shard]["background_final_guard_failed_closed"] += 1
            return None
        token = runtime.finalize_background_final_guard(
            preparation,
            _ReviewAttemptShim(
                review=_final_background_review(result.payload.get("review")),
                raw_response=result.payload.get("raw_response"),
            ),
        )
        if self._mark_once(f"guard_applied-{job.job_id()}"):
            key = (
                "background_final_guard_accepted"
                if token is not None
                else "background_final_guard_rejected"
            )
            self._guard_counters[shard][key] += 1
        return token

    # -- primary publication ----------------------------------------------

    def _publish_primary(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        frames: Any,
        entity_states: Sequence[EntityReferenceState],
        temporary_images: Mapping[str, tuple[Path, Image.Image]],
    ) -> ModelJob | None:
        clip = storage.read_clip(clip_uid)
        retained = [state.entity_id for state in entity_states if state.status == "ready"]
        if not set(retained).intersection(clip.coverage.qualifying_entity_ids):
            pairing = PairingState(
                status="rejected", reason="no_qualifying_ready_reference"
            )
            token: str | None = None
        else:
            token, pending = self._background_token(
                shard, clip_uid, frames, CALL_SITE_PRIMARY
            )
            if pending is not None:
                self._discard_temporary(temporary_images)
                return pending
            pairing = PairingState(
                status="ready",
                retained_entity_ids=retained,
                tokens=_tokens_for_retained(
                    retained, {e.entity_id: e for e in clip.annotation.entities}
                ),
                background_token=token,
            )
        # The publication transaction itself is the authority, not a marker: a
        # marker written before it could outlive a failed publish and stop the
        # restart from ever publishing again.
        if clip.pairing is not None:
            self._verify_existing_publication(
                storage, clip, pairing, entity_states, temporary_images
            )
            self._discard_temporary(temporary_images)
        else:
            _publish_pair_result(
                self.config,
                storage,
                clip_uid=clip_uid,
                references=ReferencesState(
                    entities=list(entity_states),
                    background=clip.references.background,
                ),
                pairing=pairing,
                temporary_images=dict(temporary_images),
            )
        if self._mark_once(f"published-{shard}-{clip_uid}"):
            counters = self.stats[shard]
            ready_count = sum(state.status == "ready" for state in entity_states)
            counters["entities_ready"] += ready_count
            counters["entities_rejected"] += len(entity_states) - ready_count
            counters[pairing.status] += 1
            counters["backgrounds_bound"] += int(pairing.background_token is not None)
        return None

    @staticmethod
    def _verify_existing_publication(
        storage: RunStorage,
        clip: Any,
        pairing: PairingState,
        entity_states: Sequence[EntityReferenceState],
        temporary_images: Mapping[str, tuple[Path, Image.Image]],
    ) -> None:
        """Restart after a successful publish must reproduce it byte for byte."""
        if clip.pairing != pairing:
            raise PairEpochError(
                f"clip {clip.clip_uid} pairing drifted from the reconstructed "
                "primary outcome"
            )
        current = [state.model_dump(mode="json") for state in clip.references.entities]
        expected = [state.model_dump(mode="json") for state in entity_states]
        if current != expected:
            raise PairEpochError(
                f"clip {clip.clip_uid} references drifted from the reconstructed "
                "primary outcome"
            )
        for entity_id, (temporary, _) in temporary_images.items():
            published = storage.selected_entity_path(clip.clip_uid, entity_id)
            if not published.is_file():
                raise PairEpochError(
                    f"clip {clip.clip_uid} entity {entity_id} lost its published PNG"
                )
            if _sha256_bytes(published.read_bytes()) != _sha256_bytes(
                temporary.read_bytes()
            ):
                raise PairEpochError(
                    f"clip {clip.clip_uid} entity {entity_id} published PNG differs "
                    "from the reconstructed reference"
                )

    # -- seeding and primary status ---------------------------------------

    def seed_primary_jobs(self) -> list[ModelJob]:
        """Advance every fresh Pair target to its CPU fixed point.

        CPU work does not stop just because this is the seeding call: a
        deterministic terminal entity is recomputed, a clip whose entities are
        all deterministic is published straight away, and a clip whose primary
        pass is complete but whose guard needs Qwen seeds the guard job. Only
        a genuinely unresolved model job is handed back.
        """
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            plan = self._primary_plan(shard)
            storage = self._storage_for(shard)
            for clip_uid in plan["eligible_clip_uids"]:
                if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                    continue
                context = self._primary_context(storage, clip_uid)
                if context is None:
                    continue
                jobs.extend(
                    self._advance_primary_clip(shard, storage, clip_uid, context)
                )
        jobs.sort(key=lambda job: job.job_id())
        self.phase.write_plan(jobs)
        return jobs

    def _advance_primary_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        context: Any,
    ) -> list[ModelJob]:
        """CPU -> MODEL -> CPU until a fixed point is reached.

        Returns every independent model job the clip still needs - one per
        annotated entity that requires a judge - or an empty list when the
        clip's primary Pair is terminal. The clip is only published once all of
        them have receipts.
        """
        states, temporary, pending = self._replay_primary_clip(
            shard, storage, clip_uid, context, stop_at_unresolved=True
        )
        if pending:
            return pending
        guard = self._publish_primary(
            shard, storage, clip_uid, context[1], states, temporary
        )
        return [guard] if guard is not None else []

    # -- durable primary PairStats reconciliation (4c1) -------------------



    def _pair_inputs_valid(self, storage: RunStorage, clip_uid: str) -> bool:
        """Read-only legacy input validation. Never mutates, never appends."""
        from r2v_data_v2.v3.pair import _validate_pair_inputs as legacy_inputs

        clip = storage.read_clip(clip_uid)
        try:
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
            legacy_inputs(clip, frames, masks)
        except Exception:  # noqa: BLE001 - incomplete inputs are not eligible
            return False
        return True

    def _existing_pairing_valid(self, storage: RunStorage, clip_uid: str) -> bool:
        """Read-only legacy existing-pairing validation for reconciliation."""
        from r2v_data_v2.v3.pair import _validate_existing_pairing as legacy_validate

        clip = storage.read_clip(clip_uid)
        try:
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
            _ = frames, masks
            legacy_validate(self.config, storage, clip, frames=frames, masks=masks)
        except Exception:  # noqa: BLE001 - invalid existing pairing is a failure
            return False
        return True

    def _planned_jobs(self, shard: str) -> tuple[dict[str, Any], ...]:
        """Every planned Pair job of one shard, across ALL ledger phases.

        The real scheduler commits receipts under ``r000-qwen``-style phase
        ids, while ``pair`` only holds the seed/semantic plan, so scanning one
        phase is not the receipt authority. The same job may be planned in
        more than one phase; identical records are one job, a drifting record
        is durable-state corruption and fails closed.
        """
        seen: dict[str, dict[str, Any]] = {}
        for phase_id in self.ledger.phase_ids():
            phase = self.ledger.phase(phase_id)
            if not phase.plan_path.is_file():
                continue
            try:
                records = phase.read_plan()
            except Exception as exc:  # a broken plan is corruption, not empty
                raise PairEpochError(
                    f"invalid Pair phase plan in {phase_id}"
                ) from exc
            for record in records:
                if record.get("canonical_shard") != shard:
                    continue
                if record.get("job_type") not in PAIR_JOB_TYPES:
                    continue
                job_id = record.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    continue
                previous = seen.get(job_id)
                if previous is not None and previous != record:
                    raise PairEpochError(
                        f"pair job {job_id} has drifting plan records across phases"
                    )
                seen.setdefault(job_id, record)
        return tuple(seen.values())

    def _job_from_plan_record(self, record: Mapping[str, Any]) -> ModelJob:
        """Rebuild the ModelJob its immutable plan record describes.

        ``job_id``/``job_identity`` are re-derived and must match the record,
        so a tampered or miswritten plan can never reach ``classify``.
        """
        job = ModelJob(
            job_type=str(record["job_type"]),
            resource=str(record["resource"]),
            canonical_shard=str(record["canonical_shard"]),
            clip_uid=str(record["clip_uid"]),
            input_digest=str(record["input_digest"]),
            model_identity=str(record["model_identity"]),
            target=tuple(
                (str(key), str(value))
                for key, value in (record.get("target") or [])
            ),
            attempt_index=int(record.get("attempt_index", 0)),
            seed=record.get("seed"),
            dependency_digests=tuple(
                (str(key), str(value))
                for key, value in (record.get("dependency_digests") or [])
            ),
            schema_version=str(record.get("schema_version", JOB_SCHEMA_VERSION)),
        )
        if job.job_id() != str(record.get("job_id", "")) or job.identity() != str(
            record.get("job_identity", "")
        ):
            raise PairEpochError(
                f"pair plan record {record.get('job_id', '')} does not rebuild"
            )
        return job

    def _validated_committed_payload(
        self, record: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Committed result payload, receipt-verified. None when pending."""
        from r2v_data_v2.v3.post_mask_epoch_state import STATE_MISMATCH

        job = self._job_from_plan_record(record)
        state = self.ledger.classify(job)
        if state.state == STATE_MISMATCH:
            raise PairEpochError(
                f"pair job {job.job_id()} receipt mismatch: {state.detail or ''}"
            )
        if not state.skippable:
            return None
        result = self.ledger.load_committed_result(job)
        if result is None:
            raise PairEpochError(
                f"committed pair job {job.job_id()} has no replayable result"
            )
        return dict(result.payload)

    def _committed_payload(self, job_id: str) -> dict[str, Any] | None:
        """Legacy 4c1 lookup kept for the primary reconciliation helpers."""
        result = self.ledger.load_committed_result(_JobIdRef(job_id))
        if result is None:
            return None
        return dict(result.payload)

    def reconcile_primary_stats(self, shard: str) -> Any:
        """Primary PairStats rebuilt from durable state only (4c1).

        Authority is the frozen primary plan, durable committed receipts and
        published Pair state -- never the invocation-local ``self.stats``. A
        fresh runner with zero model calls must return identical stats, so this
        is pure observation: no receipt, no Pair state and no failure
        diagnostic is written.

        Cross and prefilter fields are still 0: they are 4c2. Completion fields
        stay 0 because the epoch Pair path requires reference_edit.enabled.
        """
        from r2v_data_v2.v3.pair import PairStats

        storage = self._storage_for(shard)
        # Read-only: reconciliation must never create the plan it reads.
        plan = self._existing_primary_plan_for_reconcile(shard)
        # PairStats is frozen, so reconciliation accumulates into a plain dict.
        counts: dict[str, int] = {field: 0 for field in PairStats.__dataclass_fields__}
        stats = _MutableStats(counts)
        for clip_uid in sorted(plan.get("clips", {})):
            classification = plan["clips"][clip_uid]["classification"]
            if classification == CLIP_INELIGIBLE:
                stats.skipped_not_ready += 1
                continue
            if classification == CLIP_EXISTING_PAIRING:
                if self._existing_pairing_valid(storage, clip_uid):
                    stats.skipped_existing += 1
                else:
                    stats.failed += 1
                continue
            if not self._pair_inputs_valid(storage, clip_uid):
                stats.skipped_not_ready += 1
                continue
            stats.processed += 1
            pairing = self._reconcile_primary_pairing(shard, storage, clip_uid)
            if pairing is None:
                continue
            status = str(pairing.get("status", ""))
            if status == "ready":
                stats.ready += 1
            elif status == "rejected":
                stats.rejected += 1
            if pairing.get("background_token") is not None:
                stats.backgrounds_bound += 1
            for state in self._reconcile_primary_entities(shard, storage, clip_uid):
                if str(state.get("status", "")) == "ready":
                    stats.entities_ready += 1
                else:
                    stats.entities_rejected += 1

        # Repairs: one per committed primary entity decision, never the raw sum.
        for record in self._planned_jobs(shard):
            if record.get("job_type") != PAIR_ENTITY_JUDGE_JOB:
                continue
            payload = self._committed_payload(str(record.get("job_id", "")))
            if payload is None:
                continue
            stats.repaired += int(int(payload.get("repair_attempts", 0)) > 0)

        # Primary background guard counters, rebuilt from durable markers.
        if self.config.pair.background_final_guard_mode != "off":
            for record in self._planned_jobs(shard):
                if record.get("job_type") != PAIR_BACKGROUND_GUARD_JOB:
                    continue
                if dict(record.get("target") or {}).get("call_site") != "primary":
                    continue
                job_id = str(record.get("job_id", ""))
                if not self._marker_exists(f"guard_attempted-{job_id}"):
                    continue
                stats.background_final_guard_attempted += 1
                if not self._marker_exists(f"guard_applied-{job_id}"):
                    continue
                status = str((self._committed_payload(job_id) or {}).get("status", ""))
                if status == "failed_closed":
                    stats.background_final_guard_failed_closed += 1
                elif status == "accepted":
                    stats.background_final_guard_accepted += 1
                elif status == "rejected":
                    stats.background_final_guard_rejected += 1
                else:
                    raise PairEpochError(
                        f"committed primary guard result has no durable status: {job_id}"
                    )
            for clip_uid in sorted(plan.get("clips", {})):
                key = f"guard_deterministic-{shard}-primary-{clip_uid}"
                if self._marker_exists(key):
                    stats.background_final_guard_attempted += 1
                    stats.background_final_guard_failed_closed += 1
        return PairStats(**counts)

    # -- full legacy-equivalent PairStats reconciliation (4c2) -------------

    def reconcile_stats(self, shard: str) -> Any:
        """Complete legacy-equivalent PairStats for a TERMINAL Pair stage.

        Authorities: the frozen primary plan, receipt-validated scheduler
        commits, the frozen cross baseline / cross terminal markers, the final
        published live Pair state and a read-only prefilter replay. The
        invocation-local ``self.stats`` and ``_guard_counters`` are never read.

        Refuses incomplete work instead of guessing: any planned Pair job that
        is still pending, or any frozen cross target without a durable cross
        terminal marker, raises ``PairEpochError``.
        """
        from r2v_data_v2.v3.pair import PairStats

        primary = self.reconcile_primary_stats(shard)
        counts = primary.to_dict()
        storage = self._storage_for(shard)
        plan = self._existing_primary_plan_for_reconcile(shard)
        records = self._planned_jobs(shard)

        # Refuse incomplete: every planned Pair job must be committed.
        for record in records:
            job = self._job_from_plan_record(record)
            state = self.ledger.classify(job)
            if state.state != "pending" and not state.skippable:
                raise PairEpochError(
                    f"pair job {job.job_id()} receipt mismatch: {state.detail or ''}"
                )
            if not state.skippable:
                raise PairEpochError(
                    f"pair job {job.job_id()} is not terminal; "
                    "reconcile_stats refuses an incomplete Pair stage"
                )

        # Read-only prefilter replay for fresh targets with valid inputs.
        if self.config.pair.reference_prefilter_mode == "conservative_v1":
            self._reconcile_prefilter(shard, storage, plan, counts)

        # Cross statistics, only when the donor snapshot was frozen.
        snapshot_payload = _read_json(self._snapshot_path(shard))
        if snapshot_payload is not None:
            self._reconcile_cross(shard, storage, snapshot_payload, records, counts)

        # Final live net state for every processed fresh target.
        counts["ready"] = 0
        counts["rejected"] = 0
        counts["entities_ready"] = 0
        counts["entities_rejected"] = 0
        counts["backgrounds_bound"] = 0
        for clip_uid in sorted(plan.get("clips", {})):
            if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                continue
            if not self._pair_inputs_valid(storage, clip_uid):
                continue
            clip = storage.read_clip(clip_uid)
            if clip.pairing is None:
                continue
            if clip.pairing.status == "ready":
                counts["ready"] += 1
            elif clip.pairing.status == "rejected":
                counts["rejected"] += 1
            if clip.pairing.background_token is not None:
                counts["backgrounds_bound"] += 1
            for state in clip.references.entities if clip.references else ():
                if state.status == "ready":
                    counts["entities_ready"] += 1
                else:
                    counts["entities_rejected"] += 1

        # Full guard counters = primary (already counted) + cross call site.
        if self.config.pair.background_final_guard_mode != "off":
            self._reconcile_guard_site(
                shard, storage, plan, records, counts, CALL_SITE_CROSS_PAIR
            )
        return PairStats(**counts)

    def _reconcile_prefilter(
        self,
        shard: str,
        storage: RunStorage,
        plan: Mapping[str, Any],
        counts: dict[str, int],
    ) -> None:
        """Replay the legacy primary prefilter counters. Never writes debug."""
        from r2v_data_v2.v3.pair import (
            _build_entity_reference_candidates,
            _load_source_images,
            _record_prefilter_stats,
            prefilter_entity_reference_candidates,
        )

        for clip_uid in sorted(plan.get("clips", {})):
            if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                continue
            if not self._pair_inputs_valid(storage, clip_uid):
                continue
            clip = storage.read_clip(clip_uid)
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
            assert clip.annotation is not None
            for entity in clip.annotation.entities:
                tracked = masks.entities[entity.entity_id]
                if tracked.status != "ready":
                    continue
                candidates, _tiny, _fragmented = _build_entity_reference_candidates(
                    self.config,
                    storage,
                    clip_uid=clip_uid,
                    entity=entity,
                    frames=frames,
                    masks=masks,
                )
                if not candidates:
                    continue
                source_images = _load_source_images(storage, candidates)
                try:
                    result = prefilter_entity_reference_candidates(
                        entity, candidates, source_images
                    )
                except Exception:  # noqa: BLE001 - legacy fail-open semantics
                    counts["prefilter_candidates_examined"] += len(candidates)
                    counts["prefilter_fail_open_entities"] += 1
                else:
                    _record_prefilter_stats(counts, result)

    def _reconcile_cross(
        self,
        shard: str,
        storage: RunStorage,
        snapshot_payload: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        counts: dict[str, int],
    ) -> None:
        """Cross statistics from receipt-validated commits and terminals."""
        cross_records = [
            record
            for record in records
            if record.get("job_type") == PAIR_CROSS_JUDGE_JOB
        ]
        # Every committed cross job is one legacy attempted call.
        for record in cross_records:
            payload = self._validated_committed_payload(record)
            assert payload is not None, "incomplete cross work was refused above"
            counts["cross_pair_attempted"] += 1
            if str(payload.get("status", "")) == "decision" and int(
                payload.get("repair_attempts", 0)
            ) > 0:
                counts["cross_pair_repaired"] += 1

        targets = list(snapshot_payload.get("cross_pair_target_clip_uids", ()))
        for clip_uid in targets:
            terminal = self._cross_terminal(shard, clip_uid)
            if terminal is None:
                raise PairEpochError(
                    f"cross target {clip_uid!r} has no durable cross terminal; "
                    "reconcile_stats refuses an incomplete Pair stage"
                )
            if str(terminal.get("status", "")) == "failed":
                counts["failed"] += 1
            if str(terminal.get("status", "")) != "completed" or str(
                terminal.get("reason_kind", "")
            ) != "published":
                continue
            # A published target: every accepted donor decision made one
            # ready cross reference, and the live entity must match the donor
            # the job carried.
            for record in cross_records:
                if record.get("clip_uid") != clip_uid:
                    continue
                payload = self._validated_committed_payload(record)
                assert payload is not None
                if str(payload.get("status", "")) != "decision":
                    continue
                decision = payload.get("decision") or {}
                if str(decision.get("verdict", "")) != "accept":
                    continue
                counts["cross_pair_ready"] += 1
                self._verify_cross_published_entity(
                    storage, clip_uid, record, decision
                )

    def _verify_cross_published_entity(
        self,
        storage: RunStorage,
        clip_uid: str,
        record: Mapping[str, Any],
        decision: Mapping[str, Any],
    ) -> None:
        target = dict(record.get("target") or [])
        entity_id = str(target.get("target_entity_id", ""))
        donor_clip_uid = str(target.get("donor_clip_uid", ""))
        donor_entity_id = str(target.get("donor_entity_id", ""))
        clip = storage.read_clip(clip_uid)
        for state in clip.references.entities if clip.references else ():
            if state.entity_id != entity_id:
                continue
            if state.status != "ready":
                raise PairEpochError(
                    f"cross-published entity {clip_uid}/{entity_id} is not ready"
                )
            if (
                state.source_clip_uid != donor_clip_uid
                or state.source_entity_id != donor_entity_id
            ):
                raise PairEpochError(
                    f"cross-published entity {clip_uid}/{entity_id} does not "
                    "match its committed donor"
                )
            return
        raise PairEpochError(
            f"cross-published entity {clip_uid}/{entity_id} is missing"
        )

    def _reconcile_guard_site(
        self,
        shard: str,
        storage: RunStorage,
        plan: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
        counts: dict[str, int],
        call_site: str,
    ) -> None:
        """Guard counters for one call site, from durable markers."""
        for record in records:
            if record.get("job_type") != PAIR_BACKGROUND_GUARD_JOB:
                continue
            if dict(record.get("target") or {}).get("call_site") != call_site:
                continue
            job_id = str(record.get("job_id", ""))
            if not self._marker_exists(f"guard_attempted-{job_id}"):
                continue
            counts["background_final_guard_attempted"] += 1
            if not self._marker_exists(f"guard_applied-{job_id}"):
                continue
            status = str(
                (self._validated_committed_payload(record) or {}).get("status", "")
            )
            if status == "failed_closed":
                counts["background_final_guard_failed_closed"] += 1
            elif status == "accepted":
                counts["background_final_guard_accepted"] += 1
            elif status == "rejected":
                counts["background_final_guard_rejected"] += 1
            else:
                raise PairEpochError(
                    f"committed {call_site} guard result has no durable "
                    f"status: {job_id}"
                )
        for clip_uid in sorted(plan.get("clips", {})):
            key = f"guard_deterministic-{shard}-{call_site}-{clip_uid}"
            if self._marker_exists(key):
                counts["background_final_guard_attempted"] += 1
                counts["background_final_guard_failed_closed"] += 1

    def _reconcile_primary_pairing(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> dict[str, Any] | None:
        """Primary pairing as it was *before* any cross upgrade."""
        baseline = _read_json(self._cross_baseline_path(shard, clip_uid))
        if baseline is not None:
            return dict(baseline.get("primary_pairing") or {}) or None
        clip = storage.read_clip(clip_uid)
        if clip.pairing is None:
            return None
        return clip.pairing.model_dump(mode="json")

    def _reconcile_primary_entities(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> tuple[dict[str, Any], ...]:
        """Primary entity reference states as they were before any cross pass."""
        baseline = _read_json(self._cross_baseline_path(shard, clip_uid))
        if baseline is not None:
            return tuple(baseline.get("primary_entity_references") or ())
        clip = storage.read_clip(clip_uid)
        return tuple(
            state.model_dump(mode="json")
            for state in (clip.references.entities if clip.references is not None else ())
        )

    def primary_unresolved_job_ids(self) -> tuple[str, ...]:
        """Job ids of primary work that is still outstanding.

        A later commit consumes this to decide which clips are settled before
        the donor snapshot is frozen.
        """
        unresolved: list[str] = []
        for shard in sorted(self.storages):
            plan = self._primary_plan(shard)
            storage = self._storage_for(shard)
            for clip_uid in plan["eligible_clip_uids"]:
                if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                    continue
                context = self._primary_context(storage, clip_uid)
                if context is None:
                    continue
                unresolved.extend(
                    job.job_id()
                    for job in self._advance_primary_clip(shard, storage, clip_uid, context)
                )
        return tuple(unresolved)

    # -- frozen donor snapshot --------------------------------------------

    def _clip_job_ids(self, shard: str, clip_uid: str) -> set[str]:
        """Job ids currently planned for one clip, for barrier bookkeeping."""
        return {
            str(record.get("job_id", ""))
            for record in self.phase.read_plan()
            if record.get("clip_uid") == clip_uid
            and record.get("canonical_shard") == shard
        }


    def _shard_view(self, shard: str, *, exclude: Sequence[str] = ()) -> Any:
        """Storage view limited to this shard's current eligible Pair view."""
        from r2v_data_v2.v3.post_mask_runtime import _ShardStorage

        blocked = set(exclude)
        storage = self._storage_for(shard)
        return _ShardStorage(
            storage, tuple(uid for uid in self._eligible_for(shard) if uid not in blocked)
        )

    def _unresolved_primary_clip_uids(
        self, shard: str, unresolved_job_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Fresh targets whose primary work has not settled."""
        plan = self._primary_plan(shard)
        storage = self._storage_for(shard)
        blocked: list[str] = []
        for clip_uid in plan["eligible_clip_uids"]:
            if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                continue
            context = self._primary_context(storage, clip_uid)
            if context is None:
                continue
            _, temporary, pending = self._replay_primary_clip(
                shard, storage, clip_uid, context, stop_at_unresolved=True
            )
            self._discard_temporary(temporary)
            known = self._clip_job_ids(shard, clip_uid)
            if pending and (
                not unresolved_job_ids or not known & set(unresolved_job_ids)
            ) or known & set(unresolved_job_ids):
                blocked.append(clip_uid)
        return tuple(sorted(blocked))

    def _cross_pair_targets(
        self, shard: str, *, blocked: Sequence[str]
    ) -> tuple[str, ...]:
        """Cross targets: fresh in the durable plan, published, not blocked.

        A clip that already had a pairing before this launch is
        ``existing_pairing`` and can only ever be a donor, never a target --
        even though a fresh target that has since published looks identical
        if you only look at clip.pairing.
        """
        plan = self._primary_plan(shard)
        storage = self._storage_for(shard)
        skip = set(blocked)
        targets: list[str] = []
        for clip_uid in plan["eligible_clip_uids"]:
            if plan["clips"][clip_uid]["classification"] != CLIP_FRESH_TARGET:
                continue
            if clip_uid in skip:
                continue
            if storage.read_clip(clip_uid).pairing is None:
                continue
            targets.append(clip_uid)
        return tuple(targets)

    # -- cross target primary baseline ------------------------------------
    #
    # Cross replay must start from the PRIMARY PairingState / ReferencesState
    # that existed when the donor snapshot was frozen, not from whatever the
    # live clip currently holds: after a successful cross publication the live
    # state already contains the cross upgrade, and replaying from it would
    # treat an upgraded e2 as a pre-existing full self reference and skip it.

    def _cross_baseline_path(self, shard: str, clip_uid: str) -> Path:
        return (
            _semantic_root(self.ledger)
            / "cross_baselines"
            / shard
            / f"{clip_uid}.json"
        )

    def _freeze_cross_baseline(self, shard: str, clip_uid: str) -> dict[str, Any]:
        storage = self._storage_for(shard)
        clip = storage.read_clip(clip_uid)
        payload = {
            "schema": PAIR_CROSS_BASELINE_SCHEMA,
            "canonical_shard": shard,
            "clip_uid": clip_uid,
            "primary_pairing": (
                clip.pairing.model_dump(mode="json") if clip.pairing is not None else None
            ),
            "primary_entity_references": [
                state.model_dump(mode="json")
                for state in (
                    clip.references.entities if clip.references is not None else ()
                )
            ],
        }
        _write_json_once(self._cross_baseline_path(shard, clip_uid), payload)
        return payload

    def cross_baseline(self, shard: str, clip_uid: str) -> dict[str, Any]:
        payload = _read_json(self._cross_baseline_path(shard, clip_uid))
        if payload is None:
            raise PairEpochError(
                f"no frozen cross baseline for {clip_uid!r} in {shard!r}"
            )
        if payload.get("schema") != PAIR_CROSS_BASELINE_SCHEMA:
            raise PairEpochError(f"unsupported cross baseline schema for {clip_uid!r}")
        if payload.get("canonical_shard") != shard or payload.get("clip_uid") != clip_uid:
            raise PairEpochError(f"cross baseline identity mismatch for {clip_uid!r}")
        return payload

    def _freeze_cross_baselines(self, shard: str, targets: Sequence[str]) -> None:
        for clip_uid in targets:
            self._freeze_cross_baseline(shard, clip_uid)

    def _snapshot_path(self, shard: str) -> Path:
        return _semantic_root(self.ledger) / "donor_snapshots" / f"{shard}.json"

    def _snapshot_identity(self, payload: Mapping[str, Any]) -> str:
        return semantic_input_digest(payload)

    def freeze_donor_snapshot(
        self, shard: str, *, unresolved_job_ids: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Freeze this shard's donor index once. Create-or-validate.

        The donor view excludes clips whose primary work has not settled, so
        a half-finished primary can never contribute a stale donor.
        """
        if self._snapshot_path(shard).is_file():
            # Already frozen: return it as-is. Donor membership is never
            # re-derived from live storage, so a full reference produced later
            # by cross-pair can neither expand nor invalidate the snapshot.
            return self.donor_snapshot(shard)

        blocked = self._unresolved_primary_clip_uids(shard, unresolved_job_ids)
        targets = self._cross_pair_targets(shard, blocked=blocked)
        view = self._shard_view(shard, exclude=blocked)
        live = _build_same_parent_donor_index(self.config, view)
        storage = self._storage_for(shard)
        groups = []
        for key in sorted(live, key=lambda item: (str(item[0]), str(item[1]))):
            parent, reference_type = key
            groups.append(
                {
                    "parent_video_id": parent,
                    "reference_type": str(reference_type),
                    "donors": [
                        dict(ordinal=ordinal, **_donor_projection(donor, storage))
                        for ordinal, donor in enumerate(live[key])
                    ],
                }
            )
        payload = {
            "schema": PAIR_DONOR_SNAPSHOT_SCHEMA,
            "canonical_shard": shard,
            "eligible_view": list(self._eligible_for(shard)),
            "primary_plan_digest": semantic_input_digest(
                self._primary_plan(shard)
            ),
            "blocked_primary_clip_uids": list(blocked),
            "cross_pair_target_clip_uids": list(targets),
            "groups": groups,
        }
        _write_json_once(self._snapshot_path(shard), payload)
        # Baseline the cross targets against their frozen primary state before
        # any cross-pair work can mutate it.
        self._freeze_cross_baselines(shard, targets)
        return payload

    def donor_snapshot(self, shard: str) -> dict[str, Any]:
        """Load the frozen snapshot. Never rebuilt, never expanded."""
        payload = _read_json(self._snapshot_path(shard))
        if payload is None:
            raise PairEpochError(f"no frozen donor snapshot for shard {shard!r}")
        if payload.get("schema") != PAIR_DONOR_SNAPSHOT_SCHEMA:
            raise PairEpochError(f"unsupported donor snapshot schema for {shard!r}")
        expected = {
            "canonical_shard": shard,
            "eligible_view": list(self._eligible_for(shard)),
            "primary_plan_digest": semantic_input_digest(self._primary_plan(shard)),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise PairEpochError(f"frozen donor snapshot {key} drifted for {shard!r}")
        return payload

    def frozen_donor_index(self, shard: str) -> dict[tuple[str, str], tuple[Any, ...]]:
        """Restore the frozen index for _donors_for_target."""
        return load_frozen_donor_index(self._storage_for(shard), self.donor_snapshot(shard))

    # -- cross terminal marker ---------------------------------------------

    def _cross_terminal_path(self, shard: str, clip_uid: str) -> Path:
        return (
            _semantic_root(self.ledger) / "cross_terminal" / shard / f"{clip_uid}.json"
        )

    def _cross_terminal_identity(self, shard: str, clip_uid: str) -> tuple[str, str]:
        return (
            self._snapshot_identity(self.donor_snapshot(shard)),
            semantic_input_digest(self.cross_baseline(shard, clip_uid)),
        )

    def _cross_terminal(self, shard: str, clip_uid: str) -> dict[str, Any] | None:
        payload = _read_json(self._cross_terminal_path(shard, clip_uid))
        if payload is None:
            return None
        if payload.get("schema") != PAIR_CROSS_TERMINAL_SCHEMA:
            raise PairEpochError(f"unsupported cross terminal schema for {clip_uid!r}")
        snapshot_identity, baseline_digest = self._cross_terminal_identity(shard, clip_uid)
        if payload.get("snapshot_identity") != snapshot_identity:
            raise PairEpochError(f"cross terminal snapshot drifted for {clip_uid!r}")
        if payload.get("baseline_digest") != baseline_digest:
            raise PairEpochError(f"cross terminal baseline drifted for {clip_uid!r}")
        return payload

    def _mark_cross_terminal(
        self,
        shard: str,
        clip_uid: str,
        *,
        status: str,
        reason_kind: str,
        reason: str,
    ) -> None:
        """Record that this target's fallback pass is terminal.

        Written only after the legacy side effects (debug, append_failure or
        successful publication) have run, so a crash before it can duplicate a
        diagnostic but can never drop one.
        """
        snapshot_identity, baseline_digest = self._cross_terminal_identity(shard, clip_uid)
        _write_json_once(
            self._cross_terminal_path(shard, clip_uid),
            {
                "schema": PAIR_CROSS_TERMINAL_SCHEMA,
                "status": status,
                "canonical_shard": shard,
                "clip_uid": clip_uid,
                "snapshot_identity": snapshot_identity,
                "baseline_digest": baseline_digest,
                "reason_kind": reason_kind,
                "reason": reason,
            },
        )

    # -- cross job identity -------------------------------------------------

    def _cross_job(
        self,
        shard: str,
        clip_uid: str,
        entity: AnnotationEntity,
        index: int,
        evidence: Any,
        donor: Any,
        rank: int,
        snapshot_identity: str,
        donor_image: Any,
    ) -> ModelJob:
        from r2v_data_v2.v3.cross_pair_judge import build_cross_pair_request_payload

        service = self.config.qwen.cross_pair_judge
        if service is None:
            raise PairEpochError("cross-pair judge is not configured")
        semantic_inputs = {
            "clip_uid": clip_uid,
            "entity": _entity_semantics(entity, index),
            "request": build_cross_pair_request_payload(
                target_clip_uid=clip_uid,
                target_entity=entity,
                target_evidence_mode=evidence.evidence_mode,
                donor_clip_uid=donor.clip.clip_uid,
                donor_entity=donor.entity,
            ),
            "evidence_mode": str(evidence.evidence_mode),
            "frame_slots": list(evidence.frame_slots),
            "context_image": _image_identity(evidence.context_image),
            "entity_crop": (
                _image_identity(evidence.entity_crop)
                if evidence.entity_crop is not None
                else None
            ),
            "donor": {
                "clip_uid": donor.clip.clip_uid,
                "entity_id": donor.entity.entity_id,
                "projection": donor.reference.model_dump(mode="json"),
                "image": _image_identity(donor_image),
            },
            "donor_ordinal": rank,
            "snapshot_identity": snapshot_identity,
            "policy": _pair_policy_semantics(self.config),
        }
        return ModelJob.create(
            job_type=PAIR_CROSS_JUDGE_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs=semantic_inputs,
            model_identity=_model_identity(service),
            target={
                "target_entity_id": entity.entity_id,
                "target_entity_index": str(index),
                "donor_clip_uid": donor.clip.clip_uid,
                "donor_entity_id": donor.entity.entity_id,
                "donor_ordinal": str(rank),
            },
            attempt_index=rank,
        )

    # -- cross CPU replay ---------------------------------------------------

    def _cross_context(self, shard: str, clip_uid: str) -> Any:
        """Frozen snapshot, donor index, baseline states and target inputs."""
        storage = self._storage_for(shard)
        snapshot = self.donor_snapshot(shard)
        baseline = self.cross_baseline(shard, clip_uid)
        target_clip = storage.read_clip(clip_uid)
        states = {
            item["entity_id"]: EntityReferenceState.model_validate(item)
            for item in baseline["primary_entity_references"]
        }
        frames = _validate_frames(storage, clip_uid)
        masks = storage.read_masks(clip_uid)
        return (storage, snapshot, baseline, target_clip, states, frames, masks)

    def _cross_position(
        self, shard: str, clip_uid: str, target: Mapping[str, str]
    ) -> tuple[Any, Any, Any, int, Any]:
        """Rebuild the exact entity/donor position a cross job points at."""
        from r2v_data_v2.v3.pair import build_entity_reference_candidates

        storage, _snapshot, _b, target_clip, _s, frames, masks = self._cross_context(
            shard, clip_uid
        )
        index = int(target["target_entity_index"])
        entity = target_clip.annotation.entities[index]
        donors = _donors_for_target(
            self.config,
            self.frozen_donor_index(shard),
            target_clip=target_clip,
            target_entity=entity,
        )
        rank = int(target["donor_ordinal"])
        donor = next(
            (
                item
                for item in donors
                if item.clip.clip_uid == target["donor_clip_uid"]
                and item.entity.entity_id == target["donor_entity_id"]
            ),
            None,
        )
        if donor is None:
            raise PairEpochError(
                f"frozen donor {target['donor_clip_uid']} is no longer in the target list"
            )
        candidates = build_entity_reference_candidates(
            self.config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            frames=frames,
            masks=masks,
        )
        evidence = prepare_cross_pair_target_evidence(
            self.config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            frames=frames,
            target_candidates=candidates,
        )
        prepared = prepare_cross_pair_attempt(evidence, donor)
        return prepared, evidence, donor, rank, target_clip

    def legacy_cross_in_progress(self) -> bool:
        """Whether this ledger already entered the frozen legacy Cross pass.

        The Post-Mask Resource Epoch composition does not own Cross Pair, so a
        fresh run never freezes a donor snapshot: Primary Pair is the final Pair
        state. A snapshot that already exists can therefore only come from an
        older launch that stopped inside legacy Cross, and that frozen work must
        not be silently reinterpreted as complete. Read-only.
        """
        return any(
            self._snapshot_path(shard).is_file() for shard in sorted(self.storages)
        )

    def freeze_cross_pair_after_primary_quiescence(
        self, *, unresolved_job_ids: Sequence[str] = ()
    ) -> Sequence[ModelJob]:
        """Whole-shard barrier: seed one chain head per frozen cross target.

        Called once the primary Qwen ready-set has quiesced. It never blocks
        the rest of the shard: a target whose primary is unresolved is simply
        absent from the frozen target list.
        """
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            snapshot = self.freeze_donor_snapshot(
                shard, unresolved_job_ids=unresolved_job_ids
            )
            for clip_uid in snapshot.get("cross_pair_target_clip_uids", ()):
                if self._cross_terminal(shard, clip_uid) is not None:
                    continue
                pending = self._advance_cross_pair_target(shard, clip_uid)
                if pending is not None:
                    jobs.append(pending)
        return tuple(jobs)

    def _advance_cross_pair_target(self, shard: str, clip_uid: str) -> ModelJob | None:
        """Replay one target's fallback from the frozen baseline.

        Returns the single chain-head job, or None when the fallback pass is
        terminal (accepted donors published, or nothing to do).
        """
        from r2v_data_v2.v3.pair import (
            _pairing_from_references,
            _publish_cross_pair_result,
            build_entity_reference_candidates,
        )

        if self._cross_terminal(shard, clip_uid) is not None:
            return None
        storage, snapshot, _b, target_clip, states_by_id, frames, masks = (
            self._cross_context(shard, clip_uid)
        )
        snapshot_identity = self._snapshot_identity(snapshot)
        index = self.frozen_donor_index(shard)
        accepted: dict[str, Any] = {}

        for entity_index, entity in enumerate(target_clip.annotation.entities):
            current = states_by_id.get(entity.entity_id)
            if (
                current is not None
                and current.status == "ready"
                and current.reference_scope == "full"
            ):
                continue
            try:
                candidates = build_entity_reference_candidates(
                    self.config,
                    storage,
                    clip_uid=clip_uid,
                    entity=entity,
                    frames=frames,
                    masks=masks,
                )
                donors = _donors_for_target(
                    self.config, index, target_clip=target_clip, target_entity=entity
                )
                if not donors:
                    # Frozen ordering: no donor means no evidence materialization.
                    continue
                evidence = prepare_cross_pair_target_evidence(
                    self.config,
                    storage,
                    clip_uid=clip_uid,
                    entity=entity,
                    frames=frames,
                    target_candidates=candidates,
                )
            except PairEpochError:
                raise
            except Exception as exc:  # noqa: BLE001 - legacy target isolation
                return self._finish_cross_cpu_failure(shard, clip_uid, exc)
            for rank, donor in enumerate(donors):
                try:
                    prepared = prepare_cross_pair_attempt(evidence, donor)
                except PairEpochError:
                    raise
                except Exception as exc:  # noqa: BLE001 - legacy target isolation
                    return self._finish_cross_cpu_failure(shard, clip_uid, exc)
                job = self._cross_job(
                    shard,
                    clip_uid,
                    entity,
                    entity_index,
                    evidence,
                    donor,
                    rank,
                    snapshot_identity,
                    prepared.donor_reference_image,
                )
                result = self._committed(job)
                if result is None:
                    return job
                if str(result.payload.get("status")) == "cross_pair_failed":
                    return self._finish_cross_failure(
                        shard, clip_uid, target_clip, entity, evidence, donor, result
                    )
                try:
                    outcome = finalize_cross_pair_judge(
                        storage,
                        target_clip=target_clip,
                        prepared=prepared,
                        attempt=_cross_payload_attempt(result),
                        counters=self._scratch(shard),
                    )
                except PairEpochError:
                    raise
                except Exception as exc:  # noqa: BLE001 - legacy target isolation
                    return self._finish_cross_cpu_failure(shard, clip_uid, exc)
                if outcome.verdict != "accepted":
                    continue
                states_by_id[entity.entity_id] = outcome.state
                accepted[entity.entity_id] = outcome.donor_path
                break

        entity_states = [
            states_by_id[entity.entity_id]
            for entity in target_clip.annotation.entities
            if entity.entity_id in states_by_id
        ]
        if not accepted:
            self._mark_cross_terminal(
                shard,
                clip_uid,
                status=CROSS_TERMINAL_COMPLETED,
                reason_kind="no_accepted_donor",
                reason="fallback completed without an accepted donor",
            )
            return None
        try:
            pairing = _pairing_from_references(
                target_clip, entity_states, bind_ready_background=False
            )
        except PairEpochError:
            raise
        except Exception as exc:  # noqa: BLE001 - legacy target isolation
            return self._finish_cross_cpu_failure(shard, clip_uid, exc)
        if pairing.status == "ready":
            # Part of the frozen legacy cross target outer try: an ordinary
            # CPU failure here (background artifact validation, deterministic
            # preparation, mode-off validation) terminates the target, while a
            # pending guard ModelJob is a normal unlock and guard semantic
            # drift stays retryable inside run(). PairEpochError is frozen
            # durable-state drift and must keep propagating.
            try:
                token, pending = self._background_token(
                    shard, clip_uid, frames, CALL_SITE_CROSS_PAIR
                )
            except PairEpochError:
                raise
            except Exception as exc:  # noqa: BLE001 - legacy target isolation
                return self._finish_cross_cpu_failure(shard, clip_uid, exc)
            if pending is not None:
                return pending
            pairing = pairing.model_copy(update={"background_token": token})
        references = ReferencesState(
            entities=entity_states, background=target_clip.references.background
        )
        live = storage.read_clip(clip_uid)
        baseline = self.cross_baseline(shard, clip_uid)
        if self._live_matches_cross_baseline(live, baseline):
            # Case A: still the frozen primary state -> publish once.
            _publish_cross_pair_result(
                storage,
                clip_uid=clip_uid,
                references=references,
                pairing=pairing,
                donor_paths=accepted,
            )
        elif live.pairing == pairing:
            # Case B: already published -> verify, never republish. Marker
            # state is not consulted: the publication transaction is the
            # authority.
            self._verify_existing_cross_publication(
                storage, clip_uid, references, pairing, accepted
            )
        else:
            # Case C: neither primary nor the reconstructed cross state.
            raise PairEpochError(
                f"clip {clip_uid} is neither the frozen primary state nor the "
                "reconstructed cross outcome"
            )
        self._mark_cross_terminal(
            shard,
            clip_uid,
            status=CROSS_TERMINAL_COMPLETED,
            reason_kind="published",
            reason=f"accepted donors: {sorted(accepted)}",
        )
        return None

    def _live_matches_cross_baseline(self, clip: Any, baseline: Mapping[str, Any]) -> bool:
        """True when the live clip is still exactly the frozen primary state."""
        pairing = (
            clip.pairing.model_dump(mode="json") if clip.pairing is not None else None
        )
        if pairing != baseline.get("primary_pairing"):
            return False
        entities = [
            state.model_dump(mode="json")
            for state in (clip.references.entities if clip.references is not None else ())
        ]
        return entities == baseline.get("primary_entity_references")

    def _verify_existing_cross_publication(
        self,
        storage: RunStorage,
        clip_uid: str,
        expected_references: Any,
        expected_pairing: Any,
        accepted_donor_paths: Mapping[str, Any],
    ) -> None:
        """Verify an already-published cross result instead of republishing."""
        clip = storage.read_clip(clip_uid)
        if clip.pairing != expected_pairing:
            raise PairEpochError(
                f"clip {clip_uid} cross pairing drifted from the reconstructed outcome"
            )
        live = [
            state.model_dump(mode="json")
            for state in (clip.references.entities if clip.references is not None else ())
        ]
        if live != [state.model_dump(mode="json") for state in expected_references.entities]:
            raise PairEpochError(
                f"clip {clip_uid} cross references drifted from the reconstructed outcome"
            )
        for entity_id, donor_path in accepted_donor_paths.items():
            published = storage.selected_entity_path(clip_uid, entity_id)
            if not published.is_file():
                raise PairEpochError(
                    f"clip {clip_uid} entity {entity_id} lost its cross-published PNG"
                )
            if _sha256_bytes(published.read_bytes()) != _sha256_bytes(
                Path(donor_path).read_bytes()
            ):
                raise PairEpochError(
                    f"clip {clip_uid} entity {entity_id} cross-published PNG differs "
                    "from the frozen donor reference"
                )

    def _finish_cross_cpu_failure(
        self, shard: str, clip_uid: str, exc: BaseException
    ) -> None:
        """Legacy target-block failure for an ordinary fallback CPU exception.

        Only called for real fallback execution failures. Frozen-state drift
        (snapshot mismatch, donor PNG mismatch, job digest mismatch,
        already-published mismatch) never reaches here: those stay fail closed
        so a corrupted durable state is never recorded as a Pair terminal.
        """
        from r2v_data_v2.v3.pair import _failure_details

        storage = self._storage_for(shard)
        try:
            storage.cleanup_pair_artifacts(clip_uid)
        except Exception as cleanup_error:  # noqa: BLE001
            self.emit("cross cleanup skipped", cleanup_error)
        storage.append_failure(
            stage="pair",
            clip_uid=clip_uid,
            reason=str(exc),
            details=_failure_details(exc),
        )
        self.stats[shard]["failed"] += 1
        self._mark_cross_terminal(
            shard,
            clip_uid,
            status=CROSS_TERMINAL_FAILED,
            reason_kind="cross_pair_cpu_failure",
            reason=f"{type(exc).__name__}: {exc}",
        )

    def _finish_cross_failure(
        self,
        shard: str,
        clip_uid: str,
        target_clip: Any,
        entity: Any,
        evidence: Any,
        donor: Any,
        result: JobResult,
    ) -> None:
        """Legacy target-block failure: debug, diagnostic, then the marker."""
        from r2v_data_v2.v3.pair import _failure_details, _write_cross_pair_debug

        storage = self._storage_for(shard)
        failure = _cross_failure_from_payload(result.payload)
        _write_cross_pair_debug(
            storage,
            target_clip=target_clip,
            target_entity=entity,
            target_evidence_mode=evidence.evidence_mode,
            target_frame_slots=evidence.frame_slots,
            target_context_image=evidence.context_image,
            donor=donor,
            failure=failure,
        )
        storage.append_failure(
            stage="pair",
            clip_uid=clip_uid,
            reason=str(failure),
            details=_failure_details(failure),
        )
        self.stats[shard]["failed"] += 1
        self._mark_cross_terminal(
            shard,
            clip_uid,
            status=CROSS_TERMINAL_FAILED,
            reason_kind="cross_pair_judge_failure",
            reason=str(result.payload.get("error") or ""),
        )

    # -- model calls -------------------------------------------------------

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        if job.job_type == PAIR_ENTITY_JUDGE_JOB:
            return self._run_entity_job(job, handle)
        if job.job_type == PAIR_BACKGROUND_GUARD_JOB:
            return self._run_guard_job(job, handle)
        if job.job_type == PAIR_CROSS_JUDGE_JOB:
            return self._run_cross_job(job, handle)
        raise PairEpochError(f"unknown Pair job type: {job.job_type}")

    def _run_cross_job(self, job: ModelJob, handle: Any) -> JobResult:
        from r2v_data_v2.v3.cross_pair_judge import (
            CrossPairJudgeFailure,
            QwenCrossPairJudge,
        )

        shard = job.canonical_shard
        clip_uid = job.clip_uid
        try:
            prepared, evidence, donor, rank, target_clip = self._cross_position(
                shard, clip_uid, dict(job.target)
            )
        except PairEpochError as exc:
            # Frozen durable-state drift must never be recorded as the legacy
            # target-block terminal outcome.
            return JobResult(OUTCOME_RETRYABLE_FAILED, detail=str(exc))
        snapshot_identity = self._snapshot_identity(self.donor_snapshot(shard))
        index = int(dict(job.target)["target_entity_index"])
        entity = target_clip.annotation.entities[index]
        rebuilt = self._cross_job(
            shard,
            clip_uid,
            entity,
            index,
            evidence,
            donor,
            rank,
            snapshot_identity,
            prepared.donor_reference_image,
        )
        if rebuilt.input_digest != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="cross-pair semantic inputs changed",
            )
        resolved = _resolve_qwen_judge(
            handle,
            self.config.qwen.cross_pair_judge,
            lambda svc: QwenCrossPairJudge(
                svc, repair_retries=self.config.pair.repair_retries
            ),
        )
        try:
            attempt = run_cross_pair_judge(prepared, resolved.judge)
        except CrossPairJudgeFailure as exc:
            # Legacy: the fallback aborts the whole target and is never retried.
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "cross_pair_failed",
                    "failure": exc.to_dict(),
                    "error": str(exc),
                },
            )
        except Exception as exc:  # noqa: BLE001 - infrastructure, not semantic
            return JobResult(OUTCOME_RETRYABLE_FAILED, detail=str(exc))
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "decision",
                "decision": attempt.decision.model_dump(mode="json"),
                "raw_responses": list(attempt.raw_responses),
                "repair_attempts": int(attempt.repair_attempts),
            },
        )

    def _run_entity_job(self, job: ModelJob, handle: Any) -> JobResult:
        storage = self._storage_for(job.canonical_shard)
        clip = storage.read_clip(job.clip_uid)
        try:
            frames = _validate_frames(storage, job.clip_uid)
            masks = storage.read_masks(job.clip_uid)
        except Exception as exc:  # noqa: BLE001 - drift, no receipt
            return JobResult(OUTCOME_RETRYABLE_FAILED, detail=str(exc))
        index = int(dict(job.target)["entity_index"])
        entity = clip.annotation.entities[index]
        prepared = self._prepared_for_job(job)
        if prepared is None:
            prepared = prepare_entity_reference(
                self.config,
                storage,
                clip_uid=job.clip_uid,
                entity=entity,
                frames=frames,
                masks=masks,
                counters=self._scratch(job.canonical_shard),
            )
            self._remember_prepared(job, prepared)
        if not isinstance(prepared, PreparedEntityReferenceDecision):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail=f"entity {entity.entity_id} became deterministic",
            )
        rebuilt = self._entity_job(
            job.canonical_shard, storage, job.clip_uid, entity, index, prepared
        )
        if rebuilt.input_digest != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="primary entity semantic inputs changed",
            )
        resolved = _resolve_qwen_judge(
            handle,
            self.config.qwen.candidate_judge,
            lambda svc: _entity_judge_factory(svc, self.config),
        )
        try:
            attempt = run_entity_reference_judge(prepared, resolved.judge)
        except Exception as exc:  # noqa: BLE001 - retryable, no receipt
            return JobResult(OUTCOME_RETRYABLE_FAILED, detail=str(exc))
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "decision": attempt.decision.model_dump(mode="json"),
                "raw_responses": list(attempt.raw_responses),
                "repair_attempts": int(attempt.repair_attempts),
            },
        )

    def _run_guard_job(self, job: ModelJob, handle: Any) -> JobResult:
        # Re-derivation must never move stage counters, so the runtime here is
        # scratch: `attempted` was already accounted once when the job was
        # planned.
        runtime = self._guard_runtime(job.canonical_shard)
        runtime.counters = self._scratch(job.canonical_shard)
        storage = self._storage_for(job.canonical_shard)
        clip = storage.read_clip(job.clip_uid)
        try:
            frames = _validate_frames(storage, job.clip_uid)
        except Exception as exc:  # noqa: BLE001 - drift, never a committed guard outcome
            return JobResult(OUTCOME_RETRYABLE_FAILED, detail=str(exc))
        call_site = dict(job.target)["call_site"]
        try:
            step = runtime.prepare_background_final_guard(clip=clip, frames=frames)
        except _GuardFailClosed:
            # The job was planned against earlier background semantics. If the
            # planned inputs no longer exist this is drift, not the semantic
            # failure that job was created to observe, so it must not be
            # committed under the old job identity.
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="background guard semantic inputs changed",
            )
        if step.preparation is None:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="background guard no longer requires a model call",
            )
        rebuilt = self._guard_job(
            job.canonical_shard, job.clip_uid, call_site, step.preparation
        )
        if rebuilt.input_digest != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="background guard semantic inputs changed",
            )
        resolved = _resolve_qwen_judge(
            handle, self.config.qwen.background_final_judge, _guard_judge_factory
        )
        runtime.active_judge = resolved.judge
        try:
            attempt = runtime.run_background_final_guard_judge(step.preparation)
        except _GuardFailClosed as closed:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "failed_closed",
                    "raw_response": closed.raw_response,
                    "error": str(closed.error),
                },
            )
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": (
                    "accepted" if attempt.review.verdict == "accept" else "rejected"
                ),
                "review": attempt.review.model_dump(mode="json"),
                "raw_response": attempt.raw_response,
            },
        )

    # -- finalization -------------------------------------------------------

    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        del result  # the committed receipt is the authority, not this object
        if job.job_type == PAIR_ENTITY_JUDGE_JOB:
            return self._finalize_primary_clip(job)
        if job.job_type == PAIR_BACKGROUND_GUARD_JOB:
            if dict(job.target).get("call_site") == CALL_SITE_CROSS_PAIR:
                pending = self._advance_cross_pair_target(
                    job.canonical_shard, job.clip_uid
                )
                return (pending,) if pending is not None else ()
            return self._finalize_primary_clip(job)
        if job.job_type == PAIR_CROSS_JUDGE_JOB:
            pending = self._advance_cross_pair_target(job.canonical_shard, job.clip_uid)
            return (pending,) if pending is not None else ()
        raise PairEpochError(f"unknown Pair job type: {job.job_type}")

    def _finalize_primary_clip(self, job: ModelJob) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        # Terminal barrier. A clip is reconstructed and published exactly once,
        # when its *last* primary entity receipt lands. Until then there is
        # nothing to do: replaying the clip would re-run every entity's
        # prepare, re-crop and re-encode the reference PNGs and immediately
        # discard them again, for every single sibling completion.
        if self._clip_primary_settled(shard, job.clip_uid) is False:
            return ()
        # Crash resume: replay from a clean temporary state.
        storage.cleanup_pair_artifacts(job.clip_uid)
        context = self._primary_context(storage, job.clip_uid)
        if context is None:
            return ()
        states, temporary, pending = self._replay_primary_clip(
            shard, storage, job.clip_uid, context, stop_at_unresolved=True
        )
        if pending:
            # Other entities still need Qwen: nothing is published yet, and
            # every one of them is independent so all go back to the pool.
            return tuple(pending)
        guard = self._publish_primary(
            shard, storage, job.clip_uid, context[1], states, temporary
        )
        if guard is None:
            return ()
        if guard.job_id() in self._clip_job_ids(shard, job.clip_uid):
            # Clip-level work, already handed to the graph by whichever entity
            # reached publication first. Every entity shares one publication, so
            # the guard is unlocked once, not once per entity.
            return ()
        self.phase.write_plan([guard])
        return (guard,)


# ---------------------------------------------------------------------------
# Frozen per-shard donor snapshot (Commit 3b)
#
# Donor eligibility, natural ordering, self exclusion and the donor cap all
# stay in pair.py. This module only freezes the ordered index that
# _build_same_parent_donor_index() returned and restores it later. It never
# re-filters, re-sorts, re-caps or re-derives eligibility.
# ---------------------------------------------------------------------------

PAIR_CROSS_JUDGE_JOB = "pair_cross_pair_judge"

#: Job types owned by the Pair stage; used for cross-phase plan scans.
PAIR_JOB_TYPES = (
    PAIR_ENTITY_JUDGE_JOB,
    PAIR_BACKGROUND_GUARD_JOB,
    PAIR_CROSS_JUDGE_JOB,
)
CALL_SITE_CROSS_PAIR = "cross_pair"

PAIR_DONOR_SNAPSHOT_SCHEMA = "post_mask_epoch_pair_donor_snapshot/2"
PAIR_CROSS_BASELINE_SCHEMA = "post_mask_epoch_pair_cross_baseline/1"


def _donor_projection(donor: Any, storage: RunStorage) -> dict[str, Any]:
    path = Path(donor.image_path)
    payload = path.read_bytes() if path.is_file() else b""
    return {
        "clip_uid": donor.clip.clip_uid,
        "clip_suffix": donor.clip.source.clip_suffix,
        "entity_id": donor.entity.entity_id,
        "entity": donor.entity.model_dump(mode="json"),
        "reference": donor.reference.model_dump(mode="json"),
        "image_sha256": _sha256_bytes(payload),
        "image_size": len(payload),
    }


def _restore_donor(
    storage: RunStorage, parent: str, entry: Mapping[str, Any]
) -> Any:
    """Validate one frozen donor projection and rebuild its carrier."""
    from r2v_data_v2.v3.pair import _CrossPairDonor

    clip_uid = entry["clip_uid"]
    clip = storage.read_clip(clip_uid)
    if clip.source.parent_video_id != parent:
        raise PairEpochError(f"frozen donor {clip_uid} parent drifted")
    if clip.source.clip_suffix != entry["clip_suffix"]:
        raise PairEpochError(f"frozen donor {clip_uid} clip_suffix drifted")
    if clip.pairing is None or clip.pairing.status != "ready":
        raise PairEpochError(f"frozen donor {clip_uid} pairing is not ready")

    entities = tuple(clip.annotation.entities) if clip.annotation is not None else ()
    entity = next((item for item in entities if item.entity_id == entry["entity_id"]), None)
    if entity is None:
        raise PairEpochError(f"frozen donor {clip_uid} lost entity {entry['entity_id']}")
    if entity.model_dump(mode="json") != entry["entity"]:
        raise PairEpochError(f"frozen donor {clip_uid} entity projection drifted")
    if entry["entity_id"] not in (clip.pairing.retained_entity_ids or ()):
        raise PairEpochError(f"frozen donor {clip_uid} no longer retains {entry['entity_id']}")

    states = tuple(clip.references.entities) if clip.references is not None else ()
    state = next((item for item in states if item.entity_id == entry["entity_id"]), None)
    if state is None:
        raise PairEpochError(f"frozen donor {clip_uid} lost reference {entry['entity_id']}")
    if state.model_dump(mode="json") != entry["reference"]:
        raise PairEpochError(f"frozen donor {clip_uid} reference projection drifted")

    path = storage.selected_entity_path(clip_uid, entry["entity_id"])
    if not path.is_file():
        raise PairEpochError(f"frozen donor {clip_uid} lost its selected PNG")
    payload = path.read_bytes()
    if len(payload) != entry["image_size"] or _sha256_bytes(payload) != entry["image_sha256"]:
        raise PairEpochError(f"frozen donor {clip_uid} selected PNG changed")

    return _CrossPairDonor(clip=clip, entity=entity, reference=state, image_path=path)


def load_frozen_donor_index(
    storage: RunStorage, snapshot: Mapping[str, Any]
) -> dict[tuple[str, str], tuple[Any, ...]]:
    """Reconstruct the frozen index, keeping the exact snapshot order.

    Nothing is re-sorted and eligibility is never re-derived: the result is
    fed straight to pair.py's _donors_for_target.
    """
    groups = snapshot.get("groups")
    if not isinstance(groups, list):
        raise PairEpochError("frozen donor snapshot has a malformed groups list")
    index: dict[tuple[str, str], list[Any]] = {}
    for group in groups:
        key = (group["parent_video_id"], group["reference_type"])
        if key in index:
            raise PairEpochError(f"frozen donor snapshot has duplicate group {key!r}")
        entries = group.get("donors")
        if not isinstance(entries, list):
            raise PairEpochError("frozen donor group has a malformed donors list")
        donors: list[Any] = []
        # The ordinal is the frozen order itself: it is verified, never sorted.
        for expected_ordinal, entry in enumerate(entries):
            if entry.get("ordinal") != expected_ordinal:
                raise PairEpochError(
                    f"frozen donor ordinal mismatch at {expected_ordinal} "
                    f"for group {key!r}"
                )
            donors.append(_restore_donor(storage, group["parent_video_id"], entry))
        index[key] = donors
    return {key: tuple(items) for key, items in index.items()}


# ---------------------------------------------------------------------------
# Durable cross-pair chain (Commit 3b-2)
# ---------------------------------------------------------------------------

PAIR_CROSS_TERMINAL_SCHEMA = "post_mask_epoch_pair_cross_terminal/1"

CROSS_TERMINAL_COMPLETED = "completed"
CROSS_TERMINAL_FAILED = "failed"


def _cross_failure_from_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild a real CrossPairJudgeFailure from the durable payload.

    Its constructor is keyword-only and carries raw_responses, issues and
    attempt_count, so the payload has to keep all three; `error` is only a
    human-readable hint and is not the authority.
    """
    from r2v_data_v2.structured_output import ValidationIssue
    from r2v_data_v2.v3.cross_pair_judge import CrossPairJudgeFailure

    failure_payload = dict(payload.get("failure") or {})
    issues = [
        ValidationIssue(**issue) for issue in failure_payload.get("issues", ())
    ]
    return CrossPairJudgeFailure(
        raw_responses=list(failure_payload.get("raw_responses", ())),
        issues=issues,
        attempt_count=failure_payload.get("attempt_count"),
    )


def _cross_payload_attempt(result: JobResult) -> Any:
    from r2v_data_v2.v3.cross_pair_judge import CrossPairDecisionAttempt
    from r2v_data_v2.v3.schemas import RawCrossPairDecision

    return CrossPairDecisionAttempt(
        decision=RawCrossPairDecision.model_validate(result.payload["decision"]),
        raw_responses=tuple(result.payload.get("raw_responses", ())),
        repair_attempts=int(result.payload.get("repair_attempts", 0)),
    )
