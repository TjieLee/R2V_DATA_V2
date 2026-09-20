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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pair import (
    EntityReferenceState,
    PreparedEntityReferenceDecision,
    _BackgroundFinalGuardRuntime,
    _GuardFailClosed,
    _publish_pair_result,
    finalize_entity_reference,
    prepare_entity_reference,
    run_entity_reference_judge,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    semantic_input_digest,
)
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
            factory(service.model_copy(update={"base_url": handle})), owned=True
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
            if existing.get("schema") != PAIR_PRIMARY_PLAN_SCHEMA:
                raise PairEpochError(
                    f"unsupported primary plan schema for {shard!r}"
                )
            expected = {
                "canonical_shard": shard,
                "config_fingerprint": self.config.fingerprint(),
                "pair_policy": _pair_policy_semantics(self.config),
                "eligible_clip_uids": list(self._eligible_for(shard)),
            }
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise PairEpochError(
                        f"frozen primary plan {key} drifted for {shard!r}"
                    )
            # The stored per-clip input digests are the whole point of the
            # plan: re-derive them and fail closed on any pre-Pair drift.
            storage = self._storage_for(shard)
            for clip_uid, entry in existing.get("clips", {}).items():
                actual = self._frozen_input_digest(storage, clip_uid)
                if actual != entry.get("digest"):
                    raise PairEpochError(
                        f"frozen primary input drifted for {clip_uid!r} in {shard!r}"
                    )
                if entry.get("classification") == CLIP_EXISTING_PAIRING:
                    self._validate_existing_pairing(shard, storage, clip_uid)
            return existing

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

    def _primary_context(self, storage: RunStorage, clip_uid: str) -> Any:
        from r2v_data_v2.v3.pair import _validate_pair_inputs

        clip = storage.read_clip(clip_uid)
        try:
            frames = _validate_frames(storage, clip_uid)
            masks = storage.read_masks(clip_uid)
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

    def _replay_primary_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        context: Any,
        *,
        stop_at_unresolved: bool,
    ) -> tuple[
        list[EntityReferenceState], dict[str, tuple[Path, Image.Image]], ModelJob | None
    ]:
        """Rebuild a clip's primary prefix from committed results only.

        Deterministic entities are recomputed; entity judge outcomes are read
        back from their receipts. Nothing lives only in memory, so a crash
        before publication costs zero extra Qwen calls.
        """
        clip, frames, masks = context
        counters = self._scratch(shard)
        entity_states: list[EntityReferenceState] = []
        temporary_images: dict[str, tuple[Path, Image.Image]] = {}
        for index, entity in enumerate(clip.annotation.entities):
            prepared = prepare_entity_reference(
                self.config,
                storage,
                clip_uid=clip_uid,
                entity=entity,
                frames=frames,
                masks=masks,
                counters=counters,
            )
            if isinstance(prepared, EntityReferenceState):
                entity_states.append(prepared)
                continue
            job = self._entity_job(shard, storage, clip_uid, entity, index, prepared)
            result = self._committed(job)
            if result is None:
                if not stop_at_unresolved:
                    raise PairEpochError(
                        f"primary entity {entity.entity_id} has no committed result"
                    )
                self._discard_temporary(temporary_images)
                return entity_states, {}, job
            finalization = finalize_entity_reference(
                self.config,
                storage,
                prepared=prepared,
                attempt=_attempt_from_result(result),
            )
            entity_states.append(finalization.state)
            if finalization.temporary is not None:
                temporary_images[entity.entity_id] = finalization.temporary
        return entity_states, temporary_images, None

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
                pending = self._advance_primary_clip(shard, storage, clip_uid, context)
                if pending is not None:
                    jobs.append(pending)
        jobs.sort(key=lambda job: job.job_id())
        self.phase.write_plan(jobs)
        return jobs

    def _advance_primary_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        context: Any,
    ) -> ModelJob | None:
        """CPU -> MODEL -> CPU until a fixed point is reached.

        Returns the single job the scheduler still has to run, or ``None`` when
        the clip's primary Pair is terminal.
        """
        states, temporary, pending = self._replay_primary_clip(
            shard, storage, clip_uid, context, stop_at_unresolved=True
        )
        if pending is not None:
            # Rebuilt temporaries are not progress: only receipts are.
            self._discard_temporary(temporary)
            return pending
        guard = self._publish_primary(
            shard, storage, clip_uid, context[1], states, temporary
        )
        return guard

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
                pending = self._advance_primary_clip(shard, storage, clip_uid, context)
                if pending is not None:
                    unresolved.append(pending.job_id())
        return tuple(unresolved)

    # -- model calls -------------------------------------------------------

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        if job.job_type == PAIR_ENTITY_JUDGE_JOB:
            return self._run_entity_job(job, handle)
        if job.job_type == PAIR_BACKGROUND_GUARD_JOB:
            return self._run_guard_job(job, handle)
        raise PairEpochError(f"unknown Pair job type: {job.job_type}")

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
        prepared = prepare_entity_reference(
            self.config,
            storage,
            clip_uid=job.clip_uid,
            entity=entity,
            frames=frames,
            masks=masks,
            counters=self._scratch(job.canonical_shard),
        )
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
            return self._finalize_primary_clip(job)
        raise PairEpochError(f"unknown Pair job type: {job.job_type}")

    def _finalize_primary_clip(self, job: ModelJob) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        # Crash resume: replay from a clean temporary state.
        storage.cleanup_pair_artifacts(job.clip_uid)
        context = self._primary_context(storage, job.clip_uid)
        if context is None:
            return ()
        states, temporary, pending = self._replay_primary_clip(
            shard, storage, job.clip_uid, context, stop_at_unresolved=True
        )
        if pending is not None:
            # A later entity still needs Qwen: nothing is published yet.
            return (pending,)
        guard = self._publish_primary(
            shard, storage, job.clip_uid, context[1], states, temporary
        )
        return (guard,) if guard is not None else ()
