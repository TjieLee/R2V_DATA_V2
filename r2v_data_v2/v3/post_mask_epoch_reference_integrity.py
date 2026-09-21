"""Durable Reference Integrity resource epoch (6b: CPU foundation).

The frozen legacy authority stays ``reference_integrity_clips()``: this runner
replays the exact same clip policy through durable per-shard state and reuses
the legacy CPU helpers (topology diagnostics, semantic hard-reject, semantic
risk routing, rejected-reference construction, pairing tokens, artifact
resolution) verbatim instead of copying any policy.

Scope of 6b: the two CPU-only entity terminal branches plus the durable
foundation they need (frozen plan, per-entity outcomes, clip outcomes,
publication reconstruction and strict restart verification). An entity that
requires a Qwen review is deliberately left unresolved: no marker is written,
nothing is faked, and its clip simply never publishes. Model jobs arrive in
6c/6d.

Durable root (inside the group ledger)::

    semantic/reference_integrity/
        primary/<shard>.json                          frozen plan
        entity_outcomes/<shard>/<clip>/<entity>.json  per-entity outcome
        outcomes/<shard>/<clip>.json                  clip outcome marker

No model call, no ``stage_counts`` publication and no pipeline wiring happen
here.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    canonical_json,
    semantic_input_digest,
)
from r2v_data_v2.v3.reference_integrity import (
    SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT,
    QwenReferenceIntegrityJudge,
    QwenSourceBboxFallbackJudge,
    ReferenceIntegrityJudgeFailure,
    ReferenceIntegrityStats,
    SourceBboxFallbackJudgeFailure,
    _artifact_only_bbox_eligible,
    _is_self_sourced_reference,
    _materialize_source_bbox,
    _reference_edit_after_source_alpha,
    _reference_edit_after_source_bbox,
    _rejected_reference,
    _resolve_run_artifact,
    _source_bbox_candidate,
    _source_bbox_reference,
    _source_context,
    _source_evidence,
    _tokens_for_retained,
    _topology_bbox_upgrade_eligible,
    reference_semantic_hard_reject_reason,
    reference_semantic_risk_reason,
    reference_topology_diagnostics,
)
from r2v_data_v2.v3.reference_integrity import (
    SYSTEM_PROMPT as REFERENCE_INTEGRITY_SYSTEM_PROMPT,
)
from r2v_data_v2.v3.schemas import (
    EntityReferenceState,
    ExportState,
    PairingState,
    ReferenceEditState,
    ReferenceIntegrityEntityState,
    ReferenceIntegrityReview,
    ReferenceIntegrityState,
    ReferencesState,
    SourceBboxFallbackReview,
)
from r2v_data_v2.v3.storage import RunStorage

REFERENCE_INTEGRITY_PLAN_SCHEMA = "post_mask_epoch_reference_integrity_plan/1"
REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_integrity_entity_outcome/3"
)
REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_integrity_outcome/1"
)
REFERENCE_INTEGRITY_POLICY_VERSION = "reference_integrity_epoch_policy/3"
REFERENCE_INTEGRITY_REVIEW_INPUT_SCHEMA = (
    "post_mask_epoch_reference_integrity_review_input/1"
)
REFERENCE_INTEGRITY_BBOX_INPUT_SCHEMA = (
    "post_mask_epoch_reference_integrity_bbox_input/1"
)
REFERENCE_INTEGRITY_MAIN_REVIEW_POLICY_VERSION = (
    "reference_integrity_main_review/1"
)
REFERENCE_INTEGRITY_BBOX_REVIEW_POLICY_VERSION = (
    "reference_integrity_bbox_review/1"
)

#: The one model job type this epoch owns. 6c implements the main review
#: (``variant="final"``); 6d1 adds the source-alpha review for a completion the
#: main review rejected. The source bbox fallback review stays out (6d2).
REFERENCE_INTEGRITY_REVIEW_JOB = "reference_integrity_review"
VARIANT_FINAL = "final"
VARIANT_SOURCE_ALPHA = "source_alpha"
IMPLEMENTED_REVIEW_VARIANTS = (VARIANT_FINAL, VARIANT_SOURCE_ALPHA)

#: The second Qwen job type: the source bbox fallback review of one materialized
#: raw-source candidate. 6d2a implements only the ``artifact_review_reject``
#: trigger; the topology upgrade trigger stays out (6d2b).
REFERENCE_INTEGRITY_BBOX_REVIEW_JOB = "reference_integrity_bbox_review"
BBOX_TRIGGER_ARTIFACT = "artifact_review_reject"
BBOX_REVIEW_MODE = "source_bbox_fallback_v1"
BBOX_JUDGE_FAILED_PREFIX = "source_bbox_fallback_judge_failed:"

#: Dependency address of the main review inside a source-alpha review job.
MAIN_REVIEW_JOB_DEPENDENCY = "main_review"
#: Dependency address of the parent integrity review inside a bbox review job.
PARENT_REVIEW_JOB_DEPENDENCY = "parent_review"

SOURCE_ALPHA_ACCEPTED_REASON = "completion_integrity_rejected_source_alpha_accepted"
SOURCE_ALPHA_FALLBACK_REASON = "completion_integrity_rejected_fallback_to_alpha"
SOURCE_ALPHA_JUDGE_FAILED_PREFIX = (
    "completion_rejected_then_alpha_integrity_judge_failed:"
)
MAIN_REVIEW_JUDGE_FAILED_PREFIX = "integrity_judge_failed:"

CLIP_FRESH_TARGET = "fresh_target"
CLIP_EXISTING = "existing"
CLIP_INELIGIBLE = "ineligible"

#: Entity branch statuses. ``skipped`` is CPU-only; ``rejected``/``accepted``
#: may be CPU-decided (hard reject) or produced by a committed main review.
ENTITY_OUTCOME_REJECTED = "rejected"
ENTITY_OUTCOME_SKIPPED = "skipped"
ENTITY_OUTCOME_ACCEPTED = "accepted"
CPU_ENTITY_OUTCOMES = (ENTITY_OUTCOME_REJECTED, ENTITY_OUTCOME_SKIPPED)
REVIEWED_ENTITY_OUTCOMES = (ENTITY_OUTCOME_ACCEPTED, ENTITY_OUTCOME_REJECTED)

#: Counter fields a CPU-only entity outcome may carry. ``entities_reviewed`` is
#: deliberately absent: it counts real main-judge review invocations, which 6b
#: never performs.
ENTITY_DELTA_FIELDS = (
    "topology_suspicious",
    "entities_rejected",
    "entities_accepted",
    "entities_skipped_review",
    "semantic_policy_rejected",
    "entities_reviewed",
    "judge_failed",
    "source_bbox_fallback_attempted",
    "source_bbox_fallback_accepted",
    "source_bbox_fallback_rejected",
    "source_bbox_fallback_judge_failed",
)

#: Marker keys that only a reviewed outcome may carry. ``review_variant`` and
#: ``parent_review_job_id`` make the reviewed provenance explicit: a source-alpha
#: outcome is a continuation of one exact main review receipt.
REVIEWED_MARKER_FIELDS = (
    "review_job_id",
    "review_variant",
    "parent_review_job_id",
    "source_context_path",
    "review",
    "judge_failed",
)

#: Additional marker keys an artifact source-bbox outcome carries. ``review``
#: stays the *parent integrity* review; ``source_bbox_fallback_review`` is the
#: bbox judge's own verdict, and ``bbox_judge_failed`` is this epoch's internal
#: provenance flag (legacy never sets the public
#: ``source_bbox_fallback_judge_failed`` for the artifact trigger).
BBOX_MARKER_FIELDS = (
    "bbox_review_job_id",
    "source_bbox_fallback_trigger",
    "source_bbox_fallback_candidate_path",
    "source_bbox_fallback_metadata_path",
    "source_bbox_xyxy",
    "source_bbox_fallback_review",
    "bbox_judge_failed",
)

CLIP_DELTA_FIELDS = ("processed", "failed")
#: The ready clip marker binds the FULL clip stats delta, so it also carries the
#: per-entity counters its entity outcomes contributed.
CLIP_DELTA_KNOWN_FIELDS = CLIP_DELTA_FIELDS + ENTITY_DELTA_FIELDS

CLEAN_REAL_FULL_REFERENCE = "clean_real_full_reference"
REJECTED_REFERENCE_REASON = "reference_integrity_rejected"


@dataclass(frozen=True)
class _BboxParent:
    """The exact frozen parent state one artifact bbox job continues from.

    ``variant`` is the parent integrity review variant; ``review`` is that
    parent's committed verdict, ``reference`` the reference it judged,
    ``diagnostics`` its topology diagnostics and ``context_path`` its frozen
    review context PNG. ``parent_review_job_id`` is the *grandparent* main
    review job for an alpha parent, which is what the entity marker binds.
    """

    variant: str
    job: ModelJob
    payload: dict[str, Any]
    review: ReferenceIntegrityReview
    reference: EntityReferenceState
    diagnostics: Any
    main_diagnostics: Any
    context_path: str
    context_sha256: str
    parent_review_job_id: str | None


@dataclass(frozen=True)
class _ReviewOutcome:
    """One committed review's exact durable marker plus what it unlocks.

    ``marker`` is ``None`` while a continuation still owns the entity; ``unlock``
    is the deterministic model job that continuation must run next, if this
    epoch implements it.
    """

    marker: dict[str, Any] | None
    unlock: ModelJob | None


@dataclass(frozen=True)
class ResolvedIntegrityJudge:
    """A judge handle plus whether *this* call created it."""

    judge: Any
    owned: bool


def resolve_reference_integrity_judge(
    handle: Any, config: V3Config
) -> ResolvedIntegrityJudge:
    """Turn the epoch's handle into a main integrity review judge.

    Production (6e) hands in the shared Qwen endpoint string, in which case an
    owned judge is built against that endpoint with the configured model and the
    caller closes it. An injected judge object is reused as-is and never closed.
    There is deliberately no fallback to any other model.
    """
    if handle is None:
        raise ReferenceIntegrityEpochError(
            "reference integrity review needs a judge handle"
        )
    if isinstance(handle, str):
        service = config.qwen.reference_integrity_judge
        if service is None:
            raise ReferenceIntegrityEpochError(
                "no configured reference integrity judge for the endpoint handle"
            )
        return ResolvedIntegrityJudge(
            QwenReferenceIntegrityJudge(replace(service, base_url=handle)),
            owned=True,
        )
    return ResolvedIntegrityJudge(handle, owned=False)


def resolve_source_bbox_fallback_judge(
    handle: Any, config: V3Config
) -> ResolvedIntegrityJudge:
    """Turn the epoch's handle into a source bbox fallback judge.

    Same contract as the integrity judge resolver: an endpoint string builds an
    owned judge against the configured service, an injected judge is reused and
    never closed, and ``None`` is an error rather than a fallback to another
    model.
    """
    if handle is None:
        raise ReferenceIntegrityEpochError(
            "reference integrity bbox review needs a judge handle"
        )
    if isinstance(handle, str):
        service = config.qwen.reference_integrity_judge
        if service is None:
            raise ReferenceIntegrityEpochError(
                "no configured reference integrity judge for the endpoint handle"
            )
        return ResolvedIntegrityJudge(
            QwenSourceBboxFallbackJudge(replace(service, base_url=handle)),
            owned=True,
        )
    return ResolvedIntegrityJudge(handle, owned=False)


class ReferenceIntegrityEpochError(RuntimeError):
    """Raised when durable Reference Integrity state cannot be trusted."""


class ReferenceIntegrityDurableError(ReferenceIntegrityEpochError):
    """Durable corruption/drift: never semantic-failed, always fail closed."""


def _read_json(path: Path) -> dict[str, Any] | None:
    """Durable reader: missing is ``None``, corrupt is always fail-closed."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReferenceIntegrityDurableError(
            f"invalid durable Reference Integrity JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReferenceIntegrityDurableError(
            f"durable Reference Integrity JSON is not an object: {path}"
        )
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create-or-validate with the epoch durable-state fsync writer."""
    from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise ReferenceIntegrityDurableError(
                f"durable Reference Integrity file drifted: {path}"
            )
        return
    atomic_write_json(path, dict(payload))


def _plan_pre_reference(
    plan_entry: Mapping[str, Any], entity_id: str
) -> EntityReferenceState:
    """The frozen pre-stage reference of one entity, or fail closed.

    The source-alpha continuation is decided on the *pre-stage* reference (legacy
    reads it before any replacement), so it must come from the frozen baselines.
    """
    pre_references = plan_entry.get("pre_references")
    if not isinstance(pre_references, dict):
        raise ReferenceIntegrityDurableError(
            f"frozen plan has no pre-stage references for {entity_id!r}"
        )
    raw = next(
        (
            item
            for item in pre_references.get("entities", [])
            if isinstance(item, dict) and item.get("entity_id") == entity_id
        ),
        None,
    )
    if raw is None:
        raise ReferenceIntegrityDurableError(
            f"frozen plan has no pre-stage reference for {entity_id!r}"
        )
    try:
        return EntityReferenceState.model_validate(raw)
    except Exception as exc:
        raise ReferenceIntegrityDurableError(
            f"frozen pre-stage reference is invalid for {entity_id!r}"
        ) from exc


def _source_alpha_edit_entity(
    pre_edit: ReferenceEditState | None, entity_id: str
) -> Any:
    """Legacy's ``reference_edits_by_id`` lookup, exactly.

    Legacy builds that map only from a Reference Edit state that exists *and* is
    ready, so anything else can never trigger the source-alpha continuation.
    """
    if pre_edit is None or pre_edit.status != "ready":
        return None
    return next(
        (item for item in pre_edit.entities if item.entity_id == entity_id),
        None,
    )


def _load_reference_image(storage: RunStorage, image_path: str) -> Any:
    """Read one reference PNG exactly like the legacy stage does."""
    resolved = _resolve_run_artifact(storage, image_path)
    with Image.open(resolved) as opened:
        opened.load()
        return opened.copy()


@dataclass(frozen=True)
class _EntityBranch:
    """One CPU-decided entity branch, ready to become a durable outcome.

    ``requires_model`` marks the branches 6b does not own: the entity needs a
    Qwen review, so it stays unresolved instead of being faked.
    """

    outcome: str | None
    status: str | None
    reason: str | None
    reviewed: bool
    semantic_policy_reason: str | None
    diagnostics: Any
    delta: dict[str, int] = field(default_factory=dict)
    requires_model: bool = False


class ReferenceIntegrityEpochRunner:
    """CPU-only durable foundation for the Reference Integrity stage.

    Every decision is re-derived from frozen state, so an interrupted run
    replays with zero model calls and any live drift fails closed.
    """

    def __init__(
        self,
        config: V3Config,
        storages: Mapping[str, RunStorage],
        ledger: Any,
        *,
        eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
        emit: Any = None,
    ) -> None:
        if not config.reference_integrity.enabled:
            raise ReferenceIntegrityEpochError(
                "Reference Integrity resource epoch requires "
                "reference_integrity.enabled"
            )
        self.config = config
        self.storages = dict(storages)
        self.ledger = ledger
        self.eligible = {
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        }
        self.emit = emit

    # -- durable paths ---------------------------------------------------------

    def _semantic(self, *parts: str) -> Path:
        return (
            Path(self.ledger.root)
            / "semantic"
            / "reference_integrity"
            / Path(*parts)
        )

    def _plan_path(self, shard: str) -> Path:
        return self._semantic("primary", f"{shard}.json")

    def _entity_outcome_path(self, shard: str, clip_uid: str, entity_id: str) -> Path:
        return self._semantic("entity_outcomes", shard, clip_uid, f"{entity_id}.json")

    def _clip_outcome_path(self, shard: str, clip_uid: str) -> Path:
        return self._semantic("outcomes", shard, f"{clip_uid}.json")

    def _review_input_path(
        self,
        shard: str,
        clip_uid: str,
        entity_id: str,
        variant: str = VARIANT_FINAL,
    ) -> Path:
        return self._semantic(
            "review_inputs", shard, clip_uid, entity_id, f"{variant}.json"
        )

    def _bbox_input_path(self, shard: str, clip_uid: str, entity_id: str) -> Path:
        """One entity has at most one legacy bbox route, so no attempt index."""
        return self._semantic("bbox_inputs", shard, clip_uid, f"{entity_id}.json")

    def _storage_for(self, shard: str) -> RunStorage:
        storage = self.storages.get(shard)
        if storage is None:
            raise ReferenceIntegrityEpochError(
                f"no run storage for canonical shard {shard!r}"
            )
        return storage

    # -- policy identity -------------------------------------------------------

    def _main_review_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the main integrity review model behavior.

        Only what changes *what the model is asked and judged by* is bound:
        the served model, the token budget, the system prompt, the strict review
        schema and the review mode. Endpoint, credentials, timeout and any
        placement/concurrency settings are execution-only and stay out.
        """
        service = self.config.qwen.reference_integrity_judge
        if service is None:
            raise ReferenceIntegrityEpochError(
                "reference integrity Qwen judge is not configured"
            )
        return {
            "policy_version": REFERENCE_INTEGRITY_MAIN_REVIEW_POLICY_VERSION,
            "model": str(service.model),
            "max_tokens": int(service.max_tokens),
            "system_prompt_sha256": hashlib.sha256(
                REFERENCE_INTEGRITY_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "review_schema_sha256": hashlib.sha256(
                canonical_json(
                    ReferenceIntegrityReview.model_json_schema()
                ).encode("utf-8")
            ).hexdigest(),
            "mode": "targeted_qwen_v1",
        }

    def _bbox_review_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the source bbox fallback review model behavior."""
        service = self.config.qwen.reference_integrity_judge
        if service is None:
            raise ReferenceIntegrityEpochError(
                "reference integrity Qwen judge is not configured"
            )
        return {
            "policy_version": REFERENCE_INTEGRITY_BBOX_REVIEW_POLICY_VERSION,
            "model": str(service.model),
            "max_tokens": int(service.max_tokens),
            "system_prompt_sha256": hashlib.sha256(
                SOURCE_BBOX_FALLBACK_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "review_schema_sha256": hashlib.sha256(
                canonical_json(
                    SourceBboxFallbackReview.model_json_schema()
                ).encode("utf-8")
            ).hexdigest(),
            "mode": BBOX_REVIEW_MODE,
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
        }

    def _policy_identity(self) -> dict[str, Any]:
        policy = self.config.reference_integrity
        return {
            "policy_version": REFERENCE_INTEGRITY_POLICY_VERSION,
            "enabled": bool(policy.enabled),
            "mode": str(policy.mode),
            "reference_edit_enabled": bool(self.config.reference_edit.enabled),
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
            "main_review": self._main_review_policy_identity(),
            "bbox_review": self._bbox_review_policy_identity(),
        }

    # -- clip classification ---------------------------------------------------

    def _clip_classification(self, clip: Any) -> str:
        """Exact legacy eligibility for one clip."""
        if clip.pairing is None or clip.pairing.status != "ready":
            return CLIP_INELIGIBLE
        if self.config.reference_edit.enabled and (
            clip.reference_edit is None or clip.reference_edit.status != "ready"
        ):
            return CLIP_INELIGIBLE
        if (
            clip.reference_integrity is not None
            and clip.reference_integrity.status == "ready"
        ):
            return CLIP_EXISTING
        return CLIP_FRESH_TARGET

    # -- frozen plan -----------------------------------------------------------

    def _clip_digest(self, storage: RunStorage, clip: Any) -> str:
        """Digest of the publication-stable inputs this stage reads.

        ``references`` / ``pairing`` / ``reference_edit`` are deliberately
        excluded: this stage legitimately rewrites them on publication, so they
        are frozen as baselines and compared only while the clip is unpublished.
        """
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
            "sampled_frames": storage.read_frames(clip.clip_uid).model_dump(
                mode="json"
            ),
            "tracked_masks": storage.read_masks(clip.clip_uid).model_dump(mode="json"),
            "policy": self._policy_identity(),
        }
        return semantic_input_digest(projection)

    def _plan_entry(self, storage: RunStorage, clip: Any) -> dict[str, Any]:
        classification = self._clip_classification(clip)
        return {
            "classification": classification,
            "digest": self._clip_digest(storage, clip),
            "retained_entity_ids": (
                list(clip.pairing.retained_entity_ids)
                if classification == CLIP_FRESH_TARGET
                else []
            ),
            "pre_references": clip.references.model_dump(mode="json"),
            "pre_pairing": (
                clip.pairing.model_dump(mode="json")
                if clip.pairing is not None
                else None
            ),
            "pre_reference_edit": (
                clip.reference_edit.model_dump(mode="json")
                if clip.reference_edit is not None
                else None
            ),
            "pre_reference_integrity": (
                clip.reference_integrity.model_dump(mode="json")
                if clip.reference_integrity is not None
                else None
            ),
            "pre_instruction": (
                clip.instruction.model_dump(mode="json")
                if clip.instruction is not None
                else None
            ),
            "pre_export": clip.export.model_dump(mode="json"),
        }

    def _live_baselines(self, clip: Any) -> dict[str, Any]:
        """The six pre-stage sections compared against a frozen plan entry."""
        return {
            "pre_references": clip.references.model_dump(mode="json"),
            "pre_pairing": (
                clip.pairing.model_dump(mode="json")
                if clip.pairing is not None
                else None
            ),
            "pre_reference_edit": (
                clip.reference_edit.model_dump(mode="json")
                if clip.reference_edit is not None
                else None
            ),
            "pre_reference_integrity": (
                clip.reference_integrity.model_dump(mode="json")
                if clip.reference_integrity is not None
                else None
            ),
            "pre_instruction": (
                clip.instruction.model_dump(mode="json")
                if clip.instruction is not None
                else None
            ),
            "pre_export": clip.export.model_dump(mode="json"),
        }

    @staticmethod
    def _expected_classification(config: Any, entry: Mapping[str, Any]) -> str:
        """Re-derive the legacy classification from the frozen baselines."""
        pre_pairing = entry.get("pre_pairing")
        if not isinstance(pre_pairing, dict) or pre_pairing.get("status") != "ready":
            return CLIP_INELIGIBLE
        pre_edit = entry.get("pre_reference_edit")
        if config.reference_edit.enabled and (
            not isinstance(pre_edit, dict) or pre_edit.get("status") != "ready"
        ):
            return CLIP_INELIGIBLE
        pre_integrity = entry.get("pre_reference_integrity")
        if isinstance(pre_integrity, dict) and pre_integrity.get("status") == "ready":
            return CLIP_EXISTING
        return CLIP_FRESH_TARGET

    def _verify_plan_entry(
        self, shard: str, storage: RunStorage, clip_uid: str, entry: Mapping[str, Any]
    ) -> None:
        """Strictly validate one frozen plan entry against the live clip.

        A plan entry is durable authority, so both its structure and its
        classification are re-derived; a tampered plan may never silently change
        which clips or entities this stage executes.
        """
        clip = storage.read_clip(clip_uid)
        actual = self._clip_digest(storage, clip)
        if actual != entry.get("digest"):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity input drifted for {clip_uid!r}"
            )
        classification = entry.get("classification")
        if classification not in (CLIP_FRESH_TARGET, CLIP_EXISTING, CLIP_INELIGIBLE):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity classification is unknown for "
                f"{clip_uid!r}"
            )
        expected_classification = self._expected_classification(self.config, entry)
        if classification != expected_classification:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity classification drifted for "
                f"{clip_uid!r}"
            )
        retained = entry.get("retained_entity_ids")
        if not isinstance(retained, list) or not all(
            isinstance(item, str) for item in retained
        ):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity retained entities are malformed for "
                f"{clip_uid!r}"
            )
        pre_pairing = entry.get("pre_pairing") or {}
        if classification == CLIP_FRESH_TARGET:
            if retained != list(pre_pairing.get("retained_entity_ids", [])):
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity retained entities drifted for "
                    f"{clip_uid!r}"
                )
        elif retained:
            raise ReferenceIntegrityDurableError(
                f"non-fresh clip {clip_uid!r} must not retain review entities"
            )
        if classification != CLIP_FRESH_TARGET:
            return

        marker = _read_json(self._clip_outcome_path(shard, clip_uid))
        if marker is not None:
            # The clip outcome marker is the durable terminal authority: verify
            # the live publication against it. Live state may legitimately equal
            # the frozen baseline here, because a re-run can republish the very
            # same failed state (same clip, same reason) as a no-op write.
            self._verify_published_clip(shard, storage, clip_uid, entry)
            return
        # No marker yet: the live state decides whether this epoch has published
        # anything. Equality with the frozen baseline means it has not, even when
        # that baseline already carries a failed Reference Integrity state
        # (legacy re-runs a failed clip as a fresh target). Any other state is a
        # publication-before-marker crash candidate that must verify exactly.
        baselines = self._live_baselines(clip)
        if all(entry.get(key) == value for key, value in baselines.items()):
            return
        self._verify_published_clip(shard, storage, clip_uid, entry)

    def _validate_existing_plan(
        self, shard: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The single strict validation of one frozen plan, used by every reader.

        The seed path and the reconcile path must never disagree about what a
        valid plan is, so both go through this helper.
        """
        expected = {
            "schema": REFERENCE_INTEGRITY_PLAN_SCHEMA,
            "canonical_shard": shard,
            "eligible_clip_uids": list(self.eligible.get(shard, ())),
            "policy": self._policy_identity(),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity plan {key} drifted for {shard!r}"
                )
        clips = payload.get("clips")
        if not isinstance(clips, dict):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity plan clips map is malformed for "
                f"{shard!r}"
            )
        if set(clips) != set(self.eligible.get(shard, ())):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity plan scope drifted for {shard!r}"
            )
        storage = self._storage_for(shard)
        for clip_uid, entry in clips.items():
            if not isinstance(entry, dict):
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity plan entry is malformed for "
                    f"{clip_uid!r}"
                )
            self._verify_plan_entry(shard, storage, clip_uid, entry)
        return dict(payload)

    def _plan(self, shard: str) -> dict[str, Any]:
        existing = _read_json(self._plan_path(shard))
        if existing is not None:
            return self._validate_existing_plan(shard, existing)

        storage = self._storage_for(shard)
        clips: dict[str, dict[str, Any]] = {}
        for clip_uid in self.eligible.get(shard, ()):
            clip = storage.read_clip(clip_uid)
            clips[clip_uid] = self._plan_entry(storage, clip)
        payload = {
            "schema": REFERENCE_INTEGRITY_PLAN_SCHEMA,
            "canonical_shard": shard,
            "eligible_clip_uids": list(self.eligible.get(shard, ())),
            "policy": self._policy_identity(),
            "clips": clips,
        }
        _write_json_once(self._plan_path(shard), payload)
        return payload

    def _existing_plan_for_reconcile(self, shard: str) -> dict[str, Any]:
        payload = _read_json(self._plan_path(shard))
        if payload is None:
            raise ReferenceIntegrityEpochError(
                f"reconcile_stats needs a frozen Reference Integrity plan for "
                f"{shard!r}"
            )
        return self._validate_existing_plan(shard, payload)

    # -- frozen review inputs --------------------------------------------------

    def _review_semantic_inputs(
        self,
        entity: Any,
        reference: Any,
        *,
        variant: str,
        source_context_sha256: str,
        final_reference_sha256: str,
        main_review_policy: Mapping[str, Any],
        reference_scope: str | None = None,
        synthetic: bool | None = None,
    ) -> dict[str, Any]:
        """Exactly the semantic inputs of one integrity review call.

        ``variant`` is explicit because the source-alpha call reviews a different
        reference and passes a literal ``synthetic=False`` while binding the very
        same model semantic policy.
        """
        return {
            "variant": variant,
            "entity_id": entity.entity_id,
            "reference_type": entity.reference_type,
            "phrase": entity.phrase,
            "grounding_prompt": entity.grounding_prompt,
            "reference_scope": (
                reference.reference_scope
                if reference_scope is None
                else str(reference_scope)
            ),
            "synthetic": (
                bool(reference.synthetic) if synthetic is None else bool(synthetic)
            ),
            "source_context_sha256": source_context_sha256,
            "final_reference_sha256": final_reference_sha256,
            "main_review_policy": dict(main_review_policy),
        }

    def _materialize_review_context(
        self,
        storage: RunStorage,
        clip_uid: str,
        entity_id: str,
        context: Any,
        *,
        variant: str = VARIANT_FINAL,
    ) -> tuple[str, str]:
        """Create-once Qwen context PNG with exact content validation.

        The context image is a model input, not a debug sidecar: an existing file
        whose bytes differ from the derived image is durable corruption. The path
        stays at the legacy location, which carries a variant suffix for the
        source-alpha call.
        """
        suffix = "" if variant == VARIANT_FINAL else f"_{variant}"
        path = storage.selected_path(
            clip_uid, f"integrity_context_{entity_id}{suffix}.png"
        )
        buffer = io.BytesIO()
        context.save(buffer, format="PNG")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ReferenceIntegrityDurableError(
                    f"frozen integrity context drifted: {path}"
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            try:
                temporary.write_bytes(data)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return storage.relative_artifact_path(path), digest

    def _review_input_anchor(
        self,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
        *,
        variant: str = VARIANT_FINAL,
        synthetic: bool | None = None,
    ) -> dict[str, Any]:
        """Derive the complete frozen review input of one variant."""
        evidence = _source_evidence(storage, clip_uid=clip_uid, reference=reference)
        context_path, context_sha256 = self._materialize_review_context(
            storage,
            clip_uid,
            entity.entity_id,
            _source_context(evidence),
            variant=variant,
        )
        reference_path = _resolve_run_artifact(storage, str(reference.image_path))
        reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if synthetic is not None:
            resolved_synthetic = bool(synthetic)
        elif variant == VARIANT_SOURCE_ALPHA:
            # Legacy passes a literal ``False`` here rather than reading the flag
            # off the alpha reference.
            resolved_synthetic = False
        else:
            resolved_synthetic = bool(reference.synthetic)
        policy = self._main_review_policy_identity()
        inputs = self._review_semantic_inputs(
            entity,
            reference,
            variant=variant,
            source_context_sha256=context_sha256,
            final_reference_sha256=reference_sha256,
            main_review_policy=policy,
            synthetic=resolved_synthetic,
        )
        return {
            "schema": REFERENCE_INTEGRITY_REVIEW_INPUT_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "variant": variant,
            "reference_type": entity.reference_type,
            "phrase": entity.phrase,
            "grounding_prompt": entity.grounding_prompt,
            "reference_scope": reference.reference_scope,
            "synthetic": resolved_synthetic,
            "source_clip_uid": evidence.source_clip_uid,
            "source_entity_id": evidence.source_entity_id,
            "source_frame_slot": evidence.frame_slot,
            "source_frame_index": evidence.source_frame_index,
            "source_context_path": context_path,
            "source_context_sha256": context_sha256,
            "final_reference_path": reference.image_path,
            "final_reference_sha256": reference_sha256,
            "main_review_policy": policy,
            "semantic_digest": semantic_input_digest(inputs),
        }

    def _frozen_review_input(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
        *,
        variant: str = VARIANT_FINAL,
    ) -> dict[str, Any]:
        """Create-once review input anchor, re-derived and compared exactly."""
        path = self._review_input_path(shard, clip_uid, entity.entity_id, variant)
        existing = _read_json(path)
        if existing is None:
            # First freeze. A broken reference or source evidence here is still
            # an ordinary legacy CPU failure, not durable corruption.
            expected = self._review_input_anchor(
                storage, clip_uid, entity, reference, variant=variant
            )
            _write_json_once(path, expected)
            return expected
        # The model input is already frozen, so failing to re-derive it can only
        # mean the frozen evidence changed underneath us.
        try:
            expected = self._review_input_anchor(
                storage, clip_uid, entity, reference, variant=variant
            )
        except ReferenceIntegrityDurableError:
            raise
        except Exception as exc:
            raise ReferenceIntegrityDurableError(
                f"cannot re-derive frozen Reference Integrity review input for "
                f"{clip_uid}/{entity.entity_id}"
            ) from exc
        if existing != expected:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity review input drifted for "
                f"{clip_uid}/{entity.entity_id}"
            )
        return existing

    @staticmethod
    def _frozen_source_alpha_reference(
        plan_entry: Mapping[str, Any], entity_id: str
    ) -> EntityReferenceState:
        """The exact source-alpha reference legacy would review, from frozen state.

        Legacy reads ``edit.source_reference`` off the ready Reference Edit state
        it already holds, so the frozen pre-edit is the only authority; live
        references and the Pairing state must never be used to guess it.
        """
        dump = plan_entry.get("pre_reference_edit")
        if not isinstance(dump, dict):
            raise ReferenceIntegrityDurableError(
                f"frozen source alpha needs a Reference Edit state for {entity_id!r}"
            )
        try:
            state = ReferenceEditState.model_validate(dump)
        except Exception as exc:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Edit state is invalid for {entity_id!r}"
            ) from exc
        edit_entity = _source_alpha_edit_entity(state, entity_id)
        if edit_entity is None:
            raise ReferenceIntegrityDurableError(
                f"frozen source alpha has no Reference Edit entity {entity_id!r}"
            )
        if (
            edit_entity.default_variant != "accepted_base"
            or edit_entity.source_reference.image_path is None
        ):
            raise ReferenceIntegrityDurableError(
                f"frozen source alpha prerequisites drifted for {entity_id!r}"
            )
        return edit_entity.source_reference

    def _expected_review_job(
        self,
        shard: str,
        entity: Any,
        reference: Any,
        anchor: Mapping[str, Any],
        *,
        variant: str = VARIANT_FINAL,
        dependencies: Mapping[str, str] | None = None,
    ) -> ModelJob:
        """The one deterministic review job for this entity and variant."""
        return ModelJob.create(
            job_type=REFERENCE_INTEGRITY_REVIEW_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=str(anchor["clip_uid"]),
            semantic_inputs=self._review_semantic_inputs(
                entity,
                reference,
                variant=variant,
                source_context_sha256=str(anchor["source_context_sha256"]),
                final_reference_sha256=str(anchor["final_reference_sha256"]),
                main_review_policy=dict(anchor["main_review_policy"]),
                reference_scope=str(anchor["reference_scope"]),
                synthetic=bool(anchor["synthetic"]),
            ),
            model_identity=(
                f"qwen:{self.config.qwen.reference_integrity_judge.model}"
            ),
            target={"entity_id": entity.entity_id, "variant": variant},
            dependencies=dependencies,
        )

    def _expected_review_inputs(
        self,
        *,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan_entry: Mapping[str, Any],
        entity: Any,
        reference: Any,
        variant: str,
        main_job: ModelJob | None = None,
    ) -> tuple[EntityReferenceState, dict[str, Any], ModelJob]:
        """Resolve, freeze and re-verify one variant's deterministic review job.

        Every reader of a review job identity goes through here, so the writer
        and all verifiers share exactly one derivation. For ``source_alpha`` the
        main review job is the dependency address, which makes the alpha job
        conditional on one exact main receipt instead of a standalone root job.
        """
        if variant == VARIANT_SOURCE_ALPHA and main_job is None:
            raise ReferenceIntegrityDurableError(
                "source alpha review requires its frozen main review job"
            )
        input_reference = (
            self._frozen_source_alpha_reference(plan_entry, entity.entity_id)
            if variant == VARIANT_SOURCE_ALPHA
            else reference
        )
        anchor = self._frozen_review_input(
            shard, storage, clip_uid, entity, input_reference, variant=variant
        )
        job = self._expected_review_job(
            shard,
            entity,
            input_reference,
            anchor,
            variant=variant,
            dependencies=(
                {MAIN_REVIEW_JOB_DEPENDENCY: main_job.job_id()}
                if main_job is not None
                else None
            ),
        )
        return input_reference, anchor, job

    # -- per-entity durable outcomes -------------------------------------------

    def _entity_outcome(
        self, shard: str, clip_uid: str, entity_id: str
    ) -> dict[str, Any] | None:
        """Strictly validated CPU entity outcome, or ``None`` if unresolved."""
        marker = _read_json(self._entity_outcome_path(shard, clip_uid, entity_id))
        if marker is None:
            return None
        label = f"{clip_uid}/{entity_id}"
        if marker.get("schema") != REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA:
            raise ReferenceIntegrityDurableError(
                f"entity outcome schema drifted for {label}"
            )
        if marker.get("clip_uid") != clip_uid or marker.get("entity_id") != entity_id:
            raise ReferenceIntegrityDurableError(
                f"entity outcome identity drifted for {label}"
            )
        outcome = marker.get("outcome")
        if outcome not in CPU_ENTITY_OUTCOMES + REVIEWED_ENTITY_OUTCOMES:
            raise ReferenceIntegrityDurableError(
                f"entity outcome {outcome!r} is not a known branch for {label}"
            )
        if marker.get("status") != outcome:
            raise ReferenceIntegrityDurableError(
                f"entity outcome status disagrees with its branch for {label}"
            )
        reviewed = marker.get("reviewed")
        if not isinstance(reviewed, bool):
            raise ReferenceIntegrityDurableError(
                f"entity outcome reviewed flag is not a bool for {label}"
            )
        if reviewed and outcome not in REVIEWED_ENTITY_OUTCOMES:
            raise ReferenceIntegrityDurableError(
                f"reviewed entity outcome {outcome!r} is impossible for {label}"
            )
        if not reviewed and outcome not in CPU_ENTITY_OUTCOMES:
            raise ReferenceIntegrityDurableError(
                f"CPU entity outcome {outcome!r} is impossible for {label}"
            )
        present = [key for key in REVIEWED_MARKER_FIELDS if key in marker]
        bbox_present = [key for key in BBOX_MARKER_FIELDS if key in marker]
        if reviewed:
            if len(present) != len(REVIEWED_MARKER_FIELDS):
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome is missing provenance for {label}"
                )
            review_variant = marker.get("review_variant")
            if review_variant not in IMPLEMENTED_REVIEW_VARIANTS:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome variant is unknown for {label}"
                )
            parent_job_id = marker.get("parent_review_job_id")
            if review_variant == VARIANT_FINAL:
                if parent_job_id is not None:
                    raise ReferenceIntegrityDurableError(
                        f"main reviewed entity outcome must not carry a parent job "
                        f"for {label}"
                    )
            elif not isinstance(parent_job_id, str) or not parent_job_id:
                raise ReferenceIntegrityDurableError(
                    f"source alpha entity outcome has no parent job for {label}"
                )
            job_id = marker.get("review_job_id")
            if not isinstance(job_id, str) or not job_id:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome has no review job id for {label}"
                )
            context_path = marker.get("source_context_path")
            if not isinstance(context_path, str) or not context_path:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome has no context path for {label}"
                )
            judge_failed = marker.get("judge_failed")
            if not isinstance(judge_failed, bool):
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome judge_failed is not a bool for {label}"
                )
            if bbox_present:
                if len(bbox_present) != len(BBOX_MARKER_FIELDS):
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome is missing provenance for "
                        f"{label}"
                    )
                bbox_job_id = marker.get("bbox_review_job_id")
                if not isinstance(bbox_job_id, str) or not bbox_job_id:
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome has no bbox job id for "
                        f"{label}"
                    )
                if marker.get("source_bbox_fallback_trigger") != BBOX_TRIGGER_ARTIFACT:
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome trigger is unknown for {label}"
                    )
                for key in (
                    "source_bbox_fallback_candidate_path",
                    "source_bbox_fallback_metadata_path",
                ):
                    value = marker.get(key)
                    if not isinstance(value, str) or not value:
                        raise ReferenceIntegrityDurableError(
                            f"source bbox entity outcome {key} is missing for "
                            f"{label}"
                        )
                xyxy = marker.get("source_bbox_xyxy")
                if (
                    not isinstance(xyxy, list)
                    or len(xyxy) != 4
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in xyxy
                    )
                ):
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome bbox is malformed for {label}"
                    )
                bbox_judge_failed = marker.get("bbox_judge_failed")
                if not isinstance(bbox_judge_failed, bool):
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome bbox_judge_failed is not a "
                        f"bool for {label}"
                    )
                bbox_review = marker.get("source_bbox_fallback_review")
                if bbox_judge_failed:
                    if bbox_review is not None:
                        raise ReferenceIntegrityDurableError(
                            f"judge-failed source bbox outcome must not carry a "
                            f"review for {label}"
                        )
                else:
                    try:
                        SourceBboxFallbackReview.model_validate(bbox_review)
                    except Exception as exc:
                        raise ReferenceIntegrityDurableError(
                            f"source bbox entity outcome has an invalid review for "
                            f"{label}"
                        ) from exc
            review = marker.get("review")
            if judge_failed:
                if review is not None:
                    raise ReferenceIntegrityDurableError(
                        f"judge-failed entity outcome must not carry a review for "
                        f"{label}"
                    )
            else:
                if not isinstance(review, dict):
                    raise ReferenceIntegrityDurableError(
                        f"reviewed entity outcome has no review for {label}"
                    )
                # Structural fail-closed layer: a shape-valid but content-less
                # review (for example {}) must never reach the reason comparison
                # as a bare ``marker["review"]["reason"]`` KeyError.
                try:
                    validated_review = ReferenceIntegrityReview.model_validate(review)
                except Exception as exc:
                    raise ReferenceIntegrityDurableError(
                        f"reviewed entity outcome has an invalid review for {label}"
                    ) from exc
        elif present or bbox_present:
            raise ReferenceIntegrityDurableError(
                f"CPU entity outcome must not carry review provenance for {label}"
            )
        reason = marker.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ReferenceIntegrityDurableError(
                f"entity outcome reason is missing for {label}"
            )
        diagnostics = marker.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise ReferenceIntegrityDurableError(
                f"entity outcome diagnostics are missing for {label}"
            )
        final_path = marker.get("final_reference_path")
        if not isinstance(final_path, str) or not final_path:
            raise ReferenceIntegrityDurableError(
                f"entity outcome final reference is missing for {label}"
            )
        delta = marker.get("delta")
        if not isinstance(delta, dict) or not delta:
            raise ReferenceIntegrityDurableError(
                f"entity outcome delta is missing for {label}"
            )
        for key, value in delta.items():
            if key not in ENTITY_DELTA_FIELDS:
                raise ReferenceIntegrityDurableError(
                    f"entity outcome delta field {key!r} is unknown for {label}"
                )
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReferenceIntegrityDurableError(
                    f"entity outcome delta {key} is not a non-negative int for "
                    f"{label}"
                )
        policy_reason = marker.get("semantic_policy_reason")
        if reviewed:
            if policy_reason is not None:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome must not carry a semantic policy "
                    f"reason for {label}"
                )
            if marker["judge_failed"]:
                prefix = (
                    SOURCE_ALPHA_JUDGE_FAILED_PREFIX
                    if review_variant == VARIANT_SOURCE_ALPHA
                    else MAIN_REVIEW_JUDGE_FAILED_PREFIX
                )
                if not reason.startswith(prefix):
                    raise ReferenceIntegrityDurableError(
                        f"judge-failed entity outcome reason drifted for {label}"
                    )
            elif bbox_present:
                if marker["bbox_judge_failed"]:
                    if not reason.startswith(BBOX_JUDGE_FAILED_PREFIX):
                        raise ReferenceIntegrityDurableError(
                            f"source bbox judge-failed entity outcome reason "
                            f"drifted for {label}"
                        )
                elif reason != marker["source_bbox_fallback_review"]["reason"]:
                    raise ReferenceIntegrityDurableError(
                        f"source bbox entity outcome reason disagrees with its "
                        f"review for {label}"
                    )
            elif (
                review_variant == VARIANT_SOURCE_ALPHA
                and outcome == ENTITY_OUTCOME_ACCEPTED
            ):
                # Legacy does not reuse the alpha review's own reason here.
                if validated_review.verdict != "accept":
                    raise ReferenceIntegrityDurableError(
                        f"source alpha accepted outcome needs an accept review for "
                        f"{label}"
                    )
                if reason != SOURCE_ALPHA_ACCEPTED_REASON:
                    raise ReferenceIntegrityDurableError(
                        f"source alpha entity outcome reason drifted for {label}"
                    )
            elif reason != validated_review.reason:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome reason disagrees with its review for "
                    f"{label}"
                )
        elif outcome == ENTITY_OUTCOME_REJECTED:
            if not isinstance(policy_reason, str) or not policy_reason:
                raise ReferenceIntegrityDurableError(
                    f"rejected entity outcome needs a semantic policy reason for "
                    f"{label}"
                )
            if reason != policy_reason:
                raise ReferenceIntegrityDurableError(
                    f"rejected entity outcome reason disagrees with its policy "
                    f"reason for {label}"
                )
        elif policy_reason is not None:
            raise ReferenceIntegrityDurableError(
                f"skipped entity outcome must not carry a semantic policy reason "
                f"for {label}"
            )
        return marker

    def _write_entity_outcome(
        self, shard: str, clip_uid: str, entity_id: str, payload: Mapping[str, Any]
    ) -> None:
        _write_json_once(
            self._entity_outcome_path(shard, clip_uid, entity_id),
            {
                "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                **payload,
            },
        )

    def _entity_branch(
        self, entity: Any, reference: Any, diagnostics: Any
    ) -> _EntityBranch:
        """Exactly the legacy CPU-only branching for one retained entity."""
        topology_delta = {"topology_suspicious": int(diagnostics.suspicious)}
        policy_reason = reference_semantic_hard_reject_reason(
            reference_type=entity.reference_type,
            phrase=entity.phrase,
        )
        if policy_reason is not None:
            return _EntityBranch(
                outcome=ENTITY_OUTCOME_REJECTED,
                status="rejected",
                reason=policy_reason,
                reviewed=False,
                semantic_policy_reason=policy_reason,
                diagnostics=diagnostics,
                delta={
                    **topology_delta,
                    "entities_rejected": 1,
                    "semantic_policy_rejected": 1,
                },
            )
        semantic_risk = reference_semantic_risk_reason(
            reference_type=entity.reference_type,
            phrase=entity.phrase,
            grounding_prompt=entity.grounding_prompt,
        )
        requires_review = bool(
            reference.synthetic
            or reference.reference_scope == "local"
            or reference.source_bbox_fallback
            or diagnostics.suspicious
            or semantic_risk is not None
        )
        if requires_review:
            return _EntityBranch(
                outcome=None,
                status=None,
                reason=None,
                reviewed=False,
                semantic_policy_reason=None,
                diagnostics=diagnostics,
                requires_model=True,
            )
        return _EntityBranch(
            outcome=ENTITY_OUTCOME_SKIPPED,
            status="skipped",
            reason=CLEAN_REAL_FULL_REFERENCE,
            reviewed=False,
            semantic_policy_reason=None,
            diagnostics=diagnostics,
            delta={
                **topology_delta,
                "entities_skipped_review": 1,
                "entities_accepted": 1,
            },
        )

    def _expected_entity_marker(
        self,
        clip_uid: str,
        entity: Any,
        reference: Any,
        diagnostics: Any,
    ) -> dict[str, Any] | None:
        """The exact durable marker this entity must have, or ``None`` for 6c/6d.

        One helper serves both the writer and the verifier so the branch policy
        exists exactly once.
        """
        branch = self._entity_branch(entity, reference, diagnostics)
        if branch.requires_model:
            return None
        return {
            "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "outcome": branch.outcome,
            "status": branch.status,
            "reason": branch.reason,
            "reviewed": branch.reviewed,
            "semantic_policy_reason": branch.semantic_policy_reason,
            "final_reference_path": reference.image_path,
            "diagnostics": branch.diagnostics.model_dump(mode="json"),
            "delta": branch.delta,
        }

    def _advance_entity(
        self,
        shard: str,
        storage: RunStorage,
        clip: Any,
        entity: Any,
        reference: Any,
    ) -> list[ModelJob]:
        """Terminal CPU marker, or the deterministic main review job to run."""
        review_input_path = self._review_input_path(
            shard, clip.clip_uid, entity.entity_id
        )
        if _read_json(review_input_path) is not None:
            # A durable review input already proves this entity entered the main
            # review, so re-verify the whole frozen model input before re-issuing
            # the same deterministic job. Without this fast path a missing final
            # reference would surface as an ordinary CPU failure instead of
            # frozen-input corruption.
            anchor = self._frozen_review_input(
                shard, storage, clip.clip_uid, entity, reference
            )
            return [self._expected_review_job(shard, entity, reference, anchor)]
        final_image = _load_reference_image(storage, str(reference.image_path))
        diagnostics = reference_topology_diagnostics(final_image)
        expected = self._expected_entity_marker(
            clip.clip_uid, entity, reference, diagnostics
        )
        if expected is not None:
            _write_json_once(
                self._entity_outcome_path(shard, clip.clip_uid, entity.entity_id),
                expected,
            )
            return []
        # The entity needs the main review: freeze its full input first, so the
        # job identity and its provenance are durable before any model call.
        anchor = self._frozen_review_input(
            shard, storage, clip.clip_uid, entity, reference
        )
        return [self._expected_review_job(shard, entity, reference, anchor)]

    # -- review execution ------------------------------------------------------

    def _review_variant(self, job: ModelJob) -> str:
        """The implemented review variant of one job, or fail closed."""
        if job.job_type != REFERENCE_INTEGRITY_REVIEW_JOB:
            raise ReferenceIntegrityEpochError(
                f"unsupported job type {job.job_type!r}"
            )
        variant = str(dict(job.target).get("variant", ""))
        if variant not in IMPLEMENTED_REVIEW_VARIANTS:
            raise ReferenceIntegrityEpochError(
                f"unsupported review variant {variant!r}"
            )
        return variant

    def _job_entity_reference(
        self, job: ModelJob
    ) -> tuple[str, RunStorage, Any, Any, Any]:
        """Resolve one review job back to its live entity and pre-reference."""
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip = storage.read_clip(job.clip_uid)
        entity_id = str(dict(job.target).get("entity_id", ""))
        entity = next(
            (item for item in clip.annotation.entities if item.entity_id == entity_id),
            None,
        )
        reference = next(
            (item for item in clip.references.entities if item.entity_id == entity_id),
            None,
        )
        if entity is None or reference is None:
            raise ReferenceIntegrityDurableError(
                f"review job {job.job_id()} addresses an unknown entity "
                f"{entity_id!r} of clip {job.clip_uid!r}"
            )
        return shard, storage, clip, entity, reference

    def _job_review_context(
        self, job: ModelJob
    ) -> tuple[str, RunStorage, Any, Any, Any, dict[str, Any], Any]:
        """Resolve one review job to its live entity and its frozen plan entry."""
        shard, storage, clip, entity, reference = self._job_entity_reference(job)
        plan_entry = self._existing_plan_for_reconcile(shard)["clips"][job.clip_uid]
        pre_edit_dump = plan_entry.get("pre_reference_edit")
        pre_edit = (
            ReferenceEditState.model_validate(pre_edit_dump)
            if pre_edit_dump is not None
            else None
        )
        return shard, storage, clip, entity, reference, plan_entry, pre_edit

    def _committed_review_payload(self, job: ModelJob) -> dict[str, Any]:
        """The committed receipt payload of one review job, or fail closed."""
        from r2v_data_v2.v3.post_mask_epoch_state import STATE_MISMATCH

        entity_id = str(dict(job.target).get("entity_id", ""))
        state = self.ledger.classify(job)
        if state.state == STATE_MISMATCH:
            raise ReferenceIntegrityDurableError(
                f"review receipt mismatch for {job.clip_uid}/{entity_id}: "
                f"{state.detail or ''}"
            )
        if not state.skippable:
            raise ReferenceIntegrityEpochError(
                f"review job {job.job_id()} is not terminal; the stage is "
                "incomplete"
            )
        result = self.ledger.load_committed_result(job)
        if result is None:
            raise ReferenceIntegrityDurableError(
                f"committed review job {job.job_id()} has no replayable result"
            )
        return dict(result.payload)

    @staticmethod
    def _review_continuation(
        *,
        clip_uid: str,
        entity: Any,
        reference: Any,
        diagnostics: Any,
        review: ReferenceIntegrityReview,
        pre_edit: ReferenceEditState | None,
    ) -> str | None:
        """Which continuation the frozen legacy policy takes, if any.

        Mirrors the legacy order exactly: the source-alpha completion fallback is
        decided first, then the source-bbox routes. A non-``None`` result means
        this epoch must leave the entity unresolved for 6d1/6d2.
        """
        if review.verdict == "reject" and reference.synthetic:
            edit_entity = _source_alpha_edit_entity(pre_edit, entity.entity_id)
            if (
                edit_entity is not None
                and edit_entity.default_variant == "accepted_base"
                and edit_entity.source_reference.image_path is not None
            ):
                return "source_alpha"
        if _topology_bbox_upgrade_eligible(
            review=review,
            reference_type=entity.reference_type,
            reference=reference,
            clip_uid=clip_uid,
            diagnostics=diagnostics,
        ):
            return "topology_bbox"
        if (
            _artifact_only_bbox_eligible(review)
            and not reference.source_bbox_fallback
            and _is_self_sourced_reference(clip_uid=clip_uid, reference=reference)
        ):
            return "artifact_bbox"
        return None

    def _review_diagnostics(self, storage: RunStorage, reference: Any) -> Any:
        return reference_topology_diagnostics(
            _load_reference_image(storage, str(reference.image_path))
        )

    def _expected_reviewed_marker(
        self,
        *,
        clip_uid: str,
        entity: Any,
        reference: Any,
        diagnostics: Any,
        job: ModelJob,
        payload: Mapping[str, Any],
        anchor: Mapping[str, Any],
        pre_edit: ReferenceEditState | None,
    ) -> dict[str, Any] | None:
        """The exact durable marker one committed main review must produce.

        ``None`` means a continuation still owns this entity. One helper serves
        the finalizer and the restart verifier so the policy exists exactly once.
        """
        common_delta = {"topology_suspicious": int(diagnostics.suspicious)}
        if str(payload.get("status")) == "judge_failed":
            return {
                "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "entity_id": entity.entity_id,
                "outcome": ENTITY_OUTCOME_REJECTED,
                "status": ENTITY_OUTCOME_REJECTED,
                "reason": f"{MAIN_REVIEW_JUDGE_FAILED_PREFIX}{payload.get('error')}",
                "reviewed": True,
                "semantic_policy_reason": None,
                "final_reference_path": reference.image_path,
                "diagnostics": diagnostics.model_dump(mode="json"),
                "review_job_id": job.job_id(),
                "review_variant": VARIANT_FINAL,
                "parent_review_job_id": None,
                "source_context_path": str(anchor["source_context_path"]),
                "review": None,
                "judge_failed": True,
                "delta": {
                    **common_delta,
                    "entities_reviewed": 1,
                    "judge_failed": 1,
                    "entities_rejected": 1,
                },
            }
        review = ReferenceIntegrityReview.model_validate(payload.get("review"))
        if (
            self._review_continuation(
                clip_uid=clip_uid,
                entity=entity,
                reference=reference,
                diagnostics=diagnostics,
                review=review,
                pre_edit=pre_edit,
            )
            is not None
        ):
            return None
        accepted = review.verdict == "accept"
        status = ENTITY_OUTCOME_ACCEPTED if accepted else ENTITY_OUTCOME_REJECTED
        return {
            "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "outcome": status,
            "status": status,
            "reason": review.reason,
            "reviewed": True,
            "semantic_policy_reason": None,
            "final_reference_path": reference.image_path,
            "diagnostics": diagnostics.model_dump(mode="json"),
            "review_job_id": job.job_id(),
            "review_variant": VARIANT_FINAL,
            "parent_review_job_id": None,
            "source_context_path": str(anchor["source_context_path"]),
            "review": review.model_dump(mode="json"),
            "judge_failed": False,
            "delta": {
                **common_delta,
                "entities_reviewed": 1,
                **({"entities_accepted": 1} if accepted else {"entities_rejected": 1}),
            },
        }

    def _expected_source_alpha_outcome(
        self,
        *,
        clip_uid: str,
        entity: Any,
        plan_entry: Mapping[str, Any],
        main_job: ModelJob,
        main_anchor: Mapping[str, Any],
        main_diagnostics: Any,
        alpha_job: ModelJob,
        alpha_payload: Mapping[str, Any],
        alpha_reference: EntityReferenceState,
        alpha_diagnostics: Any,
        alpha_anchor: Mapping[str, Any],
        pre_edit: ReferenceEditState | None,
    ) -> tuple[dict[str, Any] | None, ReferenceIntegrityReview | None]:
        """The exact durable outcome one committed source-alpha review produces.

        The main receipt is validated here as well: a source-alpha outcome is a
        continuation of one exact main review, never a standalone decision. The
        delta carries the whole entity's legacy counters, so both reviews are
        accounted for. ``(None, review)`` means the frozen policy continues into
        the artifact bbox route and ``review`` is the alpha review that parented
        it; ``(None, None)`` never happens because this helper is only reached
        for a committed review.
        """
        entity_id = entity.entity_id
        main_payload = self._committed_review_payload(main_job)
        if str(main_payload.get("status")) != "review":
            raise ReferenceIntegrityDurableError(
                f"source alpha continuation needs a successful main review for "
                f"{clip_uid}/{entity_id}"
            )
        main_review = ReferenceIntegrityReview.model_validate(main_payload.get("review"))
        if (
            self._review_continuation(
                clip_uid=clip_uid,
                entity=entity,
                reference=_plan_pre_reference(plan_entry, entity_id),
                diagnostics=main_diagnostics,
                review=main_review,
                pre_edit=pre_edit,
            )
            != "source_alpha"
        ):
            raise ReferenceIntegrityDurableError(
                f"frozen main review no longer routes {clip_uid}/{entity_id} to "
                "source alpha"
            )
        alpha_status = str(alpha_payload.get("status", ""))
        if alpha_status not in {"review", "judge_failed"}:
            raise ReferenceIntegrityDurableError(
                f"committed source alpha job {alpha_job.job_id()} has an unknown "
                "payload status"
            )
        if alpha_status == "review":
            alpha_review = ReferenceIntegrityReview.model_validate(
                alpha_payload.get("review")
            )
        else:
            alpha_review = None
        common: dict[str, Any] = {
            "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity_id,
            "reviewed": True,
            "semantic_policy_reason": None,
            "final_reference_path": alpha_reference.image_path,
            "diagnostics": alpha_diagnostics.model_dump(mode="json"),
            "review_job_id": alpha_job.job_id(),
            "review_variant": VARIANT_SOURCE_ALPHA,
            "parent_review_job_id": main_job.job_id(),
            "source_context_path": str(alpha_anchor["source_context_path"]),
        }
        delta = {
            "topology_suspicious": int(main_diagnostics.suspicious)
            + int(alpha_diagnostics.suspicious),
            "entities_reviewed": 2,
        }
        if alpha_status == "judge_failed":
            return {
                **common,
                "outcome": ENTITY_OUTCOME_REJECTED,
                "status": ENTITY_OUTCOME_REJECTED,
                "reason": (
                    f"{SOURCE_ALPHA_JUDGE_FAILED_PREFIX}{alpha_payload.get('error')}"
                ),
                "review": None,
                "judge_failed": True,
                "delta": {**delta, "judge_failed": 1, "entities_rejected": 1},
            }, None
        assert alpha_review is not None
        if alpha_review.verdict == "reject":
            if self._artifact_bbox_route(
                clip_uid=clip_uid, reference=alpha_reference, review=alpha_review
            ):
                # The artifact bbox continuation owns this entity now: only the
                # alpha receipt stays durable here.
                return None, alpha_review
            return {
                **common,
                "outcome": ENTITY_OUTCOME_REJECTED,
                "status": ENTITY_OUTCOME_REJECTED,
                "reason": alpha_review.reason,
                "review": alpha_review.model_dump(mode="json"),
                "judge_failed": False,
                "delta": {**delta, "entities_rejected": 1},
            }, None
        return {
            **common,
            "outcome": ENTITY_OUTCOME_ACCEPTED,
            "status": ENTITY_OUTCOME_ACCEPTED,
            "reason": SOURCE_ALPHA_ACCEPTED_REASON,
            "review": alpha_review.model_dump(mode="json"),
            "judge_failed": False,
            "delta": {**delta, "entities_accepted": 1},
        }, None

    def _write_review_debug(
        self,
        storage: RunStorage,
        clip_uid: str,
        entity_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Legacy debug sidecar. Execution-only, never a semantic authority."""
        if not self.config.debug.save_diagnostics:
            return
        path = storage.debug_path(clip_uid, f"reference_integrity_{entity_id}.json")
        if str(payload.get("status")) == "judge_failed":
            body: dict[str, Any] = {
                "judge_failed": True,
                "raw_responses": list(payload.get("raw_responses", [])),
                "finish_reasons": list(payload.get("finish_reasons", [])),
                "reason": str(payload.get("error")),
            }
        else:
            body = {
                "review": dict(payload.get("review") or {}),
                "raw_response": str(payload.get("raw_response")),
                "raw_responses": list(payload.get("raw_responses", [])),
                "finish_reasons": list(payload.get("finish_reasons", [])),
            }
        write_json_atomic(path, body)

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        """One review call. No prompt or retry policy is copied."""
        if job.job_type == REFERENCE_INTEGRITY_BBOX_REVIEW_JOB:
            return self._run_bbox_review(job, handle)
        variant = self._review_variant(job)
        shard, storage, _clip, entity, reference, plan_entry, _pre_edit = (
            self._job_review_context(job)
        )
        main_job: ModelJob | None = None
        if variant == VARIANT_SOURCE_ALPHA:
            _r, _a, main_job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=job.clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_FINAL,
            )
        _input_reference, anchor, expected = self._expected_review_inputs(
            shard=shard,
            storage=storage,
            clip_uid=job.clip_uid,
            plan_entry=plan_entry,
            entity=entity,
            reference=reference,
            variant=variant,
            main_job=main_job,
        )
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != str(anchor["semantic_digest"])
            or job.model_identity != expected.model_identity
        ):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Integrity review semantic identity drifted",
            )
        resolved = resolve_reference_integrity_judge(handle, self.config)
        try:
            with Image.open(
                _resolve_run_artifact(storage, str(anchor["source_context_path"]))
            ) as opened:
                opened.load()
                source_context = opened.copy()
            attempt = resolved.judge.review(
                source_context=source_context,
                final_reference=_load_reference_image(
                    storage, str(anchor["final_reference_path"])
                ),
                reference_type=entity.reference_type,
                phrase=entity.phrase,
                grounding_prompt=entity.grounding_prompt,
                reference_scope=str(anchor["reference_scope"]),
                synthetic=bool(anchor["synthetic"]),
            )
        except ReferenceIntegrityJudgeFailure as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "judge_failed",
                    "error": str(exc),
                    "raw_responses": list(exc.raw_responses),
                    "finish_reasons": list(exc.finish_reasons),
                },
            )
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "review": attempt.review.model_dump(mode="json"),
                "raw_response": attempt.raw_response,
                "raw_responses": list(attempt.raw_responses),
                "finish_reasons": list(attempt.finish_reasons),
            },
        )

    def _review_outcome(
        self,
        *,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
        plan_entry: Mapping[str, Any],
        pre_edit: ReferenceEditState | None,
        job: ModelJob,
        payload: Mapping[str, Any],
    ) -> _ReviewOutcome:
        """Everything one committed review job means, from frozen state.

        Returns the exact durable marker (``None`` while a continuation still
        owns the entity) plus the model job that continuation unlocks, if any.
        This is the one policy point shared by the finalizer and every restart
        verifier, so neither can drift from the other.

        The caller resolves the frozen context, so a verifier never re-reads the
        plan inside itself: publication verification already holds it.
        """
        variant = self._review_variant(job)
        main_reference, main_anchor, main_job = self._expected_review_inputs(
            shard=shard,
            storage=storage,
            clip_uid=clip_uid,
            plan_entry=plan_entry,
            entity=entity,
            reference=reference,
            variant=VARIANT_FINAL,
        )
        main_diagnostics = self._review_diagnostics(storage, main_reference)
        if variant == VARIANT_SOURCE_ALPHA:
            alpha_reference, alpha_anchor, alpha_job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_SOURCE_ALPHA,
                main_job=main_job,
            )
            if alpha_job.job_id() != job.job_id():
                raise ReferenceIntegrityDurableError(
                    f"source alpha job id drifted for {clip_uid}/{entity.entity_id}"
                )
            marker, _alpha_review = self._expected_source_alpha_outcome(
                clip_uid=clip_uid,
                entity=entity,
                plan_entry=plan_entry,
                main_job=main_job,
                main_anchor=main_anchor,
                main_diagnostics=main_diagnostics,
                alpha_job=alpha_job,
                alpha_payload=payload,
                alpha_reference=alpha_reference,
                alpha_diagnostics=self._review_diagnostics(storage, alpha_reference),
                alpha_anchor=alpha_anchor,
                pre_edit=pre_edit,
            )
            if marker is not None:
                return _ReviewOutcome(marker=marker, unlock=None)
            # The source-alpha review routes into the artifact source bbox.
            _parent, _anchor, bbox_job = self._bbox_context(
                shard=shard,
                storage=storage,
                clip_uid=clip_uid,
                entity=entity,
                pre_reference=reference,
                plan_entry=plan_entry,
                pre_edit=pre_edit,
                parent_variant=VARIANT_SOURCE_ALPHA,
            )
            return _ReviewOutcome(marker=None, unlock=bbox_job)
        if main_job.job_id() != job.job_id():
            raise ReferenceIntegrityDurableError(
                f"review job {job.job_id()} is not the frozen job for "
                f"{clip_uid}/{entity.entity_id}"
            )
        marker = self._expected_reviewed_marker(
            clip_uid=clip_uid,
            entity=entity,
            reference=main_reference,
            diagnostics=main_diagnostics,
            job=main_job,
            payload=payload,
            anchor=main_anchor,
            pre_edit=pre_edit,
        )
        if marker is not None:
            return _ReviewOutcome(marker=marker, unlock=None)
        review = ReferenceIntegrityReview.model_validate(payload.get("review"))
        continuation = self._review_continuation(
            clip_uid=clip_uid,
            entity=entity,
            reference=main_reference,
            diagnostics=main_diagnostics,
            review=review,
            pre_edit=pre_edit,
        )
        if continuation == "source_alpha":
            _alpha_reference, _alpha_anchor, alpha_job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_SOURCE_ALPHA,
                main_job=main_job,
            )
            return _ReviewOutcome(marker=None, unlock=alpha_job)
        if continuation == "artifact_bbox":
            _parent, _anchor, bbox_job = self._bbox_context(
                shard=shard,
                storage=storage,
                clip_uid=clip_uid,
                entity=entity,
                pre_reference=reference,
                plan_entry=plan_entry,
                pre_edit=pre_edit,
            )
            return _ReviewOutcome(marker=None, unlock=bbox_job)
        # The topology upgrade route belongs to 6d2b.
        return _ReviewOutcome(marker=None, unlock=None)

    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """CPU continuation of one committed review. Never unlocks a job early."""
        if job.job_type == REFERENCE_INTEGRITY_BBOX_REVIEW_JOB:
            return self._finalize_bbox_review(job, result)
        variant = self._review_variant(job)
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        if status not in {"review", "judge_failed"}:
            raise ReferenceIntegrityDurableError(
                f"committed review job {job.job_id()} has an unknown payload status"
            )
        if status == "review":
            try:
                ReferenceIntegrityReview.model_validate(payload.get("review"))
            except Exception as exc:
                raise ReferenceIntegrityDurableError(
                    f"committed review job {job.job_id()} has an invalid review"
                ) from exc
        shard, storage, _clip, entity, reference, plan_entry, pre_edit = (
            self._job_review_context(job)
        )
        if variant == VARIANT_FINAL:
            # Only the main review owns the legacy debug sidecar; the alpha
            # continuation never overwrites it.
            self._write_review_debug(storage, job.clip_uid, entity.entity_id, payload)
        outcome = self._review_outcome(
            shard=shard,
            storage=storage,
            clip_uid=job.clip_uid,
            entity=entity,
            reference=reference,
            plan_entry=plan_entry,
            pre_edit=pre_edit,
            job=job,
            payload=payload,
        )
        if outcome.marker is None:
            # A continuation owns this entity; only the durable receipt is kept.
            return (outcome.unlock,) if outcome.unlock is not None else ()
        self._write_entity_outcome(
            shard, job.clip_uid, entity.entity_id, outcome.marker
        )
        self._publish_clip_if_terminal(shard, storage, job.clip_uid)
        return ()

    # -- artifact source bbox review -------------------------------------------

    def _bbox_semantic_inputs(
        self, *, entity: Any, parent: _BboxParent, anchor: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Exactly the semantic inputs of one artifact bbox review call."""
        return {
            "entity_id": entity.entity_id,
            "trigger": BBOX_TRIGGER_ARTIFACT,
            "parent_variant": parent.variant,
            "parent_review_job_id": parent.job.job_id(),
            "reference_type": entity.reference_type,
            "phrase": entity.phrase,
            "grounding_prompt": entity.grounding_prompt,
            "reference_scope": parent.reference.reference_scope,
            "source_context_sha256": str(anchor["source_context_sha256"]),
            "current_reference_sha256": str(anchor["current_reference_sha256"]),
            "candidate_sha256": str(anchor["candidate_sha256"]),
            "bbox_xyxy": list(anchor["bbox_xyxy"]),
            "bbox_review_policy": dict(anchor["bbox_review_policy"]),
        }

    def _bbox_input_anchor(
        self,
        *,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        parent: _BboxParent,
    ) -> dict[str, Any]:
        """Derive the frozen artifact bbox input from frozen evidence.

        Everything here is deterministic CPU work: the legacy candidate crop, the
        source frame/mask identity, the current reference and the parent review
        context. The legacy materializer itself is never re-run, because its
        metadata is a two-phase artifact that may already have advanced.
        """
        evidence = _source_evidence(
            storage, clip_uid=clip_uid, reference=parent.reference
        )
        crop_padding_ratio = float(self.config.pair.crop_padding_ratio)
        candidate_image, bbox_xyxy = _source_bbox_candidate(
            evidence, crop_padding_ratio=crop_padding_ratio
        )
        buffer = io.BytesIO()
        candidate_image.save(buffer, format="PNG")
        candidate_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
        current_reference_relative = str(parent.reference.image_path)
        current_reference_sha256 = hashlib.sha256(
            _resolve_run_artifact(storage, current_reference_relative).read_bytes()
        ).hexdigest()
        context_sha256 = hashlib.sha256(
            _resolve_run_artifact(storage, parent.context_path).read_bytes()
        ).hexdigest()
        policy = self._bbox_review_policy_identity()
        anchor: dict[str, Any] = {
            "schema": REFERENCE_INTEGRITY_BBOX_INPUT_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "trigger": BBOX_TRIGGER_ARTIFACT,
            "parent_variant": parent.variant,
            "parent_review_job_id": parent.job.job_id(),
            "reference_type": entity.reference_type,
            "phrase": entity.phrase,
            "grounding_prompt": entity.grounding_prompt,
            "reference_scope": parent.reference.reference_scope,
            "source_context_path": parent.context_path,
            "source_context_sha256": context_sha256,
            "current_reference_path": current_reference_relative,
            "current_reference_sha256": current_reference_sha256,
            "source_clip_uid": evidence.source_clip_uid,
            "source_entity_id": evidence.source_entity_id,
            "source_frame_slot": evidence.frame_slot,
            "source_frame_index": evidence.source_frame_index,
            "source_frame_path": storage.relative_artifact_path(
                evidence.frame_path
            ),
            "source_frame_sha256": hashlib.sha256(
                evidence.frame_path.read_bytes()
            ).hexdigest(),
            "source_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(evidence.mask.astype(np.uint8)).tobytes()
            ).hexdigest(),
            "bbox_xyxy": [int(value) for value in bbox_xyxy],
            "crop_padding_ratio": crop_padding_ratio,
            "candidate_path": storage.relative_artifact_path(
                storage.selected_path(
                    clip_uid, f"source_bbox_fallback_{entity.entity_id}.png"
                )
            ),
            "candidate_sha256": candidate_sha256,
            "metadata_path": storage.relative_artifact_path(
                storage.selected_path(
                    clip_uid, f"source_bbox_fallback_{entity.entity_id}.json"
                )
            ),
            "parent_integrity_review": parent.review.model_dump(mode="json"),
            "bbox_review_policy": policy,
            "semantic_digest": "",
        }
        anchor["semantic_digest"] = semantic_input_digest(
            self._bbox_semantic_inputs(entity=entity, parent=parent, anchor=anchor)
        )
        return anchor

    @staticmethod
    def _bbox_anchor_projection(anchor: Mapping[str, Any]) -> dict[str, Any]:
        """Everything in an anchor that must be recomputable from frozen state."""
        return {key: value for key, value in anchor.items() if key != "base_metadata"}

    @staticmethod
    def _verify_frozen_base_metadata(anchor: Mapping[str, Any]) -> None:
        """Tie the frozen legacy materialized metadata to the derived anchor.

        The materialized metadata is legacy authority, stored verbatim; this
        check proves it still describes exactly the artifact the anchor froze.
        """
        base = anchor.get("base_metadata")
        label = f"{anchor.get('clip_uid')}/{anchor.get('entity_id')}"
        if not isinstance(base, dict):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity bbox base metadata is missing for "
                f"{label}"
            )
        expected = {
            "mode": BBOX_REVIEW_MODE,
            "trigger": f"{BBOX_TRIGGER_ARTIFACT}_v1",
            "status": "materialized",
            "review_status": "not_reviewed",
            "synthetic": False,
            "raw_source_pixels_only": True,
            "clip_uid": anchor["clip_uid"],
            "entity_id": anchor["entity_id"],
            "source_clip_uid": anchor["source_clip_uid"],
            "source_entity_id": anchor["source_entity_id"],
            "source_frame_slot": anchor["source_frame_slot"],
            "source_frame_index": anchor["source_frame_index"],
            "source_frame_path": anchor["source_frame_path"],
            "source_frame_sha256": anchor["source_frame_sha256"],
            "source_mask_sha256": anchor["source_mask_sha256"],
            "bbox_xyxy": list(anchor["bbox_xyxy"]),
            "crop_padding_ratio": anchor["crop_padding_ratio"],
            "candidate_path": anchor["candidate_path"],
            "candidate_sha256": anchor["candidate_sha256"],
            "original_reference_path": anchor["current_reference_path"],
            "original_reference_sha256": anchor["current_reference_sha256"],
            "original_integrity_review": anchor["parent_integrity_review"],
            "failed_reference_path": anchor["current_reference_path"],
            "failed_reference_sha256": anchor["current_reference_sha256"],
        }
        for key, value in expected.items():
            if base.get(key) != value:
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity bbox base metadata {key} drifted "
                    f"for {label}"
                )

    def _materialize_bbox_input(
        self, *, storage: RunStorage, clip_uid: str, entity: Any, parent: _BboxParent
    ) -> dict[str, Any]:
        """First freeze: let the legacy materializer produce both artifacts.

        Only the candidate crop and the base metadata come from legacy; the model
        call is this epoch's own job. The materializer runs exactly once per
        entity because no job can exist before its anchor does.
        """
        metadata_path = storage.selected_path(
            clip_uid, f"source_bbox_fallback_{entity.entity_id}.json"
        )
        if metadata_path.is_file():
            stale = _read_json(metadata_path)
            if stale is not None and stale.get("status") != "materialized":
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity bbox metadata drifted: "
                    f"{metadata_path}"
                )
        evaluation = _materialize_source_bbox(
            storage=storage,
            clip_uid=clip_uid,
            entity_id=entity.entity_id,
            source_evidence=_source_evidence(
                storage, clip_uid=clip_uid, reference=parent.reference
            ),
            current_reference=_load_reference_image(
                storage, str(parent.reference.image_path)
            ),
            current_reference_path=_resolve_run_artifact(
                storage, str(parent.reference.image_path)
            ),
            reference=parent.reference,
            original_review=parent.review,
            diagnostics=parent.diagnostics,
            crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
            trigger=BBOX_TRIGGER_ARTIFACT,
        )
        base = _read_json(
            _resolve_run_artifact(storage, evaluation.metadata_relative)
        )
        if base is None:
            raise ReferenceIntegrityDurableError(
                f"legacy source bbox metadata is missing: "
                f"{evaluation.metadata_relative}"
            )
        anchor = self._bbox_input_anchor(
            storage=storage, clip_uid=clip_uid, entity=entity, parent=parent
        )
        anchor["base_metadata"] = base
        self._verify_frozen_base_metadata(anchor)
        return anchor

    def _frozen_bbox_input(
        self,
        *,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        parent: _BboxParent,
    ) -> dict[str, Any]:
        """Create-once bbox input anchor, re-derived and compared exactly."""
        path = self._bbox_input_path(shard, clip_uid, entity.entity_id)
        existing = _read_json(path)
        if existing is None:
            anchor = self._materialize_bbox_input(
                storage=storage, clip_uid=clip_uid, entity=entity, parent=parent
            )
            _write_json_once(path, anchor)
            return anchor
        # The input is frozen, so neither the candidate nor any of its evidence
        # may be re-materialized: re-derive and compare instead.
        try:
            expected = self._bbox_input_anchor(
                storage=storage, clip_uid=clip_uid, entity=entity, parent=parent
            )
        except ReferenceIntegrityDurableError:
            raise
        except Exception as exc:
            raise ReferenceIntegrityDurableError(
                f"cannot re-derive frozen Reference Integrity bbox input for "
                f"{clip_uid}/{entity.entity_id}"
            ) from exc
        if self._bbox_anchor_projection(existing) != expected:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity bbox input drifted for "
                f"{clip_uid}/{entity.entity_id}"
            )
        self._verify_frozen_base_metadata(existing)
        try:
            candidate = _resolve_run_artifact(storage, str(existing["candidate_path"]))
            candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except ReferenceIntegrityDurableError:
            raise
        except Exception as exc:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity bbox candidate is unreadable for "
                f"{clip_uid}/{entity.entity_id}"
            ) from exc
        if candidate_sha256 != str(existing["candidate_sha256"]):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity bbox candidate drifted for "
                f"{clip_uid}/{entity.entity_id}"
            )
        return existing

    def _expected_bbox_job(
        self,
        *,
        shard: str,
        entity: Any,
        parent: _BboxParent,
        anchor: Mapping[str, Any],
    ) -> ModelJob:
        """The deterministic artifact bbox job of one frozen chain."""
        return ModelJob.create(
            job_type=REFERENCE_INTEGRITY_BBOX_REVIEW_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=str(anchor["clip_uid"]),
            semantic_inputs=self._bbox_semantic_inputs(
                entity=entity, parent=parent, anchor=anchor
            ),
            model_identity=(
                f"qwen:{self.config.qwen.reference_integrity_judge.model}"
            ),
            target={
                "entity_id": entity.entity_id,
                "trigger": BBOX_TRIGGER_ARTIFACT,
                "parent_variant": parent.variant,
            },
            dependencies={PARENT_REVIEW_JOB_DEPENDENCY: parent.job.job_id()},
        )

    @staticmethod
    def _artifact_bbox_route(
        *,
        clip_uid: str,
        reference: EntityReferenceState,
        review: ReferenceIntegrityReview,
    ) -> bool:
        """Legacy's artifact-only source bbox route, exactly."""
        return (
            review.verdict == "reject"
            and _artifact_only_bbox_eligible(review)
            and not reference.source_bbox_fallback
            and _is_self_sourced_reference(clip_uid=clip_uid, reference=reference)
        )

    def _bbox_parent(
        self,
        *,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        pre_reference: EntityReferenceState,
        plan_entry: Mapping[str, Any],
        pre_edit: ReferenceEditState | None,
    ) -> _BboxParent:
        """Re-derive the whole frozen artifact-bbox parent chain.

        For a final parent the main receipt must route to the artifact bbox; for
        a source-alpha parent the main receipt must route to source alpha, the
        alpha receipt must exist and succeed, and only then may the alpha review
        route to the artifact bbox. Nothing about the parent is guessed from live
        state.
        """
        label = f"{clip_uid}/{entity.entity_id}"
        main_reference, main_anchor, main_job = self._expected_review_inputs(
            shard=shard,
            storage=storage,
            clip_uid=clip_uid,
            plan_entry=plan_entry,
            entity=entity,
            reference=pre_reference,
            variant=VARIANT_FINAL,
        )
        main_payload = self._committed_review_payload(main_job)
        if str(main_payload.get("status")) != "review":
            raise ReferenceIntegrityDurableError(
                f"artifact bbox continuation needs a successful main review for "
                f"{label}"
            )
        main_review = ReferenceIntegrityReview.model_validate(main_payload.get("review"))
        main_diagnostics = self._review_diagnostics(storage, main_reference)
        continuation = self._review_continuation(
            clip_uid=clip_uid,
            entity=entity,
            reference=main_reference,
            diagnostics=main_diagnostics,
            review=main_review,
            pre_edit=pre_edit,
        )
        if continuation == "artifact_bbox":
            return _BboxParent(
                variant=VARIANT_FINAL,
                job=main_job,
                payload=main_payload,
                review=main_review,
                reference=main_reference,
                diagnostics=main_diagnostics,
                main_diagnostics=main_diagnostics,
                context_path=str(main_anchor["source_context_path"]),
                context_sha256=str(main_anchor["source_context_sha256"]),
                parent_review_job_id=None,
            )
        if continuation == "source_alpha":
            alpha_reference, alpha_anchor, alpha_job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=pre_reference,
                variant=VARIANT_SOURCE_ALPHA,
                main_job=main_job,
            )
            alpha_payload = self._committed_review_payload(alpha_job)
            if str(alpha_payload.get("status")) != "review":
                raise ReferenceIntegrityDurableError(
                    f"artifact bbox continuation needs a successful source alpha "
                    f"review for {label}"
                )
            alpha_review = ReferenceIntegrityReview.model_validate(
                alpha_payload.get("review")
            )
            if self._artifact_bbox_route(
                clip_uid=clip_uid, reference=alpha_reference, review=alpha_review
            ):
                return _BboxParent(
                    variant=VARIANT_SOURCE_ALPHA,
                    job=alpha_job,
                    payload=alpha_payload,
                    review=alpha_review,
                    reference=alpha_reference,
                    diagnostics=self._review_diagnostics(storage, alpha_reference),
                    main_diagnostics=main_diagnostics,
                    context_path=str(alpha_anchor["source_context_path"]),
                    context_sha256=str(alpha_anchor["source_context_sha256"]),
                    parent_review_job_id=main_job.job_id(),
                )
        raise ReferenceIntegrityDurableError(
            f"frozen policy no longer routes {label} to the artifact source bbox"
        )

    def _bbox_context(
        self,
        *,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        pre_reference: EntityReferenceState,
        plan_entry: Mapping[str, Any],
        pre_edit: ReferenceEditState | None,
        parent_variant: str | None = None,
    ) -> tuple[_BboxParent, dict[str, Any], ModelJob]:
        """The frozen parent, anchor and deterministic job of one bbox route."""
        parent = self._bbox_parent(
            shard=shard,
            storage=storage,
            clip_uid=clip_uid,
            entity=entity,
            pre_reference=pre_reference,
            plan_entry=plan_entry,
            pre_edit=pre_edit,
        )
        if parent_variant is not None and parent.variant != parent_variant:
            raise ReferenceIntegrityDurableError(
                f"artifact bbox parent variant drifted for "
                f"{clip_uid}/{entity.entity_id}"
            )
        anchor = self._frozen_bbox_input(
            shard=shard,
            storage=storage,
            clip_uid=clip_uid,
            entity=entity,
            parent=parent,
        )
        job = self._expected_bbox_job(
            shard=shard, entity=entity, parent=parent, anchor=anchor
        )
        return parent, anchor, job

    def _expected_bbox_metadata(
        self, *, base_metadata: Mapping[str, Any], bbox_payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The exact legacy final metadata for one committed bbox receipt."""
        metadata = dict(base_metadata)
        status = str(bbox_payload.get("status", ""))
        if status == "judge_failed":
            metadata.update(
                {
                    "status": "judge_failed",
                    "review_status": "judge_failed",
                    "reason": str(bbox_payload.get("error")),
                    "raw_response": bbox_payload.get("raw_response"),
                }
            )
            return metadata
        if status != "review":
            raise ReferenceIntegrityDurableError(
                "committed source bbox review has an unknown payload status"
            )
        review = SourceBboxFallbackReview.model_validate(bbox_payload.get("review"))
        metadata.update(
            {
                "status": "accepted" if review.verdict == "accept" else "rejected",
                "review_status": review.verdict,
                "review": review.model_dump(mode="json"),
                "raw_response": str(bbox_payload.get("raw_response")),
                "finish_reason": bbox_payload.get("finish_reason"),
            }
        )
        return metadata

    def _publish_bbox_metadata(
        self,
        *,
        storage: RunStorage,
        anchor: Mapping[str, Any],
        bbox_payload: Mapping[str, Any],
    ) -> None:
        """Advance the legacy bbox metadata from materialized to final.

        The metadata is a two-phase artifact and the model call must never touch
        it: a crash between the Qwen call and the receipt commit would otherwise
        leave a state that claims an unreceipted review. Only the frozen base may
        transition, and only to the exact expected final state.
        """
        label = f"{anchor['clip_uid']}/{anchor['entity_id']}"
        base = anchor["base_metadata"]
        path = _resolve_run_artifact(storage, str(anchor["metadata_path"]))
        current = _read_json(path)
        expected = self._expected_bbox_metadata(
            base_metadata=base, bbox_payload=bbox_payload
        )
        if current == expected:
            return
        if current != base:
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity bbox metadata drifted for {label}"
            )
        write_json_atomic(path, expected)

    def _expected_bbox_marker(
        self,
        *,
        clip_uid: str,
        entity: Any,
        parent: _BboxParent,
        bbox_job: ModelJob,
        bbox_payload: Mapping[str, Any],
        bbox_anchor: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The exact durable marker one committed artifact bbox review produces.

        The delta carries the entity's whole legacy counter accumulation: the
        parent integrity review's counters plus this bbox attempt. The bbox judge
        itself never counts as an integrity review.
        """
        if parent.variant == VARIANT_FINAL:
            topology = int(parent.diagnostics.suspicious)
            reviewed = 1
        else:
            topology = int(parent.main_diagnostics.suspicious) + int(
                parent.diagnostics.suspicious
            )
            reviewed = 2
        common: dict[str, Any] = {
            "schema": REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "reviewed": True,
            "semantic_policy_reason": None,
            "source_context_path": parent.context_path,
            "diagnostics": parent.diagnostics.model_dump(mode="json"),
            "review": parent.review.model_dump(mode="json"),
            "review_job_id": parent.job.job_id(),
            "review_variant": parent.variant,
            "parent_review_job_id": parent.parent_review_job_id,
            "bbox_review_job_id": bbox_job.job_id(),
            "source_bbox_fallback_trigger": BBOX_TRIGGER_ARTIFACT,
            "source_bbox_fallback_candidate_path": str(bbox_anchor["candidate_path"]),
            "source_bbox_fallback_metadata_path": str(bbox_anchor["metadata_path"]),
            "source_bbox_xyxy": list(bbox_anchor["bbox_xyxy"]),
        }
        delta_prefix = {
            "topology_suspicious": topology,
            "entities_reviewed": reviewed,
            "source_bbox_fallback_attempted": 1,
        }
        status = str(bbox_payload.get("status", ""))
        if status == "judge_failed":
            # Legacy keeps this a plain rejection with the parent review as its
            # public provenance; only this epoch's marker remembers the bbox job.
            return {
                **common,
                "outcome": ENTITY_OUTCOME_REJECTED,
                "status": ENTITY_OUTCOME_REJECTED,
                "final_reference_path": parent.reference.image_path,
                "judge_failed": False,
                "bbox_judge_failed": True,
                "source_bbox_fallback_review": None,
                "reason": f"{BBOX_JUDGE_FAILED_PREFIX}{bbox_payload.get('error')}",
                "delta": {
                    **delta_prefix,
                    "source_bbox_fallback_judge_failed": 1,
                    "source_bbox_fallback_rejected": 1,
                    "entities_rejected": 1,
                },
            }
        if status != "review":
            raise ReferenceIntegrityDurableError(
                "committed source bbox review has an unknown payload status"
            )
        review = SourceBboxFallbackReview.model_validate(bbox_payload.get("review"))
        accepted = review.verdict == "accept"
        outcome = ENTITY_OUTCOME_ACCEPTED if accepted else ENTITY_OUTCOME_REJECTED
        return {
            **common,
            "outcome": outcome,
            "status": outcome,
            "final_reference_path": (
                str(bbox_anchor["candidate_path"])
                if accepted
                else parent.reference.image_path
            ),
            "judge_failed": False,
            "bbox_judge_failed": False,
            "source_bbox_fallback_review": review.model_dump(mode="json"),
            "reason": review.reason,
            "delta": {
                **delta_prefix,
                **(
                    {"source_bbox_fallback_accepted": 1, "entities_accepted": 1}
                    if accepted
                    else {
                        "source_bbox_fallback_rejected": 1,
                        "entities_rejected": 1,
                    }
                ),
            },
        }

    def _run_bbox_review(self, job: ModelJob, handle: Any) -> JobResult:
        """One source bbox fallback call. No prompt or schema logic is copied."""
        if str(dict(job.target).get("trigger", "")) != BBOX_TRIGGER_ARTIFACT:
            raise ReferenceIntegrityEpochError(
                f"unsupported bbox trigger {dict(job.target).get('trigger')!r}"
            )
        parent_variant = str(dict(job.target).get("parent_variant", ""))
        shard, storage, _clip, entity, reference, plan_entry, pre_edit = (
            self._job_review_context(job)
        )
        _parent, anchor, expected = self._bbox_context(
            shard=shard,
            storage=storage,
            clip_uid=job.clip_uid,
            entity=entity,
            pre_reference=reference,
            plan_entry=plan_entry,
            pre_edit=pre_edit,
            parent_variant=parent_variant,
        )
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != str(anchor["semantic_digest"])
            or job.model_identity != expected.model_identity
        ):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Integrity bbox review semantic identity drifted",
            )
        resolved = resolve_source_bbox_fallback_judge(handle, self.config)
        try:
            with Image.open(
                _resolve_run_artifact(storage, str(anchor["source_context_path"]))
            ) as opened:
                opened.load()
                source_context = opened.copy()
            with Image.open(
                _resolve_run_artifact(storage, str(anchor["candidate_path"]))
            ) as opened:
                opened.load()
                candidate = opened.convert("RGB")
            attempt = resolved.judge.review(
                source_context=source_context,
                failed_reference=_load_reference_image(
                    storage, str(anchor["current_reference_path"])
                ),
                source_bbox_candidate=candidate,
                reference_type=entity.reference_type,
                phrase=entity.phrase,
                grounding_prompt=entity.grounding_prompt,
                reference_scope=str(anchor["reference_scope"]),
                trigger=BBOX_TRIGGER_ARTIFACT,
            )
        except SourceBboxFallbackJudgeFailure as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "judge_failed",
                    "error": str(exc),
                    "raw_response": exc.raw_response,
                },
            )
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "review": attempt.review.model_dump(mode="json"),
                "raw_response": attempt.raw_response,
                "finish_reason": attempt.finish_reason,
            },
        )

    def _finalize_bbox_review(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        """CPU continuation of one committed bbox review. Unlocks nothing."""
        if str(dict(job.target).get("trigger", "")) != BBOX_TRIGGER_ARTIFACT:
            raise ReferenceIntegrityEpochError(
                f"unsupported bbox trigger {dict(job.target).get('trigger')!r}"
            )
        parent_variant = str(dict(job.target).get("parent_variant", ""))
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        if status not in {"review", "judge_failed"}:
            raise ReferenceIntegrityDurableError(
                f"committed bbox job {job.job_id()} has an unknown payload status"
            )
        if status == "review":
            try:
                SourceBboxFallbackReview.model_validate(payload.get("review"))
            except Exception as exc:
                raise ReferenceIntegrityDurableError(
                    f"committed bbox job {job.job_id()} has an invalid review"
                ) from exc
        shard, storage, _clip, entity, reference, plan_entry, pre_edit = (
            self._job_review_context(job)
        )
        parent, anchor, expected = self._bbox_context(
            shard=shard,
            storage=storage,
            clip_uid=job.clip_uid,
            entity=entity,
            pre_reference=reference,
            plan_entry=plan_entry,
            pre_edit=pre_edit,
            parent_variant=parent_variant,
        )
        if expected.job_id() != job.job_id():
            raise ReferenceIntegrityDurableError(
                f"bbox review job {job.job_id()} is not the frozen job for "
                f"{job.clip_uid}/{entity.entity_id}"
            )
        self._publish_bbox_metadata(
            storage=storage, anchor=anchor, bbox_payload=payload
        )
        marker = self._expected_bbox_marker(
            clip_uid=job.clip_uid,
            entity=entity,
            parent=parent,
            bbox_job=expected,
            bbox_payload=payload,
            bbox_anchor=anchor,
        )
        self._write_entity_outcome(shard, job.clip_uid, entity.entity_id, marker)
        self._publish_clip_if_terminal(shard, storage, job.clip_uid)
        return ()

    def _advance_clip(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> list[ModelJob]:
        plan = self._existing_plan_for_reconcile(shard)
        entry = plan["clips"][clip_uid]
        if entry.get("classification") != CLIP_FRESH_TARGET:
            return []
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return []
        clip = storage.read_clip(clip_uid)
        references_by_id = {item.entity_id: item for item in clip.references.entities}
        entities_by_id = {
            item.entity_id: item
            for item in (
                clip.annotation.entities if clip.annotation is not None else []
            )
        }
        jobs: list[ModelJob] = []
        for entity_id in entry.get("retained_entity_ids", []):
            if self._entity_outcome(shard, clip_uid, entity_id) is not None:
                continue
            jobs.extend(
                self._advance_entity(
                    shard,
                    storage,
                    clip,
                    entities_by_id[entity_id],
                    references_by_id[entity_id],
                )
            )
        self._publish_clip_if_terminal(shard, storage, clip_uid)
        return jobs

    # -- CPU replay entry point ------------------------------------------------

    def seed_jobs(self) -> list[ModelJob]:
        """CPU fixed point, then the deterministic pending main review jobs.

        An entity with a durable outcome is never re-seeded, and a clip that is
        already terminal is skipped, so the same frozen state always yields the
        same job ids.
        """
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            self._plan(shard)
            storage = self._storage_for(shard)
            plan = self._existing_plan_for_reconcile(shard)
            for clip_uid in sorted(plan.get("clips", {})):
                if plan["clips"][clip_uid].get("classification") != CLIP_FRESH_TARGET:
                    continue
                if self._clip_outcome_path(shard, clip_uid).is_file():
                    continue
                try:
                    jobs.extend(self._advance_clip(shard, storage, clip_uid))
                except ReferenceIntegrityDurableError:
                    # Durable corruption is never a semantic clip failure.
                    raise
                except Exception as exc:  # noqa: BLE001 - legacy clip isolation
                    self._fail_clip_terminal(shard, storage, clip_uid, exc)
        return sorted(jobs, key=lambda job: job.job_id())

    # -- publication -----------------------------------------------------------

    def _entity_state(
        self,
        marker: Mapping[str, Any],
        input_reference: EntityReferenceState,
    ) -> ReferenceIntegrityEntityState:
        payload = {
            "entity_id": marker["entity_id"],
            "status": marker["status"],
            "input_reference": input_reference.model_dump(mode="json"),
            "final_reference_path": marker["final_reference_path"],
            "diagnostics": marker["diagnostics"],
            "reviewed": marker["reviewed"],
            "reason": marker["reason"],
        }
        if marker.get("semantic_policy_reason") is not None:
            payload["semantic_policy_reason"] = marker["semantic_policy_reason"]
        if marker["reviewed"]:
            payload["source_context_path"] = marker["source_context_path"]
            payload["judge_failed"] = marker["judge_failed"]
            if marker["review"] is not None:
                payload["review"] = marker["review"]
            # Legacy publishes the artifact bbox provenance only for an explicit
            # bbox verdict; a bbox judge failure keeps the parent review as its
            # whole public provenance.
            if marker.get("bbox_review_job_id") and not marker["bbox_judge_failed"]:
                payload["source_bbox_fallback_trigger"] = marker[
                    "source_bbox_fallback_trigger"
                ]
                payload["source_bbox_fallback_candidate_path"] = marker[
                    "source_bbox_fallback_candidate_path"
                ]
                payload["source_bbox_fallback_metadata_path"] = marker[
                    "source_bbox_fallback_metadata_path"
                ]
                payload["source_bbox_xyxy"] = tuple(marker["source_bbox_xyxy"])
                payload["source_bbox_fallback_review"] = marker[
                    "source_bbox_fallback_review"
                ]
        return ReferenceIntegrityEntityState.model_validate(payload)

    def _verify_entity_branch(
        self,
        shard: str,
        storage: RunStorage,
        clip: Any,
        plan_entry: Mapping[str, Any],
        entities_by_id: Mapping[str, Any],
        pre_entities: Sequence[EntityReferenceState],
        pre_edit: ReferenceEditState | None,
        marker: Mapping[str, Any],
    ) -> None:
        """Re-derive one entity's durable marker from frozen state and require it.

        CPU branches replay the frozen CPU policy; reviewed branches rebuild the
        deterministic review job, validate its committed receipt through the
        ledger and re-derive the marker from the receipt payload. Either way the
        durable marker must match exactly, every counter in ``delta`` included.
        """
        entity_id = str(marker["entity_id"])
        reference = next(item for item in pre_entities if item.entity_id == entity_id)
        entity = entities_by_id[entity_id]
        if not marker["reviewed"]:
            final_image = _load_reference_image(storage, str(reference.image_path))
            diagnostics = reference_topology_diagnostics(final_image)
            expected = self._expected_entity_marker(
                clip.clip_uid, entity, reference, diagnostics
            )
            if expected is None:
                raise ReferenceIntegrityDurableError(
                    f"entity {clip.clip_uid}/{entity_id} carries a CPU outcome but "
                    "now requires a Qwen review"
                )
            if dict(marker) != expected:
                raise ReferenceIntegrityDurableError(
                    f"entity outcome drifted for {clip.clip_uid}/{entity_id}"
                )
            return
        self._verify_reviewed_branch(
            shard,
            storage,
            clip,
            plan_entry,
            entity,
            reference,
            pre_edit,
            marker,
        )

    def _verify_reviewed_branch(
        self,
        shard: str,
        storage: RunStorage,
        clip: Any,
        plan_entry: Mapping[str, Any],
        entity: Any,
        reference: Any,
        pre_edit: ReferenceEditState | None,
        marker: Mapping[str, Any],
    ) -> None:
        """Re-derive a reviewed marker from its committed review receipts.

        The main receipt and (for a source-alpha outcome) the alpha receipt are
        both validated through the ledger, then the whole marker -- including
        both reviews' counters -- is re-derived from those payloads. Nothing the
        marker itself carries is trusted.
        """
        entity_id = entity.entity_id
        variant = marker.get("review_variant")
        if variant == VARIANT_FINAL:
            _input_reference, _anchor, job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip.clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_FINAL,
            )
        elif variant == VARIANT_SOURCE_ALPHA:
            _r, _a, main_job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip.clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_FINAL,
            )
            if marker.get("parent_review_job_id") != main_job.job_id():
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome parent job id drifted for "
                    f"{clip.clip_uid}/{entity_id}"
                )
            _r2, _a2, job = self._expected_review_inputs(
                shard=shard,
                storage=storage,
                clip_uid=clip.clip_uid,
                plan_entry=plan_entry,
                entity=entity,
                reference=reference,
                variant=VARIANT_SOURCE_ALPHA,
                main_job=main_job,
            )
        else:
            raise ReferenceIntegrityDurableError(
                f"reviewed entity outcome variant is unknown for "
                f"{clip.clip_uid}/{entity_id}"
            )
        if marker.get("review_job_id") != job.job_id():
            raise ReferenceIntegrityDurableError(
                f"reviewed entity outcome job id drifted for "
                f"{clip.clip_uid}/{entity_id}"
            )
        bbox_job_id = marker.get("bbox_review_job_id")
        if bbox_job_id is None:
            outcome = self._review_outcome(
                shard=shard,
                storage=storage,
                clip_uid=clip.clip_uid,
                entity=entity,
                reference=reference,
                plan_entry=plan_entry,
                pre_edit=pre_edit,
                job=job,
                payload=self._committed_review_payload(job),
            )
            if outcome.marker is None:
                raise ReferenceIntegrityDurableError(
                    f"entity {clip.clip_uid}/{entity_id} has a durable review "
                    "outcome but the frozen policy now defers it"
                )
            if dict(marker) != outcome.marker:
                raise ReferenceIntegrityDurableError(
                    f"reviewed entity outcome drifted for "
                    f"{clip.clip_uid}/{entity_id}"
                )
            return
        # An artifact bbox outcome must be re-derived from the whole frozen
        # chain: both parent receipts, the frozen bbox anchor and the bbox
        # receipt. Nothing the marker carries is trusted.
        parent, anchor, bbox_job = self._bbox_context(
            shard=shard,
            storage=storage,
            clip_uid=clip.clip_uid,
            entity=entity,
            pre_reference=reference,
            plan_entry=plan_entry,
            pre_edit=pre_edit,
            parent_variant=str(variant),
        )
        if bbox_job_id != bbox_job.job_id():
            raise ReferenceIntegrityDurableError(
                f"source bbox job id drifted for {clip.clip_uid}/{entity_id}"
            )
        expected = self._expected_bbox_marker(
            clip_uid=clip.clip_uid,
            entity=entity,
            parent=parent,
            bbox_job=bbox_job,
            bbox_payload=self._committed_review_payload(bbox_job),
            bbox_anchor=anchor,
        )
        if dict(marker) != expected:
            raise ReferenceIntegrityDurableError(
                f"reviewed entity outcome drifted for {clip.clip_uid}/{entity_id}"
            )

    def _reconstruct_clip_publication(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan_entry: Mapping[str, Any],
    ) -> tuple[Any, Any, Any, Any, dict[str, int]]:
        """Pure-CPU expected final publication for one fresh clip."""
        retained = list(plan_entry.get("retained_entity_ids", []))
        clip = storage.read_clip(clip_uid)
        entities_by_id = {
            item.entity_id: item
            for item in (
                clip.annotation.entities if clip.annotation is not None else []
            )
        }
        pre_pairing = dict(plan_entry.get("pre_pairing") or {})
        pre_references = dict(plan_entry.get("pre_references") or {})
        pre_entities = [
            EntityReferenceState.model_validate(item)
            for item in pre_references.get("entities", [])
        ]
        pre_edit_dump = plan_entry.get("pre_reference_edit")
        pre_edit = (
            ReferenceEditState.model_validate(pre_edit_dump)
            if pre_edit_dump is not None
            else None
        )
        markers: dict[str, dict[str, Any]] = {}
        for entity_id in retained:
            marker = self._entity_outcome(shard, clip_uid, entity_id)
            if marker is None:
                raise ReferenceIntegrityDurableError(
                    f"clip {clip_uid!r} cannot be reconstructed: entity "
                    f"{entity_id!r} has no durable outcome"
                )
            self._verify_entity_branch(
                shard,
                storage,
                clip,
                plan_entry,
                entities_by_id,
                pre_entities,
                pre_edit,
                marker,
            )
            markers[entity_id] = marker
        rejected_ids = {
            entity_id
            for entity_id, marker in markers.items()
            if marker["outcome"] == ENTITY_OUTCOME_REJECTED
        }
        # Legacy keeps two separate maps: ``fallback_references`` (published in
        # place of the pre-stage reference) only ever receives an *accepted*
        # source alpha, while ``rejected_ids`` collects every rejection. The
        # entity outcome, by contrast, records the alpha reference it judged for
        # both. The frozen pre-edit is the only source of either.
        alpha_references = {
            entity_id: self._frozen_source_alpha_reference(plan_entry, entity_id)
            for entity_id, marker in markers.items()
            if marker.get("review_variant") == VARIANT_SOURCE_ALPHA
        }
        # A source-alpha *outcome* accepted on its own falls back to the alpha
        # reference. An artifact bbox outcome that happens to have an alpha
        # parent also carries ``review_variant == source_alpha``, so it must be
        # excluded here: its Reference Edit mutation is the bbox one.
        alpha_accepted_ids = {
            entity_id
            for entity_id, marker in markers.items()
            if entity_id in alpha_references
            and not marker.get("bbox_review_job_id")
            and marker["outcome"] == ENTITY_OUTCOME_ACCEPTED
        }
        # The bbox parent's reference is what an artifact outcome judged: the
        # frozen pre-stage reference for a final parent, the frozen source alpha
        # for an alpha parent. That is the same rule as ``input_references``.
        bbox_accepted_ids = {
            entity_id
            for entity_id, marker in markers.items()
            if marker.get("bbox_review_job_id")
            and marker["outcome"] == ENTITY_OUTCOME_ACCEPTED
        }
        input_references = {
            entity_id: (alpha_references.get(entity_id) or pre_reference)
            for entity_id, pre_reference in (
                (item.entity_id, item) for item in pre_entities
            )
            if entity_id in retained
        }
        # ``_verify_entity_branch`` has already re-derived every marker from its
        # receipts and the frozen bbox anchor, so these provenance paths are the
        # verified ones.
        bbox_references = {
            entity_id: _source_bbox_reference(
                input_references[entity_id],
                clip_uid=clip_uid,
                image_path=str(markers[entity_id]["source_bbox_fallback_candidate_path"]),
                bbox_xyxy=tuple(
                    int(value)
                    for value in markers[entity_id]["source_bbox_xyxy"]
                ),
                metadata_path=str(
                    markers[entity_id]["source_bbox_fallback_metadata_path"]
                ),
            )
            for entity_id in bbox_accepted_ids
        }
        final_references = [
            (
                bbox_references[item.entity_id]
                if item.entity_id in bbox_references
                else alpha_references[item.entity_id]
                if item.entity_id in alpha_accepted_ids
                else _rejected_reference(item, REJECTED_REFERENCE_REASON)
                if item.entity_id in rejected_ids
                else item
            )
            for item in pre_entities
        ]
        retained_after = [
            entity_id
            for entity_id in pre_pairing.get("retained_entity_ids", [])
            if entity_id not in rejected_ids
        ]
        qualifying = set(
            clip.coverage.qualifying_entity_ids if clip.coverage is not None else []
        )
        if not set(retained_after).intersection(qualifying):
            pairing = PairingState(
                status="rejected",
                reason="no_qualifying_reference_after_integrity",
            )
        else:
            pairing = PairingState(
                status="ready",
                retained_entity_ids=retained_after,
                tokens=_tokens_for_retained(
                    retained_after,
                    {
                        item.entity_id: item.reference_type
                        for item in entities_by_id.values()
                    },
                ),
                background_token=pre_pairing.get("background_token"),
            )
        references_state = ReferencesState(
            entities=final_references,
            background=pre_references.get("background"),
        )
        # Legacy mutates Reference Edit per entity in the frozen retained order,
        # and several entities may fall back in one clip. An entity reaches at
        # most one of the two routes, because an accepted source alpha never
        # continues into the bbox fallback.
        reference_edit = pre_edit
        for entity_id in retained:
            try:
                if entity_id in alpha_accepted_ids:
                    reference_edit = _reference_edit_after_source_alpha(
                        reference_edit,
                        entity_id=entity_id,
                        reason=SOURCE_ALPHA_FALLBACK_REASON,
                    )
                elif entity_id in bbox_accepted_ids:
                    reference_edit = _reference_edit_after_source_bbox(
                        reference_edit,
                        entity_id=entity_id,
                        output_image_path=str(
                            markers[entity_id][
                                "source_bbox_fallback_candidate_path"
                            ]
                        ),
                        metadata_path=str(
                            markers[entity_id]["source_bbox_fallback_metadata_path"]
                        ),
                    )
            except ReferenceIntegrityDurableError:
                raise
            except Exception as exc:
                raise ReferenceIntegrityDurableError(
                    f"cannot apply the frozen source fallback for "
                    f"{clip_uid}/{entity_id}"
                ) from exc
        integrity_state = ReferenceIntegrityState(
            status="ready",
            entities=[
                self._entity_state(markers[entity_id], input_references[entity_id])
                for entity_id in retained
            ],
        )
        delta: dict[str, int] = {"processed": 1, "failed": 0}
        for entity_id in retained:
            for key, value in markers[entity_id]["delta"].items():
                delta[key] = delta.get(key, 0) + int(value)
        return references_state, pairing, reference_edit, integrity_state, delta

    def _publish_clip_if_terminal(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> None:
        plan = _read_json(self._plan_path(shard))
        if plan is None:
            raise ReferenceIntegrityDurableError(
                f"Reference Integrity publication needs the frozen plan for "
                f"{shard!r}"
            )
        plan_entry = plan["clips"][clip_uid]
        if any(
            self._entity_outcome(shard, clip_uid, entity_id) is None
            for entity_id in plan_entry.get("retained_entity_ids", [])
        ):
            return
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return
        (
            references_state,
            pairing,
            reference_edit,
            integrity_state,
            delta,
        ) = self._reconstruct_clip_publication(shard, storage, clip_uid, plan_entry)
        storage.write_reference_integrity_result(
            clip_uid,
            references_state,
            pairing,
            integrity_state,
            reference_edit=reference_edit,
        )
        _write_json_once(
            self._clip_outcome_path(shard, clip_uid),
            {
                "schema": REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "ready",
                "delta": delta,
            },
        )

    def _fail_clip_terminal(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        exc: Exception,
    ) -> None:
        """Legacy clip isolation: one ordinary CPU failure is terminal."""
        reason = str(exc)
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return
        storage.write_reference_integrity_failure(clip_uid, reason)
        storage.append_failure(
            stage="reference_integrity",
            clip_uid=clip_uid,
            reason=reason,
            details={"exception_type": type(exc).__name__},
        )
        delta: dict[str, int] = {"processed": 1, "failed": 1}
        for key, value in self._clip_entity_deltas(shard, clip_uid).items():
            delta[key] = delta.get(key, 0) + int(value)
        _write_json_once(
            self._clip_outcome_path(shard, clip_uid),
            {
                "schema": REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "failed",
                "reason": reason,
                "delta": delta,
            },
        )

    def _clip_entity_deltas(self, shard: str, clip_uid: str) -> dict[str, int]:
        """Sum the entity outcome counters already durable for one clip.

        A clip-level CPU failure happens after some entities may already have
        reached a terminal branch; the legacy counters keep those increments, so
        the failed clip delta must carry them too.
        """
        plan = _read_json(self._plan_path(shard))
        if plan is None:
            return {}
        entry = (plan.get("clips") or {}).get(clip_uid)
        if not isinstance(entry, dict):
            return {}
        totals: dict[str, int] = {}
        for entity_id in entry.get("retained_entity_ids", []):
            marker = self._entity_outcome(shard, clip_uid, entity_id)
            if marker is None:
                continue
            for key, value in marker["delta"].items():
                totals[key] = totals.get(key, 0) + int(value)
        return totals

    def _failed_clip_delta(
        self, shard: str, clip_uid: str, integrity: Any
    ) -> dict[str, int]:
        """The exact delta a failed publication must carry on back-fill."""
        target = {"processed": 1, "failed": 1}
        for key, value in self._clip_entity_deltas(shard, clip_uid).items():
            target[key] = target.get(key, 0) + int(value)
        del clip_uid, integrity
        return target

    # -- restart verification --------------------------------------------------

    def _verify_published_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan_entry: Mapping[str, Any],
    ) -> None:
        """Verify a published clip against its durable outcome marker.

        A crash between the publication and the marker is repaired by
        back-filling the exact expected marker after a byte-exact comparison;
        any other difference fails closed.
        """
        clip = storage.read_clip(clip_uid)
        marker_path = self._clip_outcome_path(shard, clip_uid)
        marker = _read_json(marker_path)
        integrity = clip.reference_integrity
        if integrity is not None and integrity.status == "failed":
            expected_marker = {
                "schema": REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "failed",
                "reason": str(integrity.reason or ""),
                "delta": self._failed_clip_delta(shard, clip_uid, integrity),
            }
            # ``write_reference_integrity_failure`` only replaces the Reference
            # Integrity section, so every other section must still equal the
            # frozen pre-stage baseline.
            baselines = self._live_baselines(clip)
            for key in (
                "pre_references",
                "pre_pairing",
                "pre_reference_edit",
                "pre_instruction",
                "pre_export",
            ):
                if plan_entry.get(key) != baselines[key]:
                    raise ReferenceIntegrityDurableError(
                        f"clip {clip_uid!r} failed publication drifted {key} away "
                        "from its frozen baseline"
                    )
            if marker is None:
                _write_json_once(marker_path, expected_marker)
                return
            if marker != expected_marker:
                raise ReferenceIntegrityDurableError(
                    f"clip {clip_uid!r} failure outcome marker does not match the "
                    "published failure"
                )
            return
        if integrity is None:
            if marker is not None:
                raise ReferenceIntegrityDurableError(
                    f"clip {clip_uid!r} has a durable outcome marker but no "
                    "published Reference Integrity state"
                )
            raise ReferenceIntegrityDurableError(
                f"clip {clip_uid!r} published Reference Integrity state without a "
                "durable outcome marker"
            )
        (
            references_state,
            pairing,
            reference_edit,
            integrity_state,
            delta,
        ) = self._reconstruct_clip_publication(shard, storage, clip_uid, plan_entry)
        live_edit = (
            clip.reference_edit.model_dump(mode="json")
            if clip.reference_edit is not None
            else None
        )
        expected_edit = (
            reference_edit.model_dump(mode="json")
            if reference_edit is not None
            else None
        )
        if (
            clip.references.model_dump(mode="json")
            != references_state.model_dump(mode="json")
            or clip.pairing.model_dump(mode="json") != pairing.model_dump(mode="json")
            or live_edit != expected_edit
            or integrity.model_dump(mode="json")
            != integrity_state.model_dump(mode="json")
        ):
            raise ReferenceIntegrityDurableError(
                f"published Reference Integrity state for {clip_uid!r} does not "
                "match its durable outcome"
            )
        if clip.instruction is not None or clip.export != ExportState():
            raise ReferenceIntegrityDurableError(
                f"published Reference Integrity clip {clip_uid!r} did not reset its "
                "instruction/export"
            )
        expected_marker = {
            "schema": REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "terminal": "ready",
            "delta": delta,
        }
        if marker is None:
            _write_json_once(marker_path, expected_marker)
            return
        if marker != expected_marker:
            raise ReferenceIntegrityDurableError(
                f"clip {clip_uid!r} ready outcome marker does not match its durable "
                "publication"
            )

    # -- durable stats ---------------------------------------------------------

    def _validated_clip_outcome(
        self, clip_uid: str, marker: Mapping[str, Any]
    ) -> str:
        if marker.get("schema") != REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA:
            raise ReferenceIntegrityDurableError(
                f"clip outcome schema drifted for {clip_uid!r}"
            )
        if marker.get("clip_uid") != clip_uid:
            raise ReferenceIntegrityDurableError(
                f"clip outcome clip_uid drifted for {clip_uid!r}"
            )
        terminal = marker.get("terminal")
        if terminal not in {"ready", "failed"}:
            raise ReferenceIntegrityDurableError(
                f"clip outcome terminal is unknown for {clip_uid!r}"
            )
        delta = marker.get("delta")
        if not isinstance(delta, dict) or not delta:
            raise ReferenceIntegrityDurableError(
                f"clip outcome delta is missing for {clip_uid!r}"
            )
        for key, value in delta.items():
            if key not in CLIP_DELTA_KNOWN_FIELDS:
                raise ReferenceIntegrityDurableError(
                    f"clip outcome delta field {key!r} is unknown for {clip_uid!r}"
                )
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReferenceIntegrityDurableError(
                    f"clip outcome delta {key} is not a non-negative int for "
                    f"{clip_uid!r}"
                )
        return str(terminal)

    def reconcile_stats(self, shard: str) -> ReferenceIntegrityStats:
        """Stats rebuilt from the frozen plan, clip markers and entity markers."""
        plan = self._existing_plan_for_reconcile(shard)
        counts = {field: 0 for field in ReferenceIntegrityStats.__dataclass_fields__}
        for clip_uid, entry in sorted(plan.get("clips", {}).items()):
            classification = entry.get("classification")
            if classification == CLIP_INELIGIBLE:
                counts["skipped_not_ready"] += 1
                continue
            if classification == CLIP_EXISTING:
                counts["skipped_existing"] += 1
                continue
            marker = _read_json(self._clip_outcome_path(shard, clip_uid))
            if marker is None:
                raise ReferenceIntegrityEpochError(
                    f"Reference Integrity clip {clip_uid!r} has no durable "
                    "outcome; the stage is not terminal"
                )
            self._validated_clip_outcome(clip_uid, marker)
            for key, value in marker.get("delta", {}).items():
                counts[key] = counts.get(key, 0) + int(value)
        return ReferenceIntegrityStats(**counts)
