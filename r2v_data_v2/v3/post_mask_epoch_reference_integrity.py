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

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.post_mask_epoch_jobs import ModelJob, semantic_input_digest
from r2v_data_v2.v3.reference_integrity import (
    ReferenceIntegrityStats,
    _rejected_reference,
    _resolve_run_artifact,
    _tokens_for_retained,
    reference_semantic_hard_reject_reason,
    reference_semantic_risk_reason,
    reference_topology_diagnostics,
)
from r2v_data_v2.v3.schemas import (
    EntityReferenceState,
    ExportState,
    PairingState,
    ReferenceEditState,
    ReferenceIntegrityEntityState,
    ReferenceIntegrityState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage

REFERENCE_INTEGRITY_PLAN_SCHEMA = "post_mask_epoch_reference_integrity_plan/1"
REFERENCE_INTEGRITY_ENTITY_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_integrity_entity_outcome/1"
)
REFERENCE_INTEGRITY_CLIP_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_integrity_outcome/1"
)
REFERENCE_INTEGRITY_POLICY_VERSION = "reference_integrity_epoch_policy/1"

CLIP_FRESH_TARGET = "fresh_target"
CLIP_EXISTING = "existing"
CLIP_INELIGIBLE = "ineligible"

#: The two CPU-only entity branches implemented in 6b. Any other branch means
#: the entity needs a model review (6c/6d) and must stay unresolved.
ENTITY_OUTCOME_REJECTED = "rejected"
ENTITY_OUTCOME_SKIPPED = "skipped"
CPU_ENTITY_OUTCOMES = (ENTITY_OUTCOME_REJECTED, ENTITY_OUTCOME_SKIPPED)

#: Counter fields a CPU-only entity outcome may carry. ``entities_reviewed`` is
#: deliberately absent: it counts real main-judge review invocations, which 6b
#: never performs.
ENTITY_DELTA_FIELDS = (
    "topology_suspicious",
    "entities_rejected",
    "entities_accepted",
    "entities_skipped_review",
    "semantic_policy_rejected",
)

CLIP_DELTA_FIELDS = ("processed", "failed")
#: The ready clip marker binds the FULL clip stats delta, so it also carries the
#: per-entity counters its entity outcomes contributed.
CLIP_DELTA_KNOWN_FIELDS = CLIP_DELTA_FIELDS + ENTITY_DELTA_FIELDS

CLEAN_REAL_FULL_REFERENCE = "clean_real_full_reference"
REJECTED_REFERENCE_REASON = "reference_integrity_rejected"


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

    def _storage_for(self, shard: str) -> RunStorage:
        storage = self.storages.get(shard)
        if storage is None:
            raise ReferenceIntegrityEpochError(
                f"no run storage for canonical shard {shard!r}"
            )
        return storage

    # -- policy identity -------------------------------------------------------

    def _policy_identity(self) -> dict[str, Any]:
        policy = self.config.reference_integrity
        return {
            "policy_version": REFERENCE_INTEGRITY_POLICY_VERSION,
            "enabled": bool(policy.enabled),
            "mode": str(policy.mode),
            "reference_edit_enabled": bool(self.config.reference_edit.enabled),
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
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

    def _verify_plan_entry(
        self, shard: str, storage: RunStorage, clip_uid: str, entry: Mapping[str, Any]
    ) -> None:
        """Validate one frozen plan entry against the live clip."""
        clip = storage.read_clip(clip_uid)
        actual = self._clip_digest(storage, clip)
        if actual != entry.get("digest"):
            raise ReferenceIntegrityDurableError(
                f"frozen Reference Integrity input drifted for {clip_uid!r}"
            )
        if entry.get("classification") != CLIP_FRESH_TARGET:
            return
        if clip.reference_integrity is not None:
            # Published: rebuild the expected publication and verify the live
            # state against it, with or without the marker present.
            self._verify_published_clip(shard, storage, clip_uid, entry)
            return
        if self._clip_outcome_path(shard, clip_uid).is_file():
            raise ReferenceIntegrityDurableError(
                f"clip {clip_uid!r} has a durable outcome marker but no published "
                "Reference Integrity state"
            )
        baselines = {
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
        for key, value in baselines.items():
            if entry.get(key) != value:
                raise ReferenceIntegrityDurableError(
                    f"frozen Reference Integrity {key} drifted for {clip_uid!r}"
                )

    def _plan(self, shard: str) -> dict[str, Any]:
        existing = _read_json(self._plan_path(shard))
        if existing is not None:
            expected = {
                "schema": REFERENCE_INTEGRITY_PLAN_SCHEMA,
                "canonical_shard": shard,
                "eligible_clip_uids": list(self.eligible.get(shard, ())),
                "policy": self._policy_identity(),
            }
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ReferenceIntegrityDurableError(
                        f"frozen Reference Integrity plan {key} drifted for "
                        f"{shard!r}"
                    )
            storage = self._storage_for(shard)
            for clip_uid, entry in existing.get("clips", {}).items():
                self._verify_plan_entry(shard, storage, clip_uid, entry)
            return existing

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
        storage = self._storage_for(shard)
        for clip_uid, entry in payload.get("clips", {}).items():
            self._verify_plan_entry(shard, storage, clip_uid, entry)
        return payload

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
        if outcome not in CPU_ENTITY_OUTCOMES:
            raise ReferenceIntegrityDurableError(
                f"entity outcome {outcome!r} is not a 6b CPU branch for {label}"
            )
        if marker.get("status") != outcome:
            raise ReferenceIntegrityDurableError(
                f"entity outcome status disagrees with its branch for {label}"
            )
        if marker.get("reviewed") is not False:
            raise ReferenceIntegrityDurableError(
                f"CPU entity outcome must be reviewed=false for {label}"
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
        if outcome == ENTITY_OUTCOME_REJECTED:
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

    def _advance_entity(
        self,
        shard: str,
        storage: RunStorage,
        clip: Any,
        entity: Any,
        reference: Any,
    ) -> None:
        """Resolve one entity's CPU branch, or leave it for 6c/6d."""
        final_image = _load_reference_image(storage, str(reference.image_path))
        diagnostics = reference_topology_diagnostics(final_image)
        branch = self._entity_branch(entity, reference, diagnostics)
        if branch.requires_model:
            # 6c/6d own this entity: stay unresolved, never fake a terminal.
            return
        self._write_entity_outcome(
            shard,
            clip.clip_uid,
            entity.entity_id,
            {
                "outcome": branch.outcome,
                "status": branch.status,
                "reason": branch.reason,
                "reviewed": branch.reviewed,
                "semantic_policy_reason": branch.semantic_policy_reason,
                "final_reference_path": reference.image_path,
                "diagnostics": branch.diagnostics.model_dump(mode="json"),
                "delta": branch.delta,
            },
        )

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
        for entity_id in entry.get("retained_entity_ids", []):
            if self._entity_outcome(shard, clip_uid, entity_id) is not None:
                continue
            self._advance_entity(
                shard,
                storage,
                clip,
                entities_by_id[entity_id],
                references_by_id[entity_id],
            )
        self._publish_clip_if_terminal(shard, storage, clip_uid)
        return []

    # -- CPU replay entry point ------------------------------------------------

    def seed_jobs(self) -> list[ModelJob]:
        """CPU fixed point over every fresh clip.

        6b has no model job at all, so this always returns an empty list: a clip
        whose entity needs a review simply stays unresolved.
        """
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
                    self._advance_clip(shard, storage, clip_uid)
                except ReferenceIntegrityDurableError:
                    # Durable corruption is never a semantic clip failure.
                    raise
                except Exception as exc:  # noqa: BLE001 - legacy clip isolation
                    self._fail_clip_terminal(shard, storage, clip_uid, exc)
        return []

    # -- publication -----------------------------------------------------------

    def _entity_state(
        self,
        marker: Mapping[str, Any],
        reference: EntityReferenceState,
    ) -> ReferenceIntegrityEntityState:
        payload = {
            "entity_id": marker["entity_id"],
            "status": marker["status"],
            "input_reference": reference.model_dump(mode="json"),
            "final_reference_path": marker["final_reference_path"],
            "diagnostics": marker["diagnostics"],
            "reviewed": marker["reviewed"],
            "reason": marker["reason"],
        }
        if marker.get("semantic_policy_reason") is not None:
            payload["semantic_policy_reason"] = marker["semantic_policy_reason"]
        return ReferenceIntegrityEntityState.model_validate(payload)

    def _verify_entity_branch(
        self,
        storage: RunStorage,
        clip: Any,
        entities_by_id: Mapping[str, Any],
        pre_entities: Sequence[EntityReferenceState],
        marker: Mapping[str, Any],
    ) -> None:
        """Re-derive one entity's CPU branch from frozen state and require it.

        The plan's pre-reference image is re-read from disk, the legacy
        diagnostics and gates are recomputed, and the durable marker must agree
        exactly: this is the drift detector for one entity.
        """
        entity_id = str(marker["entity_id"])
        reference = next(item for item in pre_entities if item.entity_id == entity_id)
        entity = entities_by_id[entity_id]
        final_image = _load_reference_image(storage, str(reference.image_path))
        diagnostics = reference_topology_diagnostics(final_image)
        if marker["diagnostics"] != diagnostics.model_dump(mode="json"):
            raise ReferenceIntegrityDurableError(
                f"entity outcome diagnostics drifted for {clip.clip_uid}/{entity_id}"
            )
        policy_reason = reference_semantic_hard_reject_reason(
            reference_type=entity.reference_type,
            phrase=entity.phrase,
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
        if policy_reason is not None:
            expected: tuple[str, str, str | None] = (
                ENTITY_OUTCOME_REJECTED,
                policy_reason,
                policy_reason,
            )
        elif not requires_review:
            expected = (ENTITY_OUTCOME_SKIPPED, CLEAN_REAL_FULL_REFERENCE, None)
        else:
            raise ReferenceIntegrityDurableError(
                f"entity {clip.clip_uid}/{entity_id} carries a CPU outcome but "
                "now requires a Qwen review"
            )
        actual = (
            marker["outcome"],
            marker["reason"],
            marker.get("semantic_policy_reason"),
        )
        if actual != expected:
            raise ReferenceIntegrityDurableError(
                f"entity outcome branch drifted for {clip.clip_uid}/{entity_id}"
            )
        if marker["final_reference_path"] != reference.image_path:
            raise ReferenceIntegrityDurableError(
                f"entity outcome final reference drifted for "
                f"{clip.clip_uid}/{entity_id}"
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
        markers: dict[str, dict[str, Any]] = {}
        for entity_id in retained:
            marker = self._entity_outcome(shard, clip_uid, entity_id)
            if marker is None:
                raise ReferenceIntegrityDurableError(
                    f"clip {clip_uid!r} cannot be reconstructed: entity "
                    f"{entity_id!r} has no durable outcome"
                )
            self._verify_entity_branch(
                storage, clip, entities_by_id, pre_entities, marker
            )
            markers[entity_id] = marker
        rejected_ids = {
            entity_id
            for entity_id, marker in markers.items()
            if marker["outcome"] == ENTITY_OUTCOME_REJECTED
        }
        final_references = [
            (
                _rejected_reference(item, REJECTED_REFERENCE_REASON)
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
        pre_edit = plan_entry.get("pre_reference_edit")
        reference_edit = (
            ReferenceEditState.model_validate(pre_edit)
            if pre_edit is not None
            else None
        )
        integrity_state = ReferenceIntegrityState(
            status="ready",
            entities=[
                self._entity_state(
                    markers[entity_id],
                    next(
                        item
                        for item in pre_entities
                        if item.entity_id == entity_id
                    ),
                )
                for entity_id in retained
            ],
        )
        # 6b never mutates Reference Edit state: the frozen baseline IS the
        # publication, so a drift here is corruption, not policy.
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
