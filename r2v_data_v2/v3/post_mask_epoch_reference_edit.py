"""Durable Reference Edit resource epoch (5b).

The frozen legacy authority stays ``reference_edit_clips()``: this runner
replays the exact same clip policy and reuses the 5a split model-call helpers
(``prepare_boogu_reference_edit_attempt``,
``run_boogu_reference_edit_generation``, ``run_boogu_qwen_review``,
``run_boogu_sam_review``, ``finalize_boogu_reference_edit_attempt``) plus the
CPU policy helpers of ``reference_edit.py``. No prompt, review rule, SAM
threshold, candidate2 rule or fallback rule is copied.

Differences from the legacy adapter are execution-only:

* every model call is a durable ModelJob committed by the scheduler;
* Boogu seeds come from a create-once durable seed plan, never from a fresh
  ``new_boogu_seed()`` on replay;
* generated candidates are stored as ledger artifacts first and only
  materialized to the frozen production paths by receipt-verified CPU steps,
  so a crash before a receipt can never strand a production candidate;
* an attempt finalizes with ``publish_final=False``; the final reference is
  published by clip-level CPU publication, exactly like legacy.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    JOB_SCHEMA_VERSION,
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    JobResult,
    ModelJob,
    semantic_input_digest,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
from r2v_data_v2.v3.reference_edit import (
    _alternate_completion_source,
    _disabled_entity_background_variant,
    _instruction,
    _instruction_rewrite_enabled,
    _operations,
    _reference_content_geometry,
    _resolve_artifact,
    _route,
    _source_gate_reason,
    _tokens_for_retained,
    _variant,
    _variant_state,
    _write_source_selection_metadata,
)
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguReferenceEditResult,
    BooguSamReview,
    finalize_boogu_reference_edit_attempt,
    prepare_boogu_reference_edit_attempt,
    run_boogu_qwen_review,
    run_boogu_reference_edit_generation,
    run_boogu_sam_review,
)
from r2v_data_v2.v3.schemas import (
    PairingState,
    ReferenceEditEntityState,
    ReferenceEditState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage

REFERENCE_EDIT_PHASE = "reference_edit"
REFERENCE_EDIT_BOOGU_GENERATE_JOB = "reference_edit_boogu_generate"
REFERENCE_EDIT_QWEN_REVIEW_JOB = "reference_edit_qwen_review"
REFERENCE_EDIT_SAM_REVIEW_JOB = "reference_edit_sam_review"
REFERENCE_EDIT_PLAN_SCHEMA = "post_mask_epoch_reference_edit_plan/1"
REFERENCE_EDIT_SEED_SCHEMA = "post_mask_epoch_reference_edit_seed/1"
REFERENCE_EDIT_ALTERNATE_SCHEMA = "post_mask_epoch_reference_edit_alternate/1"
REFERENCE_EDIT_ENTITY_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_edit_entity_outcome/1"
)
REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA = "post_mask_epoch_reference_edit_outcome/1"
REFERENCE_EDIT_ATTEMPT_OUTCOME_SCHEMA = (
    "post_mask_epoch_reference_edit_attempt_outcome/1"
)
#: Statuses the attempt-outcome writer can actually produce.
ATTEMPT_OUTCOME_STATUSES = (
    "accepted",
    "rejected",
    "generation_failed",
    "qwen_failed",
    "sam_failed",
)

CLIP_FRESH_TARGET = "fresh_target"
CLIP_EXISTING = "existing"
CLIP_INELIGIBLE = "ineligible"

#: Legacy model-call exception set. These stay *semantic* attempt failures
#: (candidate rejection, candidate2 fallback), never scheduler retries.
LEGACY_MODEL_EXCEPTIONS = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)


class ReferenceEditEpochError(RuntimeError):
    """Raised when durable Reference Edit state cannot be trusted."""


class ReferenceEditDurableError(ReferenceEditEpochError):
    """Durable corruption/drift: never semantic-failed, always fail closed."""


class _JobIdRef:
    """Minimal job reference: the GroupLedger index only needs ``job_id()``."""

    __slots__ = ("_job_id",)

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def job_id(self) -> str:
        return self._job_id


@dataclass(frozen=True)
class ResolvedReferenceEditJudge:
    """A judge handle plus whether *this* call created it."""

    judge: Any
    owned: bool


def resolve_reference_edit_judge(
    handle: Any, config: Any
) -> ResolvedReferenceEditJudge:
    """Resolve the shared Qwen executor's handle into a review judge.

    Production passes the shared vLLM endpoint string; the judge is then built
    against that endpoint with the configured model and is owned (closed) by
    the caller. An injected judge object is reused as-is and never closed.
    """
    from r2v_data_v2.v3.reference_edit_boogu import QwenBooguReferenceEditJudge

    if handle is None:
        raise ReferenceEditEpochError("qwen review epoch supplied no handle")
    if isinstance(handle, str):
        service = config.qwen.reference_edit_judge
        if service is None:
            raise ReferenceEditEpochError(
                "no configured reference edit judge for the endpoint handle"
            )
        from dataclasses import replace as _replace

        return ResolvedReferenceEditJudge(
            QwenBooguReferenceEditJudge(
                _replace(service, base_url=handle)
            ),
            owned=True,
        )
    return ResolvedReferenceEditJudge(handle, owned=False)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Durable reader: missing is ``None``, corrupt is always fail-closed.

    A durable authority file that exists but cannot be read, cannot be parsed,
    or does not hold a JSON object is corruption -- never an empty state. The
    exception is always :class:`ReferenceEditDurableError` so no caller can
    turn it into a semantic clip failure.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReferenceEditDurableError(
            f"invalid durable Reference Edit JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReferenceEditDurableError(
            f"durable Reference Edit JSON is not an object: {path}"
        )
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create-or-validate with the epoch durable-state fsync writer."""
    from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_json

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise ReferenceEditDurableError(
                f"durable Reference Edit file drifted: {path}"
            )
        return
    atomic_write_json(path, dict(payload))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


#: The frozen Boogu seed contract: ``new_boogu_seed()`` draws
#: ``secrets.randbelow(2**31)``, so any other value is durable corruption.
BOOGU_SEED_UPPER_BOUND = 2**31


def _validated_boogu_seed(seed: Any, *, clip_uid: str, entity_id: str) -> int:
    """Validate one durable seed. A malformed seed is durable corruption."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReferenceEditDurableError(
            f"durable Boogu seed for {clip_uid}/{entity_id} is not an int"
        )
    if not 0 <= seed < BOOGU_SEED_UPPER_BOUND:
        raise ReferenceEditDurableError(
            f"durable Boogu seed for {clip_uid}/{entity_id} is out of range"
        )
    return seed


class ReferenceEditEpochRunner:
    """Durable Reference Edit chain for one resource-epoch group."""

    def __init__(
        self,
        config: Any,
        storages: Mapping[str, RunStorage],
        ledger: GroupLedger,
        *,
        eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
        emit: Any = None,
        review_execution: Literal["sequential", "parallel_independent"] = "sequential",
    ) -> None:
        if not config.reference_edit.enabled:
            raise ReferenceEditEpochError(
                "Reference Edit resource epoch requires reference_edit.enabled"
            )
        if review_execution not in {"sequential", "parallel_independent"}:
            raise ValueError(
                f"unsupported review execution: {review_execution}"
            )
        self.config = config
        self.storages = dict(storages)
        self.ledger = ledger
        self.eligible = {
            shard: tuple(uids) for shard, uids in eligible_clip_uids_by_shard.items()
        }
        self.emit = emit or (lambda *args, **kwargs: None)
        self.review_execution = review_execution

    # -- durable roots -----------------------------------------------------

    def _semantic(self, *parts: str) -> Path:
        return Path(self.ledger.root).joinpath("semantic", "reference_edit", *parts)

    def _plan_path(self, shard: str) -> Path:
        return self._semantic("primary", f"{shard}.json")

    def _seed_path(self, shard: str, clip_uid: str, entity_id: str, index: int) -> Path:
        return self._semantic(
            "seeds", shard, clip_uid, entity_id, f"attempt-{index}.json"
        )

    def _alternate_path(self, shard: str, clip_uid: str, entity_id: str) -> Path:
        return self._semantic("alternate_sources", shard, clip_uid, f"{entity_id}.json")

    def _entity_outcome_path(
        self, shard: str, clip_uid: str, entity_id: str
    ) -> Path:
        return self._semantic(
            "entity_outcomes", shard, clip_uid, f"{entity_id}.json"
        )

    def _clip_outcome_path(self, shard: str, clip_uid: str) -> Path:
        return self._semantic("outcomes", shard, f"{clip_uid}.json")

    def _routing_path(self, shard: str, clip_uid: str, entity_id: str) -> Path:
        return self._semantic("routing", shard, clip_uid, f"{entity_id}.json")

    def _storage_for(self, shard: str) -> RunStorage:
        storage = self.storages.get(shard)
        if storage is None:
            raise ReferenceEditEpochError(
                f"no run storage for canonical shard {shard!r}"
            )
        return storage

    # -- frozen launch plan -------------------------------------------------

    def _clip_classification(self, clip: Any) -> str:
        if (
            clip.annotation is None
            or clip.annotation.status != "ready"
            or clip.coverage is None
            or not clip.coverage.passed
            or clip.pairing is None
            or clip.pairing.status != "ready"
        ):
            return CLIP_INELIGIBLE
        if clip.reference_edit is not None:
            return CLIP_EXISTING
        return CLIP_FRESH_TARGET

    def _chain_entity_ids(self, clip: Any) -> tuple[str, ...]:
        """Entities whose terminal outcome the clip chain must produce."""
        retained = set(clip.pairing.retained_entity_ids)
        return tuple(
            reference.entity_id
            for reference in clip.references.entities
            if reference.status == "ready" and reference.entity_id in retained
        )

    def _policy_identity(self) -> dict[str, Any]:
        reference_edit = self.config.reference_edit
        judge = self.config.qwen.reference_edit_judge
        return {
            "target_area": reference_edit.target_area,
            "alignment": reference_edit.alignment,
            "min_source_content_area_pixels": (
                reference_edit.min_source_content_area_pixels
            ),
            "min_source_content_long_side_pixels": (
                reference_edit.min_source_content_long_side_pixels
            ),
            "model_revision": reference_edit.model_revision,
            "model_path": str(reference_edit.model_path),
            "fallback_policy": reference_edit.fallback_policy,
            "completion_instruction_rewrite_enabled": (
                reference_edit.completion_instruction_rewrite_enabled
            ),
            "sam_max_area_growth_ratio": reference_edit.sam_max_area_growth_ratio,
            "sam_max_significant_components": (
                reference_edit.sam_max_significant_components
            ),
            "min_candidate_scale_ratio": reference_edit.min_candidate_scale_ratio,
            "max_candidate_center_shift": reference_edit.max_candidate_center_shift,
            "review_execution": self.review_execution,
            "reference_edit_judge_model": (
                str(judge.model) if judge is not None else None
            ),
            "crop_padding_ratio": self.config.pair.crop_padding_ratio,
        }

    def _clip_digest(self, storage: RunStorage, clip: Any) -> str:
        """Digest of the publication-stable inputs (annotation/coverage/frames).

        ``references`` and ``pairing`` are deliberately excluded: a successful
        completion publication legitimately rewrites them, so restart must not
        compare them here (see ``_verify_plan_entry``).
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
            "sampled_frames": storage
            .read_frames(clip.clip_uid)
            .model_dump(mode="json"),
            "tracked_masks": storage.read_masks(clip.clip_uid).model_dump(mode="json"),
            "policy": self._policy_identity(),
        }
        return semantic_input_digest(projection)

    def _plan_entry(self, storage: RunStorage, clip: Any) -> dict[str, Any]:
        classification = self._clip_classification(clip)
        return {
            "classification": classification,
            "digest": self._clip_digest(storage, clip),
            "chain_entity_ids": (
                list(self._chain_entity_ids(clip))
                if classification == CLIP_FRESH_TARGET
                else []
            ),
            "pre_pairing": (
                clip.pairing.model_dump(mode="json")
                if clip.pairing is not None
                else None
            ),
            "pre_references": clip.references.model_dump(mode="json"),
        }

    def _verify_plan_entry(
        self, shard: str, storage: RunStorage, clip_uid: str, entry: dict[str, Any]
    ) -> None:
        """Validate one frozen plan entry against the live clip.

        A clip that already published its Reference Edit result is verified
        against its durable outcome instead of the pre-edit baseline: the
        publication legitimately rewrites ``references``/``pairing``.
        """
        clip = storage.read_clip(clip_uid)
        actual = self._clip_digest(storage, clip)
        if actual != entry.get("digest"):
            raise ReferenceEditDurableError(
                f"frozen Reference Edit input drifted for {clip_uid!r}"
            )
        if entry.get("classification") != CLIP_FRESH_TARGET:
            return
        if clip.reference_edit is not None:
            # Published: reconstruct the expected publication and verify the
            # live state against it, with or without the marker present.
            self._verify_published_clip(shard, storage, clip_uid, entry)
            return
        if self._clip_outcome_path(shard, clip_uid).is_file():
            raise ReferenceEditDurableError(
                f"clip {clip_uid!r} has a durable outcome marker but no "
                "published Reference Edit state"
            )
        pre_pairing = (
            clip.pairing.model_dump(mode="json") if clip.pairing is not None else None
        )
        if pre_pairing != entry.get("pre_pairing"):
            raise ReferenceEditDurableError(
                f"frozen Reference Edit pairing drifted for {clip_uid!r}"
            )
        if clip.references.model_dump(mode="json") != entry.get("pre_references"):
            raise ReferenceEditDurableError(
                f"frozen Reference Edit references drifted for {clip_uid!r}"
            )

    def _plan(self, shard: str) -> dict[str, Any]:
        existing = _read_json(self._plan_path(shard))
        if existing is not None:
            expected = {
                "schema": REFERENCE_EDIT_PLAN_SCHEMA,
                "canonical_shard": shard,
                "eligible_clip_uids": list(self.eligible.get(shard, ())),
                "policy": self._policy_identity(),
            }
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ReferenceEditDurableError(
                        f"frozen Reference Edit plan {key} drifted for {shard!r}"
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
            "schema": REFERENCE_EDIT_PLAN_SCHEMA,
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
            raise ReferenceEditEpochError(
                f"reconcile_stats needs a frozen Reference Edit plan for {shard!r}"
            )
        expected = {
            "schema": REFERENCE_EDIT_PLAN_SCHEMA,
            "canonical_shard": shard,
            "eligible_clip_uids": list(self.eligible.get(shard, ())),
            "policy": self._policy_identity(),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ReferenceEditDurableError(
                    f"frozen Reference Edit plan {key} drifted for {shard!r}"
                )
        storage = self._storage_for(shard)
        for clip_uid, entry in payload.get("clips", {}).items():
            self._verify_plan_entry(shard, storage, clip_uid, entry)
        return payload

    def _entity_anchor(
        self,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
        attempt_index: int,
        *,
        alternate: Any = None,
    ) -> dict[str, Any]:
        canonical_sha256 = _sha256_bytes(
            _resolve_artifact(storage, reference.image_path).read_bytes()
        )
        from dataclasses import asdict

        geometry_sha256 = _sha256_bytes(
            json.dumps(
                asdict(
                    _reference_content_geometry(storage, reference)
                ),
                sort_keys=True,
            ).encode()
        )
        source_sha256 = canonical_sha256
        comparison_sha256 = None
        source_candidate_id = "canonical_reference"
        source_frame_index = reference.source_frame_index
        if alternate is not None:
            source_sha256 = _sha256_bytes(alternate.image_path.read_bytes())
            comparison_sha256 = canonical_sha256
            source_candidate_id = alternate.candidate.candidate_id
            source_frame_index = alternate.candidate.source_frame_index
        instruction = _instruction("complete_entity", entity)
        if alternate is not None:
            with Image.open(alternate.image_path) as opened:
                opened.load()
                source_rgba = opened.convert("RGBA")
            width, height = self._resolved_size_rgba(source_rgba)
        else:
            width, height = self._resolved_size(storage, reference)
        return {
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "operation": "complete_entity",
            "instruction": instruction,
            "entity_phrase": entity.phrase,
            "grounding_prompt": entity.grounding_prompt or entity.phrase,
            "reference_type": entity.reference_type,
            "canonical_source_sha256": canonical_sha256,
            "source_sha256": source_sha256,
            "geometry_source_sha256": (
                source_sha256 if alternate is None else canonical_sha256
            ),
            "geometry_content_digest": geometry_sha256,
            "comparison_source_sha256": comparison_sha256,
            "attempt_index": attempt_index,
            "candidate_source_id": source_candidate_id,
            "candidate_source_frame_index": source_frame_index,
            "resolved_width": width,
            "resolved_height": height,
            "thinking_enabled": True,
            "instruction_rewrite_enabled": _instruction_rewrite_enabled(
                self.config, "complete_entity"
            ),
            "target_area": self.config.reference_edit.target_area,
            "alignment": self.config.reference_edit.alignment,
            "boogu_model_revision": self.config.reference_edit.model_revision,
            "boogu_model_path": str(self.config.reference_edit.model_path),
        }

    def _resolved_size(self, storage: RunStorage, reference: Any) -> tuple[int, int]:
        with Image.open(_resolve_artifact(storage, reference.image_path)) as opened:
            opened.load()
            rgba = opened.convert("RGBA")
        return self._resolved_size_rgba(rgba)

    def _resolved_size_rgba(self, rgba: Image.Image) -> tuple[int, int]:
        from r2v_data_v2.v3.reference_edit_boogu import resolve_boogu_1k_size

        return resolve_boogu_1k_size(
            rgba.width,
            rgba.height,
            target_area=self.config.reference_edit.target_area,
            alignment=self.config.reference_edit.alignment,
        )

    def durable_seed(
        self,
        shard: str,
        clip_uid: str,
        entity_id: str,
        anchor: Mapping[str, Any],
        attempt_index: int,
    ) -> int:
        """Load the create-once durable seed for this attempt anchor."""
        path = self._seed_path(shard, clip_uid, entity_id, attempt_index)
        anchor_digest = semantic_input_digest(dict(anchor))
        existing = _read_json(path)
        if existing is not None:
            if (
                existing.get("schema") != REFERENCE_EDIT_SEED_SCHEMA
                or existing.get("anchor_digest") != anchor_digest
                or existing.get("attempt_index") != attempt_index
            ):
                raise ReferenceEditDurableError(
                    f"durable Boogu seed plan drifted for {clip_uid}/{entity_id} "
                    f"attempt {attempt_index}"
                )
            return _validated_boogu_seed(
                existing.get("seed"), clip_uid=clip_uid, entity_id=entity_id
            )
        seed = _draw_boogu_seed()
        _write_json_once(
            path,
            {
                "schema": REFERENCE_EDIT_SEED_SCHEMA,
                "shard": shard,
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "attempt_index": attempt_index,
                "anchor_digest": anchor_digest,
                "seed": seed,
            },
        )
        return seed

    # -- frozen alternate (candidate2) source plan ---------------------------

    def durable_alternate(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
    ) -> Any:
        """Resolve (once) and freeze the candidate2 alternate source.

        Once the plan is frozen the legacy re-selection is NEVER run again:
        the frozen candidate identity, the canonical SHA and the materialized
        ``alternate_source_2.png`` SHA are re-verified against disk, and any
        drift fails closed instead of being silently regenerated.
        """
        path = self._alternate_path(shard, clip_uid, entity.entity_id)
        existing = _read_json(path)
        if existing is not None:
            return self._rebuild_frozen_alternate(
                shard, storage, clip_uid, entity, reference, existing
            )
        alternate = _alternate_completion_source(
            self.config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            canonical_reference=reference,
        )
        if alternate is None:
            return None
        source_image_path = _resolve_artifact(
            storage, alternate.candidate.image_path
        )
        identity = {
            "schema": REFERENCE_EDIT_ALTERNATE_SCHEMA,
            "canonical_source_sha256": _sha256_bytes(
                _resolve_artifact(storage, reference.image_path).read_bytes()
            ),
            "candidate_id": alternate.candidate.candidate_id,
            "source_frame_index": alternate.candidate.source_frame_index,
            "source_image_path": alternate.candidate.image_path,
            "source_image_sha256": _sha256_bytes(source_image_path.read_bytes()),
            "alternate_source_path": storage.relative_artifact_path(
                alternate.image_path
            ),
            "alternate_source_sha256": _sha256_bytes(
                alternate.image_path.read_bytes()
            ),
        }
        _write_json_once(path, identity)
        return alternate

    def _rebuild_frozen_alternate(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
        frozen: Mapping[str, Any],
    ) -> Any:
        """Validate the frozen alternate plan and rebuild its descriptor.

        The frozen plan is the authority for WHICH candidate was chosen, but
        the legacy materialized artifacts (``alternate_source_2.png`` and
        ``alternate_source_2.json``) plus the original candidate image are
        independent evidence that the plan itself was not tampered with or
        corrupted. Any mismatch fails closed; the candidate is never
        re-selected and no artifact is ever regenerated.
        """
        del shard
        if frozen.get("schema") != REFERENCE_EDIT_ALTERNATE_SCHEMA:
            raise ReferenceEditDurableError(
                f"unsupported alternate plan schema for {clip_uid}/"
                f"{entity.entity_id}"
            )
        canonical_sha = _sha256_bytes(
            _resolve_artifact(storage, reference.image_path).read_bytes()
        )
        if frozen.get("canonical_source_sha256") != canonical_sha:
            raise ReferenceEditDurableError(
                f"frozen alternate canonical source drifted for {clip_uid}/"
                f"{entity.entity_id}"
            )
        alternate_rel = frozen.get("alternate_source_path")
        if not isinstance(alternate_rel, str):
            raise ReferenceEditDurableError(
                f"frozen alternate plan has no artifact path for {clip_uid}/"
                f"{entity.entity_id}"
            )
        alternate_path = _resolve_artifact(storage, alternate_rel)
        if not alternate_path.is_file():
            raise ReferenceEditDurableError(
                f"frozen alternate artifact is missing: {alternate_rel}"
            )
        if _sha256_bytes(alternate_path.read_bytes()) != frozen.get(
            "alternate_source_sha256"
        ):
            raise ReferenceEditDurableError(
                f"frozen alternate artifact drifted: {alternate_rel}"
            )
        # The legacy materialized metadata is an independent record of the
        # same decision and must agree with the frozen plan field for field.
        metadata_path = alternate_path.with_name("alternate_source_2.json")
        if not metadata_path.is_file():
            raise ReferenceEditDurableError(
                f"frozen alternate metadata is missing: {metadata_path}"
            )
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ReferenceEditDurableError(
                f"frozen alternate metadata is not an object: {metadata_path}"
            )
        if metadata.get("schema_version") != 1:
            raise ReferenceEditDurableError(
                f"unsupported alternate metadata schema: {metadata_path}"
            )
        expected_metadata = {
            "candidate_id": frozen.get("candidate_id"),
            "source_frame_index": frozen.get("source_frame_index"),
            "source_image_path": frozen.get("source_image_path"),
            "source_image_sha256": frozen.get("source_image_sha256"),
            "alternate_source_path": alternate_rel,
            "alternate_source_sha256": frozen.get("alternate_source_sha256"),
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ReferenceEditDurableError(
                    f"frozen alternate metadata {key} disagrees with the "
                    f"frozen plan for {clip_uid}/{entity.entity_id}"
                )
        # The original candidate image is a run artifact: it must still exist
        # and still hash to the frozen value.
        source_image_path = frozen.get("source_image_path")
        source_image_sha256 = frozen.get("source_image_sha256")
        if not isinstance(source_image_path, str) or not isinstance(
            source_image_sha256, str
        ):
            raise ReferenceEditDurableError(
                f"frozen alternate plan has no source image identity for "
                f"{clip_uid}/{entity.entity_id}"
            )
        resolved_source = _resolve_artifact(storage, source_image_path)
        if _sha256_bytes(resolved_source.read_bytes()) != source_image_sha256:
            raise ReferenceEditDurableError(
                f"frozen alternate source image drifted: {source_image_path}"
            )
        # Rebuild the legacy descriptor from frozen metadata only. The
        # candidate is never re-selected.
        from types import SimpleNamespace

        return SimpleNamespace(
            candidate=SimpleNamespace(
                candidate_id=str(frozen["candidate_id"]),
                source_frame_index=int(frozen["source_frame_index"]),
                image_path=str(source_image_path),
            ),
            image_path=alternate_path,
            metadata_path=metadata_path,
        )

    # -- entity/clip outcomes -------------------------------------------------

    def _write_entity_outcome(
        self,
        shard: str,
        clip_uid: str,
        outcome: Mapping[str, Any],
    ) -> None:
        payload = {"schema": REFERENCE_EDIT_ENTITY_OUTCOME_SCHEMA, **outcome}
        _write_json_once(
            self._entity_outcome_path(shard, clip_uid, str(outcome["entity_id"])),
            payload,
        )

    def _entity_outcome(
        self, shard: str, clip_uid: str, entity_id: str
    ) -> dict[str, Any] | None:
        return _read_json(self._entity_outcome_path(shard, clip_uid, entity_id))

    # -- prepared reconstruction ----------------------------------------------

    def _prepare_attempt(
        self,
        shard: str,
        clip_uid: str,
        entity: Any,
        reference: Any,
        attempt_index: int,
        *,
        alternate_image_path: Path | None,
        alternate: Any = None,
    ) -> Any:
        comparison: Path | None = None
        source_path: Path | None = None
        geometry_source: Path | None = None
        if attempt_index == 2:
            if alternate_image_path is None:
                raise ReferenceEditDurableError(
                    f"attempt 2 needs a frozen alternate for "
                    f"{clip_uid}/{entity.entity_id}"
                )
            source_path = alternate_image_path
            geometry_source = alternate_image_path
            comparison = _resolve_artifact(
                self._storage_for(shard), reference.image_path
            )
        return prepare_boogu_reference_edit_attempt(
            run_root=self._storage_for(shard).root,
            clip_uid=clip_uid,
            entity_id=entity.entity_id,
            operation="complete_entity",
            instruction=_instruction("complete_entity", entity),
            entity_phrase=entity.phrase,
            grounding_prompt=entity.grounding_prompt,
            reference_type=entity.reference_type,
            instruction_rewrite_enabled=_instruction_rewrite_enabled(
                self.config, "complete_entity"
            ),
            target_area=self.config.reference_edit.target_area,
            alignment=self.config.reference_edit.alignment,
            min_source_content_area_pixels=(
                self.config.reference_edit.min_source_content_area_pixels
            ),
            min_source_content_long_side_pixels=(
                self.config.reference_edit.min_source_content_long_side_pixels
            ),
            model_revision=self.config.reference_edit.model_revision,
            fallback_status=self.config.reference_edit.fallback_policy,
            source_image_path=source_path,
            geometry_source_image_path=geometry_source,
            comparison_source_image_path=comparison,
            completion_attempt_index=attempt_index,
            completion_source_candidate_id=(
                "canonical_reference"
                if attempt_index == 1 or alternate is None
                else alternate.candidate.candidate_id
            ),
            completion_source_frame_index=(
                reference.source_frame_index
                if attempt_index == 1 or alternate is None
                else alternate.candidate.source_frame_index
            ),
            publish_final=False,
            resume_existing_outputs=True,
        )

    # -- job construction ------------------------------------------------------

    def _generation_job(
        self,
        shard: str,
        clip_uid: str,
        entity: Any,
        reference: Any,
        attempt_index: int,
        *,
        seed: int,
        anchor: Mapping[str, Any],
    ) -> ModelJob:
        return ModelJob.create(
            job_type=REFERENCE_EDIT_BOOGU_GENERATE_JOB,
            resource=RESOURCE_BOOGU,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "anchor": dict(anchor),
                "seed": seed,
                "attempt_index": attempt_index,
            },
            model_identity=(
                f"boogu:{self.config.reference_edit.model_revision}:"
                f"{self.config.reference_edit.model_path}"
            ),
            target={
                "entity_id": entity.entity_id,
                "attempt_index": str(attempt_index),
            },
        )

    def _review_job(
        self,
        shard: str,
        clip_uid: str,
        entity: Any,
        attempt_index: int,
        *,
        job_type: str,
        resource: str,
        model_identity: str,
        generation_job_id: str,
        candidate_sha256: str,
    ) -> ModelJob:
        return ModelJob.create(
            job_type=job_type,
            resource=resource,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "entity_id": entity.entity_id,
                "entity_phrase": entity.phrase,
                "reference_type": entity.reference_type,
                "operation": "complete_entity",
                "attempt_index": attempt_index,
                "generation_job_id": generation_job_id,
                "candidate_sha256": candidate_sha256,
                "model_identity": model_identity,
            },
            model_identity=model_identity,
            dependencies={"generation": generation_job_id},
            target={
                "entity_id": entity.entity_id,
                "attempt_index": str(attempt_index),
            },
        )

    def _plan_record_index(self) -> dict[str, dict[str, Any]]:
        """Every planned Reference Edit job record of this group, by job_id."""
        seen: dict[str, dict[str, Any]] = {}
        for phase_id in self.ledger.phase_ids():
            phase = self.ledger.phase(phase_id)
            if not phase.plan_path.is_file():
                continue
            try:
                records = phase.read_plan()
            except Exception as exc:  # corruption is never empty
                raise ReferenceEditDurableError(
                    f"invalid Reference Edit phase plan in {phase_id}"
                ) from exc
            for record in records:
                if record.get("job_type") not in (
                    REFERENCE_EDIT_BOOGU_GENERATE_JOB,
                    REFERENCE_EDIT_QWEN_REVIEW_JOB,
                    REFERENCE_EDIT_SAM_REVIEW_JOB,
                ):
                    continue
                job_id = record.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    continue
                previous = seen.get(job_id)
                if previous is not None and previous != record:
                    raise ReferenceEditDurableError(
                        f"Reference Edit job {job_id} has drifting plan records"
                    )
                seen.setdefault(job_id, record)
        return seen

    def _job_from_plan_record(self, record: Mapping[str, Any]) -> ModelJob:
        """Rebuild the ModelJob its immutable plan record describes."""
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
            raise ReferenceEditDurableError(
                "Reference Edit plan record does not rebuild: "
                f"{record.get('job_id', '')}"
            )
        return job

    def _planned_record(self, job_id: str) -> dict[str, Any]:
        record = self._plan_record_index().get(job_id)
        if record is None:
            raise ReferenceEditDurableError(
                f"Reference Edit job {job_id} is not in any durable plan"
            )
        return record

    def _validated_committed(
        self, job_id: str
    ) -> tuple[ModelJob, Any, dict[str, Any]]:
        """Receipt-validated committed job: (job, result, payload).

        ``ledger.classify`` re-validates the receipt identity, artifact
        digests, result digest and outcome before anything is trusted.
        """
        from r2v_data_v2.v3.post_mask_epoch_state import STATE_MISMATCH

        job = self._job_from_plan_record(self._planned_record(job_id))
        state = self.ledger.classify(job)
        if state.state == STATE_MISMATCH:
            raise ReferenceEditDurableError(
                f"Reference Edit job {job_id} receipt mismatch: "
                f"{state.detail or ''}"
            )
        if not state.skippable:
            raise ReferenceEditEpochError(
                f"Reference Edit job {job_id} is not terminal; the stage is "
                "incomplete"
            )
        result = self.ledger.load_committed_result(job)
        if result is None:
            raise ReferenceEditDurableError(
                f"committed Reference Edit job {job_id} has no replayable result"
            )
        return job, result, dict(result.payload)

    def _phase_for(self, job_id: str) -> str:
        phase_id = self.ledger.phase_for(_JobIdRef(job_id))
        if phase_id is None:
            raise ReferenceEditDurableError(
                f"Reference Edit job {job_id} has no committed phase"
            )
        return phase_id

    def _committed_artifact(self, job_id: str, name: str) -> bytes:
        phase_id = self._phase_for(job_id)
        path = self.ledger.phase(phase_id).artifact_path(job_id, name)
        if not path.is_file():
            raise ReferenceEditDurableError(
                f"committed job {job_id} has no artifact {name!r}"
            )
        return path.read_bytes()

    def _committed_candidate_bytes(
        self, generation_job_id: str, expected_sha256: str
    ) -> bytes:
        """Receipt-validated candidate bytes for one committed generation."""
        _generation_job, _result, payload = self._validated_committed(
            generation_job_id
        )
        if payload.get("status") != "generated":
            raise ReferenceEditDurableError(
                f"generation {generation_job_id} did not commit a candidate"
            )
        candidate = self._committed_artifact(generation_job_id, "candidate.png")
        digest = _sha256_bytes(candidate)
        if digest != expected_sha256 or digest != str(
            payload.get("candidate_sha256", "")
        ):
            raise ReferenceEditDurableError(
                f"committed candidate artifact drifted for {generation_job_id}"
            )
        return candidate

    # -- chain replay ----------------------------------------------------------

    def seed_jobs(self) -> list[ModelJob]:
        """CPU fixed point over every fresh clip, then pending model jobs."""
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            plan = self._plan(shard)
            storage = self._storage_for(shard)
            for clip_uid in sorted(plan.get("clips", {})):
                entry = plan["clips"][clip_uid]
                if entry["classification"] != CLIP_FRESH_TARGET:
                    continue
                if self._clip_outcome_path(shard, clip_uid).is_file():
                    continue
                try:
                    jobs.extend(self._advance_clip(shard, storage, clip_uid))
                except ReferenceEditDurableError:
                    # Durable corruption is never a semantic clip failure.
                    raise
                except Exception as exc:  # noqa: BLE001 - legacy clip isolation
                    # Ordinary legacy CPU failure: terminal for THIS clip,
                    # other clips continue.
                    self._fail_clip_terminal(shard, storage, clip_uid, exc)
        jobs.sort(key=lambda job: job.job_id())
        self.ledger.phase(REFERENCE_EDIT_PHASE).write_plan(jobs)
        return jobs

    def _clip(self, storage: RunStorage, clip_uid: str) -> Any:
        clip = storage.read_clip(clip_uid)
        assert clip.annotation is not None
        return clip

    def _advance_clip(self, shard: str, storage: RunStorage, clip_uid: str) -> list[ModelJob]:
        """Chain every entity of one clip; publish the clip when terminal."""
        plan = self._existing_plan_for_reconcile(shard)
        clip = self._clip(storage, clip_uid)
        entities = {entity.entity_id: entity for entity in clip.annotation.entities}
        jobs: list[ModelJob] = []
        for entity_id in plan["clips"][clip_uid]["chain_entity_ids"]:
            if self._entity_outcome(shard, clip_uid, entity_id) is not None:
                continue
            reference = next(
                (
                    item
                    for item in clip.references.entities
                    if item.entity_id == entity_id
                ),
                None,
            )
            entity = entities[entity_id]
            assert reference is not None
            jobs.extend(
                self._advance_entity(shard, storage, clip_uid, entity, reference)
            )
        if not jobs:
            self._publish_clip_if_terminal(shard, storage, clip_uid)
        return jobs

    def _entity_route_context(
        self, storage: RunStorage, clip: Any, reference: Any
    ) -> tuple[Any, Any, Any]:
        source_geometry = _reference_content_geometry(storage, reference)
        gate_reason = _source_gate_reason(self.config, source_geometry)
        route = _route(
            reference,
            source_touches_boundary=source_geometry.touches_canvas_boundary,
        )
        return source_geometry, gate_reason, route

    def _advance_entity(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        entity: Any,
        reference: Any,
    ) -> list[ModelJob]:
        clip = self._clip(storage, clip_uid)
        source_geometry, gate_reason, route = self._entity_route_context(
            storage, clip, reference
        )
        entity_variant_route = bool(
            entity.reference_type in {"subject", "object"}
            and route in {"complete", "local_usable", "repairable"}
        )
        routing_delta: dict[str, int] = {}
        bbox_paths: dict[str, str | None] = {}
        if gate_reason is None and entity_variant_route:
            # Legacy deterministic source-bbox materialization (CPU only).
            from r2v_data_v2.v3.reference_edit import (
                _materialize_source_bbox,
                _source_evidence,
            )

            bbox_evaluation = None
            bbox_error: str | None = None
            try:
                evidence = _source_evidence(
                    storage, clip_uid=clip_uid, reference=reference
                )
                from r2v_data_v2.v3.reference_edit import _resolve_artifact

                with Image.open(
                    _resolve_artifact(storage, reference.image_path)
                ) as opened:
                    opened.load()
                    current_reference = opened.copy()
                bbox_evaluation = _materialize_source_bbox(
                    storage=storage,
                    clip_uid=clip_uid,
                    entity_id=reference.entity_id,
                    source_evidence=evidence,
                    current_reference=current_reference,
                    current_reference_path=_resolve_artifact(
                        storage, reference.image_path
                    ),
                    reference=reference,
                    original_review=None,
                    diagnostics=None,
                    crop_padding_ratio=self.config.pair.crop_padding_ratio,
                    trigger="variant_bbox_review",
                )
                routing_delta["bbox_variants_materialized"] = 1
                bbox_paths = {
                    "candidate_relative": (
                        bbox_evaluation.candidate_relative
                        if bbox_evaluation is not None
                        else None
                    ),
                    "metadata_relative": (
                        bbox_evaluation.metadata_relative
                        if bbox_evaluation is not None
                        else None
                    ),
                }
            except Exception as exc:  # noqa: BLE001 - alpha remains valid
                bbox_error = f"{type(exc).__name__}:{exc}"
            _write_json_once(
                self._routing_path(shard, clip_uid, entity.entity_id),
                {
                    "schema": "post_mask_epoch_reference_edit_routing/1",
                    "entity_variant_route": entity_variant_route,
                    "route": route,
                    "delta": routing_delta,
                    "bbox_paths": bbox_paths,
                    "bbox_error": bbox_error,
                },
            )
        if gate_reason is not None:
            _write_source_selection_metadata(
                storage,
                clip_uid=clip_uid,
                reference=reference,
                geometry=source_geometry,
                source_gate_reason=gate_reason,
                reason="tiny_source_entity",
            )
            self._write_entity_outcome(
                shard,
                clip_uid,
                {
                    "entity_id": entity.entity_id,
                    "outcome": "fallback",
                    "reason": "tiny_source_entity",
                    "route": route,
                    "delta": {
                        "entities_eligible": 1,
                        "entities_fallback": 1,
                    },
                },
            )
            return []
        if not _operations(route):
            delta = {
                "entities_eligible": 1,
                "entities_accepted": 1,
                **routing_delta,
            }
            self._write_entity_outcome(
                shard,
                clip_uid,
                {
                    "entity_id": entity.entity_id,
                    "outcome": "not_required",
                    "reason": "no_operation_required",
                    "route": route,
                    "delta": delta,
                },
            )
            return []
        anchor = self._entity_anchor(
            storage, clip_uid, entity, reference, attempt_index=1
        )
        seed = self.durable_seed(shard, clip_uid, entity.entity_id, anchor, 1)
        return [
            self._generation_job(
                shard,
                clip_uid,
                entity,
                reference,
                1,
                seed=seed,
                anchor=anchor,
            )
        ]

    # -- scheduler-facing run/finalize -----------------------------------------

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        if job.job_type == REFERENCE_EDIT_BOOGU_GENERATE_JOB:
            return self._run_generation(job, handle)
        if job.job_type == REFERENCE_EDIT_QWEN_REVIEW_JOB:
            return self._run_qwen(job, handle)
        if job.job_type == REFERENCE_EDIT_SAM_REVIEW_JOB:
            return self._run_sam(job, handle)
        raise ReferenceEditEpochError(f"unsupported job type {job.job_type!r}")

    def _attempt_context(
        self, job: ModelJob
    ) -> tuple[str, str, Any, Any, Any, int]:
        shard = job.canonical_shard
        clip_uid = job.clip_uid
        storage = self._storage_for(shard)
        clip = self._clip(storage, clip_uid)
        target = dict(job.target)
        entity_id = str(target["entity_id"])
        attempt_index = int(target["attempt_index"])
        entity = next(
            item for item in clip.annotation.entities if item.entity_id == entity_id
        )
        reference = next(
            item
            for item in clip.references.entities
            if item.entity_id == entity_id
        )
        return shard, clip_uid, storage, entity, reference, attempt_index

    def _rebuild_anchor(self, job: ModelJob, attempt_index: int) -> dict[str, Any]:
        """Re-derive the attempt anchor from durable plans. Drift fails closed."""
        shard, clip_uid, storage, entity, reference, _ = self._attempt_context(job)
        alternate = None
        if attempt_index == 2:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
            if alternate is None:
                raise ReferenceEditDurableError(
                    f"attempt 2 planned without a frozen alternate: "
                    f"{clip_uid}/{entity.entity_id}"
                )
        return self._entity_anchor(
            storage, clip_uid, entity, reference, attempt_index, alternate=alternate
        )

    def _rebuild_prepared(self, job: ModelJob, attempt_index: int) -> Any:
        """CPU replay prepare.

        The anchor is re-derived from durable state and validated against the
        durable seed plan's own anchor digest (the job's ``input_digest``
        belongs to this job's *own* semantic identity: generation binds
        anchor+seed, reviews bind the review identity). Drift fails closed.
        """
        anchor = self._rebuild_anchor(job, attempt_index)
        # durable_seed validates the frozen anchor digest (fail closed on
        # drift); the seed value itself is only needed by the generation job.
        self.durable_seed(
            job.canonical_shard,
            job.clip_uid,
            dict(job.target)["entity_id"],
            anchor,
            attempt_index,
        )
        shard, clip_uid, storage, entity, reference, _ = self._attempt_context(job)
        alternate = None
        if attempt_index == 2:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
        return self._prepare_attempt(
            shard,
            clip_uid,
            entity,
            reference,
            attempt_index,
            alternate_image_path=(
                alternate.image_path if alternate is not None else None
            ),
            alternate=alternate,
        )

    def _run_generation(self, job: ModelJob, handle: Any) -> JobResult:
        """Rebuild anchor+seed from durable state and compare input digest."""
        shard, clip_uid, storage, entity, reference, attempt_index = (
            self._attempt_context(job)
        )
        alternate = None
        if attempt_index == 2:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
        anchor = self._entity_anchor(
            storage, clip_uid, entity, reference, attempt_index, alternate=alternate
        )
        seed_path = self._seed_path(shard, clip_uid, entity.entity_id, attempt_index)
        existing_seed = _read_json(seed_path)
        if existing_seed is None:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit generation has no durable seed plan yet",
            )
        if existing_seed.get("schema") != REFERENCE_EDIT_SEED_SCHEMA:
            raise ReferenceEditDurableError(
                f"durable Boogu seed plan schema drifted for "
                f"{clip_uid}/{entity.entity_id}"
            )
        if existing_seed.get("anchor_digest") != semantic_input_digest(
            dict(anchor)
        ):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit generation anchor drifted",
            )
        seed = _validated_boogu_seed(
            existing_seed.get("seed"),
            clip_uid=clip_uid,
            entity_id=entity.entity_id,
        )
        inputs = {
            "anchor": dict(anchor),
            "seed": seed,
            "attempt_index": attempt_index,
        }
        if semantic_input_digest(inputs) != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit generation semantic identity drifted",
            )
        prepared = self._prepare_attempt(
            shard,
            clip_uid,
            entity,
            reference,
            attempt_index,
            alternate_image_path=(
                alternate.image_path if alternate is not None else None
            ),
        )
        try:
            generation = run_boogu_reference_edit_generation(
                prepared, handle, seed=seed, materialize_candidate=False
            )
        except LEGACY_MODEL_EXCEPTIONS as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "generation_failed",
                    "error": str(exc),
                    "seed": seed,
                },
            )
        return JobResult(
            OUTCOME_COMPLETED,
            artifacts={"candidate.png": generation.output.png_bytes},
            payload={
                "status": "generated",
                "seed": seed,
                "candidate_sha256": generation.output_sha256,
                "original_instruction": generation.output.original_instruction,
                "rewritten_instruction": generation.output.rewritten_instruction,
                "effective_instruction": generation.output.effective_instruction,
                "worker_metadata": generation.output.worker_metadata,
                "width": prepared.width,
                "height": prepared.height,
            },
        )

    def _review_identity(
        self, job: ModelJob
    ) -> tuple[str, str, dict[str, Any]]:
        """(generation_job_id, candidate_sha256, rebuilt semantic inputs)."""
        generation_job_id = str(dict(job.dependency_digests)["generation"])
        _gen_job, _result, generation_payload = self._validated_committed(
            generation_job_id
        )
        candidate_sha256 = str(generation_payload.get("candidate_sha256", ""))
        target = dict(job.target)
        attempt_index = int(target["attempt_index"])
        clip = self._clip(self._storage_for(job.canonical_shard), job.clip_uid)
        entity = next(
            item
            for item in clip.annotation.entities
            if item.entity_id == str(target["entity_id"])
        )
        inputs = {
            "entity_id": entity.entity_id,
            "entity_phrase": entity.phrase,
            "reference_type": entity.reference_type,
            "operation": "complete_entity",
            "attempt_index": attempt_index,
            "generation_job_id": generation_job_id,
            "candidate_sha256": candidate_sha256,
            "model_identity": job.model_identity,
        }
        del clip
        return generation_job_id, candidate_sha256, inputs

    def _run_qwen(self, job: ModelJob, handle: Any) -> JobResult:
        generation_job_id, candidate_sha256, inputs = self._review_identity(job)
        if semantic_input_digest(inputs) != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit qwen review semantic identity drifted",
            )
        candidate_bytes = self._committed_candidate_bytes(
            generation_job_id, candidate_sha256
        )
        prepared = self._rebuild_prepared(job, int(dict(job.target)["attempt_index"]))
        generation = _reconstructed_generation(
            self, generation_job_id, candidate_bytes, candidate_sha256
        )
        resolved = resolve_reference_edit_judge(handle, self.config)
        try:
            review = run_boogu_qwen_review(prepared, generation, resolved.judge)
        except LEGACY_MODEL_EXCEPTIONS as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={"status": "qwen_failed", "error": str(exc)},
            )
        finally:
            if resolved.owned:
                resolved.judge.close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "qwen_review": review.model_dump(mode="json"),
            },
        )

    def _run_sam(self, job: ModelJob, handle: Any) -> JobResult:
        from r2v_data_v2.v3.reference_edit_boogu import Sam3BooguReferenceReviewer

        generation_job_id, candidate_sha256, inputs = self._review_identity(job)
        if semantic_input_digest(inputs) != job.input_digest:
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit sam review semantic identity drifted",
            )
        candidate_bytes = self._committed_candidate_bytes(
            generation_job_id, candidate_sha256
        )
        prepared = self._rebuild_prepared(job, int(dict(job.target)["attempt_index"]))
        generation = _reconstructed_generation(
            self, generation_job_id, candidate_bytes, candidate_sha256
        )
        segmenter = handle
        reviewer = Sam3BooguReferenceReviewer(
            segmenter,
            temporary_root=self._storage_for(
                job.canonical_shard
            ).reference_edit_temporary_dir(),
            max_area_growth_ratio=(
                self.config.reference_edit.sam_max_area_growth_ratio
            ),
            max_significant_components=(
                self.config.reference_edit.sam_max_significant_components
            ),
            min_candidate_scale_ratio=(
                self.config.reference_edit.min_candidate_scale_ratio
            ),
            max_candidate_center_shift=(
                self.config.reference_edit.max_candidate_center_shift
            ),
        )
        try:
            review = run_boogu_sam_review(prepared, generation, reviewer)
        except LEGACY_MODEL_EXCEPTIONS as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={"status": "sam_failed", "error": str(exc)},
            )
        return JobResult(
            OUTCOME_COMPLETED,
            payload={"status": "review", "sam_review": review.model_dump(mode="json")},
        )

    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        if job.job_type == REFERENCE_EDIT_BOOGU_GENERATE_JOB:
            return self._finalize_generation(job, result)
        if job.job_type == REFERENCE_EDIT_QWEN_REVIEW_JOB:
            return self._finalize_qwen(job, result)
        if job.job_type == REFERENCE_EDIT_SAM_REVIEW_JOB:
            return self._finalize_sam(job)
        raise ReferenceEditEpochError(f"unsupported job type {job.job_type!r}")

    def _finalize_generation(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        payload = dict(result.payload)
        shard, _storage, entity, _reference, attempt_index = (
            self._generation_context(job)
        )
        judge = self.config.qwen.reference_edit_judge
        if judge is None:
            raise ReferenceEditEpochError("reference edit judge is not configured")
        if payload.get("status") == "generated":
            candidate_sha256 = str(payload["candidate_sha256"])
            qwen_job = self._review_job(
                shard,
                job.clip_uid,
                entity,
                attempt_index,
                job_type=REFERENCE_EDIT_QWEN_REVIEW_JOB,
                resource=RESOURCE_QWEN,
                model_identity=f"qwen:{judge.model}",
                generation_job_id=job.job_id(),
                candidate_sha256=candidate_sha256,
            )
            unlocked: list[ModelJob] = [qwen_job]
            if self.review_execution == "parallel_independent":
                unlocked.append(
                    self._review_job(
                        shard,
                        job.clip_uid,
                        entity,
                        attempt_index,
                        job_type=REFERENCE_EDIT_SAM_REVIEW_JOB,
                        resource=RESOURCE_SAM,
                        model_identity="sam3",
                        generation_job_id=job.job_id(),
                        candidate_sha256=candidate_sha256,
                    )
                )
            return unlocked
        # generation_failed: legacy semantic rejection; no reviews are paid.
        return self._complete_attempt(
            job,
            generation_job_id=job.job_id(),
            failure_kind="generation_failed",
        )

    def _generation_context(
        self, job: ModelJob
    ) -> tuple[str, RunStorage, Any, Any, int]:
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip = self._clip(storage, job.clip_uid)
        target = dict(job.target)
        entity_id = str(target["entity_id"])
        attempt_index = int(target["attempt_index"])
        entity = next(
            item for item in clip.annotation.entities if item.entity_id == entity_id
        )
        reference = next(
            item for item in clip.references.entities if item.entity_id == entity_id
        )
        return shard, storage, entity, reference, attempt_index

    def _finalize_qwen(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        payload = dict(result.payload)
        if payload.get("status") == "review":
            if self.review_execution == "parallel_independent":
                # The SAM job was unlocked with the Qwen job at generation
                # time; whichever review terminal arrives second finalizes.
                return self._maybe_finalize_attempt(job)
            attempt_index = int(dict(job.target)["attempt_index"])
            generation_job_id = str(dict(job.dependency_digests)["generation"])
            _gen_job, _result, gen_payload = self._validated_committed(
                generation_job_id
            )
            sam_job = self._review_job(
                job.canonical_shard,
                job.clip_uid,
                self._chain_entity(job),
                attempt_index,
                job_type=REFERENCE_EDIT_SAM_REVIEW_JOB,
                resource=RESOURCE_SAM,
                model_identity="sam3",
                generation_job_id=generation_job_id,
                candidate_sha256=str(gen_payload.get("candidate_sha256", "")),
            )
            return [sam_job]
        if self.review_execution == "parallel_independent":
            # A Qwen exception must not skip SAM: keep the failure durable and
            # let the SAM finalizer complete the attempt with both payloads.
            return self._maybe_finalize_attempt(job)
        return self._complete_attempt(
            job, generation_job_id=str(dict(job.dependency_digests)["generation"]),
            failure_kind="qwen_failed",
        )

    def _finalize_sam(self, job: ModelJob) -> Sequence[ModelJob]:
        if self.review_execution == "parallel_independent":
            return self._maybe_finalize_attempt(job)
        return self._complete_attempt(
            job, generation_job_id=str(dict(job.dependency_digests)["generation"]),
            failure_kind="none",
        )

    def _chain_entity(self, job: ModelJob) -> Any:
        _shard, _clip_uid, storage, _clip = (
            job.canonical_shard,
            job.clip_uid,
            self._storage_for(job.canonical_shard),
            self._clip(self._storage_for(job.canonical_shard), job.clip_uid),
        )
        target = dict(job.target)
        return next(
            item
            for item in self._clip(storage, job.clip_uid).annotation.entities
            if item.entity_id == str(target["entity_id"])
        )

    # -- attempt completion -----------------------------------------------------

    def _attempt_finalized_marker(self, generation_job_id: str) -> Path:
        return self._semantic(
            "accounted", "attempt_finalized", f"{generation_job_id}.json"
        )

    def _attempt_outcome_path(
        self, shard: str, clip_uid: str, entity_id: str, attempt_index: int
    ) -> Path:
        return self._semantic(
            "attempt_outcomes",
            shard,
            clip_uid,
            entity_id,
            f"attempt-{attempt_index}.json",
        )

    def _review_job_for(
        self, generation_job_id: str, job_type: str
    ) -> ModelJob | None:
        for record in self._plan_record_index().values():
            if record.get("job_type") != job_type:
                continue
            dependencies = {
                str(key): str(value)
                for key, value in (record.get("dependency_digests") or [])
            }
            if dependencies.get("generation") == generation_job_id:
                return self._job_from_plan_record(record)
        return None

    def _maybe_finalize_attempt(self, job: ModelJob) -> Sequence[ModelJob]:
        """parallel_independent: finalize once BOTH reviews are terminal.

        The attempt-outcome marker is written BEFORE ``_after_attempt`` runs,
        so the marker alone is NOT proof that the attempt's continuation
        completed: a crash in that window leaves the marker on disk with no
        entity outcome and no candidate2 unlocked. The continuation boundary is
        the entity outcome, so only a marker AND an entity outcome may short
        circuit; otherwise the review receipts are re-checked and the whole
        ``_complete_attempt`` replays (create-once marker, deterministic
        unlock, zero model calls).
        """
        generation_job_id = str(dict(job.dependency_digests)["generation"])
        target = dict(job.target)
        entity_id = str(target["entity_id"])
        attempt_index = int(target["attempt_index"])
        marker = self._validated_attempt_outcome(
            job.canonical_shard, job.clip_uid, entity_id, attempt_index
        )
        if marker is not None and self._entity_outcome(
            job.canonical_shard, job.clip_uid, entity_id
        ) is not None:
            return ()
        qwen_job = self._review_job_for(
            generation_job_id, REFERENCE_EDIT_QWEN_REVIEW_JOB
        )
        sam_job = self._review_job_for(
            generation_job_id, REFERENCE_EDIT_SAM_REVIEW_JOB
        )
        if qwen_job is None or sam_job is None:
            raise ReferenceEditDurableError(
                f"attempt {generation_job_id} is missing a planned review job"
            )
        qwen_state = self.ledger.classify(qwen_job)
        sam_state = self.ledger.classify(sam_job)
        if not qwen_state.skippable or not sam_state.skippable:
            # The sibling review is still outstanding; it finalizes the attempt.
            return ()
        return self._complete_attempt(
            job,
            generation_job_id=generation_job_id,
            failure_kind="none",
        )

    def _complete_attempt(
        self,
        job: ModelJob,
        *,
        generation_job_id: str,
        failure_kind: str,
    ) -> Sequence[ModelJob]:
        """Run the attempt CPU finalizer, then continue the clip chain."""
        generation_job, _result, generation_payload = self._validated_committed(
            generation_job_id
        )
        # The whole finalize is pure CPU on committed receipts, so replaying
        # it after a crash is safe: every durable write below is
        # create-once/idempotent and no model is ever paid again.
        qwen_payload: dict[str, Any] | None = None
        sam_payload: dict[str, Any] | None = None
        qwen_job = self._review_job_for(
            generation_job_id, REFERENCE_EDIT_QWEN_REVIEW_JOB
        )
        if qwen_job is not None:
            _job, _result, qwen_payload = self._validated_committed(qwen_job.job_id())
        sam_job = self._review_job_for(
            generation_job_id, REFERENCE_EDIT_SAM_REVIEW_JOB
        )
        if sam_job is not None:
            _job, _result, sam_payload = self._validated_committed(sam_job.job_id())
        shard, storage, entity, reference, attempt_index = (
            self._generation_context(generation_job)
        )
        if str(generation_payload.get("status", "")) == "generation_failed":
            attempt_result = self._attempt_failed_result(generation_job)
            failure_kind = "generation_failed"
        else:
            candidate_sha256 = str(generation_payload.get("candidate_sha256", ""))
            candidate_bytes = self._committed_candidate_bytes(
                generation_job_id, candidate_sha256
            )
            prepared = self._rebuild_prepared(generation_job, attempt_index)
            generation = _reconstructed_generation(
                self, generation_job_id, candidate_bytes, candidate_sha256
            )
            self._materialize_candidate(prepared.candidate_path, candidate_bytes)
            qwen_review = None
            override: str | None = None
            qwen_failed = (
                qwen_payload is not None and qwen_payload.get("status") != "review"
            )
            sam_failed = (
                sam_payload is not None and sam_payload.get("status") != "review"
            )
            if qwen_payload is not None and qwen_payload.get("status") == "review":
                qwen_review = _deserialize_qwen_review(
                    shard, qwen_payload["qwen_review"]
                )
            elif qwen_failed:
                override = (
                    f"boogu_reference_edit_failed: {qwen_payload.get('error')}"
                )
                failure_kind = "qwen_failed"
            sam_review = None
            if (
                not qwen_failed
                and sam_payload is not None
                and sam_payload.get("status") == "review"
            ):
                sam_review = BooguSamReview.model_validate(sam_payload["sam_review"])
            elif sam_failed and not qwen_failed:
                override = (
                    f"boogu_reference_edit_failed: {sam_payload.get('error')}"
                )
                failure_kind = "sam_failed"
            attempt_result = finalize_boogu_reference_edit_attempt(
                prepared,
                generation,
                qwen_review=qwen_review,
                qwen_review_skipped_reason=None,
                sam_review=sam_review,
                rejection_reason_override=override,
            )
        self._write_attempt_outcome(
            shard, job.clip_uid, entity.entity_id, attempt_index,
            attempt_result, failure_kind,
        )
        return self._after_attempt(
            generation_job,
            shard,
            storage,
            entity,
            reference,
            attempt_index,
            attempt_result,
            failure_kind=failure_kind,
            failure_reason=_rejection_reason_of(attempt_result),
        )

    def _write_attempt_outcome(
        self,
        shard: str,
        clip_uid: str,
        entity_id: str,
        attempt_index: int,
        attempt_result: BooguReferenceEditResult,
        failure_kind: str,
    ) -> None:
        _write_json_once(
            self._attempt_outcome_path(
                shard, clip_uid, entity_id, attempt_index
            ),
            {
                "schema": REFERENCE_EDIT_ATTEMPT_OUTCOME_SCHEMA,
                "attempt_index": attempt_index,
                "status": (
                    "accepted"
                    if attempt_result.status == "accepted"
                    else failure_kind if failure_kind != "none" else "rejected"
                ),
                "accepted": attempt_result.status == "accepted",
                "rejection_reason": _rejection_reason_of(attempt_result),
            },
        )

    def _materialize_candidate(self, destination: Path, payload: bytes) -> None:
        if destination.is_file():
            if _sha256_bytes(destination.read_bytes()) != _sha256_bytes(payload):
                raise ReferenceEditDurableError(
                    f"production candidate drifted: {destination}"
                )
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _attempt_failed_result(self, generation_job: ModelJob) -> BooguReferenceEditResult:
        """Re-run the attempt finalizer for a generation_failed attempt."""
        _job, result, payload = self._validated_committed(generation_job.job_id())
        del result
        prepared = self._rebuild_prepared(
            generation_job, int(dict(generation_job.target)["attempt_index"])
        )
        return finalize_boogu_reference_edit_attempt(
            prepared,
            None,
            qwen_review=None,
            qwen_review_skipped_reason=None,
            sam_review=None,
            rejection_reason_override=(
                f"boogu_reference_edit_failed: {payload.get('error')}"
            ),
        )

    def _validated_attempt_outcome(
        self,
        shard: str,
        clip_uid: str,
        entity_id: str,
        attempt_index: int,
    ) -> dict[str, Any] | None:
        """Durable per-attempt accounting marker, strictly validated.

        ``None`` means the attempt never reached a terminal outcome. Anything
        else must be exactly what the writer produces for this attempt, so a
        tampered marker can never skew the completion counters.
        """
        marker = _read_json(
            self._attempt_outcome_path(shard, clip_uid, entity_id, attempt_index)
        )
        if marker is None:
            return None
        label = f"{clip_uid}/{entity_id} attempt {attempt_index}"
        if marker.get("schema") != REFERENCE_EDIT_ATTEMPT_OUTCOME_SCHEMA:
            raise ReferenceEditDurableError(
                f"attempt outcome schema drifted for {label}"
            )
        if marker.get("attempt_index") != attempt_index:
            raise ReferenceEditDurableError(
                f"attempt outcome index drifted for {label}"
            )
        accepted = marker.get("accepted")
        if not isinstance(accepted, bool):
            raise ReferenceEditDurableError(
                f"attempt outcome accepted flag is not a bool for {label}"
            )
        status = marker.get("status")
        if status not in ATTEMPT_OUTCOME_STATUSES:
            raise ReferenceEditDurableError(
                f"attempt outcome status is unknown for {label}"
            )
        if accepted != (status == "accepted"):
            raise ReferenceEditDurableError(
                f"attempt outcome accepted flag disagrees with its status for "
                f"{label}"
            )
        if not isinstance(marker.get("rejection_reason"), str):
            raise ReferenceEditDurableError(
                f"attempt outcome rejection_reason is not a string for {label}"
            )
        return marker

    def _verify_entity_attempt_consistency(
        self,
        shard: str,
        clip_uid: str,
        entity_id: str,
        entity_outcome: Mapping[str, Any],
        attempt_markers: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Cross-check the entity outcome against its attempt markers.

        Only the relationships the frozen candidate1/candidate2 chain always
        produces are asserted; there is no general state machine here.
        """
        attempt_index = entity_outcome.get("attempt_index")
        if not isinstance(attempt_index, int):
            return
        first = attempt_markers.get(1)
        second = attempt_markers.get(2)
        outcome = entity_outcome.get("outcome")
        if outcome == "accepted" and attempt_index == 1:
            if first is None or not first.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} is accepted on attempt 1 but its "
                    "attempt-1 marker disagrees"
                )
            if second is not None:
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} is accepted on attempt 1 but an "
                    "attempt-2 marker exists"
                )
            return
        if outcome == "fallback" and attempt_index == 1:
            if first is None or first.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} fell back on attempt 1 but its "
                    "attempt-1 marker disagrees"
                )
            if second is not None:
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} fell back on attempt 1 but an "
                    "attempt-2 marker exists"
                )
            return
        if outcome == "accepted" and attempt_index == 2:
            if second is None or not second.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} is accepted on attempt 2 but its "
                    "attempt-2 marker disagrees"
                )
            if first is None or first.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} reached attempt 2 without a "
                    "rejected attempt 1"
                )
            return
        if outcome == "fallback" and attempt_index == 2:
            if second is None or second.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} fell back on attempt 2 but its "
                    "attempt-2 marker disagrees"
                )
            if first is None or first.get("accepted"):
                raise ReferenceEditDurableError(
                    f"{clip_uid}/{entity_id} reached attempt 2 without a "
                    "rejected attempt 1"
                )

    def _routing_delta(self, shard: str, clip_uid: str, entity_id: str) -> dict[str, int]:
        routing = _read_json(self._routing_path(shard, clip_uid, entity_id)) or {}
        return {
            field: int(value)
            for field, value in (routing.get("delta") or {}).items()
        }

    def _after_attempt(
        self,
        generation_job: ModelJob,
        shard: str,
        storage: RunStorage,
        entity: Any,
        reference: Any,
        attempt_index: int,
        attempt_result: BooguReferenceEditResult,
        *,
        failure_kind: str,
        failure_reason: str,
    ) -> Sequence[ModelJob]:
        clip_uid = generation_job.clip_uid
        routing_delta = self._routing_delta(shard, clip_uid, entity.entity_id)
        if attempt_result.status == "accepted":
            self._write_entity_outcome(
                shard,
                clip_uid,
                {
                    "entity_id": entity.entity_id,
                    "outcome": "accepted",
                    "attempt_index": attempt_index,
                    "route": "repairable",
                    "delta": {
                        "entities_eligible": 1,
                        "entities_accepted": 1,
                        **routing_delta,
                    },
                },
            )
            self._publish_clip_if_terminal(shard, storage, clip_uid)
            return []
        if attempt_index == 1:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
            if alternate is not None:
                anchor = self._entity_anchor(
                    storage,
                    clip_uid,
                    entity,
                    reference,
                    attempt_index=2,
                    alternate=alternate,
                )
                seed = self.durable_seed(
                    shard, clip_uid, entity.entity_id, anchor, 2
                )
                return [
                    self._generation_job(
                        shard,
                        clip_uid,
                        entity,
                        reference,
                        2,
                        seed=seed,
                        anchor=anchor,
                    )
                ]
        # Final fallback to the canonical source alpha. Legacy also counts
        # entities_failed when the FINAL attempt failed through a model
        # exception, with one idempotent failure diagnostic.
        delta = {
            "entities_eligible": 1,
            "entities_fallback": 1,
            "completion_fallback_to_alpha": 1,
            **routing_delta,
        }
        if failure_kind != "none":
            delta["entities_failed"] = 1
            self._account_failure_diagnostic(
                shard,
                clip_uid=clip_uid,
                entity_id=entity.entity_id,
                reason=failure_reason,
            )
        self._write_entity_outcome(
            shard,
            clip_uid,
            {
                "entity_id": entity.entity_id,
                "outcome": "fallback",
                "reason": (
                    "repairable_completion_rejected:"
                    f"{failure_reason or 'completion_unavailable'}"
                ),
                "route": "repairable",
                "attempt_index": attempt_index,
                "delta": delta,
            },
        )
        self._publish_clip_if_terminal(shard, storage, clip_uid)
        return []

    def _account_failure_diagnostic(
        self,
        shard: str,
        *,
        clip_uid: str,
        entity_id: str,
        reason: str,
    ) -> None:
        """Append the legacy failure diagnostic exactly once per entity."""
        marker_path = self._semantic(
            "failure_accounted", shard, clip_uid, f"{entity_id}.json"
        )
        if marker_path.is_file():
            return
        storage = self._storage_for(shard)
        storage.append_failure(
            stage="reference_edit",
            clip_uid=clip_uid,
            reason=reason,
            details={"entity_id": entity_id, "operation": "complete_entity"},
        )
        _write_json_once(
            marker_path,
            {
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "reason": reason,
            },
        )

    # -- clip publication ---    # -- clip publication --------------------------------------------------------

    def _reconstruct_clip_publication(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan_entry: Mapping[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, int]]:
        """Pure-CPU expected final publication for one fresh clip.

        Rebuilds the legacy-equivalent ``(ReferencesState, PairingState,
        ReferenceEditState, stats delta)`` from the frozen pre-edit
        pairing/references, the entity outcomes, the routing metadata and the
        production completion metadata. No model is ever called; the final
        publication artifacts are (re)written with identical content only.
        Publication and restart verification share this helper.
        """
        from r2v_data_v2.v3.schemas import EntityReferenceState

        chain_ids = list(plan_entry.get("chain_entity_ids", []))
        outcomes = {
            entity_id: self._entity_outcome(shard, clip_uid, entity_id)
            for entity_id in chain_ids
        }
        if any(outcome is None for outcome in outcomes.values()):
            raise ReferenceEditDurableError(
                f"clip {clip_uid!r} cannot be reconstructed: an entity "
                "outcome is missing"
            )
        clip = self._clip(storage, clip_uid)
        entities = {entity.entity_id: entity for entity in clip.annotation.entities}
        pre_pairing = dict(plan_entry.get("pre_pairing") or {})
        pre_references = dict(plan_entry.get("pre_references") or {})
        pre_entity_states = [
            EntityReferenceState.model_validate(item)
            for item in pre_references.get("entities", [])
        ]
        retained = set(pre_pairing.get("retained_entity_ids", []))
        background = pre_references.get("background")
        final_references = []
        edit_states: list[ReferenceEditEntityState] = []
        delta: dict[str, int] = {}
        for reference in pre_entity_states:
            entity_id = reference.entity_id
            entity = entities.get(entity_id)
            if entity_id not in outcomes:
                final_references.append(reference)
                continue
            outcome = dict(outcomes[entity_id])
            for field, value in outcome.get("delta", {}).items():
                delta[field] = delta.get(field, 0) + int(value)
            source_geometry, gate_reason, route = self._entity_route_context(
                storage, clip, reference
            )
            initial_route = route
            if gate_reason is None:
                route = _route(
                    reference,
                    source_touches_boundary=(
                        source_geometry.touches_canvas_boundary
                    ),
                )
            entity_variant_route = bool(
                entity is not None
                and entity.reference_type in {"subject", "object"}
                and route in {"complete", "local_usable", "repairable"}
            )
            alpha_variant = (
                _variant(
                    image_path=reference.image_path,
                    status="accepted",
                    reviewed=True,
                    review_status="accepted_pair_reference",
                    reason="pair_accepted_source_alpha",
                    synthetic=False,
                    source_frame_index=reference.source_frame_index,
                )
                if entity_variant_route
                else None
            )
            routing = _read_json(self._routing_path(shard, clip_uid, entity_id))
            bbox_paths = dict((routing or {}).get("bbox_paths") or {})
            bbox_error = (routing or {}).get("bbox_error")
            bbox_variant = (
                _variant(
                    image_path=bbox_paths.get("candidate_relative"),
                    status=(
                        "available"
                        if bbox_paths.get("candidate_relative")
                        else "unavailable"
                    ),
                    reviewed=False,
                    review_status=(
                        "not_reviewed"
                        if bbox_paths.get("candidate_relative")
                        else "materialization_failed"
                    ),
                    reason=(
                        "source_bbox_materialized"
                        if bbox_paths.get("candidate_relative")
                        else bbox_error
                        or "source_bbox_unavailable"
                    ),
                    synthetic=False,
                    metadata_path=bbox_paths.get("metadata_relative"),
                    source_frame_index=reference.source_frame_index,
                )
                if entity_variant_route
                else None
            )
            generated_variant = (
                _disabled_entity_background_variant(reference)
                if entity_variant_route
                else None
            )
            variants = (
                _variant_state(
                    alpha=alpha_variant,
                    bbox=bbox_variant,
                    generated_background=generated_variant,
                )
                if alpha_variant is not None
                and bbox_variant is not None
                and generated_variant is not None
                else None
            )
            if (
                entity is not None
                and entity.reference_type not in {"subject", "object"}
                and reference.completeness is not None
                and not _operations(_route(reference))
            ):
                final_references.append(reference)
                edit_states.append(
                    ReferenceEditEntityState(
                        entity_id=entity_id,
                        route=initial_route,
                        status="not_required",
                        source_reference=reference,
                        source_image_path=reference.image_path,
                        output_image_path=reference.image_path,
                    )
                )
                continue
            if outcome["outcome"] == "fallback" and outcome.get("reason") == (
                "tiny_source_entity"
            ):
                metadata_path = _write_source_selection_metadata(
                    storage,
                    clip_uid=clip_uid,
                    reference=reference,
                    geometry=source_geometry,
                    source_gate_reason=gate_reason,
                    reason="tiny_source_entity",
                )
                final_references.append(reference)
                edit_states.append(
                    ReferenceEditEntityState(
                        entity_id=entity_id,
                        route=route,
                        status="fallback",
                        source_reference=reference,
                        source_image_path=reference.image_path,
                        output_image_path=reference.image_path,
                        variants=variants,
                        metadata_path=storage.relative_artifact_path(metadata_path),
                        fallback_policy="keep_source",
                        reason="tiny_source_entity",
                    )
                )
                continue
            if outcome["outcome"] == "not_required":
                final_references.append(reference)
                edit_states.append(
                    ReferenceEditEntityState(
                        entity_id=entity_id,
                        route=route,
                        status="not_required",
                        source_reference=reference,
                        source_image_path=reference.image_path,
                        output_image_path=reference.image_path,
                        variants=variants,
                        default_variant="alpha" if entity_variant_route else None,
                        default_image_path=(
                            reference.image_path if entity_variant_route else None
                        ),
                        default_reason=(
                            "source_alpha_preferred_by_policy"
                            if entity_variant_route
                            else None
                        ),
                    )
                )
                continue
            if outcome["outcome"] == "accepted":
                attempt_index = int(outcome.get("attempt_index", 1))
                edit_dir = storage.reference_edit_dir(clip_uid) / entity_id
                candidate_name = (
                    "completion_candidate_1k.png"
                    if attempt_index == 1
                    else "completion_candidate_2_1k.png"
                )
                metadata_path = edit_dir / (
                    "completion_metadata.json"
                    if attempt_index == 1
                    else "completion_metadata_2.json"
                )
                publication = publish_final_reference(
                    storage=storage,
                    clip_uid=clip_uid,
                    entity_id=entity_id,
                    candidate_path=edit_dir / candidate_name,
                    metadata_path=metadata_path,
                    attempt_index=attempt_index,
                )
                accepted = accepted_reference_from_publication(
                    storage, reference, clip_uid, publication
                )
                final_references.append(accepted)
                edit_states.append(
                    ReferenceEditEntityState(
                        entity_id=entity_id,
                        route=route,
                        status="accepted",
                        source_reference=reference,
                        source_image_path=reference.image_path,
                        output_image_path=accepted.image_path,
                        variants=variants,
                        default_variant="accepted_base",
                        default_image_path=accepted.image_path,
                        default_reason=(
                            "completion_candidate_"
                            f"{attempt_index}_review_accepted"
                        ),
                        accepted_base_image_path=accepted.image_path,
                        operation="complete_entity",
                        metadata_path=accepted.generation_metadata_path,
                        operations=["complete_entity"],
                        completion_metadata_path=storage.relative_artifact_path(
                            metadata_path
                        ),
                    )
                )
                continue
            final_references.append(reference)
            edit_states.append(
                ReferenceEditEntityState(
                    entity_id=entity_id,
                    route=route,
                    status="fallback",
                    source_reference=reference,
                    source_image_path=reference.image_path,
                    output_image_path=reference.image_path,
                    variants=variants,
                    default_variant="alpha" if entity_variant_route else None,
                    default_image_path=(
                        reference.image_path if entity_variant_route else None
                    ),
                    default_reason=(
                        "completion_rejected_fallback_to_source_alpha"
                        if entity_variant_route
                        else None
                    ),
                    accepted_base_image_path=(
                        reference.image_path if entity_variant_route else None
                    ),
                    operation="complete_entity",
                    metadata_path=storage.relative_artifact_path(
                        storage.reference_edit_dir(clip_uid)
                        / entity_id
                        / "final_metadata.json"
                    ),
                    operations=["complete_entity"],
                    completion_metadata_path=(
                        storage.relative_artifact_path(
                            storage.reference_edit_dir(clip_uid)
                            / entity_id
                            / (
                                "completion_metadata.json"
                                if int(outcome.get("attempt_index", 1)) == 1
                                else "completion_metadata_2.json"
                            )
                        )
                        if outcome.get("attempt_index")
                        else None
                    ),
                    fallback_policy="keep_source",
                    reason=str(outcome.get("reason", "")),
                )
            )
        retained_ids = [
            reference.entity_id
            for reference in final_references
            if reference.status == "ready" and reference.entity_id in retained
        ]
        qualifying = set(clip.coverage.qualifying_entity_ids)
        if not set(retained_ids).intersection(qualifying):
            pairing = PairingState(
                status="rejected", reason="no_qualifying_ready_reference"
            )
        else:
            pairing = PairingState(
                status="ready",
                retained_entity_ids=retained_ids,
                tokens=_tokens_for_retained(retained_ids, entities),
                background_token=pre_pairing.get("background_token"),
            )
        references_state = ReferencesState(
            entities=final_references, background=background
        )
        edit_state = ReferenceEditState(status="ready", entities=edit_states)
        outcome_delta = {
            "processed": 1,
            "failed": int(
                any(o["outcome"] == "failed" for o in outcomes.values() if o)
            ),
        }
        outcome_delta.update(delta)
        return references_state, pairing, edit_state, outcome_delta

    def _publish_clip_if_terminal(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> None:
        # Read-only plan access: the strict publication verification runs in
        # reconcile_stats AFTER the clip outcome marker exists, so calling it
        # here (mid-publication) would fail against its own ordering.
        plan = _read_json(self._plan_path(shard))
        if plan is None:
            raise ReferenceEditDurableError(
                f"Reference Edit publication needs the frozen plan for {shard!r}"
            )
        plan_entry = plan["clips"][clip_uid]
        if any(
            self._entity_outcome(shard, clip_uid, entity_id) is None
            for entity_id in plan_entry.get("chain_entity_ids", [])
        ):
            return
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return
        references_state, pairing, edit_state, outcome_delta = (
            self._reconstruct_clip_publication(shard, storage, clip_uid, plan_entry)
        )
        storage.write_reference_edit_result(
            clip_uid, references_state, pairing, edit_state
        )
        _write_json_once(
            self._clip_outcome_path(shard, clip_uid),
            {
                "schema": REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "ready",
                "delta": outcome_delta,
            },
        )

    def _fail_clip_terminal(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        exc: Exception,
    ) -> None:
        """Legacy clip isolation: one ordinary CPU failure is terminal.

        The legacy diagnostic is reproduced exactly: the failure reason is
        ``str(exc)`` and the diagnostic details carry the real exception type.
        """
        reason = str(exc)
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return
        storage.write_reference_edit_failure(clip_uid, reason)
        storage.append_failure(
            stage="reference_edit",
            clip_uid=clip_uid,
            reason=reason,
            details={"exception_type": type(exc).__name__},
        )
        _write_json_once(
            self._clip_outcome_path(shard, clip_uid),
            {
                "schema": REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "failed",
                "reason": reason,
                "delta": {"processed": 1, "failed": 1},
            },
        )

    def _verify_published_clip(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan_entry: Mapping[str, Any],
    ) -> None:
        """Verify a published clip against its durable outcome marker.

        The marker is a strict authority: its exact payload is derived from the
        verified publication and compared field for field. A crash between
        ``write_reference_edit_result``/``write_reference_edit_failure`` and the
        marker is repaired by back-filling that same exact payload with zero
        model calls; any other difference fails closed.
        """
        clip = storage.read_clip(clip_uid)
        marker_path = self._clip_outcome_path(shard, clip_uid)
        marker = _read_json(marker_path)
        if clip.reference_edit is not None and clip.reference_edit.status == "failed":
            expected_marker = {
                "schema": REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "failed",
                "reason": str(clip.reference_edit.reason or ""),
                "delta": {"processed": 1, "failed": 1},
            }
            if marker is None:
                _write_json_once(marker_path, expected_marker)
                return
            if marker != expected_marker:
                raise ReferenceEditDurableError(
                    f"clip {clip_uid!r} failure outcome marker does not match "
                    "the published failure"
                )
            return
        if clip.reference_edit is None:
            if marker is not None:
                raise ReferenceEditDurableError(
                    f"clip {clip_uid!r} has a durable outcome marker but no "
                    "published Reference Edit state"
                )
            raise ReferenceEditDurableError(
                f"clip {clip_uid!r} published Reference Edit state without a "
                "durable outcome marker"
            )
        references_state, pairing, edit_state, outcome_delta = (
            self._reconstruct_clip_publication(shard, storage, clip_uid, plan_entry)
        )
        if (
            clip.references.model_dump(mode="json")
            != references_state.model_dump(mode="json")
            or clip.pairing.model_dump(mode="json") != pairing.model_dump(mode="json")
            or clip.reference_edit.model_dump(mode="json")
            != edit_state.model_dump(mode="json")
        ):
            raise ReferenceEditDurableError(
                f"published Reference Edit state for {clip_uid!r} does not "
                "match its durable outcome"
            )
        expected_marker = {
            "schema": REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "terminal": "ready",
            "delta": outcome_delta,
        }
        if marker is None:
            # Crash window: the publication is verified; back-fill the marker.
            _write_json_once(marker_path, expected_marker)
            return
        if marker != expected_marker:
            raise ReferenceEditDurableError(
                f"clip {clip_uid!r} ready outcome marker does not match its "
                "durable publication"
            )

    # -- durable stats ---------------------------------------------------------

    def reconcile_stats(self, shard: str) -> Any:
        """ReferenceEditStats rebuilt from durable plan + outcome markers."""
        from r2v_data_v2.v3.reference_edit import ReferenceEditStats

        plan = self._existing_plan_for_reconcile(shard)
        counts = {field: 0 for field in ReferenceEditStats.__dataclass_fields__}
        for clip_uid, entry in sorted(plan.get("clips", {}).items()):
            classification = entry["classification"]
            if classification == CLIP_INELIGIBLE:
                counts["skipped_not_ready"] += 1
                continue
            if classification == CLIP_EXISTING:
                counts["skipped_existing"] += 1
                continue
            outcome = _read_json(self._clip_outcome_path(shard, clip_uid))
            if outcome is None:
                raise ReferenceEditEpochError(
                    f"Reference Edit clip {clip_uid!r} has no durable outcome; "
                    "the stage is not terminal"
                )
            if outcome.get("schema") != REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA:
                raise ReferenceEditDurableError(
                    f"clip outcome schema drifted for {clip_uid!r}"
                )
            if outcome.get("clip_uid") != clip_uid:
                raise ReferenceEditDurableError(
                    f"clip outcome clip_uid drifted for {clip_uid!r}"
                )
            if outcome.get("terminal") not in {"ready", "failed"}:
                raise ReferenceEditDurableError(
                    f"clip outcome terminal is unknown for {clip_uid!r}"
                )
            delta = outcome.get("delta")
            if not isinstance(delta, dict):
                raise ReferenceEditDurableError(
                    f"clip outcome delta is not an object for {clip_uid!r}"
                )
            for field, value in delta.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ReferenceEditDurableError(
                        f"clip outcome delta {field} is not an int for "
                        f"{clip_uid!r}"
                    )
                counts[field] = counts.get(field, 0) + value
            # Completion stats come from the durable per-attempt markers.
            for entity_id in entry.get("chain_entity_ids", []):
                markers: dict[int, dict[str, Any]] = {}
                for attempt_index in (1, 2):
                    marker = self._validated_attempt_outcome(
                        shard, clip_uid, entity_id, attempt_index
                    )
                    if marker is None:
                        continue
                    markers[attempt_index] = marker
                    counts["completion_attempts"] += 1
                    if attempt_index == 1 and marker["accepted"]:
                        counts["completion_candidate1_accepted"] += 1
                    if attempt_index == 2:
                        counts["completion_candidate2_attempts"] += 1
                        if marker["accepted"]:
                            counts["completion_candidate2_accepted"] += 1
                entity_outcome = self._entity_outcome(shard, clip_uid, entity_id)
                if entity_outcome is not None:
                    self._verify_entity_attempt_consistency(
                        shard, clip_uid, entity_id, entity_outcome, markers
                    )
        return ReferenceEditStats(**counts)


def publish_final_reference(
    *,
    storage: RunStorage,
    clip_uid: str,
    entity_id: str,
    candidate_path: Path,
    metadata_path: Path,
    attempt_index: int,
) -> Any:
    from r2v_data_v2.v3.reference_edit_boogu import publish_boogu_final_reference

    return publish_boogu_final_reference(
        run_root=storage.root,
        clip_uid=clip_uid,
        entity_id=entity_id,
        candidate_path=candidate_path,
        selected_metadata_path=metadata_path,
        completion_metadata_path=metadata_path,
        final_selection="completion_candidate",
        final_selection_reason=(
            "accepted_completion_candidate_"
            f"{attempt_index}_preferred_over_canonical_source_alpha"
        ),
    )


def accepted_reference_from_publication(
    storage: RunStorage,
    reference: Any,
    clip_uid: str,
    publication: Any,
) -> Any:
    from r2v_data_v2.v3.reference_edit import _accepted_reference

    return _accepted_reference(
        storage,
        reference,
        clip_uid=clip_uid,
        output_path=publication.final_reference_path,
        metadata_path=publication.final_metadata_path,
    )


class _GenerationOutputReplay:
    """Replay of the committed generation metadata fields."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.rewritten_instruction = payload.get("rewritten_instruction")
        self.effective_instruction = payload.get("effective_instruction")
        self.worker_metadata = dict(payload.get("worker_metadata") or {})
        self.original_instruction = str(payload.get("original_instruction", ""))


def _reconstructed_generation(
    runner: ReferenceEditEpochRunner,
    generation_job_id: str,
    candidate_bytes: bytes,
    candidate_sha256: str,
) -> Any:
    from types import SimpleNamespace

    _job, _result, payload = runner._validated_committed(generation_job_id)
    payload = dict(payload)
    from r2v_data_v2.v3.reference_edit_boogu import _validated_native_png

    candidate_rgb = _validated_native_png(
        candidate_bytes,
        expected_size=(int(payload.get("width", 0)), int(payload.get("height", 0))),
    )
    return SimpleNamespace(
        png_bytes=candidate_bytes,
        output_sha256=candidate_sha256,
        seed=int(payload.get("seed", 0)),
        candidate_rgb=candidate_rgb,
        output=_GenerationOutputReplay(payload),
    )


def _rejection_reason_of(result: BooguReferenceEditResult) -> str:
    from r2v_data_v2.v3.reference_edit import _rejection_reason

    return _rejection_reason(result)


def _deserialize_qwen_review(shard: str, payload: Mapping[str, Any]) -> Any:
    from r2v_data_v2.v3.reference_edit_boogu import BooguCompletionReview

    return BooguCompletionReview.model_validate(dict(payload))


def _draw_boogu_seed() -> int:
    from r2v_data_v2.v3.boogu_seed import new_boogu_seed

    return new_boogu_seed()
