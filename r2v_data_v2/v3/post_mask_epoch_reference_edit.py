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
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from r2v_data_v2.v3.post_mask_epoch_jobs import (
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
    _instruction,
    _instruction_rewrite_enabled,
    _operations,
    _reference_content_geometry,
    _resolve_artifact,
    _route,
    _source_gate_reason,
    _tokens_for_retained,
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


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise ReferenceEditEpochError(
                f"durable Reference Edit file drifted: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
                    raise ReferenceEditEpochError(
                        f"frozen Reference Edit plan {key} drifted for {shard!r}"
                    )
            storage = self._storage_for(shard)
            for clip_uid, entry in existing.get("clips", {}).items():
                actual = self._clip_digest(storage, storage.read_clip(clip_uid))
                if actual != entry.get("digest"):
                    raise ReferenceEditEpochError(
                        f"frozen Reference Edit input drifted for {clip_uid!r}"
                    )
            return existing

        storage = self._storage_for(shard)
        clips: dict[str, dict[str, Any]] = {}
        for clip_uid in self.eligible.get(shard, ()):
            clip = storage.read_clip(clip_uid)
            clips[clip_uid] = {
                "classification": self._clip_classification(clip),
                "digest": self._clip_digest(storage, clip),
                "chain_entity_ids": (
                    list(self._chain_entity_ids(clip))
                    if self._clip_classification(clip) == CLIP_FRESH_TARGET
                    else []
                ),
            }
        payload = {
            "schema": REFERENCE_EDIT_PLAN_SCHEMA,
            "canonical_shard": shard,
            "eligible_clip_uids": list(self.eligible.get(shard, ())),
            "policy": self._policy_identity(),
            "clips": clips,
        }
        _write_json_once(self._plan_path(shard), payload)
        return payload

    def _clip_digest(self, storage: RunStorage, clip: Any) -> str:
        """Digest of every pre-Reference-Edit input the stage depends on."""
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
            "pairing": (
                clip.pairing.model_dump(mode="json")
                if clip.pairing is not None
                else None
            ),
            "references": clip.references.model_dump(mode="json"),
            "sampled_frames": storage
            .read_frames(clip.clip_uid)
            .model_dump(mode="json"),
            "tracked_masks": storage.read_masks(clip.clip_uid).model_dump(mode="json"),
            "policy": self._policy_identity(),
        }
        return semantic_input_digest(projection)

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
                raise ReferenceEditEpochError(
                    f"frozen Reference Edit plan {key} drifted for {shard!r}"
                )
        return payload

    # -- durable seed plan ---------------------------------------------------

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
        geometry_sha256 = _sha256_bytes(
            _reference_content_geometry(storage, reference).model_dump_json().encode()
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
        width, height = self._resolved_size(source_sha256, storage, reference)
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

    def _resolved_size(
        self, _source_sha256: str, storage: RunStorage, reference: Any
    ) -> tuple[int, int]:
        from r2v_data_v2.v3.reference_edit_boogu import resolve_boogu_1k_size

        with Image.open(_resolve_artifact(storage, reference.image_path)) as opened:
            opened.load()
            rgba = opened.convert("RGBA")
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
            ):
                raise ReferenceEditEpochError(
                    f"durable Boogu seed plan drifted for {clip_uid}/{entity_id} "
                    f"attempt {attempt_index}"
                )
            return int(existing["seed"])
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
        """Resolve (once) and freeze the candidate2 alternate source."""
        path = self._alternate_path(shard, clip_uid, entity.entity_id)
        alternate = _alternate_completion_source(
            self.config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            canonical_reference=reference,
        )
        if alternate is None:
            return None
        identity = {
            "schema": REFERENCE_EDIT_ALTERNATE_SCHEMA,
            "canonical_source_sha256": _sha256_bytes(
                _resolve_artifact(storage, reference.image_path).read_bytes()
            ),
            "candidate_id": alternate.candidate.candidate_id,
            "source_frame_index": alternate.candidate.source_frame_index,
            "source_image_path": alternate.candidate.image_path,
            "alternate_source_sha256": _sha256_bytes(
                alternate.image_path.read_bytes()
            ),
        }
        _write_json_once(path, identity)
        return alternate

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
    ) -> Any:
        comparison: Path | None = None
        source_path: Path | None = None
        geometry_source: Path | None = None
        if attempt_index == 2:
            if alternate_image_path is None:
                raise ReferenceEditEpochError(
                    f"attempt 2 needs a frozen alternate for {clip_uid}/{entity.entity_id}"
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
                "canonical_reference" if attempt_index == 1 else "candidate_1"
            ),
            completion_source_frame_index=reference.source_frame_index,
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

    def _phase_for(self, job_id: str) -> str:
        class _Ref:
            def __init__(self, value: str) -> None:
                self._value = value

            def job_id(self) -> str:
                return self._value

        phase_id = self.ledger.phase_for(_Ref(job_id))
        if phase_id is None:
            raise ReferenceEditEpochError(
                f"pair-style dependency {job_id} has no committed phase"
            )
        return phase_id

    def _committed_result(self, job_id: str) -> Any:
        from r2v_data_v2.v3.post_mask_epoch_jobs import JobResult as _JobResult

        phase_id = self._phase_for(job_id)
        path = self.ledger.phase(phase_id).artifact_path(job_id, "result.json")
        if not path.is_file():
            raise ReferenceEditEpochError(
                f"committed job {job_id} has no durable result"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _JobResult.from_durable(payload)

    def _committed_artifact(self, job_id: str, name: str) -> bytes:
        phase_id = self._phase_for(job_id)
        path = self.ledger.phase(phase_id).artifact_path(job_id, name)
        if not path.is_file():
            raise ReferenceEditEpochError(
                f"committed job {job_id} has no artifact {name!r}"
            )
        return path.read_bytes()

    def _validated_candidate(
        self, generation_job_id: str, expected_sha256: str
    ) -> bytes:
        payload = self._committed_result(generation_job_id).payload
        candidate = self._committed_artifact(generation_job_id, "candidate.png")
        digest = _sha256_bytes(candidate)
        if payload.get("status") != "generated":
            raise ReferenceEditEpochError(
                f"generation {generation_job_id} did not commit a candidate"
            )
        if digest != expected_sha256 or digest != payload.get("candidate_sha256"):
            raise ReferenceEditEpochError(
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
                jobs.extend(self._advance_clip(shard, storage, clip_uid))
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

    def _rebuild_prepared(self, job: ModelJob, attempt_index: int) -> Any:
        """Re-derive the frozen attempt anchor, then prepare with resume policy."""
        shard, clip_uid, storage, entity, reference, _ = self._attempt_context(job)
        stored_anchor = dict(job.semantic_inputs["anchor"])
        alternate = None
        if attempt_index == 2:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
            if alternate is None:
                raise ReferenceEditEpochError(
                    f"attempt 2 planned without a frozen alternate: "
                    f"{clip_uid}/{entity.entity_id}"
                )
        current = self._entity_anchor(
            storage,
            clip_uid,
            entity,
            reference,
            attempt_index,
            alternate=alternate,
        )
        if semantic_input_digest(current) != semantic_input_digest(stored_anchor):
            raise ReferenceEditEpochError(
                f"generation anchor drifted for {clip_uid}/{entity.entity_id}"
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
        )

    def _run_generation(self, job: ModelJob, handle: Any) -> JobResult:
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
            item for item in clip.references.entities if item.entity_id == entity_id
        )
        anchor = dict(job.semantic_inputs["anchor"])
        seed = int(job.semantic_inputs["seed"])
        alternate = None
        if attempt_index == 2:
            alternate = self.durable_alternate(
                shard, storage, clip_uid, entity, reference
            )
        current = self._entity_anchor(
            storage, clip_uid, entity, reference, attempt_index, alternate=alternate
        )
        if semantic_input_digest({**current, "seed": seed}) != semantic_input_digest(
            {**anchor, "seed": seed}
        ):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Reference Edit generation anchor drifted",
            )
        prepared = self._prepare_attempt(
            shard,
            job.clip_uid,
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

    def _run_qwen(self, job: ModelJob, handle: Any) -> JobResult:
        generation_job_id = str(job.semantic_inputs["generation_job_id"])
        candidate_sha256 = str(job.semantic_inputs["candidate_sha256"])
        try:
            candidate_bytes = self._validated_candidate(
                generation_job_id, candidate_sha256
            )
            prepared = self._rebuild_prepared(job, int(dict(job.target)["attempt_index"]))
            generation = _reconstructed_generation(
                self, generation_job_id, candidate_bytes, candidate_sha256
            )
            review = run_boogu_qwen_review(prepared, generation, handle)
        except LEGACY_MODEL_EXCEPTIONS as exc:
            return JobResult(
                OUTCOME_COMPLETED,
                payload={"status": "qwen_failed", "error": str(exc)},
            )
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "qwen_review": review.model_dump(mode="json"),
            },
        )

    def _run_sam(self, job: ModelJob, handle: Any) -> JobResult:
        from r2v_data_v2.v3.reference_edit_boogu import Sam3BooguReferenceReviewer

        generation_job_id = str(job.semantic_inputs["generation_job_id"])
        candidate_sha256 = str(job.semantic_inputs["candidate_sha256"])
        try:
            candidate_bytes = self._validated_candidate(
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
            self._finalize_attempt(job)
            return ()
        raise ReferenceEditEpochError(f"unsupported job type {job.job_type!r}")

    def _finalize_generation(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        payload = dict(result.payload)
        shard, _storage, entity, _reference, attempt_index = (
            self._generation_context(job)
        )
        judge = self.config.qwen.reference_edit_judge
        assert judge is not None
        if payload.get("status") == "generated":
            qwen_job = self._review_job(
                shard,
                job.clip_uid,
                entity,
                attempt_index,
                job_type=REFERENCE_EDIT_QWEN_REVIEW_JOB,
                resource=RESOURCE_QWEN,
                model_identity=f"qwen:{judge.model}",
                generation_job_id=job.job_id(),
                candidate_sha256=str(payload["candidate_sha256"]),
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
                        candidate_sha256=str(payload["candidate_sha256"]),
                    )
                )
            return unlocked
        # generation_failed: legacy semantic rejection; no reviews are paid.
        return self._complete_attempt_after_generation_failure(job, payload)

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
                # The SAM job was unlocked with the Qwen job at generation time.
                return ()
            attempt_index = int(dict(job.target)["attempt_index"])
            sam_job = self._review_job(
                job.canonical_shard,
                job.clip_uid,
                self._entity_of(job),
                attempt_index,
                job_type=REFERENCE_EDIT_SAM_REVIEW_JOB,
                resource=RESOURCE_SAM,
                model_identity="sam3",
                generation_job_id=str(job.semantic_inputs["generation_job_id"]),
                candidate_sha256=str(job.semantic_inputs["candidate_sha256"]),
            )
            return [sam_job]
        return self._complete_attempt_after_review_failure(job, payload)

    # -- attempt completion -----------------------------------------------------

    def _attempt_finalize(
        self,
        job: ModelJob,
        *,
        qwen_payload: Mapping[str, Any] | None,
        sam_payload: Mapping[str, Any] | None,
    ) -> BooguReferenceEditResult:
        generation_job_id = str(job.semantic_inputs["generation_job_id"])
        candidate_sha256 = str(job.semantic_inputs["candidate_sha256"])
        candidate_bytes = self._validated_candidate(
            generation_job_id, candidate_sha256
        )
        attempt_index = int(dict(job.target)["attempt_index"])
        prepared = self._rebuild_prepared(job, attempt_index)
        generation = _reconstructed_generation(
            self, generation_job_id, candidate_bytes, candidate_sha256
        )
        self._materialize_candidate(prepared.candidate_path, candidate_bytes)
        qwen_review = None
        qwen_override: str | None = None
        if qwen_payload is not None and qwen_payload.get("status") == "review":
            qwen_review = _deserialize_qwen_review(
                job.canonical_shard, qwen_payload["qwen_review"]
            )
        elif qwen_payload is not None:
            qwen_override = f"boogu_reference_edit_failed: {qwen_payload.get('error')}"
        sam_review = None
        sam_override: str | None = None
        if sam_payload is not None and sam_payload.get("status") == "review":
            sam_review = BooguSamReview.model_validate(sam_payload["sam_review"])
        elif sam_payload is not None:
            sam_override = f"boogu_reference_edit_failed: {sam_payload.get('error')}"
        override = qwen_override or sam_override
        return finalize_boogu_reference_edit_attempt(
            prepared,
            generation,
            qwen_review=qwen_review,
            qwen_review_skipped_reason=None,
            sam_review=sam_review,
            rejection_reason_override=override,
        )

    def _materialize_candidate(self, destination: Path, payload: bytes) -> None:
        if destination.is_file():
            if _sha256_bytes(destination.read_bytes()) != _sha256_bytes(payload):
                raise ReferenceEditEpochError(
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

    def _attempt_failed_result(self, job: ModelJob) -> BooguReferenceEditResult:
        """Re-run the attempt finalizer for a generation_failed attempt."""
        payload = dict(self._committed_result(job.job_id()).payload)
        shard, _storage, _entity, _reference, _attempt_index = (
            self._generation_context(job)
        )
        del shard
        prepared = self._rebuild_prepared(job, int(dict(job.target)["attempt_index"]))
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

    def _complete_attempt_after_generation_failure(
        self, job: ModelJob, payload: Mapping[str, Any]
    ) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        _shard, storage, entity, reference, attempt_index = (
            self._generation_context(job)
        )
        attempt_result = self._attempt_failed_result(job)
        return self._after_attempt(
            job,
            shard,
            storage,
            entity,
            reference,
            attempt_index,
            attempt_result,
            failure_reason=_rejection_reason_of(attempt_result),
        )

    def _complete_attempt_after_review_failure(
        self, job: ModelJob, payload: Mapping[str, Any]
    ) -> Sequence[ModelJob]:
        shard = job.canonical_shard
        _shard, storage, entity, reference, attempt_index = (
            self._generation_context(job)
        )
        generation_job_id = str(job.semantic_inputs["generation_job_id"])
        generation_job = self._generation_job_by_id(generation_job_id)
        attempt_result = self._attempt_finalize(
            generation_job,
            qwen_payload=payload if job.job_type == REFERENCE_EDIT_QWEN_REVIEW_JOB else None,
            sam_payload=payload if job.job_type == REFERENCE_EDIT_SAM_REVIEW_JOB else None,
        )
        return self._after_attempt(
            generation_job,
            shard,
            storage,
            entity,
            reference,
            attempt_index,
            attempt_result,
            failure_reason=_rejection_reason_of(attempt_result),
        )

    def _generation_job_by_id(self, job_id: str) -> ModelJob:
        for record in self._planned_records():
            if record.get("job_id") == job_id:
                return self._job_from_plan_record(record)
        raise ReferenceEditEpochError(
            f"generation job {job_id} is not in any durable plan"
        )

    def _planned_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for phase_id in self.ledger.phase_ids():
            phase = self.ledger.phase(phase_id)
            if not phase.plan_path.is_file():
                continue
            try:
                records.extend(phase.read_plan())
            except Exception as exc:
                raise ReferenceEditEpochError(
                    f"invalid Reference Edit phase plan in {phase_id}"
                ) from exc
        return records

    def _job_from_plan_record(self, record: Mapping[str, Any]) -> ModelJob:
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
            schema_version=str(record.get("schema_version", "post_mask_epoch_job/1")),
        )
        if job.job_id() != str(record.get("job_id", "")):
            raise ReferenceEditEpochError(
                f"Reference Edit plan record does not rebuild: {record.get('job_id')}"
            )
        return job

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
                        **(
                            {"completion_candidate1_accepted": 1}
                            if attempt_index == 1
                            else {
                                "completion_candidate2_accepted": 1,
                                "completion_candidate2_attempts": 1,
                            }
                        ),
                        "completion_attempts": 1,
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
        if attempt_index == 2:
            # The candidate2 attempt was planned and paid; count it.
            self._count_candidate2_attempt(shard, clip_uid, entity.entity_id)
        _write_source_selection_metadata(
            storage,
            clip_uid=clip_uid,
            reference=reference,
            geometry=_reference_content_geometry(storage, reference),
            source_gate_reason=None,
            reason=(
                "repairable_completion_rejected:"
                f"{failure_reason or 'completion_unavailable'}"
            ),
            operation_metadata_path=attempt_result.metadata_path,
        )
        delta = {
            "entities_eligible": 1,
            "entities_fallback": 1,
            "completion_fallback_to_alpha": 1,
            "completion_attempts": 1,
            **routing_delta,
        }
        if attempt_index == 2:
            delta["completion_candidate2_attempts"] = 1
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

    def _count_candidate2_attempt(
        self, shard: str, clip_uid: str, entity_id: str
    ) -> None:
        outcome = self._entity_outcome(shard, clip_uid, entity_id)
        if outcome is not None and outcome.get("candidate2_counted"):
            return
        # Counted inside the fallback delta below; marker kept for idempotence.
        marker = self._semantic(
            "accounted", "candidate2", shard, clip_uid, f"{entity_id}.json"
        )
        _write_json_once(marker, {"candidate2_counted": True, "entity_id": entity_id})

    # -- clip publication --------------------------------------------------------

    def _publish_clip_if_terminal(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> None:
        plan = self._existing_plan_for_reconcile(shard)
        chain_ids = list(plan["clips"][clip_uid]["chain_entity_ids"])
        outcomes = {
            entity_id: self._entity_outcome(shard, clip_uid, entity_id)
            for entity_id in chain_ids
        }
        if any(outcome is None for outcome in outcomes.values()):
            return
        if self._clip_outcome_path(shard, clip_uid).is_file():
            return
        clip = self._clip(storage, clip_uid)
        entities = {entity.entity_id: entity for entity in clip.annotation.entities}
        retained = set(clip.pairing.retained_entity_ids)
        final_references = []
        edit_states: list[ReferenceEditEntityState] = []
        delta: dict[str, int] = {}
        for reference in clip.references.entities:
            entity_id = reference.entity_id
            if entity_id not in outcomes:
                final_references.append(reference)
                continue
            outcome = dict(outcomes[entity_id])
            for field, value in outcome.get("delta", {}).items():
                delta[field] = delta.get(field, 0) + int(value)
            source_geometry, gate_reason, route = self._entity_route_context(
                storage, clip, reference
            )
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
                    )
                )
                continue
            if outcome["outcome"] == "accepted":
                attempt_index = int(outcome.get("attempt_index", 1))
                edit_dir = (
                    storage.reference_edit_dir(clip_uid) / entity_id
                )
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
                        default_variant=(
                            "accepted_base"
                        ),
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
            # fallback (completion rejected)
            final_references.append(reference)
            edit_states.append(
                ReferenceEditEntityState(
                    entity_id=entity_id,
                    route=route,
                    status="fallback",
                    source_reference=reference,
                    source_image_path=reference.image_path,
                    output_image_path=reference.image_path,
                    default_variant="alpha",
                    default_image_path=reference.image_path,
                    default_reason=(
                        "completion_rejected_fallback_to_source_alpha"
                    ),
                    accepted_base_image_path=reference.image_path,
                    operation="complete_entity",
                    metadata_path=storage.relative_artifact_path(
                        storage.reference_edit_dir(clip_uid)
                        / entity_id
                        / "final_metadata.json"
                    ),
                    operations=["complete_entity"],
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
                background_token=clip.pairing.background_token,
            )
        storage.write_reference_edit_result(
            clip_uid,
            ReferencesState(
                entities=final_references, background=clip.references.background
            ),
            pairing,
            ReferenceEditState(status="ready", entities=edit_states),
        )
        outcome_delta = {
            "processed": 1,
            "failed": int(any(o["outcome"] == "failed" for o in outcomes.values() if o)),
        }
        outcome_delta.update(delta)
        _write_json_once(
            self._clip_outcome_path(shard, clip_uid),
            {
                "schema": REFERENCE_EDIT_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "ready",
                "delta": outcome_delta,
            },
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
            for field, value in outcome.get("delta", {}).items():
                counts[field] = counts.get(field, 0) + int(value)
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

    payload = dict(runner._committed_result(generation_job_id).payload)
    return SimpleNamespace(
        png_bytes=candidate_bytes,
        output_sha256=candidate_sha256,
        seed=int(payload.get("seed", 0)),
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
