"""Subject Attributes resource epoch (8a): durable owner + discovery foundation.

Legacy authority is ``subject_attributes.process_subject_attribute_clip`` /
``_process_owner``. This module resource-epochs only the first half of that
pipeline: the frozen clip/owner plan, the zero-model reuse of an already valid
legacy owner artifact, and the Qwen subject attribute discovery call with its two
terminal outcomes (discovery failure, confirmed non-human owner).

A confirmed human owner is deliberately left unresolved: it keeps only its
durable discovery receipt so 8b can continue from exactly there. SAM candidate
selection, raw/candidate2/bbox review, Boogu completion, completion post-checks,
composition and export are all out of scope.

Durable roots under ``<ledger>/semantic/subject_attributes``::

    plans/<shard>/<clip>.json                        frozen clip plan
    owners/<shard>/<clip>/<owner>/plan.json          frozen owner plan
    owners/<shard>/<clip>/<owner>/outcome.json       owner outcome marker
    outcomes/<shard>/<clip>.json                     clip outcome marker

The legacy owner artifacts themselves keep their legacy path and schema
(``<storage.root>/subject_attributes/owners/<clip>/<owner>.json`` is an
``OwnerEnrichmentArtifact``); this epoch never redefines them.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.boogu_seed import new_boogu_seed
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pair import (
    EntityReferenceCandidate,
    _decode_candidate_mask,
    build_entity_reference_candidates,
    build_reference_crop,
    mask_component_diagnostics,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    canonical_json,
)
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    RESOURCE_SAM as RESOURCE_SAM_NAME,
)
from r2v_data_v2.v3.post_mask_epoch_resources import resolve_cpu_workers
from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_bytes, atomic_write_json
from r2v_data_v2.v3.storage import RunStorage, evaluate_export_state
from r2v_data_v2.v3.subject_attributes import (
    ATTRIBUTE_BBOX_REVIEW_SYSTEM_PROMPT,
    ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT,
    ATTRIBUTE_FACE_BBOX_REVIEW_SYSTEM_PROMPT,
    DISCOVERY_SYSTEM_PROMPT,
    MAX_ATTRIBUTE_SOURCE_CANDIDATES,
    MAX_ATTRIBUTES_PER_OWNER,
    MIN_ATTRIBUTE_AREA_PIXELS,
    MIN_ATTRIBUTE_LONG_SIDE_PIXELS,
    QWEN_INPUT_MAX_LONG_SIDE_PIXELS,
    REVIEW_SYSTEM_PROMPT,
    ClipEnrichmentResult,
    DiscoveredSubjectAttribute,
    EnrichmentTotals,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    QwenSubjectAttributeClient,
    QwenSubjectAttributeCompletionJudge,
    SubjectAttributeBboxReview,
    SubjectAttributeCompletionReview,
    SubjectAttributeDiscovery,
    SubjectAttributeRecord,
    SubjectAttributeReviewBatch,
    _attribute_bbox_crop,
    _build_enriched_sample,
    _clean_completion_mask,
    _clip_sample_path,
    _collect_attribute_candidates,
    _completion_mask_postcheck_rejection,
    _duplicate_attribute_mask_conflicts,
    _evaluate_completion_quality,
    _load_cached_owner_artifact,
    _owner_artifact_path,
    _process_owner,
    _rejected_record,
    _resize_qwen_input_image,
    _save_completion_input,
    _source_image,
    attribute_completion_prompt,
    build_candidate_context_image,
    evaluate_owner_eligibility,
    prefer_attribute_candidate_frames,
)

SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA = (
    "post_mask_epoch_subject_attributes_clip_plan/3"
)
SUBJECT_ATTRIBUTE_OWNER_PLAN_SCHEMA = (
    "post_mask_epoch_subject_attributes_owner_plan/3"
)
SUBJECT_ATTRIBUTE_OWNER_OUTCOME_SCHEMA = (
    "post_mask_epoch_subject_attributes_owner_outcome/2"
)
SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA = (
    "post_mask_epoch_subject_attributes_clip_outcome/1"
)
SUBJECT_ATTRIBUTE_POLICY_VERSION = "subject_attributes_epoch_policy/3"
SUBJECT_ATTRIBUTE_DISCOVERY_POLICY_VERSION = "subject_attribute_discovery/1"

SUBJECT_ATTRIBUTE_ATTRIBUTE_PLAN_SCHEMA = (
    "post_mask_epoch_subject_attribute_plan/1"
)
SUBJECT_ATTRIBUTE_SELECTION_SCHEMA = (
    "post_mask_epoch_subject_attribute_selection/1"
)
SUBJECT_ATTRIBUTE_OWNER_RAW_STATE_SCHEMA = (
    "post_mask_epoch_subject_attribute_raw_state/1"
)
SUBJECT_ATTRIBUTE_SELECTION_POLICY_VERSION = (
    "subject_attribute_candidate_selection/1"
)
SUBJECT_ATTRIBUTE_RAW_REVIEW_POLICY_VERSION = "subject_attribute_raw_review/1"

# -- 8c: completion and bbox -------------------------------------------------
SUBJECT_ATTRIBUTE_COMPLETION_SEED_SCHEMA = (
    "post_mask_epoch_subject_attribute_completion_seed/1"
)
SUBJECT_ATTRIBUTE_COMPLETION_OUTCOME_SCHEMA = (
    "post_mask_epoch_subject_attribute_completion_outcome/1"
)
SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_POLICY_VERSION = (
    "subject_attribute_completion_generate/2"
)
SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_POLICY_VERSION = (
    "subject_attribute_completion_review/2"
)
SUBJECT_ATTRIBUTE_BBOX_REVIEW_POLICY_VERSION = "subject_attribute_bbox_review/2"
SUBJECT_ATTRIBUTE_COMPLETION_POSTCHECK_POLICY_VERSION = (
    "subject_attribute_completion_postcheck/1"
)

#: Terminal completion statuses. ``review_required`` is the non-terminal stage
#: between a passing CPU postcheck and the committed review receipt, so it is
#: never persisted: the durable marker exists only once the chain is terminal.
COMPLETION_GENERATION_FAILED = "generation_failed"
COMPLETION_SAM_FAILED = "sam_failed"
COMPLETION_POSTCHECK_REJECTED = "postcheck_rejected"
COMPLETION_REVIEW_REQUIRED = "review_required"
COMPLETION_REVIEW_REJECTED = "review_rejected"
COMPLETION_ACCEPTED = "accepted"
COMPLETION_TERMINAL_STATUSES = (
    COMPLETION_GENERATION_FAILED,
    COMPLETION_SAM_FAILED,
    COMPLETION_POSTCHECK_REJECTED,
    COMPLETION_REVIEW_REJECTED,
    COMPLETION_ACCEPTED,
)

#: The only model job this epoch owns. Everything else in Subject Attributes is
#: still legacy-only.
SUBJECT_ATTRIBUTE_DISCOVERY_JOB = "subject_attribute_discovery"
#: One real legacy ``segment_frame`` call is one job, so a single job never pays
#: more than one SAM call.
SUBJECT_ATTRIBUTE_SAM_PROBE_JOB = "subject_attribute_sam_probe"
#: One owner-batched rank-0 raw review call.
SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB = "subject_attribute_raw_review"

#: 8c model calls. One completion chain is generate -> SAM -> review; the bbox
#: review is the legacy last-resort route and is one job per attribute.
SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB = "subject_attribute_completion_generate"
SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB = "subject_attribute_completion_sam"
SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_JOB = "subject_attribute_completion_review"
SUBJECT_ATTRIBUTE_BBOX_REVIEW_JOB = "subject_attribute_bbox_review"
RESOURCE_SAM = RESOURCE_SAM_NAME

CLIP_NO_WORK = "no_work"
CLIP_OWNERS = "owners"

OWNER_SOURCE_PREEXISTING = "preexisting"
OWNER_SOURCE_DISCOVERY = "discovery"
#: The owner artifact written by the epoch's own semantic graph (8c).
OWNER_SOURCE_PROCESSED = "processed"

DISCOVERY_FAILED_PREFIX = "discovery_failed:"

SELECTION_CANDIDATES = "candidates"
SELECTION_REJECTED = "rejected"

#: The only routes 8b may emit. Everything except a pure selection rejection,
#: a duplicate-mask conflict and a terminal reject still needs 8c to finish.
ROUTE_SELECTION_REJECTED = "selection_rejected"
ROUTE_DUPLICATE_CONFLICT = "duplicate_conflict"
ROUTE_RAW_ACCEPT = "raw_accept"
ROUTE_FACE_ACCEPT = "face_accept"
ROUTE_CANDIDATE2_READY = "candidate2_ready"
ROUTE_COMPLETION_REQUIRED = "completion_required"
ROUTE_BBOX_PENDING = "bbox_pending"
ROUTE_TERMINAL_REJECT = "terminal_reject"
RANK0_ROUTES = (
    ROUTE_SELECTION_REJECTED,
    ROUTE_DUPLICATE_CONFLICT,
    ROUTE_RAW_ACCEPT,
    ROUTE_FACE_ACCEPT,
    ROUTE_CANDIDATE2_READY,
    ROUTE_COMPLETION_REQUIRED,
    ROUTE_BBOX_PENDING,
    ROUTE_TERMINAL_REJECT,
)

#: The legacy output root this epoch reuses for owner artifacts and samples.
SUBJECT_ATTRIBUTE_OUTPUT_DIRNAME = "subject_attributes"


class SubjectAttributeEpochError(RuntimeError):
    """Raised when the Subject Attributes stage is not terminal."""


class SubjectAttributeDurableError(SubjectAttributeEpochError):
    """Raised when durable Subject Attributes state cannot be trusted."""


@dataclass(frozen=True)
class ResolvedDiscoveryClient:
    """A discovery client handle plus whether *this* call created it."""

    client: Any
    owned: bool


class _OwnerReplay:
    """The durable state one owner's legacy re-run replays from.

    Everything here is derived from frozen plans and committed receipts: the
    replayed legacy algorithm never sees a live model, and the first call it
    cannot answer from a receipt aborts the re-run through ``_PendingModelCall``.
    """

    def __init__(self, *, epoch, shard, storage, clip, owner, owner_plan,
                 reference, discovery, discovery_job_id, candidates, masks,
                 states) -> None:
        self.epoch = epoch
        self.shard = shard
        self.storage = storage
        self.clip = clip
        self.clip_uid = clip.clip_uid
        self.owner = owner
        self.owner_plan = owner_plan
        self.reference = reference
        self.discovery = discovery
        self.discovery_job_id = discovery_job_id
        #: The frozen owner-plan order, which is what legacy ``_process_owner``
        #: is handed as ``owner_candidates``. It is *not* the probe order: legacy
        #: re-sorts it with ``prefer_attribute_candidate_frames``.
        self.candidates = list(candidates)
        self.masks = masks
        self.states = list(states)
        #: The legacy probe order, frozen once. ``_ordered_owner_candidates``
        #: proves the frame slots are unique, and every SAM probe job, receipt and
        #: mask artifact is keyed on this order's index, so the replay must index
        #: chains and receipts with it instead of with the frozen owner-plan order.
        self._legacy_probe_candidates = self._freeze_legacy_probe_candidates()
        self._legacy_probe_slots = tuple(
            int(candidate.frame_slot) for candidate in self._legacy_probe_candidates
        )
        self._states_by_id = {state.attribute_id: state for state in self.states}
        self._chains: dict[str, list[ModelJob]] = {}
        self._receipts: dict[str, dict[int, Mapping[str, Any]]] = {}
        #: ``-1`` until legacy's first probe opens the attribute run. Legacy walks
        #: the discovery order once and never goes back, so the replay only ever
        #: stays on the current attribute or advances to the next one.
        self.attribute_cursor = -1
        #: How many legacy probes of one attribute were already served, which is
        #: also the index of its next expected probe in the legacy probe order.
        self._probe_cursors: dict[str, int] = {}
        #: 8c chain currently being replayed: legacy calls generation, then the
        #: generated-frame segmentation, then the review.
        self.completion: dict[str, Any] | None = None
        self.bbox_used: set[str] = set()
        # Model timing is accumulated from the durable receipts the replayed
        # legacy algorithm actually walked, never from wall clock, so a restart
        # never re-measures CPU replay. Discovery is always the first Qwen call.
        self._qwen_seconds = _receipt_seconds(
            self.epoch._discovery_payload(shard, self.clip_uid, owner_plan),
            "qwen_model_call_time_seconds",
            label=f"discovery of {self.clip_uid}/{self.owner_entity_id}",
        )
        self._sam3_seconds = 0.0
        self._completion_seconds = 0.0

    # -- durable model timing -------------------------------------------------

    def add_qwen_seconds(self, payload: Mapping[str, Any], *, count: bool) -> None:
        """Accumulate one Qwen call the legacy round would have measured.

        Legacy times the whole raw review round around its ``try``, so a failed
        round still contributes; the completion judge only measures a call that
        returned, so a failed completion review contributes nothing.
        """
        if not count:
            return
        self._qwen_seconds += _receipt_seconds(
            payload,
            "qwen_model_call_time_seconds",
            label=f"Qwen call of {self.clip_uid}/{self.owner_entity_id}",
        )

    def add_sam_seconds(self, payload: Mapping[str, Any]) -> None:
        self._sam3_seconds += _receipt_seconds(
            payload,
            "model_call_time_seconds",
            label=f"SAM call of {self.clip_uid}/{self.owner_entity_id}",
        )

    def add_completion_seconds(self, payload: Mapping[str, Any]) -> None:
        """Accumulate one Boogu generation the legacy accounting measured.

        Legacy adds the reported duration as soon as the backend returns, before
        it opens the generated file, so a later PIL or read failure keeps the
        duration. A backend that raised never reported one and adds nothing.
        """
        self._completion_seconds += _receipt_seconds(
            payload,
            "model_call_time_seconds",
            label=f"completion of {self.clip_uid}/{self.owner_entity_id}",
        )

    # -- lookups -------------------------------------------------------------

    @property
    def owner_entity_id(self) -> str:
        return str(self.owner_plan["owner_entity_id"])

    def state_for(self, attribute_id: str) -> _AttributeReplay:
        state = self._states_by_id.get(str(attribute_id))
        if state is None:
            raise SubjectAttributeDurableError(
                f"attribute {attribute_id!r} is not a frozen attribute of "
                f"{self.clip_uid}/{self.owner_entity_id}"
            )
        return state

    def _freeze_legacy_probe_candidates(self) -> tuple[Any, ...]:
        """Derive the legacy probe order exactly the way legacy would.

        ``_process_owner`` is handed ``self.candidates`` (the frozen owner-plan
        order) and re-sorts it with ``prefer_attribute_candidate_frames``, whose
        tie-break rank is the input order, so feeding the frozen order in
        reproduces legacy's probe order exactly. ``_ordered_owner_candidates``
        proves the frame slots are unique, so one slot names one probe index.
        """
        ordered = tuple(
            self.epoch._ordered_owner_candidates(
                self.shard,
                self.storage,
                self.clip_uid,
                self.owner_plan,
                self.reference,
            )
        )
        if not ordered:
            raise SubjectAttributeDurableError(
                f"owner {self.clip_uid}/{self.owner_entity_id} has no frozen "
                "attribute probe candidate"
            )
        return ordered

    def ordered_candidates(self) -> tuple[Any, ...]:
        """The frozen legacy probe order, derived once per owner replay.

        Every SAM probe job, receipt and mask artifact is keyed on this order's
        index, so this - not the frozen owner-plan order - is the replay's job
        index authority.
        """
        return self._legacy_probe_candidates

    def chain_for(self, state: _AttributeReplay) -> list[ModelJob]:
        chain = self._chains.get(state.attribute_id)
        if chain is None:
            chain = self.epoch._sam_probe_chain(
                self.shard,
                self.clip_uid,
                self.owner_plan,
                state.attribute_plan,
                self.ordered_candidates(),
            )
            self._chains[state.attribute_id] = chain
        return chain

    def receipts_for(self, state: _AttributeReplay) -> dict[int, Mapping[str, Any]]:
        receipts = self._receipts.get(state.attribute_id)
        if receipts is None:
            receipts = {}
            for index, job in enumerate(self.chain_for(state)):
                payload = self.epoch._committed_payload_or_none(job)
                if payload is not None:
                    receipts[index] = payload
            self._receipts[state.attribute_id] = receipts
        return receipts

    def resolve_probe(
        self, grounding_prompt: str, frame_slot: int
    ) -> tuple[_AttributeReplay, int]:
        """Which attribute legacy is probing, and at which legacy probe index.

        Legacy's ``_collect_attribute_candidates`` walks the frozen attributes in
        discovery order and never looks back, and it may leave an attribute early
        once it has collected ``MAX_ATTRIBUTE_SOURCE_CANDIDATES`` candidates. So
        the replay advances one way only: a call is either the current attribute's
        next expected probe, or it opens the *next* attribute at the *first*
        legacy probe. That is what keeps two attributes sharing one grounding
        prompt apart - a probe count cannot, because an attribute that stopped
        early never probes the tail of the probe order, so the next attribute's
        first probe would be mistaken for a continuation of the previous one.

        The returned index is the position in the legacy probe order, which is
        the index the SAM probe job, its receipt and its mask artifacts are all
        keyed on. It is deliberately not the position in the frozen owner-plan
        order: ``prefer_attribute_candidate_frames`` moves the owner reference's
        own frame to the end, so the two orders disagree whenever the reference
        frame is also a candidate.
        """
        slot = int(frame_slot)
        count = len(self.states)
        if count == 0:
            raise SubjectAttributeDurableError(
                f"owner {self.clip_uid}/{self.owner_entity_id} has no frozen "
                "attribute to probe"
            )
        if self.attribute_cursor < 0:
            # Legacy's very first probe of this owner opens attribute zero, at
            # the first slot of the legacy probe order.
            state = self.states[0]
            expected_slot = self._legacy_probe_slots[0]
            if self._prompt_of(state) != grounding_prompt or expected_slot != slot:
                raise SubjectAttributeDurableError(
                    "legacy SAM probe sequence drifted at the first probe of "
                    f"{self.clip_uid}/{self.owner_entity_id}: expected "
                    f"{self._prompt_of(state)!r} at slot {expected_slot}, got "
                    f"{grounding_prompt!r} at slot {slot}"
                )
            self.attribute_cursor = 0
            self._probe_cursors[state.attribute_id] = 1
            return state, 0
        current = self.states[self.attribute_cursor]
        probe_index = self._probe_cursors.get(current.attribute_id, 0)
        if (
            probe_index < len(self._legacy_probe_slots)
            and self._prompt_of(current) == grounding_prompt
            and self._legacy_probe_slots[probe_index] == slot
        ):
            self._probe_cursors[current.attribute_id] = probe_index + 1
            return current, probe_index
        # Legacy stopped probing the current attribute: either it exhausted the
        # probe order or it collected its candidate quota early. Only the next
        # discovery-order attribute may open here, and only at the first probe.
        next_index = self.attribute_cursor + 1
        if next_index >= count:
            raise SubjectAttributeDurableError(
                "legacy SAM probe sequence drifted past the last frozen attribute "
                f"of {self.clip_uid}/{self.owner_entity_id}: {grounding_prompt!r} "
                f"at slot {slot}"
            )
        next_state = self.states[next_index]
        expected_slot = self._legacy_probe_slots[0]
        if self._prompt_of(next_state) != grounding_prompt or expected_slot != slot:
            raise SubjectAttributeDurableError(
                "legacy SAM probe sequence drifted between frozen attributes of "
                f"{self.clip_uid}/{self.owner_entity_id}: expected "
                f"{self._prompt_of(next_state)!r} at slot {expected_slot}, got "
                f"{grounding_prompt!r} at slot {slot}"
            )
        self.attribute_cursor = next_index
        self._probe_cursors[next_state.attribute_id] = 1
        return next_state, 0

    @staticmethod
    def _prompt_of(state: _AttributeReplay) -> str:
        return str(state.attribute_plan["discovered"]["grounding_prompt"])

    # -- completion chain ----------------------------------------------------

    def begin_completion(self, state, rank, candidate, generate_job) -> None:
        self.completion = {
            "state": state,
            "rank": int(rank),
            "candidate": candidate,
            "generate_job": generate_job,
        }

    def _completion_context(self) -> dict[str, Any]:
        if self.completion is None:
            raise SubjectAttributeDurableError(
                "a completion stage was reached outside a completion chain"
            )
        return self.completion

    def completion_target(self, source_path: Path) -> tuple[Any, int, Any]:
        """Recover ``(attribute state, candidate rank, candidate)`` from a path."""
        root = self.epoch._output_root(self.storage) / "completion_candidates"
        try:
            relative = Path(source_path).resolve(strict=False).relative_to(root)
        except ValueError as exc:
            raise SubjectAttributeDurableError(
                f"completion source {source_path} is outside the legacy root"
            ) from exc
        parts = relative.parts
        if len(parts) != 6 or parts[4] != "raw" or parts[5] != "source.png":
            raise SubjectAttributeDurableError(
                f"completion source {source_path} is not a legacy candidate path"
            )
        clip_uid, owner_entity_id, attribute_id, owner_candidate_id = parts[:4]
        if clip_uid != self.clip_uid or owner_entity_id != self.owner_entity_id:
            raise SubjectAttributeDurableError(
                f"completion source {source_path} belongs to another owner"
            )
        state = self.state_for(attribute_id)
        ranks = [
            index
            for index, candidate in enumerate(state.options)
            if str(candidate.owner_candidate.candidate_id) == owner_candidate_id
        ]
        if len(ranks) != 1:
            raise SubjectAttributeDurableError(
                f"completion candidate {owner_candidate_id} is not a frozen "
                f"option of {attribute_id}"
            )
        return state, ranks[0], state.options[ranks[0]]

    def completion_cpu(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """The CPU postcheck state of one completion context."""
        epoch = self.epoch
        state = context["state"]
        candidate = context["candidate"]
        rank = int(context["rank"])
        generate_job = context["generate_job"]
        payload = epoch._committed_payload_or_none(generate_job)
        if payload is None:
            raise _PendingModelCall(generate_job)
        if str(payload.get("status")) == "completion_failed":
            raise _sam_failure_exception(payload)
        generated = epoch._generated_png_bytes(self.storage, payload)
        sam_job = epoch._expected_completion_sam_job(
            self.shard,
            self.clip_uid,
            self.owner_plan,
            state.attribute_plan,
            candidate,
            rank,
            generate_job,
            str(payload.get("generated", {}).get("sha256")),
            str(payload.get("generated_png_path", "")),
        )
        sam_payload = epoch._committed_payload_or_none(sam_job)
        if sam_payload is None:
            raise _PendingModelCall(sam_job)
        if str(sam_payload.get("status")) == "sam_failed":
            raise _sam_failure_exception(sam_payload)
        masks = epoch._load_sam_masks(sam_payload)
        return {
            "generated": generated,
            "masks": masks,
            "sam_job": sam_job,
            "sam_payload": sam_payload,
            "cpu": epoch._completion_cpu_result(generated, masks),
        }

    def segment_generated_frame(
        self, *, frame_path: Path, grounding_prompt: str
    ) -> list[Any]:
        epoch = self.epoch
        context = self._completion_context()
        state = context["state"]
        candidate = context["candidate"]
        if grounding_prompt != str(
            state.attribute_plan["discovered"]["grounding_prompt"]
        ):
            raise SubjectAttributeDurableError(
                "completion SAM prompt drifted from its frozen attribute"
            )
        expected = epoch._completion_output_path(
            self.storage,
            self.clip_uid,
            self.owner_entity_id,
            state.attribute_id,
            str(candidate.owner_candidate.candidate_id),
        )
        if Path(frame_path).resolve(strict=False) != expected.resolve(strict=False):
            raise SubjectAttributeDurableError(
                "completion SAM input drifted from its generated candidate"
            )
        evaluated = self.completion_cpu(context)
        self.add_sam_seconds(evaluated["sam_payload"])
        return list(evaluated["masks"])

    def completion_review(self, *, generated_candidate: Any) -> Any:
        epoch = self.epoch
        context = self._completion_context()
        state = context["state"]
        candidate = context["candidate"]
        rank = int(context["rank"])
        evaluated = self.completion_cpu(context)
        cpu = evaluated["cpu"]
        if cpu["status"] != COMPLETION_REVIEW_REQUIRED:
            raise SubjectAttributeDurableError(
                "completion review was reached for a rejected postcheck"
            )
        if _image_png_sha256(
            cpu["completed_rgba"].convert("RGBA")
        ) != _image_png_sha256(generated_candidate.convert("RGBA")):
            raise SubjectAttributeDurableError(
                "completion review candidate drifted from its postcheck"
            )
        review_job = epoch._expected_completion_review_job(
            self.shard,
            self.clip_uid,
            self.owner_plan,
            state.attribute_plan,
            candidate,
            rank,
            evaluated["sam_job"],
            _image_png_sha256(candidate.crop.convert("RGBA")),
            _image_png_sha256(
                _attribute_bbox_crop(
                    candidate.source_image,
                    candidate.attribute_mask,
                    crop_padding_ratio=float(epoch.config.pair.crop_padding_ratio),
                )
            ),
            _image_png_sha256(cpu["completed_rgba"].convert("RGBA")),
        )
        payload = epoch._committed_payload_or_none(review_job)
        if payload is None:
            raise _PendingModelCall(review_job)
        # Legacy only adds the review duration once ``review`` returned; a
        # failed completion review contributes nothing to the Qwen total.
        self.add_qwen_seconds(
            payload, count=str(payload.get("status")) == "review"
        )
        if str(payload.get("status")) == "review_failed":
            raise _sam_failure_exception(payload)
        try:
            return SubjectAttributeCompletionReview.model_validate(
                payload.get("review")
            )
        except Exception as exc:
            raise SubjectAttributeDurableError(
                "committed completion review payload is invalid"
            ) from exc

    # -- frozen accounting ---------------------------------------------------

    def frozen_seed_for(self, attribute_id: str) -> int | None:
        """The seed of the last completion this attribute actually attempted."""
        state = self.state_for(attribute_id)
        found: int | None = None
        for rank in range(len(state.options)):
            path = self.epoch._completion_seed_path(
                self.shard,
                self.clip_uid,
                self.owner_entity_id,
                attribute_id,
                rank,
            )
            payload = _read_json(path)
            if payload is not None:
                found = int(payload["seed"])
        return found

    @property
    def qwen_seconds(self) -> float:
        """Qwen model time from the durable receipts the replayed legacy walked."""
        return self._qwen_seconds

    @property
    def sam3_seconds(self) -> float:
        """SAM model time from the durable receipts the replayed legacy walked."""
        return self._sam3_seconds

    @property
    def completion_seconds(self) -> float:
        """Boogu model time from the durable receipts legacy measured."""
        return self._completion_seconds


class _OwnerDiscoveryClient:
    """Legacy discovery surface answered from the committed receipt."""

    def __init__(self, replay: _OwnerReplay) -> None:
        self.r = replay

    def discover(self, *, owner, owner_candidates, source_images) -> Any:
        del owner, owner_candidates, source_images
        payload = self.r.epoch._discovery_payload(
            self.r.shard, self.r.clip_uid, self.r.owner_plan
        )
        if str(payload.get("status")) == "discovery_failed":
            raise _sam_failure_exception(payload)
        return SubjectAttributeDiscovery.model_validate(payload.get("discovery"))


class _OwnerSegmentationBackend:
    """Legacy segmentation surface answered from the SAM receipts."""

    def __init__(self, replay: _OwnerReplay) -> None:
        self.r = replay

    def segment_frame(
        self, *, frame_path: Path, frame_slot: int, grounding_prompt: str
    ) -> list[Any]:
        r = self.r
        slot = int(frame_slot)
        # The legacy probe index is the only job index authority: the frozen
        # owner-plan order disagrees with it as soon as the owner reference's own
        # frame is a candidate, because legacy sorts that frame to the end.
        state, probe_index = r.resolve_probe(grounding_prompt, slot)
        candidate = r.ordered_candidates()[probe_index]
        if int(candidate.frame_slot) != slot:
            raise SubjectAttributeDurableError(
                "SAM probe frame slot drifted from its legacy probe candidate: "
                f"expected {int(candidate.frame_slot)}, got {slot}"
            )
        expected = (r.storage.root / candidate.image_path).resolve(strict=False)
        if Path(frame_path).resolve(strict=False) != expected:
            raise SubjectAttributeDurableError(
                "SAM probe input drifted from its frozen candidate"
            )
        job = r.chain_for(state)[probe_index]
        target = dict(job.target)
        if str(target.get("probe_index")) != str(probe_index) or str(
            target.get("owner_candidate_id")
        ) != str(candidate.candidate_id):
            raise SubjectAttributeDurableError(
                f"SAM probe job {job.job_id()} is not the frozen probe "
                f"{probe_index} of {r.clip_uid}/{r.owner_entity_id}"
            )
        payload = r.receipts_for(state).get(probe_index)
        if payload is None:
            raise _PendingModelCall(job)
        if str(payload.get("status")) == "sam_failed":
            raise _sam_failure_exception(payload)
        r.add_sam_seconds(payload)
        return list(r.epoch._load_sam_masks(payload))

    def segment_generated_frame(
        self, *, frame_path: Path, grounding_prompt: str
    ) -> list[Any]:
        return self.r.segment_generated_frame(
            frame_path=frame_path, grounding_prompt=grounding_prompt
        )


class _OwnerReviewClient:
    """Legacy raw review and last-resort bbox review, from the receipts."""

    def __init__(self, replay: _OwnerReplay) -> None:
        self.r = replay

    def review(self, *, owner, candidates) -> Any:
        del owner
        r = self.r
        epoch = r.epoch
        batch = list(candidates)
        if not batch:
            raise SubjectAttributeDurableError("raw review batch is empty")
        states = [r.state_for(str(candidate.attribute_id)) for candidate in batch]
        rank: int | None = None
        for state, candidate in zip(states, batch, strict=True):
            ids = [
                str(option.owner_candidate.candidate_id)
                for option in state.options
            ]
            try:
                index = ids.index(str(candidate.owner_candidate.candidate_id))
            except ValueError as exc:
                raise SubjectAttributeDurableError(
                    f"raw review candidate {candidate.owner_candidate.candidate_id} "
                    "is not a frozen option"
                ) from exc
            if rank is None:
                rank = index
            elif rank != index:
                raise SubjectAttributeDurableError(
                    "raw review batch mixes candidate ranks"
                )
        job = epoch._expected_raw_review_job(
            r.shard,
            r.clip_uid,
            r.owner_plan,
            r.discovery,
            r.discovery_job_id,
            states,
            batch,
            candidate_rank=int(rank or 0),
        )
        payload = epoch._committed_payload_or_none(job)
        if payload is None:
            raise _PendingModelCall(job)
        # Legacy wraps the whole round in its timing, so a failed round counts.
        r.add_qwen_seconds(payload, count=True)
        if str(payload.get("status")) == "review_failed":
            raise _sam_failure_exception(payload)
        try:
            return SubjectAttributeReviewBatch.model_validate(
                payload.get("review_batch")
            )
        except Exception as exc:
            raise SubjectAttributeDurableError(
                "committed raw review payload is invalid"
            ) from exc

    def review_attribute_bbox(
        self, *, bbox_candidate, owner_context, attribute_type, attribute_phrase
    ) -> Any:
        r = self.r
        epoch = r.epoch
        bbox_sha = _image_png_sha256(bbox_candidate)
        context_sha = _image_png_sha256(_resize_qwen_input_image(owner_context))
        for state in r.states:
            if state.attribute_id in r.bbox_used or not state.options:
                continue
            discovered = state.attribute_plan["discovered"]
            if str(discovered["attribute_type"]) != str(attribute_type):
                continue
            if str(discovered["phrase"]) != str(attribute_phrase):
                continue
            candidate = state.options[0]
            expected_bbox = _image_png_sha256(
                _attribute_bbox_crop(
                    candidate.source_image,
                    candidate.attribute_mask,
                    crop_padding_ratio=float(epoch.config.pair.crop_padding_ratio),
                )
            )
            if expected_bbox != bbox_sha:
                continue
            expected_context = _image_png_sha256(
                _resize_qwen_input_image(
                    build_candidate_context_image(
                        candidate.source_image, candidate.owner_candidate.mask
                    )
                )
            )
            if expected_context != context_sha:
                continue
            r.bbox_used.add(state.attribute_id)
            job = epoch._expected_bbox_review_job(
                r.shard,
                r.clip_uid,
                r.owner_plan,
                state.attribute_plan,
                candidate,
                0,
                context_sha,
                bbox_sha,
            )
            payload = epoch._committed_payload_or_none(job)
            if payload is None:
                raise _PendingModelCall(job)
            if str(payload.get("status")) == "bbox_failed":
                raise _sam_failure_exception(payload)
            try:
                return SubjectAttributeBboxReview.model_validate(
                    payload.get("bbox_review")
                )
            except Exception as exc:
                raise SubjectAttributeDurableError(
                    "committed bbox review payload is invalid"
                ) from exc
        raise SubjectAttributeDurableError(
            "bbox review has no frozen semantic candidate for this attribute"
        )


class _OwnerCompletionBackend:
    """Legacy Boogu completion surface answered from the generation receipt."""

    def __init__(self, replay: _OwnerReplay) -> None:
        self.r = replay

    def attribute_completion(
        self, *, source_path: Path, output_path: Path, instruction: str, seed: int
    ) -> dict[str, object]:
        del seed
        r = self.r
        epoch = r.epoch
        state, rank, candidate = r.completion_target(source_path)
        frozen_seed = epoch._completion_seed(
            r.shard,
            r.clip_uid,
            r.owner_entity_id,
            state.attribute_id,
            rank,
            str(candidate.owner_candidate.candidate_id),
        )
        generate_job = epoch._expected_completion_generate_job(
            r.shard,
            r.clip_uid,
            r.owner_plan,
            state.attribute_plan,
            candidate,
            rank,
            frozen_seed,
        )
        r.begin_completion(state, rank, candidate, generate_job)
        del instruction
        expected_source = epoch._completion_input_bytes(candidate.crop)
        if (
            not Path(source_path).is_file()
            or Path(source_path).read_bytes() != expected_source
        ):
            _save_completion_input(Path(source_path), candidate.crop)
        payload = epoch._committed_payload_or_none(generate_job)
        if payload is None:
            raise _PendingModelCall(generate_job)
        # Legacy adds the reported duration before it touches the output file, so
        # a generation that reported a duration and then failed still counts it.
        r.add_completion_seconds(payload)
        if str(payload.get("status")) == "completion_failed":
            raise _sam_failure_exception(payload)
        epoch._generated_png_bytes(r.storage, payload)
        return {
            "width": int(payload.get("width", 0)),
            "height": int(payload.get("height", 0)),
            "model_call_time_seconds": float(
                payload.get("model_call_time_seconds", 0.0)
            ),
        }


class _OwnerCompletionJudge:
    """Legacy completion review surface answered from the review receipt."""

    def __init__(self, replay: _OwnerReplay) -> None:
        self.r = replay

    def review(
        self,
        *,
        source_attribute,
        source_bbox,
        generated_candidate,
        attribute_type,
        attribute_phrase,
    ) -> Any:
        del source_attribute, source_bbox, attribute_type, attribute_phrase
        return self.r.completion_review(generated_candidate=generated_candidate)


@dataclass(frozen=True)
class _CompletionChain:
    """One candidate's completion chain, derived from its durable receipts.

    ``status`` is ``None`` while a prerequisite is still missing, and one of
    ``COMPLETION_TERMINAL_STATUSES`` once the chain is terminal;
    ``review_required`` is the non-terminal stage before the review receipt, so
    it is reported here but never persisted.
    """

    attribute_id: str
    attribute_type: str
    candidate_rank: int
    owner_candidate_id: str
    candidate: Any
    seed: int
    generate_job: ModelJob
    sam_job: ModelJob | None
    review_job: ModelJob | None
    status: str | None
    reason: str | None
    next_job: ModelJob | None
    review: Any | None
    completed_crop_sha256: str | None

    @property
    def accepted(self) -> bool:
        return self.status == COMPLETION_ACCEPTED

    @property
    def terminal(self) -> bool:
        return self.status in COMPLETION_TERMINAL_STATUSES


_COMPLETION_REVIEW_FAILED = "review_failed"
COMPLETION_REVIEW_FAILED = _COMPLETION_REVIEW_FAILED
COMPLETION_TERMINAL_STATUSES = COMPLETION_TERMINAL_STATUSES + (
    _COMPLETION_REVIEW_FAILED,
)


class _PendingModelCall(BaseException):
    """Internal abort: the legacy algorithm reached a call with no receipt.

    This derives from ``BaseException`` on purpose. The legacy subject attribute
    code isolates every model call with ``except Exception`` so that one bad
    frame, attribute or review round cannot abort a whole clip; an epoch cannot
    let a pending call be mistaken for a model failure, so the pending signal has
    to travel through those handlers untouched and abort the invocation. The
    legacy policy is then re-run from scratch next round with one more receipt
    committed, exactly like 8b's SAM probe chain.
    """

    def __init__(self, job: ModelJob) -> None:
        super().__init__(f"model call {job.job_id()} has no committed receipt")
        self.job = job


class _ReplaySegmentationBackend:
    """Legacy ``AttributeFrameSegmentationBackend`` backed by SAM receipts.

    Committed probes replay their exact ledger masks; a committed ``sam_failed``
    probe re-raises an exception of the same type and message so the legacy
    ``probe_failures`` reason stays identical; a probe without a receipt aborts
    the invocation with its own job through ``_PendingModelCall``.
    """

    def __init__(
        self,
        *,
        epoch: SubjectAttributeEpochRunner,
        ordered: Sequence[Any],
        receipts: Mapping[int, Mapping[str, Any]],
        storage: RunStorage,
        grounding_prompt: str,
        chain: Sequence[ModelJob] = (),
        generated: Any = None,
    ) -> None:
        self._epoch = epoch
        self._ordered = list(ordered)
        self._receipts = dict(receipts)
        self._storage = storage
        self._grounding_prompt = grounding_prompt
        self._chain = list(chain)
        #: 8c: the completion chain currently being replayed, if any. Legacy
        #: calls completion generation, then the generated-frame segmentation,
        #: then the review, so one context carries the whole chain.
        self.generated = generated
        self._by_slot: dict[int, int] = {}
        for index, candidate in enumerate(self._ordered):
            self._by_slot[int(candidate.frame_slot)] = index

    def _pending(self, index: int) -> _PendingModelCall:
        if index >= len(self._chain):
            raise SubjectAttributeDurableError(
                f"SAM probe {index} has no frozen job to unlock"
            )
        return _PendingModelCall(self._chain[index])

    def segment_frame(
        self, *, frame_path: Path, frame_slot: int, grounding_prompt: str
    ) -> list[Any]:
        index = self._by_slot.get(int(frame_slot))
        if index is None:
            raise SubjectAttributeDurableError(
                f"SAM probe frame slot {frame_slot} is not a frozen owner candidate"
            )
        candidate = self._ordered[index]
        expected_path = (self._storage.root / candidate.image_path).resolve(
            strict=False
        )
        if Path(frame_path) != expected_path or grounding_prompt != (
            self._grounding_prompt
        ):
            raise SubjectAttributeDurableError(
                "SAM probe input drifted from its frozen candidate"
            )
        payload = self._receipts.get(index)
        if payload is None:
            raise self._pending(index)
        if str(payload.get("status")) == "sam_failed":
            raise _sam_failure_exception(payload)
        return self._epoch._load_sam_masks(payload)

    def segment_generated_frame(
        self, *, frame_path: Path, grounding_prompt: str
    ) -> list[Any]:
        """The completion SAM call of the chain the replay is currently in."""
        if self.generated is None:
            raise SubjectAttributeDurableError(
                "generated-frame segmentation was reached outside a completion"
            )
        return self.generated.segment_generated_frame(
            frame_path=frame_path, grounding_prompt=grounding_prompt
        )


@dataclass(frozen=True)
class _AttributeReplay:
    """One attribute's replayed legacy selection state."""

    attribute_id: str
    attribute_type: str
    attribute_plan: dict[str, Any]
    marker: dict[str, Any] | None
    options: tuple[Any, ...]
    record: Any | None
    processing_failure: int
    pending_job: ModelJob | None
    sam_job_ids: dict[str, str]

    @property
    def has_candidate2(self) -> bool:
        return len(self.options) > 1


@dataclass(frozen=True)
class SubjectAttributeEpochStats:
    """Legacy clip counts merged across one shard.

    ``to_dict()`` is exactly the merged ``ClipEnrichmentResult.to_counts()``
    payload; the public count schema stays legacy.
    """

    counts: Mapping[str, int | float] = field(default_factory=dict)
    terminal_clips: int = 0
    no_work_clips: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.counts)


def resolve_subject_attribute_discovery_client(
    handle: Any, config: V3Config
) -> ResolvedDiscoveryClient:
    """Turn the epoch's handle into a subject attribute discovery client.

    An endpoint string builds an owned client against the configured candidate
    judge, an injected client object is reused as-is and never closed, and
    ``None`` is an error rather than a fallback to another model.
    """
    if handle is None:
        raise SubjectAttributeEpochError(
            "subject attribute discovery needs a client handle"
        )
    return resolve_subject_attribute_qwen_client(handle, config)


def resolve_subject_attribute_qwen_client(
    handle: Any, config: V3Config, *, judge: bool = False
) -> ResolvedDiscoveryClient:
    """The shared Qwen endpoint resolver for every subject attribute review call.

    A real Resource Epoch handle can be an endpoint string such as
    ``http://127.0.0.1:....../v1``, in which case an owned client is built
    against the configured candidate judge and closed after the call. An
    injected client object is reused as-is and never closed, and ``None`` is an
    error rather than a fallback to another model.
    """
    if handle is None:
        raise SubjectAttributeEpochError(
            "subject attribute review needs a client handle"
        )
    if isinstance(handle, str):
        service = replace(config.qwen.candidate_judge, base_url=handle)
        client = (
            QwenSubjectAttributeCompletionJudge(
                service,
                completion_component="qwen_attribute_completion_review",
            )
            if judge
            else QwenSubjectAttributeClient(service)
        )
        return ResolvedDiscoveryClient(client, owned=True)
    return ResolvedDiscoveryClient(handle, owned=False)


def _receipt_seconds(
    payload: Mapping[str, Any], key: str, *, label: str
) -> float:
    """One receipt-reported model duration, validated and never negative."""
    value = payload.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SubjectAttributeDurableError(
            f"committed {label} has an invalid {key}"
        )
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise SubjectAttributeDurableError(
            f"committed {label} has an invalid {key}"
        )
    return seconds


def _completion_adapter(handle: Any, storage: RunStorage, direct: bool) -> Any:
    """The production Boogu completion adapter, imported without a cycle.

    A handle that already exposes ``attribute_completion`` is used as-is; a raw
    Boogu ``.edit`` backend is wrapped by the production runtime adapter so the
    epoch never restates the completion prompt, size or seed policy.
    """
    if direct:
        return handle
    from r2v_data_v2.v3.post_mask_runtime import _AttributeCompletion

    return _AttributeCompletion(handle, storage)


def _close_owned(resolved: Any) -> None:
    """Close a client the epoch owns; an injected client is never closed."""
    if not resolved.owned:
        return
    close = getattr(resolved.client, "close", None)
    if callable(close):
        close()


def _read_json(path: Path) -> dict[str, Any] | None:
    """Durable reader: missing is ``None``, corrupt is always fail-closed."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SubjectAttributeDurableError(
            f"invalid durable Subject Attributes JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise SubjectAttributeDurableError(
            f"durable Subject Attributes JSON is not an object: {path}"
        )
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Create-or-validate with the epoch durable-state writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if _read_json(path) != dict(payload):
            raise SubjectAttributeDurableError(
                f"durable Subject Attributes file drifted: {path}"
            )
        return
    atomic_write_json(path, dict(payload))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    """Contiguous uint8 mask bytes plus its shape."""
    body = np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    shape = f":{mask.shape[0]}x{mask.shape[1]}".encode("ascii")
    return _sha256_bytes(body + shape)


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _discovery_context_sha256(storage: RunStorage, candidate: Any) -> str:
    """SHA256 of the exact PNG bytes the discovery model is shown for this candidate.

    Legacy ``QwenSubjectAttributeClient.discover`` renders
    ``_resize_qwen_input_image(build_candidate_context_image(source, mask))`` and
    encodes it as PNG, so the frozen digest binds real pixels rather than a path.
    """
    source = _source_image(storage, candidate)
    context = build_candidate_context_image(source, candidate.mask)
    image = _resize_qwen_input_image(context)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _sha256_bytes(buffer.getvalue())


def _candidate_descriptor(storage: RunStorage, candidate: Any) -> dict[str, Any]:
    """Everything about one owner candidate that the discovery call depends on."""
    source_path = storage.root / candidate.image_path
    return {
        "candidate_id": str(candidate.candidate_id),
        "entity_id": str(candidate.entity_id),
        "frame_slot": int(candidate.frame_slot),
        "source_frame_index": int(candidate.source_frame_index),
        "image_path": str(candidate.image_path),
        "bbox_xyxy": [int(value) for value in candidate.bbox_xyxy],
        "area_pixels": int(candidate.area_pixels),
        "area_ratio": float(candidate.area_ratio),
        "bbox_fill_ratio": float(candidate.bbox_fill_ratio),
        "border_contact_count": int(candidate.border_contact_count),
        "normalized_center_distance": float(candidate.normalized_center_distance),
        "sharpness_score": float(candidate.sharpness_score),
        "significant_component_count": int(candidate.significant_component_count),
        "largest_component_ratio": float(candidate.largest_component_ratio),
        "second_largest_component_ratio": float(
            candidate.second_largest_component_ratio
        ),
        "source_image_sha256": _file_sha256(source_path),
        "mask_shape": [int(candidate.mask.shape[0]), int(candidate.mask.shape[1])],
        "mask_sha256": _mask_sha256(candidate.mask),
        "discovery_context_sha256": _discovery_context_sha256(storage, candidate),
    }


class SubjectAttributeEpochRunner:
    """Durable owner/discovery foundation for the Subject Attributes stage."""

    def __init__(
        self,
        config: V3Config,
        storages: Mapping[str, RunStorage],
        ledger: Any,
        *,
        eligible_clip_uids_by_shard: Mapping[str, Sequence[str]],
        emit: Any = None,
        cpu_workers: int | None = None,
    ) -> None:
        if bool(getattr(config.subject_attribute_gme, "enabled", False)):
            raise SubjectAttributeEpochError(
                "Post-Mask Resource Epoch requires subject attribute GME disabled"
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
        self.emit = emit
        # Invocation-local execution cache only. The first access still
        # re-derives and validates the complete frozen clip plan from live
        # upstream artifacts. Subsequent accesses in the same locked resource-
        # epoch invocation reuse that validated plan instead of rebuilding every
        # owner candidate, image digest and discovery context on every receipt.
        # A restart constructs a fresh runner and therefore revalidates from
        # disk, preserving fail-closed durable semantics.
        self._clip_plan_cache: dict[tuple[str, str], dict[str, Any]] = {}
        # Candidate masks are full-resolution, so keep this invocation cache
        # deliberately bounded. It exists only to avoid rebuilding the same
        # owner evidence at every receipt boundary; a restart rebuilds and
        # revalidates everything from durable inputs.
        self._owner_candidate_cache: dict[
            tuple[str, str, str], tuple[Any, ...]
        ] = {}
        self._owner_candidate_cache_limit = 16
        # Receipt-boundary replay caches. Both hold only what a newly arriving
        # receipt cannot change: the frozen owner prefix, and an attribute
        # selection that is already terminal and whose durable marker still
        # verifies. Everything that a receipt *can* change - review verdicts,
        # completion outcomes, routes, the owner outcome - is still read from
        # the ledger every time. A restart starts cold and re-derives it all.
        self._owner_context_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._attribute_selection_cache: dict[
            tuple[str, str, str, str], Any
        ] = {}
        self._owner_context_cache_limit = 32
        self._attribute_selection_cache_limit = 32
        self._candidate_cache_lock = threading.Lock()
        # The selection cache is read by model-execution workers while the main
        # thread replays the previous receipt, so every access is guarded. The
        # heavy replay itself always happens outside the lock.
        self._attribute_selection_cache_lock = threading.Lock()
        self._replay_counter_lock = threading.Lock()
        #: Execution-only counters. They measure how much CPU replay was skipped
        #: and are never part of any job identity, receipt or public schema.
        self.replay_counters: dict[str, int] = {
            "owner_context_cache_hit": 0,
            "owner_context_cache_miss": 0,
            "attribute_selection_cache_hit": 0,
            "attribute_selection_cache_miss": 0,
            "owner_graph_rebuilds": 0,
            "attribute_selection_rebuilds": 0,
            "owner_candidate_cache_hit": 0,
            "owner_candidate_cache_miss": 0,
            "owner_candidate_rebuilds": 0,
        }
        #: Execution-only seed counters for the clip-plan derivation fan-out.
        #: ``clip_plan_derive_wall_seconds`` is a float, so this mapping stays
        #: separate from the int-only replay counters. Never part of
        #: ``SubjectAttributeEpochStats``, a ModelJob identity, a receipt, the
        #: durable stage counts, a public schema or the config fingerprint.
        self._seed_counters_lock = threading.Lock()
        self.seed_counters: dict[str, int | float] = {
            "clip_plan_derive_tasks": 0,
            "clip_plan_derive_peak_inflight": 0,
            "clip_plan_derive_batches": 0,
            "clip_plan_derive_wall_seconds": 0.0,
        }

    def _bump_replay_counter(self, key: str, delta: int = 1) -> None:
        with self._replay_counter_lock:
            self.replay_counters[key] += delta

    def _bump_seed_counter(self, key: str, delta: int = 1) -> None:
        with self._seed_counters_lock:
            self.seed_counters[key] = int(self.seed_counters.get(key, 0)) + delta

    def _note_seed_wall(self, seconds: float) -> None:
        with self._seed_counters_lock:
            self.seed_counters["clip_plan_derive_wall_seconds"] = float(
                self.seed_counters.get("clip_plan_derive_wall_seconds", 0.0)
            ) + float(seconds)

    def _note_seed_inflight(self, inflight: int) -> None:
        with self._seed_counters_lock:
            self.seed_counters["clip_plan_derive_peak_inflight"] = max(
                int(self.seed_counters.get("clip_plan_derive_peak_inflight", 0)),
                int(inflight),
            )

    def _clip_plan_derive_budget(self) -> int:
        """How many clip plans may be in flight before the batch is applied.

        Derivation is pure read-only CPU, so the only thing that grows with the
        dataset is how many derived plans wait for their canonical application
        slot. Keeping that at ``max(cpu_workers, 2 * cpu_workers)`` makes
        residency O(cpu_workers) instead of O(eligible clips), which is what a
        ten-thousand-clip shard needs.
        """
        workers = int(self.cpu_workers)
        return max(workers, workers * 2)

    # -- durable paths ---------------------------------------------------------

    def _semantic(self, *parts: str) -> Path:
        return (
            Path(self.ledger.root)
            / "semantic"
            / "subject_attributes"
            / Path(*parts)
        )

    def _clip_plan_path(self, shard: str, clip_uid: str) -> Path:
        return self._semantic("plans", shard, f"{clip_uid}.json")

    def _owner_dir(self, shard: str, clip_uid: str, owner_entity_id: str) -> Path:
        return self._semantic("owners", shard, clip_uid, owner_entity_id)

    def _owner_plan_path(self, shard: str, clip_uid: str, owner: str) -> Path:
        return self._owner_dir(shard, clip_uid, owner) / "plan.json"

    def _owner_outcome_path(self, shard: str, clip_uid: str, owner: str) -> Path:
        return self._owner_dir(shard, clip_uid, owner) / "outcome.json"

    def _clip_outcome_path(self, shard: str, clip_uid: str) -> Path:
        return self._semantic("outcomes", shard, f"{clip_uid}.json")

    def _storage_for(self, shard: str) -> RunStorage:
        storage = self.storages.get(shard)
        if storage is None:
            raise SubjectAttributeEpochError(
                f"no run storage for canonical shard {shard!r}"
            )
        return storage

    @staticmethod
    def _output_root(storage: RunStorage) -> Path:
        return storage.root / SUBJECT_ATTRIBUTE_OUTPUT_DIRNAME

    # -- policy identity -------------------------------------------------------

    def _completion_mode(self) -> str | None:
        completion = getattr(
            getattr(self.config, "subject_attributes", None), "completion", None
        )
        return (
            "boogu_completion_v1"
            if completion is not None and bool(completion.enabled)
            else None
        )

    def _discovery_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the discovery model behavior.

        Only what changes *what the model is asked and judged by* is bound.
        Endpoint, credentials, timeout, host and concurrency are execution only.
        """
        service = self.config.qwen.candidate_judge
        return {
            "policy_version": SUBJECT_ATTRIBUTE_DISCOVERY_POLICY_VERSION,
            "model": str(service.model),
            "temperature": float(service.temperature),
            "max_tokens": int(service.max_tokens),
            "system_prompt_sha256": _sha256_bytes(
                DISCOVERY_SYSTEM_PROMPT.encode("utf-8")
            ),
            "discovery_schema_sha256": _sha256_bytes(
                canonical_json(
                    SubjectAttributeDiscovery.model_json_schema()
                ).encode("utf-8")
            ),
            "max_attributes_per_owner": int(MAX_ATTRIBUTES_PER_OWNER),
            "qwen_input_max_long_side_pixels": int(QWEN_INPUT_MAX_LONG_SIDE_PIXELS),
            "duplicate_type_policy": "first_wins_original_order",
            "mode": "qwen_subject_attribute_discovery_v1",
        }

    def _policy_identity(self) -> dict[str, Any]:
        return {
            "policy_version": SUBJECT_ATTRIBUTE_POLICY_VERSION,
            "completion_mode": self._completion_mode(),
            "reference_edit_enabled": bool(self.config.reference_edit.enabled),
            "reference_integrity_enabled": bool(
                self.config.reference_integrity.enabled
            ),
            "discovery": self._discovery_policy_identity(),
        }

    # -- clip plan -------------------------------------------------------------

    def _effective_clip(self, clip: Any) -> Any:
        """The production export view: the clip with its effective export state."""
        effective_export = evaluate_export_state(
            clip,
            require_reference_edit=bool(self.config.reference_edit.enabled),
            require_reference_integrity=bool(
                self.config.reference_integrity.enabled
            ),
        )
        return clip.model_copy(update={"export": effective_export})

    @staticmethod
    def _clip_digest(clip: Any) -> str:
        """The upstream visual authority, deliberately without the live export."""
        payload = canonical_json(clip.model_dump(mode="json", exclude={"export"}))
        return _sha256_bytes(payload.encode("utf-8"))

    def _model_digest(self, value: Any) -> str:
        return _sha256_bytes(
            canonical_json(value.model_dump(mode="json")).encode("utf-8")
        )

    def _owner_candidates(
        self,
        storage: RunStorage,
        clip_uid: str,
        owner: Any,
        frames: Any,
        masks: Any,
    ) -> list[Any]:
        """The legacy candidate pool, with the legacy catch scope."""
        try:
            return build_entity_reference_candidates(
                self.config,
                storage,
                clip_uid=clip_uid,
                entity=owner,
                frames=frames,
                masks=masks,
            )[:MAX_ATTRIBUTES_PER_OWNER]
        except (OSError, ValueError):
            return []

    def _derive_clip_plan(self, storage: RunStorage, clip_uid: str) -> dict[str, Any]:
        """Derive the whole frozen clip plan from live state."""
        clip = storage.read_clip(clip_uid)
        effective_clip = self._effective_clip(clip)
        base: dict[str, Any] = {
            "schema": SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA,
            "clip_uid": clip_uid,
            "policy": self._policy_identity(),
            "clip_digest": self._clip_digest(clip),
            "effective_export": effective_clip.export.model_dump(mode="json"),
            "frames_sha256": None,
            "masks_sha256": None,
            "classification": CLIP_NO_WORK,
            "subject_owners": [],
        }
        if (
            not effective_clip.export.accepted
            or clip.annotation is None
            or clip.annotation.status != "ready"
            or clip.pairing is None
            or clip.pairing.status != "ready"
        ):
            # Legacy returns an empty clip result here: a terminal CPU branch,
            # never durable corruption.
            return base
        try:
            frames = storage.read_frames(clip_uid)
            masks = storage.read_masks(clip_uid)
        except (OSError, ValueError):
            return base
        owners: list[dict[str, Any]] = []
        for owner in clip.annotation.entities:
            if owner.reference_type != "subject":
                continue
            candidates = self._owner_candidates(
                storage, clip_uid, owner, frames, masks
            )
            eligibility = evaluate_owner_eligibility(
                effective_clip,
                entity_id=owner.entity_id,
                has_usable_candidate_evidence=bool(candidates),
            )
            owners.append(
                {
                    "owner_entity_id": str(owner.entity_id),
                    "owner_phrase": str(owner.phrase),
                    "owner_grounding_prompt": str(owner.grounding_prompt),
                    "eligible": bool(eligibility.eligible),
                    "eligibility_reason": str(eligibility.reason),
                    "candidates": [
                        _candidate_descriptor(storage, candidate)
                        for candidate in candidates
                    ],
                }
            )
        base.update(
            {
                "frames_sha256": self._model_digest(frames),
                "masks_sha256": self._model_digest(masks),
                "classification": CLIP_OWNERS,
                "subject_owners": owners,
            }
        )
        return base

    def _checked_derive_clip_plan(
        self, storage: RunStorage, clip_uid: str
    ) -> dict[str, Any]:
        """``_derive_clip_plan`` plus the frozen error envelope.

        The derivation itself is pure read-only CPU (live upstream reads, no
        durable or debug write at any point), so any thread may run it; only the
        envelope that turns an unexpected failure into a durable error has to be
        identical on the serial path, on the worker path and on a restart.
        """
        try:
            return self._derive_clip_plan(storage, clip_uid)
        except SubjectAttributeDurableError:
            raise
        except Exception as exc:
            raise SubjectAttributeDurableError(
                f"cannot derive the Subject Attributes clip plan for {clip_uid!r}"
            ) from exc

    def _apply_derived_clip_plan(
        self, shard: str, clip_uid: str, expected_plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Main-thread durable authority for one already derived clip plan.

        Everything that writes - the durable plan file and the invocation cache
        - happens here, on the caller's thread, in canonical clip order. A
        derived plan may therefore never be published speculatively.
        """
        cache_key = (shard, clip_uid)
        cached = self._clip_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        existing = _read_json(self._clip_plan_path(shard, clip_uid))
        if existing is None:
            _write_json_once(self._clip_plan_path(shard, clip_uid), expected_plan)
            self._clip_plan_cache[cache_key] = expected_plan
            return expected_plan
        if existing != expected_plan:
            # Whole-plan equality: an extra, missing or changed field is drift.
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes clip plan drifted for {clip_uid!r}"
            )
        self._clip_plan_cache[cache_key] = existing
        return existing

    def _clip_plan(self, shard: str, clip_uid: str) -> dict[str, Any]:
        """Create once, validate once per invocation, then reuse while locked."""
        cached = self._clip_plan_cache.get((shard, clip_uid))
        if cached is not None:
            return cached
        storage = self._storage_for(shard)
        expected = self._checked_derive_clip_plan(storage, clip_uid)
        return self._apply_derived_clip_plan(shard, clip_uid, expected)

    def _eligible_owners(self, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            owner for owner in plan.get("subject_owners", []) if owner.get("eligible")
        ]

    # -- owner plan ------------------------------------------------------------

    def _preexisting_owner_artifact(
        self,
        storage: RunStorage,
        clip_uid: str,
        owner_entity_id: str,
        attribute_id_start: int,
    ) -> tuple[Any | None, str | None]:
        """The legacy owner artifact cache check, with legacy semantics."""
        path = _owner_artifact_path(
            self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
        )
        artifact = _load_cached_owner_artifact(
            path,
            output_root=self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
            attribute_id_start=attribute_id_start,
            expected_gme_screen_mode=None,
            expected_completion_mode=self._completion_mode(),
        )
        if artifact is None:
            return None, None
        return artifact, _file_sha256(path)

    def _owner_plan(
        self,
        shard: str,
        clip_uid: str,
        owner: Mapping[str, Any],
        owner_index: int,
        attribute_id_start: int,
    ) -> dict[str, Any]:
        """Create-once owner plan for the current active owner."""
        storage = self._storage_for(shard)
        path = self._owner_plan_path(shard, clip_uid, str(owner["owner_entity_id"]))
        existing = _read_json(path)
        if existing is not None:
            return self._validate_owner_plan(
                clip_uid, owner, owner_index, attribute_id_start, existing
            )
        artifact, artifact_sha256 = self._preexisting_owner_artifact(
            storage,
            clip_uid,
            str(owner["owner_entity_id"]),
            attribute_id_start,
        )
        payload = {
            "schema": SUBJECT_ATTRIBUTE_OWNER_PLAN_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": str(owner["owner_entity_id"]),
            "owner_index": int(owner_index),
            "attribute_id_start": int(attribute_id_start),
            "owner_phrase": str(owner["owner_phrase"]),
            "owner_grounding_prompt": str(owner["owner_grounding_prompt"]),
            "completion_mode": self._completion_mode(),
            "gme_screen_mode": None,
            "candidates": [dict(item) for item in owner.get("candidates", [])],
            "discovery_policy": self._discovery_policy_identity(),
            "preexisting_owner_artifact": {
                "valid": artifact is not None,
                "sha256": artifact_sha256,
            },
        }
        _write_json_once(path, payload)
        return payload

    def _validate_owner_plan(
        self,
        clip_uid: str,
        owner: Mapping[str, Any],
        owner_index: int,
        attribute_id_start: int,
        existing: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Re-derive the owner plan and require exact equality.

        ``preexisting_owner_artifact`` is the one decision-time fact that cannot
        be re-derived: the epoch may already have written its own terminal
        artifact over a cache miss, and that is exactly the supported
        receipt-before-marker crash window. Its validity is frozen and, when it
        was valid, its bytes are verified instead.
        """
        owner_entity_id = str(owner["owner_entity_id"])
        expected = {
            "schema": SUBJECT_ATTRIBUTE_OWNER_PLAN_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": owner_entity_id,
            "owner_index": int(owner_index),
            "attribute_id_start": int(attribute_id_start),
            "owner_phrase": str(owner["owner_phrase"]),
            "owner_grounding_prompt": str(owner["owner_grounding_prompt"]),
            "completion_mode": self._completion_mode(),
            "gme_screen_mode": None,
            "candidates": [dict(item) for item in owner.get("candidates", [])],
            "discovery_policy": self._discovery_policy_identity(),
        }
        label = f"{clip_uid}/{owner_entity_id}"
        if set(existing) != set(expected) | {"preexisting_owner_artifact"}:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner plan keyset drifted for {label}"
            )
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SubjectAttributeDurableError(
                    f"frozen Subject Attributes owner plan {key} drifted for "
                    f"{label}"
                )
        preexisting = existing.get("preexisting_owner_artifact")
        if not isinstance(preexisting, dict) or set(preexisting) != {
            "valid",
            "sha256",
        }:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner plan artifact cache is malformed "
                f"for {label}"
            )
        valid = preexisting["valid"]
        digest = preexisting["sha256"]
        if not isinstance(valid, bool):
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact cache flag is invalid "
                f"for {label}"
            )
        if valid:
            if not isinstance(digest, str) or len(digest) != 64:
                raise SubjectAttributeDurableError(
                    f"frozen Subject Attributes owner artifact cache digest is "
                    f"invalid for {label}"
                )
        elif digest is not None:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact cache must not carry a "
                f"digest for {label}"
            )
        return dict(existing)

    def _verify_preexisting_artifact(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
    ) -> None:
        """A valid frozen legacy cache must still be the exact same artifact."""
        storage = self._storage_for(shard)
        owner_entity_id = str(owner_plan["owner_entity_id"])
        cached = owner_plan["preexisting_owner_artifact"]
        if not cached.get("valid"):
            return
        path = _owner_artifact_path(
            self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
        )
        if not path.is_file():
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact is missing for "
                f"{clip_uid}/{owner_entity_id}"
            )
        if _file_sha256(path) != str(cached.get("sha256")):
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact drifted for "
                f"{clip_uid}/{owner_entity_id}"
            )
        artifact, _sha = self._preexisting_owner_artifact(
            storage,
            clip_uid,
            owner_entity_id,
            int(owner_plan["attribute_id_start"]),
        )
        if artifact is None:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact is no longer valid for "
                f"{clip_uid}/{owner_entity_id}"
            )

    # -- discovery job ---------------------------------------------------------

    def _discovery_semantic_inputs(
        self,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidates = [dict(item) for item in owner_plan.get("candidates", [])]
        return {
            "clip_uid": clip_uid,
            "owner_entity_id": str(owner_plan["owner_entity_id"]),
            "owner_phrase": str(owner_plan["owner_phrase"]),
            "owner_grounding_prompt": str(owner_plan["owner_grounding_prompt"]),
            "candidate_provenance": [
                {
                    "candidate_id": item["candidate_id"],
                    "image_path": item["image_path"],
                    "source_frame_index": int(item["source_frame_index"]),
                    "mask_sha256": item["mask_sha256"],
                }
                for item in candidates
            ],
            "ordered_discovery_context_sha256": [
                item["discovery_context_sha256"] for item in candidates
            ],
            "discovery_policy": dict(owner_plan["discovery_policy"]),
        }

    def _expected_discovery_job(
        self, shard: str, clip_uid: str, owner_plan: Mapping[str, Any]
    ) -> ModelJob:
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs=self._discovery_semantic_inputs(clip_uid, owner_plan),
            model_identity=f"qwen:{self.config.qwen.candidate_judge.model}",
            target={"owner_entity_id": str(owner_plan["owner_entity_id"])},
        )

    def _materialize_frozen_candidate(
        self,
        storage: RunStorage,
        masks: Any,
        tracked_frames: Mapping[int, Any],
        descriptor: Mapping[str, Any],
    ) -> EntityReferenceCandidate:
        """One frozen candidate object, rebuilt from the plan and upstream data.

        Only two of its inputs cannot be serialized into the plan - the mask
        pixels and the source image the model is shown - so those are read again
        and required to hash exactly what the plan froze. Everything else is the
        frozen value: the candidate was already selected, ranked and numbered
        when the plan was derived, and re-deriving those metrics can only ever
        reproduce them or fail.
        """
        frame_slot = int(descriptor["frame_slot"])
        tracked_frame = tracked_frames.get(frame_slot)
        if (
            tracked_frame is None
            or not bool(tracked_frame.present)
            or not bool(tracked_frame.track_valid)
            or int(tracked_frame.area_pixels) <= 0
        ):
            raise ValueError("frozen candidate frame is no longer trackable")
        binary = _decode_candidate_mask(
            tracked_frame,
            expected_shape=(int(masks.height), int(masks.width)),
        )
        if _mask_sha256(binary) != str(descriptor["mask_sha256"]):
            raise ValueError("frozen candidate mask pixels changed")
        image_path = str(descriptor["image_path"])
        if _file_sha256(storage.root / image_path) != str(
            descriptor["source_image_sha256"]
        ):
            raise ValueError("frozen candidate source image changed")
        return EntityReferenceCandidate(
            candidate_id=str(descriptor["candidate_id"]),
            entity_id=str(descriptor["entity_id"]),
            frame_slot=frame_slot,
            source_frame_index=int(descriptor["source_frame_index"]),
            image_path=image_path,
            mask=binary,
            bbox_xyxy=tuple(int(value) for value in descriptor["bbox_xyxy"]),
            area_pixels=int(descriptor["area_pixels"]),
            area_ratio=float(descriptor["area_ratio"]),
            bbox_fill_ratio=float(descriptor["bbox_fill_ratio"]),
            border_contact_count=int(descriptor["border_contact_count"]),
            normalized_center_distance=float(
                descriptor["normalized_center_distance"]
            ),
            sharpness_score=float(descriptor["sharpness_score"]),
            significant_component_count=int(descriptor["significant_component_count"]),
            largest_component_ratio=float(descriptor["largest_component_ratio"]),
            second_largest_component_ratio=float(
                descriptor["second_largest_component_ratio"]
            ),
        )

    def _materialize_frozen_candidates(
        self, shard: str, clip_uid: str, owner_plan: Mapping[str, Any]
    ) -> list[EntityReferenceCandidate]:
        """The frozen owner candidates, without re-running the live selection.

        The clip plan already froze which candidates exist, in which order, with
        which identity, geometry and metrics, and the live selection that
        produced them ran once for this clip plan. Re-running it at every receipt
        boundary re-encoded every discovery context the model is shown - for
        evidence that has to come out identical or fail. This materializes the
        same objects from the plan plus the two upstream inputs that are checked
        by digest instead, so the model can only ever be shown the frozen
        evidence and a drifted upstream artifact still fails closed.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        frozen = [dict(item) for item in owner_plan.get("candidates", [])]
        try:
            storage = self._storage_for(shard)
            clip = storage.read_clip(clip_uid)
            owner = next(
                (
                    item
                    for item in clip.annotation.entities
                    if item.entity_id == owner_entity_id
                ),
                None,
            )
            if owner is None:
                raise SubjectAttributeDurableError(
                    f"owner {clip_uid}/{owner_entity_id} is missing from the "
                    "annotation"
                )
            masks = storage.read_masks(clip_uid)
            tracked = masks.entities.get(owner_entity_id)
            if tracked is None:
                raise ValueError("mask artifact has no tracked owner entity")
            if (
                tracked.reference_type != owner.reference_type
                or tracked.grounding_prompt != owner.grounding_prompt
            ):
                raise ValueError("mask artifact entity semantics changed")
            if tracked.status != "ready":
                raise ValueError("tracked owner entity is no longer ready")
            if [int(item.slot) for item in tracked.frames] != list(range(10)):
                raise ValueError("tracked entity slots are no longer ordered")
            tracked_frames = {int(item.slot): item for item in tracked.frames}
            return [
                self._materialize_frozen_candidate(
                    storage, masks, tracked_frames, descriptor
                )
                for descriptor in frozen
            ]
        except SubjectAttributeDurableError:
            raise
        except (OSError, ValueError):
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes candidate evidence drifted for "
                f"{clip_uid}/{owner_entity_id}"
            ) from None

    def _owner_candidate_objects(
        self, shard: str, clip_uid: str, owner_plan: Mapping[str, Any]
    ) -> list[Any]:
        """Materialize the frozen owner candidates once per invocation."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        cache_key = (shard, clip_uid, owner_entity_id)
        # Attribute replays can run on several threads, so the bounded cache is
        # guarded: losing an entry would only cost a materialization, but the
        # bound and the drift check must stay coherent.
        with self._candidate_cache_lock:
            cached = self._owner_candidate_cache.get(cache_key)
            if cached is not None:
                # Refresh insertion order for a tiny LRU without another dependency.
                self._owner_candidate_cache.pop(cache_key)
                self._owner_candidate_cache[cache_key] = cached
                self._bump_replay_counter("owner_candidate_cache_hit")
                return list(cached)
            self._bump_replay_counter("owner_candidate_cache_miss")
            self._bump_replay_counter("owner_candidate_rebuilds")

        candidates = self._materialize_frozen_candidates(shard, clip_uid, owner_plan)

        with self._candidate_cache_lock:
            if len(self._owner_candidate_cache) >= self._owner_candidate_cache_limit:
                oldest = next(iter(self._owner_candidate_cache))
                self._owner_candidate_cache.pop(oldest)
            self._owner_candidate_cache[cache_key] = tuple(candidates)
        return candidates

    # -- run -------------------------------------------------------------------

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        """Execute one model job. No prompt or schema is copied."""
        if job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            return self._run_sam_probe(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB:
            return self._run_raw_review(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB:
            return self._run_completion_generate(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB:
            return self._run_completion_sam(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_JOB:
            return self._run_completion_review(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_BBOX_REVIEW_JOB:
            return self._run_bbox_review(job, handle)
        if job.job_type != SUBJECT_ATTRIBUTE_DISCOVERY_JOB:
            raise SubjectAttributeEpochError(
                f"unsupported job type {job.job_type!r}"
            )
        shard = job.canonical_shard
        owner_entity_id = str(dict(job.target).get("owner_entity_id", ""))
        plan = self._clip_plan(shard, job.clip_uid)
        owner_plan = self._owner_plan_for(shard, plan, owner_entity_id)
        candidates = self._owner_candidate_objects(shard, job.clip_uid, owner_plan)
        expected = self._expected_discovery_job(shard, job.clip_uid, owner_plan)
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != expected.input_digest
            or job.model_identity != expected.model_identity
        ):
            return JobResult(
                OUTCOME_RETRYABLE_FAILED,
                detail="Subject Attributes discovery semantic identity drifted",
            )
        storage = self._storage_for(shard)
        clip = storage.read_clip(job.clip_uid)
        owner = next(
            item
            for item in clip.annotation.entities
            if item.entity_id == owner_entity_id
        )
        resolved = resolve_subject_attribute_discovery_client(handle, self.config)
        # Legacy builds the source images outside its discovery try, so an
        # unreadable source image stays an unexpected execution failure and is
        # never reported as a discovery failure.
        source_images = {
            candidate.image_path: _source_image(storage, candidate)
            for candidate in candidates
        }
        started = time.perf_counter()
        try:
            discovery = resolved.client.discover(
                owner=owner,
                owner_candidates=candidates,
                source_images=source_images,
            )
        except Exception as exc:  # noqa: BLE001 - legacy terminal failure scope
            elapsed = time.perf_counter() - started
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "discovery_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "qwen_model_call_time_seconds": elapsed,
                },
            )
        finally:
            if resolved.owned:
                close = getattr(resolved.client, "close", None)
                if callable(close):
                    close()
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "discovery",
                "discovery": discovery.model_dump(mode="json"),
                "qwen_model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _owner_plan_for(
        self, shard: str, plan: Mapping[str, Any], owner_entity_id: str
    ) -> dict[str, Any]:
        owners = self._eligible_owners(plan)
        index = next(
            (
                position
                for position, owner in enumerate(owners)
                if str(owner["owner_entity_id"]) == owner_entity_id
            ),
            None,
        )
        if index is None:
            raise SubjectAttributeDurableError(
                f"owner {plan.get('clip_uid')}/{owner_entity_id} is not an "
                "eligible frozen owner"
            )
        return self._owner_plan(
            shard,
            str(plan["clip_uid"]),
            owners[index],
            index,
            self._attribute_id_start(shard, str(plan["clip_uid"]), owners, index),
        )

    # -- owner accounting ------------------------------------------------------

    def _attribute_id_start(
        self,
        shard: str,
        clip_uid: str,
        owners: Sequence[Mapping[str, Any]],
        index: int,
    ) -> int:
        """1 + the record count of every previous *terminal* owner artifact."""
        start = 1
        for owner in owners[:index]:
            outcome = self._owner_outcome(
                shard, clip_uid, str(owner["owner_entity_id"])
            )
            if outcome is None:
                raise SubjectAttributeEpochError(
                    f"Subject Attributes clip {clip_uid!r} reached owner "
                    f"{owner['owner_entity_id']!r} before its predecessors were "
                    "terminal"
                )
            start += int(outcome["record_count"])
        return start

    @staticmethod
    def _accumulate_owner(
        totals: EnrichmentTotals,
        artifact: OwnerEnrichmentArtifact,
        *,
        preexisting: bool,
        records: list[Any],
    ) -> None:
        """The legacy per-owner accounting, exactly as legacy applies it."""
        if preexisting:
            totals.skipped_existing_owners += 1
        if artifact.failure_reason:
            totals.failure_reasons[artifact.failure_reason.split(":", 1)[0]] += 1
        totals.add_owner_metrics(artifact.metrics)
        if artifact.owner_is_human is True:
            totals.eligible_human_owners += 1
        elif artifact.owner_is_human is False:
            totals.screened_nonhuman_subjects += 1
        records.extend(artifact.records)

    # -- terminal artifacts ----------------------------------------------------

    def _terminal_artifact_for_owner(
        self,
        shard: str,
        clip_uid: str,
        owner_entity_id: str,
        attribute_id_start: int,
        outcome: Mapping[str, Any],
    ) -> OwnerEnrichmentArtifact:
        """Rebuild one terminal owner artifact from its durable outcome."""
        storage = self._storage_for(shard)
        path = _owner_artifact_path(
            self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
        )
        expected_sha = str(outcome["artifact_sha256"])
        if not path.is_file():
            raise SubjectAttributeDurableError(
                f"Subject Attributes owner artifact is missing for "
                f"{clip_uid}/{owner_entity_id}"
            )
        if _file_sha256(path) != expected_sha:
            raise SubjectAttributeDurableError(
                f"Subject Attributes owner artifact drifted for "
                f"{clip_uid}/{owner_entity_id}"
            )
        try:
            return OwnerEnrichmentArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise SubjectAttributeDurableError(
                f"Subject Attributes owner artifact is invalid for "
                f"{clip_uid}/{owner_entity_id}"
            ) from exc

    def _expected_discovery_terminal_artifact(
        self,
        *,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> OwnerEnrichmentArtifact | None:
        """The exact legacy terminal artifact of one discovery receipt.

        ``None`` means a confirmed human owner: 8a writes no artifact for it and
        leaves the owner unresolved for 8b.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        status = str(payload.get("status"))
        elapsed = float(payload.get("qwen_model_call_time_seconds", 0.0))
        metrics = OwnerEnrichmentMetrics(discovery_calls=1)
        common: dict[str, Any] = {
            "sample_id": clip_uid,
            "owner_entity_id": owner_entity_id,
            "attribute_id_start": int(owner_plan["attribute_id_start"]),
            "owner_phrase": str(owner_plan["owner_phrase"]),
            "owner_grounding_prompt": str(owner_plan["owner_grounding_prompt"]),
            "records": [],
            "gme_screen_mode": None,
            "completion_mode": owner_plan.get("completion_mode"),
        }
        if status == "discovery_failed":
            return OwnerEnrichmentArtifact(
                owner_is_human=None,
                metrics=metrics.model_copy(
                    update={
                        "qwen_model_call_time_seconds": elapsed,
                        "failures": 1,
                    }
                ),
                failure_reason=(
                    f"{DISCOVERY_FAILED_PREFIX}"
                    f"{payload.get('exception_type')}:{payload.get('error')}"
                ),
                **common,
            )
        if status != "discovery":
            raise SubjectAttributeDurableError(
                "committed subject attribute discovery has an unknown payload status"
            )
        try:
            discovery = SubjectAttributeDiscovery.model_validate(
                payload.get("discovery")
            )
        except Exception as exc:
            raise SubjectAttributeDurableError(
                "committed subject attribute discovery payload is invalid"
            ) from exc
        if discovery.owner_entity_id != owner_entity_id:
            raise SubjectAttributeDurableError(
                "committed subject attribute discovery returned the wrong owner"
            )
        if discovery.owner_is_human:
            if discovery.attributes:
                # 8b owns the attribute chain; the human owner artifact itself is
                # published by 8c once the raw routes are resolved.
                return None
            return OwnerEnrichmentArtifact(
                owner_is_human=True,
                metrics=metrics.model_copy(
                    update={"qwen_model_call_time_seconds": elapsed}
                ),
                **common,
            )
        return OwnerEnrichmentArtifact(
            owner_is_human=False,
            metrics=metrics.model_copy(
                update={"qwen_model_call_time_seconds": elapsed}
            ),
            **common,
        )

    def _publish_owner_artifact(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        artifact: OwnerEnrichmentArtifact,
    ) -> str:
        """Publish the legacy artifact, or reuse/overwrite by legacy cache rules."""
        storage = self._storage_for(shard)
        owner_entity_id = str(owner_plan["owner_entity_id"])
        path = _owner_artifact_path(
            self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
        )
        expected_bytes = _expected_artifact_bytes(artifact)
        artifact_sha256 = _sha256_bytes(expected_bytes)
        if not path.is_file():
            write_json_atomic(path, artifact.model_dump(mode="json"))
            return artifact_sha256
        if path.read_bytes() == expected_bytes:
            # A receipt-before-marker crash already published exactly this.
            return artifact_sha256
        if not owner_plan["preexisting_owner_artifact"].get("valid"):
            # Legacy semantics for an invalid cache: it is a miss and may be
            # replaced atomically.
            write_json_atomic(path, artifact.model_dump(mode="json"))
            return artifact_sha256
        raise SubjectAttributeDurableError(
            f"Subject Attributes owner artifact drifted for "
            f"{clip_uid}/{owner_entity_id}"
        )

    # -- advance ---------------------------------------------------------------

    def _advance_clip(
        self, shard: str, storage: RunStorage, clip_uid: str
    ) -> list[ModelJob]:
        plan = self._clip_plan(shard, clip_uid)
        return self._advance_clip_with_plan(shard, storage, clip_uid, plan)

    def _advance_clip_with_plan(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
    ) -> list[ModelJob]:
        """Advance one clip from an already derived and applied clip plan.

        The owner walk below is deliberately serial and canonical per clip: an
        owner's ``attribute_id_start`` depends on every earlier terminal owner's
        ``record_count``, and the walk stops at the first unresolved owner. Only
        the clip-plan derivation above it fans out across clips.
        """
        if plan["classification"] == CLIP_NO_WORK:
            self._publish_clip_outcome(shard, storage, clip_uid, plan)
            return []
        owners = self._eligible_owners(plan)
        for index, owner in enumerate(owners):
            owner_entity_id = str(owner["owner_entity_id"])
            outcome = self._owner_outcome(shard, clip_uid, owner_entity_id)
            owner_plan = self._owner_plan(
                shard,
                clip_uid,
                owner,
                index,
                self._attribute_id_start(shard, clip_uid, owners, index),
            )
            if outcome is not None:
                self._verify_owner_outcome(
                    shard, storage, clip_uid, owner_entity_id, owner_plan, outcome
                )
                continue
            if owner_plan["preexisting_owner_artifact"].get("valid"):
                # Zero-model reuse of an already valid legacy artifact.
                self._verify_preexisting_artifact(shard, clip_uid, owner_plan)
                artifact = self._load_preexisting(
                    shard, clip_uid, owner_entity_id, owner_plan
                )
                self._write_owner_outcome(
                    shard,
                    clip_uid,
                    owner_entity_id,
                    self._expected_owner_outcome(
                        clip_uid=clip_uid,
                        owner_entity_id=owner_entity_id,
                        owner_plan=owner_plan,
                        artifact=artifact,
                        source=OWNER_SOURCE_PREEXISTING,
                        discovery_job_id=None,
                    ),
                )
                continue
            # The current active owner: freeze it and stop scanning this clip.
            discovery_job = self._expected_discovery_job(
                shard, clip_uid, owner_plan
            )
            discovery_payload = self._committed_payload_or_none(discovery_job)
            if discovery_payload is not None:
                # A committed discovery failure, a non-human owner and a human
                # owner without attributes are all complete legacy terminals and
                # only ever come from the terminal artifact derivation. A failed
                # payload has no ``discovery`` key at all, so it must never be
                # validated as one.
                terminal = self._expected_discovery_terminal_artifact(
                    clip_uid=clip_uid,
                    owner_plan=owner_plan,
                    payload=discovery_payload,
                )
                if terminal is None:
                    return self._advance_attributes(
                        shard, storage, clip_uid, owner_entity_id
                    )
                self._publish_owner_artifact(
                    shard, clip_uid, owner_plan, terminal
                )
                self._write_owner_outcome(
                    shard,
                    clip_uid,
                    owner_entity_id,
                    self._expected_owner_outcome(
                        clip_uid=clip_uid,
                        owner_entity_id=owner_entity_id,
                        owner_plan=owner_plan,
                        artifact=terminal,
                        source=OWNER_SOURCE_DISCOVERY,
                        discovery_job_id=discovery_job.job_id(),
                    ),
                )
                continue
            return [discovery_job]
        self._publish_clip_outcome(shard, storage, clip_uid, plan)
        return []

    def _load_preexisting(
        self,
        shard: str,
        clip_uid: str,
        owner_entity_id: str,
        owner_plan: Mapping[str, Any],
    ) -> OwnerEnrichmentArtifact:
        storage = self._storage_for(shard)
        artifact, _sha = self._preexisting_owner_artifact(
            storage,
            clip_uid,
            owner_entity_id,
            int(owner_plan["attribute_id_start"]),
        )
        if artifact is None:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes owner artifact disappeared for "
                f"{clip_uid}/{owner_entity_id}"
            )
        return artifact

    def _seed_targets(self) -> list[tuple[str, str]]:
        """Canonical seed order: sorted shard, then the declared clip order."""
        return [
            (shard, clip_uid)
            for shard in sorted(self.storages)
            for clip_uid in self.eligible.get(shard, ())
        ]

    @staticmethod
    def _seed_batches(
        targets: Sequence[tuple[str, str]], budget: int
    ) -> Iterator[list[tuple[str, str]]]:
        """Consecutive canonical batches of at most ``budget`` targets."""
        size = max(1, int(budget))
        for start in range(0, len(targets), size):
            yield list(targets[start : start + size])

    def _timed_derive_clip_plan(
        self, storage: RunStorage, clip_uid: str
    ) -> dict[str, Any]:
        """One fan-out derivation task, with its own execution-only duration."""
        started = time.perf_counter()
        try:
            return self._checked_derive_clip_plan(storage, clip_uid)
        finally:
            self._note_seed_wall(time.perf_counter() - started)

    def seed_jobs(self) -> list[ModelJob]:
        """CPU fixed point, then the deterministic pending discovery jobs.

        Deriving a clip plan is pure read-only CPU over live upstream artifacts,
        so it fans out across clips on one executor per invocation, in bounded
        canonical batches. Everything that writes - the durable plan, the owner
        advancement, the clip outcome - stays on this thread, in canonical clip
        order, exactly as the serial path did: a batch is applied and advanced
        clip by clip as its canonical derivation becomes available, so the first
        failure is raised at the position the serial path would raise it, with
        everything before it already published and nothing after it published at
        all. ``cpu_workers <= 1`` therefore takes that serial path and builds no
        executor at all.
        """
        targets = self._seed_targets()
        jobs: list[ModelJob] = []
        if int(self.cpu_workers) <= 1 or len(targets) <= 1:
            for shard, clip_uid in targets:
                jobs.extend(
                    self._advance_clip(shard, self._storage_for(shard), clip_uid)
                )
            return sorted(jobs, key=lambda job: job.job_id())
        with ThreadPoolExecutor(max_workers=int(self.cpu_workers)) as pool:
            for batch in self._seed_batches(targets, self._clip_plan_derive_budget()):
                futures: dict[tuple[str, str], Any] = {}
                for shard, clip_uid in batch:
                    target = (shard, clip_uid)
                    # A clip whose plan this invocation already derived is not
                    # re-derived: the serial path short-circuits on the same
                    # cache, and deriving again would add a drift check the
                    # serial path never performs.
                    if self._clip_plan_cache.get(target) is None:
                        futures[target] = pool.submit(
                            self._timed_derive_clip_plan,
                            self._storage_for(shard),
                            clip_uid,
                        )
                if futures:
                    self._bump_seed_counter("clip_plan_derive_batches")
                    self._note_seed_inflight(len(futures))
                for shard, clip_uid in batch:
                    plan = self._clip_plan_cache.get((shard, clip_uid))
                    if plan is None:
                        expected = futures[(shard, clip_uid)].result()
                        self._bump_seed_counter("clip_plan_derive_tasks")
                        plan = self._apply_derived_clip_plan(
                            shard, clip_uid, expected
                        )
                    jobs.extend(
                        self._advance_clip_with_plan(
                            shard, self._storage_for(shard), clip_uid, plan
                        )
                    )
        return sorted(jobs, key=lambda job: job.job_id())

    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """CPU continuation of one committed model receipt."""
        if job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            return self._finalize_sam_probe(job, result)
        if job.job_type in {
            SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
            SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB,
            SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB,
            SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_JOB,
            SUBJECT_ATTRIBUTE_BBOX_REVIEW_JOB,
        }:
            return self._finalize_owner_chain(job, result)
        if job.job_type != SUBJECT_ATTRIBUTE_DISCOVERY_JOB:
            raise SubjectAttributeEpochError(
                f"unsupported job type {job.job_type!r}"
            )
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        if status not in {"discovery", "discovery_failed"}:
            raise SubjectAttributeDurableError(
                f"committed discovery job {job.job_id()} has an unknown payload "
                "status"
            )
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip_uid = job.clip_uid
        owner_entity_id = str(dict(job.target).get("owner_entity_id", ""))
        plan = self._clip_plan(shard, clip_uid)
        owner_plan = self._owner_plan_for(shard, plan, owner_entity_id)
        expected = self._expected_discovery_job(shard, clip_uid, owner_plan)
        if expected.job_id() != job.job_id():
            raise SubjectAttributeDurableError(
                f"discovery job {job.job_id()} is not the frozen job for "
                f"{clip_uid}/{owner_entity_id}"
            )
        artifact = self._expected_discovery_terminal_artifact(
            clip_uid=clip_uid, owner_plan=owner_plan, payload=payload
        )
        if artifact is None:
            # A confirmed human owner with attributes keeps its discovery
            # receipt and continues into the SAM / raw review chain.
            return self._advance_attributes(
                shard, storage, clip_uid, owner_entity_id
            )
        self._publish_owner_artifact(shard, clip_uid, owner_plan, artifact)
        self._write_owner_outcome(
            shard,
            clip_uid,
            owner_entity_id,
            self._expected_owner_outcome(
                clip_uid=clip_uid,
                owner_entity_id=owner_entity_id,
                owner_plan=owner_plan,
                artifact=artifact,
                source=OWNER_SOURCE_DISCOVERY,
                discovery_job_id=job.job_id(),
            ),
        )
        return self._advance_clip(shard, storage, clip_uid)

    # -- owner outcome ---------------------------------------------------------

    def _expected_owner_outcome(
        self,
        *,
        clip_uid: str,
        owner_entity_id: str,
        owner_plan: Mapping[str, Any],
        artifact: OwnerEnrichmentArtifact,
        source: str,
        discovery_job_id: str | None,
    ) -> dict[str, Any]:
        """The exact owner outcome marker one terminal owner must produce."""
        return {
            "schema": SUBJECT_ATTRIBUTE_OWNER_OUTCOME_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": owner_entity_id,
            "source": source,
            "attribute_id_start": int(owner_plan["attribute_id_start"]),
            "artifact_sha256": _sha256_bytes(_expected_artifact_bytes(artifact)),
            "record_count": len(artifact.records),
            "discovery_job_id": discovery_job_id,
        }

    def _write_owner_outcome(
        self, shard: str, clip_uid: str, owner_entity_id: str, payload: Mapping[str, Any]
    ) -> None:
        _write_json_once(
            self._owner_outcome_path(shard, clip_uid, owner_entity_id),
            dict(payload),
        )

    def _owner_outcome(
        self, shard: str, clip_uid: str, owner_entity_id: str
    ) -> dict[str, Any] | None:
        """Strictly validated owner outcome marker, or ``None`` if unresolved."""
        path = self._owner_outcome_path(shard, clip_uid, owner_entity_id)
        marker = _read_json(path)
        if marker is None:
            return None
        label = f"{clip_uid}/{owner_entity_id}"
        if marker.get("schema") != SUBJECT_ATTRIBUTE_OWNER_OUTCOME_SCHEMA:
            raise SubjectAttributeDurableError(
                f"owner outcome schema drifted for {label}"
            )
        if marker.get("clip_uid") != clip_uid:
            raise SubjectAttributeDurableError(
                f"owner outcome clip_uid drifted for {label}"
            )
        if marker.get("owner_entity_id") != owner_entity_id:
            raise SubjectAttributeDurableError(
                f"owner outcome owner drifted for {label}"
            )
        if marker.get("source") not in (
            OWNER_SOURCE_PREEXISTING,
            OWNER_SOURCE_DISCOVERY,
            OWNER_SOURCE_PROCESSED,
        ):
            raise SubjectAttributeDurableError(
                f"owner outcome source is unknown for {label}"
            )
        start = marker.get("attribute_id_start")
        if isinstance(start, bool) or not isinstance(start, int) or start < 1:
            raise SubjectAttributeDurableError(
                f"owner outcome attribute_id_start is invalid for {label}"
            )
        sha = marker.get("artifact_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise SubjectAttributeDurableError(
                f"owner outcome artifact sha is invalid for {label}"
            )
        count = marker.get("record_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SubjectAttributeDurableError(
                f"owner outcome record count is invalid for {label}"
            )
        job_id = marker.get("discovery_job_id")
        if marker["source"] == OWNER_SOURCE_PREEXISTING:
            if job_id is not None:
                raise SubjectAttributeDurableError(
                    f"preexisting owner outcome must not carry a discovery job "
                    f"for {label}"
                )
        elif not isinstance(job_id, str) or not job_id:
            raise SubjectAttributeDurableError(
                f"discovery owner outcome has no job id for {label}"
            )
        return marker

    def _verify_owner_outcome(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_entity_id: str,
        owner_plan: Mapping[str, Any],
        marker: Mapping[str, Any],
    ) -> None:
        """Re-derive one owner artifact from frozen state and require the marker.

        A discovery outcome additionally requires the exact committed receipt:
        the marker is never its own authority.
        """
        label = f"{clip_uid}/{owner_entity_id}"
        if marker["source"] == OWNER_SOURCE_PREEXISTING:
            self._verify_preexisting_artifact(shard, clip_uid, owner_plan)
            artifact = self._load_preexisting(
                shard, clip_uid, owner_entity_id, owner_plan
            )
            expected = self._expected_owner_outcome(
                clip_uid=clip_uid,
                owner_entity_id=owner_entity_id,
                owner_plan=owner_plan,
                artifact=artifact,
                source=OWNER_SOURCE_PREEXISTING,
                discovery_job_id=None,
            )
        elif marker["source"] == OWNER_SOURCE_PROCESSED:
            # The processed owner is authority-free too: the whole legacy owner
            # algorithm is re-run over the committed receipts and must land on
            # exactly the artifact and marker that were published.
            discovery_job = self._expected_discovery_job(shard, clip_uid, owner_plan)
            discovery_job_id = discovery_job.job_id()
            if marker["discovery_job_id"] != discovery_job_id:
                raise SubjectAttributeDurableError(
                    f"owner outcome discovery job drifted for {label}"
                )
            context = self._rank0_context(
                shard,
                storage,
                clip_uid,
                owner_plan,
                SubjectAttributeDiscovery.model_validate(
                    self._discovery_payload(shard, clip_uid, owner_plan).get(
                        "discovery"
                    )
                ),
                discovery_job_id,
            )
            if context["pending"]:
                raise SubjectAttributeDurableError(
                    f"owner outcome exists while owner {label} is still pending"
                )
            self._verified_raw_state(shard, storage, clip_uid, owner_plan, context)
            self._verify_completion_outcomes(
                shard,
                storage,
                clip_uid,
                owner_plan,
                self._owner_graph(shard, storage, clip_uid, owner_plan, context),
            )
            artifact = self._owner_artifact_from_receipts(
                shard, storage, clip_uid, owner_plan, context["states"]
            )
            if isinstance(artifact, _PendingModelCall):
                raise SubjectAttributeDurableError(
                    f"owner outcome exists while owner {label} still owes a model call"
                )
            expected = self._expected_owner_outcome(
                clip_uid=clip_uid,
                owner_entity_id=owner_entity_id,
                owner_plan=owner_plan,
                artifact=artifact,
                source=OWNER_SOURCE_PROCESSED,
                discovery_job_id=discovery_job_id,
            )
        else:
            job = self._expected_discovery_job(shard, clip_uid, owner_plan)
            if marker["discovery_job_id"] != job.job_id():
                raise SubjectAttributeDurableError(
                    f"owner outcome discovery job drifted for {label}"
                )
            payload = _committed_payload(self.ledger, job)
            artifact = self._expected_discovery_terminal_artifact(
                clip_uid=clip_uid, owner_plan=owner_plan, payload=payload
            )
            if artifact is None:
                raise SubjectAttributeDurableError(
                    f"owner outcome exists for a deferred human owner {label}"
                )
            expected = self._expected_owner_outcome(
                clip_uid=clip_uid,
                owner_entity_id=owner_entity_id,
                owner_plan=owner_plan,
                artifact=artifact,
                source=OWNER_SOURCE_DISCOVERY,
                discovery_job_id=job.job_id(),
            )
        path = _owner_artifact_path(
            self._output_root(storage),
            sample_id=clip_uid,
            owner_entity_id=owner_entity_id,
        )
        if not path.is_file() or _file_sha256(path) != expected["artifact_sha256"]:
            raise SubjectAttributeDurableError(
                f"owner artifact drifted for {label}"
            )
        if dict(marker) != expected:
            raise SubjectAttributeDurableError(
                f"owner outcome drifted for {label}"
            )

    # -- clip outcome ----------------------------------------------------------

    def _clip_totals(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
    ) -> tuple[EnrichmentTotals, list[Any], bool]:
        """Recompute a clip's legacy totals from its durable terminal owners."""
        totals = EnrichmentTotals()
        records: list[Any] = []
        enriched = False
        for index, owner in enumerate(self._eligible_owners(plan)):
            owner_entity_id = str(owner["owner_entity_id"])
            outcome = self._owner_outcome(shard, clip_uid, owner_entity_id)
            if outcome is None:
                raise SubjectAttributeEpochError(
                    f"Subject Attributes clip {clip_uid!r} has no terminal owner "
                    f"outcome for {owner_entity_id!r}; the stage is not terminal"
                )
            owner_plan = self._owner_plan(
                shard,
                clip_uid,
                owner,
                index,
                int(outcome["attribute_id_start"]),
            )
            self._verify_owner_outcome(
                shard, storage, clip_uid, owner_entity_id, owner_plan, outcome
            )
            artifact = self._terminal_artifact_for_owner(
                shard,
                clip_uid,
                owner_entity_id,
                int(outcome["attribute_id_start"]),
                outcome,
            )
            self._accumulate_owner(
                totals,
                artifact,
                preexisting=outcome["source"] == OWNER_SOURCE_PREEXISTING,
                records=records,
            )
            enriched = enriched or artifact.owner_is_human is True
        return totals, records, enriched

    def _clip_outcome_payload(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
    ) -> tuple[dict[str, Any], OwnerEnrichmentArtifact | None]:
        """The expected clip outcome marker plus the enriched sample, if any."""
        if plan["classification"] == CLIP_NO_WORK:
            totals = EnrichmentTotals()
            records: list[Any] = []
            enriched = False
        else:
            totals, records, enriched = self._clip_totals(
                shard, storage, clip_uid, plan
            )
        sample = None
        sample_path: str | None = None
        sample_sha256: str | None = None
        if enriched:
            clip = self._effective_clip(storage.read_clip(clip_uid))
            sample = _build_enriched_sample(
                storage=storage,
                clip=clip,
                records=records,
            )
            sample_path = _clip_sample_path(
                self._output_root(storage), sample_id=clip_uid
            ).relative_to(storage.root).as_posix()
            sample_sha256 = _sha256_bytes(
                _expected_artifact_bytes(sample)
            )
        result = ClipEnrichmentResult(
            clip_uid=clip_uid,
            totals=totals,
            owner_limit_reached=False,
            enriched_sample=sample,
        )
        return (
            {
                "schema": SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA,
                "clip_uid": clip_uid,
                "terminal": "ready",
                "counts": result.to_counts(),
                "owner_outcome_ids": [
                    str(owner["owner_entity_id"])
                    for owner in self._eligible_owners(plan)
                ],
                "enriched_sample_path": sample_path,
                "enriched_sample_sha256": sample_sha256,
            },
            sample,
        )

    def _expected_clip_outcome(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._clip_outcome_payload(shard, storage, clip_uid, plan)[0]

    def _publish_clip_outcome(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
    ) -> None:
        path = self._clip_outcome_path(shard, clip_uid)
        if path.is_file():
            return
        expected, sample = self._clip_outcome_payload(shard, storage, clip_uid, plan)
        if expected["enriched_sample_path"] is not None and sample is not None:
            target = storage.root / str(expected["enriched_sample_path"])
            expected_bytes = _expected_artifact_bytes(sample)
            if not target.is_file() or target.read_bytes() != expected_bytes:
                write_json_atomic(target, sample.model_dump(mode="json"))
        _write_json_once(path, expected)

    # -- attribute plans -------------------------------------------------------

    def _attribute_dir(
        self, shard: str, clip_uid: str, owner: str, attribute_id: str
    ) -> Path:
        return self._owner_dir(shard, clip_uid, owner) / "attributes" / attribute_id

    def _attribute_plan_path(
        self, shard: str, clip_uid: str, owner: str, attribute_id: str
    ) -> Path:
        return self._attribute_dir(shard, clip_uid, owner, attribute_id) / "plan.json"

    def _selection_path(
        self, shard: str, clip_uid: str, owner: str, attribute_id: str
    ) -> Path:
        return self._attribute_dir(shard, clip_uid, owner, attribute_id) / "selection.json"

    def _raw_state_path(self, shard: str, clip_uid: str, owner: str) -> Path:
        return self._owner_dir(shard, clip_uid, owner) / "raw_state.json"

    def _sam_mask_path(self, job_id: str, index: int) -> Path:
        return self._semantic("binary", "sam", job_id, f"mask-{index:03d}.npy")

    def _selection_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the legacy candidate-selection CPU policy."""
        completion = self.config.subject_attributes.completion
        return {
            "policy_version": SUBJECT_ATTRIBUTE_SELECTION_POLICY_VERSION,
            "max_attribute_source_candidates": int(
                MAX_ATTRIBUTE_SOURCE_CANDIDATES
            ),
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
            "completion_enabled": bool(completion.enabled),
            "completion_minimum_mean_luminance": float(
                completion.minimum_mean_luminance
            ),
            "completion_minimum_sharpness_score": float(
                completion.minimum_sharpness_score
            ),
            "completion_maximum_significant_components": int(
                completion.maximum_significant_components
            ),
            "gme_enabled": False,
            "mode": "legacy_collect_attribute_candidates_v1",
        }

    def _raw_review_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the owner-batched raw review model behavior."""
        service = self.config.qwen.candidate_judge
        return {
            "policy_version": SUBJECT_ATTRIBUTE_RAW_REVIEW_POLICY_VERSION,
            "model": str(service.model),
            "temperature": float(service.temperature),
            "max_tokens": int(service.max_tokens),
            "system_prompt_sha256": _sha256_bytes(
                REVIEW_SYSTEM_PROMPT.encode("utf-8")
            ),
            "review_schema_sha256": _sha256_bytes(
                canonical_json(
                    SubjectAttributeReviewBatch.model_json_schema()
                ).encode("utf-8")
            ),
            "qwen_input_max_long_side_pixels": int(QWEN_INPUT_MAX_LONG_SIDE_PIXELS),
            "mode": "owner_batched_raw_review_rank0_v1",
        }

    def _discovery_payload(self, shard: str, clip_uid: str, owner_plan: Mapping[str, Any]) -> dict[str, Any]:
        job = self._expected_discovery_job(shard, clip_uid, owner_plan)
        payload = self._committed_payload_or_none(job)
        if payload is None:
            raise SubjectAttributeEpochError(
                f"owner {clip_uid}/{owner_plan['owner_entity_id']} has no committed "
                "discovery receipt"
            )
        return payload

    def _committed_payload_or_none(self, job: ModelJob) -> dict[str, Any] | None:
        """The committed receipt payload of one job, or ``None`` if it is pending."""
        from r2v_data_v2.v3.post_mask_epoch_state import STATE_MISMATCH

        state = self.ledger.classify(job)
        if state.state == STATE_MISMATCH:
            raise SubjectAttributeDurableError(
                f"receipt mismatch for {job.clip_uid}: {state.detail or ''}"
            )
        if not state.skippable:
            return None
        result = self.ledger.load_committed_result(job)
        if result is None:
            raise SubjectAttributeDurableError(
                f"committed job {job.job_id()} has no replayable result"
            )
        return dict(result.payload)

    def _owner_reference(self, storage: RunStorage, clip_uid: str, owner_entity_id: str) -> Any:
        """The published ready reference of one owner, exactly like legacy."""
        clip = storage.read_clip(clip_uid)
        reference = next(
            (
                item
                for item in clip.references.entities
                if item.entity_id == owner_entity_id and item.status == "ready"
            ),
            None,
        )
        if reference is None:
            raise SubjectAttributeDurableError(
                f"owner {clip_uid}/{owner_entity_id} has no ready published reference"
            )
        return reference

    def _ordered_owner_candidates(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        *,
        candidates: Sequence[Any] | None = None,
    ) -> list[Any]:
        """The owner's candidates in the exact legacy attribute-probe order.

        ``candidates`` lets the caller pass a set it has already built on the
        main thread, so the heavy owner-candidate construction happens once per
        owner instead of once per attribute worker.
        """
        if candidates is None:
            candidates = self._owner_candidate_objects(shard, clip_uid, owner_plan)
        is_local = owner_reference.source_clip_uid in {None, clip_uid}
        ordered = prefer_attribute_candidate_frames(
            candidates,
            owner_reference_source_frame_index=(
                owner_reference.source_frame_index if is_local else -1
            ),
        )
        slots = [candidate.frame_slot for candidate in ordered]
        if len(slots) != len(set(slots)):
            raise SubjectAttributeDurableError(
                f"owner {clip_uid}/{owner_plan['owner_entity_id']} has ambiguous "
                "candidate frame slots"
            )
        return ordered

    def _expected_attribute_plan(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
        *,
        ordered: Sequence[Any],
    ) -> dict[str, Any]:
        """Derive the attribute plan bytes. Pure: no read, no write."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        discovered = discovery.attributes[attribute_index]
        expected = {
            "schema": SUBJECT_ATTRIBUTE_ATTRIBUTE_PLAN_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": owner_entity_id,
            "attribute_id": attribute_id,
            "attribute_index": int(attribute_index),
            "discovery_job_id": discovery_job_id,
            "discovered": {
                "attribute_type": str(discovered.attribute_type),
                "phrase": str(discovered.phrase),
                "grounding_prompt": str(discovered.grounding_prompt),
            },
            "owner_reference": {
                "source_clip_uid": owner_reference.source_clip_uid,
                "source_entity_id": owner_reference.source_entity_id,
                "source_frame_index": int(owner_reference.source_frame_index),
                "image_path": str(owner_reference.image_path),
            },
            "ordered_owner_candidate_ids": [
                str(candidate.candidate_id) for candidate in ordered
            ],
            "selection_policy": self._selection_policy_identity(),
        }
        return expected

    def _attribute_plan(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
        *,
        ordered: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        """Create-once attribute plan, re-derived on every read.

        Main thread only. Creating a durable plan is a publication, so it must
        never happen from a worker: ``_rank0_context`` creates and verifies every
        attribute's plan, in annotation order, before any replay is submitted.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        path = self._attribute_plan_path(shard, clip_uid, owner_entity_id, attribute_id)
        if ordered is None:
            ordered = self._ordered_owner_candidates(
                shard, storage, clip_uid, owner_plan, owner_reference
            )
        expected = self._expected_attribute_plan(
            shard,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_index,
            attribute_id,
            discovery_job_id,
            ordered=ordered,
        )
        existing = _read_json(path)
        if existing is None:
            _write_json_once(path, expected)
            return expected
        if existing != expected:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes attribute plan drifted for "
                f"{clip_uid}/{owner_entity_id}/{attribute_id}"
            )
        return existing

    def _require_attribute_plan(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
        *,
        ordered: Sequence[Any],
    ) -> dict[str, Any]:
        """Read-only attribute plan for the model-execution path.

        A ModelJob can only exist if the plan it was built from is already
        durable, so this never creates one. A missing or drifted plan is durable
        corruption and fails closed instead of being silently recreated under a
        job whose identity was frozen against the old plan.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        path = self._attribute_plan_path(shard, clip_uid, owner_entity_id, attribute_id)
        existing = _read_json(path)
        if existing is None:
            raise SubjectAttributeDurableError(
                f"attribute plan is missing for "
                f"{clip_uid}/{owner_entity_id}/{attribute_id}"
            )
        expected = self._expected_attribute_plan(
            shard,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_index,
            attribute_id,
            discovery_job_id,
            ordered=ordered,
        )
        if existing != expected:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes attribute plan drifted for "
                f"{clip_uid}/{owner_entity_id}/{attribute_id}"
            )
        return existing

    # -- SAM probe jobs --------------------------------------------------------

    def _expected_sam_probe_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        ordered: Sequence[Any],
        probe_index: int,
        previous_job: ModelJob | None,
    ) -> ModelJob:
        candidate = ordered[probe_index]
        descriptor = next(
            (
                item
                for item in owner_plan["candidates"]
                if item["candidate_id"] == candidate.candidate_id
            ),
            None,
        )
        if descriptor is None:
            raise SubjectAttributeDurableError(
                f"SAM probe candidate {candidate.candidate_id} is not frozen"
            )
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_SAM_PROBE_JOB,
            resource=RESOURCE_SAM,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "clip_uid": clip_uid,
                "owner_entity_id": str(owner_plan["owner_entity_id"]),
                "attribute_id": str(attribute_plan["attribute_id"]),
                "attribute_type": attribute_plan["discovered"]["attribute_type"],
                "grounding_prompt": attribute_plan["discovered"]["grounding_prompt"],
                "owner_candidate_id": str(candidate.candidate_id),
                "frame_slot": int(candidate.frame_slot),
                "source_frame_index": int(candidate.source_frame_index),
                "frame_path": str(candidate.image_path),
                "source_image_sha256": descriptor["source_image_sha256"],
                "selection_policy": dict(attribute_plan["selection_policy"]),
            },
            model_identity="sam3",
            target={
                "owner_entity_id": str(owner_plan["owner_entity_id"]),
                "attribute_id": str(attribute_plan["attribute_id"]),
                "owner_candidate_id": str(candidate.candidate_id),
                "probe_index": str(probe_index),
            },
            attempt_index=probe_index,
            dependencies=(
                {attribute_plan["attribute_id"]: previous_job.job_id()}
                if previous_job is not None
                else {SUBJECT_ATTRIBUTE_DISCOVERY_JOB: str(attribute_plan["discovery_job_id"])}
            ),
        )

    def _sam_probe_chain(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        ordered: Sequence[Any],
    ) -> list[ModelJob]:
        """The deterministic probe chain: probe i depends on probe i-1."""
        chain: list[ModelJob] = []
        previous: ModelJob | None = None
        for index in range(len(ordered)):
            previous = self._expected_sam_probe_job(
                shard, clip_uid, owner_plan, attribute_plan, ordered, index, previous
            )
            chain.append(previous)
        return chain

    def _load_sam_masks(self, payload: Mapping[str, Any]) -> list[Any]:
        """Read the ledger ``.npy`` mask artifacts of one committed probe."""
        masks: list[Any] = []
        for record in payload.get("masks", []):
            path = Path(self.ledger.root) / str(record["path"])
            if not path.is_file():
                raise SubjectAttributeDurableError(
                    f"SAM probe mask artifact is missing: {record['path']}"
                )
            data = path.read_bytes()
            if _sha256_bytes(data) != str(record["sha256"]):
                raise SubjectAttributeDurableError(
                    f"SAM probe mask artifact drifted: {record['path']}"
                )
            try:
                masks.append(np.load(io.BytesIO(data), allow_pickle=False))
            except (OSError, ValueError) as exc:
                raise SubjectAttributeDurableError(
                    f"SAM probe mask artifact is unreadable: {record['path']}"
                ) from exc
        if len(masks) != int(payload.get("mask_count", -1)):
            raise SubjectAttributeDurableError(
                "SAM probe mask count drifted from its receipt"
            )
        return masks

    def _run_sam_probe(self, job: ModelJob, handle: Any) -> JobResult:
        """One real legacy ``segment_frame`` call. Nothing else is caught."""
        if job.job_type != SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            raise SubjectAttributeEpochError(f"unsupported job type {job.job_type!r}")
        context = self._sam_probe_context(job)
        probe_index = context["probe_index"]
        candidate = context["ordered"][probe_index]
        if handle is None:
            raise SubjectAttributeEpochError("subject attribute SAM probe needs a handle")
        frame_path = (context["storage"].root / candidate.image_path).resolve(
            strict=False
        )
        started = time.perf_counter()
        try:
            returned = handle.segment_frame(
                frame_path=frame_path,
                frame_slot=int(candidate.frame_slot),
                grounding_prompt=str(
                    context["attribute_plan"]["discovered"]["grounding_prompt"]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - legacy per-probe isolation
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "sam_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "model_call_time_seconds": time.perf_counter() - started,
                },
            )
        records: list[dict[str, Any]] = []
        for index, mask in enumerate(returned):
            array = np.asarray(mask)
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            data = buffer.getvalue()
            path = self._sam_mask_path(job.job_id(), index)
            atomic_write_bytes(path, data)
            records.append(
                {
                    "path": path.relative_to(Path(self.ledger.root)).as_posix(),
                    "sha256": _sha256_bytes(data),
                    "dtype": str(array.dtype),
                    "shape": [int(value) for value in array.shape],
                }
            )
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "sam",
                "mask_count": len(records),
                "masks": records,
                "model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _sam_probe_context(self, job: ModelJob) -> dict[str, Any]:
        """Resolve and re-verify one SAM probe job against frozen state."""
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip_uid = job.clip_uid
        target = dict(job.target)
        owner_entity_id = str(target.get("owner_entity_id", ""))
        attribute_id = str(target.get("attribute_id", ""))
        owner_plan = self._owner_plan_for(shard, self._clip_plan(shard, clip_uid), owner_entity_id)
        discovery_payload = self._discovery_payload(shard, clip_uid, owner_plan)
        discovery = SubjectAttributeDiscovery.model_validate(
            discovery_payload.get("discovery")
        )
        owner_reference = self._owner_reference(storage, clip_uid, owner_entity_id)
        ordered = self._ordered_owner_candidates(
            shard, storage, clip_uid, owner_plan, owner_reference
        )
        # This runs in the model-execution worker, so the plan must already be
        # durable: a probe job that exists was frozen against it.
        attribute_plan = self._require_attribute_plan(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            self._attribute_index(
                discovery,
                attribute_id,
                int(owner_plan["attribute_id_start"]),
            ),
            attribute_id,
            self._expected_discovery_job(shard, clip_uid, owner_plan).job_id(),
            ordered=ordered,
        )
        probe_index = int(target.get("probe_index", -1))
        if probe_index < 0 or probe_index >= len(ordered):
            raise SubjectAttributeDurableError(
                f"SAM probe index {probe_index} is out of range for {clip_uid}"
            )
        expected = self._sam_probe_chain(
            shard, clip_uid, owner_plan, attribute_plan, ordered
        )[probe_index]
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != expected.input_digest
            or job.model_identity != expected.model_identity
        ):
            raise SubjectAttributeDurableError(
                f"SAM probe job {job.job_id()} is not the frozen probe for "
                f"{clip_uid}/{owner_entity_id}/{attribute_id}"
            )
        return {
            "storage": storage,
            "owner_plan": owner_plan,
            "attribute_plan": attribute_plan,
            "ordered": ordered,
            "probe_index": probe_index,
        }

    @staticmethod
    def _attribute_index(
        discovery: SubjectAttributeDiscovery,
        attribute_id: str,
        attribute_id_start: int,
    ) -> int:
        """Map ``a<N>`` to its offset inside *this owner's* discovery.

        Attribute ids are absolutely numbered from the owner's
        ``attribute_id_start``, so a second human owner starts at ``a3``; the
        offset is only meaningful relative to that owner.
        """
        try:
            index = int(attribute_id[1:]) - int(attribute_id_start)
        except ValueError as exc:
            raise SubjectAttributeDurableError(
                f"attribute id {attribute_id!r} is malformed"
            ) from exc
        if not 0 <= index < len(discovery.attributes):
            raise SubjectAttributeDurableError(
                f"attribute id {attribute_id!r} is outside the frozen discovery "
                f"of its owner"
            )
        return index

    # -- attribute selection replay -------------------------------------------

    def _replay_attribute_selection(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
    ) -> _AttributeReplay:
        """Create-or-verify the attribute plan, then replay. Main thread only."""
        return self._replay_attribute_selection_prepared(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_index,
            attribute_id,
            discovery_job_id,
            attribute_plan=self._attribute_plan(
                shard,
                storage,
                clip_uid,
                owner_plan,
                owner_reference,
                discovery,
                attribute_index,
                attribute_id,
                discovery_job_id,
            ),
            owner_candidates=self._owner_candidate_objects(shard, clip_uid, owner_plan),
            ordered=self._ordered_owner_candidates(
                shard, storage, clip_uid, owner_plan, owner_reference
            ),
        )

    def _replay_attribute_selection_prepared(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
        *,
        attribute_plan: Mapping[str, Any],
        owner_candidates: Sequence[Any],
        ordered: Sequence[Any],
    ) -> _AttributeReplay:
        """Replay the legacy selection CPU policy. Pure: no durable writes.

        Everything this function needs is handed in. It reads receipts, verifies
        them, loads committed SAM masks and runs the mask and image CPU, and
        returns Python objects. It must never write a plan, a selection marker,
        an owner marker, a receipt or a pending-map entry - that is what lets it
        run on a worker thread.

        The only SAM call the replay may not replay is the first probe that has
        no committed receipt: that probe is the next job, and the legacy helper
        is re-run from scratch once its receipt exists.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        chain = self._sam_probe_chain(
            shard, clip_uid, owner_plan, attribute_plan, ordered
        )
        probe_jobs = {index: job for index, job in enumerate(chain)}
        receipts: dict[int, dict[str, Any]] = {}
        for index, job in probe_jobs.items():
            payload = self._committed_payload_or_none(job)
            if payload is not None:
                receipts[index] = payload
        backend = _ReplaySegmentationBackend(
            epoch=self,
            ordered=ordered,
            receipts=receipts,
            storage=storage,
            grounding_prompt=str(attribute_plan["discovered"]["grounding_prompt"]),
            chain=list(probe_jobs.values()),
        )
        clip = storage.read_clip(clip_uid)
        owner = next(
            item
            for item in clip.annotation.entities
            if item.entity_id == owner_entity_id
        )
        discovered = DiscoveredSubjectAttribute.model_validate(
            attribute_plan["discovered"]
        )
        processing_failure = 0
        try:
            selected = _collect_attribute_candidates(
                discovered=discovered,
                attribute_id=attribute_id,
                clip_uid=clip_uid,
                owner=owner,
                owner_candidates=list(owner_candidates),
                owner_reference=owner_reference,
                masks=storage.read_masks(clip_uid),
                storage=storage,
                other_subject_ids=[
                    entity.entity_id
                    for entity in clip.annotation.entities
                    if entity.reference_type == "subject"
                    and entity.entity_id != owner_entity_id
                ],
                crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
                segmentation_backend=backend,
                gme_screener=None,
                completion_config=self.config.subject_attributes.completion,
                max_candidates=MAX_ATTRIBUTE_SOURCE_CANDIDATES,
            )
        except _PendingModelCall as pending:
            # The replay stopped at a probe that has no receipt yet: that probe
            # is the next job and everything after it is not decided.
            return _AttributeReplay(
                attribute_id=attribute_id,
                attribute_type=str(attribute_plan["discovered"]["attribute_type"]),
                attribute_plan=attribute_plan,
                marker=None,
                options=(),
                record=None,
                processing_failure=0,
                pending_job=pending.job,
                sam_job_ids={},
            )
        except Exception as exc:  # noqa: BLE001 - legacy attribute isolation
            processing_failure = 1
            selected = _rejected_record(
                discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner_entity_id,
                reason=f"attribute_processing_failed:{type(exc).__name__}:{exc}",
            )
        marker, options, record = self._selection_marker(
            clip_uid=clip_uid,
            owner_entity_id=owner_entity_id,
            attribute_id=attribute_id,
            selected=selected,
            processing_failure=processing_failure,
            ordered=ordered,
            receipts=receipts,
            probe_jobs=probe_jobs,
            storage=storage,
        )
        return _AttributeReplay(
            attribute_id=attribute_id,
            attribute_type=str(attribute_plan["discovered"]["attribute_type"]),
            attribute_plan=attribute_plan,
            marker=marker,
            options=tuple(options),
            record=record,
            processing_failure=processing_failure,
            pending_job=None,
            sam_job_ids={
                option.owner_candidate.candidate_id: probe_jobs[index].job_id()
                for index, candidate in enumerate(ordered)
                for option in options
                if option.owner_candidate.candidate_id == candidate.candidate_id
            },
        )

    def _selection_marker(
        self,
        *,
        clip_uid: str,
        owner_entity_id: str,
        attribute_id: str,
        selected: Any,
        processing_failure: int,
        ordered: Sequence[Any],
        receipts: Mapping[int, Mapping[str, Any]],
        probe_jobs: Mapping[int, ModelJob],
        storage: RunStorage,
    ) -> tuple[dict[str, Any], list[Any], Any | None]:
        """The exact terminal selection marker of one attribute."""
        base = {
            "schema": SUBJECT_ATTRIBUTE_SELECTION_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": owner_entity_id,
            "attribute_id": attribute_id,
        }
        if isinstance(selected, SubjectAttributeRecord):
            return (
                {
                    **base,
                    "status": SELECTION_REJECTED,
                    "record": selected.model_dump(mode="json"),
                    "processing_failure": int(processing_failure),
                },
                [],
                selected,
            )
        slot_index = {
            int(candidate.frame_slot): index for index, candidate in enumerate(ordered)
        }
        options: list[dict[str, Any]] = []
        for candidate in selected:
            index = slot_index.get(int(candidate.owner_candidate.frame_slot))
            if index is None:
                raise SubjectAttributeDurableError(
                    f"selected attribute candidate {candidate.owner_candidate.candidate_id} "
                    "is not a frozen owner candidate"
                )
            payload = receipts.get(index)
            if payload is None or payload.get("status") != "sam":
                raise SubjectAttributeDurableError(
                    "selected attribute candidate has no committed SAM masks"
                )
            masks = self._load_sam_masks(payload)
            mask_index = next(
                (
                    position
                    for position, mask in enumerate(masks)
                    if np.array_equal(np.asarray(mask), candidate.attribute_mask)
                ),
                None,
            )
            if mask_index is None:
                raise SubjectAttributeDurableError(
                    "selected attribute mask is not part of its SAM receipt"
                )
            options.append(
                {
                    "owner_candidate_id": str(
                        candidate.owner_candidate.candidate_id
                    ),
                    "sam_job_id": probe_jobs[index].job_id(),
                    "sam_mask_index": int(mask_index),
                    "sam_mask_sha256": str(
                        payload["masks"][mask_index]["sha256"]
                    ),
                    "geometry": candidate.geometry.model_dump(mode="json"),
                    "source_frame_slot": int(candidate.owner_candidate.frame_slot),
                    "source_frame_index": int(
                        candidate.owner_candidate.source_frame_index
                    ),
                    "crop_sha256": _image_png_sha256(candidate.crop),
                }
            )
        return (
            {**base, "status": SELECTION_CANDIDATES, "options": options},
            list(selected),
            None,
        )

    # -- rank-0 raw review -----------------------------------------------------

    def _attribute_selection(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
    ) -> _AttributeReplay:
        """Create-once terminal selection marker, re-derived on every read.

        A selection that is already terminal - every SAM probe for this attribute
        has a receipt and the durable marker has been written - cannot be changed
        by any later review, completion or bbox receipt, so it is reused instead
        of being rebuilt from JPEG and mask decodes at every receipt boundary. The
        marker is still read back and compared on every hit, so drift is detected
        exactly as before; only the heavy reconstruction is skipped. A selection
        that still owes a probe is never cached, because its own next receipt is
        what advances it.
        """
        replay = self._replay_or_cached_selection(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_index,
            attribute_id,
            discovery_job_id,
        )
        self._verify_or_write_selection_marker(
            shard, clip_uid, owner_plan, attribute_id, replay
        )
        return replay

    def _replay_or_cached_selection(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_index: int,
        attribute_id: str,
        discovery_job_id: str,
        *,
        prepared: Mapping[str, Any] | None = None,
    ) -> Any:
        """Replay one attribute's selection, or reuse a stable one.

        ``prepared`` carries the plan, ordered candidates and candidate objects
        already built on the main thread, so a worker neither creates a plan nor
        rebuilds the owner candidates. Without it (the model-execution path) the
        plan is *required* rather than created, because a job that exists was
        frozen against a plan that must already be durable.

        The cache is guarded by a lock: the scheduler now overlaps a model job's
        context lookup on a worker with the main thread's replay of the previous
        receipt, so insert, evict and lookup are all done under the lock while the
        heavy replay happens outside it.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        cache_key = (shard, clip_uid, owner_entity_id, attribute_id)
        with self._attribute_selection_cache_lock:
            cached = self._attribute_selection_cache.get(cache_key)
        if cached is not None:
            stored = _read_json(
                self._selection_path(shard, clip_uid, owner_entity_id, attribute_id)
            )
            if stored is not None and stored == cached.marker:
                self._bump_replay_counter("attribute_selection_cache_hit")
                return cached
            # The marker is gone or moved: fall through and rebuild fail-closed.
            with self._attribute_selection_cache_lock:
                self._attribute_selection_cache.pop(cache_key, None)
        self._bump_replay_counter("attribute_selection_cache_miss")
        self._bump_replay_counter("attribute_selection_rebuilds")
        if prepared is not None:
            attribute_plan = prepared["attribute_plan"]
            ordered = prepared["ordered"]
            owner_candidates = prepared["owner_candidates"]
        else:
            ordered = self._ordered_owner_candidates(
                shard, storage, clip_uid, owner_plan, owner_reference
            )
            attribute_plan = self._require_attribute_plan(
                shard,
                storage,
                clip_uid,
                owner_plan,
                owner_reference,
                discovery,
                attribute_index,
                attribute_id,
                discovery_job_id,
                ordered=ordered,
            )
            owner_candidates = self._owner_candidate_objects(shard, clip_uid, owner_plan)
        return self._replay_attribute_selection_prepared(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_index,
            attribute_id,
            discovery_job_id,
            attribute_plan=attribute_plan,
            owner_candidates=owner_candidates,
            ordered=ordered,
        )

    def _verify_or_write_selection_marker(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_id: str,
        replay: Any,
    ) -> None:
        """The durable part: create-once selection marker and drift check.

        Always runs on the main thread, in annotation order, after any parallel
        replay has completed.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        marker_path = self._selection_path(
            shard, clip_uid, owner_entity_id, attribute_id
        )
        stored = _read_json(marker_path)
        if replay.pending_job is not None:
            if stored is not None:
                raise SubjectAttributeDurableError(
                    f"attribute {clip_uid}/{attribute_id} has a terminal selection "
                    "marker but its SAM probes are incomplete"
                )
            return
        if stored is None:
            _write_json_once(marker_path, replay.marker)
        elif stored != replay.marker:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes selection marker drifted for "
                f"{clip_uid}/{attribute_id}"
            )
        if replay.marker is None:
            return
        cache_key = (shard, clip_uid, owner_entity_id, attribute_id)
        # Bounded insert, guarded: a worker may be reading the cache while the
        # main thread publishes this marker.
        with self._attribute_selection_cache_lock:
            if (
                len(self._attribute_selection_cache)
                >= self._attribute_selection_cache_limit
            ):
                self._attribute_selection_cache.pop(
                    next(iter(self._attribute_selection_cache)), None
                )
            self._attribute_selection_cache[cache_key] = replay

    def _replay_attribute_selections(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        owner_reference: Any,
        discovery: SubjectAttributeDiscovery,
        attribute_ids: Sequence[str],
        discovery_job_id: str,
        prepared: Mapping[str, Mapping[str, Any]],
    ) -> list[Any]:
        """Replay every attribute's selection, concurrently when there are several.

        ``prepared`` must already hold this owner's plan and candidates, built on
        the main thread; the workers only consume it.

        An attribute's selection depends only on the owner, the discovery and its
        own SAM probe receipts, never on a sibling attribute's outcome, so they
        can be replayed at the same time. Threads rather than processes: the work
        is JPEG and RLE decode, mask geometry and cropping inside C extensions,
        and a process pool would have to pickle every decoded image and mask.

        Results come back in annotation order, never in completion order. Nothing
        here writes durable state - markers are published afterwards, on the
        calling thread - so the parallel region cannot move a durable write.
        """
        workers = int(getattr(self, "cpu_workers", 1) or 1)
        if workers <= 1 or len(attribute_ids) < 2:
            return [
                self._replay_or_cached_selection(
                    shard,
                    storage,
                    clip_uid,
                    owner_plan,
                    owner_reference,
                    discovery,
                    index,
                    attribute_id,
                    discovery_job_id,
                    prepared=prepared[attribute_id],
                )
                for index, attribute_id in enumerate(attribute_ids)
            ]
        with ThreadPoolExecutor(max_workers=min(workers, len(attribute_ids))) as pool:
            return list(
                pool.map(
                    lambda pair: self._replay_or_cached_selection(
                        shard,
                        storage,
                        clip_uid,
                        owner_plan,
                        owner_reference,
                        discovery,
                        pair[0],
                        pair[1],
                        discovery_job_id,
                        prepared=prepared[pair[1]],
                    ),
                    list(enumerate(attribute_ids)),
                )
            )

    def _rank0_context(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        discovery: SubjectAttributeDiscovery,
        discovery_job_id: str,
    ) -> dict[str, Any]:
        """Replay every attribute selection, the duplicate CPU pass and the batch."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        owner_reference = self._owner_reference(storage, clip_uid, owner_entity_id)
        # Main thread phase 1: build the owner's candidate set exactly once, so
        # the attribute workers never race to rebuild the same JPEG/RLE evidence.
        owner_candidates = tuple(
            self._owner_candidate_objects(shard, clip_uid, owner_plan)
        )
        ordered = tuple(
            self._ordered_owner_candidates(
                shard,
                storage,
                clip_uid,
                owner_plan,
                owner_reference,
                candidates=owner_candidates,
            )
        )
        attribute_ids = [
            f"a{int(owner_plan['attribute_id_start']) + index}"
            for index in range(len(discovery.attributes))
        ]
        # Main thread phase 2: create and verify every attribute plan, in
        # annotation order, before any replay is submitted.
        prepared = {
            attribute_id: {
                "attribute_plan": self._attribute_plan(
                    shard,
                    storage,
                    clip_uid,
                    owner_plan,
                    owner_reference,
                    discovery,
                    index,
                    attribute_id,
                    discovery_job_id,
                    ordered=ordered,
                ),
                "ordered": ordered,
                "owner_candidates": owner_candidates,
            }
            for index, attribute_id in enumerate(attribute_ids)
        }
        states = self._replay_attribute_selections(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            attribute_ids,
            discovery_job_id,
            prepared=prepared,
        )
        # Durable marker work stays on this thread and in annotation order, so
        # parallel replay never changes what is published or when.
        for attribute_id, replay in zip(attribute_ids, states):
            self._verify_or_write_selection_marker(
                shard, clip_uid, owner_plan, attribute_id, replay
            )
        pending = [
            state.pending_job for state in states if state.pending_job is not None
        ]
        if pending:
            # Not every attribute has a terminal selection yet: there is no
            # rank-0 batch to reason about until the probe chain drains.
            return {
                "states": states,
                "pending": pending,
                "conflicts": set(),
                "batch": [],
                "job": None,
                "review": None,
                "reviews": {},
                "failure": None,
                "discovery": discovery,
                "discovery_job_id": discovery_job_id,
            }
        rank0 = [
            state.options[0]
            for state in states
            if state.marker is not None
            and state.marker["status"] == SELECTION_CANDIDATES
            and state.options
        ]
        conflicts = _duplicate_attribute_mask_conflicts(list(rank0))
        batch = [
            state.options[0]
            for state in states
            if state.marker is not None
            and state.marker["status"] == SELECTION_CANDIDATES
            and state.options
            and state.attribute_id not in conflicts
        ]
        job = (
            self._expected_raw_review_job(
                shard,
                clip_uid,
                owner_plan,
                discovery,
                discovery_job_id,
                states,
                batch,
            )
            if batch
            else None
        )
        payload = self._committed_payload_or_none(job) if job is not None else None
        reviews: dict[str, Any] = {}
        failure: str | None = None
        if payload is not None:
            if str(payload.get("status")) == "review":
                try:
                    batch_model = SubjectAttributeReviewBatch.model_validate(
                        payload.get("review_batch")
                    )
                except Exception as exc:
                    raise SubjectAttributeDurableError(
                        "committed raw review payload is invalid"
                    ) from exc
                reviews = {review.attribute_id: review for review in batch_model.reviews}
            elif str(payload.get("status")) == "review_failed":
                failure = (
                    f"review_failed:{payload.get('exception_type')}:"
                    f"{payload.get('error')}"
                )
            else:
                raise SubjectAttributeDurableError(
                    "committed raw review has an unknown payload status"
                )
        return {
            "states": states,
            "pending": [],
            "conflicts": conflicts,
            "batch": batch,
            "job": job,
            "review": payload,
            "reviews": reviews,
            "failure": failure,
            "discovery": discovery,
            "discovery_job_id": discovery_job_id,
        }

    def _raw_review_semantic_inputs(
        self,
        *,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        discovery: SubjectAttributeDiscovery,
        discovery_job_id: str,
        states: Sequence[_AttributeReplay],
        batch: Sequence[Any],
        candidate_rank: int = 0,
    ) -> dict[str, Any]:
        sam_job_by_attribute = {
            state.attribute_id: state.sam_job_ids for state in states
        }
        entries = []
        for candidate in batch:
            attribute_id = str(candidate.attribute_id)
            sam_jobs = sam_job_by_attribute.get(attribute_id, {})
            owner_context = build_candidate_context_image(
                candidate.source_image, candidate.owner_candidate.mask
            )
            entries.append(
                {
                    "attribute_id": attribute_id,
                    "attribute_type": str(candidate.discovered.attribute_type),
                    "phrase": str(candidate.discovered.phrase),
                    "grounding_prompt": str(candidate.discovered.grounding_prompt),
                    "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
                    "frame_slot": int(candidate.owner_candidate.frame_slot),
                    "source_frame_index": int(
                        candidate.owner_candidate.source_frame_index
                    ),
                    "image_path": str(candidate.owner_candidate.image_path),
                    "sam_job_id": sam_jobs.get(
                        str(candidate.owner_candidate.candidate_id), ""
                    ),
                    "owner_context_qwen_png_sha256": _image_png_sha256(
                        _resize_qwen_input_image(owner_context)
                    ),
                    "crop_qwen_png_sha256": _image_png_sha256(
                        _resize_qwen_input_image(candidate.crop.convert("RGBA"))
                    ),
                }
            )
        return {
            "clip_uid": clip_uid,
            "owner_entity_id": str(owner_plan["owner_entity_id"]),
            "candidate_rank": str(int(candidate_rank)),
            "discovery_job_id": discovery_job_id,
            "attribute_order": [
                f"a{int(owner_plan['attribute_id_start']) + index}"
                for index in range(len(discovery.attributes))
            ],
            "candidates": entries,
            "raw_review_policy": self._raw_review_policy_identity(),
        }

    def _expected_raw_review_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        discovery: SubjectAttributeDiscovery,
        discovery_job_id: str,
        states: Sequence[_AttributeReplay],
        batch: Sequence[Any],
        candidate_rank: int = 0,
    ) -> ModelJob:
        dependencies: dict[str, str] = {
            SUBJECT_ATTRIBUTE_DISCOVERY_JOB: discovery_job_id
        }
        for candidate in batch:
            sam_jobs = next(
                (
                    state.sam_job_ids
                    for state in states
                    if state.attribute_id == candidate.attribute_id
                ),
                {},
            )
            job_id = sam_jobs.get(str(candidate.owner_candidate.candidate_id))
            if job_id:
                dependencies[f"sam_probe:{candidate.attribute_id}"] = job_id
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs=self._raw_review_semantic_inputs(
                clip_uid=clip_uid,
                owner_plan=owner_plan,
                discovery=discovery,
                discovery_job_id=discovery_job_id,
                states=states,
                batch=batch,
                candidate_rank=candidate_rank,
            ),
            model_identity=f"qwen:{self.config.qwen.candidate_judge.model}",
            target={
                "owner_entity_id": str(owner_plan["owner_entity_id"]),
                "candidate_rank": str(int(candidate_rank)),
            },
            attempt_index=int(candidate_rank),
            dependencies=dependencies,
        )

    def _run_raw_review(self, job: ModelJob, handle: Any) -> JobResult:
        """One owner-batched rank-0 review call. Nothing else is caught."""
        if job.job_type != SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB:
            raise SubjectAttributeEpochError(f"unsupported job type {job.job_type!r}")
        context = self._raw_review_context(job)
        owner = next(
            item
            for item in context["clip"].annotation.entities
            if item.entity_id == context["owner_entity_id"]
        )
        resolved = resolve_subject_attribute_qwen_client(handle, self.config)
        started = time.perf_counter()
        try:
            batch = resolved.client.review(
                owner=owner, candidates=list(context["batch"])
            )
        except Exception as exc:  # noqa: BLE001 - legacy review round isolation
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "review_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "qwen_model_call_time_seconds": time.perf_counter() - started,
                },
            )
        finally:
            _close_owned(resolved)
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "review_batch": batch.model_dump(mode="json"),
                "qwen_model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _raw_review_context(self, job: ModelJob) -> dict[str, Any]:
        """Resolve and re-verify one rank-0 raw review job."""
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip_uid = job.clip_uid
        owner_entity_id = str(dict(job.target).get("owner_entity_id", ""))
        rank = int(dict(job.target).get("candidate_rank", "0"))
        if rank not in (0, 1):
            raise SubjectAttributeEpochError(
                f"unsupported raw review rank {rank!r}"
            )
        plan = self._clip_plan(shard, clip_uid)
        owner_plan = self._owner_plan_for(shard, plan, owner_entity_id)
        discovery_payload = self._discovery_payload(shard, clip_uid, owner_plan)
        discovery = SubjectAttributeDiscovery.model_validate(
            discovery_payload.get("discovery")
        )
        discovery_job_id = self._expected_discovery_job(
            shard, clip_uid, owner_plan
        ).job_id()
        context = self._rank0_context(
            shard, storage, clip_uid, owner_plan, discovery, discovery_job_id
        )
        if context["pending"]:
            raise SubjectAttributeDurableError(
                f"raw review job {job.job_id()} was reached before its probes"
            )
        if rank == 0:
            expected = context["job"]
            batch = context["batch"]
        else:
            # Rank 1 only ever exists once every rank-0 completion has settled,
            # so the graph is the authority for the frozen second-candidate batch.
            graph = self._owner_graph(
                shard, storage, clip_uid, owner_plan, context
            )
            expected = graph["rank1_job"]
            batch = graph["batch1"]
        if expected is None:
            raise SubjectAttributeDurableError(
                f"raw review job {job.job_id()} has no rank-{rank} batch"
            )
        self._require_same_job(job, expected)
        return {
            "storage": storage,
            "clip": storage.read_clip(clip_uid),
            "owner_entity_id": owner_entity_id,
            "owner_plan": owner_plan,
            "discovery": discovery,
            "discovery_job_id": discovery_job_id,
            "batch": batch,
            "context": context,
            "candidate_rank": rank,
        }

    # -- rank-0 routes ---------------------------------------------------------

    @staticmethod
    def _rank0_route(
        state: _AttributeReplay,
        context: Mapping[str, Any],
        *,
        has_candidate2: bool | None = None,
    ) -> str:
        """The exact legacy route of one attribute in a raw review round."""
        review = context["reviews"].get(state.attribute_id)
        if has_candidate2 is None:
            has_candidate2 = len(state.options) > 1
        if review is None:
            return (
                ROUTE_CANDIDATE2_READY if has_candidate2 else ROUTE_TERMINAL_REJECT
            )
        if not (review.matches_attribute and review.owner_binding_correct):
            return (
                ROUTE_CANDIDATE2_READY if has_candidate2 else ROUTE_TERMINAL_REJECT
            )
        completion_eligible = bool(
            context["completion_enabled"]
            and state.attribute_type != "face"
            and state.attribute_type in context["completion_eligible_types"]
        )
        if completion_eligible and not review.accepted:
            return (
                ROUTE_CANDIDATE2_READY if has_candidate2 else ROUTE_TERMINAL_REJECT
            )
        if not review.sufficient_source_evidence:
            return ROUTE_CANDIDATE2_READY if has_candidate2 else ROUTE_BBOX_PENDING
        if not completion_eligible:
            if review.accepted:
                return (
                    ROUTE_FACE_ACCEPT
                    if state.attribute_type == "face"
                    else ROUTE_RAW_ACCEPT
                )
            return ROUTE_CANDIDATE2_READY if has_candidate2 else ROUTE_BBOX_PENDING
        raw_publishable = bool(
            review.accepted
            and review.structure_complete
            and not review.completion_recommended
        )
        return ROUTE_RAW_ACCEPT if raw_publishable else ROUTE_COMPLETION_REQUIRED

    def _raw_state_payload(
        self,
        *,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        discovery: SubjectAttributeDiscovery,
        discovery_job_id: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        completion = self.config.subject_attributes.completion
        route_context = {
            **context,
            "completion_enabled": bool(completion.enabled),
            "completion_eligible_types": tuple(completion.eligible_types),
        }
        attributes = []
        for state in context["states"]:
            marker = state.marker
            if marker is None or marker["status"] == SELECTION_REJECTED:
                route = ROUTE_SELECTION_REJECTED
            elif state.attribute_id in context["conflicts"]:
                route = ROUTE_DUPLICATE_CONFLICT
            else:
                route = self._rank0_route(state, route_context)
            review = context["reviews"].get(state.attribute_id)
            attributes.append(
                {
                    "attribute_id": state.attribute_id,
                    "selection_status": (
                        marker["status"] if marker is not None else SELECTION_REJECTED
                    ),
                    "duplicate_conflict": state.attribute_id in context["conflicts"],
                    "rank0_review": (
                        review.model_dump(mode="json") if review is not None else None
                    ),
                    "rank0_review_failure": context["failure"],
                    "route_after_rank0": route,
                }
            )
        return {
            "schema": SUBJECT_ATTRIBUTE_OWNER_RAW_STATE_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": str(owner_plan["owner_entity_id"]),
            "attribute_id_start": int(owner_plan["attribute_id_start"]),
            "discovery_job_id": discovery_job_id,
            "rank0_review_job_id": (
                context["job"].job_id() if context["job"] is not None else None
            ),
            "attributes": attributes,
        }

    def _owner_prefix(
        self, shard: str, clip_uid: str, owner_entity_id: str
    ) -> dict[str, Any]:
        """The frozen owner prefix: plan, discovery and discovery job id.

        All three are fixed by the time an owner is advanced - the clip plan and
        owner plan are frozen plans, and the discovery is a durable receipt - and
        none of them can be changed by a later review, completion or bbox receipt.
        Caching them removes repeated plan derivation and discovery validation
        from every receipt boundary. Receipt-driven state is never cached here.
        """
        key = (shard, clip_uid, owner_entity_id)
        cached = self._owner_context_cache.get(key)
        if cached is not None:
            self._bump_replay_counter("owner_context_cache_hit")
            return cached
        self._bump_replay_counter("owner_context_cache_miss")
        plan = self._clip_plan(shard, clip_uid)
        owner_plan = self._owner_plan_for(shard, plan, owner_entity_id)
        discovery_payload = self._discovery_payload(shard, clip_uid, owner_plan)
        discovery = SubjectAttributeDiscovery.model_validate(
            discovery_payload.get("discovery")
        )
        payload = {
            "owner_plan": owner_plan,
            "discovery": discovery,
            "discovery_job_id": self._expected_discovery_job(
                shard, clip_uid, owner_plan
            ).job_id(),
        }
        if len(self._owner_context_cache) >= self._owner_context_cache_limit:
            self._owner_context_cache.pop(next(iter(self._owner_context_cache)), None)
        self._owner_context_cache[key] = payload
        return payload

    def _advance_attributes(
        self, shard: str, storage: RunStorage, clip_uid: str, owner_entity_id: str
    ) -> list[ModelJob]:
        """Advance one human owner's semantic graph as far as receipts allow.

        The rank-0 boundary marker is verified first (it is derived, never
        trusted), then the real legacy owner algorithm is re-run over the
        committed receipts. The first model call it cannot answer from a receipt
        is the next job, and once every call is paid the legacy artifact is
        published together with the processed owner outcome.
        """
        prefix = self._owner_prefix(shard, clip_uid, owner_entity_id)
        owner_plan = prefix["owner_plan"]
        discovery = prefix["discovery"]
        discovery_job_id = prefix["discovery_job_id"]
        context = self._rank0_context(
            shard, storage, clip_uid, owner_plan, discovery, discovery_job_id
        )
        if context["pending"]:
            return sorted(context["pending"], key=lambda job: job.job_id())
        if context["job"] is not None and context["review"] is None:
            # The owner's rank-0 raw review is still owed; the boundary marker is
            # only meaningful once every rank-0 verdict exists.
            return [context["job"]]
        self._verified_raw_state(shard, storage, clip_uid, owner_plan, context)
        outcome = self._owner_outcome(shard, clip_uid, owner_entity_id)
        if outcome is not None:
            self._verify_owner_outcome(
                shard, storage, clip_uid, owner_entity_id, owner_plan, outcome
            )
            return self._advance_clip(shard, storage, clip_uid)
        self._bump_replay_counter("owner_graph_rebuilds")
        graph = self._owner_graph(
            shard, storage, clip_uid, owner_plan, context
        )
        self._publish_owner_markers(
            shard, storage, clip_uid, owner_plan, graph
        )
        # The owner graph already knows every independent model call this owner
        # owes, so hand all of them to the pool at once instead of letting the
        # legacy replay abort on the first one it reaches. The replay is only run
        # once the graph owes nothing, which is when it can produce the final
        # owner artifact without aborting.
        owed = self._owner_owed_jobs(graph)
        if owed:
            return owed
        result = self._owner_artifact_from_receipts(
            shard, storage, clip_uid, owner_plan, context["states"]
        )
        if isinstance(result, _PendingModelCall):
            return [result.job]
        self._publish_owner_artifact(shard, clip_uid, owner_plan, result)
        self._write_owner_outcome(
            shard,
            clip_uid,
            owner_entity_id,
            self._expected_owner_outcome(
                clip_uid=clip_uid,
                owner_entity_id=owner_entity_id,
                owner_plan=owner_plan,
                artifact=result,
                source=OWNER_SOURCE_PROCESSED,
                discovery_job_id=discovery_job_id,
            ),
        )
        return self._advance_clip(shard, storage, clip_uid)

    def _finalize_sam_probe(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """CPU continuation of one committed SAM probe."""
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        if status not in {"sam", "sam_failed"}:
            raise SubjectAttributeDurableError(
                f"committed SAM probe {job.job_id()} has an unknown payload status"
            )
        context = self._sam_probe_context(job)
        if status == "sam":
            # The masks are this probe's durable artifact; verify them now.
            self._load_sam_masks(payload)
        return self._advance_attributes(
            job.canonical_shard,
            context["storage"],
            job.clip_uid,
            str(dict(job.target).get("owner_entity_id", "")),
        )

    def _finalize_owner_chain(
        self, job: ModelJob, result: JobResult
    ) -> Sequence[ModelJob]:
        """CPU continuation of any committed 8b/8c owner model call.

        Every payload shape is validated here so that a torn or foreign receipt
        fails closed, and the owner's graph is then re-derived from the receipts:
        whatever the next missing call is becomes the next job.
        """
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        expected_statuses = {
            SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB: {"review", "review_failed"},
            SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB: {
                "completion",
                "completion_failed",
            },
            SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB: {"sam", "sam_failed"},
            SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_JOB: {"review", "review_failed"},
            SUBJECT_ATTRIBUTE_BBOX_REVIEW_JOB: {"bbox", "bbox_failed"},
        }
        if status not in expected_statuses[job.job_type]:
            raise SubjectAttributeDurableError(
                f"committed {job.job_type} {job.job_id()} has an unknown payload "
                f"status {status!r}"
            )
        if status == "review" and job.job_type == SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB:
            try:
                SubjectAttributeReviewBatch.model_validate(
                    payload.get("review_batch")
                )
            except Exception as exc:
                raise SubjectAttributeDurableError(
                    f"committed raw review {job.job_id()} has an invalid batch"
                ) from exc
        if status == "sam":
            self._load_sam_masks(payload)
        if status == "completion":
            storage = self._storage_for(job.canonical_shard)
            self._generated_png_bytes(storage, payload)
        return self._advance_attributes(
            job.canonical_shard,
            self._storage_for(job.canonical_shard),
            job.clip_uid,
            str(dict(job.target).get("owner_entity_id", "")),
        )

    # -- 8c: completion seed, CPU postcheck and chain state ---------------------

    def _completion_dir(
        self, shard: str, clip_uid: str, owner: str, attribute_id: str
    ) -> Path:
        return self._attribute_dir(shard, clip_uid, owner, attribute_id) / "completion"

    def _completion_seed_path(self, shard, clip_uid, owner, attribute_id, rank) -> Path:
        return self._completion_dir(
            shard, clip_uid, owner, attribute_id
        ) / f"rank-{rank}-seed.json"

    def _completion_outcome_path(self, shard, clip_uid, owner, attribute_id, rank):
        return self._completion_dir(
            shard, clip_uid, owner, attribute_id
        ) / f"rank-{rank}-outcome.json"

    def _completion_source_path(
        self, storage: RunStorage, clip_uid, owner, attribute_id, candidate_id
    ) -> Path:
        return (
            self._output_root(storage)
            / "completion_candidates"
            / clip_uid
            / owner
            / attribute_id
            / candidate_id
            / "raw"
            / "source.png"
        )

    def _completion_output_path(
        self, storage: RunStorage, clip_uid, owner, attribute_id, candidate_id
    ) -> Path:
        return (
            self._output_root(storage)
            / "completion_candidates"
            / clip_uid
            / owner
            / attribute_id
            / candidate_id
            / "generated"
            / "candidate.png"
        )

    def _completion_seed(
        self,
        shard: str,
        clip_uid: str,
        owner_entity_id: str,
        attribute_id: str,
        rank: int,
        owner_candidate_id: str,
    ) -> int:
        """Create-once durable Boogu seed.

        Legacy draws a fresh seed per completion request, which a restart would
        redraw and thereby change both the model call and the recorded artifact.
        The epoch freezes it instead: the seed is drawn once, stored next to the
        attribute it belongs to, and reused exactly afterwards.
        """
        path = self._completion_seed_path(
            shard, clip_uid, owner_entity_id, attribute_id, rank
        )
        expected = {
            "schema": SUBJECT_ATTRIBUTE_COMPLETION_SEED_SCHEMA,
            "clip_uid": clip_uid,
            "owner_entity_id": owner_entity_id,
            "attribute_id": attribute_id,
            "candidate_rank": int(rank),
            "owner_candidate_id": owner_candidate_id,
        }
        existing = _read_json(path)
        if existing is not None:
            if set(existing) != set(expected) | {"seed"}:
                raise SubjectAttributeDurableError(
                    f"frozen completion seed keyset drifted for {path}"
                )
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise SubjectAttributeDurableError(
                        f"frozen completion seed {key} drifted for {path}"
                    )
            seed = existing.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise SubjectAttributeDurableError(
                    f"frozen completion seed is invalid for {path}"
                )
            return seed
        seed = int(new_boogu_seed())
        _write_json_once(path, {**expected, "seed": seed})
        return seed

    def _completion_input_bytes(self, crop: Any) -> bytes:
        """The exact PNG legacy ``_save_completion_input`` writes for a crop."""
        buffer = io.BytesIO()
        crop.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def _completion_cpu_result(
        self, generated_bytes: bytes, masks: Sequence[Any]
    ) -> dict[str, Any]:
        """Legacy ``_attempt_attribute_completion`` CPU stages, in legacy order.

        Only the stages that do not need a model live here: the mask postcheck,
        the cleanup, the size and component gates, the reference crop and the
        quality evaluation. The reasons are byte-identical to legacy so the
        recorded ``completion_outcome`` never diverges from the durable marker.
        """
        completion = self.config.subject_attributes.completion
        with Image.open(io.BytesIO(generated_bytes)) as opened:
            opened.load()
            completed_rgb = opened.convert("RGB")
        usable, rejection = _completion_mask_postcheck_rejection(
            tuple(masks),
            expected_shape=(completed_rgb.height, completed_rgb.width),
        )
        counters = {
            "sam_masks_returned_total": len(masks),
            "sam_zero_mask_rejects": 0,
            "postcheck_rejects": 0,
            "sam_single_mask": 0,
            "sam_multi_mask": 0,
        }
        if rejection is not None:
            counters["postcheck_rejects"] = 1
            return {
                "status": COMPLETION_POSTCHECK_REJECTED,
                "reason": f"completion_postcheck:{rejection}",
                "completed_rgba": None,
                **counters,
            }
        if not usable:
            counters["sam_zero_mask_rejects"] = 1
            counters["postcheck_rejects"] = 1
            return {
                "status": COMPLETION_POSTCHECK_REJECTED,
                "reason": "completion_postcheck:missing_entity",
                "completed_rgba": None,
                **counters,
            }
        counters["sam_single_mask"] = int(len(usable) == 1)
        counters["sam_multi_mask"] = int(len(usable) > 1)
        union_mask = np.logical_or.reduce(list(usable))
        cleaned_mask, cleanup_diagnostics = _clean_completion_mask(
            union_mask, config=completion.quality_filter
        )
        completed_rows, completed_columns = np.nonzero(cleaned_mask)
        # An empty cleaned mask raises here exactly like legacy's own reduction.
        completed_long_side = max(
            int(completed_columns.max() - completed_columns.min() + 1),
            int(completed_rows.max() - completed_rows.min() + 1),
        )
        if (
            int(cleaned_mask.sum()) < MIN_ATTRIBUTE_AREA_PIXELS
            or completed_long_side < MIN_ATTRIBUTE_LONG_SIDE_PIXELS
        ):
            counters["postcheck_rejects"] = 1
            return {
                "status": COMPLETION_POSTCHECK_REJECTED,
                "reason": "completion_postcheck:attribute_too_small",
                "completed_rgba": None,
                **counters,
            }
        components = mask_component_diagnostics(cleaned_mask)
        if (
            components.significant_component_count
            > completion.maximum_completed_significant_components
        ):
            counters["postcheck_rejects"] = 1
            return {
                "status": COMPLETION_POSTCHECK_REJECTED,
                "reason": "completion_postcheck:extra_entity_contours",
                "completed_rgba": None,
                **counters,
            }
        completed_rgba, _ = build_reference_crop(
            completed_rgb,
            cleaned_mask,
            crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
        )
        quality = _evaluate_completion_quality(
            completed_rgba,
            config=completion.quality_filter,
            minimum_mean_luminance=completion.minimum_mean_luminance,
            minimum_sharpness_score=completion.minimum_sharpness_score,
        )
        del cleanup_diagnostics
        if not quality.passed:
            counters["postcheck_rejects"] = 1
            return {
                "status": COMPLETION_POSTCHECK_REJECTED,
                "reason": "completion_postcheck:" + ",".join(quality.reasons),
                "completed_rgba": None,
                **counters,
            }
        return {
            "status": COMPLETION_REVIEW_REQUIRED,
            "reason": None,
            "completed_rgba": completed_rgba,
            "completed_rgb": completed_rgb,
            **counters,
        }

    # -- 8c: model job identities ---------------------------------------------

    def _completion_postcheck_policy_identity(self) -> dict[str, Any]:
        """Every input the completion CPU postcheck actually reads.

        These values decide the cleanup, the quality verdict and therefore the
        completed crop, so they are part of the SAM-to-review semantic identity:
        a threshold change must produce a different review job.
        """
        completion = self.config.subject_attributes.completion
        quality = completion.quality_filter
        return {
            "policy_version": (
                SUBJECT_ATTRIBUTE_COMPLETION_POSTCHECK_POLICY_VERSION
            ),
            "minimum_attribute_area_pixels": int(MIN_ATTRIBUTE_AREA_PIXELS),
            "minimum_attribute_long_side_pixels": int(
                MIN_ATTRIBUTE_LONG_SIDE_PIXELS
            ),
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
            "completion_enabled": bool(completion.enabled),
            "maximum_completed_significant_components": int(
                completion.maximum_completed_significant_components
            ),
            "minimum_mean_luminance": float(completion.minimum_mean_luminance),
            "minimum_sharpness_score": float(completion.minimum_sharpness_score),
            "quality_filter_enabled": bool(quality.enabled),
            "quality_filter_version": str(quality.version),
            "quality_filter_secondary_component_ratio_max": float(
                quality.secondary_component_ratio_max
            ),
            "quality_filter_second_component_ratio_max": float(
                quality.second_component_ratio_max
            ),
            "quality_filter_soft_alpha_ratio_max": float(
                quality.soft_alpha_ratio_max
            ),
            "quality_filter_weak_alpha_ratio_max": float(
                quality.weak_alpha_ratio_max
            ),
            "quality_filter_outer_weak_alpha_ratio_max": float(
                quality.outer_weak_alpha_ratio_max
            ),
            "quality_filter_foreground_fill_min": float(
                quality.foreground_fill_min
            ),
            "quality_filter_cleanup_component_ratio_max": float(
                quality.cleanup_component_ratio_max
            ),
            "quality_filter_cleanup_component_pixels_max": int(
                quality.cleanup_component_pixels_max
            ),
            "mode": "legacy_completion_postcheck_v1",
        }

    def _completion_generate_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of one Boogu completion request."""
        edit = self.config.reference_edit
        return {
            "policy_version": SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_POLICY_VERSION,
            "model_path": str(edit.model_path),
            "model_revision": str(edit.model_revision),
            "target_area": int(edit.target_area),
            "alignment": int(edit.alignment),
            "thinking_enabled": False,
            "instruction_rewrite_enabled": False,
            "completion_instruction_rewrite_enabled": bool(
                edit.completion_instruction_rewrite_enabled
            ),
            "add_background_to_complete": bool(edit.add_background_to_complete),
            "crop_padding_ratio": float(self.config.pair.crop_padding_ratio),
            "mode": "legacy_attempt_attribute_completion_v1",
        }

    def _completion_review_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the comparative completion review.

        ``QwenSubjectAttributeCompletionJudge`` sends a fixed request: the
        configured candidate judge model and max tokens, but temperature 0.0,
        top-p 1.0 and presence penalty 0.0 regardless of the service config, with
        one structured repair retry. The policy therefore binds those real
        request parameters and both prompt and schema digests, never the service
        temperature, and never endpoint, key or timeout.
        """
        service = self.config.qwen.candidate_judge
        return {
            "policy_version": SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_POLICY_VERSION,
            "model": str(service.model),
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": int(service.max_tokens),
            "repair_retries": 1,
            "system_prompt_sha256": _sha256_bytes(
                ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT.encode("utf-8")
            ),
            "review_schema_sha256": _sha256_bytes(
                canonical_json(
                    SubjectAttributeCompletionReview.model_json_schema()
                ).encode("utf-8")
            ),
            "qwen_input_max_long_side_pixels": int(QWEN_INPUT_MAX_LONG_SIDE_PIXELS),
            "mode": "comparative_alpha_versus_completion_v1",
        }

    def _bbox_review_policy_identity(self) -> dict[str, Any]:
        """Semantic identity of the last-resort bbox review.

        A face normally never reaches this judge, but both legacy bbox system
        prompts are bound so the identity stays complete if it ever legally did.
        """
        service = self.config.qwen.candidate_judge
        return {
            "policy_version": SUBJECT_ATTRIBUTE_BBOX_REVIEW_POLICY_VERSION,
            "model": str(service.model),
            "temperature": float(service.temperature),
            "max_tokens": int(service.max_tokens),
            "repair_retries": 1,
            "bbox_system_prompt_sha256": _sha256_bytes(
                ATTRIBUTE_BBOX_REVIEW_SYSTEM_PROMPT.encode("utf-8")
            ),
            "face_bbox_system_prompt_sha256": _sha256_bytes(
                ATTRIBUTE_FACE_BBOX_REVIEW_SYSTEM_PROMPT.encode("utf-8")
            ),
            "bbox_schema_sha256": _sha256_bytes(
                canonical_json(
                    SubjectAttributeBboxReview.model_json_schema()
                ).encode("utf-8")
            ),
            "qwen_input_max_long_side_pixels": int(QWEN_INPUT_MAX_LONG_SIDE_PIXELS),
            "mode": "last_resort_attribute_bbox_review_v1",
        }

    def _expected_completion_generate_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        candidate: Any,
        rank: int,
        seed: int,
    ) -> ModelJob:
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_id = str(attribute_plan["attribute_id"])
        owner_candidate_id = str(candidate.owner_candidate.candidate_id)
        discovered = attribute_plan["discovered"]
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB,
            resource=RESOURCE_BOOGU,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "clip_uid": clip_uid,
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": owner_candidate_id,
                "attribute_type": discovered["attribute_type"],
                "attribute_phrase": discovered["phrase"],
                "grounding_prompt": discovered["grounding_prompt"],
                "instruction": attribute_completion_prompt(
                    discovered["attribute_type"]
                ),
                "raw_crop_png_sha256": _image_png_sha256(
                    candidate.crop.convert("RGBA")
                ),
                "completion_input_png_sha256": _sha256_bytes(
                    self._completion_input_bytes(candidate.crop)
                ),
                "seed": int(seed),
                "completion_generate_policy": (
                    self._completion_generate_policy_identity()
                ),
            },
            model_identity=f"boogu:{self.config.reference_edit.model_revision}",
            target={
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": owner_candidate_id,
            },
            attempt_index=int(rank),
            dependencies={
                SUBJECT_ATTRIBUTE_DISCOVERY_JOB: str(
                    attribute_plan["discovery_job_id"]
                )
            },
        )

    def _expected_completion_sam_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        candidate: Any,
        rank: int,
        generate_job: ModelJob,
        generated_sha256: str,
        generated_path: str,
    ) -> ModelJob:
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_id = str(attribute_plan["attribute_id"])
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB,
            resource=RESOURCE_SAM,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "clip_uid": clip_uid,
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
                "grounding_prompt": attribute_plan["discovered"]["grounding_prompt"],
                "generated_png_path": str(generated_path),
                "generated_png_sha256": str(generated_sha256),
                "generation_job_id": generate_job.job_id(),
                "postcheck_policy": self._completion_postcheck_policy_identity(),
            },
            model_identity="sam3",
            target={
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
            },
            attempt_index=int(rank),
            dependencies={"completion_generate": generate_job.job_id()},
        )

    def _expected_completion_review_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        candidate: Any,
        rank: int,
        sam_job: ModelJob,
        source_attribute_png_sha256: str,
        source_bbox_png_sha256: str,
        generated_candidate_png_sha256: str,
    ) -> ModelJob:
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_id = str(attribute_plan["attribute_id"])
        discovered = attribute_plan["discovered"]
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_COMPLETION_REVIEW_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "clip_uid": clip_uid,
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
                "attribute_type": discovered["attribute_type"],
                "attribute_phrase": discovered["phrase"],
                "source_attribute_png_sha256": str(source_attribute_png_sha256),
                "source_bbox_png_sha256": str(source_bbox_png_sha256),
                "generated_candidate_png_sha256": str(
                    generated_candidate_png_sha256
                ),
                "sam_job_id": sam_job.job_id(),
                "completion_review_policy": self._completion_review_policy_identity(),
            },
            model_identity=f"qwen:{self.config.qwen.candidate_judge.model}",
            target={
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
            },
            attempt_index=int(rank),
            dependencies={"completion_sam": sam_job.job_id()},
        )

    def _expected_bbox_review_job(
        self,
        shard: str,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        candidate: Any,
        rank: int,
        owner_context_png_sha256: str,
        bbox_png_sha256: str,
    ) -> ModelJob:
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_id = str(attribute_plan["attribute_id"])
        discovered = attribute_plan["discovered"]
        return ModelJob.create(
            job_type=SUBJECT_ATTRIBUTE_BBOX_REVIEW_JOB,
            resource=RESOURCE_QWEN,
            canonical_shard=shard,
            clip_uid=clip_uid,
            semantic_inputs={
                "clip_uid": clip_uid,
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
                "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
                "attribute_type": discovered["attribute_type"],
                "attribute_phrase": discovered["phrase"],
                "owner_context_qwen_png_sha256": str(owner_context_png_sha256),
                "bbox_qwen_png_sha256": str(bbox_png_sha256),
                "bbox_review_policy": self._bbox_review_policy_identity(),
            },
            model_identity=f"qwen:{self.config.qwen.candidate_judge.model}",
            target={
                "owner_entity_id": owner_entity_id,
                "attribute_id": attribute_id,
                "candidate_rank": str(int(rank)),
            },
            attempt_index=int(rank),
            dependencies={
                SUBJECT_ATTRIBUTE_DISCOVERY_JOB: str(
                    attribute_plan["discovery_job_id"]
                )
            },
        )

    # -- 8c: durable completion chain -----------------------------------------

    def _completion_png_path(self, job_id: str) -> Path:
        return self._semantic("binary", "completion", job_id, "generated.png")

    def _generated_png_bytes(
        self, storage: RunStorage, payload: Mapping[str, Any]
    ) -> bytes:
        """The generated PNG of one committed generation, materialised on disk.

        The ledger copy is authority; the legacy working path under the storage
        root is restored from it when it is missing or was tampered with, so the
        completion SAM call always sees the exact bytes the model produced.
        """
        record = payload.get("generated")
        if not isinstance(record, Mapping):
            raise SubjectAttributeDurableError(
                "committed completion generation has no generated image"
            )
        ledger_path = Path(self.ledger.root) / str(record.get("path", ""))
        if not ledger_path.is_file():
            raise SubjectAttributeDurableError(
                f"completion generated image is missing: {record.get('path')}"
            )
        data = ledger_path.read_bytes()
        if _sha256_bytes(data) != str(record.get("sha256")):
            raise SubjectAttributeDurableError(
                "completion generated image drifted from its receipt"
            )
        target = Path(storage.root) / str(payload.get("generated_png_path", ""))
        if not target.is_file() or target.read_bytes() != data:
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target, data)
        return data

    def _completion_chain(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        attribute_plan: Mapping[str, Any],
        candidate: Any,
        rank: int,
    ) -> _CompletionChain:
        """Derive one candidate's completion chain from its durable receipts."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_id = str(attribute_plan["attribute_id"])
        seed = self._completion_seed(
            shard,
            clip_uid,
            owner_entity_id,
            attribute_id,
            rank,
            str(candidate.owner_candidate.candidate_id),
        )
        generate_job = self._expected_completion_generate_job(
            shard, clip_uid, owner_plan, attribute_plan, candidate, rank, seed
        )
        base = {
            "attribute_id": attribute_id,
            "attribute_type": str(attribute_plan["discovered"]["attribute_type"]),
            "candidate_rank": int(rank),
            "owner_candidate_id": str(candidate.owner_candidate.candidate_id),
            "candidate": candidate,
            "seed": seed,
            "generate_job": generate_job,
        }
        payload = self._committed_payload_or_none(generate_job)
        if payload is None:
            return _CompletionChain(
                sam_job=None,
                review_job=None,
                status=None,
                reason=None,
                next_job=generate_job,
                review=None,
                completed_crop_sha256=None,
                **base,
            )
        if str(payload.get("status")) == "completion_failed":
            return _CompletionChain(
                sam_job=None,
                review_job=None,
                status=COMPLETION_GENERATION_FAILED,
                reason=(
                    f"completion_failed:{payload.get('exception_type')}:"
                    f"{payload.get('error')}"
                ),
                next_job=None,
                review=None,
                completed_crop_sha256=None,
                **base,
            )
        generated = self._generated_png_bytes(storage, payload)
        sam_job = self._expected_completion_sam_job(
            shard,
            clip_uid,
            owner_plan,
            attribute_plan,
            candidate,
            rank,
            generate_job,
            str(payload.get("generated", {}).get("sha256")),
            str(payload.get("generated_png_path", "")),
        )
        sam_payload = self._committed_payload_or_none(sam_job)
        if sam_payload is None:
            return _CompletionChain(
                sam_job=sam_job,
                review_job=None,
                status=None,
                reason=None,
                next_job=sam_job,
                review=None,
                completed_crop_sha256=None,
                **base,
            )
        if str(sam_payload.get("status")) == "sam_failed":
            return _CompletionChain(
                sam_job=sam_job,
                review_job=None,
                status=COMPLETION_SAM_FAILED,
                reason=(
                    f"completion_failed:{sam_payload.get('exception_type')}:"
                    f"{sam_payload.get('error')}"
                ),
                next_job=None,
                review=None,
                completed_crop_sha256=None,
                **base,
            )
        masks = self._load_sam_masks(sam_payload)
        cpu = self._completion_cpu_result(generated, masks)
        if cpu["status"] == COMPLETION_POSTCHECK_REJECTED:
            return _CompletionChain(
                sam_job=sam_job,
                review_job=None,
                status=COMPLETION_POSTCHECK_REJECTED,
                reason=str(cpu["reason"]),
                next_job=None,
                review=None,
                completed_crop_sha256=None,
                **base,
            )
        completed_rgba = cpu["completed_rgba"]
        completed_sha = _image_png_sha256(completed_rgba.convert("RGBA"))
        review_job = self._expected_completion_review_job(
            shard,
            clip_uid,
            owner_plan,
            attribute_plan,
            candidate,
            rank,
            sam_job,
            _image_png_sha256(candidate.crop.convert("RGBA")),
            _image_png_sha256(
                _attribute_bbox_crop(
                    candidate.source_image,
                    candidate.attribute_mask,
                    crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
                )
            ),
            completed_sha,
        )
        review_payload = self._committed_payload_or_none(review_job)
        if review_payload is None:
            return _CompletionChain(
                sam_job=sam_job,
                review_job=review_job,
                status=COMPLETION_REVIEW_REQUIRED,
                reason=None,
                next_job=review_job,
                review=None,
                completed_crop_sha256=completed_sha,
                **base,
            )
        if str(review_payload.get("status")) == "review_failed":
            return _CompletionChain(
                sam_job=sam_job,
                review_job=review_job,
                status=COMPLETION_REVIEW_FAILED,
                reason=(
                    f"completion_failed:{review_payload.get('exception_type')}:"
                    f"{review_payload.get('error')}"
                ),
                next_job=None,
                review=None,
                completed_crop_sha256=completed_sha,
                **base,
            )
        try:
            review = SubjectAttributeCompletionReview.model_validate(
                review_payload.get("review")
            )
        except Exception as exc:
            raise SubjectAttributeDurableError(
                "committed completion review payload is invalid"
            ) from exc
        accepted = review.verdict == "accept"
        return _CompletionChain(
            sam_job=sam_job,
            review_job=review_job,
            status=COMPLETION_ACCEPTED if accepted else COMPLETION_REVIEW_REJECTED,
            reason=(
                "completion_identity_accepted"
                if accepted
                else "completion_qwen_review_reject"
            ),
            next_job=None,
            review=review,
            completed_crop_sha256=completed_sha,
            **base,
        )

    def _completion_outcome_payload(
        self,
        storage: RunStorage,
        chain: _CompletionChain,
    ) -> dict[str, Any]:
        """The exact durable completion outcome marker of a terminal chain."""
        if not chain.terminal:
            raise SubjectAttributeDurableError(
                f"completion chain {chain.attribute_id} is not terminal"
            )
        return {
            "schema": SUBJECT_ATTRIBUTE_COMPLETION_OUTCOME_SCHEMA,
            "clip_uid": chain.generate_job.clip_uid,
            "owner_entity_id": str(
                dict(chain.generate_job.target)["owner_entity_id"]
            ),
            "attribute_id": chain.attribute_id,
            "candidate_rank": int(chain.candidate_rank),
            "owner_candidate_id": chain.owner_candidate_id,
            "status": chain.status,
            "reason": chain.reason,
            "seed": int(chain.seed),
            "generation_job_id": chain.generate_job.job_id(),
            "sam_job_id": (
                chain.sam_job.job_id() if chain.sam_job is not None else None
            ),
            "review_job_id": (
                chain.review_job.job_id() if chain.review_job is not None else None
            ),
            "completed_crop_sha256": chain.completed_crop_sha256,
            "completion_review": (
                chain.review.model_dump(mode="json")
                if chain.review is not None
                else None
            ),
        }

    def _publish_completion_outcome(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_entity_id: str,
        chain: _CompletionChain,
    ) -> None:
        if not chain.terminal:
            return
        path = self._completion_outcome_path(
            shard, clip_uid, owner_entity_id, chain.attribute_id, chain.candidate_rank
        )
        _write_json_once(path, self._completion_outcome_payload(storage, chain))

    # -- 8c: legacy re-run over the committed receipts -------------------------

    def _owner_replay(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        states: Sequence[_AttributeReplay],
    ) -> _OwnerReplay:
        clip = storage.read_clip(clip_uid)
        owner_entity_id = str(owner_plan["owner_entity_id"])
        owner = next(
            item
            for item in clip.annotation.entities
            if item.entity_id == owner_entity_id
        )
        return _OwnerReplay(
            epoch=self,
            shard=shard,
            storage=storage,
            clip=clip,
            owner=owner,
            owner_plan=owner_plan,
            reference=self._owner_reference(storage, clip_uid, owner_entity_id),
            discovery=SubjectAttributeDiscovery.model_validate(
                self._discovery_payload(shard, clip_uid, owner_plan).get("discovery")
            ),
            discovery_job_id=self._expected_discovery_job(
                shard, clip_uid, owner_plan
            ).job_id(),
            candidates=self._owner_candidate_objects(shard, clip_uid, owner_plan),
            masks=storage.read_masks(clip_uid),
            states=list(states),
        )

    def _owner_artifact_from_receipts(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        states: Sequence[_AttributeReplay],
    ) -> Any:
        """The exact legacy owner artifact, re-run over the committed receipts.

        The real legacy ``_process_owner`` drives every route, record and
        counter; the epoch only replaces the five model surfaces with replay
        clients that answer from durable receipts. The first call without a
        receipt aborts through ``_PendingModelCall`` and becomes the next job, so
        the model graph and the recorded artifact are the legacy ones by
        construction rather than by transcription.
        """
        replay = self._owner_replay(shard, storage, clip_uid, owner_plan, states)
        try:
            artifact = _process_owner(
                config=self.config,
                storage=storage,
                output_root=self._output_root(storage),
                clip=replay.clip,
                owner=replay.owner,
                owner_candidates=replay.candidates,
                masks=replay.masks,
                attribute_id_start=int(owner_plan["attribute_id_start"]),
                discovery_client=_OwnerDiscoveryClient(replay),
                review_client=_OwnerReviewClient(replay),
                segmentation_backend=_OwnerSegmentationBackend(replay),
                gme_screener=None,
                completion_backend=_OwnerCompletionBackend(replay),
                completion_judge=_OwnerCompletionJudge(replay),
            )
        except _PendingModelCall as pending:
            return pending
        return self._reconcile_owner_artifact(replay, artifact)

    def _reconcile_owner_artifact(self, replay: _OwnerReplay, artifact: Any) -> Any:
        """Freeze the two execution-time-only legacy inputs.

        Wall-clock model timing and the freshly drawn Boogu seed are the only
        legacy values that must not survive into a durable epoch artifact: the
        times come from the committed receipts instead of restart CPU replay, and
        each record's ``completion_seed`` is the frozen per-attribute seed.
        """
        records = []
        for record in artifact.records:
            if not record.completion_attempted:
                records.append(record)
                continue
            seeded = replay.frozen_seed_for(record.attribute_id)
            records.append(
                record.model_copy(update={"completion_seed": seeded})
                if seeded is not None
                else record
            )
        metrics = artifact.metrics.model_copy(
            update={
                "qwen_model_call_time_seconds": replay.qwen_seconds,
                "sam3_model_call_time_seconds": replay.sam3_seconds,
                "completion_model_call_time_seconds": replay.completion_seconds,
            }
        )
        return artifact.model_copy(update={"records": records, "metrics": metrics})

    # -- 8c: the owner's semantic graph ---------------------------------------

    def _rank_route_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        completion = self.config.subject_attributes.completion
        return {
            **context,
            "completion_enabled": bool(completion.enabled),
            "completion_eligible_types": tuple(completion.eligible_types),
        }

    def _owner_routes(self, context: Mapping[str, Any]) -> dict[str, str]:
        """The exact legacy route of every attribute in the rank-0 round."""
        route_context = self._rank_route_context(context)
        routes: dict[str, str] = {}
        for state in context["states"]:
            marker = state.marker
            if marker is None or marker["status"] != SELECTION_CANDIDATES:
                routes[state.attribute_id] = ROUTE_SELECTION_REJECTED
            elif state.attribute_id in context["conflicts"]:
                routes[state.attribute_id] = ROUTE_DUPLICATE_CONFLICT
            else:
                routes[state.attribute_id] = self._rank0_route(state, route_context)
        return routes

    def _owner_graph(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Derive the whole owner graph from the receipts, in legacy order.

        Rank-0 completions settle before the single rank-1 batch is decided, so
        no rank-1 review is ever planned speculatively. Every seed is created
        only for a completion the routes actually attempt.
        """
        routes0 = self._owner_routes(context)
        chains0: dict[str, _CompletionChain] = {}
        for state in context["states"]:
            if routes0[state.attribute_id] != ROUTE_COMPLETION_REQUIRED:
                continue
            chains0[state.attribute_id] = self._completion_chain(
                shard,
                storage,
                clip_uid,
                owner_plan,
                state.attribute_plan,
                state.options[0],
                0,
            )
        rank1_states: list[_AttributeReplay] = []
        for state in context["states"]:
            if len(state.options) < 2:
                continue
            route = routes0[state.attribute_id]
            if route == ROUTE_CANDIDATE2_READY or (
                route == ROUTE_COMPLETION_REQUIRED
                and not chains0[state.attribute_id].accepted
            ):
                rank1_states.append(state)
        batch1 = [state.options[1] for state in rank1_states]
        discovery = context["discovery"]
        discovery_job_id = context["discovery_job_id"]
        job1 = (
            self._expected_raw_review_job(
                shard,
                clip_uid,
                owner_plan,
                discovery,
                discovery_job_id,
                rank1_states,
                batch1,
                candidate_rank=1,
            )
            if batch1
            else None
        )
        payload1 = self._committed_payload_or_none(job1) if job1 is not None else None
        reviews1: dict[str, Any] = {}
        if payload1 is not None and str(payload1.get("status")) == "review":
            try:
                batch_model = SubjectAttributeReviewBatch.model_validate(
                    payload1.get("review_batch")
                )
            except Exception as exc:
                raise SubjectAttributeDurableError(
                    "committed rank-1 raw review payload is invalid"
                ) from exc
            reviews1 = {review.attribute_id: review for review in batch_model.reviews}
        elif payload1 is not None and str(payload1.get("status")) != "review_failed":
            raise SubjectAttributeDurableError(
                "committed rank-1 raw review has an unknown payload status"
            )
        routes1: dict[str, str] = {}
        if job1 is not None:
            route_context = {**self._rank_route_context(context), "reviews": reviews1}
            for state in rank1_states:
                routes1[state.attribute_id] = self._rank0_route(
                    state, route_context, has_candidate2=False
                )
        chains1: dict[str, _CompletionChain] = {}
        # Only a committed rank-1 review can route an attribute into a rank-1
        # completion, so nothing is planned speculatively here either.
        if payload1 is not None and str(payload1.get("status")) == "review":
            for state in rank1_states:
                if routes1.get(state.attribute_id) != ROUTE_COMPLETION_REQUIRED:
                    continue
                chains1[state.attribute_id] = self._completion_chain(
                    shard,
                    storage,
                    clip_uid,
                    owner_plan,
                    state.attribute_plan,
                    state.options[1],
                    1,
                )
        bbox: list[tuple[_AttributeReplay, int, Any]] = []
        for state in context["states"]:
            if routes0[state.attribute_id] == ROUTE_BBOX_PENDING:
                bbox.append((state, 0, state.options[0]))
            elif routes1.get(state.attribute_id) == ROUTE_BBOX_PENDING:
                bbox.append((state, 1, state.options[1]))
        return {
            "routes0": routes0,
            "chains0": chains0,
            "rank1_states": rank1_states,
            "batch1": batch1,
            "rank1_job": job1,
            "reviews1": reviews1,
            "routes1": routes1,
            "chains1": chains1,
            "bbox": bbox,
        }

    def _owner_owed_jobs(self, graph: Mapping[str, Any]) -> list[ModelJob]:
        """Every independent model job this owner's graph currently owes.

        Inside one attribute the completion chain is a real dependency chain, so
        only ``chain.next_job`` is reported - the single next stage that the
        stage's own receipts unlock. Across attributes there is no dependency, so
        all of them are returned together instead of one at a time.

        Rank order is preserved: rank-1 work is only reported once every rank-0
        chain has settled, and a rank-1 completion only once the rank-1 review
        that routes it has committed. A last-resort bbox review is deliberately
        left to the legacy replay, because its job identity is bound to the
        owner-context and bbox crop SHAs that the replay derives in context.
        """
        rank0 = [
            chain.next_job
            for chain in graph["chains0"].values()
            if chain.next_job is not None
        ]
        if rank0:
            return sorted(rank0, key=lambda job: job.job_id())
        rank1_review = graph.get("rank1_job")
        if (
            rank1_review is not None
            and self._committed_payload_or_none(rank1_review) is None
        ):
            return [rank1_review]
        rank1 = [
            chain.next_job
            for chain in graph["chains1"].values()
            if chain.next_job is not None
        ]
        return sorted(rank1, key=lambda job: job.job_id())

    def _publish_owner_markers(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        graph: Mapping[str, Any],
    ) -> None:
        """Publish the terminal completion outcomes the graph has settled."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        for chain in list(graph["chains0"].values()) + list(
            graph["chains1"].values()
        ):
            self._publish_completion_outcome(
                shard, storage, clip_uid, owner_entity_id, chain
            )

    def _verify_completion_outcomes(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        graph: Mapping[str, Any],
    ) -> None:
        """Require every settled completion chain to carry its exact marker.

        A processed owner outcome is the last durable authority for this owner, so
        at that point a missing, extra or altered completion marker is durable
        corruption: it is never backfilled, because a restart must not invent a
        marker under an owner that already claims to be terminal.
        """
        for chain in list(graph["chains0"].values()) + list(
            graph["chains1"].values()
        ):
            if not chain.terminal:
                continue
            path = self._completion_outcome_path(
                shard,
                clip_uid,
                str(owner_plan["owner_entity_id"]),
                chain.attribute_id,
                chain.candidate_rank,
            )
            expected = self._completion_outcome_payload(storage, chain)
            if _read_json(path) != expected:
                raise SubjectAttributeDurableError(
                    f"completion outcome drifted for {chain.attribute_id} "
                    f"rank {chain.candidate_rank}"
                )

    def _verified_raw_state(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        owner_plan: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The rank-0 boundary marker, written once and then re-derived exactly.

        The marker is never authority: it is recomputed from the discovery
        receipt, the selection markers, the SAM receipts, the duplicate-mask CPU
        pass and the committed raw review, and any difference fails closed.
        """
        path = self._raw_state_path(
            shard, clip_uid, str(owner_plan["owner_entity_id"])
        )
        expected = self._raw_state_payload(
            clip_uid=clip_uid,
            owner_plan=owner_plan,
            discovery=context["discovery"],
            discovery_job_id=context["discovery_job_id"],
            context=context,
        )
        stored = _read_json(path)
        if stored is None:
            _write_json_once(path, expected)
            return expected
        if stored != expected:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes raw state drifted for {clip_uid}"
            )
        return expected

    # -- 8c: model job execution ----------------------------------------------

    def _completion_job_context(self, job: ModelJob) -> dict[str, Any]:
        """Resolve and re-verify one completion or bbox job."""
        shard = job.canonical_shard
        storage = self._storage_for(shard)
        clip_uid = job.clip_uid
        target = dict(job.target)
        owner_entity_id = str(target.get("owner_entity_id", ""))
        attribute_id = str(target.get("attribute_id", ""))
        rank = int(target.get("candidate_rank", "0"))
        plan = self._clip_plan(shard, clip_uid)
        owner_plan = self._owner_plan_for(shard, plan, owner_entity_id)
        discovery_payload = self._discovery_payload(shard, clip_uid, owner_plan)
        discovery = SubjectAttributeDiscovery.model_validate(
            discovery_payload.get("discovery")
        )
        discovery_job_id = self._expected_discovery_job(
            shard, clip_uid, owner_plan
        ).job_id()
        reference = self._owner_reference(storage, clip_uid, owner_entity_id)
        index = self._attribute_index(
            discovery, attribute_id, int(owner_plan["attribute_id_start"])
        )
        # Model-execution worker: require the plan, never create it. A job that
        # exists was frozen against a plan that must already be durable.
        attribute_plan = self._require_attribute_plan(
            shard,
            storage,
            clip_uid,
            owner_plan,
            reference,
            discovery,
            index,
            attribute_id,
            discovery_job_id,
            ordered=self._ordered_owner_candidates(
                shard, storage, clip_uid, owner_plan, reference
            ),
        )
        state = self._replay_or_cached_selection(
            shard,
            storage,
            clip_uid,
            owner_plan,
            reference,
            discovery,
            index,
            attribute_id,
            discovery_job_id,
        )
        if state.pending_job is not None or rank >= len(state.options):
            raise SubjectAttributeDurableError(
                f"job {job.job_id()} has no frozen candidate "
                f"{attribute_id}/rank-{rank}"
            )
        candidate = state.options[rank]
        if job.job_type == SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB:
            self._require_same_job(
                job,
                self._expected_completion_generate_job(
                    shard,
                    clip_uid,
                    owner_plan,
                    attribute_plan,
                    candidate,
                    rank,
                    self._completion_seed(
                        shard,
                        clip_uid,
                        owner_entity_id,
                        attribute_id,
                        rank,
                        str(candidate.owner_candidate.candidate_id),
                    ),
                ),
            )
        return {
            "storage": storage,
            "clip": storage.read_clip(clip_uid),
            "owner_plan": owner_plan,
            "attribute_plan": attribute_plan,
            "state": state,
            "rank": rank,
            "candidate": candidate,
            "attribute_id": attribute_id,
            "owner_entity_id": owner_entity_id,
        }

    @staticmethod
    def _require_same_job(job: ModelJob, expected: ModelJob) -> None:
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != expected.input_digest
            or job.model_identity != expected.model_identity
        ):
            raise SubjectAttributeDurableError(
                f"job {job.job_id()} is not the frozen job for its target"
            )

    def _run_completion_generate(self, job: ModelJob, handle: Any) -> JobResult:
        """One Boogu completion request. Nothing outside the call is caught."""
        context = self._completion_job_context(job)
        if handle is None:
            raise SubjectAttributeEpochError(
                "subject attribute completion needs a Boogu handle"
            )
        storage = context["storage"]
        candidate = context["candidate"]
        state = context["state"]
        rank = int(context["rank"])
        owner_candidate_id = str(candidate.owner_candidate.candidate_id)
        source_path = self._completion_source_path(
            storage,
            job.clip_uid,
            context["owner_entity_id"],
            context["attribute_id"],
            owner_candidate_id,
        )
        output_path = self._completion_output_path(
            storage,
            job.clip_uid,
            context["owner_entity_id"],
            context["attribute_id"],
            owner_candidate_id,
        )
        expected_source = self._completion_input_bytes(candidate.crop)
        if not source_path.is_file() or source_path.read_bytes() != expected_source:
            _save_completion_input(source_path, candidate.crop)
        adapter = _completion_adapter(
            handle, storage, callable(getattr(handle, "attribute_completion", None))
        )
        started = time.perf_counter()
        # Exactly the legacy failure scope: the backend call, the reported
        # duration validation, the image load and the output read are all inside
        # it, because legacy validates the reported time and loads the generated
        # image before the SAM postcheck ever runs. Source materialisation above
        # stays outside, exactly like legacy's ``_save_completion_input``.
        reported_seconds: float | None = None
        try:
            result = adapter.attribute_completion(
                source_path=source_path,
                output_path=output_path,
                instruction=attribute_completion_prompt(
                    state.attribute_plan["discovered"]["attribute_type"]
                ),
                seed=self._completion_seed(
                    job.canonical_shard,
                    job.clip_uid,
                    context["owner_entity_id"],
                    context["attribute_id"],
                    rank,
                    owner_candidate_id,
                ),
            )
            reported = result.get(
                "model_call_time_seconds", time.perf_counter() - started
            )
            if (
                isinstance(reported, bool)
                or not isinstance(reported, (int, float))
                or not math.isfinite(float(reported))
            ):
                raise ValueError("completion model-call time is invalid")
            reported_seconds = max(0.0, float(reported))
            # Legacy opens and converts the generated image before the postcheck.
            with Image.open(output_path) as opened:
                opened.load()
                opened.convert("RGB")
            data = output_path.read_bytes()
        except Exception as exc:  # noqa: BLE001 - legacy completion failure scope
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "completion_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "model_call_time_seconds": (
                        0.0 if reported_seconds is None else reported_seconds
                    ),
                },
            )
        ledger_path = self._completion_png_path(job.job_id())
        atomic_write_bytes(ledger_path, data)
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "completion",
                "generated": {
                    "path": ledger_path.relative_to(
                        Path(self.ledger.root)
                    ).as_posix(),
                    "sha256": _sha256_bytes(data),
                },
                "generated_png_path": output_path.relative_to(
                    Path(storage.root)
                ).as_posix(),
                "width": int(result.get("width", 0)),
                "height": int(result.get("height", 0)),
                "model_call_time_seconds": (
                    0.0 if reported_seconds is None else reported_seconds
                ),
            },
        )

    def _run_completion_sam(self, job: ModelJob, handle: Any) -> JobResult:
        """One generated-frame segmentation call. Nothing else is caught."""
        context = self._completion_job_context(job)
        if handle is None:
            raise SubjectAttributeEpochError(
                "subject attribute completion SAM needs a handle"
            )
        storage = context["storage"]
        candidate = context["candidate"]
        rank = int(context["rank"])
        owner_candidate_id = str(candidate.owner_candidate.candidate_id)
        generate_job = self._expected_completion_generate_job(
            job.canonical_shard,
            job.clip_uid,
            context["owner_plan"],
            context["attribute_plan"],
            candidate,
            rank,
            self._completion_seed(
                job.canonical_shard,
                job.clip_uid,
                context["owner_entity_id"],
                context["attribute_id"],
                rank,
                owner_candidate_id,
            ),
        )
        payload = self._committed_payload_or_none(generate_job)
        if payload is None:
            raise SubjectAttributeDurableError(
                f"completion SAM {job.job_id()} has no committed generation"
            )
        self._generated_png_bytes(storage, payload)
        output_path = self._completion_output_path(
            storage,
            job.clip_uid,
            context["owner_entity_id"],
            context["attribute_id"],
            owner_candidate_id,
        )
        self._require_same_job(
            job,
            self._expected_completion_sam_job(
                job.canonical_shard,
                job.clip_uid,
                context["owner_plan"],
                context["attribute_plan"],
                candidate,
                rank,
                generate_job,
                str(payload.get("generated", {}).get("sha256")),
                str(payload.get("generated_png_path", "")),
            ),
        )
        started = time.perf_counter()
        try:
            returned = handle.segment_generated_frame(
                frame_path=output_path,
                grounding_prompt=str(
                    context["attribute_plan"]["discovered"]["grounding_prompt"]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - legacy completion failure scope
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "sam_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "model_call_time_seconds": time.perf_counter() - started,
                },
            )
        records = self._publish_masks(job.job_id(), returned)
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "sam",
                "mask_count": len(records),
                "masks": records,
                "model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _run_completion_review(self, job: ModelJob, handle: Any) -> JobResult:
        """One comparative alpha-versus-completion review. Nothing else caught."""
        context = self._completion_job_context(job)
        storage = context["storage"]
        candidate = context["candidate"]
        rank = int(context["rank"])
        owner_candidate_id = str(candidate.owner_candidate.candidate_id)
        generate_job = self._expected_completion_generate_job(
            job.canonical_shard,
            job.clip_uid,
            context["owner_plan"],
            context["attribute_plan"],
            candidate,
            rank,
            self._completion_seed(
                job.canonical_shard,
                job.clip_uid,
                context["owner_entity_id"],
                context["attribute_id"],
                rank,
                owner_candidate_id,
            ),
        )
        payload = self._committed_payload_or_none(generate_job)
        if payload is None:
            raise SubjectAttributeDurableError(
                f"completion review {job.job_id()} has no committed generation"
            )
        generated = self._generated_png_bytes(storage, payload)
        sam_job = self._expected_completion_sam_job(
            job.canonical_shard,
            job.clip_uid,
            context["owner_plan"],
            context["attribute_plan"],
            candidate,
            rank,
            generate_job,
            str(payload.get("generated", {}).get("sha256")),
            str(payload.get("generated_png_path", "")),
        )
        sam_payload = self._committed_payload_or_none(sam_job)
        if sam_payload is None:
            raise SubjectAttributeDurableError(
                f"completion review {job.job_id()} has no committed SAM"
            )
        cpu = self._completion_cpu_result(
            generated, self._load_sam_masks(sam_payload)
        )
        if cpu["status"] != COMPLETION_REVIEW_REQUIRED:
            raise SubjectAttributeDurableError(
                f"completion review {job.job_id()} has a rejected postcheck"
            )
        completed_rgba = cpu["completed_rgba"]
        discovered = context["attribute_plan"]["discovered"]
        self._require_same_job(
            job,
            self._expected_completion_review_job(
                job.canonical_shard,
                job.clip_uid,
                context["owner_plan"],
                context["attribute_plan"],
                candidate,
                rank,
                sam_job,
                _image_png_sha256(candidate.crop.convert("RGBA")),
                _image_png_sha256(
                    _attribute_bbox_crop(
                        candidate.source_image,
                        candidate.attribute_mask,
                        crop_padding_ratio=float(
                            self.config.pair.crop_padding_ratio
                        ),
                    )
                ),
                _image_png_sha256(completed_rgba.convert("RGBA")),
            ),
        )
        resolved = resolve_subject_attribute_qwen_client(
            handle, self.config, judge=True
        )
        started = time.perf_counter()
        try:
            review = resolved.client.review(
                source_attribute=candidate.crop,
                source_bbox=_attribute_bbox_crop(
                    candidate.source_image,
                    candidate.attribute_mask,
                    crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
                ),
                generated_candidate=completed_rgba,
                attribute_type=discovered["attribute_type"],
                attribute_phrase=discovered["phrase"],
            )
        except Exception as exc:  # noqa: BLE001 - legacy completion failure scope
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "review_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "qwen_model_call_time_seconds": time.perf_counter() - started,
                },
            )
        finally:
            _close_owned(resolved)
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "review",
                "review": review.model_dump(mode="json"),
                "qwen_model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _run_bbox_review(self, job: ModelJob, handle: Any) -> JobResult:
        """One last-resort bbox review call. Nothing else is caught."""
        context = self._completion_job_context(job)
        candidate = context["candidate"]
        rank = int(context["rank"])
        discovered = context["attribute_plan"]["discovered"]
        bbox_image = _attribute_bbox_crop(
            candidate.source_image,
            candidate.attribute_mask,
            crop_padding_ratio=float(self.config.pair.crop_padding_ratio),
        )
        owner_context = build_candidate_context_image(
            candidate.source_image, candidate.owner_candidate.mask
        )
        self._require_same_job(
            job,
            self._expected_bbox_review_job(
                job.canonical_shard,
                job.clip_uid,
                context["owner_plan"],
                context["attribute_plan"],
                candidate,
                rank,
                _image_png_sha256(_resize_qwen_input_image(owner_context)),
                _image_png_sha256(bbox_image),
            ),
        )
        resolved = resolve_subject_attribute_qwen_client(handle, self.config)
        started = time.perf_counter()
        try:
            review = resolved.client.review_attribute_bbox(
                bbox_candidate=bbox_image,
                owner_context=owner_context,
                attribute_type=discovered["attribute_type"],
                attribute_phrase=discovered["phrase"],
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, one job per attribute
            return JobResult(
                OUTCOME_COMPLETED,
                payload={
                    "status": "bbox_failed",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "qwen_model_call_time_seconds": time.perf_counter() - started,
                },
            )
        finally:
            _close_owned(resolved)
        return JobResult(
            OUTCOME_COMPLETED,
            payload={
                "status": "bbox",
                "bbox_review": review.model_dump(mode="json"),
                "qwen_model_call_time_seconds": time.perf_counter() - started,
            },
        )

    def _publish_masks(self, job_id: str, masks: Sequence[Any]) -> list[dict[str, Any]]:
        """Store returned masks as durable ledger ``.npy`` artifacts."""
        records: list[dict[str, Any]] = []
        for index, mask in enumerate(masks):
            array = np.asarray(mask)
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            data = buffer.getvalue()
            path = self._sam_mask_path(job_id, index)
            atomic_write_bytes(path, data)
            records.append(
                {
                    "path": path.relative_to(Path(self.ledger.root)).as_posix(),
                    "sha256": _sha256_bytes(data),
                    "dtype": str(array.dtype),
                    "shape": [int(value) for value in array.shape],
                }
            )
        return records

    # -- stats -----------------------------------------------------------------

    def _verified_clip_outcome(
        self,
        shard: str,
        storage: RunStorage,
        clip_uid: str,
        plan: Mapping[str, Any],
        marker: Mapping[str, Any],
    ) -> dict[str, Any]:
        if marker.get("schema") != SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA:
            raise SubjectAttributeDurableError(
                f"clip outcome schema drifted for {clip_uid!r}"
            )
        if marker.get("clip_uid") != clip_uid:
            raise SubjectAttributeDurableError(
                f"clip outcome clip_uid drifted for {clip_uid!r}"
            )
        if marker.get("terminal") != "ready":
            raise SubjectAttributeDurableError(
                f"clip outcome terminal is unknown for {clip_uid!r}"
            )
        counts = marker.get("counts")
        if not isinstance(counts, dict) or not counts:
            raise SubjectAttributeDurableError(
                f"clip outcome counts are missing for {clip_uid!r}"
            )
        expected, sample = self._clip_outcome_payload(
            shard, storage, clip_uid, plan
        )
        if marker != expected:
            raise SubjectAttributeDurableError(
                f"clip outcome drifted for {clip_uid!r}"
            )
        sample_path = expected["enriched_sample_path"]
        if sample_path is None:
            if expected["enriched_sample_sha256"] is not None:
                raise SubjectAttributeDurableError(
                    f"clip outcome carries a sample digest without a sample for "
                    f"{clip_uid!r}"
                )
            return marker
        if sample is None:
            raise SubjectAttributeDurableError(
                f"clip outcome expects an enriched sample for {clip_uid!r}"
            )
        target = storage.root / str(sample_path)
        if not target.is_file():
            raise SubjectAttributeDurableError(
                f"enriched sample is missing for {clip_uid!r}"
            )
        expected_bytes = _expected_artifact_bytes(sample)
        if target.read_bytes() != expected_bytes:
            raise SubjectAttributeDurableError(
                f"enriched sample drifted for {clip_uid!r}"
            )
        if _sha256_bytes(expected_bytes) != expected["enriched_sample_sha256"]:
            raise SubjectAttributeDurableError(
                f"enriched sample digest drifted for {clip_uid!r}"
            )
        return marker

    def reconcile_stats(self, shard: str) -> SubjectAttributeEpochStats:
        """Legacy clip counts rebuilt from the frozen plan and durable outcomes."""
        storage = self._storage_for(shard)
        counts: Counter[str] = Counter()
        terminal_clips = 0
        no_work_clips = 0
        for clip_uid in self.eligible.get(shard, ()):
            plan = self._clip_plan(shard, clip_uid)
            marker = _read_json(self._clip_outcome_path(shard, clip_uid))
            if marker is None:
                if plan["classification"] != CLIP_NO_WORK:
                    # An eligible owner is still waiting (a human owner defers to
                    # 8b), so the stage is genuinely incomplete.
                    self._clip_totals(shard, storage, clip_uid, plan)
                raise SubjectAttributeEpochError(
                    f"Subject Attributes clip {clip_uid!r} has no durable outcome; "
                    "the stage is not terminal"
                )
            verified = self._verified_clip_outcome(
                shard, storage, clip_uid, plan, marker
            )
            for key, value in verified["counts"].items():
                counts[key] += value
            terminal_clips += 1
            no_work_clips += int(plan["classification"] == CLIP_NO_WORK)
        return SubjectAttributeEpochStats(
            counts=dict(counts),
            terminal_clips=terminal_clips,
            no_work_clips=no_work_clips,
        )


def _expected_artifact_bytes(model: Any) -> bytes:
    """The exact bytes ``write_json_atomic`` produces for one model."""
    return (
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")


def _sam_failure_exception(payload: Mapping[str, Any]) -> Exception:
    """Rebuild the legacy probe exception type and message from a receipt."""
    name = str(payload.get("exception_type", "Exception"))
    exception_type = type(name, (Exception,), {})
    return exception_type(str(payload.get("error", "")))


def _image_png_sha256(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _sha256_bytes(buffer.getvalue())


def _committed_payload(ledger: Any, job: ModelJob) -> dict[str, Any]:
    """The committed receipt payload of one discovery job, or fail closed."""
    from r2v_data_v2.v3.post_mask_epoch_state import STATE_MISMATCH

    owner_entity_id = str(dict(job.target).get("owner_entity_id", ""))
    state = ledger.classify(job)
    if state.state == STATE_MISMATCH:
        raise SubjectAttributeDurableError(
            f"discovery receipt mismatch for {job.clip_uid}/{owner_entity_id}: "
            f"{state.detail or ''}"
        )
    if not state.skippable:
        raise SubjectAttributeEpochError(
            f"discovery job {job.job_id()} is not terminal; the stage is incomplete"
        )
    result = ledger.load_committed_result(job)
    if result is None:
        raise SubjectAttributeDurableError(
            f"committed discovery job {job.job_id()} has no replayable result"
        )
    return dict(result.payload)


__all__ = [
    "CLIP_NO_WORK",
    "CLIP_OWNERS",
    "OWNER_SOURCE_DISCOVERY",
    "OWNER_SOURCE_PREEXISTING",
    "SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA",
    "SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA",
    "SUBJECT_ATTRIBUTE_DISCOVERY_JOB",
    "SUBJECT_ATTRIBUTE_DISCOVERY_POLICY_VERSION",
    "SUBJECT_ATTRIBUTE_OUTPUT_DIRNAME",
    "SUBJECT_ATTRIBUTE_OWNER_OUTCOME_SCHEMA",
    "SUBJECT_ATTRIBUTE_OWNER_PLAN_SCHEMA",
    "SUBJECT_ATTRIBUTE_POLICY_VERSION",
    "ResolvedDiscoveryClient",
    "SubjectAttributeDurableError",
    "SubjectAttributeEpochError",
    "SubjectAttributeEpochRunner",
    "SubjectAttributeEpochStats",
    "resolve_subject_attribute_discovery_client",
]
