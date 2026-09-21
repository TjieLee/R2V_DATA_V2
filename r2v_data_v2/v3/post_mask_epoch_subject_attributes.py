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
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pair import build_entity_reference_candidates
from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    OUTCOME_RETRYABLE_FAILED,
    RESOURCE_QWEN,
    JobResult,
    ModelJob,
    canonical_json,
)
from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_bytes, atomic_write_json
from r2v_data_v2.v3.storage import RunStorage, evaluate_export_state
from r2v_data_v2.v3.subject_attributes import (
    DISCOVERY_SYSTEM_PROMPT,
    MAX_ATTRIBUTE_SOURCE_CANDIDATES,
    MAX_ATTRIBUTES_PER_OWNER,
    QWEN_INPUT_MAX_LONG_SIDE_PIXELS,
    REVIEW_SYSTEM_PROMPT,
    ClipEnrichmentResult,
    DiscoveredSubjectAttribute,
    EnrichmentTotals,
    OwnerEnrichmentArtifact,
    OwnerEnrichmentMetrics,
    QwenSubjectAttributeClient,
    SubjectAttributeDiscovery,
    SubjectAttributeRecord,
    SubjectAttributeReviewBatch,
    _build_enriched_sample,
    _clip_sample_path,
    _collect_attribute_candidates,
    _duplicate_attribute_mask_conflicts,
    _load_cached_owner_artifact,
    _owner_artifact_path,
    _rejected_record,
    _resize_qwen_input_image,
    _source_image,
    build_candidate_context_image,
    evaluate_owner_eligibility,
    prefer_attribute_candidate_frames,
)

SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA = (
    "post_mask_epoch_subject_attributes_clip_plan/2"
)
SUBJECT_ATTRIBUTE_OWNER_PLAN_SCHEMA = (
    "post_mask_epoch_subject_attributes_owner_plan/2"
)
SUBJECT_ATTRIBUTE_OWNER_OUTCOME_SCHEMA = (
    "post_mask_epoch_subject_attributes_owner_outcome/1"
)
SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA = (
    "post_mask_epoch_subject_attributes_clip_outcome/1"
)
SUBJECT_ATTRIBUTE_POLICY_VERSION = "subject_attributes_epoch_policy/2"
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

#: The only model job this epoch owns. Everything else in Subject Attributes is
#: still legacy-only.
SUBJECT_ATTRIBUTE_DISCOVERY_JOB = "subject_attribute_discovery"
#: One real legacy ``segment_frame`` call is one job, so a single job never pays
#: more than one SAM call.
SUBJECT_ATTRIBUTE_SAM_PROBE_JOB = "subject_attribute_sam_probe"
#: One owner-batched rank-0 raw review call.
SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB = "subject_attribute_raw_review"
RESOURCE_SAM = "sam"

CLIP_NO_WORK = "no_work"
CLIP_OWNERS = "owners"

OWNER_SOURCE_PREEXISTING = "preexisting"
OWNER_SOURCE_DISCOVERY = "discovery"

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


class _PendingSamProbe(Exception):
    """Internal marker: the legacy selection reached an unexecuted SAM probe."""

    def __init__(self, probe_index: int, owner_candidate_id: str) -> None:
        super().__init__(f"SAM probe {probe_index} is pending")
        self.probe_index = probe_index
        self.owner_candidate_id = owner_candidate_id


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


class _ReplaySegmentationBackend:
    """Legacy ``AttributeFrameSegmentationBackend`` backed by SAM receipts.

    Committed probes replay their exact ledger masks; a committed ``sam_failed``
    probe re-raises an exception of the same type and message so the legacy
    ``probe_failures`` reason stays identical; the first probe without a receipt
    is reported through ``_PendingSamProbe`` and recorded as this attribute's
    next job.
    """

    def __init__(
        self,
        *,
        epoch: SubjectAttributeEpochRunner,
        ordered: Sequence[Any],
        receipts: Mapping[int, Mapping[str, Any]],
        storage: RunStorage,
        grounding_prompt: str,
    ) -> None:
        self._epoch = epoch
        self._ordered = list(ordered)
        self._receipts = dict(receipts)
        self._storage = storage
        self._grounding_prompt = grounding_prompt
        self._by_slot: dict[int, int] = {}
        for index, candidate in enumerate(self._ordered):
            self._by_slot[int(candidate.frame_slot)] = index
        self.pending: tuple[int, str] | None = None

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
            if self.pending is None:
                self.pending = (index, str(candidate.candidate_id))
            raise _PendingSamProbe(index, str(candidate.candidate_id))
        if str(payload.get("status")) == "sam_failed":
            raise _sam_failure_exception(payload)
        return self._epoch._load_sam_masks(payload)


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
    if isinstance(handle, str):
        service = config.qwen.candidate_judge
        return ResolvedDiscoveryClient(
            QwenSubjectAttributeClient(replace(service, base_url=handle)),
            owned=True,
        )
    return ResolvedDiscoveryClient(handle, owned=False)


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
    ) -> None:
        if bool(getattr(config.subject_attribute_gme, "enabled", False)):
            raise SubjectAttributeEpochError(
                "Post-Mask Resource Epoch requires subject attribute GME disabled"
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

    def _clip_plan(self, shard: str, clip_uid: str) -> dict[str, Any]:
        """Create-once freeze of the clip plan, re-derived on every read."""
        storage = self._storage_for(shard)
        existing = _read_json(self._clip_plan_path(shard, clip_uid))
        try:
            expected = self._derive_clip_plan(storage, clip_uid)
        except SubjectAttributeDurableError:
            raise
        except Exception as exc:
            raise SubjectAttributeDurableError(
                f"cannot derive the Subject Attributes clip plan for {clip_uid!r}"
            ) from exc
        if existing is None:
            _write_json_once(self._clip_plan_path(shard, clip_uid), expected)
            return expected
        if existing != expected:
            # Whole-plan equality: an extra, missing or changed field is drift.
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes clip plan drifted for {clip_uid!r}"
            )
        return existing

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

    def _owner_candidate_objects(
        self, shard: str, clip_uid: str, owner_plan: Mapping[str, Any]
    ) -> list[Any]:
        """Rebuild the live candidate objects the discovery job was frozen on."""
        storage = self._storage_for(shard)
        owner_entity_id = str(owner_plan["owner_entity_id"])
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
                f"owner {clip_uid}/{owner_entity_id} is missing from the annotation"
            )
        frames = storage.read_frames(clip_uid)
        masks = storage.read_masks(clip_uid)
        candidates = self._owner_candidates(storage, clip_uid, owner, frames, masks)
        descriptors = [_candidate_descriptor(storage, item) for item in candidates]
        frozen = [dict(item) for item in owner_plan.get("candidates", [])]
        if descriptors != frozen:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes candidate evidence drifted for "
                f"{clip_uid}/{owner_entity_id}"
            )
        return candidates

    # -- run -------------------------------------------------------------------

    def run(self, job: ModelJob, handle: Any) -> JobResult:
        """Execute one model job. No prompt or schema is copied."""
        if job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            return self._run_sam_probe(job, handle)
        if job.job_type == SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB:
            return self._run_raw_review(job, handle)
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
                discovery = SubjectAttributeDiscovery.model_validate(
                    discovery_payload.get("discovery")
                )
                if discovery.owner_is_human and discovery.attributes:
                    return self._advance_attributes(
                        shard, storage, clip_uid, owner_entity_id
                    )
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

    def seed_jobs(self) -> list[ModelJob]:
        """CPU fixed point, then the deterministic pending discovery jobs."""
        jobs: list[ModelJob] = []
        for shard in sorted(self.storages):
            storage = self._storage_for(shard)
            for clip_uid in self.eligible.get(shard, ()):
                jobs.extend(self._advance_clip(shard, storage, clip_uid))
        return sorted(jobs, key=lambda job: job.job_id())

    def finalize(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """CPU continuation of one committed model receipt."""
        if job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB:
            return self._finalize_sam_probe(job, result)
        if job.job_type == SUBJECT_ATTRIBUTE_RAW_REVIEW_JOB:
            return self._finalize_raw_review(job, result)
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
    ) -> list[Any]:
        """The owner's candidates in the exact legacy attribute-probe order."""
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
    ) -> dict[str, Any]:
        """Create-once attribute plan, re-derived on every read."""
        owner_entity_id = str(owner_plan["owner_entity_id"])
        path = self._attribute_plan_path(shard, clip_uid, owner_entity_id, attribute_id)
        existing = _read_json(path)
        discovered = discovery.attributes[attribute_index]
        ordered = self._ordered_owner_candidates(
            shard, storage, clip_uid, owner_plan, owner_reference
        )
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
        if existing is None:
            _write_json_once(path, expected)
            return expected
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
        attribute_plan = self._attribute_plan(
            shard,
            storage,
            clip_uid,
            owner_plan,
            owner_reference,
            discovery,
            self._attribute_index(discovery, attribute_id),
            attribute_id,
            self._expected_discovery_job(shard, clip_uid, owner_plan).job_id(),
        )
        ordered = self._ordered_owner_candidates(
            shard, storage, clip_uid, owner_plan, owner_reference
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
        discovery: SubjectAttributeDiscovery, attribute_id: str
    ) -> int:
        try:
            return int(attribute_id[1:]) - 1
        except ValueError as exc:
            raise SubjectAttributeDurableError(
                f"attribute id {attribute_id!r} is malformed"
            ) from exc

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
        """Replay the legacy selection CPU policy over committed SAM receipts.

        The only SAM call the replay may not replay is the first probe that has
        no committed receipt: that probe is the next job, and the legacy helper
        is re-run from scratch once its receipt exists.
        """
        owner_entity_id = str(owner_plan["owner_entity_id"])
        attribute_plan = self._attribute_plan(
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
        ordered = self._ordered_owner_candidates(
            shard, storage, clip_uid, owner_plan, owner_reference
        )
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
                owner_candidates=self._owner_candidate_objects(
                    shard, clip_uid, owner_plan
                ),
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
        except Exception as exc:  # noqa: BLE001 - legacy attribute isolation
            processing_failure = 1
            selected = _rejected_record(
                discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner_entity_id,
                reason=f"attribute_processing_failed:{type(exc).__name__}:{exc}",
            )
        if backend.pending is not None:
            # The replay stopped at a probe that has no receipt yet: that probe
            # is the next job and everything after it is not decided.
            pending_index = backend.pending[0]
            return _AttributeReplay(
                attribute_id=attribute_id,
                attribute_type=str(attribute_plan["discovered"]["attribute_type"]),
                attribute_plan=attribute_plan,
                marker=None,
                options=(),
                record=None,
                processing_failure=0,
                pending_job=probe_jobs[pending_index],
                sam_job_ids={},
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
        """Create-once terminal selection marker, re-derived on every read."""
        replay = self._replay_attribute_selection(
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
        stored = _read_json(
            self._selection_path(
                shard, clip_uid, str(owner_plan["owner_entity_id"]), attribute_id
            )
        )
        if replay.pending_job is not None:
            if stored is not None:
                raise SubjectAttributeDurableError(
                    f"attribute {clip_uid}/{attribute_id} has a terminal selection "
                    "marker but its SAM probes are incomplete"
                )
            return replay
        if stored is None:
            _write_json_once(
                self._selection_path(
                    shard,
                    clip_uid,
                    str(owner_plan["owner_entity_id"]),
                    attribute_id,
                ),
                replay.marker,
            )
        elif stored != replay.marker:
            raise SubjectAttributeDurableError(
                f"frozen Subject Attributes selection marker drifted for "
                f"{clip_uid}/{attribute_id}"
            )
        return replay

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
        states = [
            self._attribute_selection(
                shard,
                storage,
                clip_uid,
                owner_plan,
                owner_reference,
                discovery,
                index,
                f"a{int(owner_plan['attribute_id_start']) + index}",
                discovery_job_id,
            )
            for index in range(len(discovery.attributes))
        ]
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
            "candidate_rank": "0",
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
            ),
            model_identity=f"qwen:{self.config.qwen.candidate_judge.model}",
            target={
                "owner_entity_id": str(owner_plan["owner_entity_id"]),
                "candidate_rank": "0",
            },
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
        started = time.perf_counter()
        try:
            batch = handle.review(owner=owner, candidates=list(context["batch"]))
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
        if str(dict(job.target).get("candidate_rank")) != "0":
            raise SubjectAttributeEpochError(
                f"unsupported raw review rank {dict(job.target).get('candidate_rank')!r}"
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
        if context["job"] is None:
            raise SubjectAttributeDurableError(
                f"raw review job {job.job_id()} has no rank-0 batch"
            )
        expected = context["job"]
        if (
            expected.job_id() != job.job_id()
            or job.input_digest != expected.input_digest
            or job.model_identity != expected.model_identity
        ):
            raise SubjectAttributeDurableError(
                f"raw review job {job.job_id()} is not the frozen rank-0 job for "
                f"{clip_uid}/{owner_entity_id}"
            )
        return {
            "storage": storage,
            "clip": storage.read_clip(clip_uid),
            "owner_entity_id": owner_entity_id,
            "owner_plan": owner_plan,
            "discovery": discovery,
            "discovery_job_id": discovery_job_id,
            "batch": context["batch"],
            "context": context,
        }

    # -- rank-0 routes ---------------------------------------------------------

    @staticmethod
    def _rank0_route(
        state: _AttributeReplay, context: Mapping[str, Any]
    ) -> str:
        """The exact legacy rank-0 route of one attribute."""
        review = context["reviews"].get(state.attribute_id)
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

    def _advance_attributes(
        self, shard: str, storage: RunStorage, clip_uid: str, owner_entity_id: str
    ) -> list[ModelJob]:
        """Advance one human owner's attribute chain as far as durable state allows."""
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
            return sorted(context["pending"], key=lambda job: job.job_id())
        path = self._raw_state_path(shard, clip_uid, owner_entity_id)
        if path.is_file():
            # 8b ends here: 8c completes the raw routes and publishes the owner.
            return []
        if context["job"] is None:
            _write_json_once(
                path,
                self._raw_state_payload(
                    clip_uid=clip_uid,
                    owner_plan=owner_plan,
                    discovery=discovery,
                    discovery_job_id=discovery_job_id,
                    context=context,
                ),
            )
            return []
        return [context["job"]]

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

    def _finalize_raw_review(self, job: ModelJob, result: JobResult) -> Sequence[ModelJob]:
        """CPU continuation of one committed rank-0 raw review."""
        payload = dict(result.payload)
        status = str(payload.get("status", ""))
        if status not in {"review", "review_failed"}:
            raise SubjectAttributeDurableError(
                f"committed raw review {job.job_id()} has an unknown payload status"
            )
        if status == "review":
            try:
                SubjectAttributeReviewBatch.model_validate(payload.get("review_batch"))
            except Exception as exc:
                raise SubjectAttributeDurableError(
                    f"committed raw review {job.job_id()} has an invalid batch"
                ) from exc
        context = self._raw_review_context(job)
        raw_context = context["context"]
        _write_json_once(
            self._raw_state_path(
                job.canonical_shard, job.clip_uid, context["owner_entity_id"]
            ),
            self._raw_state_payload(
                clip_uid=job.clip_uid,
                owner_plan=context["owner_plan"],
                discovery=context["discovery"],
                discovery_job_id=context["discovery_job_id"],
                context=raw_context,
            ),
        )
        # Every rank-0 route still needs 8c (completion, candidate 2, bbox or the
        # final human owner artifact), so 8b unlocks nothing further.
        return ()

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
