from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    model_validator,
)

import r2v_data_v2.v3.config as v3_config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.boogu_seed import new_boogu_seed
from r2v_data_v2.v3.config import (
    QwenServiceConfig,
    Sam3Config,
    SubjectAttributeCompletionConfig,
    V3Config,
)
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.pair import (
    _masked_sharpness_score,
    EntityReferenceCandidate,
    build_candidate_context_image,
    build_entity_reference_candidates,
    build_reference_crop,
    mask_component_diagnostics,
)
from r2v_data_v2.v3.profiling import (
    model_profile_context,
    profile_model_call,
    profiled_openai_call,
)
from r2v_data_v2.v3.reference_edit_boogu import (
    QwenBooguReferenceEditJudge,
)
from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend, mask_iou
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    ClipRecord,
    EntityReferenceState,
    ReferenceDefaultVariant,
    ReferenceVariantState,
    ReferenceVariantsState,
    RunRecord,
    TrackedMasksArtifact,
    render_annotation_plain_text,
)
from r2v_data_v2.v3.storage import RunStorage
from r2v_data_v2.v3.subject_attribute_gme import GmeRelativeMarginResult

ATTRIBUTE_ENRICHMENT_SCHEMA_VERSION = "r2v.v3.subject_attributes.1"
ATTRIBUTE_OWNER_SCHEMA_VERSION = "r2v.v3.subject_attribute_owner.1"
ENRICHED_SAMPLE_SCHEMA_VERSION = "r2v.v3.enriched_sample.1"
MAX_ATTRIBUTES_PER_OWNER = 3
MAX_ATTRIBUTE_SOURCE_CANDIDATES = 2
QWEN_INPUT_MAX_LONG_SIDE_PIXELS = 768
MIN_ATTRIBUTE_AREA_PIXELS = 16
MIN_ATTRIBUTE_LONG_SIDE_PIXELS = 4
HAIR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS = 192
HEADWEAR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS = 128
MAX_ATTRIBUTE_TO_OWNER_AREA_RATIO = 0.85
WRONG_OWNER_MINIMUM_OVERLAP_RATIO = 0.50
CLOTHING_OWNER_LIKE_AREA_RATIO = 0.78
CLOTHING_STRIP_ASPECT_RATIO = 3.0
ATTRIBUTE_DUPLICATE_MASK_IOU = 0.90
CLOTHING_ATTRIBUTE_TYPES = frozenset(
    {"upper_clothing", "lower_clothing", "dress_or_skirt"}
)
_ATTRIBUTE_COMPLETION_ENTITY_NAME = {
    "face": "人脸",
    "headwear": "帽子",
    "accessory": "配饰",
    "upper_clothing": "衣服",
    "lower_clothing": "下装",
    "dress_or_skirt": "裙子",
}

_IMAGE_PLACEHOLDER = re.compile(r"\{\{image_([1-9]\d*)\}\}")
_ANGLE_IMAGE_LABEL = re.compile(r"<Image ([1-9]\d*)>")
_ANY_REF_TOKEN = re.compile(r"<ref_[^>]+>")


AttributeType = Literal[
    "face",
    "hair",
    "headwear",
    "glasses",
    "upper_clothing",
    "lower_clothing",
    "dress_or_skirt",
    "shoes",
    "bag",
    "accessory",
]


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DiscoveredSubjectAttribute(_SchemaModel):
    attribute_type: AttributeType
    phrase: str
    grounding_prompt: str

    @model_validator(mode="after")
    def validate_text(self) -> DiscoveredSubjectAttribute:
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("discovered attribute text must not be empty")
        if _ANY_REF_TOKEN.search(self.phrase) or _ANY_REF_TOKEN.search(
            self.grounding_prompt
        ):
            raise ValueError("discovered attribute text must not contain ref tokens")
        return self


class SubjectAttributeDiscovery(_SchemaModel):
    owner_entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    owner_is_human: StrictBool
    attributes: list[DiscoveredSubjectAttribute] = Field(
        default_factory=list,
        max_length=MAX_ATTRIBUTES_PER_OWNER,
    )

    @model_validator(mode="after")
    def validate_attributes(self) -> SubjectAttributeDiscovery:
        if not self.owner_is_human and self.attributes:
            raise ValueError("non-human subjects must not publish attributes")
        kinds = [attribute.attribute_type for attribute in self.attributes]
        if len(kinds) != len(set(kinds)):
            raise ValueError("discovery must not repeat an attribute type")
        return self


class SubjectAttributeReview(_SchemaModel):
    attribute_id: str = Field(pattern=r"^a[1-9]\d*$")
    matches_attribute: StrictBool
    owner_binding_correct: StrictBool
    recognizable: StrictBool
    characteristic_appearance_visible: StrictBool
    usable_as_attribute_condition: StrictBool
    sufficient_source_evidence: StrictBool
    structure_complete: StrictBool
    completion_recommended: StrictBool
    reason: str

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.matches_attribute,
                self.owner_binding_correct,
                self.recognizable,
                self.characteristic_appearance_visible,
                self.usable_as_attribute_condition,
                self.sufficient_source_evidence,
            )
        )

    @model_validator(mode="after")
    def validate_reason(self) -> SubjectAttributeReview:
        if not self.reason.strip():
            raise ValueError("attribute review reason must not be empty")
        return self


class SubjectAttributeReviewBatch(_SchemaModel):
    owner_entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    reviews: list[SubjectAttributeReview]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> SubjectAttributeReviewBatch:
        ids = [review.attribute_id for review in self.reviews]
        if len(ids) != len(set(ids)):
            raise ValueError("attribute review IDs must be unique")
        return self


class SubjectAttributeCompletionReview(_SchemaModel):
    """Comparative decision for replacing raw alpha with a completion."""

    same_physical_attribute: StrictBool
    original_visible_details_preserved: StrictBool
    no_wrong_new_instance: StrictBool
    no_duplicate_component: StrictBool
    no_unrelated_content: StrictBool
    no_structural_distortion: StrictBool
    target_clear_and_prominent: StrictBool
    candidate_better_than_alpha: StrictBool
    certain: StrictBool
    reason: str
    verdict: Literal["accept", "reject"]

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_review(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "same_physical_attribute" in value:
            if "candidate_better_than_alpha" in value:
                return value
            legacy_better = value.get("candidate_preferred_over_alpha")
            if not isinstance(legacy_better, bool):
                legacy_better = value.get("materially_more_complete")
            if not isinstance(legacy_better, bool):
                return value
            return {
                key: item
                for key, item in {
                    **value,
                    "candidate_better_than_alpha": legacy_better,
                }.items()
                if key
                not in {"materially_more_complete", "candidate_preferred_over_alpha"}
            }
        legacy_flags = {
            "same_physical_entity",
            "identity_preserved",
            "original_visible_attributes_preserved",
            "exactly_one_entity",
            "missing_parts_plausibly_completed",
            "no_duplicate_entity",
            "no_unrelated_entity",
            "no_severe_structure_artifact",
            "style_coherent",
            "resolution_usable",
            "reference_usable",
            "certain",
        }
        if not legacy_flags.issubset(value):
            return value
        return {
            "same_physical_attribute": value["same_physical_entity"],
            "original_visible_details_preserved": all(
                (
                    value["identity_preserved"],
                    value["original_visible_attributes_preserved"],
                    value["style_coherent"],
                )
            ),
            "no_wrong_new_instance": all(
                (
                    value["same_physical_entity"],
                    value["identity_preserved"],
                    value["exactly_one_entity"],
                )
            ),
            "no_duplicate_component": value["no_duplicate_entity"],
            "no_unrelated_content": value["no_unrelated_entity"],
            "no_structural_distortion": value[
                "no_severe_structure_artifact"
            ],
            "target_clear_and_prominent": all(
                (value["resolution_usable"], value["reference_usable"])
            ),
            "candidate_better_than_alpha": value[
                "missing_parts_plausibly_completed"
            ],
            "certain": value["certain"],
            "reason": value.get("reason"),
            "verdict": value.get("verdict"),
        }

    @model_validator(mode="after")
    def validate_review(self) -> SubjectAttributeCompletionReview:
        if not self.reason.strip():
            raise ValueError("attribute completion review reason must not be empty")
        flags = tuple(
            getattr(self, name)
            for name in type(self).model_fields
            if name not in {"reason", "verdict"}
        )
        if self.verdict != ("accept" if all(flags) else "reject"):
            raise ValueError("attribute completion verdict must match strict checks")
        return self


class SubjectAttributeBboxReview(_SchemaModel):
    correct_attribute: StrictBool
    owner_binding_correct: StrictBool
    target_is_dominant_and_identifiable: StrictBool
    no_large_competing_attribute_or_entity: StrictBool
    context_is_limited_and_supportive: StrictBool
    no_strong_owner_pose_or_scene_leakage: StrictBool
    no_severe_blur_or_artifact: StrictBool
    usable_as_attribute_condition: StrictBool
    certain: StrictBool
    reason: str
    verdict: Literal["accept", "reject"]

    @model_validator(mode="after")
    def validate_review(self) -> SubjectAttributeBboxReview:
        if not self.reason.strip():
            raise ValueError("attribute bbox review reason must not be empty")
        passed = all(
            (
                self.correct_attribute,
                self.owner_binding_correct,
                self.target_is_dominant_and_identifiable,
                self.no_large_competing_attribute_or_entity,
                self.context_is_limited_and_supportive,
                self.no_strong_owner_pose_or_scene_leakage,
                self.no_severe_blur_or_artifact,
                self.usable_as_attribute_condition,
                self.certain,
            )
        )
        if self.verdict != ("accept" if passed else "reject"):
            raise ValueError("attribute bbox verdict must match strict checks")
        return self


class OwnershipGeometry(_SchemaModel):
    passed: bool
    reason: str
    owner_overlap_ratio: float = Field(ge=0, le=1)
    maximum_other_owner_overlap_ratio: float = Field(ge=0, le=1)
    attribute_to_owner_area_ratio: float = Field(ge=0)
    near_owner_region: bool
    attribute_area_pixels: int = Field(ge=0)
    attribute_long_side_pixels: int = Field(ge=0)
    significant_component_count: int = Field(ge=0)
    largest_component_ratio: float = Field(ge=0, le=1)
    second_largest_component_ratio: float = Field(ge=0, le=1)
    bbox_fill_ratio: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_reason(self) -> OwnershipGeometry:
        if not self.reason.strip():
            raise ValueError("ownership geometry reason must not be empty")
        return self


class GmeAttributeScreenAttempt(_SchemaModel):
    owner_candidate_id: str
    source_frame_slot: int = Field(ge=0, lt=10)
    source_frame_index: int = Field(ge=0)
    bbox_fill_ratio: float = Field(ge=0, le=1)
    status: Literal["passed", "rejected", "failed"]
    positive_score: float | None = None
    negative_scores: dict[str, float] = Field(default_factory=dict)
    max_negative_score: float | None = None
    margin: float | None = None
    passed: bool | None = None
    reason: str

    @model_validator(mode="after")
    def validate_attempt(self) -> GmeAttributeScreenAttempt:
        if not self.owner_candidate_id.strip() or not self.reason.strip():
            raise ValueError("GME attempt provenance and reason must not be empty")
        scores = (
            self.positive_score,
            self.max_negative_score,
            self.margin,
        )
        if self.status == "failed":
            if any(value is not None for value in scores) or self.negative_scores:
                raise ValueError("failed GME attempt cannot publish scores")
            if self.passed is not None:
                raise ValueError("failed GME attempt cannot publish passed")
            return self
        if any(value is None for value in scores) or not self.negative_scores:
            raise ValueError("completed GME attempt requires all scores")
        numeric = [
            *[float(value) for value in scores if value is not None],
            *self.negative_scores.values(),
        ]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("GME attempt scores must be finite")
        maximum = max(self.negative_scores.values())
        assert self.max_negative_score is not None
        assert self.positive_score is not None
        assert self.margin is not None
        if self.max_negative_score != maximum:
            raise ValueError("GME maximum negative score is inconsistent")
        if self.margin != self.positive_score - maximum:
            raise ValueError("GME relative margin is inconsistent")
        expected_passed = self.status == "passed"
        if self.passed is not expected_passed:
            raise ValueError("GME attempt status must match passed")
        return self


_ATTRIBUTE_BACKGROUND_DISABLED_REASON = "attribute_background_disabled_by_policy"


def _normalize_attribute_background_policy_payload(value: object) -> object:
    """Read old Attribute variants while enforcing the current no-BG policy."""

    if not isinstance(value, dict) or value.get("status") != "accepted":
        return value
    variants_value = value.get("variants")
    if isinstance(variants_value, BaseModel):
        variants_value = variants_value.model_dump(mode="json")
    if not isinstance(variants_value, dict):
        return value
    generated_value = variants_value.get("generated_background")
    if isinstance(generated_value, BaseModel):
        generated_value = generated_value.model_dump(mode="json")
    if not isinstance(generated_value, dict):
        return value
    policy_compliant = (
        generated_value.get("image_path") is None
        and generated_value.get("status") == "unavailable"
        and generated_value.get("reviewed") is False
        and generated_value.get("review_status") == "not_applicable"
        and generated_value.get("reason") == _ATTRIBUTE_BACKGROUND_DISABLED_REASON
    )
    if policy_compliant and value.get("default_variant") != "generated_background":
        return value

    accepted_base = value.get("accepted_base_image_path")
    bbox_value = variants_value.get("bbox")
    if isinstance(bbox_value, BaseModel):
        bbox_value = bbox_value.model_dump(mode="json")
    bbox_accepted = (
        isinstance(bbox_value, dict)
        and bbox_value.get("status") == "accepted"
        and isinstance(bbox_value.get("image_path"), str)
        and bool(bbox_value["image_path"].strip())
    )
    accepted_base_available = isinstance(accepted_base, str) and bool(
        accepted_base.strip()
    )
    default_variant = "accepted_base" if accepted_base_available else "bbox"
    default_path = accepted_base if accepted_base_available else bbox_value["image_path"]
    normalized_generated = {
        "image_path": None,
        "status": "unavailable",
        "reviewed": False,
        "review_status": "not_applicable",
        "reason": _ATTRIBUTE_BACKGROUND_DISABLED_REASON,
        "synthetic": True,
        "metadata_path": None,
        "source_frame_index": generated_value.get(
            "source_frame_index", value.get("source_frame_index")
        ),
    }
    return {
        **value,
        "image_path": default_path,
        "variants": {
            **variants_value,
            "generated_background": normalized_generated,
        },
        "default_variant": default_variant,
        "default_image_path": default_path,
        "default_reason": (
            "legacy_attribute_default_normalized_to_accepted_base"
            if accepted_base_available
            else "legacy_attribute_bbox_fallback"
        ),
    }


class SubjectAttributeRecord(_SchemaModel):
    attribute_id: str = Field(pattern=r"^a[1-9]\d*$")
    owner_entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    attribute_type: AttributeType
    phrase: str
    grounding_prompt: str
    status: Literal["accepted", "rejected"]
    image_path: str | None = None
    source_frame_index: int | None = Field(default=None, ge=0)
    source_frame_slot: int | None = Field(default=None, ge=0, lt=10)
    owner_candidate_id: str | None = None
    same_frame_as_owner_reference: bool | None = None
    sam3_prompt: str
    ownership_geometry: OwnershipGeometry | None = None
    review: SubjectAttributeReview | None = None
    completion_review: SubjectAttributeCompletionReview | None = None
    gme_attempts: list[GmeAttributeScreenAttempt] = Field(default_factory=list)
    selected_gme_attempt_index: int | None = Field(default=None, ge=0)
    final_selection: Literal["raw", "completed", "bbox"] | None = None
    completion_attempted: bool = False
    completion_seed: int | None = Field(default=None, ge=0)
    completion_outcome: str | None = None
    variants: ReferenceVariantsState | None = None
    default_variant: ReferenceDefaultVariant | None = None
    default_image_path: str | None = None
    default_reason: str | None = None
    accepted_base_image_path: str | None = None
    reason: str

    @model_validator(mode="before")
    @classmethod
    def normalize_disabled_attribute_background(cls, value: object) -> object:
        normalized = _normalize_attribute_background_policy_payload(value)
        if not isinstance(normalized, dict):
            return normalized
        review = normalized.get("review")
        if not isinstance(review, dict) or "sufficient_source_evidence" in review:
            return normalized
        return {
            **normalized,
            "review": {**review, "sufficient_source_evidence": True},
        }

    @model_validator(mode="after")
    def validate_record(self) -> SubjectAttributeRecord:
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("attribute record text must not be empty")
        if not self.sam3_prompt.strip() or not self.reason.strip():
            raise ValueError("attribute record provenance must not be empty")
        provenance = (
            self.image_path,
            self.source_frame_index,
            self.source_frame_slot,
            self.owner_candidate_id,
            self.same_frame_as_owner_reference,
        )
        if self.status == "accepted":
            if any(value is None for value in provenance):
                raise ValueError("accepted attribute requires complete image provenance")
            if self.ownership_geometry is None or not self.ownership_geometry.passed:
                raise ValueError("accepted attribute requires passed ownership geometry")
            if self.review is None or self.review.attribute_id != self.attribute_id:
                raise ValueError("attribute record review ID must match")
            if self.final_selection == "completed":
                if (
                    self.completion_review is None
                    or self.completion_review.verdict != "accept"
                ):
                    raise ValueError(
                        "completed attribute requires an accepted completion review"
                    )
                if not self.completion_attempted:
                    raise ValueError("completed attribute requires a completion attempt")
            elif self.final_selection == "bbox":
                if not (
                    self.review.matches_attribute
                    and self.review.owner_binding_correct
                    and self.variants is not None
                    and self.variants.bbox.status == "accepted"
                ):
                    raise ValueError(
                        "bbox attribute requires semantic ownership and accepted bbox"
                    )
            elif not self.review.accepted:
                raise ValueError("raw attribute requires an accepted review")
        elif self.image_path is not None:
            raise ValueError("rejected attribute must not publish an image")
        if not self.completion_attempted and self.completion_seed is not None:
            raise ValueError("unattempted completion cannot publish a seed")
        required_defaults = (
            self.default_variant,
            self.default_image_path,
            self.default_reason,
        )
        if self.variants is None:
            if any(value is not None for value in required_defaults) or (
                self.accepted_base_image_path is not None
            ):
                raise ValueError("attribute variant defaults require variants")
        else:
            if self.status != "accepted" or any(
                value is None for value in required_defaults
            ):
                raise ValueError(
                    "accepted attribute variants require complete default provenance"
                )
            if self.image_path != self.default_image_path:
                raise ValueError("attribute image_path must remain the selected default")
            if self.default_variant != "accepted_base":
                assert self.default_variant is not None
                selected = getattr(self.variants, self.default_variant)
                if (
                    selected.status != "accepted"
                    or selected.image_path != self.default_image_path
                ):
                    raise ValueError(
                        "attribute default variant must be accepted and path-matched"
                    )
            elif (
                self.accepted_base_image_path is None
                or self.default_image_path != self.accepted_base_image_path
            ):
                raise ValueError("accepted-base default must preserve its base image")
        if self.selected_gme_attempt_index is not None:
            if self.selected_gme_attempt_index >= len(self.gme_attempts):
                raise ValueError("selected GME attempt index is out of range")
            selected = self.gme_attempts[self.selected_gme_attempt_index]
            if selected.status not in {"passed", "failed"}:
                raise ValueError("selected GME attempt must pass or fail open")
        return self


class OwnerEnrichmentMetrics(_SchemaModel):
    discovery_calls: int = Field(default=0, ge=0, le=1)
    review_calls: int = Field(default=0, ge=0, le=2)
    sam3_attempts: int = Field(default=0, ge=0, le=MAX_ATTRIBUTES_PER_OWNER)
    discovered_by_type: dict[str, int] = Field(default_factory=dict)
    deterministic_ownership_rejects: int = Field(default=0, ge=0)
    recognizability_rejects: int = Field(default=0, ge=0)
    accepted_attributes: int = Field(default=0, ge=0)
    same_frame_accepted: int = Field(default=0, ge=0)
    different_frame_accepted: int = Field(default=0, ge=0)
    qwen_model_call_time_seconds: float = Field(default=0.0, ge=0)
    sam3_model_call_time_seconds: float = Field(default=0.0, ge=0)
    gme_calls: int = Field(default=0, ge=0)
    gme_candidates_screened: int = Field(default=0, ge=0)
    gme_candidates_passed: int = Field(default=0, ge=0)
    gme_candidates_rejected: int = Field(default=0, ge=0)
    gme_retried_next_frame: int = Field(default=0, ge=0)
    gme_failures: int = Field(default=0, ge=0)
    gme_model_call_time_seconds: float = Field(default=0.0, ge=0)
    completion_attempts: int = Field(default=0, ge=0)
    completion_accepted: int = Field(default=0, ge=0)
    completion_rejected: int = Field(default=0, ge=0)
    completion_failures: int = Field(default=0, ge=0)
    completion_qwen_review_rejects: int = Field(default=0, ge=0)
    raw_attribute_review_accepted: int = Field(default=0, ge=0)
    raw_attribute_review_repair_recommended: int = Field(default=0, ge=0)
    raw_attribute_review_hard_rejected: int = Field(default=0, ge=0)
    completion_raw_usable_attempts: int = Field(default=0, ge=0)
    completion_raw_unusable_attempts: int = Field(default=0, ge=0)
    completion_selected_completed: int = Field(default=0, ge=0)
    completion_fallback_to_raw: int = Field(default=0, ge=0)
    completion_backend_failures: int = Field(default=0, ge=0)
    completion_postcheck_rejects: int = Field(default=0, ge=0)
    completion_sam_zero_mask_rejects: int = Field(default=0, ge=0)
    completion_sam_single_mask: int = Field(default=0, ge=0)
    completion_sam_multi_mask: int = Field(default=0, ge=0)
    completion_sam_masks_returned_total: int = Field(default=0, ge=0)
    completion_identity_review_rejects: int = Field(default=0, ge=0)
    completion_final_review_rejects: int = Field(default=0, ge=0)
    repaired_attribute_final_review_accepted: int = Field(default=0, ge=0)
    repaired_attribute_final_review_rejected: int = Field(default=0, ge=0)
    completion_review_calls: int = Field(default=0, ge=0)
    completion_model_call_time_seconds: float = Field(default=0.0, ge=0)
    completion_attempts_by_type: dict[str, int] = Field(default_factory=dict)
    completion_accepted_by_type: dict[str, int] = Field(default_factory=dict)
    attribute_bbox_variants_materialized: int = Field(default=0, ge=0)
    attribute_bbox_reviews_attempted: int = Field(default=0, ge=0)
    attribute_bbox_reviews_skipped_background_accepted: int = Field(
        default=0,
        ge=0,
    )
    attribute_background_variants_attempted: int = Field(default=0, ge=0)
    attribute_background_variants_accepted: int = Field(default=0, ge=0)
    attribute_source_candidates_considered: int = Field(default=0, ge=0)
    attribute_second_candidate_attempts: int = Field(default=0, ge=0)
    attribute_completion_candidate1_accepted: int = Field(default=0, ge=0)
    attribute_completion_candidate2_accepted: int = Field(default=0, ge=0)
    attribute_bbox_fallback_attempts: int = Field(default=0, ge=0)
    attribute_bbox_fallback_accepted: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_gme_counts(self) -> OwnerEnrichmentMetrics:
        if self.gme_candidates_screened != (
            self.gme_candidates_passed + self.gme_candidates_rejected
        ):
            raise ValueError("GME screened count must equal passed plus rejected")
        if self.gme_calls != self.gme_candidates_screened + self.gme_failures:
            raise ValueError("GME calls must equal screened plus failures")
        routing_metric_total = (
            self.completion_raw_usable_attempts
            + self.completion_raw_unusable_attempts
            + self.completion_selected_completed
            + self.completion_fallback_to_raw
            + self.completion_backend_failures
            + self.completion_postcheck_rejects
            + self.completion_sam_zero_mask_rejects
            + self.completion_sam_single_mask
            + self.completion_sam_multi_mask
            + self.completion_sam_masks_returned_total
            + self.completion_identity_review_rejects
            + self.completion_final_review_rejects
            + self.repaired_attribute_final_review_accepted
            + self.repaired_attribute_final_review_rejected
        )
        if self.completion_attempts > 0 and routing_metric_total == 0:
            if self.completion_attempts != (
                self.completion_accepted
                + self.completion_rejected
                + self.completion_failures
            ):
                raise ValueError("legacy completion counters are inconsistent")
        else:
            if self.completion_attempts != (
                self.completion_raw_usable_attempts
                + self.completion_raw_unusable_attempts
            ):
                raise ValueError(
                    "completion attempts must equal raw usable plus raw unusable attempts"
                )
            if self.completion_attempts < self.completion_selected_completed:
                raise ValueError(
                    "completion selections cannot exceed candidate attempts"
                )
            if self.completion_accepted != self.completion_selected_completed:
                raise ValueError("completion accepted must equal selected completed")
            if (
                self.completion_qwen_review_rejects
                != self.completion_identity_review_rejects
            ):
                raise ValueError("completion Qwen reject counters must match")
        if sum(self.completion_attempts_by_type.values()) != self.completion_attempts:
            raise ValueError("completion attempts by type must match attempts")
        if sum(self.completion_accepted_by_type.values()) != self.completion_accepted:
            raise ValueError("completion accepts by type must match accepts")
        if (
            self.completion_sam_zero_mask_rejects
            + self.completion_sam_single_mask
            + self.completion_sam_multi_mask
            > self.completion_attempts
        ):
            raise ValueError("completion SAM outcomes must not exceed attempts")
        if self.completion_sam_masks_returned_total < (
            self.completion_sam_single_mask + 2 * self.completion_sam_multi_mask
        ):
            raise ValueError("completion SAM returned-mask total is inconsistent")
        if (
            self.attribute_background_variants_accepted
            > self.attribute_background_variants_attempted
        ):
            raise ValueError("accepted attribute backgrounds exceed attempts")
        if (
            self.attribute_bbox_reviews_attempted
            + self.attribute_bbox_reviews_skipped_background_accepted
            > self.attribute_bbox_variants_materialized
        ):
            raise ValueError("attribute bbox review outcomes exceed materialized variants")
        for value in (
            self.qwen_model_call_time_seconds,
            self.sam3_model_call_time_seconds,
            self.gme_model_call_time_seconds,
            self.completion_model_call_time_seconds,
        ):
            if not math.isfinite(value):
                raise ValueError("attribute model-call times must be finite")
        return self


class OwnerEnrichmentArtifact(_SchemaModel):
    schema_version: Literal[
        "r2v.v3.subject_attribute_owner.1"
    ] = ATTRIBUTE_OWNER_SCHEMA_VERSION
    sample_id: str
    owner_entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    owner_is_human: bool | None
    attribute_id_start: int = Field(ge=1)
    owner_phrase: str
    owner_grounding_prompt: str
    records: list[SubjectAttributeRecord]
    metrics: OwnerEnrichmentMetrics
    gme_screen_mode: str | None = None
    completion_mode: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> OwnerEnrichmentArtifact:
        expected = [
            f"a{index}"
            for index in range(
                self.attribute_id_start,
                self.attribute_id_start + len(self.records),
            )
        ]
        actual = [record.attribute_id for record in self.records]
        if actual != expected:
            raise ValueError("owner attribute IDs must be contiguous and ordered")
        if any(record.owner_entity_id != self.owner_entity_id for record in self.records):
            raise ValueError("owner artifact records must use one owner_entity_id")
        if self.owner_is_human is not True and self.records:
            raise ValueError("only a confirmed human owner may publish attributes")
        if self.metrics.discovery_calls != 1:
            raise ValueError("eligible owner artifact requires one discovery call")
        if self.metrics.accepted_attributes != sum(
            record.status == "accepted" for record in self.records
        ):
            raise ValueError("owner accepted count must match records")
        return self


class EnrichedReference(_SchemaModel):
    image_id: str = Field(pattern=r"^image_[1-9]\d*$")
    image_index: int = Field(ge=1)
    kind: Literal["subject", "object", "group", "background", "attribute"]
    origin: Literal["visual_run", "attribute_enrichment"]
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: AttributeType | None = None
    image_path: str
    source_frame_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_binding(self) -> EnrichedReference:
        if self.image_id != f"image_{self.image_index}":
            raise ValueError("enriched reference image_id must match image_index")
        if self.kind == "attribute":
            if (
                self.origin != "attribute_enrichment"
                or self.attribute_id is None
                or self.owner_entity_id is None
                or self.attribute_type is None
                or self.entity_id is not None
            ):
                raise ValueError("attribute reference requires owner-bound provenance")
        elif (
            self.attribute_id is not None
            or self.owner_entity_id is not None
            or self.attribute_type is not None
        ):
            raise ValueError("visual references cannot publish attribute provenance")
        elif self.kind == "background":
            if self.entity_id is not None:
                raise ValueError("background reference entity_id must be null")
        elif self.entity_id is None:
            raise ValueError("entity reference requires entity_id")
        return self


class EnrichedSample(_SchemaModel):
    schema_version: Literal[
        "r2v.v3.enriched_sample.1"
    ] = ENRICHED_SAMPLE_SCHEMA_VERSION
    sample_id: str
    clip_uid: str
    source_run_root: str
    original_visual: dict[str, object]
    original_instruction: str
    enriched_instruction: str
    references: list[EnrichedReference]
    accepted_attributes: list[SubjectAttributeRecord]

    @model_validator(mode="before")
    @classmethod
    def normalize_disabled_attribute_background(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        accepted = value.get("accepted_attributes")
        references = value.get("references")
        if not isinstance(accepted, list) or not isinstance(references, list):
            return value
        normalized_records = [
            _normalize_attribute_background_policy_payload(record)
            for record in accepted
        ]
        selected_path_by_id = {
            record["attribute_id"]: record.get("image_path")
            for record in normalized_records
            if isinstance(record, dict)
            and isinstance(record.get("attribute_id"), str)
        }
        normalized_references: list[object] = []
        for reference in references:
            if not isinstance(reference, dict) or reference.get("kind") != "attribute":
                normalized_references.append(reference)
                continue
            selected_path = selected_path_by_id.get(reference.get("attribute_id"))
            normalized_references.append(
                {**reference, "image_path": selected_path}
                if isinstance(selected_path, str)
                else reference
            )
        return {
            **value,
            "accepted_attributes": normalized_records,
            "references": normalized_references,
        }

    @model_validator(mode="after")
    def validate_sample(self) -> EnrichedSample:
        indexes = [reference.image_index for reference in self.references]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("enriched reference indexes must be contiguous")
        expected_labels = {str(index) for index in indexes}
        if set(_ANGLE_IMAGE_LABEL.findall(self.enriched_instruction)) != expected_labels:
            raise ValueError("enriched instruction labels must match references")
        accepted_ids = [record.attribute_id for record in self.accepted_attributes]
        reference_ids = [
            reference.attribute_id
            for reference in self.references
            if reference.kind == "attribute"
        ]
        if reference_ids != accepted_ids:
            raise ValueError("enriched attribute references must match accepted records")
        if any(record.status != "accepted" for record in self.accepted_attributes):
            raise ValueError("enriched sample may contain only accepted attributes")
        return self


@dataclass(frozen=True)
class OwnerEligibility:
    eligible: bool
    reason: str


@dataclass(frozen=True)
class PendingAttributeCandidate:
    discovered: DiscoveredSubjectAttribute
    attribute_id: str
    owner_entity_id: str
    owner_candidate: EntityReferenceCandidate
    attribute_mask: np.ndarray
    source_image: Image.Image
    crop: Image.Image
    geometry: OwnershipGeometry
    gme_attempts: tuple[GmeAttributeScreenAttempt, ...] = ()
    selected_gme_attempt_index: int | None = None
    completion_attempted: bool = False
    completion_model_call_time_seconds: float = 0.0
    completion_review_time_seconds: float = 0.0
    raw_crop: Image.Image | None = None


@dataclass
class _GmeSelectionMetrics:
    calls: int = 0
    candidates_screened: int = 0
    candidates_passed: int = 0
    candidates_rejected: int = 0
    retried_next_frame: int = 0
    failures: int = 0
    model_call_time_seconds: float = 0.0


@dataclass
class _CompletionSelectionMetrics:
    attempts: int = 0
    accepted: int = 0
    rejected: int = 0
    failures: int = 0
    qwen_review_rejects: int = 0
    raw_review_accepted: int = 0
    raw_review_repair_recommended: int = 0
    raw_review_hard_rejected: int = 0
    raw_usable_attempts: int = 0
    raw_unusable_attempts: int = 0
    selected_completed: int = 0
    fallback_to_raw: int = 0
    backend_failures: int = 0
    postcheck_rejects: int = 0
    sam_zero_mask_rejects: int = 0
    sam_single_mask: int = 0
    sam_multi_mask: int = 0
    sam_masks_returned_total: int = 0
    identity_review_rejects: int = 0
    final_review_rejects: int = 0
    repaired_final_review_accepted: int = 0
    repaired_final_review_rejected: int = 0
    review_calls: int = 0
    model_call_time_seconds: float = 0.0
    review_time_seconds: float = 0.0
    sam3_time_seconds: float = 0.0
    attempts_by_type: Counter[str] = field(default_factory=Counter)
    accepted_by_type: Counter[str] = field(default_factory=Counter)


@dataclass
class _AttributeVariantMetrics:
    bbox_variants_materialized: int = 0
    bbox_reviews_attempted: int = 0


@dataclass
class EnrichmentTotals:
    eligible_human_owners: int = 0
    screened_nonhuman_subjects: int = 0
    skipped_existing_owners: int = 0
    discovery_calls: int = 0
    review_calls: int = 0
    sam3_attempts: int = 0
    discovered_by_type: Counter[str] = field(default_factory=Counter)
    deterministic_ownership_rejects: int = 0
    recognizability_rejects: int = 0
    accepted_attributes: int = 0
    same_frame_accepted: int = 0
    different_frame_accepted: int = 0
    qwen_model_call_time_seconds: float = 0.0
    sam3_model_call_time_seconds: float = 0.0
    gme_calls: int = 0
    gme_candidates_screened: int = 0
    gme_candidates_passed: int = 0
    gme_candidates_rejected: int = 0
    gme_retried_next_frame: int = 0
    gme_failures: int = 0
    gme_model_call_time_seconds: float = 0.0
    completion_attempts: int = 0
    completion_accepted: int = 0
    completion_rejected: int = 0
    completion_failures: int = 0
    completion_qwen_review_rejects: int = 0
    raw_attribute_review_accepted: int = 0
    raw_attribute_review_repair_recommended: int = 0
    raw_attribute_review_hard_rejected: int = 0
    completion_raw_usable_attempts: int = 0
    completion_raw_unusable_attempts: int = 0
    completion_selected_completed: int = 0
    completion_fallback_to_raw: int = 0
    completion_backend_failures: int = 0
    completion_postcheck_rejects: int = 0
    completion_sam_zero_mask_rejects: int = 0
    completion_sam_single_mask: int = 0
    completion_sam_multi_mask: int = 0
    completion_sam_masks_returned_total: int = 0
    completion_identity_review_rejects: int = 0
    completion_final_review_rejects: int = 0
    repaired_attribute_final_review_accepted: int = 0
    repaired_attribute_final_review_rejected: int = 0
    completion_review_calls: int = 0
    completion_model_call_time_seconds: float = 0.0
    completion_attempts_by_type: Counter[str] = field(default_factory=Counter)
    completion_accepted_by_type: Counter[str] = field(default_factory=Counter)
    attribute_bbox_variants_materialized: int = 0
    attribute_bbox_reviews_attempted: int = 0
    attribute_bbox_reviews_skipped_background_accepted: int = 0
    attribute_background_variants_attempted: int = 0
    attribute_background_variants_accepted: int = 0
    attribute_source_candidates_considered: int = 0
    attribute_second_candidate_attempts: int = 0
    attribute_completion_candidate1_accepted: int = 0
    attribute_completion_candidate2_accepted: int = 0
    attribute_bbox_fallback_attempts: int = 0
    attribute_bbox_fallback_accepted: int = 0
    failures: int = 0
    failure_reasons: Counter[str] = field(default_factory=Counter)

    def add_owner_metrics(self, metrics: OwnerEnrichmentMetrics) -> None:
        self.discovery_calls += metrics.discovery_calls
        self.review_calls += metrics.review_calls
        self.sam3_attempts += metrics.sam3_attempts
        self.discovered_by_type.update(metrics.discovered_by_type)
        self.deterministic_ownership_rejects += (
            metrics.deterministic_ownership_rejects
        )
        self.recognizability_rejects += metrics.recognizability_rejects
        self.accepted_attributes += metrics.accepted_attributes
        self.same_frame_accepted += metrics.same_frame_accepted
        self.different_frame_accepted += metrics.different_frame_accepted
        self.qwen_model_call_time_seconds += metrics.qwen_model_call_time_seconds
        self.sam3_model_call_time_seconds += metrics.sam3_model_call_time_seconds
        self.gme_calls += metrics.gme_calls
        self.gme_candidates_screened += metrics.gme_candidates_screened
        self.gme_candidates_passed += metrics.gme_candidates_passed
        self.gme_candidates_rejected += metrics.gme_candidates_rejected
        self.gme_retried_next_frame += metrics.gme_retried_next_frame
        self.gme_failures += metrics.gme_failures
        self.gme_model_call_time_seconds += metrics.gme_model_call_time_seconds
        self.completion_attempts += metrics.completion_attempts
        self.completion_accepted += metrics.completion_accepted
        self.completion_rejected += metrics.completion_rejected
        self.completion_failures += metrics.completion_failures
        self.completion_qwen_review_rejects += (
            metrics.completion_qwen_review_rejects
        )
        self.raw_attribute_review_accepted += metrics.raw_attribute_review_accepted
        self.raw_attribute_review_repair_recommended += (
            metrics.raw_attribute_review_repair_recommended
        )
        self.raw_attribute_review_hard_rejected += (
            metrics.raw_attribute_review_hard_rejected
        )
        self.completion_raw_usable_attempts += (
            metrics.completion_raw_usable_attempts
        )
        self.completion_raw_unusable_attempts += (
            metrics.completion_raw_unusable_attempts
        )
        self.completion_selected_completed += metrics.completion_selected_completed
        self.completion_fallback_to_raw += metrics.completion_fallback_to_raw
        self.completion_backend_failures += metrics.completion_backend_failures
        self.completion_postcheck_rejects += metrics.completion_postcheck_rejects
        self.completion_sam_zero_mask_rejects += (
            metrics.completion_sam_zero_mask_rejects
        )
        self.completion_sam_single_mask += metrics.completion_sam_single_mask
        self.completion_sam_multi_mask += metrics.completion_sam_multi_mask
        self.completion_sam_masks_returned_total += (
            metrics.completion_sam_masks_returned_total
        )
        self.completion_identity_review_rejects += (
            metrics.completion_identity_review_rejects
        )
        self.completion_final_review_rejects += (
            metrics.completion_final_review_rejects
        )
        self.repaired_attribute_final_review_accepted += (
            metrics.repaired_attribute_final_review_accepted
        )
        self.repaired_attribute_final_review_rejected += (
            metrics.repaired_attribute_final_review_rejected
        )
        self.completion_review_calls += metrics.completion_review_calls
        self.completion_model_call_time_seconds += (
            metrics.completion_model_call_time_seconds
        )
        self.completion_attempts_by_type.update(metrics.completion_attempts_by_type)
        self.completion_accepted_by_type.update(metrics.completion_accepted_by_type)
        self.attribute_bbox_variants_materialized += (
            metrics.attribute_bbox_variants_materialized
        )
        self.attribute_bbox_reviews_attempted += (
            metrics.attribute_bbox_reviews_attempted
        )
        self.attribute_bbox_reviews_skipped_background_accepted += (
            metrics.attribute_bbox_reviews_skipped_background_accepted
        )
        self.attribute_background_variants_attempted += (
            metrics.attribute_background_variants_attempted
        )
        self.attribute_background_variants_accepted += (
            metrics.attribute_background_variants_accepted
        )
        self.attribute_source_candidates_considered += (
            metrics.attribute_source_candidates_considered
        )
        self.attribute_second_candidate_attempts += (
            metrics.attribute_second_candidate_attempts
        )
        self.attribute_completion_candidate1_accepted += (
            metrics.attribute_completion_candidate1_accepted
        )
        self.attribute_completion_candidate2_accepted += (
            metrics.attribute_completion_candidate2_accepted
        )
        self.attribute_bbox_fallback_attempts += (
            metrics.attribute_bbox_fallback_attempts
        )
        self.attribute_bbox_fallback_accepted += (
            metrics.attribute_bbox_fallback_accepted
        )
        self.failures += metrics.failures


@dataclass(frozen=True)
class ClipEnrichmentResult:
    clip_uid: str
    totals: EnrichmentTotals
    owner_limit_reached: bool
    enriched_sample: EnrichedSample | None

    def to_counts(self) -> dict[str, int | float]:
        return {
            "eligible_human_owners": self.totals.eligible_human_owners,
            "screened_nonhuman_subjects": self.totals.screened_nonhuman_subjects,
            "skipped_existing_owners": self.totals.skipped_existing_owners,
            "discovery_calls": self.totals.discovery_calls,
            "review_calls": self.totals.review_calls,
            "sam3_attempts": self.totals.sam3_attempts,
            "deterministic_ownership_rejects": (
                self.totals.deterministic_ownership_rejects
            ),
            "recognizability_rejects": self.totals.recognizability_rejects,
            "accepted_attributes": self.totals.accepted_attributes,
            "failures": self.totals.failures,
            "qwen_model_call_time_seconds": (
                self.totals.qwen_model_call_time_seconds
            ),
            "sam3_model_call_time_seconds": self.totals.sam3_model_call_time_seconds,
            "gme_calls": self.totals.gme_calls,
            "gme_candidates_screened": self.totals.gme_candidates_screened,
            "gme_candidates_passed": self.totals.gme_candidates_passed,
            "gme_candidates_rejected": self.totals.gme_candidates_rejected,
            "gme_retried_next_frame": self.totals.gme_retried_next_frame,
            "gme_failures": self.totals.gme_failures,
            "gme_model_call_time_seconds": self.totals.gme_model_call_time_seconds,
            "completion_attempts": self.totals.completion_attempts,
            "completion_accepted": self.totals.completion_accepted,
            "completion_rejected": self.totals.completion_rejected,
            "completion_failures": self.totals.completion_failures,
            "completion_qwen_review_rejects": (
                self.totals.completion_qwen_review_rejects
            ),
            "raw_attribute_review_accepted": (
                self.totals.raw_attribute_review_accepted
            ),
            "raw_attribute_review_repair_recommended": (
                self.totals.raw_attribute_review_repair_recommended
            ),
            "raw_attribute_review_hard_rejected": (
                self.totals.raw_attribute_review_hard_rejected
            ),
            "completion_raw_usable_attempts": (
                self.totals.completion_raw_usable_attempts
            ),
            "completion_raw_unusable_attempts": (
                self.totals.completion_raw_unusable_attempts
            ),
            "completion_selected_completed": (
                self.totals.completion_selected_completed
            ),
            "completion_fallback_to_raw": self.totals.completion_fallback_to_raw,
            "completion_backend_failures": self.totals.completion_backend_failures,
            "completion_postcheck_rejects": self.totals.completion_postcheck_rejects,
            "completion_sam_zero_mask_rejects": (
                self.totals.completion_sam_zero_mask_rejects
            ),
            "completion_sam_single_mask": self.totals.completion_sam_single_mask,
            "completion_sam_multi_mask": self.totals.completion_sam_multi_mask,
            "completion_sam_masks_returned_total": (
                self.totals.completion_sam_masks_returned_total
            ),
            "completion_identity_review_rejects": (
                self.totals.completion_identity_review_rejects
            ),
            "completion_final_review_rejects": (
                self.totals.completion_final_review_rejects
            ),
            "repaired_attribute_final_review_accepted": (
                self.totals.repaired_attribute_final_review_accepted
            ),
            "repaired_attribute_final_review_rejected": (
                self.totals.repaired_attribute_final_review_rejected
            ),
            "completion_model_call_time_seconds": (
                self.totals.completion_model_call_time_seconds
            ),
            "attribute_source_candidates_considered": (
                self.totals.attribute_source_candidates_considered
            ),
            "attribute_second_candidate_attempts": (
                self.totals.attribute_second_candidate_attempts
            ),
            "attribute_completion_candidate1_accepted": (
                self.totals.attribute_completion_candidate1_accepted
            ),
            "attribute_completion_candidate2_accepted": (
                self.totals.attribute_completion_candidate2_accepted
            ),
            "attribute_bbox_fallback_attempts": (
                self.totals.attribute_bbox_fallback_attempts
            ),
            "attribute_bbox_fallback_accepted": (
                self.totals.attribute_bbox_fallback_accepted
            ),
            "attribute_background_variants_attempted": (
                self.totals.attribute_background_variants_attempted
            ),
            "attribute_background_variants_accepted": (
                self.totals.attribute_background_variants_accepted
            ),
        }


class SubjectAttributeDiscoveryClient(Protocol):
    def discover(
        self,
        *,
        owner: AnnotationEntity,
        owner_candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> SubjectAttributeDiscovery: ...


class SubjectAttributeReviewClient(Protocol):
    def review(
        self,
        *,
        owner: AnnotationEntity,
        candidates: list[PendingAttributeCandidate],
    ) -> SubjectAttributeReviewBatch: ...


class AttributeGmeScreener(Protocol):
    def screen(
        self,
        *,
        crop: Image.Image,
        phrase: str,
        attribute_type: str,
    ) -> GmeRelativeMarginResult: ...


class AttributeFrameSegmentationBackend(Protocol):
    def segment_frame(
        self,
        *,
        frame_path: Path,
        frame_slot: int,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]: ...

    def segment_generated_frame(
        self,
        *,
        frame_path: Path,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]: ...


class AttributeCompletionBackend(Protocol):
    def attribute_completion(
        self,
        *,
        source_path: Path,
        output_path: Path,
        instruction: str,
        seed: int,
    ) -> dict[str, object]: ...


class AttributeCompletionJudge(Protocol):
    def review(
        self,
        *,
        source_attribute: Image.Image,
        source_bbox: Image.Image,
        generated_candidate: Image.Image,
        attribute_type: str,
        attribute_phrase: str,
    ) -> SubjectAttributeCompletionReview: ...


class AttributeVariantJudge(Protocol):
    def review_attribute_bbox(
        self,
        *,
        bbox_candidate: Image.Image,
        owner_context: Image.Image,
        attribute_type: str,
        attribute_phrase: str,
    ) -> SubjectAttributeBboxReview: ...


class AttributeProbeClient(Protocol):
    def attribute_probe(
        self,
        *,
        clip_uid: str,
        frame_slot: int,
        source_frame_index: int,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]: ...

    def attribute_completion_probe(
        self,
        *,
        image_path: Path,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]: ...


class Sam3AttributeFrameSegmenter:
    """Subject-attributes-local SAM3 adapter with no temporal propagation."""

    def __init__(
        self,
        config: Sam3Config,
        *,
        backend: Sam3SegmentationBackend | None = None,
    ) -> None:
        self._backend = backend or Sam3SegmentationBackend(config)

    def segment_frame(
        self,
        *,
        frame_path: Path,
        frame_slot: int,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]:
        resolved = frame_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"attribute candidate frame is missing: {resolved}")
        if not grounding_prompt.strip():
            raise ValueError("attribute grounding prompt must not be empty")
        predictor = self._backend._load_predictor()
        observations = self._backend._prompt_frame(
            predictor,
            frames_dir=resolved.parent,
            slot=frame_slot,
            grounding_prompt=grounding_prompt,
        )
        return tuple(
            np.asarray(observation.mask, dtype=bool).copy()
            for observation in observations
            if observation.valid and np.asarray(observation.mask, dtype=bool).any()
        )

    def close(self) -> None:
        self._backend.close()

    def segment_generated_frame(
        self,
        *,
        frame_path: Path,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]:
        return self.segment_frame(
            frame_path=frame_path,
            frame_slot=0,
            grounding_prompt=grounding_prompt,
        )


class PersistentWorkerAttributeFrameSegmenter:
    """Translate frame-local probes onto the existing persistent SAM3 worker."""

    def __init__(
        self,
        storage: RunStorage,
        client: AttributeProbeClient,
        *,
        inference_lock: threading.Lock | None = None,
    ) -> None:
        self._storage = storage
        self._client = client
        self._inference_lock = inference_lock

    def segment_frame(
        self,
        *,
        frame_path: Path,
        frame_slot: int,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]:
        resolved = frame_path.expanduser().resolve(strict=False)
        relative = resolved.relative_to(self._storage.root)
        if len(relative.parts) != 4 or relative.parts[:1] != ("clips",):
            raise ValueError("attribute frame must be a clip-scoped run artifact")
        clip_uid = relative.parts[1]
        frames = self._storage.read_frames(clip_uid)
        frame = next((item for item in frames.frames if item.slot == frame_slot), None)
        if frame is None:
            raise ValueError("attribute frame slot is absent from sampled frames")
        expected = (self._storage.clip_dir(clip_uid) / frame.image_path).resolve(
            strict=False
        )
        if expected != resolved:
            raise ValueError("attribute frame path does not match sampled provenance")
        if self._inference_lock is None:
            return self._client.attribute_probe(
                clip_uid=clip_uid,
                frame_slot=frame_slot,
                source_frame_index=frame.source_frame_index,
                grounding_prompt=grounding_prompt,
            )
        with self._inference_lock:
            return self._client.attribute_probe(
                clip_uid=clip_uid,
                frame_slot=frame_slot,
                source_frame_index=frame.source_frame_index,
                grounding_prompt=grounding_prompt,
            )

    def segment_generated_frame(
        self,
        *,
        frame_path: Path,
        grounding_prompt: str,
    ) -> tuple[np.ndarray, ...]:
        resolved = frame_path.expanduser().resolve()
        relative = resolved.relative_to(self._storage.root)
        if relative.parts[:2] != (
            "subject_attributes",
            "completion_candidates",
        ):
            raise ValueError("completion image must be a run-local attribute sidecar")
        if self._inference_lock is None:
            return self._client.attribute_completion_probe(
                image_path=resolved,
                grounding_prompt=grounding_prompt,
            )
        with self._inference_lock:
            return self._client.attribute_completion_probe(
                image_path=resolved,
                grounding_prompt=grounding_prompt,
            )


DISCOVERY_SYSTEM_PROMPT = """Inspect one explicitly identified V3 subject in
up to three existing strong Visual V3 candidate frames. Return at most three
clearly visible, visually distinctive, segmentable attributes that are
unambiguously owned by that subject. Omit uncertain, tiny, generic, hidden, or
weak items; zero attributes is valid. Prefer useful face or hair, distinctive
clothing, headwear or glasses, then shoes, bag, or another clear wearable
accessory. Do not enumerate every small item. phrase and grounding_prompt must
both be written in English. grounding_prompt must include the
owner relation, for example 'the red jacket worn by the woman'. Return one JSON
object whose top level contains exactly owner_entity_id, owner_is_human, and
attributes. Do not return owner_phrase, owner_grounding_prompt, or any other
top-level field. attributes must be a JSON array. Each attribute has exactly
attribute_type, phrase, and grounding_prompt. Each returned attribute_type must
be unique. Allowed types are face, hair, headwear, glasses, upper_clothing,
lower_clothing, dress_or_skirt, shoes, bag, and accessory. The required shape is
{"owner_entity_id":"e1",
"owner_is_human":true,"attributes":[{"attribute_type":"hair",
"phrase":"black hair","grounding_prompt":"black hair worn by the woman"}]}.
If the subject is an animal, non-human creature, or non-human
character, return owner_is_human=false and attributes=[]. Return JSON only."""


REVIEW_SYSTEM_PROMPT = """Review all proposed subject-bound attribute crops for
one known owner together. The isolated transparent crop is the CROP-ONLY
QUALITY TARGET. Judge recognizable, characteristic_appearance_visible, and
usable_as_attribute_condition from that isolated crop by itself. Do not use the
owner context, supplied attribute phrase, or expected attribute type to rescue
an otherwise ambiguous crop. The owner image is OWNERSHIP-ONLY CONTEXT and may
be used only for owner_binding_correct. matches_attribute=true only when the
named component is the dominant visible content of the isolated crop. Reject
matches_attribute when substantial unrelated body regions, another garment, or
a general owner or body silhouette is present, even if the target component is
also visible. owner_binding_correct means the component belongs to the intended
person.
sufficient_source_evidence=true only when the isolated alpha crop retains
enough real visual evidence to constrain the attribute's identity, appearance,
or structure. It does not require whole-object completeness. Set it false for
a tiny fragment, isolated edge, a few colors, a blurry patch, disconnected weak
pieces, or content whose meaning can be guessed only from owner context, bbox,
attribute type, or text. A visibly incomplete garment may still be true when
its real material, color, pattern, shape cues, and component structure provide
meaningful evidence for a conservative completion.
recognizable requires sufficient component structure, not merely a guessable
category. Hair needs a coherent hairstyle region or silhouette; reject fringe,
arcs, isolated strands, or contour-only regions. Face needs enough facial
structure to function independently and must show a front or near-front face
with roughly at least 50% of the frontal facial structure visible. Reject a
strong three-quarter or near-side face below that standard, a side profile,
the back of a head, an excessively turned face, or isolated facial patches.
Apply this frontal requirement only when attribute_type is face; do not apply
it to hair, headwear, glasses, clothing, shoes, bags, or accessories. Upper or
lower clothing needs coherent garment structure; reject a narrow shoulder, sleeve,
cuff, hem, trouser edge, or arbitrary strip. Headwear, glasses, shoes, bags, and
accessories need enough of the item to identify its structure independently.
usable_as_attribute_condition=true only when the isolated crop provides clean,
useful appearance information beyond the main subject reference. Reject a
generic owner or body silhouette, a pose-dominated cutout, an arbitrary subject-
region cutout, an overly fragmentary component, or a crop whose meaning relies
on the requested attribute_type or owner context.
For the three crop-only quality booleans, ask: if this isolated crop were shown
alone, would a neutral viewer be able to identify what component it is and
recover useful appearance information from it?
Reject thin curved strips or contour-only hair; a few strands or only an edge
of a hairstyle; an isolated sleeve, cuff, shoulder, hem, or trouser edge; a
generic dark or light blob; a tiny partial wearable; a crop with color or
texture but insufficient recognizable component structure; and any crop
recognizable only because owner context or text identifies it. Do not require
whole-object completeness. A partial face, hair, or clothing region is
acceptable when the isolated crop still preserves clear component-level
structure and characteristic appearance.
A crop may still be recognizable and usable while being structurally
incomplete. Judge structural completeness independently from the five existing
usability booleans. structure_complete=true only when the target attribute
region has no obvious large broken, missing, cut-out, erased, or structurally
discontinuous area that should reasonably be reconstructed.
completion_recommended=true when filling a visible missing or broken area would
likely produce a materially better reference. If the semantic target and owner
are correct and a clearly missing or broken region exists, set
structure_complete=false and completion_recommended=true even when
usable_as_attribute_condition=true. Do not use structural incompleteness to
rescue a wrong target attribute or wrong-owner crop.
Return one JSON object whose top level contains exactly owner_entity_id and
reviews. reviews must be a JSON array preserving the supplied attribute order.
Every review array item must contain exactly attribute_id,
matches_attribute, owner_binding_correct, recognizable,
characteristic_appearance_visible, usable_as_attribute_condition,
sufficient_source_evidence, structure_complete, completion_recommended, and one
concise reason. Copy each
supplied attribute_id into its review item. Do not
return an object keyed by a1, a2, a3, or any other attribute_id. The required
shape is {"owner_entity_id":"e1","reviews":[{"attribute_id":"a1",
"matches_attribute":true,"owner_binding_correct":true,"recognizable":true,
"characteristic_appearance_visible":true,
"usable_as_attribute_condition":true,"sufficient_source_evidence":true,
"structure_complete":true,
"completion_recommended":false,"reason":"..."}]}. Return JSON only."""


ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT = """You compare an isolated source
attribute with a generated completion. Image 1 is the original alpha crop,
Image 2 is an RGB source bbox used only as physical-instance and identity
evidence, and Image 3 is the generated candidate.

Choose Image 3 only if it is a better conditioning reference than Image 1 while
remaining the same physical attribute shown by Image 2. A modest real gain in
completeness, coherence, clarity, or reference usability is enough. If Image 3
is equivalent to Image 1, keep Image 1. Preserve visible color, material,
texture, pattern, shape cues, and distinctive details. For a face, Image 3 must
depict the same person as Image 2. For headwear, accessories, clothing, and
dresses or skirts, Image 3 must remain the same physical item or component.

Reject a wrong new instance, identity or appearance drift, a duplicate
attribute or component, unrelated content, malformed or structurally distorted
parts, a target that is too small or pushed into an extreme corner or edge, or
loss of visual prominence. Image 2 is identity evidence only. Do not require
Image 3 to match Image 2's position, scale, crop, or composition. Reasonable
translation, recentering, moderate scale change, and layout change are allowed.

Judge visible facts only and fail closed when uncertain. Return one strict JSON
object matching the supplied schema and no other text."""

ATTRIBUTE_BBOX_REVIEW_SYSTEM_PROMPT = """You are reviewing a last-resort RGB
bbox reference for one subject-bound attribute. The isolated or completed
attribute alternatives were not usable, so this bbox is considered only as a
fallback.

Accept the bbox only when the target attribute remains clearly identifiable and
dominant while unrelated owner identity, face, body pose, other garments, other
people, scene layout, background, text, and unrelated objects contribute only
minimal conditioning information. Reject when the bbox contains substantial
owner identity, a large face or body, strong pose information, other major
garments, another person, or scene content that could create a strong
copy-paste shortcut. The bbox must primarily teach the requested attribute, not
the original frame. Reject a tiny, blurry, or semantically weak attribute
region even when surrounding owner context or text makes its category
guessable. Fail closed when uncertain.

Return exactly the strict JSON schema with reason before verdict and no extra
fields."""


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resize_qwen_input_image(image: Image.Image) -> Image.Image:
    long_side = max(image.size)
    if long_side <= QWEN_INPUT_MAX_LONG_SIDE_PIXELS:
        return image.copy()
    scale = QWEN_INPUT_MAX_LONG_SIDE_PIXELS / long_side
    width, height = image.size
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def _owner_candidate_provenance_key(
    candidate: EntityReferenceCandidate,
) -> tuple[str, str, int, int, str]:
    return (
        candidate.entity_id,
        candidate.candidate_id,
        candidate.source_frame_index,
        candidate.frame_slot,
        candidate.image_path,
    )


def _normalize_discovery_payload(raw: str) -> object:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("attributes"), list
    ):
        return payload
    seen_types: set[str] = set()
    attributes: list[object] = []
    for attribute in payload["attributes"]:
        attribute_type = (
            attribute.get("attribute_type") if isinstance(attribute, dict) else None
        )
        if isinstance(attribute_type, str):
            if attribute_type in seen_types:
                continue
            seen_types.add(attribute_type)
        attributes.append(attribute)
    return {**payload, "attributes": attributes}


class QwenSubjectAttributeClient:
    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(
        self,
        *,
        component: str,
        system_prompt: str,
        content: list[dict[str, object]],
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        response = profiled_openai_call(
            lambda: self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            ),
            component=component,
            operation="batch",
            retry_index=0,
            model=self.config.model,
            messages=messages,
            metadata={"response_format": "json_object"},
        )
        raw = response.choices[0].message.content
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{component} returned empty content")
        return raw

    def _structured_call(
        self,
        *,
        component: str,
        messages: list[dict[str, object]],
        response_model: type[BaseModel],
        retry_index: int,
    ) -> str:
        schema = response_model.model_json_schema()
        parameters: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        operation = "initial" if retry_index == 0 else "repair"
        try:
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **parameters,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "v3_subject_attribute_bbox_review",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                ),
                component=component,
                operation=operation,
                retry_index=retry_index,
                model=self.config.model,
                messages=messages,
                metadata={"response_format": "json_schema"},
            )
        except BadRequestError:
            fallback_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return exactly one JSON object matching this schema. "
                        "Every required field must be present. "
                        "Do not omit booleans. No extra fields.\n"
                        + json.dumps(schema, ensure_ascii=False)
                    ),
                },
            ]
            response = profiled_openai_call(
                lambda: self.client.chat.completions.create(
                    **{**parameters, "messages": fallback_messages},
                    response_format={"type": "json_object"},
                ),
                component=component,
                operation=operation,
                retry_index=retry_index,
                model=self.config.model,
                messages=fallback_messages,
                metadata={"response_format": "json_object"},
            )
        raw = response.choices[0].message.content
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{component} returned empty content")
        return raw

    @staticmethod
    def _normalize_structured_verdict(
        payload: object,
        response_model: type[BaseModel],
    ) -> object:
        if not isinstance(payload, dict):
            return payload
        flag_names = tuple(
            name
            for name in response_model.model_fields
            if name not in {"reason", "verdict"}
        )
        if flag_names and all(type(payload.get(name)) is bool for name in flag_names):
            return {
                **payload,
                "verdict": (
                    "accept" if all(payload[name] for name in flag_names) else "reject"
                ),
            }
        return payload

    def _request_structured(
        self,
        *,
        component: str,
        system_prompt: str,
        content: list[dict[str, object]],
        response_model: type[BaseModel],
    ) -> BaseModel:
        schema = response_model.model_json_schema()
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        raw = self._structured_call(
            component=component,
            messages=messages,
            response_model=response_model,
            retry_index=0,
        )
        for attempt in range(2):
            try:
                payload = self._normalize_structured_verdict(
                    json.loads(raw),
                    response_model,
                )
                return response_model.model_validate(payload)
            except (TypeError, ValueError) as exc:
                if attempt == 1:
                    raise ValueError(
                        f"invalid structured {component} response: {exc}"
                    ) from exc
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Repair only the JSON structure while preserving the "
                            "original visual judgments. Return exactly one JSON object. "
                            "Every required field must be present, including every "
                            "boolean. No extra fields.\n"
                            f"Original JSON:\n{raw}\n"
                            f"Validation error:\n{exc}\n"
                            "Required schema:\n"
                            + json.dumps(schema, ensure_ascii=False)
                        ),
                    },
                ]
                raw = self._structured_call(
                    component=component,
                    messages=repair_messages,
                    response_model=response_model,
                    retry_index=1,
                )
        raise AssertionError("unreachable")

    def discover(
        self,
        *,
        owner: AnnotationEntity,
        owner_candidates: list[EntityReferenceCandidate],
        source_images: dict[str, Image.Image],
    ) -> SubjectAttributeDiscovery:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "owner_entity_id": owner.entity_id,
                        "owner_phrase": owner.phrase,
                        "owner_grounding_prompt": owner.grounding_prompt,
                        "candidate_count": len(owner_candidates),
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        for candidate in owner_candidates[:MAX_ATTRIBUTES_PER_OWNER]:
            source = source_images[candidate.image_path]
            context = build_candidate_context_image(source, candidate.mask)
            content.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            f"Owner {owner.entity_id} strong candidate "
                            f"{candidate.candidate_id}, source frame "
                            f"{candidate.source_frame_index}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(_resize_qwen_input_image(context))
                        },
                    },
                )
            )
        payload = SubjectAttributeDiscovery.model_validate(
            _normalize_discovery_payload(
                self._request(
                    component="qwen_subject_attribute_discovery",
                    system_prompt=DISCOVERY_SYSTEM_PROMPT,
                    content=content,
                )
            )
        )
        if payload.owner_entity_id != owner.entity_id:
            raise ValueError("attribute discovery returned the wrong owner_entity_id")
        return payload

    def review(
        self,
        *,
        owner: AnnotationEntity,
        candidates: list[PendingAttributeCandidate],
    ) -> SubjectAttributeReviewBatch:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "owner_entity_id": owner.entity_id,
                        "attributes": [
                            {
                                "attribute_id": candidate.attribute_id,
                                "attribute_type": candidate.discovered.attribute_type,
                            }
                            for candidate in candidates
                        ],
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        context_groups: dict[
            tuple[str, str, int, int, str],
            list[PendingAttributeCandidate],
        ] = {}
        for candidate in candidates:
            key = _owner_candidate_provenance_key(candidate.owner_candidate)
            context_groups.setdefault(key, []).append(candidate)
        emitted_contexts: set[tuple[str, str, int, int, str]] = set()
        for candidate in candidates:
            context_key = _owner_candidate_provenance_key(candidate.owner_candidate)
            if context_key not in emitted_contexts:
                shared_attribute_ids = ", ".join(
                    item.attribute_id for item in context_groups[context_key]
                )
                owner_context = build_candidate_context_image(
                    candidate.source_image,
                    candidate.owner_candidate.mask,
                )
                content.extend(
                    (
                        {
                            "type": "text",
                            "text": (
                                "OWNERSHIP-ONLY CONTEXT for attributes "
                                f"{shared_attribute_ids}\n"
                                "(use only for owner_binding_correct)"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _png_data_url(
                                    _resize_qwen_input_image(owner_context)
                                )
                            },
                        },
                    )
                )
                emitted_contexts.add(context_key)
            content.extend(
                (
                    {
                        "type": "text",
                        "text": (
                            f"{candidate.attribute_id} CROP-ONLY QUALITY TARGET "
                            "(judge recognizable, characteristic_appearance_visible, "
                            "and usable_as_attribute_condition from this isolated "
                            "transparent crop by itself)"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(
                                _resize_qwen_input_image(
                                    candidate.crop.convert("RGBA")
                                )
                            )
                        },
                    },
                )
            )
        payload = SubjectAttributeReviewBatch.model_validate_json(
            self._request(
                component="qwen_subject_attribute_review",
                system_prompt=REVIEW_SYSTEM_PROMPT,
                content=content,
            )
        )
        expected_ids = [candidate.attribute_id for candidate in candidates]
        if payload.owner_entity_id != owner.entity_id:
            raise ValueError("attribute review returned the wrong owner_entity_id")
        if [review.attribute_id for review in payload.reviews] != expected_ids:
            raise ValueError("attribute review IDs must match proposed attributes")
        return payload

    def review_attribute_bbox(
        self,
        *,
        bbox_candidate: Image.Image,
        owner_context: Image.Image,
        attribute_type: str,
        attribute_phrase: str,
    ) -> SubjectAttributeBboxReview:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Attribute type: {attribute_type}\n"
                    f"Attribute phrase: {attribute_phrase.strip()}\n"
                    "Image 1 is ownership-only context. Image 2 is the bbox target."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": _png_data_url(_resize_qwen_input_image(owner_context))
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": _png_data_url(_resize_qwen_input_image(bbox_candidate))
                },
            },
        ]
        return SubjectAttributeBboxReview.model_validate(
            self._request_structured(
                component="qwen_attribute_bbox_review",
                system_prompt=ATTRIBUTE_BBOX_REVIEW_SYSTEM_PROMPT,
                content=content,
                response_model=SubjectAttributeBboxReview,
            )
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class QwenSubjectAttributeCompletionJudge(QwenBooguReferenceEditJudge):
    """Strict comparative alpha-versus-completion reviewer."""

    def review(
        self,
        *,
        source_attribute: Image.Image,
        source_bbox: Image.Image,
        generated_candidate: Image.Image,
        attribute_type: str,
        attribute_phrase: str,
    ) -> SubjectAttributeCompletionReview:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Attribute type: {attribute_type}\n"
                    f"Original attribute phrase: {attribute_phrase.strip()}"
                ),
            }
        ]
        for label, image in (
            ("Image 1: isolated raw/source attribute crop", source_attribute),
            ("Image 2: source RGB bbox identity evidence only", source_bbox),
            ("Image 3: generated completion candidate", generated_candidate),
        ):
            content.extend(
                (
                    {"type": "text", "text": label},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_url(
                                _resize_qwen_input_image(image.convert("RGB"))
                            )
                        },
                    },
                )
            )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": ATTRIBUTE_COMPLETION_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        with model_profile_context(
            retry_index=0,
            metadata={"edit_operation": "complete_entity"},
        ):
            raw = self._request(messages, SubjectAttributeCompletionReview)
        for attempt in range(self.repair_retries + 1):
            try:
                payload = QwenSubjectAttributeClient._normalize_structured_verdict(
                    json.loads(raw),
                    SubjectAttributeCompletionReview,
                )
                return SubjectAttributeCompletionReview.model_validate(payload)
            except (TypeError, ValueError) as exc:
                if attempt >= self.repair_retries:
                    raise ValueError(
                        f"invalid Qwen subject attribute completion review: {exc}"
                    ) from exc
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Repair only the JSON structure while preserving the "
                            "original visual judgments. Every required boolean must "
                            "be present. Return exactly one JSON object and no extra "
                            f"fields. Validation error: {exc}\nRequired schema:\n"
                            + json.dumps(
                                SubjectAttributeCompletionReview.model_json_schema(),
                                ensure_ascii=False,
                            )
                        ),
                    },
                ]
                with model_profile_context(
                    retry_index=attempt + 1,
                    metadata={"edit_operation": "complete_entity"},
                ):
                    raw = self._request(
                        repair_messages,
                        SubjectAttributeCompletionReview,
                    )
        raise AssertionError("unreachable")


def evaluate_owner_eligibility(
    clip: ClipRecord,
    *,
    entity_id: str,
    has_usable_candidate_evidence: bool,
) -> OwnerEligibility:
    if clip.annotation is None or clip.annotation.status != "ready":
        return OwnerEligibility(False, "annotation_not_ready")
    if not clip.export.accepted:
        return OwnerEligibility(False, "visual_export_not_accepted")
    if clip.pairing is None or clip.pairing.status != "ready":
        return OwnerEligibility(False, "pairing_not_ready")
    entities = {entity.entity_id: entity for entity in clip.annotation.entities}
    entity = entities.get(entity_id)
    if entity is None:
        return OwnerEligibility(False, "annotation_entity_missing")
    if entity.reference_type != "subject":
        return OwnerEligibility(False, "owner_is_not_subject")
    if entity_id not in clip.pairing.retained_entity_ids:
        return OwnerEligibility(False, "subject_not_retained")
    references = {
        reference.entity_id: reference for reference in clip.references.entities
    }
    reference = references.get(entity_id)
    if reference is None or reference.status != "ready":
        return OwnerEligibility(False, "published_subject_reference_not_ready")
    integrity = clip.reference_integrity
    if integrity is not None:
        if integrity.status != "ready":
            return OwnerEligibility(False, "reference_integrity_not_ready")
        result = next(
            (item for item in integrity.entities if item.entity_id == entity_id),
            None,
        )
        if result is not None and result.status == "rejected":
            return OwnerEligibility(False, "reference_integrity_rejected")
    if not has_usable_candidate_evidence:
        return OwnerEligibility(False, "usable_owner_candidate_evidence_missing")
    return OwnerEligibility(True, "eligible")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError("mask must not be empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _bboxes_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def evaluate_ownership_geometry(
    attribute_mask: np.ndarray,
    owner_mask: np.ndarray,
    other_owner_masks: dict[str, np.ndarray],
    *,
    owner_padding_ratio: float,
) -> OwnershipGeometry:
    attribute = np.asarray(attribute_mask, dtype=bool)
    owner = np.asarray(owner_mask, dtype=bool)
    if attribute.ndim != 2 or owner.shape != attribute.shape:
        raise ValueError("attribute and owner masks must be equal two-dimensional masks")
    if (
        not isinstance(owner_padding_ratio, float)
        or not math.isfinite(owner_padding_ratio)
        or not 0 <= owner_padding_ratio <= 0.5
    ):
        raise ValueError("owner_padding_ratio must be a finite float in [0, 0.5]")
    attribute_area = int(attribute.sum())
    owner_area = int(owner.sum())
    if owner_area == 0:
        raise ValueError("owner mask must not be empty")
    if attribute_area == 0:
        return OwnershipGeometry(
            passed=False,
            reason="empty_attribute_mask",
            owner_overlap_ratio=0.0,
            maximum_other_owner_overlap_ratio=0.0,
            attribute_to_owner_area_ratio=0.0,
            near_owner_region=False,
            attribute_area_pixels=0,
            attribute_long_side_pixels=0,
            significant_component_count=0,
            largest_component_ratio=0.0,
            second_largest_component_ratio=0.0,
        )
    attribute_bbox = _bbox(attribute)
    owner_bbox = _bbox(owner)
    width = attribute_bbox[2] - attribute_bbox[0]
    height = attribute_bbox[3] - attribute_bbox[1]
    long_side = max(width, height)
    bbox_fill_ratio = attribute_area / (width * height)
    diagnostics = mask_component_diagnostics(attribute)
    owner_overlap = float(np.logical_and(attribute, owner).sum() / attribute_area)
    other_overlaps: list[float] = []
    for other_mask in other_owner_masks.values():
        other = np.asarray(other_mask, dtype=bool)
        if other.shape != attribute.shape:
            raise ValueError("other owner mask dimensions must match attribute mask")
        other_overlaps.append(
            float(np.logical_and(attribute, other).sum() / attribute_area)
        )
    maximum_other_overlap = max(other_overlaps, default=0.0)
    owner_long_side = max(
        owner_bbox[2] - owner_bbox[0],
        owner_bbox[3] - owner_bbox[1],
    )
    padding = math.ceil(owner_long_side * owner_padding_ratio)
    expanded_owner_bbox = (
        max(0, owner_bbox[0] - padding),
        max(0, owner_bbox[1] - padding),
        min(attribute.shape[1], owner_bbox[2] + padding),
        min(attribute.shape[0], owner_bbox[3] + padding),
    )
    near_owner = _bboxes_intersect(attribute_bbox, expanded_owner_bbox)
    attribute_to_owner_ratio = attribute_area / owner_area
    reason = "passed"
    if attribute_area < MIN_ATTRIBUTE_AREA_PIXELS:
        reason = "tiny_attribute_area"
    elif long_side < MIN_ATTRIBUTE_LONG_SIDE_PIXELS:
        reason = "tiny_attribute_long_side"
    elif diagnostics.severely_fragmented:
        reason = "severely_fragmented_attribute_mask"
    elif attribute_to_owner_ratio >= MAX_ATTRIBUTE_TO_OWNER_AREA_RATIO:
        reason = "attribute_mask_primarily_contains_owner"
    elif (
        maximum_other_overlap >= WRONG_OWNER_MINIMUM_OVERLAP_RATIO
        and maximum_other_overlap > owner_overlap
    ):
        reason = "attribute_primarily_belongs_to_other_subject"
    elif owner_overlap == 0.0 and not near_owner:
        reason = "attribute_not_near_intended_owner"
    return OwnershipGeometry(
        passed=reason == "passed",
        reason=reason,
        owner_overlap_ratio=owner_overlap,
        maximum_other_owner_overlap_ratio=maximum_other_overlap,
        attribute_to_owner_area_ratio=attribute_to_owner_ratio,
        near_owner_region=near_owner,
        attribute_area_pixels=attribute_area,
        attribute_long_side_pixels=long_side,
        significant_component_count=diagnostics.significant_component_count,
        largest_component_ratio=diagnostics.largest_component_ratio,
        second_largest_component_ratio=diagnostics.second_largest_component_ratio,
        bbox_fill_ratio=bbox_fill_ratio,
    )


def _clothing_geometry_rejection_reason(
    attribute_type: AttributeType,
    attribute_mask: np.ndarray,
    geometry: OwnershipGeometry,
) -> str | None:
    if attribute_type not in CLOTHING_ATTRIBUTE_TYPES or not geometry.passed:
        return None
    if geometry.attribute_to_owner_area_ratio >= CLOTHING_OWNER_LIKE_AREA_RATIO:
        return "clothing_mask_too_owner_like"
    x1, y1, x2, y2 = _bbox(np.asarray(attribute_mask, dtype=bool))
    width = x2 - x1
    height = y2 - y1
    bbox_aspect_ratio = max(width / height, height / width)
    if bbox_aspect_ratio >= CLOTHING_STRIP_ASPECT_RATIO:
        return "clothing_mask_too_strip_like"
    return None


def _type_specific_size_rejection_reason(
    attribute_type: AttributeType,
    geometry: OwnershipGeometry,
) -> str | None:
    if not geometry.passed:
        return None
    if (
        attribute_type == "hair"
        and geometry.attribute_long_side_pixels < HAIR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS
    ):
        return "hair_attribute_too_small"
    if (
        attribute_type == "headwear"
        and geometry.attribute_long_side_pixels
        < HEADWEAR_MIN_ATTRIBUTE_LONG_SIDE_PIXELS
    ):
        return "headwear_attribute_too_small"
    return None


def _duplicate_attribute_mask_conflicts(
    candidates: list[PendingAttributeCandidate],
) -> set[str]:
    conflicts: set[str] = set()
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if first.discovered.attribute_type == second.discovered.attribute_type:
                continue
            if mask_iou(first.attribute_mask, second.attribute_mask) >= (
                ATTRIBUTE_DUPLICATE_MASK_IOU
            ):
                conflicts.update((first.attribute_id, second.attribute_id))
    return conflicts


def prefer_attribute_candidate_frames(
    candidates: list[EntityReferenceCandidate],
    *,
    owner_reference_source_frame_index: int,
) -> list[EntityReferenceCandidate]:
    rank = {candidate.candidate_id: index for index, candidate in enumerate(candidates)}
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.source_frame_index == owner_reference_source_frame_index,
            rank[candidate.candidate_id],
        ),
    )


def _decode_owner_mask(
    masks: TrackedMasksArtifact,
    *,
    entity_id: str,
    slot: int,
) -> np.ndarray | None:
    tracked = masks.entities.get(entity_id)
    if tracked is None or tracked.status != "ready":
        return None
    frame = tracked.frames[slot]
    if not frame.present or not frame.track_valid or frame.area_pixels <= 0:
        return None
    decoded = np.asarray(decode_binary_mask(frame.rle), dtype=bool)
    if decoded.shape != (masks.height, masks.width):
        raise ValueError("decoded owner mask dimensions are invalid")
    return decoded


def _source_image(
    storage: RunStorage,
    candidate: EntityReferenceCandidate,
) -> Image.Image:
    path = (storage.root / candidate.image_path).resolve(strict=False)
    path.relative_to(storage.root.resolve(strict=False))
    with Image.open(path) as opened:
        opened.load()
        return opened.convert("RGB")


def _published_reference(clip: ClipRecord, entity_id: str) -> EntityReferenceState:
    reference = next(
        (
            item
            for item in clip.references.entities
            if item.entity_id == entity_id and item.status == "ready"
        ),
        None,
    )
    if reference is None:
        raise ValueError("eligible owner is missing its ready published reference")
    return reference


def _rejected_record(
    discovered: DiscoveredSubjectAttribute,
    *,
    attribute_id: str,
    owner_entity_id: str,
    reason: str,
    geometry: OwnershipGeometry | None = None,
    source_frame_index: int | None = None,
    source_frame_slot: int | None = None,
    owner_candidate_id: str | None = None,
    same_frame: bool | None = None,
    review: SubjectAttributeReview | None = None,
    completion_review: SubjectAttributeCompletionReview | None = None,
    gme_attempts: tuple[GmeAttributeScreenAttempt, ...] = (),
    selected_gme_attempt_index: int | None = None,
    completion_attempted: bool = False,
    completion_seed: int | None = None,
    completion_outcome: str | None = None,
) -> SubjectAttributeRecord:
    return SubjectAttributeRecord(
        attribute_id=attribute_id,
        owner_entity_id=owner_entity_id,
        attribute_type=discovered.attribute_type,
        phrase=discovered.phrase,
        grounding_prompt=discovered.grounding_prompt,
        status="rejected",
        image_path=None,
        source_frame_index=source_frame_index,
        source_frame_slot=source_frame_slot,
        owner_candidate_id=owner_candidate_id,
        same_frame_as_owner_reference=same_frame,
        sam3_prompt=discovered.grounding_prompt,
        ownership_geometry=geometry,
        review=review,
        completion_review=completion_review,
        gme_attempts=list(gme_attempts),
        selected_gme_attempt_index=selected_gme_attempt_index,
        completion_attempted=completion_attempted,
        completion_seed=completion_seed,
        completion_outcome=completion_outcome,
        reason=reason,
    )


def _accepted_record(
    candidate: PendingAttributeCandidate,
    *,
    output_root: Path,
    sample_id: str,
    owner_reference: EntityReferenceState,
    review: SubjectAttributeReview,
    completion_review: SubjectAttributeCompletionReview | None = None,
    final_selection: Literal["raw", "completed", "bbox"],
    completion_attempted: bool,
    completion_outcome: str,
    completion_seed: int | None = None,
    crop_padding_ratio: float,
) -> SubjectAttributeRecord:
    relative_path = _save_attribute_crop(
        output_root,
        sample_id=sample_id,
        attribute_id=candidate.attribute_id,
        crop=candidate.crop,
    )
    same_frame = (
        owner_reference.source_clip_uid in {None, sample_id}
        and candidate.owner_candidate.source_frame_index
        == owner_reference.source_frame_index
    )
    raw_crop = candidate.raw_crop if candidate.raw_crop is not None else candidate.crop
    alpha_path = relative_path
    if final_selection in {"completed", "bbox"}:
        alpha_path = _save_attribute_variant(
            output_root,
            sample_id=sample_id,
            attribute_id=candidate.attribute_id,
            variant="alpha",
            image=raw_crop.convert("RGBA"),
        )
    bbox_path = _save_attribute_variant(
        output_root,
        sample_id=sample_id,
        attribute_id=candidate.attribute_id,
        variant="bbox",
        image=_attribute_bbox_crop(
            candidate.source_image,
            candidate.attribute_mask,
            crop_padding_ratio=crop_padding_ratio,
        ),
    )
    variants = ReferenceVariantsState(
        alpha=ReferenceVariantState(
            image_path=alpha_path,
            status="accepted" if review.accepted else "rejected",
            reviewed=True,
            review_status="accepted" if review.accepted else "rejected",
            reason=review.reason,
            synthetic=False,
            source_frame_index=candidate.owner_candidate.source_frame_index,
        ),
        bbox=ReferenceVariantState(
            image_path=bbox_path,
            status="accepted" if final_selection == "bbox" else "available",
            reviewed=final_selection == "bbox",
            review_status=("accept" if final_selection == "bbox" else "not_reviewed"),
            reason=(
                "attribute_bbox_fallback_review_accepted"
                if final_selection == "bbox"
                else "source_attribute_bbox_materialized"
            ),
            synthetic=False,
            source_frame_index=candidate.owner_candidate.source_frame_index,
        ),
        generated_background=ReferenceVariantState(
            image_path=None,
            status="unavailable",
            reviewed=False,
            review_status="not_applicable",
            reason=_ATTRIBUTE_BACKGROUND_DISABLED_REASON,
            synthetic=True,
            source_frame_index=candidate.owner_candidate.source_frame_index,
        ),
    )
    default_variant: ReferenceDefaultVariant = (
        "bbox" if final_selection == "bbox" else "accepted_base"
    )
    if final_selection == "bbox":
        relative_path = bbox_path
    return SubjectAttributeRecord(
        attribute_id=candidate.attribute_id,
        owner_entity_id=candidate.owner_entity_id,
        attribute_type=candidate.discovered.attribute_type,
        phrase=candidate.discovered.phrase,
        grounding_prompt=candidate.discovered.grounding_prompt,
        status="accepted",
        image_path=relative_path,
        source_frame_index=candidate.owner_candidate.source_frame_index,
        source_frame_slot=candidate.owner_candidate.frame_slot,
        owner_candidate_id=candidate.owner_candidate.candidate_id,
        same_frame_as_owner_reference=same_frame,
        sam3_prompt=candidate.discovered.grounding_prompt,
        ownership_geometry=candidate.geometry,
        review=review,
        completion_review=completion_review,
        gme_attempts=list(candidate.gme_attempts),
        selected_gme_attempt_index=candidate.selected_gme_attempt_index,
        final_selection=final_selection,
        completion_attempted=completion_attempted,
        completion_seed=completion_seed,
        completion_outcome=completion_outcome,
        variants=variants,
        default_variant=default_variant,
        default_image_path=relative_path,
        default_reason=(
            "raw_attribute_review_accepted"
            if final_selection == "raw"
            else "attribute_completion_review_accepted"
            if final_selection == "completed"
            else "attribute_bbox_fallback_review_accepted"
        ),
        accepted_base_image_path=(relative_path if final_selection != "bbox" else None),
        reason="accepted",
    )


def _with_attribute_variants(
    record: SubjectAttributeRecord,
    candidate: PendingAttributeCandidate,
    *,
    output_root: Path,
    judge: AttributeVariantJudge | None,
    metrics: _AttributeVariantMetrics,
) -> SubjectAttributeRecord:
    if record.status != "accepted" or record.variants is None:
        return record
    metrics.bbox_variants_materialized += 1
    variants = record.variants
    generated_variant = ReferenceVariantState(
        image_path=None,
        status="unavailable",
        reviewed=False,
        review_status="not_applicable",
        reason=_ATTRIBUTE_BACKGROUND_DISABLED_REASON,
        synthetic=True,
        source_frame_index=record.source_frame_index,
    )
    bbox_variant = variants.bbox
    del candidate, output_root, judge
    default_variant = record.default_variant
    default_image_path = record.default_image_path
    default_reason = record.default_reason
    updated = record.model_copy(
        update={
            "image_path": default_image_path,
            "variants": variants.model_copy(
                update={
                    "bbox": bbox_variant,
                    "generated_background": generated_variant,
                }
            ),
            "default_variant": default_variant,
            "default_image_path": default_image_path,
            "default_reason": default_reason,
        }
    )
    return SubjectAttributeRecord.model_validate(updated.model_dump(mode="json"))


def _masked_crop_quality_rejection(
    crop: Image.Image,
    *,
    config: object,
) -> str | None:
    rgba = np.asarray(crop.convert("RGBA"))
    mask = rgba[..., 3] > 0
    if not mask.any():
        return "empty_mask"
    rgb = rgba[..., :3]
    luminance = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ) / 255.0
    if float(luminance[mask].mean()) < float(
        getattr(config, "minimum_mean_luminance")
    ):
        return "near_silhouette"
    if _masked_sharpness_score(rgb, mask) < float(
        getattr(config, "minimum_sharpness_score")
    ):
        return "too_blurry"
    components = mask_component_diagnostics(mask)
    if components.significant_component_count > int(
        getattr(config, "maximum_significant_components")
    ):
        return "too_fragmented"
    return None


def _save_completion_input(path: Path, crop: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        crop.convert("RGB").save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def attribute_completion_prompt(attribute_type: AttributeType) -> str:
    try:
        entity_name = _ATTRIBUTE_COMPLETION_ENTITY_NAME[attribute_type]
    except KeyError as exc:
        raise ValueError(
            f"attribute type is not eligible for completion: {attribute_type}"
        ) from exc
    return f"补全成完整的{entity_name}，去除掉破碎的部分，不添加其他内容"


def _attempt_attribute_completion(
    candidate: PendingAttributeCandidate,
    *,
    clip_uid: str,
    output_root: Path,
    completion_config: object,
    completion_backend: AttributeCompletionBackend,
    completion_judge: AttributeCompletionJudge,
    metrics: _CompletionSelectionMetrics,
    raw_usable: bool,
    crop_padding_ratio: float = 0.08,
) -> tuple[
    PendingAttributeCandidate | None,
    str,
    bool,
    SubjectAttributeCompletionReview | None,
    int | None,
]:
    metrics.attempts += 1
    metrics.attempts_by_type[candidate.discovered.attribute_type] += 1
    if raw_usable:
        metrics.raw_usable_attempts += 1
    else:
        metrics.raw_unusable_attempts += 1
    candidate_root = (
        output_root
        / "completion_candidates"
        / clip_uid
        / candidate.owner_entity_id
        / candidate.attribute_id
        / candidate.owner_candidate.candidate_id
    )
    source_path = candidate_root / "raw" / "source.png"
    output_path = candidate_root / "generated" / "candidate.png"
    _save_completion_input(source_path, candidate.crop)
    completion_seed: int | None = None
    started = time.perf_counter()
    failure_stage = "backend"
    try:
        instruction = attribute_completion_prompt(
            candidate.discovered.attribute_type
        )
        completion_seed = new_boogu_seed()
        completion = completion_backend.attribute_completion(
            source_path=source_path,
            output_path=output_path,
            instruction=instruction,
            seed=completion_seed,
        )
        elapsed = time.perf_counter() - started
        reported_seconds = completion.get("model_call_time_seconds", elapsed)
        if not isinstance(reported_seconds, (int, float)) or not math.isfinite(
            float(reported_seconds)
        ):
            raise ValueError("completion model-call time is invalid")
        metrics.model_call_time_seconds += max(0.0, float(reported_seconds))
        with Image.open(output_path) as opened:
            opened.load()
            completed_rgb = opened.convert("RGB")
        failure_stage = "identity_review"
        review_started = time.perf_counter()
        metrics.review_calls += 1
        review = completion_judge.review(
            source_attribute=candidate.crop,
            source_bbox=_attribute_bbox_crop(
                candidate.source_image,
                candidate.attribute_mask,
                crop_padding_ratio=crop_padding_ratio,
            ),
            generated_candidate=completed_rgb,
            attribute_type=candidate.discovered.attribute_type,
            attribute_phrase=candidate.discovered.phrase,
        )
        review_seconds = time.perf_counter() - review_started
        metrics.review_time_seconds += review_seconds
        if (
            not isinstance(review, SubjectAttributeCompletionReview)
            or review.verdict != "accept"
        ):
            metrics.qwen_review_rejects += 1
            metrics.identity_review_rejects += 1
            return (
                None,
                "completion_qwen_review_reject",
                False,
                review,
                completion_seed,
            )
    except Exception as exc:  # noqa: BLE001 - route to raw fallback or rejection
        metrics.failures += 1
        if failure_stage == "backend":
            metrics.backend_failures += 1
        return (
            None,
            f"completion_failed:{type(exc).__name__}:{exc}",
            True,
            None,
            completion_seed,
        )
    return (
        PendingAttributeCandidate(
            discovered=candidate.discovered,
            attribute_id=candidate.attribute_id,
            owner_entity_id=candidate.owner_entity_id,
            owner_candidate=candidate.owner_candidate,
            attribute_mask=candidate.attribute_mask,
            source_image=candidate.source_image,
            crop=completed_rgb,
            geometry=candidate.geometry,
            gme_attempts=candidate.gme_attempts,
            selected_gme_attempt_index=candidate.selected_gme_attempt_index,
            completion_attempted=True,
            completion_model_call_time_seconds=max(0.0, float(reported_seconds)),
            completion_review_time_seconds=review_seconds,
            raw_crop=(
                candidate.raw_crop
                if candidate.raw_crop is not None
                else candidate.crop
            ),
        ),
        "completion_identity_accepted",
        False,
        review,
        completion_seed,
    )


def _collect_attribute_candidates(
    *,
    discovered: DiscoveredSubjectAttribute,
    attribute_id: str,
    clip_uid: str,
    owner: AnnotationEntity,
    owner_candidates: list[EntityReferenceCandidate],
    owner_reference: EntityReferenceState,
    masks: TrackedMasksArtifact,
    storage: RunStorage,
    other_subject_ids: list[str],
    crop_padding_ratio: float,
    segmentation_backend: AttributeFrameSegmentationBackend,
    gme_screener: AttributeGmeScreener | None = None,
    gme_metrics: _GmeSelectionMetrics | None = None,
    completion_config: object | None = None,
    max_candidates: int = MAX_ATTRIBUTE_SOURCE_CANDIDATES,
) -> list[PendingAttributeCandidate] | SubjectAttributeRecord:
    if max_candidates < 1 or max_candidates > MAX_ATTRIBUTE_SOURCE_CANDIDATES:
        raise ValueError("attribute source candidate bound is invalid")
    owner_reference_is_local = owner_reference.source_clip_uid in {None, clip_uid}
    ordered = prefer_attribute_candidate_frames(
        owner_candidates,
        owner_reference_source_frame_index=(
            owner_reference.source_frame_index if owner_reference_is_local else -1
        ),
    )
    geometry_rejections: list[
        tuple[EntityReferenceCandidate, OwnershipGeometry]
    ] = []
    probe_failures: list[str] = []
    gme_rejections: list[
        tuple[EntityReferenceCandidate, OwnershipGeometry]
    ] = []
    gme_attempts: list[GmeAttributeScreenAttempt] = []
    active_gme_metrics = gme_metrics or _GmeSelectionMetrics()
    raw_quality_rejections: list[
        tuple[EntityReferenceCandidate, OwnershipGeometry, str]
    ] = []
    candidates: list[PendingAttributeCandidate] = []
    for owner_candidate_index, owner_candidate in enumerate(ordered):
        gme_rejected_on_frame = False
        frame_path = (storage.root / owner_candidate.image_path).resolve(strict=False)
        try:
            attribute_masks = segmentation_backend.segment_frame(
                frame_path=frame_path,
                frame_slot=owner_candidate.frame_slot,
                grounding_prompt=discovered.grounding_prompt,
            )
        except Exception as exc:  # noqa: BLE001 - try the next bounded candidate
            probe_failures.append(f"{type(exc).__name__}:{exc}")
            continue
        other_masks = {
            entity_id: decoded
            for entity_id in other_subject_ids
            if (
                decoded := _decode_owner_mask(
                    masks,
                    entity_id=entity_id,
                    slot=owner_candidate.frame_slot,
                )
            )
            is not None
        }
        for attribute_mask in attribute_masks:
            owner_mask = owner_candidate.mask
            geometry = evaluate_ownership_geometry(
                attribute_mask,
                owner_mask,
                other_masks,
                owner_padding_ratio=crop_padding_ratio,
            )
            if not geometry.passed:
                geometry_rejections.append((owner_candidate, geometry))
                continue
            size_rejection = _type_specific_size_rejection_reason(
                discovered.attribute_type,
                geometry,
            )
            if size_rejection is not None:
                geometry_rejections.append(
                    (
                        owner_candidate,
                        geometry.model_copy(
                            update={"passed": False, "reason": size_rejection}
                        ),
                    )
                )
                continue
            clothing_rejection = _clothing_geometry_rejection_reason(
                discovered.attribute_type,
                attribute_mask,
                geometry,
            )
            if clothing_rejection is not None:
                geometry_rejections.append(
                    (
                        owner_candidate,
                        geometry.model_copy(
                            update={"passed": False, "reason": clothing_rejection}
                        ),
                    )
                )
                continue
            source = _source_image(storage, owner_candidate)
            crop, _ = build_reference_crop(
                source,
                attribute_mask,
                crop_padding_ratio=crop_padding_ratio,
            )
            if bool(
                completion_config is not None
                and getattr(completion_config, "enabled", False)
            ):
                raw_rejection = _masked_crop_quality_rejection(
                    crop,
                    config=completion_config,
                )
                if raw_rejection is not None:
                    raw_quality_rejections.append(
                        (owner_candidate, geometry, raw_rejection)
                    )
                    continue
            selected_gme_attempt_index: int | None = None
            if gme_screener is not None:
                active_gme_metrics.calls += 1
                gme_started = time.perf_counter()
                try:
                    model_name = str(
                        getattr(
                            getattr(gme_screener, "config", None),
                            "model_name",
                            "injected_gme",
                        )
                    )
                    with profile_model_call(
                        component="gme_subject_attribute_screen",
                        operation="relative_margin_v1",
                        retry_index=0,
                        model=model_name,
                        input_text_chars=(
                            len(discovered.phrase)
                            + len(discovered.grounding_prompt)
                        ),
                        input_image_count=1,
                        metadata={
                            "clip_uid": clip_uid,
                            "owner_entity_id": owner.entity_id,
                            "attribute_id": attribute_id,
                            "attribute_type": discovered.attribute_type,
                            "owner_candidate_id": owner_candidate.candidate_id,
                            "source_frame_slot": owner_candidate.frame_slot,
                        },
                    ):
                        gme_result = gme_screener.screen(
                            crop=crop,
                            phrase=discovered.phrase,
                            attribute_type=discovered.attribute_type,
                        )
                except Exception as exc:  # noqa: BLE001 - fail open to Qwen
                    active_gme_metrics.failures += 1
                    gme_attempts.append(
                        GmeAttributeScreenAttempt(
                            owner_candidate_id=owner_candidate.candidate_id,
                            source_frame_slot=owner_candidate.frame_slot,
                            source_frame_index=owner_candidate.source_frame_index,
                            bbox_fill_ratio=geometry.bbox_fill_ratio,
                            status="failed",
                            reason=f"gme_failed_open:{type(exc).__name__}:{exc}",
                        )
                    )
                    selected_gme_attempt_index = len(gme_attempts) - 1
                else:
                    active_gme_metrics.candidates_screened += 1
                    active_gme_metrics.candidates_passed += int(gme_result.passed)
                    active_gme_metrics.candidates_rejected += int(
                        not gme_result.passed
                    )
                    gme_attempts.append(
                        GmeAttributeScreenAttempt(
                            owner_candidate_id=owner_candidate.candidate_id,
                            source_frame_slot=owner_candidate.frame_slot,
                            source_frame_index=owner_candidate.source_frame_index,
                            bbox_fill_ratio=geometry.bbox_fill_ratio,
                            status="passed" if gme_result.passed else "rejected",
                            positive_score=gme_result.positive_score,
                            negative_scores=gme_result.negative_scores,
                            max_negative_score=gme_result.max_negative_score,
                            margin=gme_result.margin,
                            passed=gme_result.passed,
                            reason=gme_result.reason,
                        )
                    )
                    if not gme_result.passed:
                        gme_rejections.append((owner_candidate, geometry))
                        gme_rejected_on_frame = True
                        continue
                    selected_gme_attempt_index = len(gme_attempts) - 1
                finally:
                    active_gme_metrics.model_call_time_seconds += (
                        time.perf_counter() - gme_started
                    )
            candidates.append(PendingAttributeCandidate(
                discovered=discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner.entity_id,
                owner_candidate=owner_candidate,
                attribute_mask=attribute_mask,
                source_image=source,
                crop=crop,
                geometry=geometry,
                gme_attempts=tuple(gme_attempts),
                selected_gme_attempt_index=selected_gme_attempt_index,
                raw_crop=crop,
            ))
            break
        if len(candidates) >= max_candidates:
            return candidates
        if gme_rejected_on_frame and owner_candidate_index + 1 < len(ordered):
            active_gme_metrics.retried_next_frame += 1
    if candidates:
        return candidates
    if gme_rejections:
        owner_candidate, geometry = gme_rejections[0]
        return _rejected_record(
            discovered,
            attribute_id=attribute_id,
            owner_entity_id=owner.entity_id,
            reason="gme_semantic_quality_reject",
            geometry=geometry,
            source_frame_index=owner_candidate.source_frame_index,
            source_frame_slot=owner_candidate.frame_slot,
            owner_candidate_id=owner_candidate.candidate_id,
            same_frame=(
                owner_reference_is_local
                and owner_candidate.source_frame_index
                == owner_reference.source_frame_index
            ),
            gme_attempts=tuple(gme_attempts),
        )
    if geometry_rejections:
        owner_candidate, geometry = geometry_rejections[0]
        return _rejected_record(
            discovered,
            attribute_id=attribute_id,
            owner_entity_id=owner.entity_id,
            reason=f"ownership_geometry:{geometry.reason}",
            geometry=geometry,
            source_frame_index=owner_candidate.source_frame_index,
            source_frame_slot=owner_candidate.frame_slot,
            owner_candidate_id=owner_candidate.candidate_id,
            same_frame=(
                owner_reference_is_local
                and owner_candidate.source_frame_index
                == owner_reference.source_frame_index
            ),
        )
    if raw_quality_rejections:
        owner_candidate, geometry, reason = raw_quality_rejections[0]
        return _rejected_record(
            discovered,
            attribute_id=attribute_id,
            owner_entity_id=owner.entity_id,
            reason=f"completion_precheck:{reason}",
            geometry=geometry,
            source_frame_index=owner_candidate.source_frame_index,
            source_frame_slot=owner_candidate.frame_slot,
            owner_candidate_id=owner_candidate.candidate_id,
            same_frame=(
                owner_reference_is_local
                and owner_candidate.source_frame_index
                == owner_reference.source_frame_index
            ),
        )
    reason = "sam3_no_owner_candidate_frame_mask"
    if probe_failures:
        reason = f"sam3_candidate_frame_probe_failed:{probe_failures[0]}"
    return _rejected_record(
        discovered,
        attribute_id=attribute_id,
        owner_entity_id=owner.entity_id,
        reason=reason,
    )


def _select_attribute_candidate(
    **kwargs: object,
) -> PendingAttributeCandidate | SubjectAttributeRecord:
    """Compatibility wrapper returning the highest-ranked bounded candidate."""

    selected = _collect_attribute_candidates(**kwargs, max_candidates=1)
    if isinstance(selected, SubjectAttributeRecord):
        return selected
    return selected[0]


def _save_attribute_crop(
    output_root: Path,
    *,
    sample_id: str,
    attribute_id: str,
    crop: Image.Image,
) -> str:
    destination = output_root / "references" / sample_id / f"{attribute_id}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        crop.save(temporary, format="PNG")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.relative_to(output_root).as_posix()


def _save_attribute_variant(
    output_root: Path,
    *,
    sample_id: str,
    attribute_id: str,
    variant: Literal["alpha", "bbox", "generated_background"],
    image: Image.Image,
) -> str:
    destination = (
        output_root
        / "variants"
        / sample_id
        / attribute_id
        / f"{variant}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        image.save(temporary, format="PNG")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.relative_to(output_root).as_posix()


def _attribute_bbox_crop(
    source_image: Image.Image,
    attribute_mask: np.ndarray,
    *,
    crop_padding_ratio: float,
) -> Image.Image:
    mask = np.asarray(attribute_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("attribute bbox requires a non-empty 2D mask")
    x1, y1, x2, y2 = _bbox(mask)
    padding = math.ceil(max(x2 - x1, y2 - y1) * crop_padding_ratio)
    bbox = (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(source_image.width, x2 + padding),
        min(source_image.height, y2 + padding),
    )
    return source_image.convert("RGB").crop(bbox)


def _validate_attribute_png(
    output_root: Path,
    relative_path: str,
    *,
    final_selection: Literal["raw", "completed", "bbox"] = "raw",
    expected_mode: Literal["RGB", "RGBA"] | None = None,
) -> None:
    path = (output_root / relative_path).resolve(strict=False)
    path.relative_to(output_root.resolve(strict=False))
    with Image.open(path) as opened:
        opened.load()
        if opened.format != "PNG":
            raise ValueError("accepted attribute artifact must be PNG")
        mode = expected_mode or (
            "RGB" if final_selection in {"completed", "bbox"} else "RGBA"
        )
        if mode == "RGB":
            if opened.mode != "RGB":
                if expected_mode is None and final_selection in {"completed", "bbox"}:
                    raise ValueError("completed attribute artifact must be RGB PNG")
                raise ValueError("selected attribute artifact must be RGB PNG")
            return
        if opened.mode != "RGBA":
            raise ValueError("raw attribute artifact must be RGBA PNG")
        pixels = np.asarray(opened)
    alpha = pixels[..., 3]
    if not np.any(alpha == 255) or not np.isin(alpha, (0, 255)).all():
        raise ValueError("accepted attribute alpha must be non-empty and binary")
    if np.any(pixels[..., :3][alpha == 0] != 255):
        raise ValueError("transparent attribute RGB pixels must be white")


def _load_cached_owner_artifact(
    path: Path,
    *,
    output_root: Path,
    sample_id: str,
    owner_entity_id: str,
    attribute_id_start: int,
    expected_gme_screen_mode: str | None = None,
    expected_completion_mode: str | None = None,
) -> OwnerEnrichmentArtifact | None:
    if not path.is_file():
        return None
    try:
        artifact = OwnerEnrichmentArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            artifact.sample_id != sample_id
            or artifact.owner_entity_id != owner_entity_id
            or artifact.attribute_id_start != attribute_id_start
            or artifact.gme_screen_mode != expected_gme_screen_mode
            or artifact.completion_mode != expected_completion_mode
        ):
            return None
        for record in artifact.records:
            if record.status == "accepted":
                assert record.image_path is not None
                _validate_attribute_png(
                    output_root,
                    record.image_path,
                    final_selection=record.final_selection or "raw",
                    expected_mode=(
                        "RGB"
                        if record.default_variant
                        in {"bbox", "generated_background"}
                        else None
                    ),
                )
                if record.variants is not None:
                    for variant_name, variant in (
                        ("alpha", record.variants.alpha),
                        ("bbox", record.variants.bbox),
                        (
                            "generated_background",
                            record.variants.generated_background,
                        ),
                    ):
                        if variant.image_path is None:
                            continue
                        _validate_attribute_png(
                            output_root,
                            variant.image_path,
                            expected_mode=(
                                "RGBA" if variant_name == "alpha" else "RGB"
                            ),
                        )
                    assert record.accepted_base_image_path is not None
                    _validate_attribute_png(
                        output_root,
                        record.accepted_base_image_path,
                        final_selection=record.final_selection or "raw",
                    )
        return artifact
    except (OSError, ValueError):
        return None


def _owner_artifact_path(
    output_root: Path,
    *,
    sample_id: str,
    owner_entity_id: str,
) -> Path:
    return output_root / "owners" / sample_id / f"{owner_entity_id}.json"


def _clip_sample_path(output_root: Path, *, sample_id: str) -> Path:
    return output_root / "samples" / f"{sample_id}.json"


def _process_owner(
    *,
    config: V3Config,
    storage: RunStorage,
    output_root: Path,
    clip: ClipRecord,
    owner: AnnotationEntity,
    owner_candidates: list[EntityReferenceCandidate],
    masks: TrackedMasksArtifact,
    attribute_id_start: int,
    discovery_client: SubjectAttributeDiscoveryClient,
    review_client: SubjectAttributeReviewClient,
    segmentation_backend: AttributeFrameSegmentationBackend,
    gme_screener: AttributeGmeScreener | None = None,
    completion_backend: AttributeCompletionBackend | None = None,
    completion_judge: AttributeCompletionJudge | None = None,
) -> OwnerEnrichmentArtifact:
    metrics = OwnerEnrichmentMetrics(discovery_calls=1)
    gme_screen_mode = (
        getattr(getattr(config, "subject_attribute_gme", None), "screen_mode", None)
        if gme_screener is not None
        else None
    )
    completion_config = getattr(
        getattr(config, "subject_attributes", None),
        "completion",
        SubjectAttributeCompletionConfig(),
    )
    completion_mode = (
        "boogu_completion_v1"
        if completion_config.enabled
        else None
    )
    owner_reference = _published_reference(clip, owner.entity_id)
    source_images = {
        candidate.image_path: _source_image(storage, candidate)
        for candidate in owner_candidates
    }
    discovery_started = time.perf_counter()
    try:
        discovery = discovery_client.discover(
            owner=owner,
            owner_candidates=owner_candidates,
            source_images=source_images,
        )
    except Exception as exc:  # noqa: BLE001 - isolate one eligible owner
        elapsed = time.perf_counter() - discovery_started
        return OwnerEnrichmentArtifact(
            sample_id=clip.clip_uid,
            owner_entity_id=owner.entity_id,
            owner_is_human=None,
            attribute_id_start=attribute_id_start,
            owner_phrase=owner.phrase,
            owner_grounding_prompt=owner.grounding_prompt,
            records=[],
            metrics=metrics.model_copy(
                update={
                    "qwen_model_call_time_seconds": elapsed,
                    "failures": 1,
                }
            ),
            gme_screen_mode=gme_screen_mode,
            completion_mode=completion_mode,
            failure_reason=f"discovery_failed:{type(exc).__name__}:{exc}",
        )
    qwen_seconds = time.perf_counter() - discovery_started
    if not discovery.owner_is_human:
        return OwnerEnrichmentArtifact(
            sample_id=clip.clip_uid,
            owner_entity_id=owner.entity_id,
            owner_is_human=False,
            attribute_id_start=attribute_id_start,
            owner_phrase=owner.phrase,
            owner_grounding_prompt=owner.grounding_prompt,
            records=[],
            metrics=metrics.model_copy(
                update={"qwen_model_call_time_seconds": qwen_seconds}
            ),
            gme_screen_mode=gme_screen_mode,
            completion_mode=completion_mode,
        )
    discovered_by_type = Counter(
        attribute.attribute_type for attribute in discovery.attributes
    )
    other_subject_ids = [
        entity.entity_id
        for entity in clip.annotation.entities
        if (
            entity.reference_type == "subject"
            and entity.entity_id != owner.entity_id
        )
    ]
    pending: list[PendingAttributeCandidate] = []
    candidate_options_by_id: dict[str, list[PendingAttributeCandidate]] = {}
    records_by_id: dict[str, SubjectAttributeRecord] = {}
    sam_seconds = 0.0
    gme_metrics = _GmeSelectionMetrics()
    completion_metrics = _CompletionSelectionMetrics()
    processing_failures = 0
    for offset, discovered in enumerate(discovery.attributes):
        attribute_id = f"a{attribute_id_start + offset}"
        sam_started = time.perf_counter()
        gme_seconds_before = gme_metrics.model_call_time_seconds
        try:
            selected = _collect_attribute_candidates(
                discovered=discovered,
                attribute_id=attribute_id,
                clip_uid=clip.clip_uid,
                owner=owner,
                owner_candidates=owner_candidates,
                owner_reference=owner_reference,
                masks=masks,
                storage=storage,
                other_subject_ids=other_subject_ids,
                crop_padding_ratio=config.pair.crop_padding_ratio,
                segmentation_backend=segmentation_backend,
                gme_screener=gme_screener,
                gme_metrics=gme_metrics,
                completion_config=completion_config,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one attribute
            processing_failures += 1
            selected = _rejected_record(
                discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner.entity_id,
                reason=f"attribute_processing_failed:{type(exc).__name__}:{exc}",
            )
        selection_seconds = time.perf_counter() - sam_started
        gme_seconds = (
            gme_metrics.model_call_time_seconds - gme_seconds_before
        )
        sam_seconds += max(
            0.0,
            selection_seconds - gme_seconds,
        )
        if isinstance(selected, SubjectAttributeRecord):
            records_by_id[attribute_id] = selected
        else:
            candidate_options_by_id[attribute_id] = selected
            pending.append(selected[0])

    duplicate_conflicts = _duplicate_attribute_mask_conflicts(pending)
    if duplicate_conflicts:
        retained_pending: list[PendingAttributeCandidate] = []
        for candidate in pending:
            if candidate.attribute_id not in duplicate_conflicts:
                retained_pending.append(candidate)
                continue
            same_frame = (
                owner_reference.source_clip_uid in {None, clip.clip_uid}
                and candidate.owner_candidate.source_frame_index
                == owner_reference.source_frame_index
            )
            records_by_id[candidate.attribute_id] = _rejected_record(
                candidate.discovered,
                attribute_id=candidate.attribute_id,
                owner_entity_id=owner.entity_id,
                reason="attribute_mask_duplicate_conflict",
                geometry=candidate.geometry,
                source_frame_index=candidate.owner_candidate.source_frame_index,
                source_frame_slot=candidate.owner_candidate.frame_slot,
                owner_candidate_id=candidate.owner_candidate.candidate_id,
                same_frame=same_frame,
                gme_attempts=candidate.gme_attempts,
                selected_gme_attempt_index=(
                    candidate.selected_gme_attempt_index
                ),
            )
        pending = retained_pending
        for attribute_id in duplicate_conflicts:
            candidate_options_by_id.pop(attribute_id, None)

    new_review_calls = 0
    new_review_failures = 0
    new_review_failure: str | None = None
    second_candidate_attempts = 0
    completion_candidate_accepted = [0, 0]
    new_variant_metrics = _AttributeVariantMetrics()
    selected_candidate_by_id: dict[str, PendingAttributeCandidate] = {}
    raw_fallback_pool: dict[
        str, list[tuple[PendingAttributeCandidate, SubjectAttributeReview]]
    ] = {}
    semantic_candidate_pool: dict[
        str, list[tuple[PendingAttributeCandidate, SubjectAttributeReview]]
    ] = {}
    completion_provenance: dict[
        str, tuple[str, SubjectAttributeCompletionReview | None, int | None]
    ] = {}
    unresolved = {candidate.attribute_id for candidate in pending}

    for candidate_rank in range(MAX_ATTRIBUTE_SOURCE_CANDIDATES):
        round_candidates = [
            options[candidate_rank]
            for attribute_id, options in candidate_options_by_id.items()
            if attribute_id in unresolved and candidate_rank < len(options)
        ]
        if not round_candidates:
            continue
        if candidate_rank == 1:
            second_candidate_attempts += len(round_candidates)
        new_review_calls += 1
        review_started = time.perf_counter()
        round_reviews: dict[str, SubjectAttributeReview] = {}
        round_failure: str | None = None
        try:
            batch = review_client.review(owner=owner, candidates=round_candidates)
            round_reviews = {review.attribute_id: review for review in batch.reviews}
        except Exception as exc:  # noqa: BLE001 - fail closed for this review round
            round_failure = f"review_failed:{type(exc).__name__}:{exc}"
            new_review_failure = round_failure
            new_review_failures += 1
        qwen_seconds += time.perf_counter() - review_started

        for candidate in round_candidates:
            attribute_id = candidate.attribute_id
            if attribute_id not in unresolved:
                continue
            review = round_reviews.get(attribute_id)
            same_frame = (
                owner_reference.source_clip_uid in {None, clip.clip_uid}
                and candidate.owner_candidate.source_frame_index
                == owner_reference.source_frame_index
            )
            if review is None:
                if candidate_rank + 1 < len(candidate_options_by_id[attribute_id]):
                    continue
                records_by_id[attribute_id] = _rejected_record(
                    candidate.discovered,
                    attribute_id=attribute_id,
                    owner_entity_id=owner.entity_id,
                    reason=round_failure or "recognizability_review_missing",
                    geometry=candidate.geometry,
                    source_frame_index=candidate.owner_candidate.source_frame_index,
                    source_frame_slot=candidate.owner_candidate.frame_slot,
                    owner_candidate_id=candidate.owner_candidate.candidate_id,
                    same_frame=same_frame,
                    gme_attempts=candidate.gme_attempts,
                    selected_gme_attempt_index=candidate.selected_gme_attempt_index,
                )
                unresolved.discard(attribute_id)
                continue

            completion_metrics.raw_review_accepted += int(review.accepted)
            repair_recommended = bool(
                review.matches_attribute
                and review.owner_binding_correct
                and not review.structure_complete
                and review.completion_recommended
            )
            completion_metrics.raw_review_repair_recommended += int(
                repair_recommended
            )
            completion_metrics.raw_review_hard_rejected += int(
                not review.accepted
            )
            semantic_owner_correct = bool(
                review.matches_attribute and review.owner_binding_correct
            )
            if semantic_owner_correct:
                semantic_candidate_pool.setdefault(attribute_id, []).append(
                    (candidate, review)
                )
            if review.accepted:
                raw_fallback_pool.setdefault(attribute_id, []).append(
                    (candidate, review)
                )

            completion_eligible = bool(
                completion_config.enabled
                and candidate.discovered.attribute_type
                in completion_config.eligible_types
            )
            if not semantic_owner_correct:
                if candidate_rank + 1 < len(candidate_options_by_id[attribute_id]):
                    continue
                records_by_id[attribute_id] = _rejected_record(
                    candidate.discovered,
                    attribute_id=attribute_id,
                    owner_entity_id=owner.entity_id,
                    reason=f"recognizability:{review.reason}",
                    geometry=candidate.geometry,
                    source_frame_index=candidate.owner_candidate.source_frame_index,
                    source_frame_slot=candidate.owner_candidate.frame_slot,
                    owner_candidate_id=candidate.owner_candidate.candidate_id,
                    same_frame=same_frame,
                    review=review,
                    gme_attempts=candidate.gme_attempts,
                    selected_gme_attempt_index=candidate.selected_gme_attempt_index,
                )
                unresolved.discard(attribute_id)
                continue

            if not review.sufficient_source_evidence:
                if candidate_rank + 1 < len(candidate_options_by_id[attribute_id]):
                    continue
                # Keep only semantic provenance for the last-resort bbox route.
                continue

            if not completion_eligible:
                if review.accepted:
                    selected_candidate_by_id[attribute_id] = candidate
                    records_by_id[attribute_id] = _accepted_record(
                        candidate,
                        output_root=output_root,
                        sample_id=clip.clip_uid,
                        owner_reference=owner_reference,
                        review=review,
                        final_selection="raw",
                        completion_attempted=False,
                        completion_outcome="not_attempted",
                        crop_padding_ratio=config.pair.crop_padding_ratio,
                    )
                    unresolved.discard(attribute_id)
                elif candidate_rank + 1 >= len(
                    candidate_options_by_id[attribute_id]
                ):
                    # The last-resort bbox route is handled after both raw rounds.
                    pass
                continue

            if completion_backend is None or completion_judge is None:
                completed = None
                completion_review = None
                completion_seed = None
                completion_reason = "completion_failed:backend_not_configured"
                completion_metrics.attempts += 1
                completion_metrics.attempts_by_type[
                    candidate.discovered.attribute_type
                ] += 1
                if review.accepted:
                    completion_metrics.raw_usable_attempts += 1
                else:
                    completion_metrics.raw_unusable_attempts += 1
                completion_metrics.failures += 1
                completion_metrics.backend_failures += 1
            else:
                (
                    completed,
                    completion_reason,
                    _,
                    completion_review,
                    completion_seed,
                ) = _attempt_attribute_completion(
                    candidate,
                    clip_uid=clip.clip_uid,
                    output_root=output_root,
                    completion_config=completion_config,
                    completion_backend=completion_backend,
                    completion_judge=completion_judge,
                    metrics=completion_metrics,
                    raw_usable=review.accepted,
                    crop_padding_ratio=config.pair.crop_padding_ratio,
                )
            completion_provenance[attribute_id] = (
                completion_reason,
                completion_review,
                completion_seed,
            )
            if completed is not None:
                assert completion_review is not None
                completion_metrics.accepted += 1
                completion_metrics.selected_completed += 1
                completion_metrics.accepted_by_type[
                    candidate.discovered.attribute_type
                ] += 1
                completion_candidate_accepted[candidate_rank] += 1
                selected_candidate_by_id[attribute_id] = completed
                records_by_id[attribute_id] = _accepted_record(
                    completed,
                    output_root=output_root,
                    sample_id=clip.clip_uid,
                    owner_reference=owner_reference,
                    review=review,
                    completion_review=completion_review,
                    final_selection="completed",
                    completion_attempted=True,
                    completion_seed=completion_seed,
                    completion_outcome="selected_completed",
                    crop_padding_ratio=config.pair.crop_padding_ratio,
                )
                unresolved.discard(attribute_id)

    variant_judge = (
        review_client
        if callable(getattr(review_client, "review_attribute_bbox", None))
        else None
    )
    for attribute_id in list(unresolved):
        raw_pool = raw_fallback_pool.get(attribute_id, [])
        completion_reason, completion_review, completion_seed = (
            completion_provenance.get(
                attribute_id,
                ("not_attempted", None, None),
            )
        )
        if raw_pool:
            candidate, review = raw_pool[0]
            completion_attempted = attribute_id in completion_provenance
            if completion_attempted:
                completion_metrics.fallback_to_raw += 1
            selected_candidate_by_id[attribute_id] = candidate
            records_by_id[attribute_id] = _accepted_record(
                candidate,
                output_root=output_root,
                sample_id=clip.clip_uid,
                owner_reference=owner_reference,
                review=review,
                completion_review=completion_review,
                final_selection="raw",
                completion_attempted=completion_attempted,
                completion_seed=completion_seed,
                completion_outcome=completion_reason,
                crop_padding_ratio=config.pair.crop_padding_ratio,
            )
            unresolved.discard(attribute_id)
            continue

        semantic_pool = semantic_candidate_pool.get(attribute_id, [])
        if not semantic_pool or variant_judge is None:
            options = candidate_options_by_id[attribute_id]
            candidate = options[min(len(options) - 1, 1)]
            review = semantic_pool[0][1] if semantic_pool else None
            if attribute_id in completion_provenance:
                completion_metrics.rejected += 1
            records_by_id[attribute_id] = _rejected_record(
                candidate.discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner.entity_id,
                reason=(
                    f"recognizability:{review.reason}"
                    if semantic_pool
                    else "recognizability:no_semantic_owner_candidate"
                ),
                geometry=candidate.geometry,
                source_frame_index=candidate.owner_candidate.source_frame_index,
                source_frame_slot=candidate.owner_candidate.frame_slot,
                owner_candidate_id=candidate.owner_candidate.candidate_id,
                same_frame=(
                    owner_reference.source_clip_uid in {None, clip.clip_uid}
                    and candidate.owner_candidate.source_frame_index
                    == owner_reference.source_frame_index
                ),
                review=review,
                completion_review=completion_review,
                completion_attempted=attribute_id in completion_provenance,
                completion_seed=completion_seed,
                completion_outcome=completion_reason,
            )
            unresolved.discard(attribute_id)
            continue

        candidate, review = semantic_pool[0]
        new_variant_metrics.bbox_reviews_attempted += 1
        bbox_image = _attribute_bbox_crop(
            candidate.source_image,
            candidate.attribute_mask,
            crop_padding_ratio=config.pair.crop_padding_ratio,
        )
        owner_context = build_candidate_context_image(
            candidate.source_image,
            candidate.owner_candidate.mask,
        )
        try:
            bbox_review = variant_judge.review_attribute_bbox(
                bbox_candidate=bbox_image,
                owner_context=owner_context,
                attribute_type=candidate.discovered.attribute_type,
                attribute_phrase=candidate.discovered.phrase,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed
            bbox_review = None
            bbox_reason = f"attribute_bbox_judge_failed:{type(exc).__name__}:{exc}"
        else:
            bbox_reason = bbox_review.reason
        if bbox_review is not None and bbox_review.verdict == "accept":
            selected_candidate_by_id[attribute_id] = candidate
            records_by_id[attribute_id] = _accepted_record(
                candidate,
                output_root=output_root,
                sample_id=clip.clip_uid,
                owner_reference=owner_reference,
                review=review,
                completion_review=completion_review,
                final_selection="bbox",
                completion_attempted=attribute_id in completion_provenance,
                completion_seed=completion_seed,
                completion_outcome="bbox_fallback_accepted",
                crop_padding_ratio=config.pair.crop_padding_ratio,
            )
        else:
            if attribute_id in completion_provenance:
                completion_metrics.rejected += 1
            records_by_id[attribute_id] = _rejected_record(
                candidate.discovered,
                attribute_id=attribute_id,
                owner_entity_id=owner.entity_id,
                reason=f"attribute_bbox_fallback_rejected:{bbox_reason}",
                geometry=candidate.geometry,
                source_frame_index=candidate.owner_candidate.source_frame_index,
                source_frame_slot=candidate.owner_candidate.frame_slot,
                owner_candidate_id=candidate.owner_candidate.candidate_id,
                same_frame=(
                    owner_reference.source_clip_uid in {None, clip.clip_uid}
                    and candidate.owner_candidate.source_frame_index
                    == owner_reference.source_frame_index
                ),
                review=review,
                completion_review=completion_review,
                completion_attempted=attribute_id in completion_provenance,
                completion_seed=completion_seed,
                completion_outcome=completion_reason,
            )
        unresolved.discard(attribute_id)

    for attribute_id, candidate in selected_candidate_by_id.items():
        records_by_id[attribute_id] = _with_attribute_variants(
            records_by_id[attribute_id],
            candidate,
            output_root=output_root,
            judge=None,
            metrics=new_variant_metrics,
        )

    review_calls = new_review_calls
    review_failures = new_review_failures
    review_failure = new_review_failure
    variant_metrics = new_variant_metrics
    ordered_records = [
        records_by_id[f"a{attribute_id_start + offset}"]
        for offset in range(len(discovery.attributes))
    ]
    deterministic_rejects = sum(
        record.reason.startswith("ownership_geometry:")
        or record.reason == "attribute_mask_duplicate_conflict"
        or record.reason.startswith("completion_precheck:")
        or record.reason.startswith("completion_postcheck:")
        for record in ordered_records
    )
    recognizability_rejects = sum(
        record.status == "rejected"
        and (
            record.reason.startswith("recognizability:")
            or record.reason.startswith("review_failed:")
            or record.reason == "recognizability_review_missing"
            or record.reason.startswith("completion_final_review")
        )
        for record in ordered_records
    )
    accepted = [record for record in ordered_records if record.status == "accepted"]
    failures = (
        review_failures
        + processing_failures
        + completion_metrics.failures
    )
    return OwnerEnrichmentArtifact(
        sample_id=clip.clip_uid,
        owner_entity_id=owner.entity_id,
        owner_is_human=True,
        attribute_id_start=attribute_id_start,
        owner_phrase=owner.phrase,
        owner_grounding_prompt=owner.grounding_prompt,
        records=ordered_records,
        metrics=OwnerEnrichmentMetrics(
            discovery_calls=1,
            review_calls=review_calls,
            sam3_attempts=len(discovery.attributes),
            discovered_by_type=dict(sorted(discovered_by_type.items())),
            deterministic_ownership_rejects=deterministic_rejects,
            recognizability_rejects=recognizability_rejects,
            accepted_attributes=len(accepted),
            same_frame_accepted=sum(
                record.same_frame_as_owner_reference is True for record in accepted
            ),
            different_frame_accepted=sum(
                record.same_frame_as_owner_reference is False for record in accepted
            ),
            qwen_model_call_time_seconds=(
                qwen_seconds + completion_metrics.review_time_seconds
            ),
            sam3_model_call_time_seconds=(
                sam_seconds + completion_metrics.sam3_time_seconds
            ),
            gme_calls=gme_metrics.calls,
            gme_candidates_screened=gme_metrics.candidates_screened,
            gme_candidates_passed=gme_metrics.candidates_passed,
            gme_candidates_rejected=gme_metrics.candidates_rejected,
            gme_retried_next_frame=gme_metrics.retried_next_frame,
            gme_failures=gme_metrics.failures,
            gme_model_call_time_seconds=gme_metrics.model_call_time_seconds,
            completion_attempts=completion_metrics.attempts,
            completion_accepted=completion_metrics.accepted,
            completion_rejected=completion_metrics.rejected,
            completion_failures=completion_metrics.failures,
            completion_qwen_review_rejects=(
                completion_metrics.qwen_review_rejects
            ),
            raw_attribute_review_accepted=(
                completion_metrics.raw_review_accepted
            ),
            raw_attribute_review_repair_recommended=(
                completion_metrics.raw_review_repair_recommended
            ),
            raw_attribute_review_hard_rejected=(
                completion_metrics.raw_review_hard_rejected
            ),
            completion_raw_usable_attempts=(
                completion_metrics.raw_usable_attempts
            ),
            completion_raw_unusable_attempts=(
                completion_metrics.raw_unusable_attempts
            ),
            completion_selected_completed=(
                completion_metrics.selected_completed
            ),
            completion_fallback_to_raw=(
                completion_metrics.fallback_to_raw
            ),
            completion_backend_failures=(
                completion_metrics.backend_failures
            ),
            completion_postcheck_rejects=(
                completion_metrics.postcheck_rejects
            ),
            completion_sam_zero_mask_rejects=(
                completion_metrics.sam_zero_mask_rejects
            ),
            completion_sam_single_mask=completion_metrics.sam_single_mask,
            completion_sam_multi_mask=completion_metrics.sam_multi_mask,
            completion_sam_masks_returned_total=(
                completion_metrics.sam_masks_returned_total
            ),
            completion_identity_review_rejects=(
                completion_metrics.identity_review_rejects
            ),
            completion_final_review_rejects=(
                completion_metrics.final_review_rejects
            ),
            repaired_attribute_final_review_accepted=(
                completion_metrics.repaired_final_review_accepted
            ),
            repaired_attribute_final_review_rejected=(
                completion_metrics.repaired_final_review_rejected
            ),
            completion_review_calls=completion_metrics.review_calls,
            completion_model_call_time_seconds=(
                completion_metrics.model_call_time_seconds
            ),
            completion_attempts_by_type=dict(
                sorted(completion_metrics.attempts_by_type.items())
            ),
            completion_accepted_by_type=dict(
                sorted(completion_metrics.accepted_by_type.items())
            ),
            attribute_bbox_variants_materialized=(
                variant_metrics.bbox_variants_materialized
            ),
            attribute_bbox_reviews_attempted=variant_metrics.bbox_reviews_attempted,
            attribute_bbox_reviews_skipped_background_accepted=0,
            attribute_background_variants_attempted=0,
            attribute_background_variants_accepted=0,
            attribute_source_candidates_considered=sum(
                len(options) for options in candidate_options_by_id.values()
            ),
            attribute_second_candidate_attempts=second_candidate_attempts,
            attribute_completion_candidate1_accepted=(
                completion_candidate_accepted[0]
            ),
            attribute_completion_candidate2_accepted=(
                completion_candidate_accepted[1]
            ),
            attribute_bbox_fallback_attempts=(
                new_variant_metrics.bbox_reviews_attempted
            ),
            attribute_bbox_fallback_accepted=sum(
                record.status == "accepted" and record.final_selection == "bbox"
                for record in ordered_records
            ),
            failures=failures,
        ),
        gme_screen_mode=gme_screen_mode,
        completion_mode=completion_mode,
        failure_reason=review_failure,
    )


_RELATION_PHRASE: dict[AttributeType, str] = {
    "face": "with the facial appearance shown in {image}",
    "hair": "with the hairstyle shown in {image}",
    "headwear": "wearing the headwear shown in {image}",
    "glasses": "wearing the glasses shown in {image}",
    "upper_clothing": "wearing the clothing shown in {image}",
    "lower_clothing": "wearing the clothing shown in {image}",
    "dress_or_skirt": "wearing the dress or skirt shown in {image}",
    "shoes": "wearing the shoes shown in {image}",
    "bag": "carrying or wearing the bag shown in {image}",
    "accessory": "with the accessory shown in {image}",
}


def _render_enriched_instruction(
    clip: ClipRecord,
    accepted_attributes: list[SubjectAttributeRecord],
    *,
    source_run_root: Path,
) -> tuple[str, list[EnrichedReference]]:
    if (
        clip.annotation is None
        or clip.pairing is None
        or clip.instruction is None
        or clip.instruction.status != "ready"
    ):
        raise ValueError("enriched instruction requires ready Visual state")
    annotations = {entity.entity_id: entity for entity in clip.annotation.entities}
    references = {
        reference.entity_id: reference
        for reference in clip.references.entities
        if reference.status == "ready"
    }
    attributes_by_owner: dict[str, list[SubjectAttributeRecord]] = {}
    for record in accepted_attributes:
        attributes_by_owner.setdefault(record.owner_entity_id, []).append(record)

    enriched_references: list[EnrichedReference] = []
    old_to_new: dict[int, int] = {}
    attribute_image_indexes: dict[str, int] = {}
    old_index = 0
    for entity_id in clip.pairing.retained_entity_ids:
        old_index += 1
        annotation = annotations[entity_id]
        reference = references[entity_id]
        new_index = len(enriched_references) + 1
        old_to_new[old_index] = new_index
        enriched_references.append(
            EnrichedReference(
                image_id=f"image_{new_index}",
                image_index=new_index,
                kind=annotation.reference_type,
                origin="visual_run",
                entity_id=entity_id,
                image_path=_normalize_visual_run_image_path(
                    source_run_root,
                    clip_uid=reference.source_clip_uid or clip.clip_uid,
                    image_path=reference.image_path or "",
                ),
                source_frame_index=reference.source_frame_index,
            )
        )
        for attribute in attributes_by_owner.get(entity_id, []):
            assert attribute.image_path is not None
            new_index = len(enriched_references) + 1
            attribute_image_indexes[attribute.attribute_id] = new_index
            enriched_references.append(
                EnrichedReference(
                    image_id=f"image_{new_index}",
                    image_index=new_index,
                    kind="attribute",
                    origin="attribute_enrichment",
                    attribute_id=attribute.attribute_id,
                    owner_entity_id=entity_id,
                    attribute_type=attribute.attribute_type,
                    image_path=attribute.image_path,
                    source_frame_index=attribute.source_frame_index,
                )
            )
    if clip.pairing.background_token is not None:
        old_index += 1
        background = clip.references.background
        if background is None or background.output_image_path is None:
            raise ValueError("ready Visual background is missing its image path")
        new_index = len(enriched_references) + 1
        old_to_new[old_index] = new_index
        enriched_references.append(
            EnrichedReference(
                image_id=f"image_{new_index}",
                image_index=new_index,
                kind="background",
                origin="visual_run",
                image_path=_normalize_visual_run_image_path(
                    source_run_root,
                    clip_uid=clip.clip_uid,
                    image_path=background.output_image_path,
                ),
                source_frame_index=background.source_frame_index,
            )
        )

    def remap(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old not in old_to_new:
            raise ValueError("original instruction contains an unknown image placeholder")
        return f"{{{{image_{old_to_new[old]}}}}}"

    body = _IMAGE_PLACEHOLDER.sub(remap, clip.instruction.instruction_body_template)
    for entity_id in clip.pairing.retained_entity_ids:
        owner_attributes = attributes_by_owner.get(entity_id, [])
        if not owner_attributes:
            continue
        old_entity_index = clip.pairing.retained_entity_ids.index(entity_id) + 1
        owner_marker = f"{{{{image_{old_to_new[old_entity_index]}}}}}"
        relations = [
            _RELATION_PHRASE[attribute.attribute_type].format(
                image=f"{{{{image_{attribute_image_indexes[attribute.attribute_id]}}}}}"
            )
            for attribute in owner_attributes
        ]
        body = body.replace(
            owner_marker,
            owner_marker + ", " + " and ".join(relations),
            1,
        )
    enriched_instruction = _IMAGE_PLACEHOLDER.sub(
        lambda match: f"<Image {match.group(1)}>",
        body.strip(),
    )
    return enriched_instruction, enriched_references


def _build_enriched_sample(
    *,
    storage: RunStorage,
    clip: ClipRecord,
    records: list[SubjectAttributeRecord],
) -> EnrichedSample:
    if clip.annotation is None or clip.instruction is None:
        raise ValueError("enriched sample requires annotation and instruction")
    accepted = [record for record in records if record.status == "accepted"]
    enriched_instruction, references = _render_enriched_instruction(
        clip,
        accepted,
        source_run_root=storage.root,
    )
    return EnrichedSample(
        sample_id=clip.clip_uid,
        clip_uid=clip.clip_uid,
        source_run_root=str(storage.root),
        original_visual={
            "clip_record_path": (
                storage.clip_path(clip.clip_uid).relative_to(storage.root).as_posix()
            ),
            "target_video": clip.source.video_path,
            "t2v_caption": (
                render_annotation_plain_text(
                    clip.annotation.instruction_template,
                    clip.annotation.entities,
                    clip.annotation.background,
                )
                if clip.annotation.instruction_template
                else clip.annotation.t2v_caption
            ),
            "source": {
                "parent_video_id": clip.source.parent_video_id,
                "clip_suffix": clip.source.clip_suffix,
            },
        },
        original_instruction=clip.instruction.r2v_instruction,
        enriched_instruction=enriched_instruction,
        references=references,
        accepted_attributes=accepted,
    )


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_root(
    run_root: Path,
    export_root: Path,
    output_root: Path,
    *,
    allow_run_local_sidecar: bool = False,
) -> None:
    if not isinstance(allow_run_local_sidecar, bool):
        raise TypeError("allow_run_local_sidecar must be a boolean")
    writable_root = v3_config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    resolved_run = run_root.resolve(strict=False)
    resolved_export = export_root.resolve(strict=False)
    resolved_output = output_root.resolve(strict=False)
    if writable_root not in resolved_output.parents:
        raise ValueError(
            "attribute output_root must be inside /mnt/workspace/litengjie/data"
        )
    if allow_run_local_sidecar:
        if resolved_output != resolved_run / "subject_attributes":
            raise ValueError(
                "run-local attribute output_root must be exactly "
                "<source_run_root>/subject_attributes"
            )
    elif (
        resolved_output == resolved_run
        or resolved_output in resolved_run.parents
        or resolved_run in resolved_output.parents
    ):
        raise ValueError("attribute output_root must be separate from source run_root")
    if (
        resolved_output == resolved_export
        or resolved_output in resolved_export.parents
        or resolved_export in resolved_output.parents
    ):
        raise ValueError(
            "attribute output_root must not overlap the Visual export_root"
        )


def _normalize_visual_run_image_path(
    source_run_root: Path,
    *,
    clip_uid: str,
    image_path: str,
) -> str:
    path = Path(image_path).expanduser()
    resolved_root = source_run_root.expanduser().resolve(strict=False)
    if path.is_absolute():
        candidate = path
    elif path.parts[:1] == ("frames",):
        candidate = resolved_root / "clips" / clip_uid / path
    else:
        candidate = resolved_root / path
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Visual reference image must remain inside source run_root") from exc
    if not relative.parts:
        raise ValueError("Visual reference image path must identify an artifact")
    return relative.as_posix()


def _validate_source_run_identity(config: V3Config, run: RunRecord) -> None:
    mismatches: list[str] = []
    if run.config_hash != config.fingerprint():
        mismatches.append("config_hash")
    if run.model_identifiers != config.model_identifiers():
        mismatches.append("model_identifiers")
    expected_manifest = str(config.dataset_json.expanduser().resolve(strict=False))
    if run.source_manifest_path != expected_manifest:
        mismatches.append("source_manifest_path")
    if mismatches:
        raise ValueError(
            "supplied config does not match the source Visual run identity: "
            + ", ".join(mismatches)
        )


def _clear_enrichment_artifacts(output_root: Path) -> None:
    for directory, pattern in (
        (output_root / "references", "*.png"),
        (output_root / "owners", "*.json"),
        (output_root / "samples", "*.json"),
    ):
        if directory.is_dir():
            for path in directory.rglob(pattern):
                path.unlink()
    shutil.rmtree(output_root / "completion_candidates", ignore_errors=True)


def _clear_clip_enrichment_artifacts(output_root: Path, clip_uid: str) -> None:
    owner_directory = output_root / "owners" / clip_uid
    if owner_directory.is_dir():
        for path in owner_directory.glob("*.json"):
            path.unlink()
    reference_directory = output_root / "references" / clip_uid
    if reference_directory.is_dir():
        for path in reference_directory.glob("*.png"):
            path.unlink()
    _clip_sample_path(output_root, sample_id=clip_uid).unlink(missing_ok=True)
    shutil.rmtree(
        output_root / "completion_candidates" / clip_uid,
        ignore_errors=True,
    )


def process_subject_attribute_clip(
    config: V3Config,
    *,
    storage: RunStorage,
    output_root: Path,
    clip: ClipRecord,
    discovery_client: SubjectAttributeDiscoveryClient,
    review_client: SubjectAttributeReviewClient,
    segmentation_backend: AttributeFrameSegmentationBackend,
    gme_screener: AttributeGmeScreener | None = None,
    completion_backend: AttributeCompletionBackend | None = None,
    completion_judge: AttributeCompletionJudge | None = None,
    max_owners: int | None = None,
    overwrite: bool = False,
) -> ClipEnrichmentResult:
    completion_config = getattr(
        getattr(config, "subject_attributes", None),
        "completion",
        SubjectAttributeCompletionConfig(),
    )
    if max_owners is not None and (
        isinstance(max_owners, bool)
        or not isinstance(max_owners, int)
        or max_owners < 1
    ):
        raise ValueError("max_owners must be a positive integer")
    if clip.clip_uid != storage.read_clip(clip.clip_uid).clip_uid:
        raise ValueError("subject attribute clip does not belong to source storage")
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_clip_enrichment_artifacts(output_root, clip.clip_uid)

    totals = EnrichmentTotals()
    if (
        not clip.export.accepted
        or clip.annotation is None
        or clip.annotation.status != "ready"
        or clip.pairing is None
        or clip.pairing.status != "ready"
    ):
        return ClipEnrichmentResult(
            clip_uid=clip.clip_uid,
            totals=totals,
            owner_limit_reached=False,
            enriched_sample=None,
        )
    try:
        frames = storage.read_frames(clip.clip_uid)
        masks = storage.read_masks(clip.clip_uid)
    except (OSError, ValueError):
        return ClipEnrichmentResult(
            clip_uid=clip.clip_uid,
            totals=totals,
            owner_limit_reached=False,
            enriched_sample=None,
        )

    sample_attribute_start = 1
    clip_records: list[SubjectAttributeRecord] = []
    processed_owner = False
    owner_limit_reached = False
    for owner in clip.annotation.entities:
        if owner.reference_type != "subject":
            continue
        try:
            candidates = build_entity_reference_candidates(
                config,
                storage,
                clip_uid=clip.clip_uid,
                entity=owner,
                frames=frames,
                masks=masks,
            )[:MAX_ATTRIBUTES_PER_OWNER]
        except (OSError, ValueError):
            candidates = []
        eligibility = evaluate_owner_eligibility(
            clip,
            entity_id=owner.entity_id,
            has_usable_candidate_evidence=bool(candidates),
        )
        if not eligibility.eligible:
            continue
        if max_owners is not None and totals.eligible_human_owners >= max_owners:
            owner_limit_reached = True
            break
        artifact_path = _owner_artifact_path(
            output_root,
            sample_id=clip.clip_uid,
            owner_entity_id=owner.entity_id,
        )
        cached = None
        if not overwrite:
            cached = _load_cached_owner_artifact(
                artifact_path,
                output_root=output_root,
                sample_id=clip.clip_uid,
                owner_entity_id=owner.entity_id,
                attribute_id_start=sample_attribute_start,
                expected_gme_screen_mode=(
                    getattr(
                        getattr(config, "subject_attribute_gme", None),
                        "screen_mode",
                        None,
                    )
                    if gme_screener is not None
                    else None
                ),
                expected_completion_mode=(
                    "boogu_completion_v1"
                    if completion_config.enabled
                    else None
                ),
            )
        if cached is not None:
            artifact = cached
            totals.skipped_existing_owners += 1
        else:
            try:
                artifact = _process_owner(
                    config=config,
                    storage=storage,
                    output_root=output_root,
                    clip=clip,
                    owner=owner,
                    owner_candidates=candidates,
                    masks=masks,
                    attribute_id_start=sample_attribute_start,
                    discovery_client=discovery_client,
                    review_client=review_client,
                    segmentation_backend=segmentation_backend,
                    gme_screener=gme_screener,
                    completion_backend=completion_backend,
                    completion_judge=completion_judge,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one owner
                totals.failures += 1
                totals.failure_reasons[
                    f"owner_processing_failed:{type(exc).__name__}"
                ] += 1
                continue
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(artifact_path, artifact.model_dump(mode="json"))
        if artifact.failure_reason:
            totals.failure_reasons[artifact.failure_reason.split(":", 1)[0]] += 1
        totals.add_owner_metrics(artifact.metrics)
        if artifact.owner_is_human is True:
            totals.eligible_human_owners += 1
            processed_owner = True
        elif artifact.owner_is_human is False:
            totals.screened_nonhuman_subjects += 1
        clip_records.extend(artifact.records)
        sample_attribute_start += len(artifact.records)

    enriched_sample = None
    if processed_owner:
        enriched_sample = _build_enriched_sample(
            storage=storage,
            clip=clip,
            records=clip_records,
        )
        sample_path = _clip_sample_path(output_root, sample_id=clip.clip_uid)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(sample_path, enriched_sample.model_dump(mode="json"))
    return ClipEnrichmentResult(
        clip_uid=clip.clip_uid,
        totals=totals,
        owner_limit_reached=owner_limit_reached,
        enriched_sample=enriched_sample,
    )


def _load_durable_owner_artifact(
    path: Path,
    *,
    output_root: Path,
    sample_id: str,
    owner_entity_id: str,
) -> OwnerEnrichmentArtifact | None:
    if not path.is_file():
        return None
    try:
        artifact = OwnerEnrichmentArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return _load_cached_owner_artifact(
        path,
        output_root=output_root,
        sample_id=sample_id,
        owner_entity_id=owner_entity_id,
        attribute_id_start=artifact.attribute_id_start,
        expected_gme_screen_mode=artifact.gme_screen_mode,
        expected_completion_mode=artifact.completion_mode,
    )


def reconcile_subject_attribute_outputs(
    *,
    storage: RunStorage,
    output_root: Path,
    owner_limit: int | None,
    invocation_wall_time_seconds: float,
    skipped_existing_owners: int = 0,
    gpu_peak_memory_bytes_before: int | None = None,
    gpu_peak_memory_bytes_after: int | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    totals = EnrichmentTotals(skipped_existing_owners=skipped_existing_owners)
    all_records: list[SubjectAttributeRecord] = []
    sample_records: list[EnrichedSample] = []
    for clip in storage.iter_clips():
        owner_limit_reached = False
        if clip.annotation is not None:
            for owner in clip.annotation.entities:
                if owner.reference_type != "subject":
                    continue
                if (
                    owner_limit is not None
                    and totals.eligible_human_owners >= owner_limit
                ):
                    owner_limit_reached = True
                    break
                artifact = _load_durable_owner_artifact(
                    _owner_artifact_path(
                        output_root,
                        sample_id=clip.clip_uid,
                        owner_entity_id=owner.entity_id,
                    ),
                    output_root=output_root,
                    sample_id=clip.clip_uid,
                    owner_entity_id=owner.entity_id,
                )
                if artifact is None:
                    continue
                totals.add_owner_metrics(artifact.metrics)
                if artifact.owner_is_human is True:
                    totals.eligible_human_owners += 1
                elif artifact.owner_is_human is False:
                    totals.screened_nonhuman_subjects += 1
                if artifact.failure_reason:
                    totals.failure_reasons[
                        artifact.failure_reason.split(":", 1)[0]
                    ] += 1
                all_records.extend(artifact.records)
                if (
                    owner_limit is not None
                    and totals.eligible_human_owners >= owner_limit
                ):
                    owner_limit_reached = True
                    break
        sample_path = _clip_sample_path(output_root, sample_id=clip.clip_uid)
        if sample_path.is_file():
            try:
                sample = EnrichedSample.model_validate_json(
                    sample_path.read_text(encoding="utf-8")
                )
                if sample.sample_id == clip.clip_uid:
                    sample_records.append(sample)
            except (OSError, ValueError):
                pass
        if owner_limit_reached:
            break

    run = storage.read_run()
    owner_count = totals.eligible_human_owners
    qwen_calls = (
        totals.discovery_calls
        + totals.review_calls
        + totals.completion_review_calls
    )
    single_subject_example = next(
        (
            sample.sample_id
            for sample in sample_records
            if sample.accepted_attributes
            and sum(reference.kind == "subject" for reference in sample.references) == 1
        ),
        None,
    )
    multi_subject_example = next(
        (
            sample.sample_id
            for sample in sample_records
            if sample.accepted_attributes
            and sum(reference.kind == "subject" for reference in sample.references) >= 2
        ),
        None,
    )
    summary: dict[str, object] = {
        "schema_version": ATTRIBUTE_ENRICHMENT_SCHEMA_VERSION,
        "source_run_root": str(storage.root),
        "source_run_git_commit": run.git_commit,
        "output_root": str(output_root),
        "owner_limit": owner_limit,
        "eligible_human_owner_count": owner_count,
        "screened_nonhuman_subject_count": totals.screened_nonhuman_subjects,
        "skipped_existing_owner_count": totals.skipped_existing_owners,
        "qwen_discovery_calls": totals.discovery_calls,
        "qwen_recognizability_calls": totals.review_calls,
        "qwen_calls_total": qwen_calls,
        "attributes_discovered_by_type": dict(
            sorted(totals.discovered_by_type.items())
        ),
        "sam3_attempts": totals.sam3_attempts,
        "gme_calls": totals.gme_calls,
        "gme_candidates_screened": totals.gme_candidates_screened,
        "gme_candidates_passed": totals.gme_candidates_passed,
        "gme_candidates_rejected": totals.gme_candidates_rejected,
        "gme_retried_next_frame": totals.gme_retried_next_frame,
        "gme_failures": totals.gme_failures,
        "completion_attempts": totals.completion_attempts,
        "completion_accepted": totals.completion_accepted,
        "completion_rejected": totals.completion_rejected,
        "completion_failures": totals.completion_failures,
        "completion_qwen_review_rejects": (
            totals.completion_qwen_review_rejects
        ),
        "raw_attribute_review_accepted": totals.raw_attribute_review_accepted,
        "raw_attribute_review_repair_recommended": (
            totals.raw_attribute_review_repair_recommended
        ),
        "raw_attribute_review_hard_rejected": (
            totals.raw_attribute_review_hard_rejected
        ),
        "completion_raw_usable_attempts": totals.completion_raw_usable_attempts,
        "completion_raw_unusable_attempts": (
            totals.completion_raw_unusable_attempts
        ),
        "completion_selected_completed": totals.completion_selected_completed,
        "completion_fallback_to_raw": totals.completion_fallback_to_raw,
        "completion_backend_failures": totals.completion_backend_failures,
        "completion_postcheck_rejects": totals.completion_postcheck_rejects,
        "completion_sam_zero_mask_rejects": (
            totals.completion_sam_zero_mask_rejects
        ),
        "completion_sam_single_mask": totals.completion_sam_single_mask,
        "completion_sam_multi_mask": totals.completion_sam_multi_mask,
        "completion_sam_masks_returned_total": (
            totals.completion_sam_masks_returned_total
        ),
        "completion_identity_review_rejects": (
            totals.completion_identity_review_rejects
        ),
        "completion_final_review_rejects": (
            totals.completion_final_review_rejects
        ),
        "repaired_attribute_final_review_accepted": (
            totals.repaired_attribute_final_review_accepted
        ),
        "repaired_attribute_final_review_rejected": (
            totals.repaired_attribute_final_review_rejected
        ),
        "completion_attempts_by_type": dict(
            sorted(totals.completion_attempts_by_type.items())
        ),
        "completion_accepted_by_type": dict(
            sorted(totals.completion_accepted_by_type.items())
        ),
        "attribute_bbox_variants_materialized": (
            totals.attribute_bbox_variants_materialized
        ),
        "attribute_bbox_reviews_attempted": totals.attribute_bbox_reviews_attempted,
        "attribute_bbox_reviews_skipped_background_accepted": (
            totals.attribute_bbox_reviews_skipped_background_accepted
        ),
        "attribute_background_variants_attempted": (
            totals.attribute_background_variants_attempted
        ),
        "attribute_background_variants_accepted": (
            totals.attribute_background_variants_accepted
        ),
        "attribute_source_candidates_considered": (
            totals.attribute_source_candidates_considered
        ),
        "attribute_second_candidate_attempts": (
            totals.attribute_second_candidate_attempts
        ),
        "attribute_completion_candidate1_accepted": (
            totals.attribute_completion_candidate1_accepted
        ),
        "attribute_completion_candidate2_accepted": (
            totals.attribute_completion_candidate2_accepted
        ),
        "attribute_bbox_fallback_attempts": (
            totals.attribute_bbox_fallback_attempts
        ),
        "attribute_bbox_fallback_accepted": (
            totals.attribute_bbox_fallback_accepted
        ),
        "deterministic_ownership_rejects": totals.deterministic_ownership_rejects,
        "recognizability_rejects": totals.recognizability_rejects,
        "accepted_attribute_references": totals.accepted_attributes,
        "accepted_attributes_per_human_owner": (
            totals.accepted_attributes / owner_count if owner_count else 0.0
        ),
        "average_extra_qwen_calls_per_human_owner": (
            qwen_calls / owner_count if owner_count else 0.0
        ),
        "average_sam3_attempts_per_human_owner": (
            totals.sam3_attempts / owner_count if owner_count else 0.0
        ),
        "same_frame_accepted": totals.same_frame_accepted,
        "different_frame_accepted": totals.different_frame_accepted,
        "invocation_wall_time_seconds": invocation_wall_time_seconds,
        "qwen_model_call_time_seconds": totals.qwen_model_call_time_seconds,
        "sam3_model_call_time_seconds": totals.sam3_model_call_time_seconds,
        "gme_model_call_time_seconds": totals.gme_model_call_time_seconds,
        "completion_model_call_time_seconds": (
            totals.completion_model_call_time_seconds
        ),
        "model_call_time_seconds": (
            totals.qwen_model_call_time_seconds
            + totals.sam3_model_call_time_seconds
            + totals.gme_model_call_time_seconds
            + totals.completion_model_call_time_seconds
        ),
        "gpu_peak_memory_bytes_before": gpu_peak_memory_bytes_before,
        "gpu_peak_memory_bytes_after": gpu_peak_memory_bytes_after,
        "failures": totals.failures,
        "failure_reasons": dict(sorted(totals.failure_reasons.items())),
        "enriched_sample_count": len(sample_records),
        "attribute_record_count": len(all_records),
        "example_enriched_samples": {
            "single_subject": single_subject_example,
            "multi_subject": multi_subject_example,
        },
    }
    _write_jsonl_atomic(
        output_root / "attributes.jsonl",
        [record.model_dump(mode="json") for record in all_records],
    )
    _write_jsonl_atomic(
        output_root / "enriched_samples.jsonl",
        [sample.model_dump(mode="json") for sample in sample_records],
    )
    write_json_atomic(output_root / "summary.json", summary)
    return summary


def _gpu_peak_bytes() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None


def run_subject_attribute_enrichment(
    config: V3Config,
    *,
    run_root: Path,
    output_root: Path,
    max_owners: int | None = 50,
    overwrite: bool = False,
    discovery_client: SubjectAttributeDiscoveryClient | None = None,
    review_client: SubjectAttributeReviewClient | None = None,
    segmentation_backend: AttributeFrameSegmentationBackend | None = None,
    gme_screener: AttributeGmeScreener | None = None,
    completion_backend: AttributeCompletionBackend | None = None,
    completion_judge: AttributeCompletionJudge | None = None,
    allow_run_local_sidecar: bool = False,
) -> dict[str, object]:
    if max_owners is not None and (
        isinstance(max_owners, bool)
        or not isinstance(max_owners, int)
        or max_owners < 1
    ):
        raise ValueError("max_owners must be a positive integer")
    requested_run_root = Path(run_root).expanduser().resolve(strict=False)
    if config.resolved_run_root != requested_run_root:
        raise ValueError("supplied config run_root does not match source run_root")
    effective_config = config
    effective_config.validate()
    storage = RunStorage(effective_config)
    run = storage.read_run()
    _validate_source_run_identity(effective_config, run)
    output_root = Path(output_root).expanduser().resolve(strict=False)
    _validate_output_root(
        storage.root,
        effective_config.resolved_export_root,
        output_root,
        allow_run_local_sidecar=allow_run_local_sidecar,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _clear_enrichment_artifacts(output_root)

    service = effective_config.qwen.candidate_judge
    owned_qwen: QwenSubjectAttributeClient | None = None
    if discovery_client is None or review_client is None:
        if service is None:
            raise ValueError(
                "qwen.candidate_judge is required for subject attribute enrichment"
            )
        owned_qwen = QwenSubjectAttributeClient(service)
        discovery_client = discovery_client or owned_qwen
        review_client = review_client or owned_qwen
    owned_segmenter: Sam3AttributeFrameSegmenter | None = None
    if segmentation_backend is None:
        if effective_config.sam3.model_path is None:
            raise ValueError("sam3.model_path is required for attribute enrichment")
        owned_segmenter = Sam3AttributeFrameSegmenter(effective_config.sam3)
        segmentation_backend = owned_segmenter
    assert discovery_client is not None
    assert review_client is not None
    assert segmentation_backend is not None
    owned_completion_judge: QwenSubjectAttributeCompletionJudge | None = None
    if effective_config.subject_attributes.completion.enabled:
        if completion_backend is None:
            raise ValueError(
                "standalone enabled attribute completion requires an injected "
                "persistent Boogu backend"
            )
        if completion_judge is None:
            service = effective_config.qwen.candidate_judge
            assert service is not None
            owned_completion_judge = QwenSubjectAttributeCompletionJudge(
                service,
                completion_component="qwen_attribute_completion_review",
            )
            completion_judge = owned_completion_judge

    started = time.perf_counter()
    gpu_before = _gpu_peak_bytes()
    processed_owners = 0
    skipped_existing_owners = 0
    try:
        for clip in storage.iter_clips():
            if max_owners is not None and processed_owners >= max_owners:
                break
            result = process_subject_attribute_clip(
                effective_config,
                storage=storage,
                output_root=output_root,
                clip=clip,
                discovery_client=discovery_client,
                review_client=review_client,
                segmentation_backend=segmentation_backend,
                gme_screener=gme_screener,
                completion_backend=completion_backend,
                completion_judge=completion_judge,
                max_owners=(
                    max_owners - processed_owners
                    if max_owners is not None
                    else None
                ),
                overwrite=False,
            )
            processed_owners += result.totals.eligible_human_owners
            skipped_existing_owners += result.totals.skipped_existing_owners
    finally:
        if owned_completion_judge is not None:
            owned_completion_judge.close()
        if owned_segmenter is not None:
            owned_segmenter.close()
        if owned_qwen is not None:
            owned_qwen.close()

    return reconcile_subject_attribute_outputs(
        storage=storage,
        output_root=output_root,
        owner_limit=max_owners,
        invocation_wall_time_seconds=time.perf_counter() - started,
        skipped_existing_owners=skipped_existing_owners,
        gpu_peak_memory_bytes_before=gpu_before,
        gpu_peak_memory_bytes_after=_gpu_peak_bytes(),
    )


__all__ = [
    "AttributeCompletionBackend",
    "AttributeFrameSegmentationBackend",
    "AttributeGmeScreener",
    "AttributeProbeClient",
    "ClipEnrichmentResult",
    "DiscoveredSubjectAttribute",
    "EnrichedSample",
    "GmeAttributeScreenAttempt",
    "OwnerEligibility",
    "OwnershipGeometry",
    "PendingAttributeCandidate",
    "PersistentWorkerAttributeFrameSegmenter",
    "QwenSubjectAttributeClient",
    "Sam3AttributeFrameSegmenter",
    "SubjectAttributeDiscovery",
    "SubjectAttributeRecord",
    "SubjectAttributeReview",
    "SubjectAttributeReviewBatch",
    "attribute_completion_prompt",
    "evaluate_owner_eligibility",
    "evaluate_ownership_geometry",
    "prefer_attribute_candidate_frames",
    "process_subject_attribute_clip",
    "reconcile_subject_attribute_outputs",
    "run_subject_attribute_enrichment",
]
