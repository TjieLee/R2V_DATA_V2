from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

CLIP_SCHEMA_VERSION = "r2v.v3.clip.2"
FRAMES_SCHEMA_VERSION = "r2v.v3.frames.1"
MASK_SCHEMA_VERSION = "r2v.v3.masks.1"
RUN_SCHEMA_VERSION = "r2v.v3.run.1"
DATASET_SCHEMA_VERSION = "r2v.v3.dataset.1"
SAMPLE_SCHEMA_VERSION = "r2v.v3.sample.1"
MAX_ANNOTATION_ENTITIES = 8

_REF_TOKEN = re.compile(r"<ref_(?:subject|object|group|bg)_\d+>")
_ANY_REF_TOKEN = re.compile(r"<ref_[^>]+>")
_IMAGE_ID = re.compile(r"image_([1-9]\d*)")
_CANDIDATE_ID = re.compile(r"candidate_[1-9]\d*")
_IMAGE_PLACEHOLDER = re.compile(r"\{\{(image_[1-9]\d*)\}\}")
_ANY_TEMPLATE_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")
_DIRECT_CHINESE_IMAGE_LABEL = re.compile(r"\u56fe\s*\d+")
_DIRECT_ENGLISH_IMAGE_LABEL = re.compile(
    r"\bImage\s+[1-9]\d*\b",
    flags=re.IGNORECASE,
)
_CHINESE_IMAGE_INDEX = re.compile(r"\u56fe\s*(\d+)")
_ANGLE_ENGLISH_IMAGE_INDEX = re.compile(
    r"<Image\s+([1-9]\d*)>",
    flags=re.IGNORECASE,
)
_PLAIN_ENGLISH_IMAGE_INDEX = re.compile(
    r"(?<!<)\bImage\s+([1-9]\d*)\b(?!>)",
    flags=re.IGNORECASE,
)
_ANNOTATION_ENTITY_MARKER_WITH_SPACE = re.compile(r" \{\{entity_[1-9]\d*\}\}")
_ANNOTATION_BACKGROUND_MARKER_WITH_SPACE = re.compile(r" \{\{background\}\}")
_ANNOTATION_ENTITY_PLACEHOLDER = re.compile(r"\{\{entity_[1-9]\d*\}\}")
_ANNOTATION_BACKGROUND_PLACEHOLDER = re.compile(r"\{\{background\}\}")


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipSource(SchemaModel):
    video_path: str
    parent_video_id: str
    clip_suffix: str
    source_index: int = Field(ge=0)
    caption_raw: str
    metadata: dict[str, object]


ReferenceType = Literal["subject", "object", "group"]


class RawAnnotationEntity(SchemaModel):
    reference_type: str
    phrase: str
    grounding_prompt: str


class RawBackgroundAnnotation(SchemaModel):
    phrase: str
    grounding_prompt: str


class RawAnnotationPayload(SchemaModel):
    entities: list[RawAnnotationEntity]
    background: Optional[RawBackgroundAnnotation]
    instruction_template: str


class AnnotationEntity(SchemaModel):
    entity_id: str
    reference_type: ReferenceType
    phrase: str
    grounding_prompt: str

    @model_validator(mode="after")
    def validate_text(self) -> AnnotationEntity:
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("annotation entity text must not be empty")
        if _ANY_REF_TOKEN.search(self.phrase) or _ANY_REF_TOKEN.search(
            self.grounding_prompt
        ):
            raise ValueError("annotation entity text must not contain reference tokens")
        return self


class BackgroundAnnotation(SchemaModel):
    phrase: str
    grounding_prompt: str

    @model_validator(mode="after")
    def validate_text(self) -> BackgroundAnnotation:
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("background annotation text must not be empty")
        if _ANY_REF_TOKEN.search(self.phrase) or _ANY_REF_TOKEN.search(
            self.grounding_prompt
        ):
            raise ValueError(
                "background annotation text must not contain reference tokens"
            )
        return self


class AnnotationPayload(SchemaModel):
    instruction_template: str
    entities: list[AnnotationEntity] = Field(default_factory=list)
    background: Optional[BackgroundAnnotation] = None


class AnnotationState(SchemaModel):
    status: Literal["ready", "failed"]
    instruction_template: str = ""
    t2v_caption: str = ""
    entities: list[AnnotationEntity] = Field(default_factory=list)
    background: Optional[BackgroundAnnotation] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> AnnotationState:
        if _ANY_REF_TOKEN.search(self.t2v_caption):
            raise ValueError("t2v_caption must not contain reference tokens")
        if _ANY_REF_TOKEN.search(self.instruction_template):
            raise ValueError("instruction_template must not contain reference tokens")
        if self.status == "ready":
            has_template = bool(self.instruction_template.strip())
            has_legacy_caption = bool(self.t2v_caption.strip())
            if has_template == has_legacy_caption:
                raise ValueError(
                    "ready annotation requires exactly one of instruction_template "
                    "or legacy t2v_caption"
                )
        if self.status == "failed" and not self.reason:
            raise ValueError("failed annotation requires a reason")
        if self.status == "failed" and (
            self.instruction_template
            or self.t2v_caption
            or self.entities
            or self.background is not None
        ):
            raise ValueError("failed annotation must not publish semantic content")
        entity_ids = [entity.entity_id for entity in self.entities]
        expected_entity_ids = [f"e{index}" for index in range(1, len(entity_ids) + 1)]
        if entity_ids != expected_entity_ids:
            raise ValueError(
                "annotation entity_id values must be contiguous and ordered"
            )
        if len(entity_ids) > MAX_ANNOTATION_ENTITIES:
            raise ValueError(
                f"annotation supports at most {MAX_ANNOTATION_ENTITIES} entities"
            )
        return self


def plain_instruction_text(instruction_template: str) -> str:
    """Remove valid internal annotation markers without rewriting other text."""
    value = _ANNOTATION_ENTITY_MARKER_WITH_SPACE.sub("", instruction_template)
    value = _ANNOTATION_BACKGROUND_MARKER_WITH_SPACE.sub("", value)
    return value.strip()


def render_annotation_plain_text(
    instruction_template: str,
    entities: list[AnnotationEntity],
    background: Optional[BackgroundAnnotation],
) -> str:
    """Substitute V3 mention placeholders with their annotation phrases."""
    value = instruction_template
    for index, entity in enumerate(entities, start=1):
        value = value.replace(f"{{{{entity_{index}}}}}", entity.phrase)
    if background is not None:
        value = value.replace("{{background}}", background.phrase)
    if _ANNOTATION_ENTITY_PLACEHOLDER.search(
        value
    ) or _ANNOTATION_BACKGROUND_PLACEHOLDER.search(value):
        raise ValueError("annotation contains an unbound internal placeholder")
    return value.strip()


class SampledFrame(SchemaModel):
    slot: int = Field(ge=0, lt=10)
    source_frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frame(self) -> SampledFrame:
        if not math.isfinite(self.timestamp_seconds):
            raise ValueError("frame timestamp_seconds must be finite")
        expected_path = f"frames/{self.slot:02d}.jpg"
        if self.image_path != expected_path:
            raise ValueError(f"frame image_path must match its slot: {expected_path}")
        return self


class SampledFramesArtifact(SchemaModel):
    schema_version: Literal["r2v.v3.frames.1"] = FRAMES_SCHEMA_VERSION
    clip_uid: str
    sampled_frame_count: Literal[10] = 10
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[SampledFrame]

    @model_validator(mode="after")
    def validate_frames(self) -> SampledFramesArtifact:
        if len(self.frames) != self.sampled_frame_count:
            raise ValueError("frames artifact must contain exactly 10 frames")
        slots = [frame.slot for frame in self.frames]
        if slots != list(range(self.sampled_frame_count)):
            raise ValueError("frame slots must be ordered from 0 through 9")
        source_indices = [frame.source_frame_index for frame in self.frames]
        if any(
            source_indices[index] >= source_indices[index + 1]
            for index in range(len(source_indices) - 1)
        ):
            raise ValueError(
                "source frame indices must be unique and strictly increasing"
            )
        timestamps = [frame.timestamp_seconds for frame in self.frames]
        if any(
            timestamps[index] >= timestamps[index + 1]
            for index in range(len(timestamps) - 1)
        ):
            raise ValueError("frame timestamps must be unique and strictly increasing")
        return self


class EntityVisibilitySummary(SchemaModel):
    status: Literal["ready", "not_found", "failed"]
    visible_frame_slots: list[int] = Field(default_factory=list)
    visible_frame_count: int = Field(ge=0, le=10)
    coverage_ratio: float = Field(ge=0, le=1)
    qualifies: bool
    per_frame_area_ratio: list[float]
    per_frame_confidence: list[Optional[float]]

    @model_validator(mode="after")
    def validate_visibility(self) -> EntityVisibilitySummary:
        if len(self.per_frame_area_ratio) != 10 or len(self.per_frame_confidence) != 10:
            raise ValueError(
                "entity visibility diagnostics must contain ten frame slots"
            )
        if self.visible_frame_slots != sorted(set(self.visible_frame_slots)) or any(
            not 0 <= slot < 10 for slot in self.visible_frame_slots
        ):
            raise ValueError(
                "visible frame slots must be unique, ordered, and in range"
            )
        if self.visible_frame_count != len(self.visible_frame_slots):
            raise ValueError("visible_frame_count must match visible_frame_slots")
        if not math.isclose(
            self.coverage_ratio,
            self.visible_frame_count / 10,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("coverage_ratio must match visible_frame_count")
        diagnostics = [
            *self.per_frame_area_ratio,
            *(value for value in self.per_frame_confidence if value is not None),
        ]
        if any(not math.isfinite(value) for value in diagnostics):
            raise ValueError("entity visibility diagnostics must be finite")
        visible_slots = set(self.visible_frame_slots)
        if any(self.per_frame_confidence[slot] is None for slot in visible_slots):
            raise ValueError("visible frame slots require confidence diagnostics")
        if any(self.per_frame_area_ratio[slot] <= 0 for slot in visible_slots):
            raise ValueError("visible frame slots require positive area diagnostics")
        hidden_slots = set(range(10)) - visible_slots
        if any(
            self.per_frame_confidence[slot] is not None
            or self.per_frame_area_ratio[slot] != 0
            for slot in hidden_slots
        ):
            raise ValueError("non-visible frame slots must have empty diagnostics")
        if self.status != "ready" and self.visible_frame_count:
            raise ValueError("non-ready tracked entities cannot have visible frames")
        return self


class CoverageState(SchemaModel):
    passed: bool
    qualifying_entity_ids: list[str] = Field(default_factory=list)
    required_visible_frames: int = Field(default=7, ge=1, le=10)
    entity_visibility_summary: dict[str, EntityVisibilitySummary] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_qualifying_entities(self) -> CoverageState:
        if self.passed != bool(self.qualifying_entity_ids):
            raise ValueError(
                "coverage passed must match whether qualifying_entity_ids is non-empty"
            )
        if len(self.qualifying_entity_ids) != len(set(self.qualifying_entity_ids)):
            raise ValueError("qualifying_entity_ids must be unique")
        if self.qualifying_entity_ids and not self.entity_visibility_summary:
            raise ValueError("qualifying entities require visible-frame summaries")
        if self.entity_visibility_summary:
            expected_qualifying: list[str] = []
            for entity_id, summary in self.entity_visibility_summary.items():
                expected = (
                    summary.status == "ready"
                    and summary.visible_frame_count >= self.required_visible_frames
                )
                if summary.qualifies != expected:
                    raise ValueError(
                        "entity qualifies must match required visible frames"
                    )
                if summary.qualifies:
                    expected_qualifying.append(entity_id)
            if self.qualifying_entity_ids != expected_qualifying:
                raise ValueError(
                    "qualifying entity IDs must match visibility summaries"
                )
        return self


ReferenceScope = Literal["full", "local", "reject"]
ReferenceImageQuality = Literal["high", "acceptable", "poor"]
ReferenceCompleteness = Literal[
    "complete",
    "repairable",
    "local_usable",
    "severely_incomplete",
    "fragmented",
]
ReferenceTruncationSeverity = Literal["none", "minor", "major"]
ReferenceViewpoint = Literal[
    "front",
    "three_quarter",
    "side",
    "rear",
    "not_applicable",
]
VisibleRegion = Literal[
    "whole",
    "head_shoulders",
    "upper_body",
    "lower_body",
    "front",
    "rear",
    "side",
    "central",
    "custom",
]


class RawEntityReferenceDecision(SchemaModel):
    selected_candidate_id: Optional[str]
    image_quality: ReferenceImageQuality
    completeness: ReferenceCompleteness
    reference_scope: ReferenceScope
    visible_region: VisibleRegion
    whole_entity_recognizable: StrictBool
    identity_features_visible: StrictBool
    viewpoint: ReferenceViewpoint
    independent_reference_value: StrictBool
    requires_substantial_invention: StrictBool
    primary_identity_region_visible: StrictBool
    major_structure_visible: StrictBool
    truncation_severity: ReferenceTruncationSeverity
    discrete_foreground_instance: StrictBool
    mask_matches_target: StrictBool
    completion_needed_for_reference_use: StrictBool
    detached_target_fragments_present: StrictBool
    scope_reason: str

    @model_validator(mode="after")
    def validate_decision(self) -> RawEntityReferenceDecision:
        if not self.scope_reason.strip():
            raise ValueError("entity reference scope_reason must not be empty")
        if (
            self.selected_candidate_id is not None
            and _CANDIDATE_ID.fullmatch(self.selected_candidate_id) is None
        ):
            raise ValueError("selected_candidate_id must use candidate_N")
        if self.reference_scope == "reject":
            if self.selected_candidate_id is not None:
                raise ValueError("reject decision must not select a candidate")
        elif self.selected_candidate_id is None:
            raise ValueError("full or local decision requires selected_candidate_id")
        if self.image_quality == "poor" and self.reference_scope != "reject":
            raise ValueError("poor image quality must reject the reference")
        if not self.independent_reference_value and self.reference_scope != "reject":
            raise ValueError("a reference without independent value must be rejected")
        if self.requires_substantial_invention and self.reference_scope != "reject":
            raise ValueError(
                "a reference requiring substantial invention must be rejected"
            )
        expected_scope: ReferenceScope
        if self.completeness == "complete":
            expected_scope = "full"
        elif self.completeness in {"repairable", "local_usable"}:
            expected_scope = "local"
        else:
            expected_scope = "reject"
        if self.reference_scope != expected_scope:
            raise ValueError(
                "reference_scope must match the completeness routing decision"
            )
        return self


class RawCrossPairDecision(SchemaModel):
    verdict: Literal["accept", "reject"]
    target_entity_visible: StrictBool
    same_physical_entity: StrictBool
    identity_features_match: StrictBool
    reference_is_usable: StrictBool
    reason: str

    @model_validator(mode="after")
    def validate_decision(self) -> RawCrossPairDecision:
        if not self.reason.strip():
            raise ValueError("cross-pair reason must not be empty")
        all_passed = all(
            (
                self.target_entity_visible,
                self.same_physical_entity,
                self.identity_features_match,
                self.reference_is_usable,
            )
        )
        expected = "accept" if all_passed else "reject"
        if self.verdict != expected:
            raise ValueError(
                "cross-pair verdict must accept if and only if all checks pass"
            )
        return self


class EntityReferenceState(SchemaModel):
    entity_id: str
    status: Literal["ready", "rejected"]
    reference_scope: ReferenceScope
    visible_region: VisibleRegion
    whole_entity_recognizable: bool
    identity_features_visible: bool
    scope_reason: str
    image_path: Optional[str] = None
    source_frame_index: Optional[int] = Field(default=None, ge=0)
    source_clip_uid: Optional[str] = None
    source_entity_id: Optional[str] = None
    image_quality: Optional[ReferenceImageQuality] = None
    completeness: Optional[ReferenceCompleteness] = None
    viewpoint: Optional[ReferenceViewpoint] = None
    independent_reference_value: Optional[bool] = None
    requires_substantial_invention: Optional[bool] = None
    primary_identity_region_visible: Optional[bool] = None
    major_structure_visible: Optional[bool] = None
    truncation_severity: Optional[ReferenceTruncationSeverity] = None
    discrete_foreground_instance: Optional[bool] = None
    mask_matches_target: Optional[bool] = None
    completion_needed_for_reference_use: Optional[bool] = None
    detached_target_fragments_present: Optional[bool] = None
    synthetic: bool = False
    generation_metadata_path: Optional[str] = None
    generation_source_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    generation_output_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_reference_state(self) -> EntityReferenceState:
        if not self.scope_reason.strip():
            raise ValueError("entity reference scope_reason must not be empty")
        if not isinstance(self.synthetic, bool):
            raise TypeError("entity reference synthetic must be a boolean")
        if (self.source_clip_uid is None) != (self.source_entity_id is None):
            raise ValueError(
                "entity reference source_clip_uid and source_entity_id "
                "must be set together"
            )
        if (self.image_quality is None) != (self.completeness is None):
            raise ValueError(
                "entity reference quality and completeness must be set together"
            )
        if self.status == "ready":
            if self.independent_reference_value is False:
                raise ValueError(
                    "ready entity reference requires independent reference value"
                )
            if self.requires_substantial_invention is True:
                raise ValueError(
                    "ready entity reference cannot require substantial invention"
                )
            if self.primary_identity_region_visible is False:
                raise ValueError(
                    "ready entity reference requires its primary identity region"
                )
            if self.major_structure_visible is False:
                raise ValueError(
                    "ready entity reference requires visible major structure"
                )
            if self.truncation_severity == "major":
                raise ValueError("ready entity reference cannot have major truncation")
            if self.discrete_foreground_instance is False:
                raise ValueError(
                    "ready entity reference requires a discrete foreground instance"
                )
            if self.mask_matches_target is False:
                raise ValueError("ready entity reference mask must match its target")
            if self.reference_scope == "full":
                if (
                    self.visible_region != "whole"
                    or not self.whole_entity_recognizable
                    or not self.identity_features_visible
                ):
                    raise ValueError(
                        "ready full reference requires whole visible, "
                        "recognizable identity"
                    )
            elif self.reference_scope == "local":
                if self.visible_region == "whole" or not self.identity_features_visible:
                    raise ValueError(
                        "ready local reference requires a non-whole "
                        "identity-visible region"
                    )
            else:
                raise ValueError("ready entity reference cannot use reject scope")
            if (
                self.image_path is None
                or not self.image_path.strip()
                or self.source_frame_index is None
            ):
                raise ValueError(
                    "ready entity reference requires image_path and source_frame_index"
                )
            generation_fields = (
                self.generation_metadata_path,
                self.generation_source_sha256,
                self.generation_output_sha256,
            )
            if self.synthetic:
                if (
                    self.reference_scope == "local"
                    and self.completeness != "local_usable"
                ):
                    raise ValueError(
                        "generated local reference must preserve local completeness"
                    )
                if self.source_entity_id != self.entity_id:
                    raise ValueError(
                        "generated fallback must preserve self entity provenance"
                    )
                if any(value is None for value in generation_fields):
                    raise ValueError(
                        "generated fallback requires complete generation provenance"
                    )
            elif any(value is not None for value in generation_fields):
                raise ValueError(
                    "real entity reference cannot publish generation provenance"
                )
        else:
            if self.reference_scope != "reject":
                raise ValueError("rejected entity reference requires reject scope")
            if (
                self.image_path is not None
                or self.source_frame_index is not None
                or self.source_clip_uid is not None
                or self.source_entity_id is not None
                or self.synthetic
                or self.generation_metadata_path is not None
                or self.generation_source_sha256 is not None
                or self.generation_output_sha256 is not None
            ):
                raise ValueError(
                    "rejected entity reference cannot publish image or generation provenance"
                )
        return self


class BackgroundRemovalReview(SchemaModel):
    verdict: Literal["accept", "reject"]
    foreground_absent: StrictBool
    foreground_not_reconstructed: StrictBool
    no_new_salient_entity: StrictBool
    background_only_in_repaired_region: StrictBool
    background_continuity_ok: StrictBool
    no_visible_artifacts: StrictBool
    reason: str

    @model_validator(mode="after")
    def validate_review(self) -> BackgroundRemovalReview:
        if not self.reason.strip():
            raise ValueError("background removal review reason must not be empty")
        all_passed = all(
            (
                self.foreground_absent,
                self.foreground_not_reconstructed,
                self.no_new_salient_entity,
                self.background_only_in_repaired_region,
                self.background_continuity_ok,
                self.no_visible_artifacts,
            )
        )
        expected = "accept" if all_passed else "reject"
        if self.verdict != expected:
            raise ValueError(
                "background removal verdict must accept if and only if all checks pass"
            )
        return self


class FinalBackgroundReview(SchemaModel):
    verdict: Literal["accept", "reject"]
    background_matches_description: StrictBool
    no_unexpected_foreground_subject: StrictBool
    usable_background_information: StrictBool
    no_obvious_artifacts: StrictBool
    reason: str

    @model_validator(mode="after")
    def validate_review(self) -> FinalBackgroundReview:
        if not self.reason.strip():
            raise ValueError("final background review reason must not be empty")
        all_passed = all(
            (
                self.background_matches_description,
                self.no_unexpected_foreground_subject,
                self.usable_background_information,
                self.no_obvious_artifacts,
            )
        )
        expected = "accept" if all_passed else "reject"
        if self.verdict != expected:
            raise ValueError(
                "final background verdict must accept if and only if all checks pass"
            )
        return self


class BackgroundRemovalAttempt(SchemaModel):
    seed: int = Field(ge=0)
    status: Literal["accepted", "rejected", "failed"]
    runtime_seconds: float = Field(ge=0)
    candidate_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reason: Optional[str] = None
    review: Optional[BackgroundRemovalReview] = None

    @model_validator(mode="after")
    def validate_attempt(self) -> BackgroundRemovalAttempt:
        if not math.isfinite(self.runtime_seconds):
            raise ValueError("background removal runtime_seconds must be finite")
        if self.status == "accepted":
            if self.candidate_sha256 is None:
                raise ValueError("accepted removal attempt requires candidate_sha256")
            if self.review is None or self.review.verdict != "accept":
                raise ValueError("accepted removal attempt requires an accepted review")
            if self.reason is not None:
                raise ValueError("accepted removal attempt cannot have a reason")
        elif self.status == "rejected":
            if self.candidate_sha256 is None:
                raise ValueError("rejected removal attempt requires candidate_sha256")
            if self.review is None or self.review.verdict != "reject":
                raise ValueError("rejected removal attempt requires a rejected review")
            if self.reason is None or not self.reason.strip():
                raise ValueError("rejected removal attempt requires a reason")
        else:
            if self.reason is None or not self.reason.strip():
                raise ValueError("failed removal attempt requires a reason")
            if self.review is not None:
                raise ValueError("failed removal attempt cannot publish a review")
        return self


class BackgroundReferenceState(SchemaModel):
    status: Literal[
        "none",
        "clean_raw",
        "pending_remove",
        "ready_removed",
        "rejected",
    ]
    source_image_path: Optional[str] = None
    output_image_path: Optional[str] = None
    source_frame_slot: Optional[int] = Field(default=None, ge=0, lt=10)
    source_frame_index: Optional[int] = Field(default=None, ge=0)
    source_mask_path: Optional[str] = None
    generation_mask_path: Optional[str] = None
    source_foreground_area_pixels: Optional[int] = Field(default=None, ge=0)
    source_foreground_area_ratio: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )
    removal_backend: Optional[str] = None
    removal_seed: Optional[int] = Field(default=None, ge=0)
    generation_mask_dilation_pixels: Optional[int] = Field(default=None, ge=0)
    generation_mask_area_pixels: Optional[int] = Field(default=None, ge=0)
    generation_mask_area_ratio: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )
    output_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    removal_attempts: list[BackgroundRemovalAttempt] = Field(
        default_factory=list,
    )
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_reference_state(self) -> BackgroundReferenceState:
        source_fields = (
            self.source_image_path,
            self.source_frame_slot,
            self.source_frame_index,
        )
        diagnostics = (
            self.source_foreground_area_pixels,
            self.source_foreground_area_ratio,
        )
        if (diagnostics[0] is None) != (diagnostics[1] is None):
            raise ValueError(
                "background foreground-area diagnostics must be set together"
            )
        if self.status == "none":
            if any(
                value is not None
                for value in (
                    *source_fields,
                    self.output_image_path,
                    self.source_mask_path,
                    self.generation_mask_path,
                    *diagnostics,
                )
            ):
                raise ValueError("none background cannot reference artifacts")
            return self
        if self.status in {
            "clean_raw",
            "pending_remove",
            "ready_removed",
        } and (
            self.source_image_path is None
            or self.source_frame_slot is None
            or self.source_frame_index is None
        ):
            raise ValueError(
                f"{self.status} background requires source_image_path, "
                "source_frame_slot, and source_frame_index"
            )
        if self.status == "clean_raw":
            if self.output_image_path is None:
                raise ValueError("clean_raw background requires output_image_path")
            if self.output_image_path != self.source_image_path:
                raise ValueError(
                    "clean_raw output_image_path must equal source_image_path"
                )
            if diagnostics != (0, 0.0):
                raise ValueError("clean_raw background requires zero foreground area")
            if (
                self.source_mask_path is not None
                or self.generation_mask_path is not None
            ):
                raise ValueError("clean_raw background cannot reference mask artifacts")
        if self.status == "pending_remove":
            if self.source_mask_path is None:
                raise ValueError("pending_remove background requires source_mask_path")
            if self.output_image_path is not None:
                raise ValueError(
                    "pending_remove background cannot publish output_image_path"
                )
            if self.generation_mask_path is not None:
                raise ValueError(
                    "pending_remove background cannot publish generation_mask_path"
                )
            if (
                self.source_foreground_area_pixels is None
                or self.source_foreground_area_pixels <= 0
                or self.source_foreground_area_ratio is None
                or self.source_foreground_area_ratio <= 0
            ):
                raise ValueError(
                    "pending_remove background requires positive foreground area"
                )
        if self.status == "ready_removed" and (
            self.output_image_path is None
            or self.source_mask_path is None
            or self.generation_mask_path is None
        ):
            raise ValueError(
                "ready_removed background requires output_image_path, "
                "source_mask_path, and generation_mask_path"
            )
        if self.status == "ready_removed" and (
            self.source_foreground_area_pixels is None
            or self.source_foreground_area_pixels <= 0
            or self.source_foreground_area_ratio is None
            or self.source_foreground_area_ratio <= 0
        ):
            raise ValueError(
                "ready_removed background requires positive foreground area"
            )
        if self.status == "rejected":
            if not self.reason:
                raise ValueError("rejected background requires a reason")
            if self.output_image_path is not None:
                raise ValueError("rejected background cannot publish output_image_path")
            if self.generation_mask_path is not None:
                raise ValueError(
                    "rejected background cannot publish generation_mask_path"
                )
            if any(value is not None for value in source_fields) and any(
                value is None for value in source_fields
            ):
                raise ValueError(
                    "rejected background source provenance must be complete"
                )
            if all(value is None for value in source_fields) and (
                self.source_mask_path is not None
                or any(value is not None for value in diagnostics)
            ):
                raise ValueError(
                    "rejected background without a source cannot reference "
                    "source artifacts"
                )
        return self

    @model_validator(mode="after")
    def validate_removal_metadata(self) -> BackgroundReferenceState:
        generation_diagnostics = (
            self.generation_mask_dilation_pixels,
            self.generation_mask_area_pixels,
            self.generation_mask_area_ratio,
        )
        if any(value is not None for value in generation_diagnostics) and any(
            value is None for value in generation_diagnostics
        ):
            raise ValueError(
                "background generation-mask diagnostics must be set together"
            )
        if self.generation_mask_area_ratio is not None and not math.isfinite(
            self.generation_mask_area_ratio
        ):
            raise ValueError("background generation-mask ratio must be finite")
        if self.removal_backend is not None and not self.removal_backend.strip():
            raise ValueError("removal_backend must not be empty")

        if self.status in {"none", "clean_raw"}:  # noqa: SIM102
            if (
                self.removal_backend is not None
                or self.removal_seed is not None
                or any(value is not None for value in generation_diagnostics)
                or self.output_sha256 is not None
                or self.removal_attempts
            ):
                raise ValueError(
                    f"{self.status} background cannot publish removal metadata"
                )

        if self.status == "pending_remove":
            has_attempts = bool(self.removal_attempts)
            if has_attempts:
                if self.removal_backend is None:
                    raise ValueError(
                        "retryable pending_remove requires removal_backend"
                    )
                if self.reason is None or not self.reason.strip():
                    raise ValueError("retryable pending_remove requires a reason")
                if any(
                    attempt.status == "accepted" for attempt in self.removal_attempts
                ):
                    raise ValueError(
                        "retryable pending_remove cannot contain an accepted attempt"
                    )
                if not any(
                    attempt.status == "failed" for attempt in self.removal_attempts
                ):
                    raise ValueError(
                        "retryable pending_remove requires a failed attempt"
                    )
            elif self.removal_backend is not None or self.reason is not None:
                raise ValueError(
                    "fresh pending_remove cannot publish removal diagnostics"
                )
            if (
                self.removal_seed is not None
                or any(value is not None for value in generation_diagnostics)
                or self.output_sha256 is not None
            ):
                raise ValueError(
                    "pending_remove cannot publish generated output metadata"
                )

        if self.status == "ready_removed":
            if (
                self.removal_backend is None
                or self.removal_seed is None
                or self.generation_mask_dilation_pixels is None
                or self.generation_mask_area_pixels is None
                or self.generation_mask_area_pixels <= 0
                or self.generation_mask_area_ratio is None
                or self.generation_mask_area_ratio <= 0
                or self.output_sha256 is None
                or not self.removal_attempts
            ):
                raise ValueError(
                    "ready_removed background requires complete removal metadata"
                )
            accepted = [
                attempt
                for attempt in self.removal_attempts
                if attempt.status == "accepted"
            ]
            if len(accepted) != 1:
                raise ValueError(
                    "ready_removed background requires exactly one accepted attempt"
                )
            if (
                accepted[0].seed != self.removal_seed
                or accepted[0].candidate_sha256 != self.output_sha256
            ):
                raise ValueError(
                    "accepted removal attempt must match published seed and hash"
                )
            if self.reason is not None:
                raise ValueError("ready_removed background cannot have a reason")

        if self.status == "rejected":
            if self.output_sha256 is not None:
                raise ValueError("rejected background cannot publish output_sha256")
            if self.removal_seed is not None:
                raise ValueError("rejected background cannot publish removal_seed")
            if self.removal_attempts:
                if self.removal_backend is None:
                    raise ValueError("remove-stage rejection requires removal_backend")
                if (
                    self.source_image_path is None
                    or self.source_frame_slot is None
                    or self.source_frame_index is None
                    or self.source_mask_path is None
                    or self.source_foreground_area_pixels is None
                    or self.source_foreground_area_pixels <= 0
                    or self.source_foreground_area_ratio is None
                    or self.source_foreground_area_ratio <= 0
                ):
                    raise ValueError(
                        "remove-stage rejection requires source provenance"
                    )
                if any(
                    attempt.status == "accepted" for attempt in self.removal_attempts
                ):
                    raise ValueError(
                        "rejected background cannot contain an accepted attempt"
                    )
            elif self.removal_backend is not None:
                raise ValueError(
                    "background-stage rejection cannot publish removal_backend"
                )
        return self


class ReferencesState(SchemaModel):
    entities: list[EntityReferenceState] = Field(default_factory=list)
    background: Optional[BackgroundReferenceState] = None

    @model_validator(mode="after")
    def validate_entity_ids(self) -> ReferencesState:
        entity_ids = [reference.entity_id for reference in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity references must have unique entity_id values")
        return self


class PairingState(SchemaModel):
    status: Literal["ready", "rejected"]
    retained_entity_ids: list[str] = Field(default_factory=list)
    tokens: dict[str, str] = Field(default_factory=dict)
    background_token: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_bindings(self) -> PairingState:
        if self.status == "rejected":
            if self.reason is None or not self.reason.strip():
                raise ValueError("rejected pairing requires a reason")
            if (
                self.retained_entity_ids
                or self.tokens
                or self.background_token is not None
            ):
                raise ValueError("rejected pairing must clear retained IDs and tokens")
            return self
        if self.reason is not None:
            raise ValueError("ready pairing reason must be null")
        if not self.retained_entity_ids:
            raise ValueError("ready pairing requires at least one retained entity")
        if set(self.tokens) != set(self.retained_entity_ids):
            raise ValueError("ready pairing tokens must match retained_entity_ids")
        if len(self.retained_entity_ids) != len(set(self.retained_entity_ids)):
            raise ValueError("retained_entity_ids must be unique")
        entity_tokens = list(self.tokens.values())
        if len(entity_tokens) != len(set(entity_tokens)):
            raise ValueError("pairing tokens must be unique")
        if any(_REF_TOKEN.fullmatch(token) is None for token in entity_tokens):
            raise ValueError("pairing contains an invalid reference token")
        if any(token.startswith("<ref_bg_") for token in entity_tokens):
            raise ValueError("entity tokens cannot use background reference tokens")
        if self.background_token not in {None, "<ref_bg_1>"}:
            raise ValueError('background_token must be null or "<ref_bg_1>"')
        return self


ReferenceEditOperation = Literal[
    "complete_entity",
    "add_entity_background",
]
ReferenceEditFallbackPolicy = Literal[
    "not_used",
    "keep_source",
    "completion_candidate",
    "reject_entity",
]


class ReferenceEditEntityState(SchemaModel):
    entity_id: str
    route: ReferenceCompleteness
    status: Literal["accepted", "fallback", "rejected", "not_required"]
    source_reference: EntityReferenceState
    source_image_path: str
    output_image_path: Optional[str] = None
    operation: Optional[ReferenceEditOperation] = None
    metadata_path: Optional[str] = None
    operations: list[ReferenceEditOperation] = Field(default_factory=list)
    completion_metadata_path: Optional[str] = None
    background_metadata_path: Optional[str] = None
    background_fallback: Literal["none", "completion_candidate"] = "none"
    fallback_policy: ReferenceEditFallbackPolicy = "not_used"
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> ReferenceEditEntityState:
        if not self.source_image_path.strip():
            raise ValueError("reference edit source_image_path must not be empty")
        if (
            self.source_reference.entity_id != self.entity_id
            or self.source_reference.status != "ready"
            or self.source_reference.image_path != self.source_image_path
            or self.source_reference.synthetic
        ):
            raise ValueError(
                "reference edit source evidence must be a ready real reference"
            )
        allowed_sequences = {
            (),
            ("complete_entity",),
            ("add_entity_background",),
            ("complete_entity", "add_entity_background"),
        }
        sequence = tuple(self.operations)
        if sequence not in allowed_sequences:
            raise ValueError("reference edit operations are not in production order")
        if self.operation is not None and sequence and self.operation != sequence[-1]:
            raise ValueError("reference edit operation must match the last operation")
        if self.completion_metadata_path is not None and not (
            self.completion_metadata_path.strip()
            and (
                "complete_entity" in sequence
                or not sequence
                and self.operation == "complete_entity"
            )
        ):
            raise ValueError("completion metadata requires a completion operation")
        if self.background_metadata_path is not None and not (
            self.background_metadata_path.strip()
            and (
                "add_entity_background" in sequence
                or not sequence
                and self.operation == "add_entity_background"
            )
        ):
            raise ValueError("background metadata requires a background operation")
        if self.status == "accepted":
            if (
                self.operation is None
                or self.output_image_path is None
                or not self.output_image_path.strip()
                or self.metadata_path is None
                or not self.metadata_path.strip()
            ):
                raise ValueError(
                    "accepted reference edit requires operation, output, and metadata"
                )
            if self.fallback_policy != "not_used" or self.reason is not None:
                raise ValueError("accepted reference edit cannot use fallback")
            if self.background_fallback != "none":
                raise ValueError(
                    "accepted reference edit cannot mark background fallback"
                )
        elif self.status == "fallback":
            if self.reason is None or not self.reason.strip():
                raise ValueError("reference edit fallback requires a reason")
            if self.fallback_policy == "keep_source":
                if self.output_image_path != self.source_image_path:
                    raise ValueError(
                        "keep-source fallback must publish the source image"
                    )
                if self.background_fallback != "none":
                    raise ValueError(
                        "keep-source fallback cannot mark background fallback"
                    )
            elif self.fallback_policy == "completion_candidate":
                if (
                    self.route != "repairable"
                    or sequence != ("complete_entity", "add_entity_background")
                    or self.operation != "add_entity_background"
                    or self.output_image_path is None
                    or self.output_image_path == self.source_image_path
                    or self.metadata_path is None
                    or self.completion_metadata_path is None
                    or self.background_metadata_path is None
                    or self.background_fallback != "completion_candidate"
                ):
                    raise ValueError(
                        "completion fallback requires both operations and final output"
                    )
            else:
                raise ValueError(
                    "reference edit fallback must keep source or completion candidate"
                )
        elif self.status == "not_required":
            if (
                self.operation is not None
                or self.metadata_path is not None
                or self.operations
                or self.completion_metadata_path is not None
                or self.background_metadata_path is not None
                or self.background_fallback != "none"
                or self.fallback_policy != "not_used"
            ):
                raise ValueError("not-required reference edit cannot run generation")
            if self.output_image_path != self.source_image_path:
                raise ValueError("not-required reference edit must preserve its source")
            if self.reason is not None:
                raise ValueError("not-required reference edit cannot have a reason")
        else:
            if self.output_image_path is not None:
                raise ValueError("rejected reference edit cannot publish an output")
            if self.fallback_policy != "reject_entity":
                raise ValueError("rejected reference edit must reject the entity")
            if self.background_fallback != "none":
                raise ValueError(
                    "rejected reference edit cannot mark background fallback"
                )
            if self.reason is None or not self.reason.strip():
                raise ValueError("rejected reference edit requires a reason")
        return self


class ReferenceEditState(SchemaModel):
    status: Literal["ready", "failed"]
    entities: list[ReferenceEditEntityState] = Field(default_factory=list)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> ReferenceEditState:
        entity_ids = [item.entity_id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("reference edit entity IDs must be unique")
        if self.status == "failed":
            if self.reason is None or not self.reason.strip():
                raise ValueError("failed reference edit requires a reason")
            if self.entities:
                raise ValueError("failed reference edit must clear entity results")
        elif self.reason is not None:
            raise ValueError("ready reference edit cannot have a failure reason")
        return self


class ReferenceTopologyDiagnostics(SchemaModel):
    alpha_available: StrictBool
    significant_component_count: int = Field(ge=0)
    largest_component_ratio: float = Field(ge=0, le=1)
    second_component_ratio: float = Field(ge=0, le=1)
    bbox_fill_ratio: float = Field(ge=0, le=1)
    border_contact: StrictBool
    enclosed_transparent_hole_count: int = Field(ge=0)
    largest_enclosed_hole_area: int = Field(ge=0)
    enclosed_hole_bbox_ratio: float = Field(ge=0, le=1)
    suspicious: StrictBool
    suspicion_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_diagnostics(self) -> ReferenceTopologyDiagnostics:
        ratios = (
            self.largest_component_ratio,
            self.second_component_ratio,
            self.bbox_fill_ratio,
            self.enclosed_hole_bbox_ratio,
        )
        if not all(math.isfinite(value) for value in ratios):
            raise ValueError("reference topology ratios must be finite")
        if len(self.suspicion_reasons) != len(set(self.suspicion_reasons)):
            raise ValueError("reference topology suspicion reasons must be unique")
        if self.suspicious != bool(self.suspicion_reasons):
            raise ValueError("reference topology suspicion flag must match reasons")
        return self


class ReferenceIntegrityReview(SchemaModel):
    matches_target: StrictBool
    preserves_annotated_entity_semantics: StrictBool
    preserves_primary_identity_region: StrictBool
    recognizable_as_named_entity: StrictBool
    structurally_complete_for_scope: StrictBool
    no_major_missing_regions: StrictBool
    no_unnatural_holes_or_surface_loss: StrictBool
    no_unrelated_entity_dominance: StrictBool
    usable_as_independent_reference: StrictBool
    verdict: Literal["accept", "reject"]
    reason: str

    @model_validator(mode="after")
    def validate_review(self) -> ReferenceIntegrityReview:
        if not self.reason.strip():
            raise ValueError("reference integrity review reason must not be empty")
        passed = all(
            (
                self.matches_target,
                self.preserves_annotated_entity_semantics,
                self.preserves_primary_identity_region,
                self.recognizable_as_named_entity,
                self.structurally_complete_for_scope,
                self.no_major_missing_regions,
                self.no_unnatural_holes_or_surface_loss,
                self.no_unrelated_entity_dominance,
                self.usable_as_independent_reference,
            )
        )
        if self.verdict != ("accept" if passed else "reject"):
            raise ValueError(
                "reference integrity verdict must match all integrity checks"
            )
        return self


class ReferenceIntegrityEntityState(SchemaModel):
    entity_id: str
    status: Literal["accepted", "rejected", "skipped"]
    input_reference: EntityReferenceState
    final_reference_path: str
    source_context_path: Optional[str] = None
    diagnostics: ReferenceTopologyDiagnostics
    reviewed: StrictBool
    review: Optional[ReferenceIntegrityReview] = None
    judge_failed: StrictBool = False
    reason: str

    @model_validator(mode="after")
    def validate_entity_state(self) -> ReferenceIntegrityEntityState:
        if (
            self.input_reference.entity_id != self.entity_id
            or self.input_reference.status != "ready"
            or self.input_reference.image_path != self.final_reference_path
        ):
            raise ValueError("integrity input must be the ready published reference")
        if not self.reason.strip():
            raise ValueError("reference integrity entity reason must not be empty")
        if self.status == "skipped":
            if self.reviewed or self.review is not None or self.judge_failed:
                raise ValueError("skipped integrity result cannot contain a review")
        else:
            if not self.reviewed:
                raise ValueError("reviewed integrity result must mark reviewed=true")
            if self.judge_failed:
                if self.status != "rejected" or self.review is not None:
                    raise ValueError("integrity judge failure must fail closed")
            elif self.review is None:
                raise ValueError("reviewed integrity result requires a review")
            elif (self.status == "accepted") != (self.review.verdict == "accept"):
                raise ValueError("integrity entity status must match review verdict")
        return self


class ReferenceIntegrityState(SchemaModel):
    status: Literal["ready", "failed"]
    entities: list[ReferenceIntegrityEntityState] = Field(default_factory=list)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> ReferenceIntegrityState:
        entity_ids = [item.entity_id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("reference integrity entity IDs must be unique")
        if self.status == "failed":
            if self.reason is None or not self.reason.strip():
                raise ValueError("failed reference integrity requires a reason")
            if self.entities:
                raise ValueError("failed reference integrity must clear entity results")
        elif self.reason is not None:
            raise ValueError("ready reference integrity cannot have a failure reason")
        return self


InstructionReferenceType = Literal["subject", "object", "group", "background"]


class InstructionBinding(SchemaModel):
    image_id: str
    image_index: int = Field(ge=1)
    reference_type: InstructionReferenceType
    entity_id: Optional[str]
    phrase: str
    grounding_prompt: str

    @model_validator(mode="after")
    def validate_binding(self) -> InstructionBinding:
        match = _IMAGE_ID.fullmatch(self.image_id)
        if match is None or int(match.group(1)) != self.image_index:
            raise ValueError("image_id must match image_index")
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("instruction binding text must not be empty")
        if self.reference_type == "background":
            if self.entity_id is not None:
                raise ValueError("background binding entity_id must be null")
        elif self.entity_id is None:
            raise ValueError("entity binding requires entity_id")
        return self


class RawInstructionLegend(SchemaModel):
    image_id: str
    description: str


class RawInstructionOutput(SchemaModel):
    instruction_body_template: str
    reference_legend: list[RawInstructionLegend]


class InstructionLegendEntry(SchemaModel):
    image_id: str
    description: str

    @model_validator(mode="after")
    def validate_entry(self) -> InstructionLegendEntry:
        if _IMAGE_ID.fullmatch(self.image_id) is None:
            raise ValueError("legend image_id must use image_N")
        if not self.description.strip():
            raise ValueError("legend description must not be empty")
        if _ANY_REF_TOKEN.search(self.description):
            raise ValueError("legend description must not contain reference tokens")
        if _DIRECT_CHINESE_IMAGE_LABEL.search(self.description):
            raise ValueError(
                "raw legend description must not contain Chinese image labels"
            )
        return self


def render_instruction_text(
    instruction_body_template: str,
    reference_legend: list[InstructionLegendEntry],
) -> str:
    rendered_body = instruction_body_template.strip()
    legend_lines: list[str] = []
    for entry in reference_legend:
        match = _IMAGE_ID.fullmatch(entry.image_id)
        if match is None:
            raise ValueError("legend image_id must use image_N")
        display_label = f"<Image {match.group(1)}>"
        rendered_body = rendered_body.replace(
            f"{{{{{entry.image_id}}}}}",
            display_label,
        )
        legend_lines.append(f"{display_label}: {entry.description.strip()}")
    return f"{rendered_body}\n\n" + "\n".join(legend_lines)


def render_inline_instruction_text(instruction_body_template: str) -> str:
    return _IMAGE_PLACEHOLDER.sub(
        lambda match: f"<Image {match.group(1).removeprefix('image_')}>",
        instruction_body_template.strip(),
    )


def _render_legacy_english_instruction_text(
    instruction_body_template: str,
    reference_legend: list[InstructionLegendEntry],
) -> str:
    rendered_body = instruction_body_template.strip()
    legend_lines: list[str] = []
    for entry in reference_legend:
        match = _IMAGE_ID.fullmatch(entry.image_id)
        if match is None:
            raise ValueError("legend image_id must use image_N")
        display_label = f"Image {match.group(1)}"
        rendered_body = rendered_body.replace(
            f"{{{{{entry.image_id}}}}}",
            display_label,
        )
        legend_lines.append(f"{display_label}: {entry.description.strip()}")
    return f"{rendered_body}\n\n" + "\n".join(legend_lines)


def _render_legacy_instruction_text(
    instruction_body_template: str,
    reference_legend: list[InstructionLegendEntry],
) -> str:
    rendered_body = instruction_body_template.strip()
    legend_lines: list[str] = []
    for entry in reference_legend:
        match = _IMAGE_ID.fullmatch(entry.image_id)
        if match is None:
            raise ValueError("legend image_id must use image_N")
        display_label = f"\u56fe{match.group(1)}"
        rendered_body = rendered_body.replace(
            f"{{{{{entry.image_id}}}}}",
            display_label,
        )
        legend_lines.append(f"{display_label}\uff1a{entry.description.strip()}")
    return f"{rendered_body}\n\n" + "\n".join(legend_lines)


class InstructionState(SchemaModel):
    status: Literal["ready", "failed"]
    instruction_body_template: str = ""
    reference_legend: list[InstructionLegendEntry] = Field(default_factory=list)
    r2v_instruction: str = ""
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> InstructionState:
        if self.status == "failed":
            if not self.reason:
                raise ValueError("failed instruction requires a reason")
            if (
                self.instruction_body_template
                or self.reference_legend
                or self.r2v_instruction
            ):
                raise ValueError("failed instruction must clear generated content")
            return self
        if self.reason is not None:
            raise ValueError("ready instruction must not have a failure reason")
        if not self.instruction_body_template.strip():
            raise ValueError("ready instruction requires a non-empty body template")
        if not self.reference_legend:
            raise ValueError("ready instruction requires a reference legend")
        raw_text = " ".join(
            [
                self.instruction_body_template,
                *(entry.description for entry in self.reference_legend),
            ]
        )
        if _ANY_REF_TOKEN.search(raw_text):
            raise ValueError("ready instruction must not contain reference tokens")
        if _DIRECT_CHINESE_IMAGE_LABEL.search(
            raw_text
        ) or _DIRECT_ENGLISH_IMAGE_LABEL.search(raw_text):
            raise ValueError(
                "raw instruction fields must use image_N placeholders instead "
                "of rendered image labels"
            )
        legend_ids = [entry.image_id for entry in self.reference_legend]
        expected_ids = [f"image_{index}" for index in range(1, len(legend_ids) + 1)]
        if legend_ids != expected_ids:
            raise ValueError("reference legend IDs must be contiguous and ordered")
        exact_placeholders = _IMAGE_PLACEHOLDER.findall(self.instruction_body_template)
        all_placeholders = _ANY_TEMPLATE_PLACEHOLDER.findall(
            self.instruction_body_template
        )
        placeholder_remainder = _IMAGE_PLACEHOLDER.sub(
            "",
            self.instruction_body_template,
        )
        if (
            len(exact_placeholders) != len(all_placeholders)
            or "{{" in placeholder_remainder
            or "}}" in placeholder_remainder
        ):
            raise ValueError("instruction contains an invalid image placeholder")
        if set(exact_placeholders) != set(legend_ids):
            raise ValueError(
                "instruction placeholders must exactly match reference legend"
            )
        expected_renderings = {
            render_inline_instruction_text(self.instruction_body_template),
        }
        expected_renderings.update(
            renderer(
                self.instruction_body_template,
                self.reference_legend,
            )
            for renderer in (
                render_instruction_text,
                _render_legacy_english_instruction_text,
                _render_legacy_instruction_text,
            )
        )
        if self.r2v_instruction not in expected_renderings:
            raise ValueError(
                "r2v_instruction must match deterministic inline rendering, "
                "angle-bracket English rendering with legend, exact legacy plain "
                "English rendering, or exact legacy Chinese rendering"
            )
        return self


class ExportState(SchemaModel):
    accepted: bool = False
    reason: Optional[str] = "not_evaluated"

    @model_validator(mode="after")
    def validate_state(self) -> ExportState:
        if self.accepted and self.reason is not None:
            raise ValueError("accepted export must not have a rejection reason")
        if not self.accepted and not self.reason:
            raise ValueError("non-accepted export requires a reason")
        return self


class ClipRecord(SchemaModel):
    schema_version: Literal["r2v.v3.clip.2"] = CLIP_SCHEMA_VERSION
    clip_uid: str
    source: ClipSource
    annotation: Optional[AnnotationState] = None
    coverage: Optional[CoverageState] = None
    references: ReferencesState = Field(default_factory=ReferencesState)
    pairing: Optional[PairingState] = None
    reference_edit: Optional[ReferenceEditState] = None
    reference_integrity: Optional[ReferenceIntegrityState] = None
    instruction: Optional[InstructionState] = None
    export: ExportState = Field(default_factory=ExportState)

    @model_validator(mode="after")
    def validate_section_consistency(self) -> ClipRecord:
        annotation_entities = (
            self.annotation.entities if self.annotation is not None else []
        )
        annotation_order = [entity.entity_id for entity in annotation_entities]
        annotation_by_id = {entity.entity_id: entity for entity in annotation_entities}
        annotation_ids = set(annotation_order)
        integrity_rejected_ids = {
            item.entity_id
            for item in (
                self.reference_integrity.entities
                if self.reference_integrity is not None
                and self.reference_integrity.status == "ready"
                else []
            )
            if item.status == "rejected"
        }
        if self.coverage is not None:
            unknown_qualifying = (
                set(self.coverage.qualifying_entity_ids) - annotation_ids
            )
            if unknown_qualifying:
                raise ValueError(
                    "coverage qualifying entity IDs must exist in annotation"
                )
            if set(self.coverage.entity_visibility_summary) != annotation_ids:
                raise ValueError(
                    "coverage visibility summaries must match annotation entities"
                )
        reference_order = [
            reference.entity_id for reference in self.references.entities
        ]
        reference_ids = set(reference_order)
        if reference_ids - annotation_ids:
            raise ValueError("entity references must correspond to annotation entities")
        expected_reference_order = [
            entity_id for entity_id in annotation_order if entity_id in reference_ids
        ]
        if reference_order != expected_reference_order:
            raise ValueError("entity references must follow annotation entity order")
        for reference in self.references.entities:
            if reference.synthetic and reference.source_clip_uid != self.clip_uid:
                raise ValueError(
                    "generated fallback must preserve self clip provenance"
                )
        if self.pairing is not None and self.pairing.status == "ready":
            retained = self.pairing.retained_entity_ids
            retained_ids = set(retained)
            expected_retained_order = [
                entity_id for entity_id in annotation_order if entity_id in retained_ids
            ]
            if retained != expected_retained_order:
                raise ValueError(
                    "ready pairing retained IDs must follow annotation order"
                )
            ready_reference_order = [
                reference.entity_id
                for reference in self.references.entities
                if reference.status == "ready"
            ]
            if retained != ready_reference_order:
                raise ValueError(
                    "ready pairing must retain every ready entity reference"
                )
            qualifying_ids = (
                set(self.coverage.qualifying_entity_ids)
                if self.coverage is not None and self.coverage.passed
                else set()
            )
            if not retained_ids.intersection(qualifying_ids):
                raise ValueError(
                    "ready pairing must retain at least one qualifying entity"
                )
            counters = {"subject": 0, "object": 0, "group": 0}
            expected_tokens: dict[str, str] = {}
            for entity_id in retained:
                reference_type = annotation_by_id[entity_id].reference_type
                counters[reference_type] += 1
                expected_tokens[entity_id] = (
                    f"<ref_{reference_type}_{counters[reference_type]}>"
                )
            if self.pairing.tokens != expected_tokens:
                raise ValueError(
                    "ready pairing tokens must use deterministic per-type numbering"
                )
            if self.pairing.background_token is not None:
                background = self.references.background
                if background is None or background.status not in {
                    "clean_raw",
                    "ready_removed",
                }:
                    raise ValueError(
                        "ready pairing background token requires a ready background"
                    )
        if self.reference_edit is not None and self.reference_edit.status == "ready":
            edit_order = [item.entity_id for item in self.reference_edit.entities]
            expected_edit_order = [
                entity_id for entity_id in annotation_order if entity_id in edit_order
            ]
            if edit_order != expected_edit_order:
                raise ValueError(
                    "reference edit entities must follow annotation entity order"
                )
            if set(edit_order) - annotation_ids:
                raise ValueError(
                    "reference edit entities must correspond to annotation entities"
                )
            references_by_id = {
                reference.entity_id: reference for reference in self.references.entities
            }
            edits_by_id = {
                item.entity_id: item for item in self.reference_edit.entities
            }
            if self.pairing is not None and self.pairing.status == "ready":
                missing_edits = set(self.pairing.retained_entity_ids) - set(edits_by_id)
                if missing_edits:
                    raise ValueError(
                        "ready pairing references require reference edit results"
                    )
            for entity_id, item in edits_by_id.items():
                reference = references_by_id.get(entity_id)
                if item.status == "rejected":
                    if reference is not None and reference.status != "rejected":
                        raise ValueError(
                            "rejected reference edit must reject its entity reference"
                        )
                    continue
                if reference is None or reference.status != "ready":
                    if entity_id in integrity_rejected_ids:
                        continue
                    raise ValueError(
                        "published reference edit requires a ready entity reference"
                    )
                if item.output_image_path != reference.image_path:
                    raise ValueError(
                        "reference edit output must match the published reference path"
                    )
                if item.status == "accepted" and not reference.synthetic:
                    raise ValueError(
                        "accepted reference edit must publish a synthetic reference"
                    )
        if (
            self.reference_integrity is not None
            and self.reference_integrity.status == "ready"
        ):
            integrity_order = [
                item.entity_id for item in self.reference_integrity.entities
            ]
            expected_integrity_order = [
                entity_id
                for entity_id in annotation_order
                if entity_id in set(integrity_order)
            ]
            if integrity_order != expected_integrity_order:
                raise ValueError(
                    "reference integrity entities must follow annotation order"
                )
            references_by_id = {
                reference.entity_id: reference for reference in self.references.entities
            }
            for item in self.reference_integrity.entities:
                published = references_by_id.get(item.entity_id)
                if item.status == "rejected":
                    if published is None or published.status != "rejected":
                        raise ValueError(
                            "rejected integrity result must reject its reference"
                        )
                elif published is None or published.status != "ready":
                    raise ValueError(
                        "accepted integrity result requires a ready reference"
                    )
        if self.instruction is not None and self.instruction.status == "ready":
            if self.pairing is None or self.pairing.status != "ready":
                raise ValueError("ready instruction requires ready pairing")
            if (
                self.reference_edit is not None
                and self.reference_edit.status != "ready"
            ):
                raise ValueError("ready instruction requires ready reference edit")
            if (
                self.reference_integrity is not None
                and self.reference_integrity.status != "ready"
            ):
                raise ValueError("ready instruction requires ready reference integrity")
            binding_count = len(self.pairing.retained_entity_ids)
            if self.pairing.background_token is not None:
                binding_count += 1
            expected_image_ids = [
                f"image_{index}" for index in range(1, binding_count + 1)
            ]
            legend_image_ids = [
                entry.image_id for entry in self.instruction.reference_legend
            ]
            if legend_image_ids != expected_image_ids:
                raise ValueError(
                    "ready instruction legend must match final pairing order"
                )
        if self.export.accepted:
            if self.annotation is None or self.annotation.status != "ready":
                raise ValueError("accepted export requires ready annotation")
            if self.coverage is None or not self.coverage.passed:
                raise ValueError("accepted export requires passed coverage")
            if self.pairing is None or self.pairing.status != "ready":
                raise ValueError("accepted export requires ready pairing")
            if self.instruction is None or self.instruction.status != "ready":
                raise ValueError("accepted export requires ready instruction")
            if (
                self.reference_edit is not None
                and self.reference_edit.status != "ready"
            ):
                raise ValueError("accepted export requires ready reference edit")
            if (
                self.reference_integrity is not None
                and self.reference_integrity.status != "ready"
            ):
                raise ValueError("accepted export requires ready reference integrity")
        return self


class MaskRle(SchemaModel):
    size: tuple[int, int]
    counts: list[int]

    @model_validator(mode="after")
    def validate_rle(self) -> MaskRle:
        height, width = self.size
        if height < 1 or width < 1:
            raise ValueError("mask RLE dimensions must be positive")
        if not self.counts:
            raise ValueError("mask RLE counts must not be empty")
        if self.counts[0] < 0 or any(count < 1 for count in self.counts[1:]):
            raise ValueError("mask RLE counts must contain valid run lengths")
        if sum(self.counts) != height * width:
            raise ValueError("mask RLE counts do not match its dimensions")
        return self


class TrackedMaskFrame(SchemaModel):
    slot: int = Field(ge=0, lt=10)
    present: bool
    track_valid: bool = True
    confidence: Optional[float] = None
    backend_confidences: list[float] = Field(default_factory=list)
    backend_object_ids: list[str] = Field(default_factory=list)
    area_pixels: int = Field(ge=0)
    area_ratio: float = Field(ge=0, le=1)
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    rle: MaskRle

    @model_validator(mode="after")
    def validate_frame(self) -> TrackedMaskFrame:
        confidence_values = [
            *self.backend_confidences,
            *(() if self.confidence is None else (self.confidence,)),
        ]
        if any(not math.isfinite(value) for value in confidence_values):
            raise ValueError("tracked mask confidence must be finite")
        if len(self.backend_object_ids) != len(self.backend_confidences):
            raise ValueError(
                "backend object IDs and confidences must have equal lengths"
            )
        if len(self.backend_object_ids) != len(set(self.backend_object_ids)):
            raise ValueError("backend object IDs must be unique per frame")
        encoded_area = sum(self.rle.counts[1::2])
        if self.area_pixels != encoded_area:
            raise ValueError("tracked mask area_pixels must match its RLE")
        height, width = self.rle.size
        expected_ratio = encoded_area / (height * width)
        if not math.isclose(
            self.area_ratio,
            expected_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("tracked mask area_ratio must match its RLE")
        if self.present:
            if (
                encoded_area == 0
                or not self.track_valid
                or self.confidence is None
                or not self.backend_object_ids
                or self.bbox_xyxy is None
            ):
                raise ValueError(
                    "present tracked mask requires valid non-empty evidence"
                )
            if self.confidence != min(self.backend_confidences):
                raise ValueError(
                    "tracked mask confidence must preserve the minimum "
                    "backend confidence"
                )
            x1, y1, x2, y2 = self.bbox_xyxy
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("tracked mask bbox is outside mask dimensions")
        elif (
            encoded_area != 0
            or self.confidence is not None
            or self.backend_confidences
            or self.backend_object_ids
            or self.bbox_xyxy is not None
        ):
            raise ValueError("absent tracked mask must not publish mask evidence")
        return self


class TrackedEntityMasks(SchemaModel):
    status: Literal["ready", "not_found", "failed"]
    reference_type: ReferenceType
    grounding_prompt: str
    backend_object_ids: list[str] = Field(default_factory=list)
    frames: list[TrackedMaskFrame]
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_entity(self) -> TrackedEntityMasks:
        if not self.grounding_prompt.strip():
            raise ValueError("tracked entity grounding_prompt must not be empty")
        if len(self.backend_object_ids) != len(set(self.backend_object_ids)):
            raise ValueError("tracked entity backend object IDs must be unique")
        if len(self.frames) != 10 or [frame.slot for frame in self.frames] != list(
            range(10)
        ):
            raise ValueError("tracked entity must contain ten ordered frame slots")
        present_frames = [frame for frame in self.frames if frame.present]
        if self.status == "ready" and not present_frames:
            raise ValueError("ready tracked entity requires a present mask")
        if self.status != "ready" and present_frames:
            raise ValueError("not-found or failed tracked entity cannot publish masks")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed tracked entity requires a reason")
        published_ids = {
            object_id
            for frame in present_frames
            for object_id in frame.backend_object_ids
        }
        if published_ids != set(self.backend_object_ids):
            raise ValueError(
                "tracked entity backend IDs must match published frame IDs"
            )
        return self


class TrackedMasksArtifact(SchemaModel):
    schema_version: Literal["r2v.v3.masks.1"] = MASK_SCHEMA_VERSION
    clip_uid: str
    sampled_frame_count: Literal[10] = 10
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    entities: dict[str, TrackedEntityMasks] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_masks(self) -> TrackedMasksArtifact:
        expected_size = (self.height, self.width)
        for entity_id, entity in self.entities.items():
            if not entity_id.strip():
                raise ValueError("tracked mask entity IDs must not be empty")
            if any(frame.rle.size != expected_size for frame in entity.frames):
                raise ValueError("tracked masks must match artifact dimensions")
        return self


class RunRecord(SchemaModel):
    schema_version: Literal["r2v.v3.run.1"] = RUN_SCHEMA_VERSION
    run_id: str
    created_at: str
    git_commit: str
    config_hash: str
    model_identifiers: dict[str, Optional[str]]
    source_manifest_path: str
    counts: dict[str, int] = Field(default_factory=dict)


class FailureRecord(SchemaModel):
    clip_uid: Optional[str] = None
    stage: str
    reason: str
    created_at: str
    details: dict[str, object] = Field(default_factory=dict)


DatasetReferenceType = Literal["entity", "background"]
DatasetReferenceScope = Literal["full", "local", "scene"]


class DatasetReference(SchemaModel):
    token: str
    type: DatasetReferenceType
    entity_id: Optional[str]
    scope: DatasetReferenceScope
    visible_region: VisibleRegion
    image_path: str
    source_frame_index: int = Field(ge=0)
    source_clip_uid: Optional[str] = None
    source_entity_id: Optional[str] = None
    synthetic: bool

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: str) -> str:
        if _REF_TOKEN.fullmatch(value) is None:
            raise ValueError("invalid reference token")
        return value

    @field_validator("image_path")
    @classmethod
    def validate_relative_image_path(cls, value: str) -> str:
        if value.startswith("/") or not value.startswith("references/"):
            raise ValueError("dataset reference image_path must be relative")
        if ".." in value.split("/"):
            raise ValueError(
                "dataset reference image_path must not escape dataset root"
            )
        return value

    @model_validator(mode="after")
    def validate_type_fields(self) -> DatasetReference:
        if (self.source_clip_uid is None) != (self.source_entity_id is None):
            raise ValueError(
                "dataset reference source_clip_uid and source_entity_id "
                "must be set together"
            )
        if self.type == "entity":
            if self.entity_id is None:
                raise ValueError("entity reference requires entity_id")
            if self.token.startswith("<ref_bg_"):
                raise ValueError("entity reference cannot use a background token")
            if self.scope not in {"full", "local"}:
                raise ValueError("entity reference scope must be full or local")
        else:
            if self.source_clip_uid is not None:
                raise ValueError(
                    "background reference cannot publish entity provenance"
                )
            if self.entity_id is not None or self.scope != "scene":
                raise ValueError(
                    "background reference must use scene scope and no entity_id"
                )
            if not self.token.startswith("<ref_bg_"):
                raise ValueError("background reference must use a background token")
        return self


class DatasetSampleSource(SchemaModel):
    parent_video_id: str
    clip_suffix: str


class DatasetSample(SchemaModel):
    schema_version: Literal["r2v.v3.sample.1"] = SAMPLE_SCHEMA_VERSION
    sample_id: str
    target_video: str
    t2v_caption: str
    r2v_instruction: str
    references: list[DatasetReference]
    source: DatasetSampleSource

    @model_validator(mode="after")
    def validate_bindings(self) -> DatasetSample:
        if _ANY_REF_TOKEN.search(self.t2v_caption):
            raise ValueError("t2v_caption must not contain reference tokens")
        if " ".join(self.r2v_instruction.split()) == " ".join(self.t2v_caption.split()):
            raise ValueError("r2v_instruction must differ from t2v_caption")
        if not self.references:
            raise ValueError("dataset sample requires at least one reference")
        if not any(reference.type == "entity" for reference in self.references):
            raise ValueError("dataset sample requires at least one entity reference")
        tokens = [reference.token for reference in self.references]
        if len(tokens) != len(set(tokens)):
            raise ValueError("dataset reference tokens must be unique")
        if _ANY_REF_TOKEN.search(self.r2v_instruction):
            raise ValueError(
                "r2v_instruction must not expose internal reference tokens"
            )
        if _ANY_TEMPLATE_PLACEHOLDER.search(self.r2v_instruction):
            raise ValueError("r2v_instruction must not expose raw image placeholders")
        expected_indexes = {str(index) for index in range(1, len(self.references) + 1)}
        angle_english_indexes = set(
            _ANGLE_ENGLISH_IMAGE_INDEX.findall(self.r2v_instruction)
        )
        plain_english_indexes = set(
            _PLAIN_ENGLISH_IMAGE_INDEX.findall(self.r2v_instruction)
        )
        legacy_indexes = set(_CHINESE_IMAGE_INDEX.findall(self.r2v_instruction))
        if not (
            (
                angle_english_indexes == expected_indexes
                and not plain_english_indexes
                and not legacy_indexes
            )
            or (
                plain_english_indexes == expected_indexes
                and not angle_english_indexes
                and not legacy_indexes
            )
            or (
                legacy_indexes == expected_indexes
                and not angle_english_indexes
                and not plain_english_indexes
            )
        ):
            raise ValueError(
                "r2v_instruction image labels must match dataset references"
            )
        return self


class DatasetRecord(SchemaModel):
    schema_version: Literal["r2v.v3.dataset.1"] = DATASET_SCHEMA_VERSION
    dataset_version: str
    created_at: str
    git_commit: str
    config_hash: str
    annotation_model: str
    background_remove_backend: str
    sample_count: int = Field(ge=0)
    reference_count: int = Field(ge=0)
