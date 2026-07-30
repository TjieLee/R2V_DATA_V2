from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QwenAnnotationEntity(SchemaModel):
    entity_id: str
    phrase: str
    grounding_prompt: str
    canonical_label: str
    category: Literal["person", "animal", "character", "object", "product", "vehicle"]
    reference_worthy: bool
    salience: Literal["primary", "secondary", "incidental"]
    genericity: Literal["named", "descriptive", "generic"]
    name_evidence: Literal["none", "draft_caption", "metadata", "visible_text"]
    separability: Literal[
        "independent",
        "attached_accessory",
        "important_independent_object",
        "composite_candidate",
    ]
    selection_reason: str


class AnnotationEntity(QwenAnnotationEntity):
    ref_token: Optional[str] = None


class EntityRelation(SchemaModel):
    subject_id: str
    predicate: str
    object_id: str


class QwenBackgroundAnnotation(SchemaModel):
    phrase: str
    grounding_prompt: str
    reference_worthy: bool


class BackgroundAnnotation(QwenBackgroundAnnotation):
    ref_token: Optional[str] = None


class QwenAnnotationResult(SchemaModel):
    caption: str
    entities: list[QwenAnnotationEntity]
    relations: list[EntityRelation] = Field(default_factory=list)
    background: Optional[QwenBackgroundAnnotation] = None


class AnnotationResult(QwenAnnotationResult):
    prompt_with_refs: str
    entities: list[AnnotationEntity]
    background: Optional[BackgroundAnnotation] = None


class CandidateVisualReview(SchemaModel):
    frame_slot: int
    completeness: float = Field(ge=0.0, le=1.0)
    recognizability: float = Field(ge=0.0, le=1.0)
    occlusion: float = Field(ge=0.0, le=1.0)
    mask_quality: float = Field(ge=0.0, le=1.0)
    visual_quality: float = Field(ge=0.0, le=1.0)
    identity_features_visible: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    viewpoint: Literal[
        "front",
        "front_three_quarter",
        "side",
        "back",
        "unclear",
        "not_applicable",
    ] = "unclear"
    canonical_view_score: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateJudgeResult(SchemaModel):
    entity_id: str
    candidates: list[CandidateVisualReview]
    best_frame_slot: int


class BackgroundVisualReview(SchemaModel):
    frame_slot: int
    scene_completeness: float = Field(ge=0.0, le=1.0)
    scene_recognizability: float = Field(ge=0.0, le=1.0)
    foreground_distraction: float = Field(ge=0.0, le=1.0)
    visual_quality: float = Field(ge=0.0, le=1.0)
    reusable_as_background: bool
    rejection_reasons: list[str] = Field(default_factory=list)


class BackgroundJudgeResult(SchemaModel):
    candidates: list[BackgroundVisualReview]
    best_frame_slot: int


class CrossPairJudgeResult(SchemaModel):
    same_exact_instance: Literal["yes", "no", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    context_difference: Literal["small", "moderate", "large"]
    near_duplicate: bool
    conflicting_attributes: list[str] = Field(default_factory=list)
    reason: str


class InpaintingSemanticReview(SchemaModel):
    same_semantic_content: bool
    identity_preserved: bool
    reference_phrase_supported: bool
    new_salient_objects: bool
    reason: str


class BackgroundFillPrompt(SchemaModel):
    fill_prompt: str
    visible_background_elements: list[str]
    reason: str


class MaskedContentOutcome(str, Enum):
    REMOVED_TO_BACKGROUND = "removed_to_background"
    FOREGROUND_REMAINS = "foreground_remains"
    FOREGROUND_RECONSTRUCTED = "foreground_reconstructed"
    REPLACED_BY_NEW_OBJECT = "replaced_by_new_object"
    UNCERTAIN = "uncertain"


class BackgroundArtifactType(str, Enum):
    VISIBLE_SEAM = "visible_seam"
    GHOSTING = "ghosting"
    DOUBLE_EXPOSURE = "double_exposure"
    INSET_IMAGE = "inset_image"
    TEXTURE_DISCONTINUITY = "texture_discontinuity"
    ARTIFICIAL_BLOB = "artificial_blob"
    COLOR_OR_EXPOSURE_MISMATCH = "color_or_exposure_mismatch"


class BackgroundInpaintingReview(SchemaModel):
    masked_content_outcome: MaskedContentOutcome
    background_continuity_preserved: bool
    reference_phrase_supported: bool
    artifact_types: list[BackgroundArtifactType]
    reason: str


class ForegroundRemovalReview(SchemaModel):
    original_foreground_still_visible: bool
    original_foreground_reconstructed: bool
    new_salient_entity_visible: bool
    visible_entities: list[str]
    background_only_inside_mask: bool
    uncertain: bool
    reason: str


class BackgroundContinuityReview(SchemaModel):
    background_continuity_preserved: bool
    visible_seam: bool
    ghosting: bool
    double_exposure: bool
    artificial_blob: bool
    texture_discontinuity: bool
    color_or_exposure_mismatch: bool
    uncertain: bool
    reason: str


class FullSceneReview(SchemaModel):
    reference_phrase_supported: bool
    global_scene_consistent: bool
    reason: str
