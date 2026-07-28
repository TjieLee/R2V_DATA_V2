from __future__ import annotations

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


class CandidateJudgeResult(SchemaModel):
    entity_id: str
    candidates: list[CandidateVisualReview]
    best_frame_slot: int


class CrossPairJudgeResult(SchemaModel):
    same_exact_instance: Literal["yes", "no", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    context_difference: Literal["small", "moderate", "large"]
    near_duplicate: bool
    conflicting_attributes: list[str] = Field(default_factory=list)
    reason: str
