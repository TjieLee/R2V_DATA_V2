from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotationEntity(SchemaModel):
    entity_id: str
    phrase: str
    grounding_prompt: str
    canonical_label: str
    category: Literal["person", "animal", "character", "object", "product", "vehicle"]
    ref_token: Optional[str] = None
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


class EntityRelation(SchemaModel):
    subject_id: str
    predicate: str
    object_id: str


class BackgroundAnnotation(SchemaModel):
    phrase: str
    grounding_prompt: str
    ref_token: Optional[str] = None
    reference_worthy: bool


class AnnotationResult(SchemaModel):
    caption: str
    prompt_with_refs: str
    entities: list[AnnotationEntity]
    relations: list[EntityRelation] = Field(default_factory=list)
    background: Optional[BackgroundAnnotation] = None
