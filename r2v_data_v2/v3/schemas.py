from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLIP_SCHEMA_VERSION = "r2v.v3.clip.1"
MASK_SCHEMA_VERSION = "r2v.v3.masks.1"
RUN_SCHEMA_VERSION = "r2v.v3.run.1"
DATASET_SCHEMA_VERSION = "r2v.v3.dataset.1"
SAMPLE_SCHEMA_VERSION = "r2v.v3.sample.1"

_REF_TOKEN = re.compile(r"<ref_(?:subject|object|group|bg)_\d+>")
_ANY_REF_TOKEN = re.compile(r"<ref_[^>]+>")


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipSource(SchemaModel):
    video_path: str
    parent_video_id: str
    clip_suffix: str


class AnnotationEntity(SchemaModel):
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


class EntityRelation(SchemaModel):
    subject_id: str
    predicate: str
    object_id: str


class BackgroundAnnotation(SchemaModel):
    phrase: str
    grounding_prompt: str
    reference_worthy: bool


class AnnotationState(SchemaModel):
    status: Literal["ready", "failed"]
    t2v_caption: str = ""
    entities: list[AnnotationEntity] = Field(default_factory=list)
    relations: list[EntityRelation] = Field(default_factory=list)
    background: Optional[BackgroundAnnotation] = None
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> AnnotationState:
        if _ANY_REF_TOKEN.search(self.t2v_caption):
            raise ValueError("t2v_caption must not contain reference tokens")
        if self.status == "ready" and not self.t2v_caption.strip():
            raise ValueError("ready annotation requires a non-empty t2v_caption")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed annotation requires a reason")
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("annotation entity_id values must be unique")
        known_ids = set(entity_ids)
        if any(
            relation.subject_id not in known_ids
            or relation.object_id not in known_ids
            for relation in self.relations
        ):
            raise ValueError("annotation relations must reference known entities")
        return self


class CoverageState(SchemaModel):
    passed: bool
    qualifying_entity_ids: list[str] = Field(default_factory=list)
    required_visible_frames: Literal[8] = 8
    entity_visibility_summary: dict[str, dict[str, object]] = Field(
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
        return self


ReferenceScope = Literal["full", "local", "reject"]
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
    synthetic: Literal[False] = False

    @model_validator(mode="after")
    def validate_reference_state(self) -> EntityReferenceState:
        if self.status == "ready":
            if self.reference_scope == "reject":
                raise ValueError("ready entity reference cannot use reject scope")
            if self.image_path is None or self.source_frame_index is None:
                raise ValueError(
                    "ready entity reference requires image_path and source_frame_index"
                )
        if self.reference_scope == "reject" and self.status != "rejected":
            raise ValueError("reject scope requires rejected status")
        if self.status == "rejected" and self.image_path is not None:
            raise ValueError("rejected entity reference cannot publish an image_path")
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
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_reference_state(self) -> BackgroundReferenceState:
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
        if self.status == "pending_remove":
            if self.source_mask_path is None:
                raise ValueError("pending_remove background requires source_mask_path")
            if self.output_image_path is not None:
                raise ValueError(
                    "pending_remove background cannot publish output_image_path"
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
        if self.status == "rejected":
            if not self.reason:
                raise ValueError("rejected background requires a reason")
            if self.output_image_path is not None:
                raise ValueError(
                    "rejected background cannot publish output_image_path"
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
            if not self.reason:
                raise ValueError("rejected pairing requires a reason")
            if (
                self.retained_entity_ids
                or self.tokens
                or self.background_token is not None
            ):
                raise ValueError(
                    "rejected pairing must clear retained IDs and tokens"
                )
            return self
        if not self.retained_entity_ids:
            raise ValueError("ready pairing requires at least one retained entity")
        if set(self.tokens) != set(self.retained_entity_ids):
            raise ValueError("ready pairing tokens must match retained_entity_ids")
        if len(self.retained_entity_ids) != len(set(self.retained_entity_ids)):
            raise ValueError("retained_entity_ids must be unique")
        all_tokens = [*self.tokens.values()]
        if self.background_token is not None:
            all_tokens.append(self.background_token)
        if len(all_tokens) != len(set(all_tokens)):
            raise ValueError("pairing tokens must be unique")
        if any(_REF_TOKEN.fullmatch(token) is None for token in all_tokens):
            raise ValueError("pairing contains an invalid reference token")
        if any(token.startswith("<ref_bg_") for token in self.tokens.values()):
            raise ValueError("entity tokens cannot use background reference tokens")
        if self.background_token is not None and not self.background_token.startswith(
            "<ref_bg_"
        ):
            raise ValueError("background_token must use a background reference token")
        return self


class InstructionState(SchemaModel):
    status: Literal["ready", "failed"]
    r2v_instruction: str = ""
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> InstructionState:
        if self.status == "ready" and not self.r2v_instruction.strip():
            raise ValueError("ready instruction requires non-empty text")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed instruction requires a reason")
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
    schema_version: Literal["r2v.v3.clip.1"] = CLIP_SCHEMA_VERSION
    clip_uid: str
    source: ClipSource
    annotation: Optional[AnnotationState] = None
    coverage: Optional[CoverageState] = None
    references: ReferencesState = Field(default_factory=ReferencesState)
    pairing: Optional[PairingState] = None
    instruction: Optional[InstructionState] = None
    export: ExportState = Field(default_factory=ExportState)

    @model_validator(mode="after")
    def validate_section_consistency(self) -> ClipRecord:
        annotation_ids = (
            {entity.entity_id for entity in self.annotation.entities}
            if self.annotation is not None
            else set()
        )
        if self.coverage is not None:
            unknown_qualifying = (
                set(self.coverage.qualifying_entity_ids) - annotation_ids
            )
            if unknown_qualifying:
                raise ValueError(
                    "coverage qualifying entity IDs must exist in annotation"
                )
        reference_ids = {
            reference.entity_id for reference in self.references.entities
        }
        if reference_ids - annotation_ids:
            raise ValueError(
                "entity references must correspond to annotation entities"
            )
        if self.pairing is not None and self.pairing.status == "ready":
            ready_reference_ids = {
                reference.entity_id
                for reference in self.references.entities
                if reference.status == "ready"
            }
            retained_ids = set(self.pairing.retained_entity_ids)
            if not retained_ids:
                raise ValueError(
                    "ready pairing requires at least one retained entity"
                )
            if not retained_ids.issubset(ready_reference_ids):
                raise ValueError(
                    "ready pairing retained IDs must have ready references"
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
            if self.pairing.background_token is not None:
                background = self.references.background
                if background is None or background.status not in {
                    "clean_raw",
                    "ready_removed",
                }:
                    raise ValueError(
                        "ready pairing background token requires a ready background"
                    )
        if self.instruction is not None and self.instruction.status == "ready":
            if self.pairing is None or self.pairing.status != "ready":
                raise ValueError("ready instruction requires ready pairing")
            expected_tokens = set(self.pairing.tokens.values())
            if self.pairing.background_token is not None:
                expected_tokens.add(self.pairing.background_token)
            instruction_tokens = _ANY_REF_TOKEN.findall(
                self.instruction.r2v_instruction
            )
            if set(instruction_tokens) != expected_tokens:
                raise ValueError(
                    "ready instruction tokens must exactly match pairing tokens"
                )
            if len(instruction_tokens) != len(expected_tokens):
                raise ValueError(
                    "ready instruction must contain each pairing token exactly once"
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
        return self


class TrackedMasksArtifact(SchemaModel):
    schema_version: Literal["r2v.v3.masks.1"] = MASK_SCHEMA_VERSION
    clip_uid: str
    sampled_frame_count: Literal[10] = 10
    entities: dict[str, dict[str, object]] = Field(default_factory=dict)


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
            raise ValueError("dataset reference image_path must not escape dataset root")
        return value

    @model_validator(mode="after")
    def validate_type_fields(self) -> DatasetReference:
        if self.type == "entity":
            if self.entity_id is None:
                raise ValueError("entity reference requires entity_id")
            if self.token.startswith("<ref_bg_"):
                raise ValueError("entity reference cannot use a background token")
            if self.scope not in {"full", "local"}:
                raise ValueError("entity reference scope must be full or local")
            if self.synthetic:
                raise ValueError("V3 entity references must not be synthetic")
        else:
            if self.entity_id is not None or self.scope != "scene":
                raise ValueError("background reference must use scene scope and no entity_id")
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
        if " ".join(self.r2v_instruction.split()) == " ".join(
            self.t2v_caption.split()
        ):
            raise ValueError("r2v_instruction must differ from t2v_caption")
        if not self.references:
            raise ValueError("dataset sample requires at least one reference")
        if not any(reference.type == "entity" for reference in self.references):
            raise ValueError("dataset sample requires at least one entity reference")
        tokens = [reference.token for reference in self.references]
        if len(tokens) != len(set(tokens)):
            raise ValueError("dataset reference tokens must be unique")
        instruction_tokens = _ANY_REF_TOKEN.findall(self.r2v_instruction)
        if sorted(instruction_tokens) != sorted(tokens):
            raise ValueError(
                "r2v_instruction tokens must exactly match dataset references"
            )
        if any(self.r2v_instruction.count(token) != 1 for token in tokens):
            raise ValueError("each reference token must occur exactly once")
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
