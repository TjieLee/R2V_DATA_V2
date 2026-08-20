from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from r2v_data_v2.v3.schemas import SchemaModel, VisibleRegion
from r2v_data_v2.v3.subject_attributes import AttributeType

PRODUCTION_SAMPLE_SCHEMA_VERSION = "r2v.v3.production_sample.1"

ProductionReferenceKind = Literal[
    "subject",
    "object",
    "group",
    "background",
    "attribute",
]
_IMAGE_LABEL = re.compile(
    r"<Image\s+([1-9]\d*)>|(?<!<)\bImage\s+([1-9]\d*)\b(?!>)|图\s*(\d+)",
    flags=re.IGNORECASE,
)


class ProductionReference(SchemaModel):
    image_id: str = Field(pattern=r"^image_[1-9]\d*$")
    image_index: int = Field(ge=1)
    kind: ProductionReferenceKind
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: AttributeType | None = None
    image_path: str
    source_frame_index: int = Field(ge=0)
    scope: Literal["full", "local", "scene"] | None = None
    visible_region: VisibleRegion | None = None
    synthetic: bool

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, value: str) -> str:
        parts = value.split("/")
        if not value or value.startswith("/") or ".." in parts:
            raise ValueError("production reference image_path must be relative")
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> ProductionReference:
        if self.image_id != f"image_{self.image_index}":
            raise ValueError("production reference image_id must match image_index")
        attribute_fields = (
            self.attribute_id,
            self.owner_entity_id,
            self.attribute_type,
        )
        if self.kind == "attribute":
            if self.entity_id is not None or any(
                value is None for value in attribute_fields
            ):
                raise ValueError("attribute reference requires owner-bound provenance")
            if self.scope is not None or self.visible_region is not None:
                raise ValueError("attribute reference cannot publish Visual geometry")
        elif any(value is not None for value in attribute_fields):
            raise ValueError("Visual reference cannot publish attribute provenance")
        elif self.kind == "background":
            if self.entity_id is not None or self.scope != "scene":
                raise ValueError("background reference requires scene scope")
        elif self.entity_id is None or self.scope not in {"full", "local"}:
            raise ValueError("entity reference requires entity_id and entity scope")
        return self


class ProductionSampleSource(SchemaModel):
    parent_video_id: str
    clip_suffix: str
    shard_id: str


class ProductionSample(SchemaModel):
    schema_version: Literal[
        "r2v.v3.production_sample.1"
    ] = PRODUCTION_SAMPLE_SCHEMA_VERSION
    sample_id: str
    clip_uid: str
    target_video: str
    t2v_caption: str
    r2v_instruction: str
    references: list[ProductionReference]
    source: ProductionSampleSource

    @model_validator(mode="after")
    def validate_references(self) -> ProductionSample:
        indexes = [reference.image_index for reference in self.references]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("production reference indexes must be contiguous")
        instruction_indexes = [
            int(next(group for group in match.groups() if group is not None))
            for match in _IMAGE_LABEL.finditer(self.r2v_instruction)
        ]
        if instruction_indexes != indexes:
            raise ValueError(
                "production instruction image ordering must match references"
            )
        return self
