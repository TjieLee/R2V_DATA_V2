from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, StrictBool, field_validator, model_validator

from r2v_data_v2.h3.schemas import SchemaModel


class _VisualClipProjectionModel(SchemaModel):
    """H3-owned clip fields with Visual-internal sections ignored."""

    model_config = ConfigDict(extra="ignore")


class VisualClipSourceMetadata(_VisualClipProjectionModel):
    source_relative_video_path: str
    source_relative_source_video_path: str

    @field_validator(
        "source_relative_video_path",
        "source_relative_source_video_path",
    )
    @classmethod
    def validate_relative_source_path(cls, value: str) -> str:
        if not value.strip() or "\\" in value:
            raise ValueError("Visual source path must be a non-empty POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("Visual source path must remain relative and contained")
        return value


class VisualClipSource(_VisualClipProjectionModel):
    video_path: str
    metadata: VisualClipSourceMetadata

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Visual source video_path must not be empty")
        return value


class VisualClipAnnotationEntity(_VisualClipProjectionModel):
    entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    reference_type: Literal["subject", "object", "group"]
    phrase: str

    @field_validator("phrase")
    @classmethod
    def validate_phrase(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Visual annotation entity phrase must not be empty")
        return value


class VisualClipAnnotation(_VisualClipProjectionModel):
    status: Literal["ready", "failed"]
    entities: list[VisualClipAnnotationEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_entity_ids(self) -> VisualClipAnnotation:
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Visual annotation entity IDs must be unique")
        return self


class VisualClipCoverage(_VisualClipProjectionModel):
    passed: StrictBool


class VisualClipEntityReference(_VisualClipProjectionModel):
    entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    status: Literal["ready", "rejected"]
    image_path: str | None = None

    @model_validator(mode="after")
    def validate_ready_image(self) -> VisualClipEntityReference:
        if self.status == "ready" and (
            self.image_path is None or not self.image_path.strip()
        ):
            raise ValueError("ready Visual entity reference requires image_path")
        return self


class VisualClipReferences(_VisualClipProjectionModel):
    entities: list[VisualClipEntityReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_entity_ids(self) -> VisualClipReferences:
        entity_ids = [reference.entity_id for reference in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Visual entity reference IDs must be unique")
        return self


class VisualClipPairing(_VisualClipProjectionModel):
    status: Literal["ready", "rejected"]
    retained_entity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_retained_ids(self) -> VisualClipPairing:
        if len(self.retained_entity_ids) != len(set(self.retained_entity_ids)):
            raise ValueError("Visual pairing retained entity IDs must be unique")
        if self.status == "ready" and not self.retained_entity_ids:
            raise ValueError("ready Visual pairing requires a retained entity")
        if self.status == "rejected" and self.retained_entity_ids:
            raise ValueError("rejected Visual pairing cannot retain entities")
        return self


class VisualClipRecord(_VisualClipProjectionModel):
    """Stable Audio/H3 projection of a Visual V3 clip record."""

    schema_version: Literal["r2v.v3.clip.2"]
    clip_uid: str
    source: VisualClipSource
    annotation: VisualClipAnnotation | None = None
    coverage: VisualClipCoverage | None = None
    references: VisualClipReferences = Field(default_factory=VisualClipReferences)
    pairing: VisualClipPairing | None = None

    @field_validator("clip_uid")
    @classmethod
    def validate_clip_uid(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Visual clip_uid must not be empty")
        return value

    @model_validator(mode="after")
    def validate_entity_bindings(self) -> VisualClipRecord:
        annotation_order = (
            [entity.entity_id for entity in self.annotation.entities]
            if self.annotation is not None
            else []
        )
        annotation_ids = set(annotation_order)
        reference_order = [reference.entity_id for reference in self.references.entities]
        reference_ids = set(reference_order)
        if reference_ids - annotation_ids:
            raise ValueError("Visual references must correspond to annotation entities")
        expected_reference_order = [
            entity_id for entity_id in annotation_order if entity_id in reference_ids
        ]
        if reference_order != expected_reference_order:
            raise ValueError("Visual references must follow annotation entity order")

        if self.pairing is None or self.pairing.status != "ready":
            return self
        if self.annotation is None or self.annotation.status != "ready":
            raise ValueError("ready Visual pairing requires ready annotation")
        if self.coverage is None or not self.coverage.passed:
            raise ValueError("ready Visual pairing requires passed coverage")
        retained = self.pairing.retained_entity_ids
        retained_ids = set(retained)
        if retained_ids - annotation_ids:
            raise ValueError("Visual pairing retains unknown annotation entities")
        expected_retained_order = [
            entity_id for entity_id in annotation_order if entity_id in retained_ids
        ]
        if retained != expected_retained_order:
            raise ValueError("Visual pairing retained IDs must follow annotation order")
        ready_reference_order = [
            reference.entity_id
            for reference in self.references.entities
            if reference.status == "ready"
        ]
        if retained != ready_reference_order:
            raise ValueError("ready Visual pairing requires matching ready references")
        return self


def load_visual_clip_record(
    path: Path,
    *,
    expected_clip_uid: str | None = None,
) -> VisualClipRecord:
    clip = VisualClipRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if expected_clip_uid is not None and clip.clip_uid != expected_clip_uid:
        raise ValueError("Visual clip record does not match expected clip_uid")
    return clip
