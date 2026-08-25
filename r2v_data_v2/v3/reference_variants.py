from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from r2v_data_v2.v3.schemas import (
    ReferenceDefaultVariant,
    ReferenceVariantsState,
)

REFERENCE_VARIANTS_SCHEMA_VERSION = "r2v.v3.reference_variants.1"


class ReferenceVariantsManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "r2v.v3.reference_variants.1"
    ] = REFERENCE_VARIANTS_SCHEMA_VERSION
    sample_id: str
    clip_uid: str
    kind: Literal["subject", "object", "attribute"]
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None
    default_variant: ReferenceDefaultVariant
    default_image_path: str
    default_reason: str
    accepted_base_image_path: str | None = None
    variants: ReferenceVariantsState

    @model_validator(mode="after")
    def validate_binding(self) -> ReferenceVariantsManifestRecord:
        if not self.sample_id.strip() or self.clip_uid != self.sample_id:
            raise ValueError("reference variants require matching sample and clip IDs")
        if not self.default_image_path.strip() or not self.default_reason.strip():
            raise ValueError("reference variants require default provenance")
        if self.kind == "attribute":
            if (
                self.entity_id is not None
                or self.attribute_id is None
                or self.owner_entity_id is None
                or self.attribute_type is None
                or self.accepted_base_image_path is None
            ):
                raise ValueError("attribute variants require owner-bound provenance")
        elif (
            self.entity_id is None
            or self.attribute_id is not None
            or self.owner_entity_id is not None
            or self.attribute_type is not None
        ):
            raise ValueError("entity variants require entity-only provenance")
        elif self.default_variant == "accepted_base" and (
            self.accepted_base_image_path is None
        ):
            raise ValueError("completed entity variants require accepted base provenance")
        if self.default_variant != "accepted_base":
            selected = getattr(self.variants, self.default_variant)
            if (
                selected.status != "accepted"
                or selected.image_path != self.default_image_path
            ):
                raise ValueError("manifest default must match an accepted variant")
        elif self.default_image_path != self.accepted_base_image_path:
            raise ValueError("accepted-base manifest default must match its image path")
        return self


__all__ = [
    "REFERENCE_VARIANTS_SCHEMA_VERSION",
    "ReferenceVariantsManifestRecord",
]
