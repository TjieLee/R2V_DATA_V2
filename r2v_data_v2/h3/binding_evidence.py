"""Explicit optional speaker-binding priors, preserving legacy serialization."""

from typing import Literal

from pydantic import model_serializer

from r2v_data_v2.h3.schemas import SchemaModel

BindingEvidenceMode = Literal["legacy_lr_asd", "none"]


class BindingEvidenceSource(SchemaModel):
    binding_evidence_mode: BindingEvidenceMode = "legacy_lr_asd"

    @model_serializer(mode="wrap")
    def serialize_binding_mode(self, handler):
        values = handler(self)
        if self.binding_evidence_mode == "legacy_lr_asd":
            values.pop("binding_evidence_mode", None)
        return values
