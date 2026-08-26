from __future__ import annotations

from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.schemas import SchemaModel


class SpeakerTimeRange(SchemaModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> SpeakerTimeRange:
        if self.end_time <= self.start_time:
            raise ValueError("speaker time range must be positive")
        return self


class SpeakerClusterEvidence(SchemaModel):
    speaker_cluster_id: str
    entity_id: str | None = None
    active_time_ranges: list[SpeakerTimeRange]

    @model_validator(mode="after")
    def validate_cluster(self) -> SpeakerClusterEvidence:
        if not self.speaker_cluster_id.strip() or not self.active_time_ranges:
            raise ValueError("speaker cluster evidence must be complete")
        if self.entity_id is not None and not self.entity_id.strip():
            raise ValueError("speaker entity ID must be non-empty or null")
        if self.active_time_ranges != sorted(
            self.active_time_ranges,
            key=lambda item: (item.start_time, item.end_time),
        ):
            raise ValueError("speaker time ranges must be chronological")
        return self


class ModelSpeakerDelivery(SchemaModel):
    speaker_cluster_id: StrictStr
    delivery_style: StrictStr | None = None

    @model_validator(mode="after")
    def validate_delivery(self) -> ModelSpeakerDelivery:
        if not self.speaker_cluster_id.strip():
            raise ValueError("speaker cluster ID must not be empty")
        if self.delivery_style is not None and not self.delivery_style.strip():
            raise ValueError("speaker delivery style must be non-empty or null")
        return self


class TargetAudioCaptionResponse(SchemaModel):
    background_audio_prompt: StrictStr | None = None
    speaker_delivery: list[ModelSpeakerDelivery]

    @model_validator(mode="after")
    def validate_response(self) -> TargetAudioCaptionResponse:
        if (
            self.background_audio_prompt is not None
            and not self.background_audio_prompt.strip()
        ):
            raise ValueError("background audio prompt must be non-empty or null")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("speaker delivery cluster IDs must be unique")
        return self


class TargetSpeakerDelivery(ModelSpeakerDelivery):
    entity_id: str | None = None


__all__ = [
    "ModelSpeakerDelivery",
    "SpeakerClusterEvidence",
    "SpeakerTimeRange",
    "TargetAudioCaptionResponse",
    "TargetSpeakerDelivery",
]
