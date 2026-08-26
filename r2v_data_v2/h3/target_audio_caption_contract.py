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


class LegacyTargetAudioCaptionResponse(SchemaModel):
    background_audio_prompt: StrictStr | None = None
    speaker_delivery: list[ModelSpeakerDelivery]

    @model_validator(mode="after")
    def validate_response(self) -> LegacyTargetAudioCaptionResponse:
        if (
            self.background_audio_prompt is not None
            and not self.background_audio_prompt.strip()
        ):
            raise ValueError("background audio prompt must be non-empty or null")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("speaker delivery cluster IDs must be unique")
        return self


class TemporalAudioEvent(SchemaModel):
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    description: StrictStr

    @model_validator(mode="after")
    def validate_event(self) -> TemporalAudioEvent:
        if self.end_time <= self.start_time:
            raise ValueError("temporal audio event range must be positive")
        if not self.description.strip():
            raise ValueError("temporal audio event description must not be empty")
        return self


class TargetAudioCaptionResponse(SchemaModel):
    overall_soundscape: StrictStr | None = None
    non_diegetic_music: StrictStr | None = None
    temporal_audio_events: list[TemporalAudioEvent] = Field(default_factory=list)
    speaker_delivery: list[ModelSpeakerDelivery]

    @model_validator(mode="after")
    def validate_response(self) -> TargetAudioCaptionResponse:
        for field_name in ("overall_soundscape", "non_diegetic_music"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty or null")
        if self.temporal_audio_events != sorted(
            self.temporal_audio_events,
            key=lambda item: (item.start_time, item.end_time),
        ):
            raise ValueError("temporal audio events must be chronological")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("speaker delivery cluster IDs must be unique")
        return self


class TargetSpeakerDelivery(ModelSpeakerDelivery):
    entity_id: str | None = None


__all__ = [
    "LegacyTargetAudioCaptionResponse",
    "ModelSpeakerDelivery",
    "SpeakerClusterEvidence",
    "SpeakerTimeRange",
    "TargetAudioCaptionResponse",
    "TargetSpeakerDelivery",
    "TemporalAudioEvent",
]
