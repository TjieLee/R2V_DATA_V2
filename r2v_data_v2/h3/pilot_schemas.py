from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.schemas import ASDModelProvenance, SchemaModel

_FACE_TRACK_ID = re.compile(r"face_[1-9]\d*")


class LRASDNativeSample(SchemaModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    raw_class1_logit: float
    backend_native_active: bool

    @model_validator(mode="after")
    def validate_sample(self) -> LRASDNativeSample:
        if not math.isfinite(self.raw_class1_logit):
            raise ValueError("LR-ASD native logit must be finite")
        if self.backend_native_active != (self.raw_class1_logit >= 0):
            raise ValueError("LR-ASD native active decision must preserve score >= 0")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("LR-ASD face bbox values must be finite")
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("LR-ASD face bbox must have positive extent")
        return self


class LRASDNativeTrack(SchemaModel):
    face_track_id: str
    samples: list[LRASDNativeSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_track(self) -> LRASDNativeTrack:
        if _FACE_TRACK_ID.fullmatch(self.face_track_id) is None:
            raise ValueError("LR-ASD face_track_id must use face_N")
        order = [
            (sample.frame_index, sample.timestamp_seconds) for sample in self.samples
        ]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("LR-ASD samples must use unique deterministic order")
        return self


class LRASDNativeArtifact(SchemaModel):
    schema_version: Literal["r2v.h3.lr_asd_native.1"] = "r2v.h3.lr_asd_native.1"
    clip_uid: str
    source_video_path: str
    model_video_path: str
    audio_path: str
    official_visualization_path: str | None = None
    model_fps: Literal[25.0] = 25.0
    audio_sample_rate_hz: Literal[16000] = 16000
    audio_channels: Literal[1] = 1
    face_detector: Literal["S3FD"] = "S3FD"
    face_tracking: Literal["shot_aware_iou"] = "shot_aware_iou"
    face_crop_preprocessing: Literal["official_lr_asd"] = "official_lr_asd"
    score_semantics: Literal["lr_asd_native_class_1_logit"] = (
        "lr_asd_native_class_1_logit"
    )
    active_decision_rule: Literal["score_greater_than_or_equal_to_zero"] = (
        "score_greater_than_or_equal_to_zero"
    )
    model_provenance: ASDModelProvenance
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    tracks: list[LRASDNativeTrack] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> LRASDNativeArtifact:
        if not self.clip_uid.strip():
            raise ValueError("LR-ASD clip_uid must not be empty")
        required_paths = (
            self.source_video_path,
            self.model_video_path,
            self.audio_path,
        )
        if any(not path.strip() for path in required_paths):
            raise ValueError("LR-ASD source, model-video, and audio paths are required")
        track_ids = [track.face_track_id for track in self.tracks]
        expected_track_ids = [f"face_{index}" for index in range(1, len(track_ids) + 1)]
        if track_ids != expected_track_ids:
            raise ValueError("LR-ASD face track IDs must use deterministic face_1..N order")
        for track in self.tracks:
            for sample in track.samples:
                if sample.timestamp_seconds >= self.duration_seconds:
                    raise ValueError("LR-ASD sample exceeds model-video duration")
                if not math.isclose(
                    sample.timestamp_seconds,
                    sample.frame_index / self.model_fps,
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("LR-ASD timestamp must match its 25-FPS frame index")
                _, _, x2, y2 = sample.bbox_xyxy
                if x2 > self.width or y2 > self.height:
                    raise ValueError("LR-ASD face bbox exceeds model-video dimensions")
        return self


class SpeechActivityInterval(SchemaModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> SpeechActivityInterval:
        if self.end_time <= self.start_time:
            raise ValueError("speech interval end must exceed start")
        return self


class SpeechActivityArtifact(SchemaModel):
    schema_version: Literal["r2v.h3.speech_activity.1"] = (
        "r2v.h3.speech_activity.1"
    )
    clip_uid: str
    backend: str
    model_identifier: str
    source_audio_path: str
    duration_seconds: float = Field(gt=0)
    intervals: list[SpeechActivityInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> SpeechActivityArtifact:
        if not self.clip_uid.strip():
            raise ValueError("speech activity clip_uid must not be empty")
        if not self.backend.strip() or not self.model_identifier.strip():
            raise ValueError("speech backend provenance must not be empty")
        if not self.source_audio_path.strip():
            raise ValueError("speech source audio path must not be empty")
        ordered = sorted(
            self.intervals,
            key=lambda interval: (interval.start_time, interval.end_time),
        )
        if self.intervals != ordered:
            raise ValueError("speech intervals must be deterministically ordered")
        if any(
            ordered[index].end_time > ordered[index + 1].start_time
            for index in range(len(ordered) - 1)
        ):
            raise ValueError("speech intervals must not overlap")
        if any(interval.end_time > self.duration_seconds for interval in ordered):
            raise ValueError("speech interval exceeds audio duration")
        return self


class H3AudioBindingPilotSummary(SchemaModel):
    schema_version: Literal["r2v.h3.lr_asd_pilot.1"] = "r2v.h3.lr_asd_pilot.1"
    source_run_root: str
    output_root: str
    clips_attempted: int = Field(ge=0)
    clips_succeeded: int = Field(ge=0)
    clips_failed: int = Field(ge=0)
    clips_with_speech: int = Field(ge=0)
    bound_intervals: int = Field(ge=0)
    overlap_intervals: int = Field(ge=0)
    offscreen_intervals: int = Field(ge=0)
    ambiguous_intervals: int = Field(ge=0)
    no_speech_intervals: int = Field(ge=0)
    face_entity_association_failures: int = Field(ge=0)
    asd_runtime_failures: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> H3AudioBindingPilotSummary:
        if not self.source_run_root.strip() or not self.output_root.strip():
            raise ValueError("pilot summary roots must not be empty")
        if self.clips_attempted != self.clips_succeeded + self.clips_failed:
            raise ValueError("pilot attempted clip count must reconcile")
        return self
