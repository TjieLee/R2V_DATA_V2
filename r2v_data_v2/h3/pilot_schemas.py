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


class LaserASDNativeSample(SchemaModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float | None = Field(default=None, ge=0, le=1)
    raw_backend_score: float
    backend_native_active: bool
    landmark_available: bool

    @model_validator(mode="after")
    def validate_sample(self) -> LaserASDNativeSample:
        if not math.isfinite(self.raw_backend_score):
            raise ValueError("LASER native score must be finite")
        if self.backend_native_active != (self.raw_backend_score >= 0):
            raise ValueError("LASER native active decision must preserve score >= 0")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("LASER face bbox values must be finite")
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("LASER face bbox must have positive bounded extent")
        return self


class LaserASDNativeTrack(SchemaModel):
    face_track_id: str
    samples: list[LaserASDNativeSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_track(self) -> LaserASDNativeTrack:
        if _FACE_TRACK_ID.fullmatch(self.face_track_id) is None:
            raise ValueError("LASER face_track_id must use face_N")
        order = [
            (sample.frame_index, sample.timestamp_seconds) for sample in self.samples
        ]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("LASER samples must use unique deterministic order")
        return self


class LaserASDNativeArtifact(SchemaModel):
    schema_version: Literal["r2v.h3.laser_asd_native.2"] = (
        "r2v.h3.laser_asd_native.2"
    )
    clip_uid: str
    source_video_path: str
    model_video_path: str
    audio_path: str
    debug_visualization_path: str | None = None
    model_fps: Literal[25.0] = 25.0
    audio_sample_rate_hz: Literal[16000] = 16000
    audio_channels: Literal[1] = 1
    face_detector: Literal["S3FD"] = "S3FD"
    face_tracking: Literal["shot_aware_iou"] = "shot_aware_iou"
    face_crop_preprocessing: Literal["laser_loconet_official_face_crop"] = (
        "laser_loconet_official_face_crop"
    )
    landmark_backend: Literal["mediapipe_face_landmarker_82_lips"] = (
        "mediapipe_face_landmarker_82_lips"
    )
    score_semantics: Literal["laser_loconet_native_score"] = (
        "laser_loconet_native_score"
    )
    active_decision_rule: Literal["score_greater_than_or_equal_to_zero"] = (
        "score_greater_than_or_equal_to_zero"
    )
    upstream_repository: Literal["https://github.com/plnguyen2908/LASER_ASD"] = (
        "https://github.com/plnguyen2908/LASER_ASD"
    )
    upstream_commit: Literal["3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9"] = (
        "3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9"
    )
    model_provenance: ASDModelProvenance
    config_path: str
    config_sha256: str
    landmark_model_path: str
    landmark_model_sha256: str
    s3fd_model_path: str
    s3fd_model_sha256: str
    resolved_n_channel: int = Field(gt=0)
    resolved_layer: int = Field(ge=0)
    device: str = Field(pattern=r"^cuda:\d+$")
    cuda_visible_devices: str = Field(pattern=r"^\d+(,\d+)*$")
    mediapipe_version: str
    torch_version: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    landmark_sample_count: int = Field(ge=0)
    landmark_available_count: int = Field(ge=0)
    deterministic_context_selection: Literal["stable_first_last"] = (
        "stable_first_last"
    )
    tracks: list[LaserASDNativeTrack] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact(self) -> LaserASDNativeArtifact:
        if not self.clip_uid.strip():
            raise ValueError("LASER clip_uid must not be empty")
        required_paths = (
            self.source_video_path,
            self.model_video_path,
            self.audio_path,
            self.config_path,
            self.landmark_model_path,
            self.s3fd_model_path,
        )
        if any(not path.strip() for path in required_paths):
            raise ValueError("LASER runtime artifact paths are required")
        for digest in (
            self.config_sha256,
            self.landmark_model_sha256,
            self.s3fd_model_sha256,
        ):
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("LASER runtime hashes must be lowercase SHA-256")
        if not self.mediapipe_version.strip() or not self.torch_version.strip():
            raise ValueError("LASER runtime package versions are required")
        if int(self.device.split(":", 1)[1]) >= len(
            self.cuda_visible_devices.split(",")
        ):
            raise ValueError("LASER device is outside isolated CUDA visibility")
        track_ids = [track.face_track_id for track in self.tracks]
        expected = [f"face_{index}" for index in range(1, len(track_ids) + 1)]
        if track_ids != expected:
            raise ValueError("LASER face track IDs must use deterministic face_1..N")
        samples = [sample for track in self.tracks for sample in track.samples]
        if self.landmark_sample_count != len(samples):
            raise ValueError("LASER landmark sample count must reconcile")
        if self.landmark_available_count != sum(
            sample.landmark_available for sample in samples
        ):
            raise ValueError("LASER available-landmark count must reconcile")
        for sample in samples:
            if sample.timestamp_seconds >= self.duration_seconds:
                raise ValueError("LASER sample exceeds model-video duration")
            if not math.isclose(
                sample.timestamp_seconds,
                sample.frame_index / self.model_fps,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("LASER timestamp must match its 25-FPS frame index")
            _, _, x2, y2 = sample.bbox_xyxy
            if x2 > self.width or y2 > self.height:
                raise ValueError("LASER face bbox exceeds model-video dimensions")
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


class LaserASDPilotSummary(SchemaModel):
    schema_version: Literal["r2v.h3.laser_asd_pilot.1"] = (
        "r2v.h3.laser_asd_pilot.1"
    )
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
    voice_quality_evaluated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> LaserASDPilotSummary:
        if not self.source_run_root.strip() or not self.output_root.strip():
            raise ValueError("LASER pilot summary roots must not be empty")
        if self.clips_attempted != self.clips_succeeded + self.clips_failed:
            raise ValueError("LASER pilot attempted clip count must reconcile")
        return self


class LRASDScoreDiagnostics(SchemaModel):
    mean: float
    min: float
    p10: float


class AssociationConfidenceDiagnostics(SchemaModel):
    mean: float = Field(ge=0, le=1)
    min: float = Field(ge=0, le=1)


class VoiceReferenceTurnDiagnostics(SchemaModel):
    clip_uid: str
    turn_id: str
    entity_id: str
    face_track_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sample_count: int = Field(gt=0)
    rms_amplitude: float = Field(ge=0, le=1)
    rms_dbfs: float | None = None
    peak_amplitude: float = Field(ge=0, le=1)
    peak_dbfs: float | None = None
    clipping_ratio: float = Field(ge=0, le=1)
    local_noise_context_available: bool
    local_noise_sample_count: int = Field(ge=0)
    local_noise_duration_seconds: float = Field(ge=0)
    local_noise_rms_amplitude: float | None = Field(default=None, ge=0, le=1)
    local_noise_rms_dbfs: float | None = None
    estimated_snr_db: float | None = None
    lr_asd_raw_native_score: LRASDScoreDiagnostics
    association_confidence: AssociationConfidenceDiagnostics
    voice_reference_eligible: Literal[True] = True

    @model_validator(mode="after")
    def validate_turn(self) -> VoiceReferenceTurnDiagnostics:
        if not self.clip_uid.strip() or not self.turn_id.strip():
            raise ValueError("voice-quality turn IDs must not be empty")
        if not self.entity_id.strip() or not self.face_track_id.strip():
            raise ValueError("voice-quality entity and face IDs must not be empty")
        if self.end_time <= self.start_time or not math.isclose(
            self.duration_seconds,
            self.end_time - self.start_time,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("voice-quality turn duration must match its extent")
        numeric = (
            self.rms_amplitude,
            self.peak_amplitude,
            self.clipping_ratio,
            self.lr_asd_raw_native_score.mean,
            self.lr_asd_raw_native_score.min,
            self.lr_asd_raw_native_score.p10,
            self.association_confidence.mean,
            self.association_confidence.min,
        )
        optional = (
            self.rms_dbfs,
            self.peak_dbfs,
            self.local_noise_rms_amplitude,
            self.local_noise_rms_dbfs,
            self.estimated_snr_db,
        )
        if not all(math.isfinite(value) for value in numeric) or any(
            value is not None and not math.isfinite(value) for value in optional
        ):
            raise ValueError("voice-quality metrics must be finite")
        if not math.isclose(
            self.local_noise_duration_seconds,
            self.local_noise_sample_count / 16000,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("local-noise duration must match its sample count")
        if self.local_noise_context_available:
            if (
                self.local_noise_sample_count < 3200
                or self.local_noise_rms_amplitude is None
            ):
                raise ValueError("available local-noise context requires 0.20 seconds")
        elif (
            self.local_noise_sample_count >= 3200
            or self.local_noise_rms_amplitude is not None
            or self.local_noise_rms_dbfs is not None
            or self.estimated_snr_db is not None
        ):
            raise ValueError("unavailable local-noise context cannot publish estimates")
        expected_noise_dbfs = (
            20 * math.log10(self.local_noise_rms_amplitude)
            if self.local_noise_rms_amplitude is not None
            and self.local_noise_rms_amplitude > 0
            else None
        )
        if expected_noise_dbfs is None:
            if self.local_noise_rms_dbfs is not None or self.estimated_snr_db is not None:
                raise ValueError("silent local-noise context has no finite dBFS or SNR")
        elif self.local_noise_rms_dbfs is None or not math.isclose(
            self.local_noise_rms_dbfs,
            expected_noise_dbfs,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("local-noise dBFS must match robust RMS amplitude")
        if self.rms_dbfs is not None and self.local_noise_rms_dbfs is not None:
            if self.estimated_snr_db is None or not math.isclose(
                self.estimated_snr_db,
                self.rms_dbfs - self.local_noise_rms_dbfs,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("estimated SNR must match turn and local-noise dBFS")
        elif self.estimated_snr_db is not None:
            raise ValueError("estimated SNR requires finite turn and local-noise dBFS")
        return self


class VoiceReferenceClipDiagnostics(SchemaModel):
    schema_version: Literal["r2v.h3.voice_reference_quality.2"] = (
        "r2v.h3.voice_reference_quality.2"
    )
    clip_uid: str
    source_audio_path: str
    status: Literal["ready", "failed"]
    thresholds_calibrated: Literal[False] = False
    candidate_turns: list[VoiceReferenceTurnDiagnostics] = Field(
        default_factory=list
    )
    reason: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> VoiceReferenceClipDiagnostics:
        if not self.clip_uid.strip() or not self.source_audio_path.strip():
            raise ValueError("voice-quality clip and source audio are required")
        if self.status == "ready" and self.reason is not None:
            raise ValueError("ready voice-quality diagnostics cannot have a reason")
        if self.status == "failed":
            if self.reason is None or not self.reason.strip():
                raise ValueError("failed voice-quality diagnostics require a reason")
            if self.candidate_turns:
                raise ValueError("failed voice-quality diagnostics cannot publish turns")
        if self.candidate_turns != sorted(
            self.candidate_turns,
            key=lambda item: (item.start_time, item.end_time, item.turn_id),
        ):
            raise ValueError("voice-quality turns must use deterministic order")
        return self


class VoiceQualityMetricDistribution(SchemaModel):
    count: int = Field(ge=0)
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None


class VoiceReferenceQualityPilotReport(SchemaModel):
    schema_version: Literal["r2v.h3.voice_reference_quality_report.2"] = (
        "r2v.h3.voice_reference_quality_report.2"
    )
    thresholds_calibrated: Literal[False] = False
    clip_report_count: int = Field(ge=0)
    diagnostics_ready_clip_count: int = Field(ge=0)
    diagnostics_failed_clip_count: int = Field(ge=0)
    clips_with_candidate_turns: int = Field(ge=0)
    candidate_turn_count: int = Field(ge=0)
    noise_context_available_count: int = Field(ge=0)
    noise_context_unavailable_count: int = Field(ge=0)
    noise_context_availability_rate: float = Field(ge=0, le=1)
    metric_distributions: dict[str, VoiceQualityMetricDistribution]

    @model_validator(mode="after")
    def validate_counts(self) -> VoiceReferenceQualityPilotReport:
        if self.clip_report_count != (
            self.diagnostics_ready_clip_count + self.diagnostics_failed_clip_count
        ):
            raise ValueError("voice-quality clip counts must reconcile")
        if self.candidate_turn_count != (
            self.noise_context_available_count
            + self.noise_context_unavailable_count
        ):
            raise ValueError("voice-quality noise-context counts must reconcile")
        expected_rate = (
            self.noise_context_available_count / self.candidate_turn_count
            if self.candidate_turn_count
            else 0.0
        )
        if not math.isclose(
            self.noise_context_availability_rate,
            expected_rate,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("voice-quality noise-context rate must match counts")
        return self
