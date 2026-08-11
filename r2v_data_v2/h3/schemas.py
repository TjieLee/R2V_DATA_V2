from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

H3_AUDIO_BINDING_SCHEMA_VERSION = "r2v.h3.audio_binding.2"
H3_AUDIO_IR_SCHEMA_VERSION = "r2v.h3.audio_ir.2"

_ENTITY_ID = re.compile(r"e[1-9]\d*")
_FACE_TRACK_ID = re.compile(r"face_[1-9]\d*")
_PICTURE_ID = re.compile(r"picture_[1-9]\d*")
_SUBJECT_ID = re.compile(r"subject_[1-9]\d*")
_AUDIO_ID = re.compile(r"audio_[1-9]\d*")
_VOICE_REFERENCE_ID = re.compile(r"voice_reference_[1-9]\d*")
_H3_TASK_COMPONENT_ORDER = (
    "reference_generation",
    "audio_reference",
    "audio_reuse",
)


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AudioTrackMetadata(SchemaModel):
    status: Literal["ready", "missing", "corrupted"]
    source_video_path: str
    full_audio_path: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, gt=0)
    sample_rate_hz: Optional[int] = Field(default=None, gt=0)
    channels: Optional[int] = Field(default=None, gt=0)
    quality_score: Optional[float] = Field(default=None, ge=0, le=1)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_state(self) -> AudioTrackMetadata:
        if not self.source_video_path.strip():
            raise ValueError("audio source_video_path must not be empty")
        if self.status == "ready":
            if (
                self.full_audio_path is None
                or not self.full_audio_path.strip()
                or self.duration_seconds is None
                or self.sample_rate_hz is None
                or self.channels is None
            ):
                raise ValueError("ready audio requires complete track metadata")
            if self.reason is not None:
                raise ValueError("ready audio cannot have a failure reason")
        elif self.reason is None or not self.reason.strip():
            raise ValueError("unavailable audio requires a reason")
        return self


class FaceGeometrySample(SchemaModel):
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_geometry(self) -> FaceGeometrySample:
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(math.isfinite(value) for value in self.bbox_xyxy):
            raise ValueError("face geometry bbox values must be finite")
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("face geometry bbox must have positive bounded extent")
        return self


class FaceTrack(SchemaModel):
    face_track_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    sample_count: int = Field(gt=0)
    mean_detection_confidence: float = Field(ge=0, le=1)
    geometry_samples: list[FaceGeometrySample] = Field(
        min_length=1,
        max_length=32,
    )

    @model_validator(mode="after")
    def validate_track(self) -> FaceTrack:
        if _FACE_TRACK_ID.fullmatch(self.face_track_id) is None:
            raise ValueError("face_track_id must use face_N")
        if not self.end_time > self.start_time:
            raise ValueError("face track end_time must exceed start_time")
        if self.sample_count < len(self.geometry_samples):
            raise ValueError("face sample_count cannot be below sampled geometry count")
        if any(
            sample.timestamp < self.start_time or sample.timestamp > self.end_time
            for sample in self.geometry_samples
        ):
            raise ValueError("face geometry timestamp must be within track extent")
        sample_order = [
            (sample.timestamp, sample.frame_index) for sample in self.geometry_samples
        ]
        if sample_order != sorted(sample_order):
            raise ValueError("face geometry samples must be chronologically ordered")
        frame_indices = [sample.frame_index for sample in self.geometry_samples]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("face geometry samples must use unique frame indices")
        return self


AssociationMethod = Literal[
    "mask_overlap",
    "tracked_geometry",
    "precomputed",
    "test_fixture",
]


class EntityFaceAssociation(SchemaModel):
    face_track_id: str
    entity_id: str
    confidence: float = Field(ge=0, le=1)
    method: AssociationMethod
    evidence: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ids(self) -> EntityFaceAssociation:
        if _FACE_TRACK_ID.fullmatch(self.face_track_id) is None:
            raise ValueError("association face_track_id must use face_N")
        if _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ValueError("association entity_id must use eN")
        return self


class ActiveSpeakerInterval(SchemaModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    speech_present: bool
    visible_face_track_ids: list[str] = Field(default_factory=list)
    face_speaking_probabilities: dict[str, float] = Field(default_factory=dict)
    asd_coverage_ratio: float = Field(ge=0, le=1)
    audio_quality_usable: bool
    synchronization_plausible: bool

    @model_validator(mode="after")
    def validate_interval(self) -> ActiveSpeakerInterval:
        if not self.end_time > self.start_time:
            raise ValueError("active-speaker end_time must exceed start_time")
        if len(self.visible_face_track_ids) != len(set(self.visible_face_track_ids)):
            raise ValueError("visible face track IDs must be unique")
        if any(
            _FACE_TRACK_ID.fullmatch(face_track_id) is None
            for face_track_id in self.visible_face_track_ids
        ):
            raise ValueError("visible face track IDs must use face_N")
        for face_track_id, probability in self.face_speaking_probabilities.items():
            if _FACE_TRACK_ID.fullmatch(face_track_id) is None:
                raise ValueError("ASD probability keys must use face_N")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("ASD probabilities must be finite and in [0, 1]")
        visible = set(self.visible_face_track_ids)
        scored = set(self.face_speaking_probabilities)
        if scored - visible:
            raise ValueError("ASD scores must refer to visible face tracks")
        expected_coverage = len(scored) / len(visible) if visible else 1.0
        if not math.isclose(
            self.asd_coverage_ratio,
            expected_coverage,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("ASD coverage ratio must match scored visible faces")
        return self


class VoiceReferenceCandidate(SchemaModel):
    path: str
    source_start: float = Field(ge=0)
    source_end: float = Field(gt=0)
    quality_score: float = Field(ge=0, le=1)
    quality_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_candidate(self) -> VoiceReferenceCandidate:
        if not self.path.strip():
            raise ValueError("voice reference path must not be empty")
        if not self.source_end > self.source_start:
            raise ValueError("voice reference source_end must exceed source_start")
        return self


class AudioBindingEvidence(SchemaModel):
    clip_uid: str
    audio: AudioTrackMetadata
    face_tracks: list[FaceTrack] = Field(default_factory=list)
    associations: list[EntityFaceAssociation] = Field(default_factory=list)
    active_speaker_intervals: list[ActiveSpeakerInterval] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> AudioBindingEvidence:
        if not self.clip_uid.strip():
            raise ValueError("audio evidence clip_uid must not be empty")
        track_ids = [track.face_track_id for track in self.face_tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("face track IDs must be unique")
        known_tracks = set(track_ids)
        association_tracks = [item.face_track_id for item in self.associations]
        if len(association_tracks) != len(set(association_tracks)):
            raise ValueError("each face track may have at most one entity association")
        if set(association_tracks) - known_tracks:
            raise ValueError("entity association references an unknown face track")
        for interval in self.active_speaker_intervals:
            if set(interval.visible_face_track_ids) - known_tracks:
                raise ValueError(
                    "ASD interval references an unknown visible face track"
                )
            if set(interval.face_speaking_probabilities) - known_tracks:
                raise ValueError("ASD interval references an unknown face track")
        ordered = sorted(
            self.active_speaker_intervals,
            key=lambda item: (item.start_time, item.end_time),
        )
        if self.active_speaker_intervals != ordered:
            raise ValueError("ASD intervals must be deterministically ordered")
        if any(
            ordered[index].end_time > ordered[index + 1].start_time
            for index in range(len(ordered) - 1)
        ):
            raise ValueError("ASD intervals must not overlap")
        if self.audio.status == "ready":
            assert self.audio.duration_seconds is not None
            end_times = [
                *(track.end_time for track in self.face_tracks),
                *(interval.end_time for interval in self.active_speaker_intervals),
            ]
            if any(value > self.audio.duration_seconds for value in end_times):
                raise ValueError("audio evidence exceeds the source audio duration")
        return self


BindingStatus = Literal["bound", "overlap", "offscreen", "ambiguous", "no_speech"]


class BindingEvidence(SchemaModel):
    active_face_track_ids: list[str] = Field(default_factory=list)
    face_speaking_probabilities: dict[str, float] = Field(default_factory=dict)
    association_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    audio_quality_usable: bool
    synchronization_plausible: bool
    clean_training_eligible: bool
    reason_codes: list[str] = Field(default_factory=list)


class AudioEntityBinding(SchemaModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    entity_id: Optional[str] = None
    face_track_id: Optional[str] = None
    status: BindingStatus
    confidence: float = Field(ge=0, le=1)
    evidence: BindingEvidence

    @model_validator(mode="after")
    def validate_binding(self) -> AudioEntityBinding:
        if not self.end_time > self.start_time:
            raise ValueError("binding end_time must exceed start_time")
        if self.status == "bound":
            if self.entity_id is None or self.face_track_id is None:
                raise ValueError("bound audio requires entity and face track IDs")
            if _ENTITY_ID.fullmatch(self.entity_id) is None:
                raise ValueError("bound entity_id must use eN")
            if _FACE_TRACK_ID.fullmatch(self.face_track_id) is None:
                raise ValueError("bound face_track_id must use face_N")
        elif self.entity_id is not None or self.face_track_id is not None:
            raise ValueError("non-bound audio must not claim an entity or face track")
        if self.evidence.clean_training_eligible and self.status != "bound":
            raise ValueError("only bound audio may be clean training eligible")
        return self


class VoiceReference(SchemaModel):
    voice_reference_id: str
    entity_id: str
    path: str
    source_start: float = Field(ge=0)
    source_end: float = Field(gt=0)
    quality_score: float = Field(ge=0, le=1)
    quality_metadata: dict[str, object] = Field(default_factory=dict)
    binding_start: float = Field(ge=0)
    binding_end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_intervals(self) -> VoiceReference:
        if _VOICE_REFERENCE_ID.fullmatch(self.voice_reference_id) is None:
            raise ValueError("voice_reference_id must use voice_reference_N")
        if _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ValueError("voice reference entity_id must use eN")
        if not self.path.strip():
            raise ValueError("voice reference path must not be empty")
        if not self.source_end > self.source_start:
            raise ValueError("voice reference source interval is invalid")
        if not self.binding_end > self.binding_start:
            raise ValueError("voice reference binding interval is invalid")
        if self.source_start < self.binding_start or self.source_end > self.binding_end:
            raise ValueError(
                "voice reference must be contained in its binding interval"
            )
        return self


class PictureAsset(SchemaModel):
    picture_id: str
    entity_id: str
    path: str

    @model_validator(mode="after")
    def validate_asset(self) -> PictureAsset:
        if _PICTURE_ID.fullmatch(self.picture_id) is None:
            raise ValueError("picture_id must use picture_N")
        if _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ValueError("picture entity_id must use eN")
        if not self.path.strip():
            raise ValueError("picture path must not be empty")
        return self


class SemanticSubject(SchemaModel):
    subject_id: str
    entity_id: str
    reference_type: Literal["subject", "object", "group"]
    phrase: str
    source_assets: list[str]

    @model_validator(mode="after")
    def validate_subject(self) -> SemanticSubject:
        if _SUBJECT_ID.fullmatch(self.subject_id) is None:
            raise ValueError("subject_id must use subject_N")
        if _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ValueError("subject entity_id must use eN")
        if not self.phrase.strip() or not self.source_assets:
            raise ValueError("semantic subject requires phrase and source assets")
        if any(_PICTURE_ID.fullmatch(value) is None for value in self.source_assets):
            raise ValueError("subject source_assets must use picture_N")
        return self


class H3AudioAsset(SchemaModel):
    audio_id: str
    role: Literal["voice_reference", "full_audio"]
    voice_reference_id: Optional[str] = None
    entity_id: Optional[str] = None
    path: str
    source_start: Optional[float] = Field(default=None, ge=0)
    source_end: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_asset(self) -> H3AudioAsset:
        if _AUDIO_ID.fullmatch(self.audio_id) is None:
            raise ValueError("audio_id must use audio_N")
        if not self.path.strip():
            raise ValueError("audio asset path must not be empty")
        if self.role == "voice_reference":
            if (
                self.voice_reference_id is None
                or _VOICE_REFERENCE_ID.fullmatch(self.voice_reference_id) is None
                or self.entity_id is None
                or self.source_start is None
                or self.source_end is None
                or not self.source_end > self.source_start
            ):
                raise ValueError("voice-reference audio requires entity and interval")
        elif (
            self.voice_reference_id is not None
            or self.entity_id is not None
            or self.source_start is not None
            or self.source_end is not None
        ):
            raise ValueError("full-audio asset cannot claim an entity interval")
        return self


H3TaskComponent = Literal[
    "reference_generation",
    "audio_reference",
    "audio_reuse",
]


class H3TaskSpecification(SchemaModel):
    components: list[H3TaskComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_components(self) -> H3TaskSpecification:
        if len(self.components) != len(set(self.components)):
            raise ValueError("H3 task components must be unique")
        expected = [
            component
            for component in _H3_TASK_COMPONENT_ORDER
            if component in self.components
        ]
        if self.components != expected:
            raise ValueError("H3 task components must use deterministic order")
        return self


class H3AudioBindingIR(SchemaModel):
    schema_version: Literal["r2v.h3.audio_ir.2"] = H3_AUDIO_IR_SCHEMA_VERSION
    clip_uid: str
    task: H3TaskSpecification
    picture_assets: list[PictureAsset]
    subjects: list[SemanticSubject]
    audio_assets: list[H3AudioAsset]
    bindings: list[AudioEntityBinding]

    @model_validator(mode="after")
    def validate_numbering(self) -> H3AudioBindingIR:
        picture_ids = [item.picture_id for item in self.picture_assets]
        expected_pictures = [
            f"picture_{index}" for index in range(1, len(picture_ids) + 1)
        ]
        if picture_ids != expected_pictures:
            raise ValueError("picture assets must be contiguous and ordered")
        subject_ids = [item.subject_id for item in self.subjects]
        expected_subjects = [
            f"subject_{index}" for index in range(1, len(subject_ids) + 1)
        ]
        if subject_ids != expected_subjects:
            raise ValueError("subjects must be contiguous and ordered")
        audio_ids = [item.audio_id for item in self.audio_assets]
        expected_audio = [f"audio_{index}" for index in range(1, len(audio_ids) + 1)]
        if audio_ids != expected_audio:
            raise ValueError("audio assets must be contiguous and ordered")
        if len(self.picture_assets) != len(self.subjects):
            raise ValueError("each H3 subject requires one picture asset in V1")
        for picture, subject in zip(self.picture_assets, self.subjects, strict=True):
            if picture.entity_id != subject.entity_id or subject.source_assets != [
                picture.picture_id
            ]:
                raise ValueError("subject and picture ordering must match")
        subject_by_entity = {subject.entity_id: subject for subject in self.subjects}
        if len(subject_by_entity) != len(self.subjects):
            raise ValueError("H3 subjects must use unique entity IDs")
        voice_assets = [
            item for item in self.audio_assets if item.role == "voice_reference"
        ]
        full_audio_assets = [
            item for item in self.audio_assets if item.role == "full_audio"
        ]
        voice_entities = [item.entity_id for item in voice_assets]
        if len(voice_entities) != len(set(voice_entities)):
            raise ValueError("V1 allows at most one voice reference per entity")
        voice_reference_ids = [item.voice_reference_id for item in voice_assets]
        if len(voice_reference_ids) != len(set(voice_reference_ids)):
            raise ValueError("H3 voice-reference IDs must be unique")
        for asset in voice_assets:
            assert asset.entity_id is not None
            subject = subject_by_entity.get(asset.entity_id)
            if subject is None:
                raise ValueError("voice-reference entity must exist in H3 subjects")
            if subject.reference_type != "subject":
                raise ValueError("V1 voice references only support subject entities")
        if len(full_audio_assets) > 1:
            raise ValueError("H3 IR allows at most one full-audio asset")
        components = set(self.task.components)
        if ("audio_reference" in components) != bool(voice_assets):
            raise ValueError(
                "audio_reference task component must match voice-reference assets"
            )
        if ("audio_reuse" in components) != bool(full_audio_assets):
            raise ValueError("audio_reuse task component must match full-audio asset")
        return self


class AudioBindingSidecar(SchemaModel):
    schema_version: Literal["r2v.h3.audio_binding.2"] = H3_AUDIO_BINDING_SCHEMA_VERSION
    clip_uid: str
    source_run_root: str
    source_video_path: str
    status: Literal["ready", "ineligible", "failed"]
    reason: Optional[str] = None
    evidence: Optional[AudioBindingEvidence] = None
    bindings: list[AudioEntityBinding] = Field(default_factory=list)
    voice_references: list[VoiceReference] = Field(default_factory=list)
    h3_ir: Optional[H3AudioBindingIR] = None

    @model_validator(mode="after")
    def validate_state(self) -> AudioBindingSidecar:
        if self.status == "ready":
            if self.reason is not None or self.evidence is None or self.h3_ir is None:
                raise ValueError("ready sidecar requires evidence and H3 IR")
        elif self.reason is None or not self.reason.strip():
            raise ValueError("non-ready sidecar requires a reason")
        reference_ids = [
            reference.voice_reference_id for reference in self.voice_references
        ]
        expected_ids = [
            f"voice_reference_{index}" for index in range(1, len(reference_ids) + 1)
        ]
        if reference_ids != expected_ids:
            raise ValueError("voice references must be contiguous and ordered")
        reference_entities = [
            reference.entity_id for reference in self.voice_references
        ]
        if len(reference_entities) != len(set(reference_entities)):
            raise ValueError("V1 allows at most one voice reference per entity")
        if self.h3_ir is not None:
            asset_references = [
                (
                    asset.voice_reference_id,
                    asset.entity_id,
                    asset.path,
                    asset.source_start,
                    asset.source_end,
                )
                for asset in self.h3_ir.audio_assets
                if asset.role == "voice_reference"
            ]
            sidecar_references = [
                (
                    reference.voice_reference_id,
                    reference.entity_id,
                    reference.path,
                    reference.source_start,
                    reference.source_end,
                )
                for reference in self.voice_references
            ]
            if asset_references != sidecar_references:
                raise ValueError("H3 voice assets must match sidecar voice references")
        elif self.voice_references:
            raise ValueError("voice references require an H3 IR")
        return self


class PrecomputedClipEvidence(SchemaModel):
    evidence: AudioBindingEvidence
    voice_reference_candidates: list[VoiceReferenceCandidate] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_candidate_intervals(self) -> PrecomputedClipEvidence:
        if self.evidence.audio.status == "ready":
            assert self.evidence.audio.duration_seconds is not None
            if any(
                candidate.source_end > self.evidence.audio.duration_seconds
                for candidate in self.voice_reference_candidates
            ):
                raise ValueError("voice candidate exceeds source audio duration")
        return self


class PrecomputedEvidenceFile(SchemaModel):
    clips: list[PrecomputedClipEvidence]

    @model_validator(mode="after")
    def validate_clip_ids(self) -> PrecomputedEvidenceFile:
        clip_ids = [item.evidence.clip_uid for item in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("precomputed evidence clip_uid values must be unique")
        return self


class AudioBindingRunSummary(SchemaModel):
    schema_version: Literal["r2v.h3.audio_binding.2"] = H3_AUDIO_BINDING_SCHEMA_VERSION
    source_run_root: str
    clip_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    ineligible_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    bound_interval_count: int = Field(ge=0)
    overlap_interval_count: int = Field(ge=0)
    offscreen_interval_count: int = Field(ge=0)
    ambiguous_interval_count: int = Field(ge=0)
    no_speech_interval_count: int = Field(ge=0)
    voice_reference_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> AudioBindingRunSummary:
        if (
            self.clip_count
            != self.ready_count + self.ineligible_count + self.failed_count
        ):
            raise ValueError("audio binding clip counts must reconcile")
        return self
