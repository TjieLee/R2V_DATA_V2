from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import Field, model_validator

from r2v_data_v2.h3.schemas import BindingStatus, SchemaModel

AUDIO_CLIP_BINDING_SCHEMA_VERSION = "r2v.audio.clip_binding.1"
AUDIO_PAIR_SAMPLE_SCHEMA_VERSION = "r2v.audio.pair_sample.1"
H3_SAMPLE_SCHEMA_VERSION = "r2v.h3.sample.1"
CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS = 0.10

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENTITY_ID = re.compile(r"e[1-9]\d*")
_TURN_ID = re.compile(r"turn_[1-9]\d*")
_PICTURE_ID = re.compile(r"picture_[1-9]\d*")
_SUBJECT_ID = re.compile(r"subject_[1-9]\d*")
_AUDIO_ID = re.compile(r"audio_[1-9]\d*")


def _validate_relative_path(value: str, field_name: str) -> str:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        not normalized.strip()
        or "\x00" in normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:/", normalized) is not None
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{field_name} must be a safe export-relative path")
    return normalized


class FileAsset(SchemaModel):
    path: str
    sha256: str
    byte_size: int = Field(ge=0)
    media_type: str

    @model_validator(mode="after")
    def validate_asset(self) -> FileAsset:
        self.path = _validate_relative_path(self.path, "asset path")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("asset SHA-256 must be lowercase hexadecimal")
        if not self.media_type.strip():
            raise ValueError("asset media_type must not be empty")
        return self


class EmbeddingAsset(FileAsset):
    model_identifier: str
    checkpoint_sha256: Optional[str] = None
    dimension: int = Field(gt=0)
    dtype: Literal["float32"] = "float32"
    normalized: Literal[True] = True
    backend_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_embedding(self) -> EmbeddingAsset:
        if not self.model_identifier.strip():
            raise ValueError("embedding model identifier must not be empty")
        if self.checkpoint_sha256 is not None and _SHA256.fullmatch(
            self.checkpoint_sha256
        ) is None:
            raise ValueError("embedding checkpoint SHA-256 must be hexadecimal")
        return self


class AudioStreamProvenance(SchemaModel):
    stream_index: int = Field(ge=0)
    codec_name: str
    original_sample_rate_hz: int = Field(gt=0)
    original_channels: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    time_base: str

    @model_validator(mode="after")
    def validate_stream(self) -> AudioStreamProvenance:
        if not self.codec_name.strip() or not self.time_base.strip():
            raise ValueError("audio stream codec and time base must not be empty")
        if not math.isfinite(self.duration_seconds):
            raise ValueError("audio stream duration must be finite")
        return self


class FullAudioArtifact(SchemaModel):
    asset: FileAsset
    stream: AudioStreamProvenance
    output_sample_rate_hz: int = Field(gt=0)
    output_channels: int = Field(gt=0)
    output_format: Literal["flac", "wav"]


class SpeechTurn(SchemaModel):
    turn_id: str
    clip_uid: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    entity_id: Optional[str] = None
    face_track_id: Optional[str] = None
    status: BindingStatus
    binding_confidence: float = Field(ge=0, le=1)
    source_binding_ids: list[str] = Field(min_length=1)
    source_frame_ranges: list[tuple[int, int]] = Field(min_length=1)
    voice_reference_eligible: bool
    text: Optional[str] = None
    text_status: Literal["missing", "auto", "reviewed"] = "missing"

    @model_validator(mode="after")
    def validate_turn(self) -> SpeechTurn:
        if _TURN_ID.fullmatch(self.turn_id) is None:
            raise ValueError("speech turn ID must use turn_N")
        if not self.clip_uid.strip() or self.end_time <= self.start_time:
            raise ValueError("speech turn requires a clip and positive duration")
        if self.end_sample <= self.start_sample or self.end_frame < self.start_frame:
            raise ValueError("speech turn sample/frame extent is invalid")
        if len(self.source_binding_ids) != len(set(self.source_binding_ids)):
            raise ValueError("speech turn source binding IDs must be unique")
        if len(self.source_binding_ids) != len(self.source_frame_ranges):
            raise ValueError("speech turn IDs and frame ranges must align")
        if self.status == "bound":
            if self.entity_id is None or self.face_track_id is None:
                raise ValueError("bound speech turn requires entity and face track")
            if _ENTITY_ID.fullmatch(self.entity_id) is None:
                raise ValueError("bound speech turn entity must use eN")
        elif self.entity_id is not None or self.face_track_id is not None:
            raise ValueError("non-bound speech turn cannot claim an entity")
        if self.voice_reference_eligible and self.status != "bound":
            raise ValueError("only bound speech turns may be voice-reference eligible")
        if self.text_status == "missing":
            if self.text is not None:
                raise ValueError("missing transcript status requires null text")
        elif self.text is None or not self.text.strip():
            raise ValueError("available transcript status requires non-empty text")
        return self


class VisualReferenceProvenance(SchemaModel):
    image_asset: FileAsset
    token: str
    reference_scope: Literal["full", "local"]
    visible_region: str
    source_frame_index: Optional[int] = Field(default=None, ge=0)
    source_clip_uid: Optional[str] = None
    source_entity_id: Optional[str] = None
    synthetic: bool

    @model_validator(mode="after")
    def validate_reference(self) -> VisualReferenceProvenance:
        if not self.token.startswith("<ref_") or not self.visible_region.strip():
            raise ValueError("visual reference token/region is invalid")
        return self


class VoiceReferenceArtifact(SchemaModel):
    voice_reference_id: str
    entity_occurrence_id: str
    source_turn_id: str
    source_start: float = Field(ge=0)
    source_end: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    asset: FileAsset
    quality_score: float = Field(ge=0, le=1)
    quality_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_voice(self) -> VoiceReferenceArtifact:
        if not self.voice_reference_id.strip():
            raise ValueError("voice_reference_id must not be empty")
        if _TURN_ID.fullmatch(self.source_turn_id) is None:
            raise ValueError("voice reference source turn must use turn_N")
        if self.source_end <= self.source_start or (
            self.source_end_sample <= self.source_start_sample
        ):
            raise ValueError("voice reference interval is invalid")
        return self


class LocalBindingSummary(SchemaModel):
    bound_turn_ids: list[str] = Field(default_factory=list)
    total_bound_seconds: float = Field(ge=0)
    best_binding_confidence: float = Field(ge=0, le=1)
    high_confidence_bound: bool


class EntityOccurrence(SchemaModel):
    entity_occurrence_id: str
    entity_id: str
    reference_type: Literal["subject", "object", "group"]
    phrase: str
    grounding_prompt: str
    identity_text: Optional[str] = None
    identity_text_specific: bool = False
    visual_reference: VisualReferenceProvenance
    face_identity_status: Literal["available", "unavailable", "failed"]
    face_crop_asset: Optional[FileAsset] = None
    face_embedding_asset: Optional[EmbeddingAsset] = None
    identity_text_embedding_asset: Optional[EmbeddingAsset] = None
    primary_voice_reference: Optional[VoiceReferenceArtifact] = None
    voice_embedding_asset: Optional[EmbeddingAsset] = None
    local_binding_summary: LocalBindingSummary
    in_pair_eligible: bool
    cross_pair_eligible: bool
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_occurrence(self) -> EntityOccurrence:
        occurrence_parts = self.entity_occurrence_id.split("/")
        if len(occurrence_parts) != 2 or occurrence_parts[1] != self.entity_id:
            raise ValueError("entity occurrence ID must end with its entity_id")
        if _ENTITY_ID.fullmatch(self.entity_id) is None:
            raise ValueError("entity occurrence entity_id must use eN")
        if not self.phrase.strip() or not self.grounding_prompt.strip():
            raise ValueError("entity occurrence requires phrase and grounding prompt")
        if self.face_identity_status == "available":
            if self.face_crop_asset is None or self.face_embedding_asset is None:
                raise ValueError("available face identity requires crop and embedding")
        elif self.face_crop_asset is not None or self.face_embedding_asset is not None:
            raise ValueError("unavailable face identity cannot publish face assets")
        if self.primary_voice_reference is None:
            if self.voice_embedding_asset is not None or self.in_pair_eligible:
                raise ValueError("voice embedding/in-pair eligibility requires voice reference")
        else:
            if (
                self.reference_type != "subject"
                or self.primary_voice_reference.entity_occurrence_id
                != self.entity_occurrence_id
                or self.voice_embedding_asset is None
            ):
                raise ValueError("V1 voice reference requires its subject occurrence")
        if self.cross_pair_eligible and (
            not self.in_pair_eligible
            or self.face_embedding_asset is None
            or self.voice_embedding_asset is None
        ):
            raise ValueError("cross-pair eligibility requires face and voice evidence")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("entity occurrence reason codes must be unique")
        return self


class SourceVideoProvenance(SchemaModel):
    path: str
    sha256: str

    @model_validator(mode="after")
    def validate_video(self) -> SourceVideoProvenance:
        if not self.path.strip() or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source video provenance is invalid")
        return self


class VisualSourceProvenance(SchemaModel):
    run_root: str
    export_root: str
    visual_sample_id: str
    visual_sample_sha256: str

    @model_validator(mode="after")
    def validate_visual_source(self) -> VisualSourceProvenance:
        if any(
            not value.strip()
            for value in (self.run_root, self.export_root, self.visual_sample_id)
        ) or _SHA256.fullmatch(self.visual_sample_sha256) is None:
            raise ValueError("visual source provenance is incomplete")
        return self


class ProducerProvenance(SchemaModel):
    producer: str
    version: str
    config_fingerprint: str
    thresholds_calibrated: bool = False

    @model_validator(mode="after")
    def validate_producer(self) -> ProducerProvenance:
        if not self.producer.strip() or not self.version.strip():
            raise ValueError("producer identity must not be empty")
        if _SHA256.fullmatch(self.config_fingerprint) is None:
            raise ValueError("producer config fingerprint must be SHA-256")
        return self


class TranscriptProvenance(SchemaModel):
    backend: Optional[str] = None
    source_path: Optional[str] = None
    status: Literal["missing", "precomputed", "automatic", "reviewed"] = "missing"

    @model_validator(mode="after")
    def validate_transcript(self) -> TranscriptProvenance:
        if self.status == "missing":
            if self.backend is not None or self.source_path is not None:
                raise ValueError("missing transcript cannot claim provenance")
        elif self.backend is None or not self.backend.strip():
            raise ValueError("available transcript requires backend provenance")
        return self


class AudioClipBinding(SchemaModel):
    schema_version: Literal["r2v.audio.clip_binding.1"] = (
        AUDIO_CLIP_BINDING_SCHEMA_VERSION
    )
    clip_binding_id: str
    clip_uid: str
    sample_id: str
    parent_video_id: str
    clip_suffix: str
    source_video: SourceVideoProvenance
    visual_source: VisualSourceProvenance
    full_audio: FullAudioArtifact
    raw_frame_bindings_path: str
    speech_turns: list[SpeechTurn]
    entity_occurrences: list[EntityOccurrence]
    transcript_provenance: TranscriptProvenance
    visual_t2v_caption: str
    visual_r2v_instruction: str
    producer_provenance: ProducerProvenance

    @model_validator(mode="after")
    def validate_binding(self) -> AudioClipBinding:
        if self.clip_binding_id != f"clip_binding/{self.clip_uid}":
            raise ValueError("clip binding ID must derive from clip_uid")
        self.raw_frame_bindings_path = _validate_relative_path(
            self.raw_frame_bindings_path,
            "raw frame bindings path",
        )
        turn_ids = [turn.turn_id for turn in self.speech_turns]
        if turn_ids != [f"turn_{index}" for index in range(1, len(turn_ids) + 1)]:
            raise ValueError("speech turn IDs must be contiguous and ordered")
        if self.speech_turns != sorted(
            self.speech_turns,
            key=lambda item: (item.start_time, item.end_time, item.turn_id),
        ):
            raise ValueError("speech turns must be chronologically ordered")
        occurrence_ids = [item.entity_occurrence_id for item in self.entity_occurrences]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("entity occurrence IDs must be unique")
        if any(not value.startswith(f"{self.clip_uid}/") for value in occurrence_ids):
            raise ValueError("entity occurrence must belong to clip")
        known_entities = {item.entity_id for item in self.entity_occurrences}
        if any(
            turn.entity_id is not None and turn.entity_id not in known_entities
            for turn in self.speech_turns
        ):
            raise ValueError("speech turn references an unknown entity")
        return self


class SamePersonEvidence(SchemaModel):
    status: Literal["accepted", "candidate", "rejected"]
    face_similarity: float = Field(ge=-1, le=1)
    text_similarity: Optional[float] = Field(default=None, ge=-1, le=1)
    face_rank_left_to_right: Optional[int] = Field(default=None, gt=0)
    face_rank_right_to_left: Optional[int] = Field(default=None, gt=0)
    face_margin: float
    policy_version: str
    reason_codes: list[str]

    @model_validator(mode="after")
    def validate_evidence(self) -> SamePersonEvidence:
        if not math.isfinite(self.face_margin):
            raise ValueError("face margin must be finite")
        if not self.policy_version.strip() or len(self.reason_codes) != len(
            set(self.reason_codes)
        ):
            raise ValueError("same-person policy/reason provenance is invalid")
        return self


class SameVoiceEvidence(SchemaModel):
    status: Literal["accepted", "candidate", "rejected"]
    voice_similarity: float = Field(ge=-1, le=1)
    voice_rank_left_to_right: Optional[int] = Field(default=None, gt=0)
    voice_rank_right_to_left: Optional[int] = Field(default=None, gt=0)
    voice_margin: float
    policy_version: str
    reason_codes: list[str]

    @model_validator(mode="after")
    def validate_evidence(self) -> SameVoiceEvidence:
        if not math.isfinite(self.voice_margin):
            raise ValueError("voice margin must be finite")
        if not self.policy_version.strip() or len(self.reason_codes) != len(
            set(self.reason_codes)
        ):
            raise ValueError("same-voice policy/reason provenance is invalid")
        return self


class PairEvidence(SchemaModel):
    target_entity_occurrence_id: str
    reference_entity_occurrence_id: str
    same_person: SamePersonEvidence
    same_voice: SameVoiceEvidence
    combined_score: float


class PairTarget(SchemaModel):
    clip_binding_id: str
    clip_uid: str
    video: SourceVideoProvenance
    full_audio: FullAudioArtifact


class PairSubjectBinding(SchemaModel):
    subject_id: str
    target_entity_occurrence_id: str
    picture_entity_occurrence_id: str
    voice_entity_occurrence_id: str
    target_entity_id: str


class PairSubjectAudioBinding(SchemaModel):
    subject_id: str
    voice_reference_id: str
    entity_occurrence_id: str


class DraftAnnotation(SchemaModel):
    annotation_status: Literal["draft"] = "draft"
    is_final_annotation: Literal[False] = False
    renderer_profile: str
    renderer_version: str
    input_sha256: str
    text: str

    @model_validator(mode="after")
    def validate_draft(self) -> DraftAnnotation:
        if not self.renderer_profile.strip() or not self.renderer_version.strip():
            raise ValueError("draft annotation renderer provenance is required")
        if _SHA256.fullmatch(self.input_sha256) is None or not self.text.strip():
            raise ValueError("draft annotation input hash/text is invalid")
        return self


class AudioPairSample(SchemaModel):
    schema_version: Literal["r2v.audio.pair_sample.1"] = AUDIO_PAIR_SAMPLE_SCHEMA_VERSION
    pair_id: str
    pair_kind: Literal["in_pair", "cross_pair"]
    source_clip_binding_ids: list[str]
    target: PairTarget
    subjects: list[PairSubjectBinding]
    voice_references: list[VoiceReferenceArtifact]
    subject_audio_bindings: list[PairSubjectAudioBinding]
    speech_turns: list[SpeechTurn]
    pair_evidence: list[PairEvidence]
    eligibility: Literal["eligible"] = "eligible"
    annotation_draft: DraftAnnotation
    producer_provenance: ProducerProvenance

    @model_validator(mode="after")
    def validate_pair(self) -> AudioPairSample:
        if (
            not self.pair_id.strip()
            or not self.subjects
            or not self.source_clip_binding_ids
            or len(self.source_clip_binding_ids) != len(set(self.source_clip_binding_ids))
        ):
            raise ValueError("pair sample requires ID and subjects")
        if self.source_clip_binding_ids[0] != self.target.clip_binding_id:
            raise ValueError("pair source provenance must begin with target clip")
        subject_ids = [item.subject_id for item in self.subjects]
        if subject_ids != [
            f"subject_{index}" for index in range(1, len(subject_ids) + 1)
        ]:
            raise ValueError("pair subjects must be contiguous and ordered")
        target_occurrences = [
            item.target_entity_occurrence_id for item in self.subjects
        ]
        picture_occurrences = [
            item.picture_entity_occurrence_id for item in self.subjects
        ]
        voice_subject_occurrences = [
            item.voice_entity_occurrence_id for item in self.subjects
        ]
        if any(
            len(values) != len(set(values))
            for values in (
                target_occurrences,
                picture_occurrences,
                voice_subject_occurrences,
            )
        ):
            raise ValueError("pair occurrence mappings must be one-to-one")
        if len(self.voice_references) != len(self.subjects):
            raise ValueError("strict pair requires one voice reference per subject")
        if len(self.subject_audio_bindings) != len(self.subjects):
            raise ValueError("strict pair requires one subject/audio binding per subject")
        voice_occurrences = {
            voice.entity_occurrence_id for voice in self.voice_references
        }
        if len(voice_occurrences) != len(self.voice_references):
            raise ValueError("pair voice references must use unique occurrences")
        if voice_occurrences != {
            subject.voice_entity_occurrence_id for subject in self.subjects
        }:
            raise ValueError("pair subject voice mapping must match voice assets")
        if any(
            subject.picture_entity_occurrence_id
            != subject.target_entity_occurrence_id
            for subject in self.subjects
        ):
            raise ValueError("V1 picture must use the target visual occurrence")
        voice_by_occurrence = {
            voice.entity_occurrence_id: voice.voice_reference_id
            for voice in self.voice_references
        }
        expected_audio_bindings = [
            (
                subject.subject_id,
                voice_by_occurrence[subject.voice_entity_occurrence_id],
                subject.voice_entity_occurrence_id,
            )
            for subject in self.subjects
        ]
        actual_audio_bindings = [
            (
                binding.subject_id,
                binding.voice_reference_id,
                binding.entity_occurrence_id,
            )
            for binding in self.subject_audio_bindings
        ]
        if actual_audio_bindings != expected_audio_bindings:
            raise ValueError("pair subject/audio bindings must follow subject order")
        if any(
            turn.clip_uid != self.target.clip_uid for turn in self.speech_turns
        ):
            raise ValueError("pair speech turns must come from target clip")
        if self.pair_kind == "in_pair":
            if (
                self.pair_id != f"in_pair/{self.target.clip_uid}"
                or self.source_clip_binding_ids != [self.target.clip_binding_id]
                or self.pair_evidence
            ):
                raise ValueError("in-pair ID/evidence/source provenance is invalid")
            if any(
                subject.target_entity_occurrence_id
                != subject.voice_entity_occurrence_id
                for subject in self.subjects
            ):
                raise ValueError("in-pair voice must come from target occurrence")
        else:
            if not self.pair_id.startswith(
                f"cross_pair/{self.target.clip_uid}/"
            ) or len(self.source_clip_binding_ids) < 2:
                raise ValueError("cross-pair ID/source provenance is invalid")
            if any(
                subject.target_entity_occurrence_id
                == subject.voice_entity_occurrence_id
                for subject in self.subjects
            ):
                raise ValueError("cross-pair voice must use another occurrence")
            if len(self.pair_evidence) != len(self.subjects) or any(
                evidence.same_person.status != "accepted"
                or evidence.same_voice.status != "accepted"
                for evidence in self.pair_evidence
            ):
                raise ValueError("cross-pair requires strict accepted pair evidence")
            evidence_pairs = [
                (
                    evidence.target_entity_occurrence_id,
                    evidence.reference_entity_occurrence_id,
                )
                for evidence in self.pair_evidence
            ]
            expected_pairs = [
                (
                    subject.target_entity_occurrence_id,
                    subject.voice_entity_occurrence_id,
                )
                for subject in self.subjects
            ]
            if evidence_pairs != expected_pairs:
                raise ValueError("cross-pair evidence must follow subject mapping order")
        return self


class H3PictureAsset(SchemaModel):
    picture_id: str
    subject_id: str
    path: str
    sha256: str

    @model_validator(mode="after")
    def validate_picture(self) -> H3PictureAsset:
        if _PICTURE_ID.fullmatch(self.picture_id) is None:
            raise ValueError("H3 picture ID must use picture_N")
        if _SUBJECT_ID.fullmatch(self.subject_id) is None:
            raise ValueError("H3 picture subject ID must use subject_N")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("H3 picture SHA-256 is invalid")
        self.path = _validate_relative_path(self.path, "H3 picture path")
        return self


class H3SubjectAsset(SchemaModel):
    subject_id: str
    entity_occurrence_id: str
    entity_id: str
    phrase: str
    picture_id: str

    @model_validator(mode="after")
    def validate_subject(self) -> H3SubjectAsset:
        if _SUBJECT_ID.fullmatch(self.subject_id) is None:
            raise ValueError("H3 subject ID must use subject_N")
        if _PICTURE_ID.fullmatch(self.picture_id) is None:
            raise ValueError("H3 subject picture ID must use picture_N")
        if _ENTITY_ID.fullmatch(self.entity_id) is None or not self.phrase.strip():
            raise ValueError("H3 subject entity provenance is invalid")
        return self


class H3SampleAudioAsset(SchemaModel):
    audio_id: str
    role: Literal["target_full_audio", "voice_reference"]
    path: str
    sha256: str
    entity_occurrence_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_audio(self) -> H3SampleAudioAsset:
        if _AUDIO_ID.fullmatch(self.audio_id) is None:
            raise ValueError("H3 audio ID must use audio_N")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("H3 audio SHA-256 is invalid")
        self.path = _validate_relative_path(self.path, "H3 audio path")
        if self.role == "target_full_audio" and self.entity_occurrence_id is not None:
            raise ValueError("target full audio cannot claim one entity")
        if self.role == "voice_reference" and not self.entity_occurrence_id:
            raise ValueError("voice-reference audio requires entity occurrence")
        return self


class H3SubjectAudioBinding(SchemaModel):
    subject_id: str
    audio_id: str
    entity_occurrence_id: str

    @model_validator(mode="after")
    def validate_binding(self) -> H3SubjectAudioBinding:
        if _SUBJECT_ID.fullmatch(self.subject_id) is None:
            raise ValueError("H3 audio binding subject ID must use subject_N")
        if _AUDIO_ID.fullmatch(self.audio_id) is None:
            raise ValueError("H3 audio binding audio ID must use audio_N")
        return self


class H3Sample(SchemaModel):
    schema_version: Literal["r2v.h3.sample.1"] = H3_SAMPLE_SCHEMA_VERSION
    sample_id: str
    pair_kind: Literal["in_pair", "cross_pair"]
    target_video: SourceVideoProvenance
    target_full_audio: H3SampleAudioAsset
    pictures: list[H3PictureAsset]
    subjects: list[H3SubjectAsset]
    voice_reference_audio: list[H3SampleAudioAsset]
    subject_audio_bindings: list[H3SubjectAudioBinding]
    speech_turns: list[SpeechTurn]
    rendered_h3_draft_annotation: DraftAnnotation
    source_clip_binding_ids: list[str]
    pair_evidence: list[PairEvidence]
    producer_provenance: ProducerProvenance

    @model_validator(mode="after")
    def validate_h3(self) -> H3Sample:
        picture_ids = [item.picture_id for item in self.pictures]
        subject_ids = [item.subject_id for item in self.subjects]
        voice_ids = [item.audio_id for item in self.voice_reference_audio]
        if len(self.pictures) != len(self.subjects) or len(
            self.voice_reference_audio
        ) != len(self.subjects):
            raise ValueError("H3 V1 requires one picture and voice asset per subject")
        if picture_ids != [f"picture_{index}" for index in range(1, len(picture_ids) + 1)]:
            raise ValueError("H3 pictures must be contiguous")
        if subject_ids != [f"subject_{index}" for index in range(1, len(subject_ids) + 1)]:
            raise ValueError("H3 subjects must be contiguous")
        expected_voice_ids = [
            f"audio_{index}" for index in range(2, len(voice_ids) + 2)
        ]
        if self.target_full_audio.audio_id != "audio_1" or voice_ids != expected_voice_ids:
            raise ValueError("H3 audio assets must be contiguous with target first")
        if any(
            _validate_relative_path(asset.path, "H3 asset path") != asset.path
            for asset in [self.target_full_audio, *self.voice_reference_audio]
        ):
            raise ValueError("H3 audio path normalization changed")
        if any(
            _validate_relative_path(asset.path, "H3 picture path") != asset.path
            for asset in self.pictures
        ):
            raise ValueError("H3 picture path normalization changed")
        if {item.subject_id for item in self.subject_audio_bindings} != set(subject_ids):
            raise ValueError("every H3 subject requires one audio binding")
        if [item.picture_id for item in self.subjects] != picture_ids:
            raise ValueError("H3 subjects must map to pictures in deterministic order")
        expected_audio_bindings = [
            (
                subject.subject_id,
                self.voice_reference_audio[index].audio_id,
                self.voice_reference_audio[index].entity_occurrence_id,
            )
            for index, subject in enumerate(self.subjects)
        ]
        actual_audio_bindings = [
            (item.subject_id, item.audio_id, item.entity_occurrence_id)
            for item in self.subject_audio_bindings
        ]
        if actual_audio_bindings != expected_audio_bindings:
            raise ValueError("H3 subject/audio mappings must be ordered and exact")
        return self


class AudioDatasetManifest(SchemaModel):
    schema_version: Literal["r2v.audio.dataset.1"] = "r2v.audio.dataset.1"
    clip_binding_count: int = Field(ge=0)
    failed_clip_count: int = Field(default=0, ge=0)
    pair_sample_count: int = Field(ge=0)
    producer_provenance: ProducerProvenance


class H3DatasetManifest(SchemaModel):
    schema_version: Literal["r2v.h3.dataset.1"] = "r2v.h3.dataset.1"
    sample_count: int = Field(ge=0)
    producer_provenance: ProducerProvenance
