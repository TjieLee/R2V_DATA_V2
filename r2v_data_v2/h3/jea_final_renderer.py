from __future__ import annotations

import hashlib
import html
import json
import shutil
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from r2v_data_v2.h3.binding_audit import (
    BINDING_AUDIT_POLICY_VERSION,
    SpeakerBindingAuditSummary,
    SpeakerBindingSegmentAudit,
)
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClipResult,
    DiarizationInventory,
)
from r2v_data_v2.h3.jea_audio_production import (
    CANONICAL_AUDIO_CHANNELS,
    CANONICAL_AUDIO_SAMPLE_RATE_HZ,
    CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS,
    VOICE_SAMPLE_MAPPING_POLICY,
    CanonicalAudioClip,
    JEACrossPair,
    JEAInPair,
)
from r2v_data_v2.h3.primary_voice import PrimaryVoiceReferenceSelection
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.specialized_audio_semantics import (
    SpecializedAudioSemanticsRecord,
)
from r2v_data_v2.h3.target_audio_caption_contract import (
    TargetSpeakerDelivery,
    TemporalAudioEvent,
)
from r2v_data_v2.h3.visual_production_source import (
    NormalizedVisualReference,
    VisualProductionInventory,
)
from r2v_data_v2.v3.production_export import ProductionReference

FINAL_SAMPLE_VERSION = "r2v.h3.final_sample.5"
FINAL_SUMMARY_VERSION = "r2v.h3.final_summary.6"


class FinalVisualReference(ProductionReference):
    image_artifact_path: str

    @field_validator("image_artifact_path")
    @classmethod
    def validate_image_artifact_path(cls, value: str) -> str:
        path = Path(value)
        if not value.strip() or not path.is_absolute():
            raise ValueError("final Visual image_artifact_path must be absolute")
        if not path.exists() or not path.is_file():
            raise ValueError("final Visual image_artifact_path must be an existing file")
        return value

    @classmethod
    def from_visual(cls, value: NormalizedVisualReference) -> FinalVisualReference:
        payload = value.model_dump(mode="json", exclude={"artifact_path"})
        payload["image_artifact_path"] = value.artifact_path
        return cls.model_validate(payload)


class FinalSubjectVoice(SchemaModel):
    subject_index: int = Field(gt=0)
    entity_id: str
    target_occurrence_id: str
    voice_reference_path: str
    voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    voice_sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    voice_channels: Literal[2] = CANONICAL_AUDIO_CHANNELS
    source_start: float = Field(ge=0, allow_inf_nan=False)
    source_end: float = Field(gt=0, allow_inf_nan=False)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    sample_mapping_policy: Literal["round_time_seconds_times_32000_v1"]
    voice_source: Literal["target", "cross_donor"]
    donor_occurrence_id: str | None = None
    donor_clip_uid: str | None = None
    donor_clip_display_path: str | None = None

    @model_validator(mode="after")
    def validate_voice(self) -> FinalSubjectVoice:
        if (
            self.source_end <= self.source_start
            or self.source_end_sample <= self.source_start_sample
        ):
            raise ValueError("final H3 voice interval is invalid")
        if (self.source_start_sample, self.source_end_sample) != (
            round(self.source_start * self.voice_sample_rate_hz),
            round(self.source_end * self.voice_sample_rate_hz),
        ):
            raise ValueError("final H3 voice samples differ from authoritative times")
        donor_values = (
            self.donor_occurrence_id,
            self.donor_clip_uid,
            self.donor_clip_display_path,
        )
        if self.voice_source == "target" and any(
            value is not None for value in donor_values
        ):
            raise ValueError("in-pair voice cannot publish donor provenance")
        if self.voice_source == "cross_donor" and any(
            value is None for value in donor_values
        ):
            raise ValueError("cross voice requires complete donor provenance")
        return self


class FinalQwen3SpeechSegment(SchemaModel):
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    text: str
    language: str | None = None
    asr_model: Literal["Qwen/Qwen3-ASR-1.7B"] = "Qwen/Qwen3-ASR-1.7B"

    @model_validator(mode="after")
    def validate_canonical_sample_domain(self) -> FinalQwen3SpeechSegment:
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("final speech segment sample range must be positive")
        if (self.source_start_sample, self.source_end_sample) != (
            round(self.start_time * self.source_sample_rate_hz),
            round(self.end_time * self.source_sample_rate_hz),
        ):
            raise ValueError("final speech samples must use canonical 32 kHz time")
        return self


class FinalFullClipAudioSemantics(SchemaModel):
    source_kind: Literal["specialized_audio_semantics"] = (
        "specialized_audio_semantics"
    )
    source_schema_version: str
    source_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["complete", "partial", "failed"]
    overall_audio_description: str | None = None
    overall_soundscape: str | None = None
    non_diegetic_music: str | None = None
    temporal_audio_events: list[TemporalAudioEvent]
    speaker_delivery: list[TargetSpeakerDelivery]


class FinalH3SampleV2(SchemaModel):
    schema_version: Literal["r2v.h3.final_sample.5"] = FINAL_SAMPLE_VERSION
    sample_id: str
    pair_id: str
    pair_type: Literal["canonical", "in_pair", "cross_pair"]
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video: str
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    target_audio_channels: Literal[2] = CANONICAL_AUDIO_CHANNELS
    r2v_instruction: str
    visual_references: list[FinalVisualReference]
    subject_voices: list[FinalSubjectVoice]
    speech_segments: list[FinalQwen3SpeechSegment]
    full_clip_audio_semantics: FinalFullClipAudioSemantics | None = None

    @model_validator(mode="after")
    def validate_sample(self) -> FinalH3SampleV2:
        indexes = [item.image_index for item in self.visual_references]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("final Visual references must preserve canonical order")
        subject_ids = [
            item.entity_id for item in self.visual_references if item.kind == "subject"
        ]
        subject_index_by_entity = {
            entity_id: index for index, entity_id in enumerate(subject_ids, start=1)
        }
        voice_entity_ids = [item.entity_id for item in self.subject_voices]
        if len(voice_entity_ids) != len(set(voice_entity_ids)):
            raise ValueError("subject voice bindings must have unique entity IDs")
        if not set(voice_entity_ids).issubset(subject_index_by_entity):
            raise ValueError(
                "only canonical subject references may receive voice binding"
            )
        for voice in self.subject_voices:
            if voice.target_occurrence_id != f"{self.clip_uid}/{voice.entity_id}":
                raise ValueError("subject voice target occurrence is inconsistent")
            if voice.subject_index != subject_index_by_entity[voice.entity_id]:
                raise ValueError(
                    "subject voice index must match canonical subject order"
                )
        if self.pair_type == "canonical" and self.subject_voices:
            raise ValueError("canonical base sample cannot publish voice references")
        if any(not item.text.strip() for item in self.speech_segments):
            raise ValueError("final Qwen3 speech segments must be non-empty")
        return self


class FinalH3SummaryV2(SchemaModel):
    schema_version: Literal["r2v.h3.final_summary.6"] = FINAL_SUMMARY_VERSION
    final_sample_schema_version: Literal[
        "r2v.h3.final_sample.5"
    ] = FINAL_SAMPLE_VERSION
    canonical_clip_count: int = Field(ge=0)
    canonical_base_sample_count: int = Field(ge=0)
    in_pair_sample_count: int = Field(ge=0)
    cross_pair_sample_count: int = Field(ge=0)
    final_sample_count: int = Field(ge=0)
    canonical_clips_without_target_voice_variant_count: int = Field(ge=0)
    canonical_clips_with_empty_speech_count: int = Field(ge=0)
    audio_semantics_available_clip_count: int = Field(ge=0)
    audio_semantics_complete_clip_count: int = Field(ge=0)
    audio_semantics_partial_clip_count: int = Field(ge=0)
    audio_semantics_failed_clip_count: int = Field(ge=0)
    audio_semantics_missing_clip_count: int = Field(ge=0)
    speech_segment_count: int = Field(ge=0)
    entity_bindings_removed_by_direct_anchor_gate: int = Field(ge=0)
    visual_reference_kind_counts: dict[str, int]
    speaker_binding_audit_policy_version: Literal[
        "h3_speaker_binding_structural_audit_v1"
    ] = BINDING_AUDIT_POLICY_VERSION
    source_binding_audit_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_audit_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_canonical_audio_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_semantics_records_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    asr_model: Literal["Qwen/Qwen3-ASR-1.7B"] = "Qwen/Qwen3-ASR-1.7B"
    whisper_rows_consumed: Literal[0] = 0
    language_probability_gate_applied: Literal[False] = False
    dots3_used: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> FinalH3SummaryV2:
        if self.canonical_base_sample_count != self.canonical_clip_count:
            raise ValueError("final H3 must publish one canonical base per clip")
        if self.final_sample_count != (
            self.canonical_base_sample_count
            + self.in_pair_sample_count
            + self.cross_pair_sample_count
        ):
            raise ValueError("final H3 sample counts do not reconcile")
        if self.audio_semantics_available_clip_count != (
            self.audio_semantics_complete_clip_count
            + self.audio_semantics_partial_clip_count
            + self.audio_semantics_failed_clip_count
        ):
            raise ValueError("final H3 Audio semantics counts do not reconcile")
        if self.canonical_clip_count != (
            self.audio_semantics_available_clip_count
            + self.audio_semantics_missing_clip_count
        ):
            raise ValueError("final H3 Audio semantics coverage is inconsistent")
        if (self.source_audio_semantics_records_sha256 is None) != (
            self.audio_semantics_available_clip_count == 0
        ):
            raise ValueError("final H3 Audio semantics provenance is inconsistent")
        return self


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_binding_audit(
    *,
    binding_audit_root: Path,
    diarization_root: Path,
    expected_clip_count: int,
) -> tuple[
    dict[tuple[str, str], SpeakerBindingSegmentAudit],
    str,
    str,
]:
    root = binding_audit_root.expanduser().resolve(strict=True)
    summary_path = root / "summary.json"
    segments_path = root / "segments.jsonl"
    if not summary_path.is_file() or not segments_path.is_file():
        raise ValueError("speaker-binding audit artifacts are incomplete")
    summary = SpeakerBindingAuditSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    expected_production_root = root.parent.resolve(strict=True)
    expected_diarization_root = diarization_root.expanduser().resolve(strict=True)
    if (
        Path(summary.source_audio_production_root).expanduser().resolve(strict=True)
        != expected_production_root
        or Path(summary.source_diarization_root).expanduser().resolve(strict=True)
        != expected_diarization_root
        or summary.clip_count != expected_clip_count
    ):
        raise ValueError("speaker-binding audit production provenance is inconsistent")
    bound_path = expected_diarization_root / "bound_segments.jsonl"
    inventory_path = expected_diarization_root / "inventory.json"
    if (
        summary.source_artifact_sha256.get("bound_segments") != _sha256(bound_path)
        or summary.source_artifact_sha256.get("inventory")
        != _sha256(inventory_path)
    ):
        raise ValueError("speaker-binding audit differs from bound DiariZen evidence")
    segments = [
        SpeakerBindingSegmentAudit.model_validate(row)
        for row in _read_rows(segments_path)
    ]
    by_key = {(item.clip_uid, item.segment_id): item for item in segments}
    if len(by_key) != len(segments) or summary.segment_count != len(segments):
        raise ValueError("speaker-binding audit segment inventory is inconsistent")
    return by_key, _sha256(summary_path), _sha256(segments_path)


def _load_canonical_audio(
    *,
    audio_root: Path,
    visual_inventory: VisualProductionInventory,
) -> tuple[dict[str, CanonicalAudioClip], str]:
    records_path = audio_root.expanduser().resolve(strict=True) / "canonical_clips.jsonl"
    records = [
        CanonicalAudioClip.model_validate(row) for row in _read_rows(records_path)
    ]
    by_clip = {item.clip_uid: item for item in records}
    visual_by_clip = {
        item.identity.clip_uid: item for item in visual_inventory.canonical_clips
    }
    if len(by_clip) != len(records) or set(by_clip) != set(visual_by_clip):
        raise ValueError("canonical Visual and Audio inventories differ")
    for clip_uid, record in by_clip.items():
        visual = visual_by_clip[clip_uid]
        video = Path(record.target_video_path).expanduser().resolve(strict=True)
        audio = Path(record.target_full_audio_path).expanduser().resolve(strict=True)
        if (
            video != Path(visual.sample.target_video).expanduser().resolve(strict=True)
            or _sha256(video) != record.target_video_sha256
            or _sha256(audio) != record.target_full_audio_sha256
        ):
            raise ValueError("canonical target media provenance differs")
    return by_clip, _sha256(records_path)


def _load_primary_voice_references(
    root: Path,
) -> dict[str, tuple[PrimaryVoiceReferenceSelection, Path]]:
    resolved_root = root.expanduser().resolve(strict=True)
    records_path = resolved_root / "primary_voice_references.jsonl"
    records = [
        PrimaryVoiceReferenceSelection.model_validate(row)
        for row in _read_rows(records_path)
    ]
    occurrence_ids = [item.entity_occurrence_id for item in records]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("primary voice selections contain duplicate occurrences")
    by_occurrence: dict[str, tuple[PrimaryVoiceReferenceSelection, Path]] = {}
    for item in records:
        artifact = item.primary_voice_reference
        if artifact is None:
            continue
        path = (resolved_root / artifact.asset.path).resolve(strict=True)
        path.relative_to(resolved_root)
        metadata = artifact.quality_metadata
        if (
            _sha256(path) != artifact.asset.sha256
            or metadata.get("source_sample_rate_hz")
            != CANONICAL_AUDIO_SAMPLE_RATE_HZ
            or metadata.get("source_channels") != CANONICAL_AUDIO_CHANNELS
            or metadata.get("sample_mapping_policy") != VOICE_SAMPLE_MAPPING_POLICY
        ):
            raise ValueError("primary voice canonical Audio provenance differs")
        by_occurrence[item.entity_occurrence_id] = (item, path)
    return by_occurrence


def _load_audio_semantics(
    *,
    root: Path | None,
    canonical_by_clip: dict[str, CanonicalAudioClip],
) -> tuple[dict[str, FinalFullClipAudioSemantics], str | None]:
    if root is None:
        return {}, None
    resolved = root.expanduser().resolve(strict=True)
    candidates = (resolved / "assembled/records.jsonl", resolved / "records.jsonl")
    records_path = next((path for path in candidates if path.is_file()), None)
    if records_path is None:
        raise ValueError("Audio semantics root has no assembled records.jsonl")
    records = [
        SpecializedAudioSemanticsRecord.model_validate(row)
        for row in _read_rows(records_path)
    ]
    by_clip = {item.target_clip_uid: item for item in records}
    if len(by_clip) != len(records) or set(by_clip) != set(canonical_by_clip):
        raise ValueError("Audio semantics must exactly cover canonical clips")
    projected: dict[str, FinalFullClipAudioSemantics] = {}
    for clip_uid, record in by_clip.items():
        canonical = canonical_by_clip[clip_uid]
        video = Path(record.target_video_path).expanduser().resolve(strict=True)
        audio = Path(record.target_full_audio_path).expanduser().resolve(strict=True)
        if (
            video
            != Path(canonical.target_video_path).expanduser().resolve(strict=True)
            or audio
            != Path(canonical.target_full_audio_path).expanduser().resolve(strict=True)
            or record.target_video_sha256 != canonical.target_video_sha256
            or record.target_full_audio_sha256 != canonical.target_full_audio_sha256
            or abs(record.target_duration_seconds - canonical.target_duration_seconds)
            > CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS
        ):
            raise ValueError("Audio semantics target provenance differs")
        projected[clip_uid] = FinalFullClipAudioSemantics(
            source_schema_version=record.schema_version,
            source_record_fingerprint=record.assemble_fingerprint,
            status=record.status,
            overall_audio_description=record.overall_audio_description,
            overall_soundscape=record.overall_soundscape,
            non_diegetic_music=record.non_diegetic_music,
            temporal_audio_events=record.temporal_audio_events,
            speaker_delivery=record.speaker_delivery,
        )
    return projected, _sha256(records_path)


def _load_optional_pairs(
    pairs_root: Path | None,
) -> tuple[list[JEAInPair], list[JEACrossPair]]:
    if pairs_root is None:
        return [], []
    root = pairs_root.expanduser().resolve(strict=True)
    in_path = root / "in_pairs.jsonl"
    cross_path = root / "cross_pairs.jsonl"
    if not in_path.is_file() or not cross_path.is_file():
        raise ValueError("optional pair artifacts are incomplete")
    return (
        [JEAInPair.model_validate(row) for row in _read_rows(in_path)],
        [JEACrossPair.model_validate(row) for row in _read_rows(cross_path)],
    )


def _validate_pair_target(
    *,
    pair: JEAInPair,
    canonical: CanonicalAudioClip,
    visual_references: list[FinalVisualReference],
) -> None:
    reference_by_entity = {
        item.entity_id: item
        for item in visual_references
        if item.kind == "subject" and item.entity_id is not None
    }
    if (
        pair.target_clip_uid != canonical.clip_uid
        or Path(pair.target_video_path).expanduser().resolve(strict=True)
        != Path(canonical.target_video_path).expanduser().resolve(strict=True)
        or Path(pair.target_full_audio_path).expanduser().resolve(strict=True)
        != Path(canonical.target_full_audio_path).expanduser().resolve(strict=True)
    ):
        raise ValueError("pair target media differs from canonical target")
    for subject in pair.subjects:
        reference = reference_by_entity.get(subject.target_entity_id)
        if (
            reference is None
            or subject.target_occurrence_id
            != f"{canonical.clip_uid}/{subject.target_entity_id}"
            or Path(subject.target_visual_reference_path).expanduser().resolve(
                strict=True
            )
            != Path(reference.image_artifact_path).expanduser().resolve(strict=True)
        ):
            raise ValueError("pair subject differs from canonical Visual reference")


def _final_voice(
    *,
    subject_index: int,
    entity_id: str,
    target_occurrence_id: str,
    pair_voice_path: str,
    source: tuple[PrimaryVoiceReferenceSelection, Path],
    voice_source: Literal["target", "cross_donor"],
    donor_occurrence_id: str | None = None,
    donor_clip_uid: str | None = None,
    donor_clip_display_path: str | None = None,
) -> FinalSubjectVoice:
    selection, source_path = source
    artifact = selection.primary_voice_reference
    assert artifact is not None
    if selection.entity_occurrence_id != (
        target_occurrence_id if voice_source == "target" else donor_occurrence_id
    ):
        raise ValueError("pair voice occurrence differs from H3 voice provenance")
    if Path(pair_voice_path).expanduser().resolve(strict=True) != source_path:
        raise ValueError("pair voice differs from primary voice provenance")
    return FinalSubjectVoice(
        subject_index=subject_index,
        entity_id=entity_id,
        target_occurrence_id=target_occurrence_id,
        voice_reference_path=str(source_path),
        voice_reference_sha256=artifact.asset.sha256,
        source_start=artifact.source_start,
        source_end=artifact.source_end,
        source_start_sample=artifact.source_start_sample,
        source_end_sample=artifact.source_end_sample,
        sample_mapping_policy=VOICE_SAMPLE_MAPPING_POLICY,
        voice_source=voice_source,
        donor_occurrence_id=donor_occurrence_id,
        donor_clip_uid=donor_clip_uid,
        donor_clip_display_path=donor_clip_display_path,
    )


def _target_voices(
    pair: JEAInPair,
    voices: dict[str, tuple[PrimaryVoiceReferenceSelection, Path]],
) -> list[FinalSubjectVoice]:
    return [
        _final_voice(
            subject_index=item.subject_index,
            entity_id=item.target_entity_id,
            target_occurrence_id=item.target_occurrence_id,
            pair_voice_path=item.target_primary_voice_reference_path,
            source=voices[item.target_occurrence_id],
            voice_source="target",
        )
        for item in pair.subjects
    ]


def _cross_voices(
    pair: JEAInPair,
    cross: JEACrossPair,
    voices: dict[str, tuple[PrimaryVoiceReferenceSelection, Path]],
) -> list[FinalSubjectVoice]:
    entity_by_occurrence = {
        item.target_occurrence_id: item.target_entity_id for item in pair.subjects
    }
    return [
        _final_voice(
            subject_index=item.subject_index,
            entity_id=entity_by_occurrence[item.target_occurrence_id],
            target_occurrence_id=item.target_occurrence_id,
            pair_voice_path=item.donor_primary_voice_reference_path,
            source=voices[item.donor_occurrence_id],
            voice_source="cross_donor",
            donor_occurrence_id=item.donor_occurrence_id,
            donor_clip_uid=item.donor_clip_uid,
            donor_clip_display_path=item.donor_clip_display_path,
        )
        for item in cross.mappings
    ]


def _speech(
    value: Qwen3ASRSegment,
    audit: SpeakerBindingSegmentAudit,
) -> tuple[FinalQwen3SpeechSegment, bool]:
    assert value.text is not None
    directly_anchored = (
        audit.current_mapping_status == "candidate_mapped"
        and audit.direct_anchor_seconds > 0
    )
    fully_propagated = (
        audit.current_mapping_status == "candidate_mapped"
        and audit.direct_anchor_seconds == 0
        and audit.flags.fully_propagated_segment
    )
    if audit.current_mapping_status == "candidate_mapped" and not (
        directly_anchored or fully_propagated
    ):
        raise ValueError("mapped speaker-binding audit eligibility is inconsistent")
    if directly_anchored and audit.flags.fully_propagated_segment:
        raise ValueError("directly anchored segment cannot be fully propagated")
    entity_id = value.entity_id if directly_anchored else None
    entity_occurrence_id = value.entity_occurrence_id if directly_anchored else None
    return (
        FinalQwen3SpeechSegment(
            segment_id=value.segment_id,
            speaker_cluster_id=value.speaker_cluster_id,
            entity_id=entity_id,
            entity_occurrence_id=entity_occurrence_id,
            source_start_sample=value.source_start_sample,
            source_end_sample=value.source_end_sample,
            source_sample_rate_hz=value.source_sample_rate_hz,
            start_time=value.start_time,
            end_time=value.end_time,
            text=value.text,
            language=value.language,
        ),
        value.entity_id is not None and entity_id is None,
    )


def render_jea_final_samples(
    *,
    visual_inventory: VisualProductionInventory,
    audio_root: Path,
    pairs_root: Path | None,
    diarization_root: Path,
    binding_audit_root: Path,
    qwen3_asr_root: Path,
    output_root: Path,
    primary_voice_root: Path | None = None,
    audio_semantics_root: Path | None = None,
    overwrite: bool = False,
) -> FinalH3SummaryV2:
    canonical_by_clip, canonical_manifest_sha256 = _load_canonical_audio(
        audio_root=audio_root,
        visual_inventory=visual_inventory,
    )
    semantics_by_clip, semantics_records_sha256 = _load_audio_semantics(
        root=audio_semantics_root,
        canonical_by_clip=canonical_by_clip,
    )
    in_pairs, cross_pairs = _load_optional_pairs(pairs_root)
    primary_voices = (
        {}
        if not in_pairs
        else _load_primary_voice_references(
            primary_voice_root
            if primary_voice_root is not None
            else audio_root.expanduser().resolve(strict=True).parent / "primary_voice"
        )
    )
    in_by_clip = {item.target_clip_uid: item for item in in_pairs}
    if len(in_by_clip) != len(in_pairs) or not set(in_by_clip).issubset(
        canonical_by_clip
    ):
        raise ValueError("in-pair inventory is not a unique canonical subset")
    cross_by_clip = {item.target_clip_uid: item for item in cross_pairs}
    if (
        len(cross_by_clip) != len(cross_pairs)
        or not set(cross_by_clip).issubset(in_by_clip)
    ):
        raise ValueError("more than one cross pair was published for a target clip")

    diarization = diarization_root.expanduser().resolve(strict=True)
    diarization_inventory = DiarizationInventory.model_validate_json(
        (diarization / "inventory.json").read_text(encoding="utf-8")
    )
    canonical_manifest_path = (
        audio_root.expanduser().resolve(strict=True) / "canonical_clips.jsonl"
    )
    if (
        diarization_inventory.source_inventory_kind != "canonical_audio_manifest"
        or diarization_inventory.selection_mode
        != "canonical_visual_target_inventory_v1"
        or diarization_inventory.source_canonical_audio_manifest_path is None
        or Path(
            diarization_inventory.source_canonical_audio_manifest_path
        ).expanduser().resolve(strict=True)
        != canonical_manifest_path
        or diarization_inventory.source_canonical_audio_manifest_sha256
        != canonical_manifest_sha256
    ):
        raise ValueError("DiariZen inventory is not rooted in canonical Audio")
    diarization_ids = [item.target_clip_uid for item in diarization_inventory.targets]
    if (
        len(diarization_ids) != len(set(diarization_ids))
        or set(diarization_ids) != set(canonical_by_clip)
    ):
        raise ValueError("DiariZen targets do not exactly cover canonical clips")
    clip_results = [
        DiarizationClipResult.model_validate(row)
        for row in _read_rows(diarization / "clip_results.jsonl")
    ]
    result_by_clip = {item.target_clip_uid: item for item in clip_results}
    if len(result_by_clip) != len(clip_results) or set(result_by_clip) != set(
        canonical_by_clip
    ):
        raise ValueError("DiariZen clip results do not exactly cover canonical clips")
    failed = sorted(
        item.target_clip_uid for item in clip_results if item.status == "failed"
    )
    if failed:
        raise ValueError(f"final H3 cannot consume failed diarization clips: {failed}")
    for target in diarization_inventory.targets:
        canonical = canonical_by_clip[target.target_clip_uid]
        if (
            Path(target.source_audio_path).expanduser().resolve(strict=True)
            != Path(canonical.target_full_audio_path).expanduser().resolve(strict=True)
            or target.source_audio_sha256 != canonical.target_full_audio_sha256
            or Path(target.target_video_path).expanduser().resolve(strict=True)
            != Path(canonical.target_video_path).expanduser().resolve(strict=True)
        ):
            raise ValueError("DiariZen target provenance differs from canonical media")
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read_rows(diarization / "bound_segments.jsonl")
    ]
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    audit_by_key, audit_summary_sha256, audit_segments_sha256 = _load_binding_audit(
        binding_audit_root=binding_audit_root,
        diarization_root=diarization,
        expected_clip_count=len(canonical_by_clip),
    )
    if set(audit_by_key) != set(bound_by_key):
        raise ValueError("speaker-binding audit does not cover bound DiariZen segments")
    for key, source in bound_by_key.items():
        audit = audit_by_key[key]
        if (
            audit.speaker_cluster_id != source.speaker_cluster_id
            or audit.current_mapping_status != source.cluster_binding_status
            or audit.current_entity_id != source.entity_id
            or audit.start_time != source.start_time
            or audit.end_time != source.end_time
        ):
            raise ValueError("speaker-binding audit differs from DiariZen identity")
    qwen_rows = [
        Qwen3ASRSegment.model_validate(row)
        for row in _read_rows(
            qwen3_asr_root.expanduser().resolve(strict=True) / "segments.jsonl"
        )
    ]
    qwen_by_key = {(item.clip_uid, item.segment_id): item for item in qwen_rows}
    if len(qwen_by_key) != len(qwen_rows) or set(qwen_by_key) != set(bound_by_key):
        raise ValueError("Qwen3 ASR rows do not exactly cover DiariZen segments")
    failed_asr = sorted(
        f"{item.clip_uid}/{item.segment_id}"
        for item in qwen_rows
        if item.status == "failed"
    )
    if failed_asr:
        preview = failed_asr[:10]
        raise ValueError(
            "final H3 cannot consume failed Qwen3-ASR segments: "
            f"count={len(failed_asr)}, first={preview}"
        )
    for row in qwen_rows:
        source = bound_by_key.get((row.clip_uid, row.segment_id))
        if source is None or (
            source.source_start_sample,
            source.source_end_sample,
            source.speaker_cluster_id,
            source.entity_id,
        ) != (
            row.source_start_sample,
            row.source_end_sample,
            row.speaker_cluster_id,
            row.entity_id,
        ):
            raise ValueError("Qwen3 row differs from its exact DiariZen segment")
    speech_by_clip: dict[str, list[FinalQwen3SpeechSegment]] = {}
    removed_entity_binding_count = 0
    for row in qwen_rows:
        if row.status != "transcribed":
            continue
        audit = audit_by_key[(row.clip_uid, row.segment_id)]
        speech, removed = _speech(row, audit)
        removed_entity_binding_count += int(removed)
        speech_by_clip.setdefault(row.clip_uid, []).append(speech)
    for rows in speech_by_clip.values():
        rows.sort(key=lambda item: (item.source_start_sample, item.segment_id))

    samples: list[FinalH3SampleV2] = []
    for visual in visual_inventory.canonical_clips:
        clip_uid = visual.identity.clip_uid
        canonical = canonical_by_clip[clip_uid]
        references = [
            FinalVisualReference.from_visual(item)
            for item in visual.sample.references
        ]
        pair = in_by_clip.get(clip_uid)
        if pair is not None:
            _validate_pair_target(
                pair=pair,
                canonical=canonical,
                visual_references=references,
            )
        common = {
            **visual.identity.model_dump(mode="python", exclude={"clip_uid"}),
            "clip_uid": clip_uid,
            "target_video": visual.sample.target_video,
            "target_full_audio_path": canonical.target_full_audio_path,
            "target_full_audio_sha256": canonical.target_full_audio_sha256,
            "r2v_instruction": visual.sample.r2v_instruction,
            "visual_references": references,
            "speech_segments": speech_by_clip.get(clip_uid, []),
            "full_clip_audio_semantics": semantics_by_clip.get(clip_uid),
        }
        samples.append(
            FinalH3SampleV2(
                sample_id=f"{visual.sample.sample_id}/canonical",
                pair_id=f"canonical/{clip_uid}",
                pair_type="canonical",
                subject_voices=[],
                **common,
            )
        )
        if pair is not None:
            samples.append(
                FinalH3SampleV2(
                    sample_id=f"{visual.sample.sample_id}/in_pair",
                    pair_id=pair.pair_id,
                    pair_type="in_pair",
                    subject_voices=_target_voices(pair, primary_voices),
                    **common,
                )
            )
        cross = cross_by_clip.get(clip_uid)
        if cross is not None:
            assert pair is not None
            if (
                Path(cross.target_video_path).expanduser().resolve(strict=True)
                != Path(canonical.target_video_path).expanduser().resolve(strict=True)
                or Path(cross.target_full_audio_path).expanduser().resolve(strict=True)
                != Path(canonical.target_full_audio_path).expanduser().resolve(
                    strict=True
                )
            ):
                raise ValueError("cross-pair target media differs from canonical target")
            samples.append(
                FinalH3SampleV2(
                    sample_id=f"{visual.sample.sample_id}/cross_pair/1",
                    pair_id=cross.pair_id,
                    pair_type="cross_pair",
                    subject_voices=_cross_voices(pair, cross, primary_voices),
                    **common,
                )
            )

    kind_counts = Counter(
        reference.kind for sample in samples for reference in sample.visual_references
    )
    summary = FinalH3SummaryV2(
        canonical_clip_count=visual_inventory.canonical_sample_count,
        canonical_base_sample_count=sum(
            item.pair_type == "canonical" for item in samples
        ),
        in_pair_sample_count=sum(item.pair_type == "in_pair" for item in samples),
        cross_pair_sample_count=sum(item.pair_type == "cross_pair" for item in samples),
        final_sample_count=len(samples),
        canonical_clips_without_target_voice_variant_count=(
            visual_inventory.canonical_sample_count - len(in_by_clip)
        ),
        canonical_clips_with_empty_speech_count=sum(
            not speech_by_clip.get(item.identity.clip_uid)
            for item in visual_inventory.canonical_clips
        ),
        audio_semantics_available_clip_count=len(semantics_by_clip),
        audio_semantics_complete_clip_count=sum(
            item.status == "complete" for item in semantics_by_clip.values()
        ),
        audio_semantics_partial_clip_count=sum(
            item.status == "partial" for item in semantics_by_clip.values()
        ),
        audio_semantics_failed_clip_count=sum(
            item.status == "failed" for item in semantics_by_clip.values()
        ),
        audio_semantics_missing_clip_count=(
            visual_inventory.canonical_sample_count - len(semantics_by_clip)
        ),
        speech_segment_count=sum(len(item.speech_segments) for item in samples),
        entity_bindings_removed_by_direct_anchor_gate=removed_entity_binding_count,
        visual_reference_kind_counts=dict(sorted(kind_counts.items())),
        source_binding_audit_summary_sha256=audit_summary_sha256,
        source_binding_audit_segments_sha256=audit_segments_sha256,
        source_canonical_audio_manifest_sha256=canonical_manifest_sha256,
        source_audio_semantics_records_sha256=semantics_records_sha256,
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"final H3 output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "samples.jsonl").write_text(
            "".join(
                json.dumps(
                    item.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for item in samples
            ),
            encoding="utf-8",
        )
        for sample in samples:
            relative = PurePosixPath(sample.clip_display_path)
            name = {
                "canonical": "canonical.json",
                "in_pair": "in_pair.json",
                "cross_pair": "cross_pair_1.json",
            }[sample.pair_type]
            path = temporary / "samples" / Path(*relative.parts) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sample.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        rows = "".join(
            f"<tr><td>{html.escape(item.clip_display_path)}</td><td>{item.pair_type}</td><td>{len(item.visual_references)}</td><td>{len(item.speech_segments)}</td></tr>"
            for item in samples
        )
        (temporary / "review.html").write_text(
            "<!doctype html><meta charset='utf-8'><table><tr><th>clip</th><th>pair</th><th>visual refs</th><th>speech</th></tr>"
            + rows
            + "</table>",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
