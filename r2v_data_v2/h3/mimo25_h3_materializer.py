from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import AudioMediaBackend, FFmpegAudioMediaBackend
from r2v_data_v2.h3.jea_audio_production import (
    CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS,
)
from r2v_data_v2.h3.jea_final_renderer import (
    FinalH3SampleV2,
    FinalQwen3SpeechSegment,
    FinalSubjectVoice,
)
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoInventory,
    MimoRecord,
    project_mimo_h3_sample_references,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_MATERIALIZER_VERSION,
    MimoAudioEvent,
    MimoH3AudioEventPart,
    MimoH3Shot,
    MimoH3SpeechPart,
    MimoH3VisualPart,
    MimoSegmentDecision,
    MimoSubjectDefinitionDraft,
)
from r2v_data_v2.h3.mimo25_recovered_voice import (
    RECOVERED_VOICE_POLICY_VERSION,
    FFmpegRecoveredVoiceAudioAnalyzer,
    MimoRecoveredVoiceReference,
    RecoveredVoiceAudioAnalyzer,
    recover_mimo_target_voices,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    AudioFactAuditItem,
    ConditioningVariant,
    Qwen38DraftShot,
    Qwen38H3DraftResponse,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRequest,
    RecaptionAudioContract,
    RecaptionAudioFacts,
    RecaptionNonSpeechFact,
    RecaptionReferenceContract,
    RecaptionSpeechFact,
    RecaptionSubjectContract,
    _render_locked_speech,
    build_reference_contract,
    materialize_h3_draft,
    render_h3_prompt,
    validate_h3_response,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speech_presentation import (
    render_speech_presentation_clause,
)
from r2v_data_v2.structured_output import ValidationIssue

MIMO25_SHADOW_RECORD_VERSION = "r2v.h3.mimo25_h3_shadow.11"
MIMO25_SHADOW_SUMMARY_VERSION = "r2v.h3.mimo25_h3_shadow_summary.12"
MUSIC_REFERENCE_POLICY_VERSION = "h3_mimo25_clean_music_reference_v1"
MUSIC_REFERENCE_SAMPLE_RATE_HZ = 32000
MUSIC_REFERENCE_CHANNELS = 2
MUSIC_REFERENCE_MINIMUM_DURATION_SECONDS = 1.0
MUSIC_REFERENCE_MINIMUM_RMS_DBFS = -60.0
MUSIC_REFERENCE_MAXIMUM_CLIPPING_RATIO = 0.001


class MimoH3MaterializationContractError(ValueError):
    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "MiMo H3 materialization violates reference contract: "
            + _compact_json([item.to_dict() for item in self.issues])
        )

    def failure_reason(self) -> str:
        return _compact_json(
            {
                "category": "materialization_contract_failed",
                "issues": [item.to_dict() for item in self.issues],
            }
        )


class MimoShadowAudioReference(SchemaModel):
    audio_index: int = Field(gt=0)
    role: Literal["voice_reference", "music_reference", "full_audio_reuse"]
    subject_index: int | None = Field(default=None, gt=0)
    entity_id: str | None = None
    speaker_id: str | None = Field(default=None, pattern=r"^S[1-9]\d*$")
    source_type: Literal[
        "existing_target_voice",
        "mimo_recovered_target_voice",
        "cross_donor",
        "mimo_non_diegetic_music_event",
        "canonical_full_audio",
    ]
    audio_path: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_segment_id: str | None = None
    source_event_id: str | None = None
    source_start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    source_end_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    source_start_sample: int | None = Field(default=None, ge=0)
    source_end_sample: int | None = Field(default=None, gt=0)
    sample_rate_hz: Literal[32000] = MUSIC_REFERENCE_SAMPLE_RATE_HZ
    channels: Literal[2] = MUSIC_REFERENCE_CHANNELS
    interval_provenance: str | None = None
    music_description: str | None = None
    rms_dbfs: float | None = Field(default=None, allow_inf_nan=False)
    clipping_ratio: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_reference(self) -> MimoShadowAudioReference:
        interval = (
            self.source_start_time,
            self.source_end_time,
            self.source_start_sample,
            self.source_end_sample,
        )
        if self.role == "voice_reference":
            if (
                self.subject_index is None
                or self.entity_id is None
                or self.source_type
                not in {
                    "existing_target_voice",
                    "mimo_recovered_target_voice",
                    "cross_donor",
                }
                or self.source_event_id is not None
                or self.music_description is not None
                or any(value is not None for value in interval)
            ):
                raise ValueError("voice Audio reference provenance is invalid")
        elif self.role == "music_reference":
            if (
                self.subject_index is not None
                or self.entity_id is not None
                or self.speaker_id is not None
                or self.source_type != "mimo_non_diegetic_music_event"
                or self.source_segment_id is not None
                or self.source_event_id is None
                or any(value is None for value in interval)
                or self.interval_provenance
                != "mimo_approximate_event_times_rounded_to_32000_v1"
                or not self.music_description
                or self.rms_dbfs is None
                or self.clipping_ratio is None
            ):
                raise ValueError("music Audio reference provenance is invalid")
        elif (
            self.subject_index is not None
            or self.entity_id is not None
            or self.speaker_id is not None
            or self.source_type != "canonical_full_audio"
            or self.source_segment_id is not None
            or self.source_event_id is not None
            or any(value is not None for value in interval)
            or self.interval_provenance != "canonical_full_audio_exact_v1"
            or self.music_description is not None
            or self.rms_dbfs is not None
            or self.clipping_ratio is not None
        ):
            raise ValueError("full-audio reuse provenance is invalid")
        return self


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(item.model_dump(mode="json")) + "\n" for item in values),
        encoding="utf-8",
    )


class MimoH3ShadowRecord(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_h3_shadow.11"] = MIMO25_SHADOW_RECORD_VERSION
    sample_id: str
    source_h3_sample_id: str
    clip_uid: str
    pair_type: Literal["canonical", "in_pair", "cross_pair"]
    derived_from_pair_type: Literal["canonical"] | None = None
    conditioning_variant: ConditioningVariant
    status: Literal["ready", "failed"]
    source_mimo_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materializer_version: Literal[
        "h3_mimo25_materializer_v11",
        "h3_mimo25_materializer_v12",
        "h3_mimo25_materializer_v13",
        "h3_mimo25_materializer_v14",
    ] = (
        MIMO25_MATERIALIZER_VERSION
    )
    corrected_speech_segments: list[FinalQwen3SpeechSegment]
    effective_subject_voices: list[FinalSubjectVoice]
    audio_references: list[MimoShadowAudioReference]
    recovered_voice_references: list[MimoRecoveredVoiceReference]
    rendered_h3_prompt: str | None = None
    warnings: list[str]
    failure_reason: str | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> MimoH3ShadowRecord:
        if self.status == "ready":
            if self.rendered_h3_prompt is None or self.failure_reason is not None:
                raise ValueError("ready MiMo H3 shadow record is incomplete")
        elif (
            self.rendered_h3_prompt is not None
            or not self.failure_reason
            or self.effective_subject_voices
            or self.audio_references
            or self.recovered_voice_references
        ):
            raise ValueError("failed MiMo H3 shadow record requires only failure")
        if (self.derived_from_pair_type is None) != (
            self.sample_id == self.source_h3_sample_id
        ):
            raise ValueError("derived MiMo H3 shadow provenance is inconsistent")
        if self.pair_type != "in_pair" and self.recovered_voice_references:
            raise ValueError("only target in-pair shadow may publish recovered voices")
        voice_references = [
            item for item in self.audio_references if item.role == "voice_reference"
        ]
        if len(voice_references) != len(self.effective_subject_voices):
            raise ValueError("MiMo shadow Audio provenance count is inconsistent")
        if [item.audio_index for item in self.audio_references] != list(
            range(1, len(self.audio_references) + 1)
        ):
            raise ValueError("MiMo shadow Audio provenance order is inconsistent")
        recovered_by_entity = {
            item.entity_id: item
            for item in self.recovered_voice_references
            if item.status == "selected"
        }
        for voice, audio in zip(self.effective_subject_voices, voice_references):
            recovered = recovered_by_entity.get(voice.entity_id)
            expected_source = (
                "cross_donor"
                if voice.voice_source == "cross_donor"
                else (
                    "mimo_recovered_target_voice"
                    if recovered is not None
                    else "existing_target_voice"
                )
            )
            if (
                audio.subject_index != voice.subject_index
                or audio.entity_id != voice.entity_id
                or audio.source_type != expected_source
                or audio.audio_path != voice.voice_reference_path
                or audio.audio_sha256 != voice.voice_reference_sha256
                or audio.source_segment_id
                != (None if recovered is None else recovered.source_segment_id)
            ):
                raise ValueError("MiMo shadow Audio provenance differs from voice asset")
        expected_variant: ConditioningVariant
        if self.sample_id == self.source_h3_sample_id:
            expected_variant = {
                "canonical": "visual_only",
                "in_pair": "target_voice_reference",
                "cross_pair": "cross_voice_reference",
            }[self.pair_type]
        elif self.pair_type == "in_pair":
            expected_variant = "target_voice_reference"
        elif self.sample_id.endswith("/audio_reuse"):
            expected_variant = "full_audio_reuse"
        elif self.sample_id.endswith("/music_reference"):
            expected_variant = "music_reference"
        else:
            raise ValueError("derived MiMo H3 shadow variant is unknown")
        if self.conditioning_variant != expected_variant:
            raise ValueError("MiMo shadow conditioning variant is inconsistent")
        if self.conditioning_variant == "visual_only":
            if self.effective_subject_voices or self.audio_references:
                raise ValueError("visual-only MiMo shadow must remain Audio-free")
        elif self.conditioning_variant in {
            "target_voice_reference",
            "cross_voice_reference",
        }:
            if len(voice_references) != len(self.audio_references):
                raise ValueError("voice MiMo shadow cannot publish non-voice Audio")
        else:
            expected_role = {
                "music_reference": "music_reference",
                "full_audio_reuse": "full_audio_reuse",
            }[self.conditioning_variant]
            if (
                self.effective_subject_voices
                or len(self.audio_references) != 1
                or self.audio_references[0].role != expected_role
            ):
                raise ValueError("derived Audio-only MiMo shadow is inconsistent")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo H3 shadow record fingerprint is invalid")
        return self


class MimoH3ShadowSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_h3_shadow_summary.12"] = (
        MIMO25_SHADOW_SUMMARY_VERSION
    )
    source_mimo_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_mimo_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    materialization_failure_count: int = Field(ge=0)
    materialization_failure_code_counts: dict[str, int]
    warning_counts: dict[str, int]
    target_clip_annotation_reuse_count: int = Field(ge=0)
    recovered_voice_quality_policy_version: Literal[
        "h3_mimo25_recovered_voice_quality_v1"
    ] = RECOVERED_VOICE_POLICY_VERSION
    existing_target_voice_reference_count: int = Field(ge=0)
    recovered_voice_candidate_count: int = Field(ge=0)
    recovered_voice_accepted_count: int = Field(ge=0)
    recovered_voice_rejected_count: int = Field(ge=0)
    recovered_voice_entity_count: int = Field(ge=0)
    derived_in_pair_count: int = Field(ge=0)
    enriched_existing_in_pair_count: int = Field(ge=0)
    recovery_rejection_reason_counts: dict[str, int]
    full_audio_reuse_count: int = Field(ge=0)
    music_reference_count: int = Field(ge=0)
    music_reference_rejection_reason_counts: dict[str, int]
    music_reference_policy_version: Literal[
        "h3_mimo25_clean_music_reference_v1"
    ] = MUSIC_REFERENCE_POLICY_VERSION
    asr_text_modified: Literal[False] = False
    asr_language_modified: Literal[False] = False
    source_segments_deleted: Literal[False] = False
    production_h3_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MimoH3ShadowSummary:
        if self.sample_count != self.ready_count + self.failed_count:
            raise ValueError("MiMo H3 shadow summary counts must reconcile")
        if self.materialization_failure_count > self.failed_count:
            raise ValueError("MiMo materialization failure count exceeds failed records")
        if bool(self.materialization_failure_count) != bool(
            self.materialization_failure_code_counts
        ):
            raise ValueError("MiMo materialization failure summary is incomplete")
        if any(count <= 0 for count in self.materialization_failure_code_counts.values()):
            raise ValueError("MiMo materialization failure code counts must be positive")
        if self.recovered_voice_candidate_count != (
            self.recovered_voice_accepted_count + self.recovered_voice_rejected_count
        ):
            raise ValueError("MiMo recovered voice counts must reconcile")
        return self


def _variant(sample: FinalH3SampleV2) -> ConditioningVariant:
    return {
        "canonical": "visual_only",
        "in_pair": "target_voice_reference",
        "cross_pair": "cross_voice_reference",
    }[sample.pair_type]


def _corrected_segments(
    sample: FinalH3SampleV2,
    job: MimoClipJob,
    record: MimoRecord,
) -> tuple[list[FinalQwen3SpeechSegment], list[str]]:
    assert record.annotation is not None
    audio_decisions = {
        item.segment_id: item
        for item in record.annotation.audio_observation.segment_decisions
    }
    groundings = {
        item.segment_id: item
        for item in record.annotation.av_grounding.segment_groundings
    }
    source_by_id = {item.segment_id: item for item in job.segments}
    sample_by_id = {item.segment_id: item for item in sample.speech_segments}
    transcribed_ids = [
        item.segment_id for item in job.segments if item.asr_status == "transcribed"
    ]
    if set(sample_by_id) != set(transcribed_ids):
        raise ValueError("source H3 speech inventory differs from authoritative ASR")
    corrected: list[FinalQwen3SpeechSegment] = []
    warnings: list[str] = []
    for segment_id in transcribed_ids:
        source = source_by_id[segment_id]
        original = sample_by_id[segment_id]
        audio_decision = audio_decisions[segment_id]
        grounding = groundings[segment_id]
        if (
            original.text != source.asr_text
            or original.language != source.asr_language
            or original.start_time != source.start_time
            or original.end_time != source.end_time
            or original.source_start_sample != source.source_start_sample
            or original.source_end_sample != source.source_end_sample
            or original.source_sample_rate_hz != source.source_sample_rate_hz
            or original.speaker_cluster_id != source.source_speaker_cluster_id
        ):
            raise ValueError("source H3 speech differs from authoritative Qwen3-ASR")
        resolved = (
            audio_decision.resolution == "resolved"
            and audio_decision.primary_speaker_group is not None
        )
        group = (
            audio_decision.primary_speaker_group
            if resolved
            else f"fallback__{source.source_speaker_cluster_id}"
        )
        entity_id = (
            grounding.entity_id
            if resolved
            and grounding.binding_status == "visible_entity"
            and grounding.speech_presentation == "onscreen_spoken"
            else None
        )
        if not resolved:
            warnings.append(f"{segment_id}:acoustic_refinement_unresolved")
        corrected.append(
            FinalQwen3SpeechSegment(
                segment_id=segment_id,
                speaker_cluster_id=group,
                entity_id=entity_id,
                entity_occurrence_id=(
                    None if entity_id is None else f"{sample.clip_uid}/{entity_id}"
                ),
                source_start_sample=source.source_start_sample,
                source_end_sample=source.source_end_sample,
                source_sample_rate_hz=source.source_sample_rate_hz,
                start_time=source.start_time,
                end_time=source.end_time,
                text=source.asr_text or "",
                language=source.asr_language,
            )
        )
    return corrected, warnings


def _speaker_ids(segments: Sequence[FinalQwen3SpeechSegment]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in segments:
        if segment.speaker_cluster_id not in result:
            result[segment.speaker_cluster_id] = f"S{len(result) + 1}"
    return result


def _audio_facts(
    *,
    sample: FinalH3SampleV2,
    corrected: list[FinalQwen3SpeechSegment],
    record: MimoRecord,
    contract: object,
) -> RecaptionAudioFacts:
    assert record.annotation is not None
    typed_contract = RecaptionReferenceContract.model_validate(contract)
    entity_subject = {
        item.entity_id: item.subject_label
        for item in typed_contract.subjects
        if item.kind == "entity" and item.entity_id is not None
    }
    speaker_ids = _speaker_ids(corrected)
    delivery_by_segment = {
        item.segment_id: item.delivery_style
        for item in record.annotation.audio_observation.segment_decisions
    }
    speech = []
    for segment in corrected:
        language = segment.language or "Unknown"
        speech.append(
            RecaptionSpeechFact(
                fact_id=segment.segment_id,
                segment_id=segment.segment_id,
                speaker_cluster_id=segment.speaker_cluster_id,
                entity_id=segment.entity_id,
                entity_subject_label=(
                    None
                    if segment.entity_id is None
                    else entity_subject.get(segment.entity_id)
                ),
                speaker_id=speaker_ids[segment.speaker_cluster_id],
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=segment.text,
                language=language,
                delivery=delivery_by_segment[segment.segment_id],
                locked_dialogue_block=f"<d>[{language}] {segment.text}</d>",
            )
        )
    events = [
        RecaptionNonSpeechFact(
            fact_id=item.event_id,
            start_time=item.approximate_start_time,
            end_time=item.approximate_end_time,
            category=item.category,
            description=item.description,
            source_attribution=item.source_grounding,
            provenance="mimo25.audio_semantics.temporal_non_speech_events",
        )
        for item in record.annotation.audio_semantics.temporal_non_speech_events
    ]
    semantics = record.annotation.audio_semantics
    return RecaptionAudioFacts(
        speech=speech,
        non_speech_events=events,
        overall_soundscape_hint=semantics.overall_soundscape,
        non_diegetic_music_hint=semantics.non_diegetic_music,
        audio_grounding_complete=True,
        provenance={
            "speech": "qwen3_asr_segments_exact",
            "speaker_structure": "mimo25_shadow_annotation",
            "audio_semantics": "mimo25_shadow_annotation",
            "source_sample_id": sample.sample_id,
        },
    )


def _render_mimo_speech_clause(
    speech: RecaptionSpeechFact,
    decisions: dict[str, MimoSegmentDecision],
    contract: RecaptionReferenceContract,
    *,
    include_audio_reference: bool,
) -> str:
    presentation = decisions[speech.segment_id].speech_presentation
    return render_speech_presentation_clause(
        speech=speech,
        base_clause=_render_locked_speech(
            speech,
            contract,
            include_audio_reference=include_audio_reference,
        ),
        presentation=presentation,
    )


def _contract_with_voice_profiles(
    contract: RecaptionReferenceContract,
    *,
    corrected: Sequence[FinalQwen3SpeechSegment],
    record: MimoRecord,
) -> RecaptionReferenceContract:
    assert record.annotation is not None
    speaker_by_group = _speaker_ids(corrected)
    profile_by_speaker = {
        speaker_by_group[profile.speaker_group]: profile.voice_characteristics
        for profile in record.annotation.speaker_voice_profiles
        if profile.speaker_group in speaker_by_group
    }
    values = contract.model_dump(mode="python")
    for audio in values["audios"]:
        speaker_id = audio.get("speaker_id")
        if speaker_id in profile_by_speaker:
            audio["voice_characteristics"] = profile_by_speaker[speaker_id]
    return RecaptionReferenceContract.model_validate(values)


def _render_timeline_parts(
    shot: MimoH3Shot,
    *,
    visual_block_by_id: dict[str, str],
    audio_event_by_id: dict[str, str],
) -> str:
    rendered: list[str] = []
    for part in shot.timeline_parts:
        if isinstance(part, MimoH3VisualPart):
            rendered.append(visual_block_by_id[part.block_id].strip())
        elif isinstance(part, MimoH3SpeechPart):
            rendered.append(f"[[{part.segment_id}]]")
        elif isinstance(part, MimoH3AudioEventPart):
            rendered.append(audio_event_by_id[part.event_id])
        else:  # pragma: no cover - discriminated schema makes this unreachable
            raise TypeError(f"unsupported MiMo timeline part: {type(part).__name__}")
    return " ".join(rendered)


def _join_picture_labels(labels: Sequence[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _render_subject_definition(
    draft: MimoSubjectDefinitionDraft,
    contract: RecaptionSubjectContract,
) -> str:
    if draft.subject_label != contract.subject_label:
        raise ValueError("MiMo Subject definition differs from frozen Subject contract")
    description = draft.description.strip()
    if description.casefold().startswith("is "):
        description = description[3:].lstrip()
    description = description.rstrip(". ")
    pictures = _join_picture_labels(contract.source_picture_labels)
    connector = {
        "entity": "shown in",
        "attribute": "with its visual detail sourced from",
        "background": "depicted in",
    }[contract.kind]
    return f"{contract.subject_label} is {description}, {connector} {pictures}."


def _materialize_sample(
    sample: FinalH3SampleV2,
    job: MimoClipJob,
    record: MimoRecord,
    *,
    conditioning_variant: ConditioningVariant | None = None,
    extra_audio_contract: RecaptionAudioContract | None = None,
) -> tuple[list[FinalQwen3SpeechSegment], str, list[str]]:
    assert record.annotation is not None
    projected_sample = project_mimo_h3_sample_references(
        sample,
        reference_images=job.reference_images,
        reference_selection=job.reference_selection,
    )
    corrected, warnings = _corrected_segments(projected_sample, job, record)
    corrected_payload = projected_sample.model_dump(mode="python")
    corrected_payload["speech_segments"] = [
        item.model_dump(mode="python") for item in corrected
    ]
    corrected_sample = FinalH3SampleV2.model_validate(corrected_payload)
    variant = conditioning_variant or _variant(corrected_sample)
    contract = _contract_with_voice_profiles(
        build_reference_contract(
            corrected_sample,
            "visual_only" if extra_audio_contract is not None else variant,
        ),
        corrected=corrected,
        record=record,
    )
    if extra_audio_contract is not None:
        contract = RecaptionReferenceContract.model_validate(
            {
                **contract.model_dump(mode="json"),
                "audios": [extra_audio_contract.model_dump(mode="json")],
            }
        )
    facts = _audio_facts(
        sample=corrected_sample,
        corrected=corrected,
        record=record,
        contract=contract,
    )
    case = Qwen38RecaptionManifestCase(
        sample_id=sample.sample_id,
        conditioning_variant=variant,
        note="MiMo-V2.5 target-level AV shadow annotation",
    )
    request = Qwen38RecaptionRequest(
        sample=corrected_sample,
        case=case,
        reference_contract=contract,
        audio_facts=facts,
        request_fingerprint=record.request_fingerprint,
    )
    semantics = record.annotation.audio_observation.audio_semantics
    audio_event_by_id = {
        item.event_id: item.description for item in semantics.temporal_non_speech_events
    }
    prefix = {
        "visual_only": "[reference generation]",
        "target_voice_reference": "[reference generation + audio reference]",
        "cross_voice_reference": "[reference generation + audio reference]",
        "music_reference": "[reference generation + audio reference]",
        "full_audio_reuse": "[reference generation + audio reuse]",
    }[variant]
    music = (
        semantics.non_diegetic_music
        if semantics.non_diegetic_music_status == "present"
        else "N/A"
    )
    if semantics.overall_soundscape_status == "present":
        soundscape = semantics.overall_soundscape
    elif semantics.overall_soundscape_status == "absent":
        soundscape = "N/A"
    else:
        raise ValueError(
            "unknown MiMo soundscape cannot be materialized as confirmed silence"
        )
    visual_block_by_id = {
        block.block_id: block.text
        for shot in record.annotation.visual_observation.shots
        for block in shot.visual_blocks
    }
    draft = Qwen38H3DraftResponse(
        subject_definitions=[
            _render_subject_definition(item, subject)
            for item, subject in zip(
                record.annotation.h3_semantics.subject_definitions,
                job.reference_subjects,
                strict=True,
            )
        ],
        summary=f"{prefix} {record.annotation.h3_semantics.summary}",
        retention_analysis=[
            item.render()
            for item in record.annotation.h3_semantics.visual_retention_analysis
        ],
        shots=[
            Qwen38DraftShot(
                shot_index=item.shot_index,
                start_time=(
                    None
                    if item.shot_index == 1 and item.start_time == 0
                    else item.start_time
                ),
                description_template=_render_timeline_parts(
                    item,
                    visual_block_by_id=visual_block_by_id,
                    audio_event_by_id=audio_event_by_id,
                ),
            )
            for item in record.annotation.h3_projection.shots
        ],
        overall_soundscape=soundscape,
        non_diegetic_music=music,
        audio_fact_audit=[
            AudioFactAuditItem(fact_id=item.fact_id, action="preserved")
            for item in facts.non_speech_events
        ],
    )
    decisions = {
        item.segment_id: item
        for item in record.annotation.av_grounding.segment_groundings
    }
    shot_by_segment = {
        part.segment_id: shot.shot_index
        for shot in record.annotation.h3_projection.shots
        for part in shot.timeline_parts
        if isinstance(part, MimoH3SpeechPart)
    }
    cited_speakers: set[tuple[int, str]] = set()

    def render_speech(speech: RecaptionSpeechFact, _base_clause: str) -> str:
        key = (shot_by_segment[speech.segment_id], speech.speaker_id)
        include_audio_reference = key not in cited_speakers
        cited_speakers.add(key)
        return _render_mimo_speech_clause(
            speech,
            decisions,
            contract,
            include_audio_reference=include_audio_reference,
        )

    structured = materialize_h3_draft(
        draft,
        request,
        speech_clause_transform=render_speech,
    )
    if variant == "music_reference":
        assert extra_audio_contract is not None
        assert extra_audio_contract.music_characteristics is not None
        structured = structured.model_copy(
            update={
                "non_diegetic_music": (
                    "A newly generated audience-only score follows <Audio 1>'s "
                    f"{extra_audio_contract.music_characteristics} without directly "
                    "reusing the source signal."
                )
            }
        )
    if "[[audio_event:" in structured.detailed_description:
        raise ValueError("MiMo Audio-event placeholder survived materialization")
    issues, validation_warnings = validate_h3_response(structured, request)
    if issues:
        raise MimoH3MaterializationContractError(issues)
    warnings.extend(validation_warnings)
    if record.annotation.warnings:
        warnings.extend(
            f"{item.segment_id}:{item.code}" for item in record.annotation.warnings
        )
    return corrected, render_h3_prompt(structured), sorted(set(warnings))


def _record(values: dict[str, object]) -> MimoH3ShadowRecord:
    return MimoH3ShadowRecord(
        **values,
        record_fingerprint=_sha256_text(_compact_json(values)),
    )


def _materialization_failure_values(
    base: dict[str, object],
    error: MimoH3MaterializationContractError,
) -> dict[str, object]:
    return {
        **base,
        "status": "failed",
        "corrected_speech_segments": [],
        "effective_subject_voices": [],
        "audio_references": [],
        "recovered_voice_references": [],
        "rendered_h3_prompt": None,
        "warnings": [],
        "failure_reason": error.failure_reason(),
    }


def _sample_with_voices(
    sample: FinalH3SampleV2,
    *,
    voices: Sequence[FinalSubjectVoice],
    sample_id: str | None = None,
    pair_id: str | None = None,
    pair_type: Literal["canonical", "in_pair", "cross_pair"] | None = None,
) -> FinalH3SampleV2:
    values = sample.model_dump(mode="python")
    values["subject_voices"] = [item.model_dump(mode="python") for item in voices]
    if sample_id is not None:
        values["sample_id"] = sample_id
    if pair_id is not None:
        values["pair_id"] = pair_id
    if pair_type is not None:
        values["pair_type"] = pair_type
    return FinalH3SampleV2.model_validate(values)


def _subject_index_by_entity(job: MimoClipJob) -> dict[str, int]:
    return {
        item.entity_id: item.subject_index
        for item in job.reference_subjects
        if item.kind == "entity" and item.entity_id is not None
    }


def _audio_references(
    *,
    voices: Sequence[FinalSubjectVoice],
    corrected: Sequence[FinalQwen3SpeechSegment],
    recovered: Sequence[MimoRecoveredVoiceReference],
) -> list[MimoShadowAudioReference]:
    speaker_ids = _speaker_ids(corrected)
    speaker_by_entity: dict[str, str] = {}
    for speech in corrected:
        if speech.entity_id is not None and speech.entity_id not in speaker_by_entity:
            speaker_by_entity[speech.entity_id] = speaker_ids[speech.speaker_cluster_id]
    recovered_by_entity = {
        item.entity_id: item for item in recovered if item.status == "selected"
    }
    result: list[MimoShadowAudioReference] = []
    for index, voice in enumerate(
        sorted(voices, key=lambda item: (item.subject_index, item.entity_id)),
        start=1,
    ):
        recovered_voice = recovered_by_entity.get(voice.entity_id)
        result.append(
            MimoShadowAudioReference(
                audio_index=index,
                role="voice_reference",
                subject_index=voice.subject_index,
                entity_id=voice.entity_id,
                speaker_id=speaker_by_entity.get(voice.entity_id),
                source_type=(
                    "cross_donor"
                    if voice.voice_source == "cross_donor"
                    else (
                        "mimo_recovered_target_voice"
                        if recovered_voice is not None
                        else "existing_target_voice"
                    )
                ),
                audio_path=voice.voice_reference_path,
                audio_sha256=voice.voice_reference_sha256,
                source_segment_id=(
                    None
                    if recovered_voice is None
                    else recovered_voice.source_segment_id
                ),
            )
        )
    return result


def _full_audio_reuse_reference(job: MimoClipJob) -> MimoShadowAudioReference:
    source = Path(job.target_full_audio_path).expanduser().resolve(strict=True)
    return MimoShadowAudioReference(
        audio_index=1,
        role="full_audio_reuse",
        source_type="canonical_full_audio",
        audio_path=str(source),
        audio_sha256=job.target_full_audio_sha256,
        interval_provenance="canonical_full_audio_exact_v1",
    )


def _extra_audio_contract(
    reference: MimoShadowAudioReference,
    *,
    render_path: Path,
) -> RecaptionAudioContract:
    if reference.role == "full_audio_reuse":
        return RecaptionAudioContract(
            audio_index=1,
            audio_label="<Audio 1>",
            kind="full_audio_reuse",
            path=str(render_path),
            sha256=reference.audio_sha256,
            retention_marker="fully_copy",
        )
    if reference.role != "music_reference":  # pragma: no cover - internal contract
        raise ValueError("extra shadow Audio must be music or full reuse")
    return RecaptionAudioContract(
        audio_index=1,
        audio_label="<Audio 1>",
        kind="music_reference",
        path=str(render_path),
        sha256=reference.audio_sha256,
        music_characteristics=reference.music_description,
        retention_marker="reference",
    )


def _probe_canonical_audio(
    *,
    job: MimoClipJob,
    audio_backend: AudioMediaBackend,
) -> None:
    path = Path(job.target_full_audio_path).expanduser().resolve(strict=True)
    probe = audio_backend.probe_audio_file(path)
    if (
        probe.sample_rate_hz != MUSIC_REFERENCE_SAMPLE_RATE_HZ
        or probe.channels != MUSIC_REFERENCE_CHANNELS
        or "flac" not in probe.format_name.lower()
        or abs(probe.duration_seconds - job.target_duration_seconds)
        > CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS
    ):
        raise ValueError("MiMo Audio conditioning source must be canonical 32 kHz stereo FLAC")


def _interval_overlaps_speech(
    start_time: float,
    end_time: float,
    job: MimoClipJob,
) -> bool:
    return any(
        start_time < segment.end_time and end_time > segment.start_time
        for segment in job.segments
    )


def _music_candidate_rank(
    item: tuple[MimoAudioEvent, int, int, float, float],
) -> tuple[float, float, float, float, str]:
    event, _start_sample, _end_sample, rms_dbfs, clipping_ratio = item
    duration = event.approximate_end_time - event.approximate_start_time
    return (
        -duration,
        clipping_ratio,
        -rms_dbfs,
        event.approximate_start_time,
        event.event_id,
    )


def _select_music_reference(
    *,
    job: MimoClipJob,
    record: MimoRecord,
    temporary_root: Path,
    final_root: Path,
    audio_backend: AudioMediaBackend,
    analyzer: RecoveredVoiceAudioAnalyzer,
) -> tuple[MimoShadowAudioReference | None, Path | None, list[str]]:
    assert record.annotation is not None
    semantics = record.annotation.audio_semantics
    if semantics.non_diegetic_music_status != "present":
        return None, None, []
    # TODO: A future policy may merge contiguous compatible spans from one cue.
    # V1 intentionally ranks individual model events without changing boundaries.
    events = [
        item
        for item in semantics.temporal_non_speech_events
        if item.category == "non_diegetic_music"
    ]
    if not events:
        return None, None, []
    source = Path(job.target_full_audio_path).expanduser().resolve(strict=True)
    analysis = analyzer.load(source)
    if (
        analysis.probe.sample_rate_hz != MUSIC_REFERENCE_SAMPLE_RATE_HZ
        or analysis.probe.channels != MUSIC_REFERENCE_CHANNELS
        or "flac" not in analysis.probe.format_name.lower()
        or analysis.mono_pcm16.ndim != 1
        or analysis.mono_pcm16.dtype != np.dtype("int16")
        or analysis.mono_pcm16.size != analysis.probe.frame_count
    ):
        raise ValueError("MiMo music source must be canonical 32 kHz stereo FLAC")
    rejection_reasons: list[str] = []
    eligible: list[tuple[MimoAudioEvent, int, int, float, float]] = []
    for event in events:
        duration = event.approximate_end_time - event.approximate_start_time
        start_sample = round(event.approximate_start_time * MUSIC_REFERENCE_SAMPLE_RATE_HZ)
        end_sample = round(event.approximate_end_time * MUSIC_REFERENCE_SAMPLE_RATE_HZ)
        reasons: list[str] = []
        if duration < MUSIC_REFERENCE_MINIMUM_DURATION_SECONDS:
            reasons.append("music_reference_too_short")
        if _interval_overlaps_speech(
            event.approximate_start_time,
            event.approximate_end_time,
            job,
        ):
            reasons.append("music_reference_overlaps_speech")
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:bpm|beats?\s+per\s+minute)\b",
            event.description,
            flags=re.IGNORECASE,
        ):
            reasons.append("music_reference_unmeasured_exact_tempo")
        if not 0 <= start_sample < end_sample <= analysis.probe.frame_count:
            reasons.append("music_reference_invalid_sample_range")
        if reasons:
            rejection_reasons.extend(reasons)
            continue
        samples = analysis.mono_pcm16[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
        rms_dbfs = 20.0 * math.log10(rms / 32768.0) if rms > 0 else -math.inf
        clipping_ratio = float(
            np.mean(np.abs(samples.astype(np.int32)) >= np.iinfo(np.int16).max)
        )
        if not math.isfinite(rms_dbfs) or rms_dbfs < MUSIC_REFERENCE_MINIMUM_RMS_DBFS:
            rejection_reasons.append("music_reference_rms_unusable")
            continue
        if clipping_ratio > MUSIC_REFERENCE_MAXIMUM_CLIPPING_RATIO:
            rejection_reasons.append("music_reference_clipping_excessive")
            continue
        eligible.append((event, start_sample, end_sample, rms_dbfs, clipping_ratio))
    if not eligible:
        return None, None, rejection_reasons
    event, start_sample, end_sample, rms_dbfs, clipping_ratio = min(
        eligible,
        key=_music_candidate_rank,
    )
    relative = Path("music_references") / job.clip_uid / "music_reference.flac"
    temporary_path = temporary_root / relative
    published_path = final_root / relative
    audio_backend.extract_voice_reference(
        clip_uid=job.clip_uid,
        entity_id="music_reference",
        full_audio_path=source,
        start_time=event.approximate_start_time,
        end_time=event.approximate_end_time,
        destination=temporary_path,
        sample_rate_hz=MUSIC_REFERENCE_SAMPLE_RATE_HZ,
        channels=MUSIC_REFERENCE_CHANNELS,
        output_format="flac",
        source_audio_path=source,
        source_start_sample=start_sample,
        source_end_sample=end_sample,
    )
    if not temporary_path.is_file():
        raise ValueError("MiMo music reference extraction produced no file")
    output_probe = audio_backend.probe_audio_file(temporary_path)
    if (
        output_probe.sample_rate_hz != MUSIC_REFERENCE_SAMPLE_RATE_HZ
        or output_probe.channels != MUSIC_REFERENCE_CHANNELS
        or "flac" not in output_probe.format_name.lower()
        or output_probe.frame_count != end_sample - start_sample
    ):
        raise ValueError("MiMo music reference output is not exact 32 kHz stereo FLAC")
    return (
        MimoShadowAudioReference(
            audio_index=1,
            role="music_reference",
            source_type="mimo_non_diegetic_music_event",
            audio_path=str(published_path.resolve(strict=False)),
            audio_sha256=_sha256_file(temporary_path),
            source_event_id=event.event_id,
            source_start_time=event.approximate_start_time,
            source_end_time=event.approximate_end_time,
            source_start_sample=start_sample,
            source_end_sample=end_sample,
            interval_provenance="mimo_approximate_event_times_rounded_to_32000_v1",
            music_description=event.description,
            rms_dbfs=rms_dbfs,
            clipping_ratio=clipping_ratio,
        ),
        temporary_path,
        rejection_reasons,
    )


def _validate_job_media_integrity(job: MimoClipJob) -> None:
    media = (
        (
            Path(job.target_video_path),
            job.target_video_sha256,
            "MiMo target video changed after AV annotation",
        ),
        (
            Path(job.target_full_audio_path),
            job.target_full_audio_sha256,
            "MiMo target full audio changed after AV annotation",
        ),
        *(
            (
                Path(reference.image_artifact_path),
                reference.image_sha256,
                "MiMo frozen reference image changed after AV annotation",
            )
            for reference in job.reference_images
        ),
    )
    for path, expected_sha256, message in media:
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise ValueError(message)


def _validate_materializer_provenance(
    *,
    inventory: MimoInventory,
    records: Sequence[MimoRecord],
    source_samples: Sequence[FinalH3SampleV2],
) -> None:
    job_by_clip = {item.clip_uid: item for item in inventory.jobs}
    sample_ids_by_clip: dict[str, list[str]] = {
        clip_uid: [] for clip_uid in job_by_clip
    }
    for sample in source_samples:
        if sample.clip_uid in sample_ids_by_clip:
            sample_ids_by_clip[sample.clip_uid].append(sample.sample_id)
    for record in records:
        job = job_by_clip.get(record.clip_uid)
        if record.inventory_fingerprint != inventory.inventory_fingerprint:
            raise ValueError("MiMo record inventory fingerprint mismatch")
        if job is None or record.request_fingerprint != job.request_fingerprint:
            raise ValueError("MiMo record request fingerprint mismatch")
    for job in inventory.jobs:
        if sorted(sample_ids_by_clip[job.clip_uid]) != sorted(job.source_h3_sample_ids):
            raise ValueError("MiMo source H3 sample IDs changed after AV annotation")
        _validate_job_media_integrity(job)


def materialize_mimo25_h3_shadow(
    *,
    mimo_root: Path,
    source_h3_root: Path,
    output_root: Path,
    overwrite: bool = False,
    enable_full_audio_reuse: bool = False,
    enable_music_reference: bool = False,
    audio_backend: AudioMediaBackend | None = None,
    recovered_voice_analyzer: RecoveredVoiceAudioAnalyzer | None = None,
) -> MimoH3ShadowSummary:
    source_mimo = mimo_root.expanduser().resolve(strict=True)
    source_h3 = source_h3_root.expanduser().resolve(strict=True)
    source_h3_samples_path = source_h3 / "samples.jsonl"
    inventory = MimoInventory.model_validate_json(
        (source_mimo / "inventory.json").read_text(encoding="utf-8")
    )
    if _sha256_file(source_h3_samples_path) != inventory.source_h3_samples_sha256:
        raise ValueError("MiMo source H3 inventory changed after AV annotation")
    mimo_records = [
        MimoRecord.model_validate(row)
        for row in _read_jsonl(source_mimo / "records.jsonl")
    ]
    all_source_samples = [
        FinalH3SampleV2.model_validate(row)
        for row in _read_jsonl(source_h3_samples_path)
    ]
    job_by_clip = {item.clip_uid: item for item in inventory.jobs}
    source_samples = [
        item for item in all_source_samples if item.clip_uid in job_by_clip
    ]
    record_by_clip = {item.clip_uid: item for item in mimo_records}
    if len(record_by_clip) != len(mimo_records):
        raise ValueError("MiMo records contain duplicate target clips")
    if set(record_by_clip) != set(job_by_clip):
        raise ValueError("MiMo records do not exactly cover the selected inventory")
    _validate_materializer_provenance(
        inventory=inventory,
        records=mimo_records,
        source_samples=all_source_samples,
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    records: list[MimoH3ShadowRecord] = []
    warning_counts: Counter[str] = Counter()
    materialization_failure_count = 0
    materialization_failure_code_counts: Counter[str] = Counter()
    recovery_records: list[MimoRecoveredVoiceReference] = []
    temporary_recovered_by_clip: dict[str, list[FinalSubjectVoice]] = {}
    published_recovered_by_clip: dict[str, list[FinalSubjectVoice]] = {}
    music_rejection_counts: Counter[str] = Counter()
    active_audio_backend = audio_backend or FFmpegAudioMediaBackend()
    active_analyzer = recovered_voice_analyzer or FFmpegRecoveredVoiceAudioAnalyzer()
    existing_target_by_clip: dict[str, list[FinalSubjectVoice]] = {}
    canonical_by_clip: dict[str, FinalH3SampleV2] = {}
    in_pair_by_clip: dict[str, FinalH3SampleV2] = {}
    for sample in source_samples:
        if sample.pair_type == "canonical":
            if sample.clip_uid in canonical_by_clip:
                raise ValueError("MiMo source H3 contains duplicate canonical samples")
            canonical_by_clip[sample.clip_uid] = sample
        elif sample.pair_type == "in_pair":
            if sample.clip_uid in in_pair_by_clip:
                raise ValueError("MiMo source H3 contains duplicate in-pair samples")
            in_pair_by_clip[sample.clip_uid] = sample
            existing_target_by_clip[sample.clip_uid] = [
                item for item in sample.subject_voices if item.voice_source == "target"
            ]
    try:
        for job in inventory.jobs:
            mimo_record = record_by_clip[job.clip_uid]
            canonical = canonical_by_clip.get(job.clip_uid)
            if canonical is None:
                raise ValueError("MiMo source H3 canonical sample is missing")
            existing = existing_target_by_clip.get(job.clip_uid, [])
            existing_ids = {item.entity_id for item in existing}
            if len(existing_ids) != len(existing):
                raise ValueError("MiMo source target voices contain duplicate entities")
            capacity = min(
                3 - len(existing),
                12 - len(job.reference_images) - len(existing),
            )
            if mimo_record.status != "ready":
                continue
            clip_records, temporary_voices, published_voices = (
                recover_mimo_target_voices(
                    job=job,
                    record=mimo_record,
                    subject_index_by_entity=_subject_index_by_entity(job),
                    existing_entity_ids=existing_ids,
                    reference_capacity=max(0, capacity),
                    temporary_root=temporary,
                    final_root=destination,
                    audio_backend=active_audio_backend,
                    analyzer=active_analyzer,
                )
            )
            recovery_records.extend(clip_records)
            temporary_recovered_by_clip[job.clip_uid] = temporary_voices
            published_recovered_by_clip[job.clip_uid] = published_voices

        for sample in source_samples:
            source_hash = _sha256_text(_compact_json(sample.model_dump(mode="json")))
            job = job_by_clip.get(sample.clip_uid)
            mimo_record = record_by_clip.get(sample.clip_uid)
            published_recovered = published_recovered_by_clip.get(sample.clip_uid, [])
            temporary_recovered = temporary_recovered_by_clip.get(sample.clip_uid, [])
            base = {
                "schema_version": MIMO25_SHADOW_RECORD_VERSION,
                "sample_id": sample.sample_id,
                "source_h3_sample_id": sample.sample_id,
                "clip_uid": sample.clip_uid,
                "pair_type": sample.pair_type,
                "derived_from_pair_type": None,
                "conditioning_variant": _variant(sample),
                "source_mimo_record_fingerprint": (
                    "0" * 64 if mimo_record is None else mimo_record.record_fingerprint
                ),
                "source_h3_sample_sha256": source_hash,
                "materializer_version": MIMO25_MATERIALIZER_VERSION,
            }
            if job is None or mimo_record is None or mimo_record.status != "ready":
                values = {
                    **base,
                    "status": "failed",
                    "corrected_speech_segments": [],
                    "effective_subject_voices": [],
                    "audio_references": [],
                    "recovered_voice_references": [],
                    "rendered_h3_prompt": None,
                    "warnings": [],
                    "failure_reason": "ready MiMo target annotation is unavailable",
                }
            else:
                try:
                    render_sample = sample
                    published_voices = list(sample.subject_voices)
                    sample_recovery: list[MimoRecoveredVoiceReference] = []
                    if sample.pair_type == "in_pair" and published_recovered:
                        temporary_effective = sorted(
                            [*sample.subject_voices, *temporary_recovered],
                            key=lambda item: (item.subject_index, item.entity_id),
                        )
                        render_sample = _sample_with_voices(
                            sample,
                            voices=temporary_effective,
                        )
                        published_voices = sorted(
                            [*sample.subject_voices, *published_recovered],
                            key=lambda item: (item.subject_index, item.entity_id),
                        )
                        sample_recovery = [
                            item
                            for item in recovery_records
                            if item.clip_uid == sample.clip_uid
                            and item.status == "selected"
                        ]
                    corrected, rendered, warnings = _materialize_sample(
                        render_sample, job, mimo_record
                    )
                    warning_counts.update(warnings)
                    values = {
                        **base,
                        "status": "ready",
                        "corrected_speech_segments": [
                            item.model_dump(mode="json") for item in corrected
                        ],
                        "effective_subject_voices": [
                            item.model_dump(mode="json") for item in published_voices
                        ],
                        "audio_references": [
                            item.model_dump(mode="json")
                            for item in _audio_references(
                                voices=published_voices,
                                corrected=corrected,
                                recovered=sample_recovery,
                            )
                        ],
                        "recovered_voice_references": [
                            item.model_dump(mode="json") for item in sample_recovery
                        ],
                        "rendered_h3_prompt": rendered,
                        "warnings": warnings,
                        "failure_reason": None,
                    }
                except MimoH3MaterializationContractError as exc:
                    materialization_failure_count += 1
                    materialization_failure_code_counts.update(
                        item.code for item in exc.issues
                    )
                    values = _materialization_failure_values(base, exc)
            records.append(_record(values))

        for clip_uid, recovered_voices in sorted(published_recovered_by_clip.items()):
            if not recovered_voices or clip_uid in in_pair_by_clip:
                continue
            canonical = canonical_by_clip[clip_uid]
            job = job_by_clip[clip_uid]
            mimo_record = record_by_clip[clip_uid]
            temporary_voices = temporary_recovered_by_clip[clip_uid]
            render_sample = _sample_with_voices(
                canonical,
                voices=temporary_voices,
                sample_id=f"{canonical.sample_id}/mimo_recovered_in_pair",
                pair_id=f"in_pair/{clip_uid}/mimo_recovered",
                pair_type="in_pair",
            )
            published_sample = _sample_with_voices(
                canonical,
                voices=recovered_voices,
                sample_id=render_sample.sample_id,
                pair_id=render_sample.pair_id,
                pair_type="in_pair",
            )
            selected_recovery = [
                item
                for item in recovery_records
                if item.clip_uid == clip_uid and item.status == "selected"
            ]
            base = {
                "schema_version": MIMO25_SHADOW_RECORD_VERSION,
                "sample_id": published_sample.sample_id,
                "source_h3_sample_id": canonical.sample_id,
                "clip_uid": clip_uid,
                "pair_type": "in_pair",
                "derived_from_pair_type": "canonical",
                "conditioning_variant": "target_voice_reference",
                "source_mimo_record_fingerprint": mimo_record.record_fingerprint,
                "source_h3_sample_sha256": _sha256_text(
                    _compact_json(canonical.model_dump(mode="json"))
                ),
                "materializer_version": MIMO25_MATERIALIZER_VERSION,
            }
            try:
                corrected, rendered, warnings = _materialize_sample(
                    render_sample, job, mimo_record
                )
            except MimoH3MaterializationContractError as exc:
                materialization_failure_count += 1
                materialization_failure_code_counts.update(
                    item.code for item in exc.issues
                )
                records.append(_record(_materialization_failure_values(base, exc)))
                continue
            warning_counts.update(warnings)
            records.append(
                _record(
                    {
                        **base,
                        "status": "ready",
                        "corrected_speech_segments": [
                            item.model_dump(mode="json") for item in corrected
                        ],
                        "effective_subject_voices": [
                            item.model_dump(mode="json") for item in recovered_voices
                        ],
                        "audio_references": [
                            item.model_dump(mode="json")
                            for item in _audio_references(
                                voices=recovered_voices,
                                corrected=corrected,
                                recovered=selected_recovery,
                            )
                        ],
                        "recovered_voice_references": [
                            item.model_dump(mode="json") for item in selected_recovery
                        ],
                        "rendered_h3_prompt": rendered,
                        "warnings": warnings,
                        "failure_reason": None,
                    }
                )
            )

        for clip_uid in sorted(canonical_by_clip):
            canonical = canonical_by_clip[clip_uid]
            job = job_by_clip[clip_uid]
            mimo_record = record_by_clip[clip_uid]
            if mimo_record.status != "ready":
                continue
            source_hash = _sha256_text(
                _compact_json(canonical.model_dump(mode="json"))
            )
            if enable_full_audio_reuse:
                if len(job.reference_images) + 1 > 12:
                    warning_counts["full_audio_reuse_reference_limit"] += 1
                else:
                    _probe_canonical_audio(job=job, audio_backend=active_audio_backend)
                    audio_reference = _full_audio_reuse_reference(job)
                    base = {
                        "schema_version": MIMO25_SHADOW_RECORD_VERSION,
                        "sample_id": f"{clip_uid}/audio_reuse",
                        "source_h3_sample_id": canonical.sample_id,
                        "clip_uid": clip_uid,
                        "pair_type": "canonical",
                        "derived_from_pair_type": "canonical",
                        "conditioning_variant": "full_audio_reuse",
                        "source_mimo_record_fingerprint": (
                            mimo_record.record_fingerprint
                        ),
                        "source_h3_sample_sha256": source_hash,
                        "materializer_version": MIMO25_MATERIALIZER_VERSION,
                    }
                    try:
                        corrected, rendered, warnings = _materialize_sample(
                            canonical,
                            job,
                            mimo_record,
                            conditioning_variant="full_audio_reuse",
                        )
                    except MimoH3MaterializationContractError as exc:
                        materialization_failure_count += 1
                        materialization_failure_code_counts.update(
                            item.code for item in exc.issues
                        )
                        records.append(
                            _record(_materialization_failure_values(base, exc))
                        )
                    else:
                        warning_counts.update(warnings)
                        records.append(
                            _record(
                                {
                                    **base,
                                    "status": "ready",
                                    "corrected_speech_segments": [
                                        item.model_dump(mode="json")
                                        for item in corrected
                                    ],
                                    "effective_subject_voices": [],
                                    "audio_references": [
                                        audio_reference.model_dump(mode="json")
                                    ],
                                    "recovered_voice_references": [],
                                    "rendered_h3_prompt": rendered,
                                    "warnings": warnings,
                                    "failure_reason": None,
                                }
                            )
                        )
            if enable_music_reference:
                if len(job.reference_images) + 1 > 12:
                    music_rejection_counts["music_reference_limit"] += 1
                    continue
                reference, temporary_path, rejections = _select_music_reference(
                    job=job,
                    record=mimo_record,
                    temporary_root=temporary,
                    final_root=destination,
                    audio_backend=active_audio_backend,
                    analyzer=active_analyzer,
                )
                music_rejection_counts.update(rejections)
                if reference is None or temporary_path is None:
                    continue
                base = {
                    "schema_version": MIMO25_SHADOW_RECORD_VERSION,
                    "sample_id": f"{clip_uid}/music_reference",
                    "source_h3_sample_id": canonical.sample_id,
                    "clip_uid": clip_uid,
                    "pair_type": "canonical",
                    "derived_from_pair_type": "canonical",
                    "conditioning_variant": "music_reference",
                    "source_mimo_record_fingerprint": mimo_record.record_fingerprint,
                    "source_h3_sample_sha256": source_hash,
                    "materializer_version": MIMO25_MATERIALIZER_VERSION,
                }
                try:
                    corrected, rendered, warnings = _materialize_sample(
                        canonical,
                        job,
                        mimo_record,
                        conditioning_variant="music_reference",
                        extra_audio_contract=_extra_audio_contract(
                            reference,
                            render_path=temporary_path,
                        ),
                    )
                except MimoH3MaterializationContractError as exc:
                    materialization_failure_count += 1
                    materialization_failure_code_counts.update(
                        item.code for item in exc.issues
                    )
                    records.append(_record(_materialization_failure_values(base, exc)))
                    continue
                warning_counts.update(warnings)
                records.append(
                    _record(
                        {
                            **base,
                            "status": "ready",
                            "corrected_speech_segments": [
                                item.model_dump(mode="json") for item in corrected
                            ],
                            "effective_subject_voices": [],
                            "audio_references": [reference.model_dump(mode="json")],
                            "recovered_voice_references": [],
                            "rendered_h3_prompt": rendered,
                            "warnings": warnings,
                            "failure_reason": None,
                        }
                    )
                )
        records.sort(key=lambda item: (item.clip_uid, item.sample_id))
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "recovered_voice_references.jsonl", recovery_records)
        selected_recovery = [
            item for item in recovery_records if item.status == "selected"
        ]
        rejection_counts = Counter(
            code
            for item in recovery_records
            if item.status == "rejected"
            for code in item.reason_codes
        )
        summary = MimoH3ShadowSummary(
            source_mimo_inventory_fingerprint=inventory.inventory_fingerprint,
            source_mimo_records_sha256=_sha256_file(source_mimo / "records.jsonl"),
            source_h3_samples_sha256=_sha256_file(source_h3_samples_path),
            sample_count=len(records),
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            materialization_failure_count=materialization_failure_count,
            materialization_failure_code_counts=dict(
                sorted(materialization_failure_code_counts.items())
            ),
            warning_counts=dict(sorted(warning_counts.items())),
            target_clip_annotation_reuse_count=sum(
                max(0, count - 1)
                for count in Counter(item.clip_uid for item in records).values()
            ),
            existing_target_voice_reference_count=sum(
                len(items) for items in existing_target_by_clip.values()
            ),
            recovered_voice_candidate_count=len(recovery_records),
            recovered_voice_accepted_count=len(selected_recovery),
            recovered_voice_rejected_count=(
                len(recovery_records) - len(selected_recovery)
            ),
            recovered_voice_entity_count=len(
                {(item.clip_uid, item.entity_id) for item in selected_recovery}
            ),
            derived_in_pair_count=sum(
                item.derived_from_pair_type == "canonical"
                and item.pair_type == "in_pair"
                for item in records
            ),
            enriched_existing_in_pair_count=sum(
                item.pair_type == "in_pair"
                and item.derived_from_pair_type is None
                and bool(item.recovered_voice_references)
                for item in records
            ),
            recovery_rejection_reason_counts=dict(sorted(rejection_counts.items())),
            full_audio_reuse_count=sum(
                item.conditioning_variant == "full_audio_reuse" for item in records
            ),
            music_reference_count=sum(
                item.conditioning_variant == "music_reference" for item in records
            ),
            music_reference_rejection_reason_counts=dict(
                sorted(music_rejection_counts.items())
            ),
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.backup"
        if destination.exists():
            destination.replace(backup)
        try:
            temporary.replace(destination)
        except Exception:
            if backup.exists():
                backup.replace(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "MimoH3ShadowRecord",
    "MimoH3ShadowSummary",
    "MimoShadowAudioReference",
    "materialize_mimo25_h3_shadow",
]
