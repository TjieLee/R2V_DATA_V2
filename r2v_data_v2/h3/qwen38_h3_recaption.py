from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.specialized_audio_semantics import (
    SpecializedAudioSemanticsRecord,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

QWEN38_RECAPTION_PROMPT_VERSION = "h3_qwen38_ref2va_recaption_v5"
QWEN38_RECAPTION_POLICY_VERSION = "h3_qwen38_ref2va_contract_v3"
QWEN38_RECAPTION_DRAFT_VERSION = "r2v.h3.qwen38_recaption_draft.1"
QWEN38_RECAPTION_MATERIALIZER_VERSION = "h3_qwen38_materializer_v1"
QWEN38_RECAPTION_MANIFEST_VERSION = "r2v.h3.qwen38_recaption_manifest.1"
QWEN38_RECAPTION_RECORD_VERSION = "r2v.h3.qwen38_recaption_record.1"
QWEN38_RECAPTION_SUMMARY_VERSION = "r2v.h3.qwen38_recaption_summary.1"
QWEN38_RECAPTION_BACKEND_VERSION = "r2v.h3.qwen38_recaption_backend.1"
DEFAULT_MODEL = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_CHECKPOINT_ID = "/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next"
OUTPUT_DIRECTORY_NAME = "qwen38_h3_recaption_v1"
OFFICIAL_H3_CONTRACT_VERSION = "MiniMax-H3 Ref2VA guide, current main"
OFFICIAL_H3_SOURCE_FILES = (
    "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md",
    "docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md",
    "skills/h3-prompt-writing/SKILL.md",
    "skills/h3-prompt-writing/references/ref-en.txt",
    "skills/h3-prompt-writing/references/base-en.txt",
)

ConditioningVariant = Literal[
    "visual_only",
    "target_voice_reference",
    "cross_voice_reference",
    "full_audio_reuse",
]
RecaptionStatus = Literal["ready", "failed", "unsupported"]
SubjectKind = Literal["entity", "attribute", "background"]
AudioKind = Literal["target_voice", "cross_voice", "full_audio_reuse"]

_REFERENCE_LABEL = re.compile(r"<(Picture|Subject|Video|Audio)\s+([1-9]\d*)>")
_SPEAKER_LABEL = re.compile(r"\(S([1-9]\d*)\)")
_DIALOGUE_BLOCK = re.compile(r"<d>\[[^\]\r\n]+\][\s\S]*?</d>")
_SPEECH_PLACEHOLDER = re.compile(r"\[\[([^\[\]\r\n]+)\]\]")
_SPEECH_LEAD_IN = re.compile(
    r"\b(?:says?|speaks?|asks?|replies?|shouts?|whispers?)\s*,?\s*$",
    flags=re.IGNORECASE,
)
_SHOT_LABEL = re.compile(r"\[Shot\s+([1-9]\d*)\]")
_KEYFRAME_ROLE = re.compile(
    r"<Picture\s+[1-9]\d*>[^.\n]{0,100}\b(first frame|last frame|keyframe)\b"
    r"|\b(first frame|last frame|keyframe)\b[^.\n]{0,100}<Picture\s+[1-9]\d*>",
    flags=re.IGNORECASE,
)


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


class Qwen38RecaptionManifestCase(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_recaption_manifest.1"] = (
        QWEN38_RECAPTION_MANIFEST_VERSION
    )
    sample_id: str = Field(min_length=1)
    conditioning_variant: ConditioningVariant
    note: str | None = None

    @model_validator(mode="after")
    def validate_case(self) -> Qwen38RecaptionManifestCase:
        if self.note is not None and not self.note.strip():
            raise ValueError("recaption case note must be non-empty or null")
        return self


class RecaptionPictureContract(SchemaModel):
    image_index: int = Field(gt=0)
    picture_label: str = Field(pattern=r"^<Picture [1-9]\d*>$")
    image_id: str
    kind: Literal["subject", "object", "group", "background", "attribute"]
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None
    image_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_picture(self) -> RecaptionPictureContract:
        if self.picture_label != f"<Picture {self.image_index}>":
            raise ValueError("Picture label must match immutable image_index")
        return self


class RecaptionSubjectContract(SchemaModel):
    subject_index: int = Field(gt=0)
    subject_label: str = Field(pattern=r"^<Subject [1-9]\d*>$")
    kind: SubjectKind
    entity_id: str | None = None
    attribute_id: str | None = None
    owner_entity_id: str | None = None
    attribute_type: str | None = None
    source_picture_labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_subject(self) -> RecaptionSubjectContract:
        if self.subject_label != f"<Subject {self.subject_index}>":
            raise ValueError("Subject label must match subject_index")
        values = (self.attribute_id, self.owner_entity_id, self.attribute_type)
        if self.kind == "entity":
            if self.entity_id is None or any(value is not None for value in values):
                raise ValueError("entity Subject requires only entity_id")
        elif self.kind == "attribute":
            if self.entity_id is not None or any(value is None for value in values):
                raise ValueError("attribute Subject requires owner-bound provenance")
        elif self.entity_id is not None or any(value is not None for value in values):
            raise ValueError("background Subject cannot claim entity ownership")
        return self


class RecaptionAudioContract(SchemaModel):
    audio_index: int = Field(gt=0)
    audio_label: str = Field(pattern=r"^<Audio [1-9]\d*>$")
    kind: AudioKind
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_label: str | None = None
    entity_id: str | None = None
    speaker_id: str | None = Field(default=None, pattern=r"^S[1-9]\d*$")
    retention_marker: Literal["fully_copy", "reference"]

    @model_validator(mode="after")
    def validate_audio(self) -> RecaptionAudioContract:
        if self.audio_label != f"<Audio {self.audio_index}>":
            raise ValueError("Audio label must match audio_index")
        if self.kind == "full_audio_reuse":
            if (
                self.retention_marker != "fully_copy"
                or self.subject_label is not None
                or self.entity_id is not None
                or self.speaker_id is not None
            ):
                raise ValueError("full-audio reuse must be an unbound full copy")
        elif (
            self.retention_marker != "reference"
            or self.subject_label is None
            or self.entity_id is None
        ):
            raise ValueError("voice reference requires a bound Subject and entity")
        return self


class RecaptionReferenceContract(SchemaModel):
    pictures: list[RecaptionPictureContract] = Field(min_length=1)
    subjects: list[RecaptionSubjectContract] = Field(min_length=1)
    audios: list[RecaptionAudioContract]
    target_video_is_observation_only: Literal[True] = True
    h3_reference_video_count: Literal[0] = 0

    @model_validator(mode="after")
    def validate_contract(self) -> RecaptionReferenceContract:
        picture_indexes = [item.image_index for item in self.pictures]
        if picture_indexes != list(range(1, len(self.pictures) + 1)):
            raise ValueError("recaption Pictures must preserve canonical contiguous order")
        subject_indexes = [item.subject_index for item in self.subjects]
        if subject_indexes != list(range(1, len(self.subjects) + 1)):
            raise ValueError("recaption Subjects must use deterministic contiguous order")
        audio_indexes = [item.audio_index for item in self.audios]
        if audio_indexes != list(range(1, len(self.audios) + 1)):
            raise ValueError("recaption Audio labels must be contiguous")
        picture_labels = {item.picture_label for item in self.pictures}
        if any(
            not set(item.source_picture_labels).issubset(picture_labels)
            for item in self.subjects
        ):
            raise ValueError("Subject contract references an unknown Picture")
        entity_ids = {
            item.entity_id
            for item in self.subjects
            if item.kind == "entity" and item.entity_id is not None
        }
        if any(
            item.kind == "attribute" and item.owner_entity_id not in entity_ids
            for item in self.subjects
        ):
            raise ValueError("attribute Subject owner is not a referenced entity")
        subject_labels = {item.subject_label for item in self.subjects}
        if any(
            item.subject_label is not None
            and item.subject_label not in subject_labels
            for item in self.audios
        ):
            raise ValueError("Audio contract references an unknown Subject")
        if len(self.pictures) > 9 or len(self.audios) > 3:
            raise ValueError("official H3 per-modality reference limit exceeded")
        if len(self.pictures) + len(self.audios) > 12:
            raise ValueError("official H3 total reference-file limit exceeded")
        return self


class RecaptionSpeechFact(SchemaModel):
    fact_id: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_subject_label: str | None = None
    speaker_id: str = Field(pattern=r"^S[1-9]\d*$")
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    text: StrictStr
    language: StrictStr
    delivery: StrictStr | None = None
    locked_dialogue_block: StrictStr

    @model_validator(mode="after")
    def validate_speech(self) -> RecaptionSpeechFact:
        if self.end_time <= self.start_time or not self.text.strip():
            raise ValueError("recaption speech fact is incomplete")
        if (self.entity_id is None) != (self.entity_subject_label is None):
            raise ValueError("speech entity and Subject binding must be paired")
        expected = f"<d>[{self.language}] {self.text}</d>"
        if self.locked_dialogue_block != expected:
            raise ValueError("locked dialogue block differs from exact ASR text")
        return self


class RecaptionNonSpeechFact(SchemaModel):
    fact_id: str
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_time: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    category: str
    description: str
    source_attribution: str | None = None
    provenance: str

    @model_validator(mode="after")
    def validate_fact(self) -> RecaptionNonSpeechFact:
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("non-speech timing must be fully present or absent")
        if self.start_time is not None and self.end_time <= self.start_time:
            raise ValueError("non-speech timing must be positive")
        if any(not value.strip() for value in (self.category, self.description, self.provenance)):
            raise ValueError("non-speech fact text must not be empty")
        return self


class RecaptionAudioFacts(SchemaModel):
    speech: list[RecaptionSpeechFact]
    non_speech_events: list[RecaptionNonSpeechFact]
    overall_soundscape_hint: str | None = None
    non_diegetic_music_hint: str | None = None
    audio_grounding_complete: bool
    provenance: dict[str, str]

    @model_validator(mode="after")
    def validate_facts(self) -> RecaptionAudioFacts:
        fact_ids = [item.fact_id for item in (*self.speech, *self.non_speech_events)]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("recaption Audio fact IDs must be unique")
        order = [(item.start_time, item.end_time, item.segment_id) for item in self.speech]
        if order != sorted(order):
            raise ValueError("recaption speech facts must be chronological")
        return self


class AudioFactAuditItem(SchemaModel):
    fact_id: str
    action: Literal["preserved", "attribution_generalized"]
    rewritten_description: str | None = None

    @model_validator(mode="after")
    def validate_audit(self) -> AudioFactAuditItem:
        if self.action == "preserved" and self.rewritten_description is not None:
            raise ValueError("preserved Audio fact cannot be rewritten")
        if self.action == "attribution_generalized" and (
            self.rewritten_description is None or not self.rewritten_description.strip()
        ):
            raise ValueError("generalized Audio attribution requires replacement text")
        return self


class Qwen38H3StructuredResponse(SchemaModel):
    subject_definitions: list[StrictStr] = Field(min_length=1)
    summary: StrictStr
    retention_analysis: list[StrictStr] = Field(min_length=1)
    detailed_description: StrictStr
    overall_soundscape: StrictStr
    non_diegetic_music: StrictStr
    audio_fact_audit: list[AudioFactAuditItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nonempty(self) -> Qwen38H3StructuredResponse:
        strings = (
            *self.subject_definitions,
            self.summary,
            *self.retention_analysis,
            self.detailed_description,
            self.overall_soundscape,
            self.non_diegetic_music,
        )
        if any(not value.strip() for value in strings):
            raise ValueError("H3 rewrite sections must not be empty")
        return self


class Qwen38DraftShot(SchemaModel):
    shot_index: int = Field(ge=1)
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    description_template: StrictStr

    @model_validator(mode="after")
    def validate_description(self) -> Qwen38DraftShot:
        if not self.description_template.strip():
            raise ValueError("Qwen3.8 draft shot description must not be empty")
        return self


class Qwen38H3DraftResponse(SchemaModel):
    subject_definitions: list[StrictStr] = Field(min_length=1)
    summary: StrictStr
    retention_analysis: list[StrictStr] = Field(min_length=1)
    shots: list[Qwen38DraftShot] = Field(min_length=1)
    overall_soundscape: StrictStr
    non_diegetic_music: StrictStr
    audio_fact_audit: list[AudioFactAuditItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_draft(self) -> Qwen38H3DraftResponse:
        strings = (
            *self.subject_definitions,
            self.summary,
            *self.retention_analysis,
            *(shot.description_template for shot in self.shots),
            self.overall_soundscape,
            self.non_diegetic_music,
        )
        if any(not value.strip() for value in strings):
            raise ValueError("H3 draft sections must not be empty")
        indices = [shot.shot_index for shot in self.shots]
        if indices != list(range(1, len(self.shots) + 1)):
            raise ValueError("Qwen3.8 draft shot indices must be contiguous")
        if self.shots[0].start_time is not None:
            raise ValueError("Qwen3.8 draft Shot 1 cannot have a start time")
        later_times = [shot.start_time for shot in self.shots[1:]]
        if any(value is None for value in later_times):
            raise ValueError("later Qwen3.8 draft shots require start times")
        numeric_times = [value for value in later_times if value is not None]
        if numeric_times != sorted(numeric_times) or len(numeric_times) != len(
            set(numeric_times)
        ):
            raise ValueError("Qwen3.8 draft shot start times must increase")
        return self


class Qwen38BackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_recaption_backend.1"] = (
        QWEN38_RECAPTION_BACKEND_VERSION
    )
    backend: Literal["vllm", "sglang"] = "sglang"
    served_model_name: str
    checkpoint_id: str
    base_url: str
    media_mode: Literal["file", "http"]
    media_root: str
    media_base_url: str | None = None
    input_modality: Literal[
        "target_video_observation_plus_frozen_reference_images_plus_audio_text"
    ] = "target_video_observation_plus_frozen_reference_images_plus_audio_text"
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    prompt_version: Literal["h3_qwen38_ref2va_recaption_v5"] = (
        QWEN38_RECAPTION_PROMPT_VERSION
    )
    policy_version: Literal["h3_qwen38_ref2va_contract_v3"] = (
        QWEN38_RECAPTION_POLICY_VERSION
    )
    draft_schema_version: Literal["r2v.h3.qwen38_recaption_draft.1"] = (
        QWEN38_RECAPTION_DRAFT_VERSION
    )
    materializer_version: Literal["h3_qwen38_materializer_v1"] = (
        QWEN38_RECAPTION_MATERIALIZER_VERSION
    )
    official_h3_contract_version: str = OFFICIAL_H3_CONTRACT_VERSION
    official_h3_source_files: list[str] = Field(
        default_factory=lambda: list(OFFICIAL_H3_SOURCE_FILES)
    )
    enable_thinking: Literal[False] = False
    temperature: float = Field(gt=0, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(gt=0)
    min_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    presence_penalty: float = Field(allow_inf_nan=False)
    repetition_penalty: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_vllm_provenance(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("backend") != "vllm":
            return value
        if "video_fps" not in value:
            return value
        legacy_values = dict(value)
        fingerprint = legacy_values.pop("configuration_fingerprint", None)
        if fingerprint != _sha256_text(_compact_json(legacy_values)):
            raise ValueError("Qwen3.8 backend configuration fingerprint is invalid")
        normalized = dict(legacy_values)
        normalized.pop("video_fps")
        normalized.setdefault("min_p", None)
        normalized.setdefault("repetition_penalty", None)
        normalized["configuration_fingerprint"] = _sha256_text(
            _compact_json(normalized)
        )
        return normalized

    @model_validator(mode="after")
    def validate_provenance(self) -> Qwen38BackendProvenance:
        if self.backend == "sglang" and (
            self.min_p is None or self.repetition_penalty is None
        ):
            raise ValueError("SGLang sampling provenance is incomplete")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("Qwen3.8 backend configuration fingerprint is invalid")
        return self


class RecaptionCompletionDiagnostic(SchemaModel):
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class RecaptionFailure(SchemaModel):
    code: str
    reason: str
    attempt_count: int = Field(ge=0)
    issues: list[dict[str, str | None]] = Field(default_factory=list)


class Qwen38RecaptionRecord(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_recaption_record.1"] = (
        QWEN38_RECAPTION_RECORD_VERSION
    )
    status: RecaptionStatus
    sample_id: str
    clip_uid: str
    conditioning_variant: ConditioningVariant
    note: str | None = None
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rough_t2v_caption: str | None = None
    rough_r2v_instruction: str
    reference_contract: RecaptionReferenceContract | None = None
    audio_facts: RecaptionAudioFacts | None = None
    audio_fact_provenance: dict[str, str] = Field(default_factory=dict)
    audio_grounding_complete: bool
    backend_provenance: Qwen38BackendProvenance
    prompt_version: Literal["h3_qwen38_ref2va_recaption_v5"] = (
        QWEN38_RECAPTION_PROMPT_VERSION
    )
    official_h3_source_files: list[str] = Field(
        default_factory=lambda: list(OFFICIAL_H3_SOURCE_FILES)
    )
    request_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    structured_h3_sections: Qwen38H3StructuredResponse | None = None
    rendered_h3_prompt: str | None = None
    detailed_description_word_count: int | None = Field(default=None, ge=0)
    validation_warnings: list[str] = Field(default_factory=list)
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    completion_diagnostics: list[RecaptionCompletionDiagnostic] = Field(
        default_factory=list
    )
    failure: RecaptionFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> Qwen38RecaptionRecord:
        if self.raw_response_count != len(self.completion_diagnostics):
            raise ValueError("recaption raw-response diagnostics differ")
        outputs = (
            self.structured_h3_sections,
            self.rendered_h3_prompt,
            self.detailed_description_word_count,
        )
        if self.status == "ready":
            if (
                self.reference_contract is None
                or self.audio_facts is None
                or self.request_fingerprint is None
                or any(value is None for value in outputs)
                or self.failure is not None
            ):
                raise ValueError("ready recaption record is incomplete")
        elif any(value is not None for value in outputs) or self.failure is None:
            raise ValueError("non-ready recaption record cannot publish H3 output")
        if self.status == "unsupported" and (self.model_call_count or self.raw_response_count):
            raise ValueError("unsupported recaption case cannot claim model work")
        if self.audio_facts is not None and (
            self.audio_grounding_complete != self.audio_facts.audio_grounding_complete
        ):
            raise ValueError("record Audio grounding flag differs from normalized facts")
        return self


class Qwen38RecaptionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_recaption_summary.1"] = (
        QWEN38_RECAPTION_SUMMARY_VERSION
    )
    source_h3_samples_path: str
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_semantics_root: str | None = None
    audio_semantics_records_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    backend_provenance: Qwen38BackendProvenance
    case_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    conditioning_variant_counts: dict[str, int]
    failure_counts: dict[str, int]
    target_video_reference_count: Literal[0] = 0
    checkpoint_written: Literal[False] = False
    production_h3_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> Qwen38RecaptionSummary:
        if self.case_count != self.ready_count + self.failed_count + self.unsupported_count:
            raise ValueError("recaption summary counts do not reconcile")
        if (self.audio_semantics_root is None) != (
            self.audio_semantics_records_sha256 is None
        ):
            raise ValueError("recaption Audio semantics provenance must be paired")
        return self


@dataclass(frozen=True)
class Qwen38RecaptionRequest:
    sample: FinalH3SampleV2
    case: Qwen38RecaptionManifestCase
    reference_contract: RecaptionReferenceContract
    audio_facts: RecaptionAudioFacts
    request_fingerprint: str


@dataclass(frozen=True)
class Qwen38BackendResult:
    response: Qwen38H3StructuredResponse
    raw_responses: tuple[str, ...]
    diagnostics: tuple[RecaptionCompletionDiagnostic, ...]
    model_call_count: int
    validation_warnings: tuple[str, ...]


class Qwen38BackendFailure(ValueError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: Sequence[str] = (),
        diagnostics: Sequence[RecaptionCompletionDiagnostic] = (),
        issues: Sequence[ValidationIssue] = (),
        model_call_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = tuple(raw_responses)
        self.diagnostics = tuple(diagnostics)
        self.issues = tuple(issues)
        self.model_call_count = model_call_count


class Qwen38RecaptionBackend(Protocol):
    @property
    def provenance(self) -> Qwen38BackendProvenance: ...

    def recaption(self, request: Qwen38RecaptionRequest) -> Qwen38BackendResult: ...


def _subject_by_entity(contract: RecaptionReferenceContract) -> dict[str, str]:
    return {
        item.entity_id: item.subject_label
        for item in contract.subjects
        if item.kind == "entity" and item.entity_id is not None
    }


def build_reference_contract(
    sample: FinalH3SampleV2,
    variant: ConditioningVariant,
) -> RecaptionReferenceContract:
    pictures = [
        RecaptionPictureContract(
            image_index=item.image_index,
            picture_label=f"<Picture {item.image_index}>",
            image_id=item.image_id,
            kind=item.kind,
            entity_id=item.entity_id,
            attribute_id=item.attribute_id,
            owner_entity_id=item.owner_entity_id,
            attribute_type=item.attribute_type,
            image_path=item.image_artifact_path,
            image_sha256=_sha256_file(Path(item.image_artifact_path).resolve(strict=True)),
        )
        for item in sample.visual_references
    ]
    entity_sources: dict[str, list[str]] = {}
    entity_order: list[str] = []
    for picture in pictures:
        if picture.kind not in {"subject", "object", "group"}:
            continue
        assert picture.entity_id is not None
        if picture.entity_id not in entity_sources:
            entity_sources[picture.entity_id] = []
            entity_order.append(picture.entity_id)
        entity_sources[picture.entity_id].append(picture.picture_label)
    subjects: list[RecaptionSubjectContract] = []
    for entity_id in entity_order:
        subjects.append(
            RecaptionSubjectContract(
                subject_index=len(subjects) + 1,
                subject_label=f"<Subject {len(subjects) + 1}>",
                kind="entity",
                entity_id=entity_id,
                source_picture_labels=entity_sources[entity_id],
            )
        )
    attribute_pictures = [item for item in pictures if item.kind == "attribute"]
    attribute_ids = [item.attribute_id for item in attribute_pictures]
    if len(attribute_ids) != len(set(attribute_ids)):
        raise ValueError("recaption attribute references must be unique")
    for picture in attribute_pictures:
        subjects.append(
            RecaptionSubjectContract(
                subject_index=len(subjects) + 1,
                subject_label=f"<Subject {len(subjects) + 1}>",
                kind="attribute",
                attribute_id=picture.attribute_id,
                owner_entity_id=picture.owner_entity_id,
                attribute_type=picture.attribute_type,
                source_picture_labels=[picture.picture_label],
            )
        )
    background_labels = [
        item.picture_label for item in pictures if item.kind == "background"
    ]
    if background_labels:
        subjects.append(
            RecaptionSubjectContract(
                subject_index=len(subjects) + 1,
                subject_label=f"<Subject {len(subjects) + 1}>",
                kind="background",
                source_picture_labels=background_labels,
            )
        )
    subject_by_entity = _subject_by_entity(
        RecaptionReferenceContract(pictures=pictures, subjects=subjects, audios=[])
    )
    audios: list[RecaptionAudioContract] = []
    if variant in {"target_voice_reference", "cross_voice_reference"}:
        expected_source = "target" if variant == "target_voice_reference" else "cross_donor"
        voices = sorted(sample.subject_voices, key=lambda item: (item.subject_index, item.entity_id))
        if not voices or any(item.voice_source != expected_source for item in voices):
            raise ValueError("sample does not provide the requested voice-reference variant")
        speaker_by_entity: dict[str, str] = {}
        speaker_ids = _speaker_ids(sample)
        for speech in sorted(
            sample.speech_segments,
            key=lambda item: (item.start_time, item.end_time, item.segment_id),
        ):
            if speech.entity_id is not None and speech.entity_id not in speaker_by_entity:
                speaker_by_entity[speech.entity_id] = speaker_ids[
                    speech.speaker_cluster_id
                ]
        for voice in voices:
            path = Path(voice.voice_reference_path).expanduser().resolve(strict=True)
            audios.append(
                RecaptionAudioContract(
                    audio_index=len(audios) + 1,
                    audio_label=f"<Audio {len(audios) + 1}>",
                    kind=("target_voice" if expected_source == "target" else "cross_voice"),
                    path=str(path),
                    sha256=_sha256_file(path),
                    subject_label=subject_by_entity[voice.entity_id],
                    entity_id=voice.entity_id,
                    speaker_id=speaker_by_entity.get(voice.entity_id),
                    retention_marker="reference",
                )
            )
    elif variant == "full_audio_reuse":
        path = Path(sample.target_full_audio_path).expanduser().resolve(strict=True)
        audios.append(
            RecaptionAudioContract(
                audio_index=1,
                audio_label="<Audio 1>",
                kind="full_audio_reuse",
                path=str(path),
                sha256=_sha256_file(path),
                retention_marker="fully_copy",
            )
        )
    return RecaptionReferenceContract(pictures=pictures, subjects=subjects, audios=audios)


def _speaker_ids(sample: FinalH3SampleV2) -> dict[str, str]:
    result: dict[str, str] = {}
    for speech in sorted(
        sample.speech_segments,
        key=lambda item: (item.start_time, item.end_time, item.segment_id),
    ):
        if speech.speaker_cluster_id not in result:
            result[speech.speaker_cluster_id] = f"S{len(result) + 1}"
    return result


def build_audio_facts(
    sample: FinalH3SampleV2,
    contract: RecaptionReferenceContract,
    semantics: SpecializedAudioSemanticsRecord | None,
    *,
    semantics_records_sha256: str | None,
) -> RecaptionAudioFacts:
    if semantics is not None:
        target_video = Path(sample.target_video).expanduser().resolve(strict=True)
        target_audio = Path(sample.target_full_audio_path).expanduser().resolve(
            strict=True
        )
        if (
            semantics.target_clip_uid != sample.clip_uid
            or Path(semantics.target_video_path).expanduser().resolve(strict=True)
            != target_video
            or semantics.target_video_sha256 != _sha256_file(target_video)
            or Path(semantics.target_full_audio_path).expanduser().resolve(strict=True)
            != target_audio
            or semantics.target_full_audio_sha256 != _sha256_file(target_audio)
        ):
            raise ValueError("Audio semantics media provenance differs from H3 sample")
    delivery_by_cluster = {
        item.speaker_cluster_id: item.delivery_style
        for item in semantics.speaker_delivery
    } if semantics is not None else {}
    speaker_ids = _speaker_ids(sample)
    subject_by_entity = _subject_by_entity(contract)
    speech = []
    for index, item in enumerate(
        sorted(
            sample.speech_segments,
            key=lambda value: (value.start_time, value.end_time, value.segment_id),
        ),
        start=1,
    ):
        language = item.language or "Unknown"
        speech.append(
            RecaptionSpeechFact(
                fact_id=f"speech_{index}",
                segment_id=item.segment_id,
                speaker_cluster_id=item.speaker_cluster_id,
                entity_id=item.entity_id,
                entity_subject_label=(
                    None if item.entity_id is None else subject_by_entity.get(item.entity_id)
                ),
                speaker_id=speaker_ids[item.speaker_cluster_id],
                start_time=item.start_time,
                end_time=item.end_time,
                text=item.text,
                language=language,
                delivery=delivery_by_cluster.get(item.speaker_cluster_id),
                locked_dialogue_block=f"<d>[{language}] {item.text}</d>",
            )
        )
    non_speech = []
    if semantics is not None:
        for index, event in enumerate(semantics.temporal_audio_events, start=1):
            non_speech.append(
                RecaptionNonSpeechFact(
                    fact_id=f"non_speech_{index}",
                    start_time=event.start_time,
                    end_time=event.end_time,
                    category="temporal_audio_event",
                    description=event.description,
                    provenance="specialized_audio_semantics.temporal_audio_events",
                )
            )
    provenance = {"speech": "current_h3_final_sample.speech_segments"}
    if semantics_records_sha256 is not None:
        provenance["audio_semantics_records_sha256"] = semantics_records_sha256
    grounding_complete = semantics is not None and (
        semantics.global_semantics_status == "ready"
        and semantics.local_semantics_status == "ready"
    )
    return RecaptionAudioFacts(
        speech=speech,
        non_speech_events=non_speech,
        overall_soundscape_hint=(
            None if semantics is None else semantics.overall_soundscape
        ),
        non_diegetic_music_hint=(
            None if semantics is None else semantics.non_diegetic_music
        ),
        audio_grounding_complete=grounding_complete,
        provenance=provenance,
    )


def _summary_prefix(variant: ConditioningVariant) -> str:
    if variant == "visual_only":
        return "[reference generation]"
    if variant in {"target_voice_reference", "cross_voice_reference"}:
        return "[reference generation + audio reference]"
    return "[reference generation + audio reuse]"


def _reference_contract_text(contract: RecaptionReferenceContract) -> str:
    payload = contract.model_dump(mode="json")
    for picture in payload["pictures"]:
        picture.pop("image_path")
        picture.pop("image_sha256")
    for audio in payload["audios"]:
        audio.pop("path")
        audio.pop("sha256")
    return _compact_json(payload)


UNGROUNDED_OVERALL_SOUNDSCAPE = (
    "No additional soundscape is established by the supplied upstream Audio facts."
)
UNGROUNDED_NON_DIEGETIC_MUSIC = (
    "Non-diegetic music is not established by the supplied upstream Audio facts."
)


SYNTHETIC_FORMAT_EXAMPLE = """subject_definitions:
<Subject 1> is a cyclist sourced from <Picture 1>.
<Subject 2> is the riverside environment sourced from <Picture 2>.
<Audio 1> is the voice-timbre reference for <Subject 1>.
summary:
[reference generation + audio reference] A cyclist pauses in the referenced riverside setting and speaks once.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the cyclist's referenced appearance is retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the riverside environment is retained.
<Audio 1>: reference - the speaker follows the referenced delivery without copying the signal.
shots:
- shot_index: 1
  start_time: null
  description_template: The target uses a natural daylight documentary style. <Subject 1> stops beside <Subject 2> and looks across the river. [[speech_1]]
overall_soundscape:
The supplied upstream Audio facts establish light riverside ambience under the speech.
non_diegetic_music:
The supplied upstream Audio facts establish that no non-diegetic music is present."""


SYSTEM_PROMPT = f"""You are a MiniMax H3 full-reference (Ref2VA) recaptioner.

Observe one ground-truth TARGET VIDEO, frozen reference images, rough Visual hints,
and trusted upstream Audio TEXT facts. Write a generation-quality H3 Ref2VA prompt
for how that exact target video is depicted. The target video is observation-only;
it is never a reference asset and must never receive a <Video N> label. You receive
no raw audio and are not ASR, diarization, speaker identification, or binding logic.

Reference labels, Picture numbers, Subject numbers, Audio numbers, speaker IDs,
entity bindings, timestamps, and transcripts are pipeline-owned. Never renumber,
remap, merge, split, delete, correct, translate, or invent them. The target video is
the visual truth; rough captions are hints only.

Follow the current official MiniMax H3 Ref2VA format. Return a structured draft
with subject_definitions, summary, retention_analysis, shots,
overall_soundscape, non_diegetic_music, and audio_fact_audit. The pipeline, not
you, materializes the final detailed_description from shots. Write English except
lyrics and visible scene text.

Treat every shot description_template as a generation-quality dense video
description, not an ordinary caption or plot summary. For each observed shot,
systematically cover every applicable dimension in this visual checklist:
- visual style
- shot scale and framing
- camera angle
- foreground, midground, and background composition
- every salient visible subject's appearance
- spatial positions and relationships
- pose
- body, arm, and hand motion
- head motion
- gaze
- facial expression and visible expression changes
- interactions
- object state and state changes
- environment, materials, and readable text
- lighting and color
- camera motion, or an explicit stable/static camera
- temporal progression from the early through middle to late portions of the shot
- speech placeholders at their correct observed temporal positions

Use observed evidence, not plausible filler. Do not infer psychology or emotion
that is not visually supported, intent, causality, relationships, identity not
supplied upstream, sounds from visible actions, invisible or offscreen events, or
invented object details. For information-rich generation recaption, target roughly
300-450 English words across the materialized detailed_description when the video
supports it. Do not pad a visually simple clip merely to hit the target; prefer
concrete temporal and visual detail over repetition. Current clips are expected to
be single-shot. Add a later shot only for a real hard cut and supply its start_time
in seconds. Never write [Shot N] syntax; the pipeline owns shot headers and
timestamps.

Pictures are content references, not first/last frames or keyframes. Use only the
supplied labels. Summary must begin with the supplied task prefix. For the current
recaption contract, the only allowed visible retention markers are:
- fully_preserved
- partially_preserved
- weak_reference
attribute_transfer is not assigned by the current conditioning contract and MUST
NOT be emitted. Audio markers are fully_copy, partially_copy, reference,
weak_reference.

Insert every supplied [[speech_N]] placeholder exactly once in chronological
order. A placeholder represents the complete source-and-dialogue clause; never
prefix it with says, speaks, a pronoun, a role, or a character name. Never write
pipeline-owned (Sx), <Subject N> (Sx), <d> dialogue, transcript text, or unknown
speech placeholders. The pipeline deterministically materializes each complete
speech clause after validating your draft. Do not copy donor/reference dialogue.

Audio text facts are trusted. Do not invent sounds from visible actions. Preserve
an audible event even when its source is not visible. Visual evidence may only
generalize an obviously wrong visual source attribution, such as changing "the
visible woman slams the door" to "a door slam is heard"; it may not delete the
event or alter ASR, speaker, entity, or timing. When Audio grounding is incomplete,
only supplied speech facts may be asserted as audible. Do not infer ambience,
footsteps, room tone, clothing noise, or other sounds from visible actions. When
the supplied overall soundscape or music hint is absent, use the exact conservative
wording supplied in the user contract rather than writing N/A or claiming absence.
overall_soundscape summarizes grounded ambience, physical sounds, and non-verbal
human sounds without repeating full dialogue. non_diegetic_music is audience-only
BGM.

Return one compact JSON object matching the supplied schema. audio_fact_audit is
diagnostic only: include every supplied non-speech fact exactly once with action
preserved or attribution_generalized. Speech fact IDs are never audit entries.
Never emit deleted/rejected/removed. No markdown, explanation, or chain-of-thought.

Small format-only synthetic example (do not copy its scene content):
{SYNTHETIC_FORMAT_EXAMPLE}
"""


def _speech_exact_source(speech: RecaptionSpeechFact) -> str:
    if speech.entity_subject_label is None:
        return f"({speech.speaker_id})"
    return f"{speech.entity_subject_label} ({speech.speaker_id})"


def _speech_placeholder_contract(request: Qwen38RecaptionRequest) -> str:
    lines = ["SPEECH PLACEHOLDER CONTRACT:"]
    for speech in request.audio_facts.speech:
        lines.append(f"- {speech.fact_id}: [[{speech.fact_id}]]")
    lines.extend(
        (
            "Rules:",
            "- every placeholder must appear exactly once across shot description_template fields",
            "- placeholders must appear in the listed chronological order",
            "- a placeholder is the complete speech clause; do not prefix it with says or a speaker description",
            "- do not emit unknown placeholders, (Sx), <d> dialogue, or transcript text",
        )
    )
    return "\n".join(lines)


def _audio_fact_audit_contract(request: Qwen38RecaptionRequest) -> str:
    fact_ids = [item.fact_id for item in request.audio_facts.non_speech_events]
    lines = [
        "AUDIO FACT AUDIT CONTRACT:",
        f"allowed_fact_ids={_compact_json(fact_ids)}",
    ]
    if not fact_ids:
        lines.append("audio_fact_audit MUST be exactly [].")
    else:
        lines.append("Each allowed fact_id must appear exactly once.")
    lines.extend(
        (
            "No other fact_id is permitted.",
            "Speech fact IDs are never audit entries.",
        )
    )
    return "\n".join(lines)


def _audio_grounding_contract(request: Qwen38RecaptionRequest) -> str:
    facts = request.audio_facts
    if facts.audio_grounding_complete:
        return "AUDIO GROUNDING CONTRACT: Use only supplied upstream Audio facts."
    lines = [
        "AUDIO GROUNDING CONTRACT: INCOMPLETE",
        "Only supplied speech facts may be asserted as audible.",
        "Do not infer ambience or non-speech sounds from visual evidence.",
    ]
    if facts.overall_soundscape_hint is None and not facts.non_speech_events:
        lines.append(
            f"overall_soundscape MUST be exactly: {UNGROUNDED_OVERALL_SOUNDSCAPE}"
        )
    if facts.non_diegetic_music_hint is None:
        lines.append(
            f"non_diegetic_music MUST be exactly: {UNGROUNDED_NON_DIEGETIC_MUSIC}"
        )
    return "\n".join(lines)


def _user_contract(request: Qwen38RecaptionRequest) -> str:
    return (
        "TARGET VIDEO is observation-only and must not become <Video N>.\n"
        f"conditioning_variant={request.case.conditioning_variant}\n"
        f"required_summary_prefix={_summary_prefix(request.case.conditioning_variant)}\n"
        "REFERENCE CONTRACT (immutable):\n"
        f"{_reference_contract_text(request.reference_contract)}\n"
        "ROUGH VISUAL HINTS (non-authoritative):\n"
        f"t2v_caption=null\nr2v_instruction={request.sample.r2v_instruction}\n"
        "LOCKED AUDIO TEXT FACTS:\n"
        f"{_compact_json(request.audio_facts.model_dump(mode='json'))}\n"
        f"{_speech_placeholder_contract(request)}\n"
        f"{_audio_fact_audit_contract(request)}\n"
        f"{_audio_grounding_contract(request)}\n"
        "Return this strict schema:\n"
        f"{_compact_json(Qwen38H3DraftResponse.model_json_schema())}"
    )


def _repair_prompt(
    request: Qwen38RecaptionRequest,
    *,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        _user_contract(request)
        + "\nRepair only the listed format/contract errors. Preserve visual meaning and "
        "all locked labels, facts, placeholder IDs/order, entity bindings, and shot "
        "semantics. Do not emit pipeline-owned dialogue or speaker serialization. "
        "Return one compact JSON object only.\n"
        + "REPAIR EXACT CONTRACTS:\n"
        + _speech_placeholder_contract(request)
        + "\n"
        + _audio_fact_audit_contract(request)
        + "\n"
        + _audio_grounding_contract(request)
        + "\n"
        + f"issues={_compact_json([item.to_dict() for item in issues])}\n"
        + f"invalid_response={invalid_response}"
    )


def _draft_non_shot_text(draft: Qwen38H3DraftResponse) -> str:
    return "\n".join(
        (
            *draft.subject_definitions,
            draft.summary,
            *draft.retention_analysis,
            draft.overall_soundscape,
            draft.non_diegetic_music,
        )
    )


def validate_h3_draft(
    draft: Qwen38H3DraftResponse,
    request: Qwen38RecaptionRequest,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    templates = [shot.description_template for shot in draft.shots]
    actual_placeholders = [
        match.group(1)
        for template in templates
        for match in _SPEECH_PLACEHOLDER.finditer(template)
    ]
    expected_placeholders = [speech.fact_id for speech in request.audio_facts.speech]
    expected_set = set(expected_placeholders)
    actual_counts = Counter(actual_placeholders)
    unknown = sorted(set(actual_placeholders) - expected_set)
    missing = [item for item in expected_placeholders if actual_counts[item] == 0]
    duplicate = sorted(item for item, count in actual_counts.items() if count > 1)
    if unknown:
        issues.append(
            ValidationIssue(
                "unknown_speech_placeholder",
                "shots",
                f"unknown placeholders: {unknown}",
            )
        )
    if missing:
        issues.append(
            ValidationIssue(
                "missing_speech_placeholder",
                "shots",
                f"missing placeholders: {missing}",
            )
        )
    if duplicate:
        issues.append(
            ValidationIssue(
                "duplicate_speech_placeholder",
                "shots",
                f"duplicate placeholders: {duplicate}",
            )
        )
    if not unknown and not missing and not duplicate and (
        actual_placeholders != expected_placeholders
    ):
        issues.append(
            ValidationIssue(
                "speech_placeholder_order_mismatch",
                "shots",
                "speech placeholders must follow chronological fact order",
            )
        )
    non_shot_text = _draft_non_shot_text(draft)
    if _SPEECH_PLACEHOLDER.search(non_shot_text):
        issues.append(
            ValidationIssue(
                "speech_placeholder_outside_shot",
                None,
                "speech placeholders are allowed only in shot descriptions",
            )
        )
    all_draft_text = "\n".join((*templates, non_shot_text))
    if "<d>" in all_draft_text or "</d>" in all_draft_text:
        issues.append(
            ValidationIssue(
                "draft_contains_dialogue_markup",
                None,
                "draft contains pipeline-owned dialogue markup",
            )
        )
    if any(
        speech.text and speech.text in all_draft_text
        for speech in request.audio_facts.speech
    ):
        issues.append(
            ValidationIssue(
                "draft_contains_locked_dialogue",
                None,
                "draft copies pipeline-owned dialogue text",
            )
        )
    if _SPEAKER_LABEL.search(all_draft_text):
        issues.append(
            ValidationIssue(
                "draft_contains_speech_source",
                None,
                "draft contains pipeline-owned speaker syntax",
            )
        )
    for shot in draft.shots:
        template = shot.description_template
        if "[Shot " in template:
            issues.append(
                ValidationIssue(
                    "draft_contains_shot_header",
                    "shots",
                    f"shot {shot.shot_index} contains pipeline-owned shot syntax",
                )
            )
        if any(
            _SPEECH_LEAD_IN.search(template[max(0, match.start() - 80) : match.start()])
            for match in _SPEECH_PLACEHOLDER.finditer(template)
        ):
            issues.append(
                ValidationIssue(
                    "draft_prefixes_complete_speech_placeholder",
                    "shots",
                    f"shot {shot.shot_index} prefixes a complete speech clause",
                )
            )
    return issues


def _render_locked_speech(speech: RecaptionSpeechFact) -> str:
    return (
        f"{_speech_exact_source(speech)} says, {speech.locked_dialogue_block}"
    )


def _format_shot_timestamp(seconds: float) -> str:
    total_milliseconds = math.floor(seconds * 1000.0 + 0.5)
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def materialize_h3_draft(
    draft: Qwen38H3DraftResponse,
    request: Qwen38RecaptionRequest,
) -> Qwen38H3StructuredResponse:
    issues = validate_h3_draft(draft, request)
    if issues:
        raise ValueError(
            "Qwen3.8 draft cannot be materialized: "
            + _compact_json([item.to_dict() for item in issues])
        )
    speech_by_id = {
        speech.fact_id: speech for speech in request.audio_facts.speech
    }
    rendered_shots = []
    for shot in draft.shots:
        description = _SPEECH_PLACEHOLDER.sub(
            lambda match: _render_locked_speech(speech_by_id[match.group(1)]),
            shot.description_template.strip(),
        )
        if shot.shot_index == 1:
            header = "[Shot 1]"
        else:
            if shot.start_time is None:
                raise AssertionError("validated later shot is missing start_time")
            header = (
                f"[Shot {shot.shot_index}] At "
                f"{_format_shot_timestamp(shot.start_time)},"
            )
        rendered_shots.append(f"{header} {description}")
    return Qwen38H3StructuredResponse(
        subject_definitions=draft.subject_definitions,
        summary=draft.summary,
        retention_analysis=draft.retention_analysis,
        detailed_description="\n".join(rendered_shots),
        overall_soundscape=draft.overall_soundscape,
        non_diegetic_music=draft.non_diegetic_music,
        audio_fact_audit=draft.audio_fact_audit,
    )


def render_h3_prompt(response: Qwen38H3StructuredResponse) -> str:
    return (
        "subject_definitions:\n"
        + "\n".join(response.subject_definitions)
        + "\n\nsummary:\n"
        + response.summary
        + "\n\nretention_analysis:\n"
        + "\n".join(response.retention_analysis)
        + "\n\ndetailed_description:\n"
        + response.detailed_description
        + "\n\noverall_soundscape:\n"
        + response.overall_soundscape
        + "\n\nnon_diegetic_music:\n"
        + response.non_diegetic_music
    )


def _all_section_text(response: Qwen38H3StructuredResponse) -> str:
    return "\n".join(
        (
            *response.subject_definitions,
            response.summary,
            *response.retention_analysis,
            response.detailed_description,
            response.overall_soundscape,
            response.non_diegetic_music,
        )
    )


def _normalized_contract_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_h3_response(
    response: Qwen38H3StructuredResponse,
    request: Qwen38RecaptionRequest,
) -> tuple[list[ValidationIssue], list[str]]:
    issues: list[ValidationIssue] = []
    text = _all_section_text(response)
    expected_prefix = _summary_prefix(request.case.conditioning_variant)
    if not response.summary.startswith(expected_prefix):
        issues.append(ValidationIssue("wrong_summary_prefix", "summary", expected_prefix))

    expected_labels = {
        "Picture": {str(item.image_index) for item in request.reference_contract.pictures},
        "Subject": {str(item.subject_index) for item in request.reference_contract.subjects},
        "Video": set(),
        "Audio": {str(item.audio_index) for item in request.reference_contract.audios},
    }
    actual_labels: dict[str, set[str]] = {key: set() for key in expected_labels}
    for kind, index in _REFERENCE_LABEL.findall(text):
        actual_labels[kind].add(index)
        if index not in expected_labels[kind]:
            issues.append(
                ValidationIssue("unknown_reference_label", None, f"<{kind} {index}>")
            )
    for kind in ("Picture", "Subject", "Audio"):
        missing = expected_labels[kind] - actual_labels[kind]
        if missing:
            issues.append(
                ValidationIssue(
                    "missing_reference_label",
                    None,
                    f"missing {kind}: {sorted(missing, key=int)}",
                )
            )
    if actual_labels["Video"]:
        issues.append(
            ValidationIssue(
                "target_video_became_reference",
                None,
                "current recaption contract permits no <Video N>",
            )
        )

    for subject in request.reference_contract.subjects:
        definitions = [
            item
            for item in response.subject_definitions
            if item.lstrip().startswith(subject.subject_label)
        ]
        if len(definitions) != 1 or any(
            label not in definitions[0]
            for label in subject.source_picture_labels
        ):
            issues.append(
                ValidationIssue(
                    "subject_definition_contract_mismatch",
                    "subject_definitions",
                    f"{subject.subject_label} must retain its source Pictures",
                )
            )
    for audio in request.reference_contract.audios:
        definitions = [
            item
            for item in response.subject_definitions
            if item.lstrip().startswith(audio.audio_label)
        ]
        if len(definitions) != 1 or (
            audio.subject_label is not None
            and audio.subject_label not in definitions[0]
        ):
            issues.append(
                ValidationIssue(
                    "audio_definition_contract_mismatch",
                    "subject_definitions",
                    f"{audio.audio_label} definition differs from its owner",
                )
            )

    expected_speakers = {item.speaker_id[1:] for item in request.audio_facts.speech}
    actual_speakers = set(_SPEAKER_LABEL.findall(text))
    if actual_speakers - expected_speakers:
        issues.append(ValidationIssue("unknown_speaker_id", None, "unknown (Sx)"))
    if expected_speakers - actual_speakers:
        issues.append(ValidationIssue("missing_speaker_id", None, "missing (Sx)"))

    actual_dialogue = _DIALOGUE_BLOCK.findall(response.detailed_description)
    expected_dialogue = [item.locked_dialogue_block for item in request.audio_facts.speech]
    if Counter(actual_dialogue) != Counter(expected_dialogue):
        issues.append(
            ValidationIssue(
                "locked_dialogue_mismatch",
                "detailed_description",
                "every exact locked dialogue block must appear once",
            )
        )
    dialogue_occurrences: dict[str, list[tuple[int, int]]] = {}
    for match in _DIALOGUE_BLOCK.finditer(response.detailed_description):
        dialogue_occurrences.setdefault(match.group(0), []).append(match.span())
    used_occurrences: Counter[str] = Counter()
    all_dialogue_ends = sorted(
        end
        for occurrences in dialogue_occurrences.values()
        for _, end in occurrences
    )
    for speech in request.audio_facts.speech:
        occurrence_index = used_occurrences[speech.locked_dialogue_block]
        used_occurrences[speech.locked_dialogue_block] += 1
        occurrences = dialogue_occurrences.get(speech.locked_dialogue_block, [])
        if occurrence_index >= len(occurrences):
            continue
        block_index, _ = occurrences[occurrence_index]
        previous_dialogue_end = max(
            (end for end in all_dialogue_ends if end <= block_index),
            default=0,
        )
        prefix = response.detailed_description[
            max(previous_dialogue_end, block_index - 180) : block_index
        ]
        expected_source = _speech_exact_source(speech)
        if expected_source not in prefix:
            issues.append(
                ValidationIssue(
                    "locked_dialogue_source_mismatch",
                    "detailed_description",
                    f"{speech.fact_id} requires {expected_source}",
                )
            )
        if speech.entity_subject_label is None and re.search(
            r"<Subject [1-9]\d*>\s*\(" + re.escape(speech.speaker_id) + r"\)",
            prefix,
        ):
            issues.append(
                ValidationIssue(
                    "unbound_speaker_gained_subject",
                    "detailed_description",
                    f"{speech.speaker_id} is not entity-bound",
                )
            )

    expected_fact_ids = {item.fact_id for item in request.audio_facts.non_speech_events}
    actual_fact_ids = [item.fact_id for item in response.audio_fact_audit]
    if len(actual_fact_ids) != len(set(actual_fact_ids)) or set(actual_fact_ids) != expected_fact_ids:
        issues.append(
            ValidationIssue(
                "audio_fact_audit_mismatch",
                "audio_fact_audit",
                "every supplied non-speech fact must be audited exactly once",
            )
        )

    facts = request.audio_facts
    if not facts.audio_grounding_complete:
        if facts.non_diegetic_music_hint is None and _normalized_contract_text(
            response.non_diegetic_music
        ) != _normalized_contract_text(UNGROUNDED_NON_DIEGETIC_MUSIC):
            issues.append(
                ValidationIssue(
                    "ungrounded_non_diegetic_music",
                    "non_diegetic_music",
                    "music status is not established by upstream Audio facts",
                )
            )
        if (
            facts.overall_soundscape_hint is None
            and not facts.non_speech_events
            and _normalized_contract_text(response.overall_soundscape)
            != _normalized_contract_text(UNGROUNDED_OVERALL_SOUNDSCAPE)
        ):
            issues.append(
                ValidationIssue(
                    "ungrounded_overall_soundscape",
                    "overall_soundscape",
                    "additional soundscape is not established by upstream Audio facts",
                )
            )

    retention = "\n".join(response.retention_analysis)
    for audio in request.reference_contract.audios:
        pattern = re.compile(
            re.escape(audio.audio_label) + r"[^\n]*:\s*" + audio.retention_marker + r"\b"
        )
        if pattern.search(retention) is None:
            issues.append(
                ValidationIssue(
                    "audio_retention_mismatch",
                    "retention_analysis",
                    f"{audio.audio_label} requires {audio.retention_marker}",
                )
            )
    if request.case.conditioning_variant == "visual_only" and "<Audio " in text:
        issues.append(ValidationIssue("unexpected_audio_reference", None, "visual_only"))
    if _KEYFRAME_ROLE.search(text):
        issues.append(
            ValidationIssue(
                "unassigned_picture_keyframe_role",
                None,
                "content references are not frame anchors",
            )
        )
    if "[Shot 1]" not in response.detailed_description:
        issues.append(
            ValidationIssue(
                "missing_opening_shot",
                "detailed_description",
                "Ref2VA detailed_description requires [Shot 1]",
            )
        )
    for match in _SHOT_LABEL.finditer(response.detailed_description):
        shot_number = int(match.group(1))
        if shot_number == 1:
            continue
        suffix = response.detailed_description[match.end() : match.end() + 22]
        if re.match(r" At \d{2}:\d{2}\.\d{3},", suffix) is None:
            issues.append(
                ValidationIssue(
                    "invalid_shot_timing_notation",
                    "detailed_description",
                    f"[Shot {shot_number}] requires At MM:SS.mmm,",
                )
            )
    if "attribute_transfer" in retention:
        issues.append(
            ValidationIssue(
                "unassigned_attribute_transfer",
                "retention_analysis",
                "current contract does not transfer attributes to another subject",
            )
        )
    word_count = len(re.findall(r"\b[\w'-]+\b", response.detailed_description))
    warnings = []
    if word_count < 250:
        warnings.append("detailed_description_below_250_words")
    if word_count > 500:
        warnings.append("detailed_description_above_500_words")
    return issues, warnings


def _diagnostic(completion: object, choice: object) -> RecaptionCompletionDiagnostic:
    usage = getattr(completion, "usage", None)

    def value(field: str) -> object | None:
        return usage.get(field) if isinstance(usage, dict) else getattr(usage, field, None)

    return RecaptionCompletionDiagnostic(
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=value("prompt_tokens"),
        completion_tokens=value("completion_tokens"),
        total_tokens=value("total_tokens"),
    )


@dataclass(frozen=True)
class Qwen38RecaptionConfig:
    base_url: str
    media_resolver: MediaURLResolver
    api_key: str = "EMPTY"
    served_model_name: str = DEFAULT_MODEL
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    timeout_seconds: float = 900.0
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        strings = (self.base_url, self.api_key, self.served_model_name, self.checkpoint_id)
        if any(not value.strip() for value in strings):
            raise ValueError("Qwen3.8 recaption backend configuration is incomplete")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("Qwen3.8 recaption runtime limits must be positive")
        sampling_values = (
            self.temperature,
            self.top_p,
            self.min_p,
            self.presence_penalty,
            self.repetition_penalty,
        )
        if (
            any(not math.isfinite(value) for value in sampling_values)
            or self.temperature <= 0
            or not 0 < self.top_p <= 1
            or self.top_k <= 0
            or not 0 <= self.min_p <= 1
            or self.repetition_penalty <= 0
        ):
            raise ValueError("Qwen3.8 recaption sampling configuration is invalid")

    def provenance(self) -> Qwen38BackendProvenance:
        values = {
            "schema_version": QWEN38_RECAPTION_BACKEND_VERSION,
            "backend": "sglang",
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "base_url": self.base_url,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "input_modality": (
                "target_video_observation_plus_frozen_reference_images_plus_audio_text"
            ),
            "output_modalities": ["text"],
            "prompt_version": QWEN38_RECAPTION_PROMPT_VERSION,
            "policy_version": QWEN38_RECAPTION_POLICY_VERSION,
            "draft_schema_version": QWEN38_RECAPTION_DRAFT_VERSION,
            "materializer_version": QWEN38_RECAPTION_MATERIALIZER_VERSION,
            "official_h3_contract_version": OFFICIAL_H3_CONTRACT_VERSION,
            "official_h3_source_files": list(OFFICIAL_H3_SOURCE_FILES),
            "enable_thinking": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "repair_retries": 1,
        }
        return Qwen38BackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


class OpenAIQwen38RecaptionBackend:
    def __init__(self, config: Qwen38RecaptionConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    def provenance(self) -> Qwen38BackendProvenance:
        return self.config.provenance()

    def _request(self, request: Qwen38RecaptionRequest, prompt: str) -> tuple[str, RecaptionCompletionDiagnostic]:
        content: list[dict[str, object]] = [
            {"type": "text", "text": "TARGET VIDEO - observation only, not an H3 reference asset"},
            {
                "type": "video_url",
                "video_url": {
                    "url": self.config.media_resolver.resolve(Path(request.sample.target_video))
                },
            },
        ]
        for picture in request.reference_contract.pictures:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"{picture.picture_label} - canonical Visual image {picture.image_index}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.config.media_resolver.resolve(Path(picture.image_path))
                        },
                    },
                ]
            )
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.config.served_model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "qwen38_h3_recaption_draft",
                    "strict": True,
                    "schema": Qwen38H3DraftResponse.model_json_schema(),
                },
            },
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "modalities": ["text"],
            "extra_body": {
                "top_k": self.config.top_k,
                "min_p": self.config.min_p,
                "repetition_penalty": self.config.repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        completion = self.client.chat.completions.create(**payload)
        choices = getattr(completion, "choices", None)
        if not choices:
            raise TypeError("Qwen3.8 recaption response has no choices")
        choice = choices[0]
        content_value = getattr(choice.message, "content", None)
        if not isinstance(content_value, str):
            raise TypeError("Qwen3.8 recaption response content must be text")
        return content_value, _diagnostic(completion, choice)

    def recaption(self, request: Qwen38RecaptionRequest) -> Qwen38BackendResult:
        raw_responses: list[str] = []
        diagnostics: list[RecaptionCompletionDiagnostic] = []
        issues: list[ValidationIssue] = []
        warnings: list[str] = []
        for attempt in range(2):
            prompt = (
                _user_contract(request)
                if attempt == 0
                else _repair_prompt(
                    request,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            )
            try:
                raw, diagnostic = self._request(request, prompt)
            except Exception as exc:
                raise Qwen38BackendFailure(
                    code="qwen38_request_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    raw_responses=raw_responses,
                    diagnostics=diagnostics,
                    issues=issues,
                    model_call_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            draft, issues = parse_structured_json_issues(raw, Qwen38H3DraftResponse)
            response = None
            if draft is not None:
                issues = validate_h3_draft(draft, request)
            if draft is not None and not issues:
                response = materialize_h3_draft(draft, request)
                issues, warnings = validate_h3_response(response, request)
            if response is not None and not issues:
                return Qwen38BackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    model_call_count=attempt + 1,
                    validation_warnings=tuple(warnings),
                )
        raise Qwen38BackendFailure(
            code="qwen38_structured_output_failed",
            reason="Qwen3.8 H3 recaption failed after one repair",
            raw_responses=raw_responses,
            diagnostics=diagnostics,
            issues=issues,
            model_call_count=2,
        )


def _load_semantics(
    root: Path | None,
) -> tuple[dict[str, SpecializedAudioSemanticsRecord], str | None, str | None]:
    if root is None:
        return {}, None, None
    resolved = root.expanduser().resolve(strict=True)
    candidates = (resolved / "assembled/records.jsonl", resolved / "records.jsonl")
    records_path = next((path for path in candidates if path.is_file()), None)
    if records_path is None:
        raise ValueError("audio-semantics root has no assembled records.jsonl")
    records = [
        SpecializedAudioSemanticsRecord.model_validate(row)
        for row in _read_jsonl(records_path)
    ]
    by_clip = {item.target_clip_uid: item for item in records}
    if len(by_clip) != len(records):
        raise ValueError("audio-semantics records contain duplicate clip IDs")
    return by_clip, str(resolved), _sha256_file(records_path)


def _request_fingerprint(
    *,
    sample: FinalH3SampleV2,
    case: Qwen38RecaptionManifestCase,
    contract: RecaptionReferenceContract,
    facts: RecaptionAudioFacts,
    provenance: Qwen38BackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "sample": sample.model_dump(mode="json"),
                "target_video_sha256": _sha256_file(
                    Path(sample.target_video).expanduser().resolve(strict=True)
                ),
                "case": case.model_dump(mode="json"),
                "reference_contract": contract.model_dump(mode="json"),
                "audio_facts": facts.model_dump(mode="json"),
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
                "prompt_version": QWEN38_RECAPTION_PROMPT_VERSION,
            }
        )
    )


def build_qwen38_pilot_manifest(
    *,
    h3_samples_path: Path,
    output_path: Path,
    size: int,
    conditioning_variant: ConditioningVariant,
) -> list[Qwen38RecaptionManifestCase]:
    if size <= 0:
        raise ValueError("pilot manifest size must be positive")
    samples = [FinalH3SampleV2.model_validate(row) for row in _read_jsonl(h3_samples_path)]
    if size > len(samples):
        raise ValueError("pilot manifest size exceeds current H3 sample inventory")
    cases = [
        Qwen38RecaptionManifestCase(
            sample_id=item.sample_id,
            conditioning_variant=conditioning_variant,
        )
        for item in samples[:size]
    ]
    output = output_path.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"pilot manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, cases)
    return cases


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve(strict=True))
    except OSError:
        shutil.copy2(source, destination)


def _materialize_review_media(
    root: Path,
    records: Sequence[Qwen38RecaptionRecord],
) -> dict[str, tuple[str, dict[int, str]]]:
    result: dict[str, tuple[str, dict[int, str]]] = {}
    for case_index, record in enumerate(records):
        case_root = root / "media" / f"{case_index:04d}"
        target = Path(record.target_video_path)
        target_name = "target" + (target.suffix or ".mp4")
        _link_or_copy(target, case_root / target_name)
        picture_paths: dict[int, str] = {}
        if record.reference_contract is not None:
            for picture in record.reference_contract.pictures:
                source = Path(picture.image_path)
                picture_name = (
                    f"picture-{picture.image_index}" + (source.suffix or ".png")
                )
                _link_or_copy(source, case_root / picture_name)
                picture_paths[picture.image_index] = (
                    f"media/{case_index:04d}/{picture_name}"
                )
        result[record.sample_id] = (
            f"media/{case_index:04d}/{target_name}",
            picture_paths,
        )
    return result


def _review_html(
    records: Sequence[Qwen38RecaptionRecord],
    media_by_sample: dict[str, tuple[str, dict[int, str]]],
) -> str:
    cards = []
    for record in records:
        target_media, picture_media = media_by_sample[record.sample_id]
        pictures = ""
        if record.reference_contract is not None:
            pictures = "".join(
                "<figure><img src='"
                + html.escape(picture_media[item.image_index], quote=True)
                + "'><figcaption>"
                + html.escape(item.picture_label)
                + "</figcaption></figure>"
                for item in record.reference_contract.pictures
            )
        prompt = record.rendered_h3_prompt or "[unavailable]"
        audit = (
            "[unavailable]"
            if record.structured_h3_sections is None
            else json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in record.structured_h3_sections.audio_fact_audit
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        facts = (
            "[unavailable]"
            if record.audio_facts is None
            else json.dumps(record.audio_facts.model_dump(mode="json"), ensure_ascii=False, indent=2)
        )
        contract = (
            "[unavailable]"
            if record.reference_contract is None
            else json.dumps(
                record.reference_contract.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        cards.append(
            "<article><h2>"
            + html.escape(record.sample_id)
            + "</h2><p><b>status:</b> "
            + record.status
            + " &nbsp; <b>variant:</b> "
            + record.conditioning_variant
            + "</p><video controls src='"
            + html.escape(target_media, quote=True)
            + "'></video><div class='pictures'>"
            + pictures
            + "</div><details><summary>Reference contract</summary><pre>"
            + html.escape(contract)
            + "</pre></details><details><summary>Rough Visual instruction</summary><pre>"
            + html.escape(record.rough_r2v_instruction)
            + "</pre></details><details><summary>Audio facts</summary><pre>"
            + html.escape(facts)
            + "</pre></details><h3>Rendered H3 prompt</h3><pre>"
            + html.escape(prompt)
            + "</pre><h3>Audio fact audit</h3><pre>"
            + html.escape(audit)
            + "</pre><h3>Warnings</h3><pre>"
            + html.escape("\n".join(record.validation_warnings) or "[none]")
            + "</pre></article>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>Qwen3.8 H3 recaption pilot</title><style>
body{font:14px system-ui;margin:24px;background:#f4f4f1;color:#171717}article{background:white;border:1px solid #ccc;margin:0 0 24px;padding:18px;max-width:1200px}video{width:min(720px,100%)}.pictures{display:flex;gap:12px;overflow:auto}.pictures figure{margin:12px 0}.pictures img{height:180px;max-width:300px;object-fit:contain;background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f7f7;padding:12px}</style></head><body><h1>Qwen3.8 H3 recaption pilot</h1>""" + "".join(cards) + "</body></html>"


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"recaption output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_qwen38_h3_recaption_pilot(
    *,
    audio_production_root: Path,
    case_manifest_path: Path,
    backend: Qwen38RecaptionBackend,
    audio_semantics_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> Qwen38RecaptionSummary:
    paths = jea_production_paths(audio_production_root)
    samples_path = paths.h3 / "samples.jsonl"
    if not samples_path.is_file():
        raise ValueError("current H3 samples.jsonl is unavailable")
    samples = [FinalH3SampleV2.model_validate(row) for row in _read_jsonl(samples_path)]
    sample_by_id = {item.sample_id: item for item in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("current H3 sample inventory has duplicate sample IDs")
    manifest_path = case_manifest_path.expanduser().resolve(strict=True)
    cases = [
        Qwen38RecaptionManifestCase.model_validate(row)
        for row in _read_jsonl(manifest_path)
    ]
    case_ids = [item.sample_id for item in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("recaption case manifest must be non-empty and unique")
    if any(sample_id not in sample_by_id for sample_id in case_ids):
        raise ValueError("recaption manifest references an unknown H3 sample")
    semantics_by_clip, semantics_root_value, semantics_sha = _load_semantics(
        audio_semantics_root
    )
    destination = (
        output_root.expanduser().resolve(strict=False)
        if output_root is not None
        else paths.root / OUTPUT_DIRECTORY_NAME
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"recaption output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[Qwen38RecaptionRecord] = []
    raw_payloads: list[tuple[str, dict[str, object]]] = []
    try:
        temporary.mkdir()
        (temporary / "raw_responses").mkdir()
        for index, case in enumerate(cases):
            sample = sample_by_id[case.sample_id]
            target_path = Path(sample.target_video).expanduser().resolve(strict=True)
            target_sha = _sha256_file(target_path)
            contract: RecaptionReferenceContract | None = None
            facts: RecaptionAudioFacts | None = None
            try:
                contract = build_reference_contract(sample, case.conditioning_variant)
                facts = build_audio_facts(
                    sample,
                    contract,
                    semantics_by_clip.get(sample.clip_uid),
                    semantics_records_sha256=semantics_sha,
                )
                fingerprint = _request_fingerprint(
                    sample=sample,
                    case=case,
                    contract=contract,
                    facts=facts,
                    provenance=backend.provenance,
                )
                request = Qwen38RecaptionRequest(
                    sample=sample,
                    case=case,
                    reference_contract=contract,
                    audio_facts=facts,
                    request_fingerprint=fingerprint,
                )
            except (OSError, ValueError) as exc:
                record = Qwen38RecaptionRecord(
                    status="unsupported",
                    sample_id=case.sample_id,
                    clip_uid=sample.clip_uid,
                    conditioning_variant=case.conditioning_variant,
                    note=case.note,
                    target_video_path=str(target_path),
                    target_video_sha256=target_sha,
                    rough_r2v_instruction=sample.r2v_instruction,
                    reference_contract=contract,
                    audio_facts=facts,
                    audio_fact_provenance={} if facts is None else facts.provenance,
                    audio_grounding_complete=False if facts is None else facts.audio_grounding_complete,
                    backend_provenance=backend.provenance,
                    model_call_count=0,
                    raw_response_count=0,
                    failure=RecaptionFailure(
                        code="unsupported_reference_contract",
                        reason=f"{type(exc).__name__}: {exc}",
                        attempt_count=0,
                    ),
                )
                records.append(record)
                continue
            try:
                result = backend.recaption(request)
                rendered = render_h3_prompt(result.response)
                record = Qwen38RecaptionRecord(
                    status="ready",
                    sample_id=case.sample_id,
                    clip_uid=sample.clip_uid,
                    conditioning_variant=case.conditioning_variant,
                    note=case.note,
                    target_video_path=str(target_path),
                    target_video_sha256=target_sha,
                    rough_r2v_instruction=sample.r2v_instruction,
                    reference_contract=contract,
                    audio_facts=facts,
                    audio_fact_provenance=facts.provenance,
                    audio_grounding_complete=facts.audio_grounding_complete,
                    backend_provenance=backend.provenance,
                    request_fingerprint=fingerprint,
                    structured_h3_sections=result.response,
                    rendered_h3_prompt=rendered,
                    detailed_description_word_count=len(
                        re.findall(r"\b[\w'-]+\b", result.response.detailed_description)
                    ),
                    validation_warnings=list(result.validation_warnings),
                    model_call_count=result.model_call_count,
                    raw_response_count=len(result.raw_responses),
                    completion_diagnostics=list(result.diagnostics),
                )
                raw_payload = {
                    "sample_id": case.sample_id,
                    "status": "ready",
                    "raw_responses": list(result.raw_responses),
                    "completion_diagnostics": [
                        item.model_dump(mode="json") for item in result.diagnostics
                    ],
                }
            except Qwen38BackendFailure as exc:
                record = Qwen38RecaptionRecord(
                    status="failed",
                    sample_id=case.sample_id,
                    clip_uid=sample.clip_uid,
                    conditioning_variant=case.conditioning_variant,
                    note=case.note,
                    target_video_path=str(target_path),
                    target_video_sha256=target_sha,
                    rough_r2v_instruction=sample.r2v_instruction,
                    reference_contract=contract,
                    audio_facts=facts,
                    audio_fact_provenance=facts.provenance,
                    audio_grounding_complete=facts.audio_grounding_complete,
                    backend_provenance=backend.provenance,
                    request_fingerprint=fingerprint,
                    model_call_count=exc.model_call_count,
                    raw_response_count=len(exc.raw_responses),
                    completion_diagnostics=list(exc.diagnostics),
                    failure=RecaptionFailure(
                        code=exc.code,
                        reason=exc.reason,
                        attempt_count=exc.model_call_count,
                        issues=[item.to_dict() for item in exc.issues],
                    ),
                )
                raw_payload = {
                    "sample_id": case.sample_id,
                    "status": "failed",
                    "raw_responses": list(exc.raw_responses),
                    "completion_diagnostics": [
                        item.model_dump(mode="json") for item in exc.diagnostics
                    ],
                    "issues": [item.to_dict() for item in exc.issues],
                }
            records.append(record)
            raw_payloads.append((f"{index:04d}-{_sha256_text(case.sample_id)[:12]}.json", raw_payload))

        failure_counts = Counter(
            item.failure.code for item in records if item.failure is not None
        )
        summary = Qwen38RecaptionSummary(
            source_h3_samples_path=str(samples_path.resolve(strict=True)),
            source_h3_samples_sha256=_sha256_file(samples_path),
            case_manifest_sha256=_sha256_file(manifest_path),
            audio_semantics_root=semantics_root_value,
            audio_semantics_records_sha256=semantics_sha,
            backend_provenance=backend.provenance,
            case_count=len(records),
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            unsupported_count=sum(item.status == "unsupported" for item in records),
            model_call_count=sum(item.model_call_count for item in records),
            raw_response_count=sum(item.raw_response_count for item in records),
            conditioning_variant_counts=dict(
                sorted(Counter(item.conditioning_variant for item in records).items())
            ),
            failure_counts=dict(sorted(failure_counts.items())),
        )
        _write_jsonl(temporary / "manifest.jsonl", cases)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        for name, payload in raw_payloads:
            _write_json(temporary / "raw_responses" / name, payload)
        review_media = _materialize_review_media(temporary, records)
        (temporary / "review.html").write_text(
            _review_html(records, review_media), encoding="utf-8"
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "DEFAULT_CHECKPOINT_ID",
    "DEFAULT_MODEL",
    "OFFICIAL_H3_SOURCE_FILES",
    "OUTPUT_DIRECTORY_NAME",
    "QWEN38_RECAPTION_DRAFT_VERSION",
    "QWEN38_RECAPTION_MATERIALIZER_VERSION",
    "QWEN38_RECAPTION_POLICY_VERSION",
    "QWEN38_RECAPTION_PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "AudioFactAuditItem",
    "OpenAIQwen38RecaptionBackend",
    "Qwen38BackendFailure",
    "Qwen38BackendResult",
    "Qwen38DraftShot",
    "Qwen38H3DraftResponse",
    "Qwen38H3StructuredResponse",
    "Qwen38RecaptionConfig",
    "Qwen38RecaptionManifestCase",
    "Qwen38RecaptionRequest",
    "RecaptionAudioFacts",
    "RecaptionNonSpeechFact",
    "RecaptionReferenceContract",
    "RecaptionSpeechFact",
    "build_audio_facts",
    "build_qwen38_pilot_manifest",
    "build_reference_contract",
    "materialize_h3_draft",
    "render_h3_prompt",
    "run_qwen38_h3_recaption_pilot",
    "validate_h3_draft",
    "validate_h3_response",
]
