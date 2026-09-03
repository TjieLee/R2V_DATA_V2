from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_backend import SpeechPresentation
from r2v_data_v2.h3.qwen38_h3_recaption import (
    OFFICIAL_H3_CONTRACT_VERSION,
    OFFICIAL_H3_SOURCE_FILES,
    UNGROUNDED_NON_DIEGETIC_MUSIC,
    UNGROUNDED_OVERALL_SOUNDSCAPE,
    Qwen38H3DraftResponse,
    Qwen38H3StructuredResponse,
    Qwen38RecaptionManifestCase,
    Qwen38RecaptionRequest,
    RecaptionAudioFacts,
    RecaptionCompletionDiagnostic,
    RecaptionFailure,
    RecaptionReferenceContract,
    build_audio_facts,
    build_reference_contract,
    materialize_h3_draft,
    render_h3_prompt,
    validate_h3_draft,
    validate_h3_response,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.speech_presentation import (
    render_speech_presentation_clause,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

QWEN_PRESENTATION_PROMPT_VERSION = "h3_qwen_ref2va_speech_presentation_v1"
QWEN_PRESENTATION_POLICY_VERSION = (
    "h3_qwen_ref2va_speech_presentation_contract_v1"
)
QWEN_PRESENTATION_DRAFT_VERSION = "r2v.h3.qwen_speech_presentation_draft.1"
QWEN_PRESENTATION_BACKEND_VERSION = "r2v.h3.qwen_speech_presentation_backend.1"
QWEN_PRESENTATION_RECORD_VERSION = "r2v.h3.qwen_speech_presentation_record.1"
QWEN_PRESENTATION_SUMMARY_VERSION = "r2v.h3.qwen_speech_presentation_summary.1"
QWEN_PRESENTATION_MATERIALIZER_VERSION = (
    "h3_qwen_speech_presentation_materializer_v1"
)

PresentationEvidenceCode = Literal[
    "visible_lip_motion",
    "no_visible_lip_motion",
    "message_text_alignment",
    "device_playback_context",
    "offscreen_visual_context",
    "voice_over_visual_context",
    "visual_insufficient",
]
PresentationStatus = Literal["ready", "failed", "unsupported"]
_PRESENTATION_VALUES: tuple[SpeechPresentation, ...] = (
    "onscreen_spoken",
    "offscreen_spoken",
    "voice_over",
    "message_voice_over",
    "device_playback",
    "uncertain",
)


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
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
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


class QwenSpeechPresentationDecision(SchemaModel):
    fact_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    speech_presentation: SpeechPresentation
    visible_entity_id: str | None = None
    evidence_codes: list[PresentationEvidenceCode] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]

    @model_validator(mode="after")
    def validate_decision(self) -> QwenSpeechPresentationDecision:
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("speech-presentation evidence codes must be unique")
        if self.speech_presentation == "onscreen_spoken":
            if "visible_lip_motion" not in self.evidence_codes:
                raise ValueError("onscreen_spoken requires visible_lip_motion")
        elif self.visible_entity_id is not None:
            raise ValueError("non-onscreen speech cannot claim a visible entity")
        if self.visible_entity_id is not None and (
            self.speech_presentation != "onscreen_spoken"
            or "visible_lip_motion" not in self.evidence_codes
        ):
            raise ValueError("visible entity requires observed onscreen lip motion")
        return self


class QwenPresentationAwareDraftResponse(Qwen38H3DraftResponse):
    schema_version: Literal["r2v.h3.qwen_speech_presentation_draft.1"] = (
        QWEN_PRESENTATION_DRAFT_VERSION
    )
    speech_presentations: list[QwenSpeechPresentationDecision]


class QwenPresentationBackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.qwen_speech_presentation_backend.1"] = (
        QWEN_PRESENTATION_BACKEND_VERSION
    )
    backend: Literal["sglang"] = "sglang"
    served_model_name: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    media_mode: Literal["file", "http"]
    media_root: str = Field(min_length=1)
    media_base_url: str | None = None
    input_modality: Literal[
        "target_video_observation_plus_frozen_reference_images_plus_locked_speech_text"
    ] = "target_video_observation_plus_frozen_reference_images_plus_locked_speech_text"
    audio_acoustic_observation: Literal[False] = False
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    prompt_version: Literal["h3_qwen_ref2va_speech_presentation_v1"] = (
        QWEN_PRESENTATION_PROMPT_VERSION
    )
    policy_version: Literal["h3_qwen_ref2va_speech_presentation_contract_v1"] = (
        QWEN_PRESENTATION_POLICY_VERSION
    )
    draft_schema_version: Literal["r2v.h3.qwen_speech_presentation_draft.1"] = (
        QWEN_PRESENTATION_DRAFT_VERSION
    )
    materializer_version: Literal[
        "h3_qwen_speech_presentation_materializer_v1"
    ] = QWEN_PRESENTATION_MATERIALIZER_VERSION
    official_h3_contract_version: str = OFFICIAL_H3_CONTRACT_VERSION
    official_h3_source_files: list[str] = Field(
        default_factory=lambda: list(OFFICIAL_H3_SOURCE_FILES)
    )
    enable_thinking: Literal[False] = False
    temperature: float = Field(gt=0, allow_inf_nan=False)
    top_p: float = Field(gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(gt=0)
    min_p: float = Field(ge=0, le=1, allow_inf_nan=False)
    presence_penalty: float = Field(allow_inf_nan=False)
    repetition_penalty: float = Field(gt=0, allow_inf_nan=False)
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> QwenPresentationBackendProvenance:
        if (self.media_mode == "http") != (self.media_base_url is not None):
            raise ValueError("HTTP media mode requires one media base URL")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("Qwen presentation backend fingerprint is invalid")
        return self


@dataclass(frozen=True)
class QwenPresentationRequest:
    sample: FinalH3SampleV2
    case: Qwen38RecaptionManifestCase
    reference_contract: RecaptionReferenceContract
    original_audio_facts: RecaptionAudioFacts
    request_fingerprint: str


@dataclass(frozen=True)
class QwenPresentationBackendResult:
    draft: QwenPresentationAwareDraftResponse
    corrected_audio_facts: RecaptionAudioFacts
    response: Qwen38H3StructuredResponse
    raw_responses: tuple[str, ...]
    diagnostics: tuple[RecaptionCompletionDiagnostic, ...]
    model_call_count: int
    validation_warnings: tuple[str, ...]


class QwenPresentationBackendFailure(ValueError):
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


class QwenPresentationBackend(Protocol):
    @property
    def provenance(self) -> QwenPresentationBackendProvenance: ...

    def recaption(
        self, request: QwenPresentationRequest
    ) -> QwenPresentationBackendResult: ...


class QwenPresentationRecord(SchemaModel):
    schema_version: Literal["r2v.h3.qwen_speech_presentation_record.1"] = (
        QWEN_PRESENTATION_RECORD_VERSION
    )
    sample_id: str
    clip_uid: str
    conditioning_variant: Literal["visual_only"] = "visual_only"
    status: PresentationStatus
    note: str | None = None
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: QwenPresentationBackendProvenance
    request_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_contract: RecaptionReferenceContract | None = None
    original_audio_facts: RecaptionAudioFacts | None = None
    speech_presentation_decisions: list[QwenSpeechPresentationDecision]
    corrected_audio_facts: RecaptionAudioFacts | None = None
    structured_h3_sections: Qwen38H3StructuredResponse | None = None
    rendered_h3_prompt: str | None = None
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    completion_diagnostics: list[RecaptionCompletionDiagnostic] = Field(
        default_factory=list
    )
    validation_warnings: list[str] = Field(default_factory=list)
    failure: RecaptionFailure | None = None
    production_h3_modified: Literal[False] = False
    mimo_annotation_created: Literal[False] = False
    speaker_identity_modified: Literal[False] = False
    asr_text_modified: Literal[False] = False
    asr_language_modified: Literal[False] = False
    speech_timing_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_record(self) -> QwenPresentationRecord:
        if self.raw_response_count != len(self.completion_diagnostics):
            raise ValueError("Qwen presentation response diagnostics differ")
        outputs = (
            self.corrected_audio_facts,
            self.structured_h3_sections,
            self.rendered_h3_prompt,
        )
        if self.status == "ready":
            if (
                self.request_fingerprint is None
                or self.reference_contract is None
                or self.original_audio_facts is None
                or not self.speech_presentation_decisions
                and self.original_audio_facts.speech
                or any(value is None for value in outputs)
                or self.failure is not None
            ):
                raise ValueError("ready Qwen presentation record is incomplete")
        elif any(value is not None for value in outputs) or self.failure is None:
            raise ValueError("non-ready Qwen presentation record cannot publish output")
        if self.status == "unsupported" and (
            self.model_call_count or self.raw_response_count
        ):
            raise ValueError("unsupported Qwen presentation case cannot claim model work")
        return self


class QwenPresentationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.qwen_speech_presentation_summary.1"] = (
        QWEN_PRESENTATION_SUMMARY_VERSION
    )
    source_h3_samples_path: str
    source_h3_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_manifest_path: str
    case_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: QwenPresentationBackendProvenance
    case_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unsupported_count: int = Field(ge=0)
    first_pass_ready_count: int = Field(ge=0)
    repaired_ready_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    served_model_name: str
    checkpoint_id: str
    presentation_counts: dict[str, int]
    current_visible_entity_binding_count: int = Field(ge=0)
    predicted_visible_entity_binding_count: int = Field(ge=0)
    visible_entity_binding_removed_count: int = Field(ge=0)
    visible_entity_binding_added_count: int = Field(ge=0)
    visible_entity_binding_changed_count: int = Field(ge=0)
    non_onscreen_speech_count: int = Field(ge=0)
    failure_counts: dict[str, int]
    production_h3_modified: Literal[False] = False
    mimo_annotation_created: Literal[False] = False
    speaker_identity_modified: Literal[False] = False
    asr_text_modified: Literal[False] = False
    asr_language_modified: Literal[False] = False
    speech_timing_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> QwenPresentationSummary:
        if self.case_count != self.ready_count + self.failed_count + self.unsupported_count:
            raise ValueError("Qwen presentation summary counts do not reconcile")
        if self.ready_count != self.first_pass_ready_count + self.repaired_ready_count:
            raise ValueError("Qwen presentation ready counts do not reconcile")
        return self


SYSTEM_PROMPT = """You are a MiniMax H3 full-reference (Ref2VA) recaptioner and a
precision-first visual speech-presentation observer.

Observe the TARGET VIDEO and frozen reference images. Produce the same complete,
generation-quality H3 visual draft contract as the supplied schema, plus exactly
one speech-presentation decision for every locked speech fact. The target video is
observation-only and never becomes <Video N>. Pictures are content references, not
keyframes. The pipeline owns all final dialogue clauses and inserts exact ASR text
deterministically; never transcribe, translate, paraphrase, quote, correct, or copy
dialogue into the draft.

Speaker identity, visible entity identity, and audiovisual speech presentation are
three different questions. A visible person is NOT a visible speaker unless
synchronized speech-like mouth, lip, or jaw articulation is visually observed
during that authoritative segment. A current entity binding is only a fallible
proposal. If no synchronized articulation is visible, set visible_entity_id to
null even when the proposal names a visible person. This visual experiment does
not infer speaker identity from voice and provides no acoustic observation.

Use onscreen_spoken only with visible_lip_motion evidence. A visible_entity_id may
name only a supplied entity and only for onscreen_spoken. For offscreen_spoken,
voice_over, message_voice_over, device_playback, or uncertain, visible_entity_id
must be null. If visual evidence cannot reliably distinguish offscreen speech,
voice-over, and device playback, use uncertain rather than guessing.

Use message_voice_over when clear messaging visuals align with the speech segment:
a chat interface or message text appears or changes, or a visible person silently
reads, types, or sends a message without synchronized mouth motion. A person merely
looking at a phone, typing, reacting to a message, or listening must not become an
onscreen speaker. Use device_playback only for unmistakable visible playback
context. Never claim acoustic evidence that this request does not provide.

The visual draft must respect each decision. For non-onscreen or uncertain speech,
describe only observed visual action around its placeholder; never say that a
visible person speaks, says, replies verbally, talks, whispers, shouts, utters,
lip-syncs, or opens their mouth to say the dialogue merely because speech exists.
For example, describe a person looking at a phone and the interface changing, then
place [[speech_2]]; do not write that the person says [[speech_2]].

Keep all supplied Picture/Subject labels and the required summary prefix. Use only
fully_preserved, partially_preserved, or weak_reference retention markers. Never
emit attribute_transfer or any <Audio N> label. Insert every supplied [[speech_N]]
placeholder exactly once, in chronological order, only in shot descriptions. A
placeholder is the complete source-and-dialogue clause: do not prefix it with a
speaker, name, pronoun, says/speaks wording, (Sx), <Subject N> (Sx), or <d> text.
Do not alter fact IDs, segment IDs, speakers, timestamps, language, or transcript.

Write a concrete generation-quality visual account: style, framing, angle,
foreground/midground/background composition, salient appearances, positions,
poses, body/hand/head motion, gaze, visible expression, interactions, object and
environment state, materials, readable text, lighting, color, camera motion or
stability, and temporal progression when observed. Do not invent psychology,
intent, causality, relationships, unseen events, sounds from visible actions, or
unsupported object detail. Audio fields remain conservative because no acoustic
observation is supplied.

Return one compact JSON object matching the strict schema. No markdown,
explanation, or chain-of-thought."""


def _model_speech_contract(request: QwenPresentationRequest) -> list[dict[str, object]]:
    return [
        {
            "fact_id": item.fact_id,
            "segment_id": item.segment_id,
            "speaker_id": item.speaker_id,
            "speaker_cluster_id": item.speaker_cluster_id,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "text": item.text,
            "language": item.language,
            "current_entity_id": item.entity_id,
            "current_entity_subject_label": item.entity_subject_label,
        }
        for item in request.original_audio_facts.speech
    ]


def _reference_contract_text(contract: RecaptionReferenceContract) -> str:
    payload = contract.model_dump(mode="json")
    for picture in payload["pictures"]:
        picture.pop("image_path")
        picture.pop("image_sha256")
    return _compact_json(payload)


def _user_contract(request: QwenPresentationRequest) -> str:
    expected = [
        [item.fact_id, item.segment_id] for item in request.original_audio_facts.speech
    ]
    return (
        "TARGET VIDEO is observation-only and must not become <Video N>.\n"
        "conditioning_variant=visual_only\n"
        "required_summary_prefix=[reference generation]\n"
        "REFERENCE CONTRACT (immutable):\n"
        f"{_reference_contract_text(request.reference_contract)}\n"
        "ROUGH VISUAL HINT (non-authoritative):\n"
        f"r2v_instruction={request.sample.r2v_instruction}\n"
        "LOCKED SPEECH FACTS plus CURRENT FALLIBLE VISIBLE-ENTITY PROPOSALS:\n"
        f"{_compact_json(_model_speech_contract(request))}\n"
        "The locked fields are fact_id, segment_id, speaker_id, speaker_cluster_id, "
        "start_time, end_time, exact text, and exact language. Only the separate "
        "presentation decision may choose visible_entity_id.\n"
        f"Required decision order={_compact_json(expected)}\n"
        "Return exactly one decision per listed pair, in exactly this order.\n"
        "Audio grounding is incomplete: do not assert non-speech audio.\n"
        f"overall_soundscape MUST be exactly: {UNGROUNDED_OVERALL_SOUNDSCAPE}\n"
        f"non_diegetic_music MUST be exactly: {UNGROUNDED_NON_DIEGETIC_MUSIC}\n"
        "audio_fact_audit MUST be exactly [].\n"
        "Return this strict schema:\n"
        f"{_compact_json(QwenPresentationAwareDraftResponse.model_json_schema())}"
    )


def _repair_prompt(
    request: QwenPresentationRequest,
    *,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        _user_contract(request)
        + "\nRepair only the listed schema or contract errors. Preserve visual meaning, "
        "locked facts, exact decision inventory/order, labels, shots, and placeholders. "
        "Return one compact JSON object only.\n"
        + f"issues={_compact_json([item.to_dict() for item in issues])}\n"
        + f"invalid_response={invalid_response}"
    )


def validate_presentation_draft(
    draft: QwenPresentationAwareDraftResponse,
    request: QwenPresentationRequest,
) -> list[ValidationIssue]:
    parent_request = _as_qwen_request(request, request.original_audio_facts)
    issues = validate_h3_draft(draft, parent_request)
    expected = [
        (item.fact_id, item.segment_id) for item in request.original_audio_facts.speech
    ]
    actual = [(item.fact_id, item.segment_id) for item in draft.speech_presentations]
    if actual != expected:
        issues.append(
            ValidationIssue(
                "speech_presentation_inventory_mismatch",
                "speech_presentations",
                "decisions must exactly match authoritative speech fact order",
            )
        )
    supplied_entities = {
        item.entity_id
        for item in request.reference_contract.subjects
        if item.kind == "entity" and item.entity_id is not None
    }
    for decision in draft.speech_presentations:
        if (
            decision.visible_entity_id is not None
            and decision.visible_entity_id not in supplied_entities
        ):
            issues.append(
                ValidationIssue(
                    "unknown_visible_entity",
                    "speech_presentations",
                    f"{decision.segment_id} names an unsupplied visible entity",
                )
            )
    return issues


def build_corrected_audio_facts(
    request: QwenPresentationRequest,
    decisions: Sequence[QwenSpeechPresentationDecision],
) -> RecaptionAudioFacts:
    if [(item.fact_id, item.segment_id) for item in decisions] != [
        (item.fact_id, item.segment_id) for item in request.original_audio_facts.speech
    ]:
        raise ValueError("speech presentation decisions differ from authoritative facts")
    entity_subjects = {
        item.entity_id: item.subject_label
        for item in request.reference_contract.subjects
        if item.kind == "entity" and item.entity_id is not None
    }
    corrected = []
    for original, decision in zip(
        request.original_audio_facts.speech, decisions, strict=True
    ):
        entity_id = (
            decision.visible_entity_id
            if decision.speech_presentation == "onscreen_spoken"
            else None
        )
        corrected.append(
            original.model_copy(
                update={
                    "entity_id": entity_id,
                    "entity_subject_label": (
                        None if entity_id is None else entity_subjects[entity_id]
                    ),
                }
            )
        )
    facts = request.original_audio_facts.model_copy(update={"speech": corrected})
    for original, updated in zip(
        request.original_audio_facts.speech, facts.speech, strict=True
    ):
        if (
            original.fact_id != updated.fact_id
            or original.segment_id != updated.segment_id
            or original.speaker_cluster_id != updated.speaker_cluster_id
            or original.speaker_id != updated.speaker_id
            or original.start_time != updated.start_time
            or original.end_time != updated.end_time
            or original.text != updated.text
            or original.language != updated.language
            or original.delivery != updated.delivery
            or original.locked_dialogue_block != updated.locked_dialogue_block
        ):
            raise ValueError("corrected speech facts modified an authoritative field")
    return facts


def _as_qwen_request(
    request: QwenPresentationRequest,
    facts: RecaptionAudioFacts,
) -> Qwen38RecaptionRequest:
    return Qwen38RecaptionRequest(
        sample=request.sample,
        case=request.case,
        reference_contract=request.reference_contract,
        audio_facts=facts,
        request_fingerprint=request.request_fingerprint,
    )


def materialize_presentation_draft(
    draft: QwenPresentationAwareDraftResponse,
    request: QwenPresentationRequest,
) -> tuple[RecaptionAudioFacts, Qwen38H3StructuredResponse, list[str]]:
    issues = validate_presentation_draft(draft, request)
    if issues:
        raise ValueError(_compact_json([item.to_dict() for item in issues]))
    corrected = build_corrected_audio_facts(request, draft.speech_presentations)
    corrected_request = _as_qwen_request(request, corrected)
    presentation_by_fact = {
        item.fact_id: item.speech_presentation
        for item in draft.speech_presentations
    }
    structured = materialize_h3_draft(
        draft,
        corrected_request,
        speech_clause_transform=lambda speech, clause: (
            render_speech_presentation_clause(
                speech=speech,
                base_clause=clause,
                presentation=presentation_by_fact[speech.fact_id],
            )
        ),
    )
    response_issues, warnings = validate_h3_response(structured, corrected_request)
    if response_issues:
        raise ValueError(_compact_json([item.to_dict() for item in response_issues]))
    return corrected, structured, warnings


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
class QwenPresentationConfig:
    base_url: str
    media_resolver: MediaURLResolver
    api_key: str = "EMPTY"
    served_model_name: str = "Qwen/Qwen3.8-Flash-Next"
    checkpoint_id: str = "/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next"
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
            raise ValueError("Qwen presentation backend configuration is incomplete")
        numbers = (
            self.timeout_seconds,
            self.temperature,
            self.top_p,
            self.min_p,
            self.presence_penalty,
            self.repetition_penalty,
        )
        if (
            any(not math.isfinite(value) for value in numbers)
            or self.timeout_seconds <= 0
            or self.max_tokens <= 0
            or self.temperature <= 0
            or not 0 < self.top_p <= 1
            or self.top_k <= 0
            or not 0 <= self.min_p <= 1
            or self.repetition_penalty <= 0
        ):
            raise ValueError("Qwen presentation sampling configuration is invalid")

    def provenance(self) -> QwenPresentationBackendProvenance:
        values = {
            "schema_version": QWEN_PRESENTATION_BACKEND_VERSION,
            "backend": "sglang",
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "base_url": self.base_url,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "input_modality": (
                "target_video_observation_plus_frozen_reference_images_plus_locked_speech_text"
            ),
            "audio_acoustic_observation": False,
            "output_modalities": ["text"],
            "prompt_version": QWEN_PRESENTATION_PROMPT_VERSION,
            "policy_version": QWEN_PRESENTATION_POLICY_VERSION,
            "draft_schema_version": QWEN_PRESENTATION_DRAFT_VERSION,
            "materializer_version": QWEN_PRESENTATION_MATERIALIZER_VERSION,
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
        return QwenPresentationBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


class OpenAIQwenPresentationBackend:
    def __init__(
        self,
        config: QwenPresentationConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    def provenance(self) -> QwenPresentationBackendProvenance:
        return self.config.provenance()

    def _request(
        self,
        request: QwenPresentationRequest,
        prompt: str,
    ) -> tuple[str, RecaptionCompletionDiagnostic]:
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "TARGET VIDEO - observation only, not an H3 reference asset",
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": self.config.media_resolver.resolve(
                        Path(request.sample.target_video)
                    )
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
                            "url": self.config.media_resolver.resolve(
                                Path(picture.image_path)
                            )
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
                    "name": "qwen_speech_presentation_draft",
                    "strict": True,
                    "schema": QwenPresentationAwareDraftResponse.model_json_schema(),
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
            raise TypeError("Qwen presentation response has no choices")
        choice = choices[0]
        response_text = getattr(choice.message, "content", None)
        if not isinstance(response_text, str):
            raise TypeError("Qwen presentation response content must be text")
        return response_text, _diagnostic(completion, choice)

    def recaption(
        self, request: QwenPresentationRequest
    ) -> QwenPresentationBackendResult:
        raw_responses: list[str] = []
        diagnostics: list[RecaptionCompletionDiagnostic] = []
        issues: list[ValidationIssue] = []
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
                raise QwenPresentationBackendFailure(
                    code="qwen_presentation_request_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    raw_responses=raw_responses,
                    diagnostics=diagnostics,
                    issues=issues,
                    model_call_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            draft, issues = parse_structured_json_issues(
                raw, QwenPresentationAwareDraftResponse
            )
            corrected = None
            structured = None
            warnings: list[str] = []
            if draft is not None:
                issues = validate_presentation_draft(draft, request)
            if draft is not None and not issues:
                try:
                    corrected, structured, warnings = materialize_presentation_draft(
                        draft, request
                    )
                except ValueError as exc:
                    issues = [
                        ValidationIssue(
                            "presentation_materialization_failed", None, str(exc)
                        )
                    ]
            if (
                draft is not None
                and corrected is not None
                and structured is not None
                and not issues
            ):
                return QwenPresentationBackendResult(
                    draft=draft,
                    corrected_audio_facts=corrected,
                    response=structured,
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    model_call_count=attempt + 1,
                    validation_warnings=tuple(warnings),
                )
        raise QwenPresentationBackendFailure(
            code="qwen_presentation_structured_output_failed",
            reason="Qwen presentation recaption failed after one repair",
            raw_responses=raw_responses,
            diagnostics=diagnostics,
            issues=issues,
            model_call_count=2,
        )


def _canonical_speech_facts(
    sample: FinalH3SampleV2,
    contract: RecaptionReferenceContract,
) -> RecaptionAudioFacts:
    built = build_audio_facts(
        sample,
        contract,
        None,
        semantics_records_sha256=None,
    )
    speech = [item.model_copy(update={"delivery": None}) for item in built.speech]
    return RecaptionAudioFacts(
        speech=speech,
        non_speech_events=[],
        overall_soundscape_hint=None,
        non_diegetic_music_hint=None,
        audio_grounding_complete=False,
        provenance={"speech": "current_h3_final_sample.speech_segments"},
    )


def _request_fingerprint(
    *,
    sample: FinalH3SampleV2,
    case: Qwen38RecaptionManifestCase,
    contract: RecaptionReferenceContract,
    facts: RecaptionAudioFacts,
    provenance: QwenPresentationBackendProvenance,
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
                "prompt_version": QWEN_PRESENTATION_PROMPT_VERSION,
            }
        )
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve(strict=True))
    except OSError:
        shutil.copy2(source, destination)


def _materialize_review_media(
    root: Path,
    records: Sequence[QwenPresentationRecord],
) -> dict[str, dict[str, object]]:
    media_by_sample: dict[str, dict[str, object]] = {}
    for record in records:
        sample_key = _sha256_text(record.sample_id)
        case_root = root / "media" / sample_key
        target = Path(record.target_video_path).expanduser().resolve(strict=True)
        if _sha256_file(target) != record.target_video_sha256:
            raise ValueError("Qwen presentation review target video differs from record")
        target_name = f"target-{record.target_video_sha256[:12]}{target.suffix or '.mp4'}"
        _link_or_copy(target, case_root / target_name)
        pictures: dict[int, str] = {}
        if record.reference_contract is not None:
            for picture in record.reference_contract.pictures:
                source = Path(picture.image_path).expanduser().resolve(strict=True)
                if _sha256_file(source) != picture.image_sha256:
                    raise ValueError("Qwen presentation review Picture differs from record")
                name = (
                    f"picture-{picture.image_index}-{picture.image_sha256[:12]}"
                    f"{source.suffix or '.png'}"
                )
                _link_or_copy(source, case_root / name)
                pictures[picture.image_index] = f"media/{sample_key}/{name}"
        media_by_sample[record.sample_id] = {
            "target": f"media/{sample_key}/{target_name}",
            "pictures": pictures,
        }
    return media_by_sample


def render_review_html(
    records: Sequence[QwenPresentationRecord],
    media_by_sample: dict[str, dict[str, object]],
) -> str:
    cards = []
    for record in records:
        media = media_by_sample[record.sample_id]
        picture_paths = media["pictures"]
        pictures = "" if record.reference_contract is None else "".join(
            "<figure><img loading='lazy' src='"
            + html.escape(picture_paths[item.image_index], quote=True)
            + "'><figcaption>"
            + html.escape(item.picture_label)
            + " · "
            + html.escape(item.kind)
            + " · entity="
            + html.escape(str(item.entity_id))
            + "</figcaption></figure>"
            for item in record.reference_contract.pictures
        )
        original = (
            [] if record.original_audio_facts is None else record.original_audio_facts.speech
        )
        decisions = {
            (item.fact_id, item.segment_id): item
            for item in record.speech_presentation_decisions
        }
        rows = []
        for speech in original:
            decision = decisions.get((speech.fact_id, speech.segment_id))
            rows.append(
                "<tr><td>"
                + html.escape(speech.fact_id)
                + "</td><td>"
                + html.escape(speech.segment_id)
                + "</td><td>"
                + f"{speech.start_time:.3f}-{speech.end_time:.3f}"
                + "</td><td>"
                + html.escape(speech.speaker_id)
                + "</td><td>"
                + html.escape(speech.text)
                + "</td><td>"
                + html.escape(str(speech.entity_id))
                + "</td><td>"
                + html.escape("-" if decision is None else decision.speech_presentation)
                + "</td><td>"
                + html.escape("-" if decision is None else str(decision.visible_entity_id))
                + "</td><td>"
                + html.escape("-" if decision is None else ", ".join(decision.evidence_codes))
                + "</td><td>"
                + html.escape("-" if decision is None else decision.confidence)
                + "</td></tr>"
            )
        prompt = record.rendered_h3_prompt or "[unavailable]"
        failure = "" if record.failure is None else record.failure.reason
        cards.append(
            "<article><h2>"
            + html.escape(record.sample_id)
            + "</h2><p>status="
            + record.status
            + " · model calls="
            + str(record.model_call_count)
            + " · warnings="
            + html.escape(", ".join(record.validation_warnings) or "none")
            + "</p><video controls preload='metadata' src='"
            + html.escape(str(media["target"]), quote=True)
            + "'></video><div class='pictures'>"
            + pictures
            + "</div><table><thead><tr><th>fact</th><th>segment</th><th>time</th>"
            + "<th>speaker</th><th>ASR text</th><th>current entity</th>"
            + "<th>presentation</th><th>visible entity</th><th>evidence</th>"
            + "<th>confidence</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table><h3>NEW rendered H3 prompt</h3><pre>"
            + html.escape(prompt)
            + "</pre><p class='failure'>"
            + html.escape(failure)
            + "</p></article>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>Qwen speech presentation shadow</title>
<style>body{font:14px system-ui;margin:24px;background:#f5f5f3;color:#171717}article{background:white;border:1px solid #ccc;padding:18px;margin:0 0 22px}video{width:min(900px,100%);max-height:520px;background:#111}.pictures{display:flex;gap:12px;overflow:auto;margin:12px 0}.pictures figure{margin:0;min-width:180px}.pictures img{width:180px;height:150px;object-fit:contain;background:#eee}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;background:#f3f3f3;padding:12px}.failure{color:#a00}</style></head><body><h1>Qwen speech-presentation A/B shadow</h1>""" + "".join(cards) + "</body></html>"


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"Qwen presentation output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def _binding_counts(record: QwenPresentationRecord) -> Counter[str]:
    counts: Counter[str] = Counter()
    if record.status != "ready" or record.original_audio_facts is None:
        return counts
    for original, decision in zip(
        record.original_audio_facts.speech,
        record.speech_presentation_decisions,
        strict=True,
    ):
        current = original.entity_id
        predicted = decision.visible_entity_id
        counts["current"] += int(current is not None)
        counts["predicted"] += int(predicted is not None)
        counts["removed"] += int(current is not None and predicted is None)
        counts["added"] += int(current is None and predicted is not None)
        counts["changed"] += int(
            current is not None and predicted is not None and current != predicted
        )
        counts["non_onscreen"] += int(
            decision.speech_presentation != "onscreen_spoken"
        )
    return counts


def run_qwen_speech_presentation_recaption(
    *,
    audio_production_root: Path,
    case_manifest_path: Path,
    output_root: Path,
    backend: QwenPresentationBackend,
    overwrite: bool = False,
) -> QwenPresentationSummary:
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
        raise ValueError("Qwen presentation manifest must be non-empty and unique")
    if any(sample_id not in sample_by_id for sample_id in case_ids):
        raise ValueError("Qwen presentation manifest references an unknown H3 sample")
    for case in cases:
        sample = sample_by_id[case.sample_id]
        if case.conditioning_variant != "visual_only" or sample.pair_type != "canonical":
            raise ValueError(
                "Qwen presentation shadow supports canonical visual_only cases only"
            )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qwen presentation output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[QwenPresentationRecord] = []
    raw_payloads: list[tuple[str, dict[str, object]]] = []
    try:
        temporary.mkdir()
        (temporary / "raw_responses").mkdir()
        for index, case in enumerate(cases):
            sample = sample_by_id[case.sample_id]
            target = Path(sample.target_video).expanduser().resolve(strict=True)
            contract = build_reference_contract(sample, "visual_only")
            facts = _canonical_speech_facts(sample, contract)
            fingerprint = _request_fingerprint(
                sample=sample,
                case=case,
                contract=contract,
                facts=facts,
                provenance=backend.provenance,
            )
            request = QwenPresentationRequest(
                sample=sample,
                case=case,
                reference_contract=contract,
                original_audio_facts=facts,
                request_fingerprint=fingerprint,
            )
            try:
                result = backend.recaption(request)
                record = QwenPresentationRecord(
                    status="ready",
                    sample_id=case.sample_id,
                    clip_uid=sample.clip_uid,
                    note=case.note,
                    target_video_path=str(target),
                    target_video_sha256=_sha256_file(target),
                    backend_provenance=backend.provenance,
                    request_fingerprint=fingerprint,
                    reference_contract=contract,
                    original_audio_facts=facts,
                    speech_presentation_decisions=result.draft.speech_presentations,
                    corrected_audio_facts=result.corrected_audio_facts,
                    structured_h3_sections=result.response,
                    rendered_h3_prompt=render_h3_prompt(result.response),
                    model_call_count=result.model_call_count,
                    raw_response_count=len(result.raw_responses),
                    completion_diagnostics=list(result.diagnostics),
                    validation_warnings=list(result.validation_warnings),
                )
                raw_payload = {
                    "sample_id": case.sample_id,
                    "status": "ready",
                    "raw_responses": list(result.raw_responses),
                    "completion_diagnostics": [
                        item.model_dump(mode="json") for item in result.diagnostics
                    ],
                }
            except QwenPresentationBackendFailure as exc:
                record = QwenPresentationRecord(
                    status="failed",
                    sample_id=case.sample_id,
                    clip_uid=sample.clip_uid,
                    note=case.note,
                    target_video_path=str(target),
                    target_video_sha256=_sha256_file(target),
                    backend_provenance=backend.provenance,
                    request_fingerprint=fingerprint,
                    reference_contract=contract,
                    original_audio_facts=facts,
                    speech_presentation_decisions=[],
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
            raw_payloads.append(
                (f"{index:04d}-{_sha256_text(case.sample_id)[:12]}.json", raw_payload)
            )
        presentation_counts = Counter(
            decision.speech_presentation
            for record in records
            if record.status == "ready"
            for decision in record.speech_presentation_decisions
        )
        binding_counts: Counter[str] = Counter()
        for record in records:
            binding_counts.update(_binding_counts(record))
        failure_counts = Counter(
            record.failure.code for record in records if record.failure is not None
        )
        summary = QwenPresentationSummary(
            source_h3_samples_path=str(samples_path.resolve(strict=True)),
            source_h3_samples_sha256=_sha256_file(samples_path),
            case_manifest_path=str(manifest_path),
            case_manifest_sha256=_sha256_file(manifest_path),
            backend_provenance=backend.provenance,
            case_count=len(records),
            ready_count=sum(record.status == "ready" for record in records),
            failed_count=sum(record.status == "failed" for record in records),
            unsupported_count=sum(record.status == "unsupported" for record in records),
            first_pass_ready_count=sum(
                record.status == "ready" and record.model_call_count == 1
                for record in records
            ),
            repaired_ready_count=sum(
                record.status == "ready" and record.model_call_count == 2
                for record in records
            ),
            model_call_count=sum(record.model_call_count for record in records),
            served_model_name=backend.provenance.served_model_name,
            checkpoint_id=backend.provenance.checkpoint_id,
            presentation_counts={
                value: presentation_counts[value] for value in _PRESENTATION_VALUES
            },
            current_visible_entity_binding_count=binding_counts["current"],
            predicted_visible_entity_binding_count=binding_counts["predicted"],
            visible_entity_binding_removed_count=binding_counts["removed"],
            visible_entity_binding_added_count=binding_counts["added"],
            visible_entity_binding_changed_count=binding_counts["changed"],
            non_onscreen_speech_count=binding_counts["non_onscreen"],
            failure_counts=dict(sorted(failure_counts.items())),
        )
        _write_jsonl(temporary / "manifest.jsonl", cases)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        for name, payload in raw_payloads:
            _write_json(temporary / "raw_responses" / name, payload)
        media = _materialize_review_media(temporary, records)
        (temporary / "review.html").write_text(
            render_review_html(records, media), encoding="utf-8"
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "QWEN_PRESENTATION_BACKEND_VERSION",
    "QWEN_PRESENTATION_DRAFT_VERSION",
    "QWEN_PRESENTATION_MATERIALIZER_VERSION",
    "QWEN_PRESENTATION_POLICY_VERSION",
    "QWEN_PRESENTATION_PROMPT_VERSION",
    "QWEN_PRESENTATION_RECORD_VERSION",
    "QWEN_PRESENTATION_SUMMARY_VERSION",
    "SYSTEM_PROMPT",
    "OpenAIQwenPresentationBackend",
    "QwenPresentationAwareDraftResponse",
    "QwenPresentationBackendFailure",
    "QwenPresentationBackendProvenance",
    "QwenPresentationBackendResult",
    "QwenPresentationConfig",
    "QwenPresentationRecord",
    "QwenPresentationRequest",
    "QwenPresentationSummary",
    "QwenSpeechPresentationDecision",
    "build_corrected_audio_facts",
    "materialize_presentation_draft",
    "render_review_html",
    "run_qwen_speech_presentation_recaption",
    "validate_presentation_draft",
]
