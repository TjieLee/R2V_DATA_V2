from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlsplit

from openai import OpenAI
from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

MIMO25_MODEL = "mimo-v2.5"
MIMO25_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO25_PROMPT_VERSION = "h3_mimo25_unified_av_reconcile_v1"
MIMO25_POLICY_VERSION = "h3_mimo25_av_authority_contract_v1"
MIMO25_SCHEMA_VERSION = "r2v.h3.mimo25_av_annotation.1"
MIMO25_BACKEND_VERSION = "r2v.h3.mimo25_backend.1"
MIMO25_MATERIALIZER_VERSION = "h3_mimo25_materializer_v1"
DEFAULT_BASE64_LIMIT_BYTES = 50 * 1024 * 1024

_GROUP = re.compile(r"g([1-9]\d*)")
_PLACEHOLDER = re.compile(r"\[\[segment:([^\[\]\r\n]+)\]\]")
_PICTURE_OR_SUBJECT = re.compile(r"<(Picture|Subject) ([1-9]\d*)>")
_PIPELINE_OWNED = re.compile(r"<Audio [1-9]\d*>|\(S[1-9]\d*\)|<d>|</d>")

VocalComposition = Literal[
    "single_speaker",
    "same_speaker_nonlexical",
    "secondary_non_speech_vocalization",
    "overlapping_secondary_speech",
    "sequential_multi_speaker_speech",
    "uncertain",
]
Resolution = Literal["resolved", "needs_acoustic_refinement", "uncertain"]
BindingStatus = Literal[
    "visible_entity",
    "offscreen",
    "no_reliable_entity",
    "uncertain",
]
EvidenceCode = Literal[
    "visible_lip_motion",
    "av_temporal_alignment",
    "voice_continuity",
    "speaker_turn_change",
    "offscreen_audio",
    "lr_asd_support",
    "lr_asd_conflict",
    "source_cluster_support",
    "source_cluster_conflict",
    "insufficient_evidence",
]


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MimoSecondaryVocalActivity(SchemaModel):
    present: bool
    speaker_relation: Literal[
        "none", "same_speaker", "different_speaker", "uncertain"
    ]
    kind: Literal[
        "discourse_particle",
        "interjection",
        "laughter",
        "cough",
        "sigh",
        "breath",
        "gasp",
        "crying",
        "non_lyrical_singing",
        "speech",
        "other_nonlexical",
        "unknown",
    ] | None = None

    @model_validator(mode="after")
    def validate_activity(self) -> MimoSecondaryVocalActivity:
        if not self.present and (
            self.speaker_relation != "none" or self.kind is not None
        ):
            raise ValueError("absent secondary vocal activity cannot claim semantics")
        if self.present and (
            self.speaker_relation == "none" or self.kind is None
        ):
            raise ValueError("present secondary vocal activity requires semantics")
        return self


class MimoSegmentDecision(SchemaModel):
    segment_id: str = Field(min_length=1)
    vocal_composition: VocalComposition
    resolution: Resolution
    primary_speaker_group: str | None = Field(default=None, pattern=r"^g[1-9]\d*$")
    binding_status: BindingStatus
    entity_id: str | None = None
    secondary_vocal_activity: MimoSecondaryVocalActivity
    confidence: Literal["high", "medium", "low"]
    evidence_codes: list[EvidenceCode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> MimoSegmentDecision:
        if self.resolution == "resolved" and self.primary_speaker_group is None:
            raise ValueError("resolved segment requires one primary speaker group")
        if self.binding_status == "visible_entity":
            if self.entity_id is None:
                raise ValueError("visible_entity requires supplied entity_id")
        elif self.entity_id is not None:
            raise ValueError("only visible_entity may publish entity_id")
        if self.vocal_composition in {
            "overlapping_secondary_speech",
            "sequential_multi_speaker_speech",
        } and self.resolution != "needs_acoustic_refinement":
            raise ValueError("multi-speaker speech requires acoustic refinement")
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("segment evidence codes must be unique")
        return self


class MimoAudioEvent(SchemaModel):
    approximate_start_time: float = Field(ge=0, allow_inf_nan=False)
    approximate_end_time: float = Field(gt=0, allow_inf_nan=False)
    category: Literal[
        "physical",
        "environmental",
        "mechanical",
        "electronic",
        "human_non_speech",
        "diegetic_music",
        "other",
    ]
    pattern: Literal["single", "repeated", "continuous"]
    description: StrictStr
    source_grounding: Literal[
        "audible_only", "audiovisually_grounded", "uncertain_source"
    ]

    @model_validator(mode="after")
    def validate_event(self) -> MimoAudioEvent:
        if self.approximate_end_time <= self.approximate_start_time:
            raise ValueError("MiMo Audio event must have positive duration")
        if not self.description.strip():
            raise ValueError("MiMo Audio event description must not be empty")
        return self


class MimoSpeakerDelivery(SchemaModel):
    speaker_group: str = Field(pattern=r"^g[1-9]\d*$")
    delivery_style: StrictStr

    @model_validator(mode="after")
    def validate_delivery(self) -> MimoSpeakerDelivery:
        if not self.delivery_style.strip():
            raise ValueError("MiMo speaker delivery must not be empty")
        return self


class MimoAudioSemantics(SchemaModel):
    temporal_non_speech_events: list[MimoAudioEvent]
    speaker_delivery: list[MimoSpeakerDelivery]
    overall_soundscape: StrictStr | None = None
    non_diegetic_music_status: Literal["present", "absent", "unknown"]
    non_diegetic_music: StrictStr | None = None
    audiovisual_summary: StrictStr | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> MimoAudioSemantics:
        if self.non_diegetic_music_status == "present":
            if self.non_diegetic_music is None or not self.non_diegetic_music.strip():
                raise ValueError("present music requires a concise description")
        elif self.non_diegetic_music is not None:
            raise ValueError("absent or unknown music cannot publish a description")
        rows = [event.model_dump(mode="json") for event in self.temporal_non_speech_events]
        if len({_compact_json(row) for row in rows}) != len(rows):
            raise ValueError("exact duplicate MiMo Audio events are forbidden")
        groups = [item.speaker_group for item in self.speaker_delivery]
        if len(groups) != len(set(groups)):
            raise ValueError("MiMo speaker delivery groups must be unique")
        optional_strings = (self.overall_soundscape, self.audiovisual_summary)
        if any(value is not None and not value.strip() for value in optional_strings):
            raise ValueError("optional MiMo Audio text must be non-empty or null")
        return self


class MimoH3Shot(SchemaModel):
    shot_index: int = Field(gt=0)
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    description_template: StrictStr

    @model_validator(mode="after")
    def validate_shot(self) -> MimoH3Shot:
        if not self.description_template.strip():
            raise ValueError("MiMo H3 shot description must not be empty")
        return self


class MimoH3Draft(SchemaModel):
    subject_definitions: list[StrictStr] = Field(min_length=1)
    summary: StrictStr
    visual_retention_analysis: list[StrictStr] = Field(min_length=1)
    shots: list[MimoH3Shot] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_draft(self) -> MimoH3Draft:
        values = (
            *self.subject_definitions,
            self.summary,
            *self.visual_retention_analysis,
            *(shot.description_template for shot in self.shots),
        )
        if any(not value.strip() for value in values):
            raise ValueError("MiMo H3 draft text must not be empty")
        indexes = [shot.shot_index for shot in self.shots]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("MiMo H3 shots must use contiguous indexes")
        if self.shots[0].start_time is not None:
            raise ValueError("MiMo H3 Shot 1 cannot have a start time")
        later = [shot.start_time for shot in self.shots[1:]]
        if any(value is None for value in later):
            raise ValueError("later MiMo H3 shots require hard-cut times")
        numeric = [value for value in later if value is not None]
        if numeric != sorted(numeric) or len(numeric) != len(set(numeric)):
            raise ValueError("MiMo H3 hard-cut times must strictly increase")
        return self


class MimoAVAnnotationDraft(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_av_annotation.1"] = MIMO25_SCHEMA_VERSION
    segment_decisions: list[MimoSegmentDecision]
    audio_semantics: MimoAudioSemantics
    h3_draft: MimoH3Draft
    warnings: list[Literal["possible_asr_conflict"]] = Field(default_factory=list)


class MimoBackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_backend.1"] = MIMO25_BACKEND_VERSION
    backend: Literal["xiaomi_openai_compatible"] = "xiaomi_openai_compatible"
    model: Literal["mimo-v2.5"] = MIMO25_MODEL
    base_url: str
    video_fps: Literal[4.0] = 4.0
    media_resolution: Literal["default"] = "default"
    thinking: Literal["disabled"] = "disabled"
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_completion_tokens: int = Field(gt=0)
    response_format: Literal["json_object"] = "json_object"
    stream: Literal[False] = False
    media_mode: Literal["base64", "http"]
    media_root: str
    media_base_url: str | None = None
    prompt_version: Literal["h3_mimo25_unified_av_reconcile_v1"] = (
        MIMO25_PROMPT_VERSION
    )
    policy_version: Literal["h3_mimo25_av_authority_contract_v1"] = (
        MIMO25_POLICY_VERSION
    )
    annotation_schema_version: Literal["r2v.h3.mimo25_av_annotation.1"] = (
        MIMO25_SCHEMA_VERSION
    )
    materializer_version: Literal["h3_mimo25_materializer_v1"] = (
        MIMO25_MATERIALIZER_VERSION
    )
    http_max_attempts: int = Field(ge=1, le=5)
    structured_repair_retries: Literal[1] = 1
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> MimoBackendProvenance:
        if not self.base_url.strip() or not self.media_root.strip():
            raise ValueError("MiMo endpoint and media root are required")
        if (self.media_mode == "http") != (self.media_base_url is not None):
            raise ValueError("MiMo HTTP media mode requires one public base URL")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo backend configuration fingerprint is invalid")
        return self


class MimoUsage(SchemaModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    video_tokens: int | None = Field(default=None, ge=0)
    audio_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class MimoCompletionDiagnostic(SchemaModel):
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "structure_only_repair",
    ]
    finish_reason: str | None = None
    usage: MimoUsage
    http_attempt_count: int = Field(ge=1)
    warning: str | None = None


@dataclass(frozen=True)
class MimoBackendResult:
    annotation: MimoAVAnnotationDraft
    raw_responses: tuple[str, ...]
    diagnostics: tuple[MimoCompletionDiagnostic, ...]
    model_call_count: int
    http_retry_count: int
    repair_count: int
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
    ]


class MimoBackendFailure(ValueError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: tuple[str, ...] = (),
        diagnostics: tuple[MimoCompletionDiagnostic, ...] = (),
        issues: tuple[ValidationIssue, ...] = (),
        model_call_count: int = 0,
        http_retry_count: int = 0,
        repair_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = raw_responses
        self.diagnostics = diagnostics
        self.issues = issues
        self.model_call_count = model_call_count
        self.http_retry_count = http_retry_count
        self.repair_count = repair_count


class MimoBackendJob(Protocol):
    clip_uid: str
    target_video_path: str
    target_full_audio_path: str
    target_duration_seconds: float

    def model_dump(self, *, mode: str, exclude: set[str] | None = None) -> dict[str, Any]: ...


class MimoMediaResolver:
    def __init__(
        self,
        *,
        mode: Literal["base64", "http"],
        media_root: Path,
        media_base_url: str | None = None,
        maximum_base64_bytes: int = DEFAULT_BASE64_LIMIT_BYTES,
    ) -> None:
        self.mode = mode
        self.media_root = media_root.expanduser().resolve(strict=True)
        self.media_base_url = media_base_url
        self.maximum_base64_bytes = maximum_base64_bytes
        if not self.media_root.is_dir() or maximum_base64_bytes <= 0:
            raise ValueError("MiMo media root and Base64 limit must be valid")
        if mode == "http":
            if media_base_url is None:
                raise ValueError("MiMo HTTP media mode requires a public base URL")
            parsed = urlsplit(media_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("MiMo media base URL must be public HTTP(S)")
            self.media_base_url = media_base_url.rstrip("/") + "/"
        elif media_base_url is not None:
            raise ValueError("MiMo Base64 media mode cannot define an HTTP URL")

    def resolve(self, path: Path) -> str:
        source = path.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("MiMo media input must be a readable file")
        try:
            relative = source.relative_to(self.media_root)
        except ValueError as exc:
            raise ValueError("MiMo media is outside the configured media root") from exc
        if self.mode == "http":
            assert self.media_base_url is not None
            return self.media_base_url + quote(relative.as_posix(), safe="/")
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        encoded_size = 4 * ((source.stat().st_size + 2) // 3)
        if encoded_size > self.maximum_base64_bytes:
            raise MimoBackendFailure(
                code="media_too_large",
                reason=f"Base64 media exceeds {self.maximum_base64_bytes} bytes",
            )
        return f"data:{mime};base64," + base64.b64encode(source.read_bytes()).decode("ascii")


@dataclass(frozen=True)
class MimoBackendConfig:
    media_resolver: MimoMediaResolver
    api_key: str
    base_url: str = MIMO25_DEFAULT_BASE_URL
    model: str = MIMO25_MODEL
    video_fps: float = 4.0
    media_resolution: str = "default"
    thinking: str = "disabled"
    temperature: float = 0.2
    max_completion_tokens: int = 16384
    timeout_seconds: float = 900.0
    http_max_attempts: int = 3

    def __post_init__(self) -> None:
        if (
            not self.api_key.strip()
            or not self.base_url.strip()
            or self.model != MIMO25_MODEL
            or self.video_fps != 4.0
            or self.media_resolution != "default"
            or self.thinking != "disabled"
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.max_completion_tokens <= 0
            or self.timeout_seconds <= 0
            or not 1 <= self.http_max_attempts <= 5
        ):
            raise ValueError("MiMo backend configuration violates the v1 contract")

    def provenance(self) -> MimoBackendProvenance:
        values = {
            "schema_version": MIMO25_BACKEND_VERSION,
            "backend": "xiaomi_openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "video_fps": self.video_fps,
            "media_resolution": self.media_resolution,
            "thinking": self.thinking,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": "json_object",
            "stream": False,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "prompt_version": MIMO25_PROMPT_VERSION,
            "policy_version": MIMO25_POLICY_VERSION,
            "annotation_schema_version": MIMO25_SCHEMA_VERSION,
            "materializer_version": MIMO25_MATERIALIZER_VERSION,
            "http_max_attempts": self.http_max_attempts,
            "structured_repair_retries": 1,
        }
        return MimoBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


SYSTEM_PROMPT = """You are the unified audiovisual reconciliation model for an H3 shadow pipeline.

AUTHORITATIVE PIPELINE FACTS
1. Supplied DiariZen segment boundaries and sample ranges are authoritative acoustic boundaries for this pass. Never change, split, merge, delete, or invent them and never invent precise subsegment timing.
2. Supplied Qwen3-ASR text and language are authoritative transcript facts. Never transcribe, correct, rewrite, paraphrase, translate, repunctuate, complete, split, merge, or move their text. Your output schema contains no transcript field.
3. Frozen Visual references define the only valid entities and reference ownership. Never invent an entity, Picture, Subject, or attribute owner.
4. LR-ASD evidence, DiariZen source clusters, and current entity bindings are fallible proposals. Verify and correct only their speaker/entity structure using synchronized audiovisual evidence.
5. The full target video is audiovisual observation evidence, not an H3 reference.
6. Sound existence requires audible evidence. Video may identify or disambiguate an audible source but must not create an inaudible sound.
7. Speaker identity is not vocal-event count. Multiple vocal sounds inside one segment never make that segment invalid.
8. Never modify upstream files. Return exactly one compact JSON object matching the supplied schema.

PASS 1 - SEGMENT / SPEAKER / ENTITY RECONCILIATION
Return exactly one decision for every supplied segment ID, no more and no fewer. Preserve every segment. Temporary groups are clip-local g1, g2, ... ordered contiguously by chronological first appearance. Do not emit S1/S2. A source cluster may split and different source clusters may merge when AV evidence supports it. Offscreen speech may retain a prior speaker group but must use entity_id=null. Only visible_entity may name one supplied entity.

Do NOT interpret more than one audible vocal event as evidence that an upstream segment is invalid. A valid segment may include discourse particles or interjections from its primary speaker; sighing, breathing, laughter, coughing, hesitation, or other non-lexical vocalization from the same speaker; a brief non-speech vocalization from another person; overlapping secondary speech; or sequential speech by two speakers. Short particles such as 嗯, 啊, 唉, 哦, 诶, uh, um, hm, oh, and ah do not by themselves prove a speaker change. same_speaker_nonlexical preserves the group. secondary_non_speech_vocalization preserves primary dialogue ownership. overlapping_secondary_speech and sequential_multi_speaker_speech preserve the source segment but require needs_acoustic_refinement; never invent an internal change timestamp.

PASS 2 - AV AUDIO SEMANTICS
Describe only meaningfully audible non-speech events, delivery, soundscape, and music. Synchronized video may ground an audible door close, cup placement, or other source; visible action alone cannot establish sound. Use generic impact/clink when the source is uncertain. Coalesce repeated acoustically similar micro-events from one action into one event with pattern=repeated. Split only genuinely distinct sources, meanings, or times. Do not guess HVAC, room acoustics, ventilation, machinery, or material source from scene plausibility. Music requires audible musical structure; visual context may distinguish diegetic from non-diegetic music. Speaker delivery describes only audible pace, energy, loudness, articulation, hesitation, questioning, shouting, whispering, rhythm, and pauses, never transcript, gender, age, nationality, role, or identity.

PASS 3 - H3 VISUAL / TEMPORAL DRAFT
Write dense generation-quality visual observations, subject definitions, summary, retention analysis, and true hard-cut shots. Use target video plus frozen references. Do not infer plot, psychology, relationships, intent, causality, invisible events, or sounds. Each supplied transcribed segment must appear exactly once as [[segment:<segment_id>]] at its observed temporal position. Do not copy dialogue. Do not emit <Audio N>, (Sx), <d>, final reference numbering, donor provenance, or final H3 formatting. The pipeline owns those fields.

PASS 4 - CONSISTENCY CHECK BEFORE JSON
Check: every source segment exactly once; no unknown segment/entity/reference; no timestamp or transcript changes; contiguous gN first appearance; no segment dropped for multiple vocal events; repeated micro-events coalesced; no sound from visual evidence alone; every transcribed placeholder exactly once in chronological order; no <Audio N>, (Sx), or <d>; JSON only with no markdown or extra fields."""


def _value(value: object, name: str) -> object | None:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _token(value: object, *names: str) -> int | None:
    current: object | None = value
    for name in names:
        if current is None:
            return None
        current = _value(current, name)
    return current if isinstance(current, int) and current >= 0 else None


def _completion_diagnostic(
    completion: object,
    choice: object,
    *,
    modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "structure_only_repair",
    ],
    http_attempt_count: int,
) -> MimoCompletionDiagnostic:
    usage = _value(completion, "usage")
    details = _value(usage, "prompt_tokens_details")
    warning = None if details is not None else "prompt_tokens_details_unavailable"
    cached_tokens = _token(details, "cached_tokens")
    if cached_tokens is None:
        cached_tokens = _token(usage, "prompt_tokens_details", "cached_tokens")
    return MimoCompletionDiagnostic(
        input_modality=modality,
        finish_reason=_value(choice, "finish_reason"),
        usage=MimoUsage(
            prompt_tokens=_token(usage, "prompt_tokens"),
            completion_tokens=_token(usage, "completion_tokens"),
            total_tokens=_token(usage, "total_tokens"),
            video_tokens=_token(details, "video_tokens"),
            audio_tokens=_token(details, "audio_tokens"),
            cached_tokens=cached_tokens,
        ),
        http_attempt_count=http_attempt_count,
        warning=warning,
    )


def validate_annotation(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str],
    transcribed_segment_ids: list[str],
    allowed_entity_ids: set[str],
    allowed_reference_labels: set[str],
    target_duration_seconds: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    decisions = annotation.segment_decisions
    actual_ids = [item.segment_id for item in decisions]
    if actual_ids != segment_ids:
        issues.append(
            ValidationIssue(
                "segment_inventory_mismatch",
                "segment_decisions",
                "segment decisions must exactly follow supplied chronological IDs",
            )
        )
    for decision in decisions:
        if decision.entity_id is not None and decision.entity_id not in allowed_entity_ids:
            issues.append(
                ValidationIssue(
                    "unknown_entity",
                    decision.segment_id,
                    f"unknown supplied entity: {decision.entity_id}",
                )
            )
    first_groups: list[str] = []
    for decision in decisions:
        group = decision.primary_speaker_group
        if group is not None and group not in first_groups:
            first_groups.append(group)
    expected_groups = [f"g{index}" for index in range(1, len(first_groups) + 1)]
    if first_groups != expected_groups:
        issues.append(
            ValidationIssue(
                "non_contiguous_speaker_groups",
                "segment_decisions",
                "speaker groups must be contiguous by first chronological appearance",
            )
        )
    known_groups = {
        item.primary_speaker_group
        for item in decisions
        if item.primary_speaker_group is not None
    }
    for delivery in annotation.audio_semantics.speaker_delivery:
        if delivery.speaker_group not in known_groups:
            issues.append(
                ValidationIssue(
                    "unknown_delivery_speaker_group",
                    "audio_semantics.speaker_delivery",
                    delivery.speaker_group,
                )
            )
    for event in annotation.audio_semantics.temporal_non_speech_events:
        if event.approximate_end_time > target_duration_seconds + 1e-6:
            issues.append(
                ValidationIssue(
                    "audio_event_exceeds_target",
                    "audio_semantics.temporal_non_speech_events",
                    str(event.approximate_end_time),
                )
            )
    draft_text = "\n".join(
        (
            *annotation.h3_draft.subject_definitions,
            annotation.h3_draft.summary,
            *annotation.h3_draft.visual_retention_analysis,
            *(shot.description_template for shot in annotation.h3_draft.shots),
        )
    )
    placeholders = _PLACEHOLDER.findall(draft_text)
    if placeholders != transcribed_segment_ids:
        issues.append(
            ValidationIssue(
                "speech_placeholder_inventory_mismatch",
                "h3_draft.shots",
                "transcribed segment placeholders must appear exactly once in order",
            )
        )
    if _PIPELINE_OWNED.search(draft_text):
        issues.append(
            ValidationIssue(
                "draft_contains_pipeline_owned_syntax",
                "h3_draft",
                "MiMo draft contains Audio, Sx, or dialogue syntax",
            )
        )
    unknown_labels = {
        match.group(0)
        for match in _PICTURE_OR_SUBJECT.finditer(draft_text)
        if match.group(0) not in allowed_reference_labels
    }
    if unknown_labels:
        issues.append(
            ValidationIssue(
                "draft_contains_unknown_reference",
                "h3_draft",
                str(sorted(unknown_labels)),
            )
        )
    return issues


class OpenAIMimo25Backend:
    def __init__(
        self,
        config: MimoBackendConfig,
        *,
        client: Any | None = None,
        sleep: Any = time.sleep,
        jitter: Any = random.random,
    ) -> None:
        self.config = config
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        self._sleep = sleep
        self._jitter = jitter

    @property
    def provenance(self) -> MimoBackendProvenance:
        return self.config.provenance()

    def _call(self, payload: dict[str, object]) -> tuple[object, int, int]:
        retries = 0
        for attempt in range(1, self.config.http_max_attempts + 1):
            try:
                return self.client.chat.completions.create(**payload), attempt, retries
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = status == 429 or (
                    isinstance(status, int) and 500 <= status <= 599
                ) or isinstance(exc, (TimeoutError, ConnectionError, OSError)) or type(
                    exc
                ).__name__ in {"APITimeoutError", "APIConnectionError"}
                if not retryable or attempt == self.config.http_max_attempts:
                    raise
                retries += 1
                self._sleep((2 ** (attempt - 1)) + self._jitter())
        raise AssertionError("bounded MiMo retry loop did not terminate")

    def _media_content(
        self,
        job: MimoBackendJob,
        *,
        include_audio_fallback: bool,
    ) -> list[dict[str, object]]:
        job_payload = job.model_dump(mode="json")
        references = job_payload.get("reference_images")
        if not isinstance(references, list):
            raise TypeError("MiMo job reference inventory is invalid")
        content: list[dict[str, object]] = []
        for reference in references:
            if not isinstance(reference, dict):
                raise TypeError("MiMo reference metadata is invalid")
            path = Path(str(reference["image_artifact_path"]))
            metadata = {
                key: value
                for key, value in reference.items()
                if key not in {"image_artifact_path", "image_sha256"}
            }
            content.extend(
                [
                    {"type": "text", "text": _compact_json(metadata)},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.config.media_resolver.resolve(path)},
                    },
                ]
            )
        content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": self.config.media_resolver.resolve(
                        Path(job.target_video_path)
                    ),
                    "fps": self.config.video_fps,
                    "media_resolution": self.config.media_resolution,
                },
            }
        )
        if include_audio_fallback:
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": self.config.media_resolver.resolve(
                            Path(job.target_full_audio_path)
                        )
                    },
                }
            )
        return content

    def _prompt(self, job: MimoBackendJob) -> str:
        payload = job.model_dump(mode="json")
        for reference in payload.get("reference_images", []):
            reference.pop("image_artifact_path", None)
            reference.pop("image_sha256", None)
        payload.pop("target_video_path", None)
        payload.pop("target_full_audio_path", None)
        payload.pop("target_video_sha256", None)
        payload.pop("target_full_audio_sha256", None)
        return (
            "Return exactly one JSON object matching this schema, with no markdown or extra fields.\n"
            + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            + "\nCOMPACT AUTHORITATIVE INPUT:\n"
            + _compact_json(payload)
        )

    def _request(
        self,
        job: MimoBackendJob,
        *,
        include_audio_fallback: bool,
    ) -> tuple[str, MimoCompletionDiagnostic, int]:
        modality = (
            "target_video_plus_canonical_full_audio_fallback"
            if include_audio_fallback
            else "target_video_with_embedded_audio"
        )
        content = self._media_content(job, include_audio_fallback=include_audio_fallback)
        content.append({"type": "text", "text": self._prompt(job)})
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "stream": False,
            "extra_body": {"thinking": self.config.thinking},
        }
        completion, attempts, retries = self._call(payload)
        choices = _value(completion, "choices")
        if not isinstance(choices, list) or not choices:
            raise TypeError("MiMo response has no choices")
        choice = choices[0]
        message = _value(choice, "message")
        raw = _value(message, "content")
        if not isinstance(raw, str):
            raise TypeError("MiMo response content must be text")
        return raw, _completion_diagnostic(
            completion,
            choice,
            modality=modality,
            http_attempt_count=attempts,
        ), retries

    def reconcile(
        self,
        job: MimoBackendJob,
        *,
        segment_ids: list[str],
        transcribed_segment_ids: list[str],
        allowed_entity_ids: set[str],
        allowed_reference_labels: set[str],
    ) -> MimoBackendResult:
        raw_responses: list[str] = []
        diagnostics: list[MimoCompletionDiagnostic] = []
        http_retries = 0
        issues: list[ValidationIssue] = []
        modality: Literal[
            "target_video_with_embedded_audio",
            "target_video_plus_canonical_full_audio_fallback",
        ] = "target_video_with_embedded_audio"
        try:
            raw, diagnostic, retries = self._request(
                job, include_audio_fallback=False
            )
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            http_retries += retries
            if diagnostic.usage.audio_tokens == 0:
                raw, diagnostic, retries = self._request(
                    job, include_audio_fallback=True
                )
                raw_responses.append(raw)
                diagnostics.append(diagnostic)
                http_retries += retries
                modality = "target_video_plus_canonical_full_audio_fallback"
        except MimoBackendFailure:
            raise
        except Exception as exc:
            raise MimoBackendFailure(
                code="mimo_request_failed",
                reason=f"{type(exc).__name__}: {exc}",
                raw_responses=tuple(raw_responses),
                diagnostics=tuple(diagnostics),
                model_call_count=len(diagnostics) + 1,
                http_retry_count=http_retries,
            ) from exc
        annotation, issues = parse_structured_json_issues(
            raw_responses[-1], MimoAVAnnotationDraft
        )
        if annotation is not None:
            issues = validate_annotation(
                annotation,
                segment_ids=segment_ids,
                transcribed_segment_ids=transcribed_segment_ids,
                allowed_entity_ids=allowed_entity_ids,
                allowed_reference_labels=allowed_reference_labels,
                target_duration_seconds=job.target_duration_seconds,
            )
        if annotation is None or issues:
            repair_prompt = (
                "Repair only JSON structure and the listed contract errors. Do not add facts, "
                "change transcript evidence, timestamps, segment IDs, entities, or speaker meaning. "
                "Return one JSON object only.\nALLOWED SEGMENTS: "
                + _compact_json(segment_ids)
                + "\nALLOWED ENTITIES: "
                + _compact_json(sorted(allowed_entity_ids))
                + "\nSCHEMA: "
                + _compact_json(MimoAVAnnotationDraft.model_json_schema())
                + "\nISSUES: "
                + _compact_json([item.to_dict() for item in issues])
                + "\nINVALID JSON: "
                + raw_responses[-1]
            )
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": repair_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_completion_tokens,
                "stream": False,
                "extra_body": {"thinking": self.config.thinking},
            }
            try:
                completion, attempts, retries = self._call(payload)
                http_retries += retries
                choices = _value(completion, "choices")
                if not isinstance(choices, list) or not choices:
                    raise TypeError("MiMo repair response has no choices")
                choice = choices[0]
                raw = _value(_value(choice, "message"), "content")
                if not isinstance(raw, str):
                    raise TypeError("MiMo repair response must be text")
                raw_responses.append(raw)
                diagnostics.append(
                    _completion_diagnostic(
                        completion,
                        choice,
                        modality="structure_only_repair",
                        http_attempt_count=attempts,
                    )
                )
            except Exception as exc:
                raise MimoBackendFailure(
                    code="mimo_repair_request_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    raw_responses=tuple(raw_responses),
                    diagnostics=tuple(diagnostics),
                    issues=tuple(issues),
                    model_call_count=len(diagnostics) + 1,
                    http_retry_count=http_retries,
                    repair_count=1,
                ) from exc
            annotation, issues = parse_structured_json_issues(
                raw_responses[-1], MimoAVAnnotationDraft
            )
            if annotation is not None:
                issues = validate_annotation(
                    annotation,
                    segment_ids=segment_ids,
                    transcribed_segment_ids=transcribed_segment_ids,
                    allowed_entity_ids=allowed_entity_ids,
                    allowed_reference_labels=allowed_reference_labels,
                    target_duration_seconds=job.target_duration_seconds,
                )
        if annotation is None or issues:
            raise MimoBackendFailure(
                code="mimo_structured_output_failed",
                reason="MiMo AV annotation failed after one structural repair",
                raw_responses=tuple(raw_responses),
                diagnostics=tuple(diagnostics),
                issues=tuple(issues),
                model_call_count=len(diagnostics),
                http_retry_count=http_retries,
                repair_count=1,
            )
        return MimoBackendResult(
            annotation=annotation,
            raw_responses=tuple(raw_responses),
            diagnostics=tuple(diagnostics),
            model_call_count=len(diagnostics),
            http_retry_count=http_retries,
            repair_count=int(any(item.input_modality == "structure_only_repair" for item in diagnostics)),
            input_modality=modality,
        )


__all__ = [
    "DEFAULT_BASE64_LIMIT_BYTES",
    "MIMO25_MATERIALIZER_VERSION",
    "MIMO25_MODEL",
    "MIMO25_POLICY_VERSION",
    "MIMO25_PROMPT_VERSION",
    "MIMO25_SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "MimoAVAnnotationDraft",
    "MimoAudioEvent",
    "MimoAudioSemantics",
    "MimoBackendConfig",
    "MimoBackendFailure",
    "MimoBackendProvenance",
    "MimoBackendResult",
    "MimoCompletionDiagnostic",
    "MimoH3Draft",
    "MimoH3Shot",
    "MimoMediaResolver",
    "MimoSegmentDecision",
    "OpenAIMimo25Backend",
    "sha256_file",
    "validate_annotation",
]
