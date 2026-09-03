from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import random
import re
import time
import unicodedata
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
MIMO25_PROMPT_VERSION = "h3_mimo25_unified_av_reconcile_v4"
MIMO25_POLICY_VERSION = "h3_mimo25_av_authority_contract_v4"
MIMO25_SCHEMA_VERSION = "r2v.h3.mimo25_av_annotation.4"
MIMO25_BACKEND_VERSION = "r2v.h3.mimo25_backend.4"
MIMO25_MATERIALIZER_VERSION = "h3_mimo25_materializer_v4"
DEFAULT_BASE64_LIMIT_BYTES = 50 * 1024 * 1024

_GROUP = re.compile(r"g([1-9]\d*)")
_PLACEHOLDER = re.compile(r"\[\[segment:([^\[\]\r\n]+)\]\]")
_AUDIO_EVENT_PLACEHOLDER = re.compile(r"\[\[audio_event:([^\[\]\r\n]+)\]\]")
_PICTURE_OR_SUBJECT = re.compile(r"<(Picture|Subject) (\d+)>")
_PIPELINE_OWNED = re.compile(
    r"<Audio \d+>|<Video \d+>|\(S\d+\)|<d>|</d>|\[Shot \d+\]"
)
_SPEECH_LEAD_IN = re.compile(
    r"\b(?:says?|speaks?|asks?|replies?|shouts?|whispers?)\s*,?\s*$",
    flags=re.IGNORECASE,
)
_KEYFRAME_ROLE = re.compile(
    r"<Picture\s+[1-9]\d*>[^.\n]{0,100}\b(first frame|last frame|keyframe)\b"
    r"|\b(first frame|last frame|keyframe)\b[^.\n]{0,100}<Picture\s+[1-9]\d*>",
    flags=re.IGNORECASE,
)
_RETENTION_MARKER = re.compile(
    r"^\s*<(?:Picture|Subject) [1-9]\d*>[^\n]*?:\s*([a-z]+_[a-z_]+)\b",
    flags=re.MULTILINE,
)
_ALLOWED_RETENTION_MARKERS = {
    "fully_preserved",
    "partially_preserved",
    "weak_reference",
}
_ALLOWED_RETENTION_MARKER_OCCURRENCE = re.compile(
    r"\b(?:fully_preserved|partially_preserved|weak_reference)\b"
)
_TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_completion_tokens",
    "token_limit",
}

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
SpeechPresentation = Literal[
    "onscreen_spoken",
    "offscreen_spoken",
    "voice_over",
    "message_voice_over",
    "device_playback",
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
    "no_visible_lip_motion",
    "message_text_alignment",
    "voice_over_context",
    "device_playback_context",
]


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _significant_transcript(value: str) -> bool:
    normalized = _normalized_text(value)
    cjk_count = sum("\u3400" <= character <= "\u9fff" for character in normalized)
    alphanumeric_count = sum(character.isalnum() for character in normalized)
    return cjk_count >= 4 or alphanumeric_count >= 12


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
    speech_presentation: SpeechPresentation
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
            if self.speech_presentation != "onscreen_spoken":
                raise ValueError("visible_entity requires onscreen_spoken presentation")
            if "visible_lip_motion" not in self.evidence_codes:
                raise ValueError("visible_entity requires visible_lip_motion evidence")
        elif self.entity_id is not None:
            raise ValueError("only visible_entity may publish entity_id")
        if self.speech_presentation == "onscreen_spoken":
            if "visible_lip_motion" not in self.evidence_codes:
                raise ValueError("onscreen_spoken requires visible_lip_motion evidence")
            if self.binding_status == "offscreen":
                raise ValueError("onscreen_spoken cannot use offscreen binding status")
        elif self.binding_status == "visible_entity" or self.entity_id is not None:
            raise ValueError("non-onscreen speech cannot claim a visible entity")
        secondary = self.secondary_vocal_activity
        if self.vocal_composition == "single_speaker":
            if secondary.present or secondary.speaker_relation != "none" or secondary.kind is not None:
                raise ValueError("single_speaker forbids secondary vocal activity")
        elif self.vocal_composition == "same_speaker_nonlexical":
            if (
                not secondary.present
                or secondary.speaker_relation != "same_speaker"
                or secondary.kind in {None, "speech"}
            ):
                raise ValueError("same-speaker nonlexical composition is inconsistent")
        elif self.vocal_composition == "secondary_non_speech_vocalization":
            if (
                not secondary.present
                or secondary.speaker_relation not in {"different_speaker", "uncertain"}
                or secondary.kind in {None, "speech"}
            ):
                raise ValueError("secondary non-speech vocalization is inconsistent")
        elif (
            self.vocal_composition
            in {"overlapping_secondary_speech", "sequential_multi_speaker_speech"}
            and (
                not secondary.present
                or secondary.speaker_relation != "different_speaker"
                or secondary.kind != "speech"
                or self.resolution != "needs_acoustic_refinement"
            )
        ):
            raise ValueError("multi-speaker speech requires acoustic refinement")
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("segment evidence codes must be unique")
        return self


class MimoAudioEvent(SchemaModel):
    event_id: str = Field(pattern=r"^ae[1-9]\d*$")
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
        if (
            re.search(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b", self.description)
            or _PIPELINE_OWNED.search(self.description)
            or _PICTURE_OR_SUBJECT.search(self.description)
            or _PLACEHOLDER.search(self.description)
            or _AUDIO_EVENT_PLACEHOLDER.search(self.description)
            or "[[" in self.description
            or "]]" in self.description
        ):
            raise ValueError("MiMo Audio event description contains forbidden syntax")
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
    audiovisual_summary: StrictStr

    @model_validator(mode="after")
    def validate_semantics(self) -> MimoAudioSemantics:
        if self.non_diegetic_music_status == "present":
            if self.non_diegetic_music is None or not self.non_diegetic_music.strip():
                raise ValueError("present music requires a concise description")
        elif self.non_diegetic_music is not None:
            raise ValueError("absent or unknown music cannot publish a description")
        events = self.temporal_non_speech_events
        rows = [event.model_dump(mode="json") for event in events]
        if len({_compact_json(row) for row in rows}) != len(rows):
            raise ValueError("exact duplicate MiMo Audio events are forbidden")
        event_ids = [event.event_id for event in events]
        if event_ids != [f"ae{index}" for index in range(1, len(events) + 1)]:
            raise ValueError("MiMo Audio event IDs must be contiguous")
        event_start_times = [event.approximate_start_time for event in events]
        if event_start_times != sorted(event_start_times):
            raise ValueError("MiMo Audio events must be chronological")
        groups = [item.speaker_group for item in self.speaker_delivery]
        if len(groups) != len(set(groups)):
            raise ValueError("MiMo speaker delivery groups must be unique")
        if not self.audiovisual_summary.strip():
            raise ValueError("MiMo audiovisual summary must not be empty")
        if self.overall_soundscape is not None and not self.overall_soundscape.strip():
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


class MimoAnnotationWarning(SchemaModel):
    code: Literal["possible_asr_conflict"]
    segment_id: str | None = None

    @model_validator(mode="after")
    def validate_warning(self) -> MimoAnnotationWarning:
        if self.code == "possible_asr_conflict" and self.segment_id is None:
            raise ValueError("possible ASR conflict warning requires segment_id")
        return self


class MimoAVAnnotationDraft(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_av_annotation.4"] = MIMO25_SCHEMA_VERSION
    segment_decisions: list[MimoSegmentDecision]
    audio_semantics: MimoAudioSemantics
    h3_draft: MimoH3Draft
    warnings: list[MimoAnnotationWarning] = Field(default_factory=list)


class MimoThinkingContract(SchemaModel):
    type: Literal["disabled"] = "disabled"


class MimoBackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_backend.4"] = MIMO25_BACKEND_VERSION
    backend: Literal["xiaomi_openai_compatible"] = "xiaomi_openai_compatible"
    model: Literal["mimo-v2.5"] = MIMO25_MODEL
    base_url: str
    video_fps: Literal[4.0] = 4.0
    media_resolution: Literal["default"] = "default"
    thinking: MimoThinkingContract
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_completion_tokens: int = Field(gt=0)
    response_format: Literal["json_object"] = "json_object"
    stream: Literal[False] = False
    media_mode: Literal["base64", "http"]
    media_root: str
    media_base_url: str | None = None
    prompt_version: Literal["h3_mimo25_unified_av_reconcile_v4"] = (
        MIMO25_PROMPT_VERSION
    )
    policy_version: Literal["h3_mimo25_av_authority_contract_v4"] = (
        MIMO25_POLICY_VERSION
    )
    annotation_schema_version: Literal["r2v.h3.mimo25_av_annotation.4"] = (
        MIMO25_SCHEMA_VERSION
    )
    materializer_version: Literal["h3_mimo25_materializer_v4"] = (
        MIMO25_MATERIALIZER_VERSION
    )
    http_max_attempts: int = Field(ge=1, le=5)
    full_av_recheck_limit: Literal[1] = 1
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
    image_tokens: int | None = Field(default=None, ge=0)
    video_tokens: int | None = Field(default=None, ge=0)
    audio_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class MimoCompletionDiagnostic(SchemaModel):
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "full_av_recheck_embedded_audio",
        "full_av_recheck_with_canonical_audio",
    ]
    finish_reason: str | None = None
    usage: MimoUsage
    http_attempt_count: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)
    request_error: str | None = None


@dataclass(frozen=True)
class MimoBackendResult:
    annotation: MimoAVAnnotationDraft
    raw_responses: tuple[str, ...]
    diagnostics: tuple[MimoCompletionDiagnostic, ...]
    model_call_count: int
    http_attempt_count: int
    http_retry_count: int
    recheck_count: int
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
        http_attempt_count: int = 0,
        http_retry_count: int = 0,
        recheck_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = raw_responses
        self.diagnostics = diagnostics
        self.issues = issues
        self.model_call_count = model_call_count
        self.http_attempt_count = http_attempt_count
        self.http_retry_count = http_retry_count
        self.recheck_count = recheck_count


class MimoBackendJob(Protocol):
    clip_uid: str
    r2v_instruction: str
    target_video_path: str
    target_full_audio_path: str
    target_duration_seconds: float
    reference_subjects: list[Any]
    reference_images: list[Any]
    segments: list[Any]

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
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.max_completion_tokens <= 0
            or self.timeout_seconds <= 0
            or not 1 <= self.http_max_attempts <= 5
        ):
            raise ValueError("MiMo backend configuration violates the v3 contract")

    def provenance(self) -> MimoBackendProvenance:
        values = {
            "schema_version": MIMO25_BACKEND_VERSION,
            "backend": "xiaomi_openai_compatible",
            "model": self.model,
            "base_url": self.base_url,
            "video_fps": self.video_fps,
            "media_resolution": self.media_resolution,
            "thinking": {"type": "disabled"},
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
            "full_av_recheck_limit": 1,
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
Return exactly one decision for every supplied segment ID, no more and no fewer. Preserve every segment. Temporary groups are clip-local g1, g2, ... ordered contiguously by chronological first appearance. Do not emit S1/S2. A source cluster may split and different source clusters may merge when AV evidence supports it.

Speaker identity, visible-entity binding, and speech audiovisual presentation are three different questions. A visible person being present is NOT evidence that they are speaking. A voice matching a visible character is NOT evidence that the character's mouth produces the current speech. visible_entity may be emitted only when synchronized visual evidence shows speech-like mouth, lip, or jaw motion during that segment; then speech_presentation must be onscreen_spoken, entity_id must name one supplied entity, and evidence_codes must include visible_lip_motion. LR-ASD activity, source-cluster support, voice continuity, body or head motion, and visibility alone are insufficient. onscreen_spoken with no reliable supplied identity is legal with entity_id=null and still requires visible_lip_motion.

If a visible person silently reads a phone, looks at a message, types, reacts, or listens while a voice is heard, do not classify the voice as onscreen_spoken merely because it sounds like that character. If speech audibly reads or represents displayed or typed messaging while the visible person does not speak, use message_voice_over with entity_id=null. Use voice_over for narration, inner voice, editorial voice-over, or other speech not produced by a visible speaking mouth. Use offscreen_spoken for speech from outside the frame. Use device_playback for speech audibly emitted by an in-scene phone, television, radio, video, or recording. When the audiovisual evidence is insufficient, use uncertain and remove visible-entity binding. Never delete an authoritative segment because its presentation is non-onscreen: Qwen3-ASR transcript/language and DiariZen exact timing remain authoritative.

Conceptual examples: a visible person silently reads or types on a phone while a voice reads the message means speech_presentation=message_voice_over, binding_status must not be visible_entity, and entity_id=null. A visible person who visibly articulates speech in sync means speech_presentation=onscreen_spoken; binding_status may be visible_entity only when identity is independently supported.

Do NOT interpret more than one audible vocal event as evidence that an upstream segment is invalid. A valid segment may include discourse particles or interjections from its primary speaker; sighing, breathing, laughter, coughing, hesitation, or other non-lexical vocalization from the same speaker; a brief non-speech vocalization from another person; overlapping secondary speech; or sequential speech by two speakers. Short particles such as 嗯, 啊, 唉, 哦, 诶, uh, um, hm, oh, and ah do not by themselves prove a speaker change. same_speaker_nonlexical preserves the group. secondary_non_speech_vocalization preserves primary dialogue ownership. overlapping_secondary_speech and sequential_multi_speaker_speech preserve the source segment but require needs_acoustic_refinement; never invent an internal change timestamp.

PASS 2 - AV AUDIO SEMANTICS
Describe only meaningfully audible non-speech events, delivery, soundscape, and music. Synchronized video may ground an audible door close, cup placement, or other source; visible action alone cannot establish sound. Use generic impact/clink when the source is uncertain. Coalesce repeated acoustically similar micro-events from one action into one event with pattern=repeated. Split only genuinely distinct sources, meanings, or times. Assign every temporal_non_speech_event a stable contiguous event_id ae1, ae2, ... in chronological order. Each event description must be one concise standalone English audible clause with no timestamp text, transcript, reference label, speaker syntax, dialogue markup, or placeholder. Do not guess HVAC, room acoustics, ventilation, machinery, or material source from scene plausibility. Music requires audible musical structure; visual context may distinguish diegetic from non-diegetic music. Speaker delivery describes only audible pace, energy, loudness, articulation, hesitation, questioning, shouting, whispering, rhythm, and pauses, never transcript, gender, age, nationality, role, or identity. For every resolved primary speaker group owning an authoritative transcribed segment, speaker_delivery must include that group exactly once; never emit an unknown group.

audiovisual_summary is a concise summary of directly observed audiovisual context. It must not quote or paraphrase dialogue and must not infer plot, relationship, intention, causality, or psychology.

PASS 3 - H3 VISUAL / TEMPORAL DRAFT
Write a generation-quality dense video description, not an ordinary caption or plot summary. For each real shot, cover every applicable observed dimension: visual style; shot scale and framing; camera angle; foreground, midground, and background composition; every salient visible subject's appearance; spatial positions and relationships; pose; body, arm, hand, and head motion; gaze; facial expression and visible expression changes; interactions; object state and state changes; environment; visible materials; readable scene text; lighting; color; camera motion or an explicitly stable/static camera; temporal progression through early, middle, and late portions; and supplied speech placeholders at their correct observed temporal positions.

Use observed evidence, never plausible filler. Do not infer psychology or unsupported emotion, intent, causality, relationships, identity not supplied upstream, sounds from visible actions, invisible or offscreen events, or invented object details. For any segment whose speech_presentation is not onscreen_spoken, do not describe a visible subject as speaking, saying, replying verbally, whispering, shouting, uttering, moving lips to the dialogue, opening the mouth to say it, or lip-syncing. For message_voice_over, describe only observed messaging behavior such as looking at a phone, typing, reading, displayed text, or a visible reaction; never convert the voice-over into mouth-speaking action. The exact dialogue remains pipeline-owned placeholder material. For information-rich clips, target roughly 300-450 English words across the materialized detailed description when visual evidence supports that amount. Do not pad a simple clip to hit a word count. Current clips are normally single-shot; add a later shot only for a real hard cut. Never write [Shot N]; the pipeline owns shot headers.

SUPPLIED <Picture N> AND <Subject N> LABELS are immutable pipeline-provided labels. Reuse them exactly where required; never renumber or invent them. Use supplied Picture and Subject labels in subject definitions and visual retention analysis. Do not emit <Video N>, <Audio N>, unknown Picture/Subject labels, your own reference numbering, (Sx), <d>, donor provenance, or final H3 formatting. The target video is observation only and is never <Video N>. Pictures are content references, not first frames, last frames, or keyframes.

The only allowed visual retention markers are fully_preserved, partially_preserved, and weak_reference. attribute_transfer is forbidden by the current conditioning contract. Do not invent another marker.

Each supplied transcribed segment must appear exactly once as [[segment:<segment_id>]] in shot description_template fields, in authoritative chronological order and at its observed temporal position. A placeholder represents the complete speech clause. Never prefix it with says, speaks, asks, replies, shouts, whispers, "the woman says", "the man replies", a character name, a pronoun, or a role. Do not place a placeholder in subject_definitions, summary, or visual_retention_analysis. Do not copy dialogue.

Insert every emitted Audio event exactly once as [[audio_event:<event_id>]] at its approximate observed position in the appropriate shot. Audio-event placeholders must follow event chronological order, may overlap speech placeholders, and are allowed only in shot description_template fields. Each placeholder represents the complete grounded Audio-event clause. Do not restate or paraphrase that event elsewhere in the shot.

Every supplied <Subject N> must have exactly one visual_retention_analysis line beginning with that exact Subject label and containing exactly one allowed retention marker. Do not use Picture labels as retention owners and do not emit unknown or duplicate Subject retention lines.

PASS 4 - CONSISTENCY CHECK BEFORE JSON
Check: every source segment exactly once; no unknown segment/entity/reference; no timestamp or transcript changes; contiguous gN first appearance; visible_entity only with onscreen_spoken plus visible_lip_motion; every non-onscreen or uncertain presentation with entity_id=null; no segment dropped for multiple vocal events; repeated micro-events coalesced; no sound from visual evidence alone; every transcribed placeholder exactly once in chronological order and only in shots; every Audio-event placeholder exactly once in event order and only in shots; all supplied Subject definitions retain their source Pictures; every supplied Subject has exactly one retention line; no pipeline-owned syntax; JSON only with no markdown or extra fields."""


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
        "full_av_recheck_embedded_audio",
        "full_av_recheck_with_canonical_audio",
    ],
    http_attempt_count: int,
) -> MimoCompletionDiagnostic:
    usage = _value(completion, "usage")
    prompt_details = _value(usage, "prompt_tokens_details")
    completion_details = _value(usage, "completion_tokens_details")
    warnings = []
    if prompt_details is None:
        warnings.append("prompt_tokens_details_unavailable")
    if completion_details is None:
        warnings.append("completion_tokens_details_unavailable")
    cached_tokens = _token(prompt_details, "cached_tokens")
    if cached_tokens is None:
        cached_tokens = _token(usage, "prompt_tokens_details", "cached_tokens")
    reasoning_tokens = _token(completion_details, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _token(
            usage, "completion_tokens_details", "reasoning_tokens"
        )
    if reasoning_tokens is not None and reasoning_tokens > 0:
        warnings.append("reasoning_tokens_nonzero_under_disabled_thinking")
    return MimoCompletionDiagnostic(
        input_modality=modality,
        finish_reason=_value(choice, "finish_reason"),
        usage=MimoUsage(
            prompt_tokens=_token(usage, "prompt_tokens"),
            completion_tokens=_token(usage, "completion_tokens"),
            total_tokens=_token(usage, "total_tokens"),
            image_tokens=_token(prompt_details, "image_tokens"),
            video_tokens=_token(prompt_details, "video_tokens"),
            audio_tokens=_token(prompt_details, "audio_tokens"),
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        http_attempt_count=http_attempt_count,
        warnings=warnings,
    )


def _validate_finish_reason(diagnostic: MimoCompletionDiagnostic) -> None:
    value = diagnostic.finish_reason
    if value is None:
        diagnostic.warnings.append("finish_reason_unavailable")
        return
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "stop":
        return
    if normalized in _TRUNCATED_FINISH_REASONS or "token_limit" in normalized:
        raise MimoBackendFailure(
            code="mimo_output_truncated",
            reason="MiMo AV response ended at the token limit",
        )
    raise MimoBackendFailure(
        code="mimo_incomplete_finish_reason",
        reason=f"MiMo AV response did not finish normally: {value}",
    )


def _validate_av_observation_usage(
    diagnostic: MimoCompletionDiagnostic,
    *,
    require_explicit_audio: bool,
) -> None:
    if diagnostic.usage.image_tokens == 0:
        raise MimoBackendFailure(
            code="mimo_reference_images_not_observed",
            reason="MiMo reported zero frozen-reference image tokens",
        )
    if diagnostic.usage.image_tokens is None:
        diagnostic.warnings.append("image_tokens_unavailable")
    if diagnostic.usage.video_tokens == 0:
        raise MimoBackendFailure(
            code="mimo_target_video_not_observed",
            reason="MiMo reported zero target-video tokens",
        )
    if diagnostic.usage.video_tokens is None:
        diagnostic.warnings.append("video_tokens_unavailable")
    if require_explicit_audio:
        if diagnostic.usage.audio_tokens == 0:
            raise MimoBackendFailure(
                code="mimo_target_audio_not_observed",
                reason="MiMo reported zero explicit canonical-audio tokens",
            )
        if diagnostic.usage.audio_tokens is None:
            diagnostic.warnings.append(
                "audio_tokens_unavailable_after_explicit_fallback"
            )
    elif diagnostic.usage.audio_tokens is None:
        diagnostic.warnings.append("audio_tokens_unavailable")


class _MimoHTTPAttemptsExhausted(RuntimeError):
    def __init__(self, original: Exception, *, attempts: int, retries: int) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
        self.attempts = attempts
        self.retries = retries


class _MimoResponseContractError(RuntimeError):
    def __init__(self, original: Exception, *, attempts: int, retries: int) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
        self.attempts = attempts
        self.retries = retries


def validate_annotation(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str],
    segment_intervals: dict[str, tuple[float, float]],
    transcribed_segment_ids: list[str],
    authoritative_transcripts: list[str],
    allowed_entity_ids: set[str],
    allowed_reference_labels: set[str],
    reference_subjects: list[Any],
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
        if decision.binding_status == "visible_entity" and (
            decision.speech_presentation != "onscreen_spoken"
            or decision.entity_id is None
            or "visible_lip_motion" not in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "visible_entity_requires_confirmed_onscreen_speech",
                    decision.segment_id,
                    "visible_entity requires onscreen_spoken, entity_id, and visible_lip_motion",
                )
            )
        if decision.speech_presentation == "onscreen_spoken" and (
            "visible_lip_motion" not in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_speech_requires_visible_lip_motion",
                    decision.segment_id,
                    "onscreen_spoken requires visible_lip_motion",
                )
            )
        if (
            decision.speech_presentation == "onscreen_spoken"
            and decision.binding_status == "offscreen"
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_speech_cannot_be_offscreen",
                    decision.segment_id,
                    "onscreen_spoken requires visible_entity, no_reliable_entity, or uncertain",
                )
            )
        if decision.speech_presentation != "onscreen_spoken" and (
            decision.binding_status == "visible_entity" or decision.entity_id is not None
        ):
            issues.append(
                ValidationIssue(
                    "non_onscreen_speech_claims_visible_entity",
                    decision.segment_id,
                    "non-onscreen speech cannot publish a visible entity",
                )
            )
        if decision.entity_id is not None and decision.entity_id not in allowed_entity_ids:
            issues.append(
                ValidationIssue(
                    "unknown_entity",
                    decision.segment_id,
                    f"unknown supplied entity: {decision.entity_id}",
                )
            )
    warning_segment_ids = [item.segment_id for item in annotation.warnings]
    for segment_id in warning_segment_ids:
        if segment_id not in segment_ids:
            issues.append(
                ValidationIssue(
                    "warning_unknown_segment",
                    "warnings",
                    f"unknown warning segment: {segment_id}",
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
    entities_by_group: dict[str, set[str]] = {}
    groups_by_entity: dict[str, set[str]] = {}
    for decision in decisions:
        if (
            decision.resolution == "resolved"
            and decision.binding_status == "visible_entity"
            and decision.primary_speaker_group is not None
            and decision.entity_id is not None
        ):
            entities_by_group.setdefault(decision.primary_speaker_group, set()).add(
                decision.entity_id
            )
            groups_by_entity.setdefault(decision.entity_id, set()).add(
                decision.primary_speaker_group
            )
    for group, entity_ids in entities_by_group.items():
        if len(entity_ids) > 1:
            issues.append(
                ValidationIssue(
                    "speaker_group_entity_contradiction",
                    "segment_decisions",
                    f"{group} maps to multiple visible entities: {sorted(entity_ids)}",
                )
            )
    for entity_id, groups in groups_by_entity.items():
        if len(groups) > 1:
            issues.append(
                ValidationIssue(
                    "visible_entity_speaker_group_contradiction",
                    "segment_decisions",
                    f"{entity_id} maps to multiple resolved groups: {sorted(groups)}",
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
    required_delivery_groups = {
        item.primary_speaker_group
        for item in decisions
        if item.segment_id in transcribed_segment_ids
        and item.resolution == "resolved"
        and item.primary_speaker_group is not None
    }
    actual_delivery_groups = {
        item.speaker_group for item in annotation.audio_semantics.speaker_delivery
    }
    missing_delivery_groups = sorted(required_delivery_groups - actual_delivery_groups)
    if missing_delivery_groups:
        issues.append(
            ValidationIssue(
                "missing_resolved_transcribed_speaker_delivery",
                "audio_semantics.speaker_delivery",
                str(missing_delivery_groups),
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
    draft = annotation.h3_draft
    shot_templates = [shot.description_template for shot in draft.shots]
    placeholders = [
        match.group(1)
        for template in shot_templates
        for match in _PLACEHOLDER.finditer(template)
    ]
    placeholder_counts = {item: placeholders.count(item) for item in set(placeholders)}
    if (
        placeholders != transcribed_segment_ids
        or any(count != 1 for count in placeholder_counts.values())
    ):
        issues.append(
            ValidationIssue(
                "speech_placeholder_inventory_mismatch",
                "h3_draft.shots",
                "transcribed segment placeholders must appear exactly once in order",
            )
        )
    event_placeholders = [
        match.group(1)
        for template in shot_templates
        for match in _AUDIO_EVENT_PLACEHOLDER.finditer(template)
    ]
    expected_event_ids = [
        event.event_id
        for event in annotation.audio_semantics.temporal_non_speech_events
    ]
    event_counts = {
        item: event_placeholders.count(item) for item in set(event_placeholders)
    }
    unknown_events = sorted(set(event_placeholders) - set(expected_event_ids))
    missing_events = [item for item in expected_event_ids if event_counts.get(item, 0) == 0]
    duplicate_events = sorted(
        item for item, count in event_counts.items() if count > 1
    )
    if unknown_events:
        issues.append(
            ValidationIssue(
                "unknown_audio_event_placeholder",
                "h3_draft.shots",
                str(unknown_events),
            )
        )
    if missing_events:
        issues.append(
            ValidationIssue(
                "missing_audio_event_placeholder",
                "h3_draft.shots",
                str(missing_events),
            )
        )
    if duplicate_events:
        issues.append(
            ValidationIssue(
                "duplicate_audio_event_placeholder",
                "h3_draft.shots",
                str(duplicate_events),
            )
        )
    if not unknown_events and not missing_events and not duplicate_events and (
        event_placeholders != expected_event_ids
    ):
        issues.append(
            ValidationIssue(
                "audio_event_placeholder_order_mismatch",
                "h3_draft.shots",
                "Audio-event placeholders must follow chronological event order",
            )
        )
    non_shot_text = "\n".join(
        (
            *draft.subject_definitions,
            draft.summary,
            *draft.visual_retention_analysis,
        )
    )
    if _PLACEHOLDER.search(non_shot_text):
        issues.append(
            ValidationIssue(
                "speech_placeholder_outside_shot",
                "h3_draft",
                "speech placeholders are allowed only in shot descriptions",
            )
        )
    if _AUDIO_EVENT_PLACEHOLDER.search(non_shot_text):
        issues.append(
            ValidationIssue(
                "audio_event_placeholder_outside_shot",
                "h3_draft",
                "Audio-event placeholders are allowed only in shot descriptions",
            )
        )
    for shot in draft.shots:
        if shot.shot_index > 1 and (
            shot.start_time is None
            or shot.start_time <= 0
            or shot.start_time >= target_duration_seconds
        ):
            issues.append(
                ValidationIssue(
                    "shot_start_outside_target",
                    "h3_draft.shots",
                    f"shot {shot.shot_index} start must be inside target duration",
                )
            )
        for match in _PLACEHOLDER.finditer(shot.description_template):
            prefix = shot.description_template[
                max(0, match.start() - 100) : match.start()
            ]
            if _SPEECH_LEAD_IN.search(prefix):
                issues.append(
                    ValidationIssue(
                        "draft_prefixes_complete_speech_placeholder",
                        "h3_draft.shots",
                        f"shot {shot.shot_index} prefixes a complete speech clause",
                    )
                )
    shot_intervals = [
        (
            0.0 if index == 0 else float(shot.start_time),
            (
                target_duration_seconds
                if index + 1 == len(draft.shots)
                else float(draft.shots[index + 1].start_time)
            ),
        )
        for index, shot in enumerate(draft.shots)
        if (index == 0 or shot.start_time is not None)
        and (
            index + 1 == len(draft.shots)
            or draft.shots[index + 1].start_time is not None
        )
    ]
    if len(shot_intervals) == len(draft.shots):
        event_by_id = {
            event.event_id: (
                event.approximate_start_time,
                event.approximate_end_time,
            )
            for event in annotation.audio_semantics.temporal_non_speech_events
        }
        for index, shot in enumerate(draft.shots):
            shot_start, shot_end = shot_intervals[index]
            for match in _PLACEHOLDER.finditer(shot.description_template):
                interval = segment_intervals.get(match.group(1))
                if interval is not None and not (
                    interval[0] < shot_end and interval[1] > shot_start
                ):
                    issues.append(
                        ValidationIssue(
                            "speech_placeholder_wrong_shot",
                            "h3_draft.shots",
                            f"{match.group(1)} does not overlap shot {shot.shot_index}",
                        )
                    )
            for match in _AUDIO_EVENT_PLACEHOLDER.finditer(
                shot.description_template
            ):
                interval = event_by_id.get(match.group(1))
                if interval is not None and not (
                    interval[0] < shot_end and interval[1] > shot_start
                ):
                    issues.append(
                        ValidationIssue(
                            "audio_event_placeholder_wrong_shot",
                            "h3_draft.shots",
                            f"{match.group(1)} does not overlap shot {shot.shot_index}",
                        )
                    )
    draft_text = "\n".join((*shot_templates, non_shot_text))
    if _PIPELINE_OWNED.search(draft_text):
        issues.append(
            ValidationIssue(
                "draft_contains_pipeline_owned_syntax",
                "h3_draft",
                "MiMo draft contains pipeline-owned H3 syntax",
            )
        )
    if _KEYFRAME_ROLE.search(draft_text):
        issues.append(
            ValidationIssue(
                "unassigned_picture_keyframe_role",
                "h3_draft",
                "Pictures are content references, not frame anchors",
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
    retention_text = "\n".join(draft.visual_retention_analysis)
    if "attribute_transfer" in retention_text:
        issues.append(
            ValidationIssue(
                "unassigned_attribute_transfer",
                "h3_draft.visual_retention_analysis",
                "current contract does not assign attribute transfer",
            )
        )
    unknown_markers = sorted(
        {
            marker
            for marker in _RETENTION_MARKER.findall(retention_text)
            if marker not in _ALLOWED_RETENTION_MARKERS
            and marker != "attribute_transfer"
        }
    )
    if unknown_markers:
        issues.append(
            ValidationIssue(
                "unknown_visual_retention_marker",
                "h3_draft.visual_retention_analysis",
                str(unknown_markers),
            )
        )
    supplied_subject_labels = {str(item.subject_label) for item in reference_subjects}
    retention_subject_labels: list[str] = []
    for line in draft.visual_retention_analysis:
        match = re.match(r"^\s*(<Subject [1-9]\d*>)", line)
        if match is not None:
            retention_subject_labels.append(match.group(1))
    unknown_retention_subjects = sorted(
        set(retention_subject_labels) - supplied_subject_labels
    )
    if unknown_retention_subjects:
        issues.append(
            ValidationIssue(
                "unknown_subject_retention",
                "h3_draft.visual_retention_analysis",
                str(unknown_retention_subjects),
            )
        )
    if len(draft.visual_retention_analysis) != len(reference_subjects):
        issues.append(
            ValidationIssue(
                "subject_retention_contract_mismatch",
                "h3_draft.visual_retention_analysis",
                "retention rows must exactly cover supplied Subjects",
            )
        )
    for subject_label in sorted(supplied_subject_labels):
        rows = [
            line
            for line in draft.visual_retention_analysis
            if line.lstrip().startswith(subject_label)
        ]
        markers = [
            marker
            for line in rows
            for marker in _ALLOWED_RETENTION_MARKER_OCCURRENCE.findall(line)
        ]
        subject_labels = (
            [
                match.group(0)
                for match in _PICTURE_OR_SUBJECT.finditer(rows[0])
                if match.group(1) == "Subject"
            ]
            if len(rows) == 1
            else []
        )
        if (
            len(rows) != 1
            or len(markers) != 1
            or subject_labels != [subject_label]
        ):
            issues.append(
                ValidationIssue(
                    "subject_retention_contract_mismatch",
                    "h3_draft.visual_retention_analysis",
                    f"{subject_label} requires exactly one allowed retention line",
                )
            )
    if len(draft.subject_definitions) != len(reference_subjects):
        issues.append(
            ValidationIssue(
                "subject_definition_contract_mismatch",
                "h3_draft.subject_definitions",
                "definition rows must exactly cover supplied Subjects",
            )
        )
    for subject in reference_subjects:
        subject_label = str(subject.subject_label)
        expected_pictures = set(subject.source_picture_labels)
        definitions = [
            item
            for item in draft.subject_definitions
            if item.lstrip().startswith(subject_label)
        ]
        actual_pictures = (
            {
                match.group(0)
                for match in _PICTURE_OR_SUBJECT.finditer(definitions[0])
                if match.group(1) == "Picture"
            }
            if len(definitions) == 1
            else set()
        )
        definition_subjects = (
            [
                match.group(0)
                for match in _PICTURE_OR_SUBJECT.finditer(definitions[0])
                if match.group(1) == "Subject"
            ]
            if len(definitions) == 1
            else []
        )
        if (
            len(definitions) != 1
            or definition_subjects != [subject_label]
            or actual_pictures != expected_pictures
        ):
            issues.append(
                ValidationIssue(
                    "subject_definition_contract_mismatch",
                    "h3_draft.subject_definitions",
                    f"{subject_label} must retain exactly its source Pictures",
                )
            )
    model_owned_shots = [
        _AUDIO_EVENT_PLACEHOLDER.sub(
            " ", _PLACEHOLDER.sub(" ", shot.description_template)
        )
        for shot in draft.shots
    ]
    model_owned_text = _normalized_text(
        "\n".join(
            (
                *draft.subject_definitions,
                draft.summary,
                *draft.visual_retention_analysis,
                *model_owned_shots,
            )
        )
    )
    for transcript in authoritative_transcripts:
        normalized_transcript = _normalized_text(transcript)
        if (
            _significant_transcript(transcript)
            and normalized_transcript
            and normalized_transcript in model_owned_text
        ):
            issues.append(
                ValidationIssue(
                    "draft_contains_authoritative_transcript",
                    "h3_draft",
                    "model-authored draft copies exact authoritative ASR text",
                )
            )
            break
    audio_semantics = annotation.audio_semantics
    model_owned_audio_fields = [
        *(event.description for event in audio_semantics.temporal_non_speech_events),
        *(item.delivery_style for item in audio_semantics.speaker_delivery),
        *(
            [audio_semantics.overall_soundscape]
            if audio_semantics.overall_soundscape is not None
            else []
        ),
        *(
            [audio_semantics.non_diegetic_music]
            if audio_semantics.non_diegetic_music is not None
            else []
        ),
        audio_semantics.audiovisual_summary,
    ]
    normalized_audio_fields = [
        _normalized_text(value) for value in model_owned_audio_fields
    ]
    for transcript in authoritative_transcripts:
        normalized_transcript = _normalized_text(transcript)
        if (
            _significant_transcript(transcript)
            and normalized_transcript
            and any(
                normalized_transcript in field for field in normalized_audio_fields
            )
        ):
            issues.append(
                ValidationIssue(
                    "audio_semantics_contains_authoritative_transcript",
                    "audio_semantics",
                    "model-authored Audio semantics copies exact authoritative ASR text",
                )
            )
            break
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
                    raise _MimoHTTPAttemptsExhausted(
                        exc,
                        attempts=attempt,
                        retries=retries,
                    ) from exc
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
                },
                "fps": self.config.video_fps,
                "media_resolution": self.config.media_resolution,
            }
        )
        if include_audio_fallback:
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": self.config.media_resolver.resolve(
                            Path(job.target_full_audio_path)
                        )
                    },
                }
            )
        return content

    @staticmethod
    def build_compact_task_contract(job: MimoBackendJob) -> dict[str, object]:
        payload = job.model_dump(mode="json")
        references = payload.get("reference_images", [])
        if not isinstance(references, list):
            raise TypeError("MiMo reference inventory is invalid")
        compact_references = []
        for reference in references:
            if not isinstance(reference, dict):
                raise TypeError("MiMo reference metadata is invalid")
            compact_references.append(
                {
                    key: value
                    for key, value in reference.items()
                    if key not in {"image_artifact_path", "image_sha256"}
                }
            )
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            raise TypeError("MiMo segment inventory is invalid")
        return {
            "clip_uid": job.clip_uid,
            "r2v_instruction": job.r2v_instruction,
            "target_duration_seconds": job.target_duration_seconds,
            "reference_images": compact_references,
            "reference_subjects": payload.get("reference_subjects", []),
            "segments": segments,
            "allowed_segment_ids": [item.segment_id for item in job.segments],
            "transcribed_segment_ids": [
                item.segment_id
                for item in job.segments
                if item.asr_status == "transcribed"
            ],
            "allowed_speaker_bindable_entity_ids": sorted(
                {
                    item.entity_id
                    for item in job.reference_images
                    if item.kind == "subject" and item.entity_id is not None
                }
            ),
            "allowed_picture_labels": [
                item.picture_label for item in job.reference_images
            ],
            "allowed_subject_labels": [
                item.subject_label for item in job.reference_subjects
            ],
        }

    def _prompt(self, job: MimoBackendJob) -> str:
        return (
            "Return exactly one JSON object matching this schema, with no markdown or extra fields.\n"
            + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            + "\nR2V INSTRUCTION:\n"
            + job.r2v_instruction
            + "\nThis is task intent and a non-authoritative visual hint. It may tell you how "
            "the supplied references are intended to participate in generation, but it must "
            "never override what is actually observed in the target video."
            + "\nCOMPACT AUTHORITATIVE INPUT:\n"
            + _compact_json(self.build_compact_task_contract(job))
        )

    def _request(
        self,
        job: MimoBackendJob,
        *,
        include_audio_fallback: bool,
        recheck_prompt: str | None = None,
    ) -> tuple[str, MimoCompletionDiagnostic, int]:
        if recheck_prompt is not None:
            modality = (
                "full_av_recheck_with_canonical_audio"
                if include_audio_fallback
                else "full_av_recheck_embedded_audio"
            )
        else:
            modality = (
                "target_video_plus_canonical_full_audio_fallback"
                if include_audio_fallback
                else "target_video_with_embedded_audio"
            )
        content = self._media_content(job, include_audio_fallback=include_audio_fallback)
        content.append(
            {"type": "text", "text": recheck_prompt or self._prompt(job)}
        )
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
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        completion, attempts, retries = self._call(payload)
        try:
            choices = _value(completion, "choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("MiMo response has no choices")
            choice = choices[0]
            message = _value(choice, "message")
            raw = _value(message, "content")
            if not isinstance(raw, str):
                raise TypeError("MiMo response content must be text")
        except Exception as exc:
            raise _MimoResponseContractError(
                exc,
                attempts=attempts,
                retries=retries,
            ) from exc
        return raw, _completion_diagnostic(
            completion,
            choice,
            modality=modality,
            http_attempt_count=attempts,
        ), retries

    def _full_av_recheck_prompt(
        self,
        job: MimoBackendJob,
        *,
        invalid_response: str,
        issues: list[ValidationIssue],
    ) -> str:
        return (
            "Reinspect the SAME audiovisual evidence. Do not merely edit fields to "
            "satisfy validation. Resolve speaker/entity/audio-semantic contradictions "
            "from the audiovisual evidence while preserving all authoritative DiariZen "
            "timing and Qwen3-ASR text. Do not change segment boundaries, sample "
            "boundaries, ASR text, ASR language, or the reference inventory. Return "
            "exactly one compact JSON object with no markdown or extra fields."
            "\nCOMPACT AUTHORITATIVE TASK CONTRACT: "
            + _compact_json(self.build_compact_task_contract(job))
            + "\nSCHEMA: "
            + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            + "\nVALIDATION ISSUES: "
            + _compact_json([item.to_dict() for item in issues])
            + "\nPREVIOUS INVALID RESPONSE: "
            + invalid_response
        )

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
        modality: Literal[
            "target_video_with_embedded_audio",
            "target_video_plus_canonical_full_audio_fallback",
        ] = "target_video_with_embedded_audio"
        recheck_count = 0

        def contextual_failure(
            failure: MimoBackendFailure,
            *,
            validation_issues: list[ValidationIssue] | None = None,
        ) -> MimoBackendFailure:
            return MimoBackendFailure(
                code=failure.code,
                reason=failure.reason,
                raw_responses=tuple(raw_responses),
                diagnostics=tuple(diagnostics),
                issues=tuple(validation_issues or failure.issues),
                model_call_count=len(diagnostics),
                http_attempt_count=sum(item.http_attempt_count for item in diagnostics),
                http_retry_count=sum(
                    item.http_attempt_count - 1 for item in diagnostics
                ),
                recheck_count=recheck_count,
            )

        def perform_request(
            *,
            include_audio: bool,
            recheck_prompt: str | None = None,
        ) -> tuple[str, MimoCompletionDiagnostic]:
            attempted_modality = (
                "full_av_recheck_with_canonical_audio"
                if recheck_prompt is not None and include_audio
                else "full_av_recheck_embedded_audio"
                if recheck_prompt is not None
                else "target_video_plus_canonical_full_audio_fallback"
                if include_audio
                else "target_video_with_embedded_audio"
            )
            try:
                raw, diagnostic, _ = self._request(
                    job,
                    include_audio_fallback=include_audio,
                    recheck_prompt=recheck_prompt,
                )
            except MimoBackendFailure as failure:
                raise contextual_failure(failure) from failure
            except (_MimoHTTPAttemptsExhausted, _MimoResponseContractError) as exc:
                diagnostics.append(
                    MimoCompletionDiagnostic(
                        input_modality=attempted_modality,
                        usage=MimoUsage(),
                        http_attempt_count=exc.attempts,
                        warnings=["http_request_failed"],
                        request_error=str(exc),
                    )
                )
                raise contextual_failure(
                    MimoBackendFailure(code="mimo_request_failed", reason=str(exc))
                ) from exc.original
            except Exception as exc:
                raise contextual_failure(
                    MimoBackendFailure(
                        code="mimo_request_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                ) from exc
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            try:
                _validate_finish_reason(diagnostic)
                _validate_av_observation_usage(
                    diagnostic,
                    require_explicit_audio=include_audio,
                )
                if recheck_prompt is not None and diagnostic.usage.audio_tokens == 0:
                    raise MimoBackendFailure(
                        code="mimo_target_audio_not_observed",
                        reason="MiMo full AV recheck reported zero target-audio tokens",
                    )
            except MimoBackendFailure as failure:
                raise contextual_failure(failure) from failure
            return raw, diagnostic

        raw, diagnostic = perform_request(include_audio=False)
        include_audio = diagnostic.usage.audio_tokens == 0
        if include_audio:
            raw, _ = perform_request(include_audio=True)
            modality = "target_video_plus_canonical_full_audio_fallback"

        def parse_and_validate(value: str) -> tuple[MimoAVAnnotationDraft | None, list[ValidationIssue]]:
            annotation, validation_issues = parse_structured_json_issues(
                value, MimoAVAnnotationDraft
            )
            if annotation is not None:
                validation_issues = validate_annotation(
                    annotation,
                    segment_ids=segment_ids,
                    segment_intervals={
                        item.segment_id: (item.start_time, item.end_time)
                        for item in job.segments
                    },
                    transcribed_segment_ids=transcribed_segment_ids,
                    authoritative_transcripts=[
                        item.asr_text
                        for item in job.segments
                        if item.asr_status == "transcribed" and item.asr_text is not None
                    ],
                    allowed_entity_ids=allowed_entity_ids,
                    allowed_reference_labels=allowed_reference_labels,
                    reference_subjects=job.reference_subjects,
                    target_duration_seconds=job.target_duration_seconds,
                )
            return annotation, validation_issues

        annotation, issues = parse_and_validate(raw)
        if annotation is None or issues:
            recheck_count = 1
            raw, _ = perform_request(
                include_audio=include_audio,
                recheck_prompt=self._full_av_recheck_prompt(
                    job,
                    invalid_response=raw,
                    issues=issues,
                ),
            )
            annotation, issues = parse_and_validate(raw)
        if annotation is None or issues:
            raise contextual_failure(
                MimoBackendFailure(
                    code="mimo_structured_output_failed",
                    reason="MiMo AV annotation failed after one full AV recheck",
                ),
                validation_issues=issues,
            )
        return MimoBackendResult(
            annotation=annotation,
            raw_responses=tuple(raw_responses),
            diagnostics=tuple(diagnostics),
            model_call_count=len(diagnostics),
            http_attempt_count=sum(item.http_attempt_count for item in diagnostics),
            http_retry_count=sum(
                item.http_attempt_count - 1 for item in diagnostics
            ),
            recheck_count=recheck_count,
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
    "SpeechPresentation",
    "sha256_file",
    "validate_annotation",
]
