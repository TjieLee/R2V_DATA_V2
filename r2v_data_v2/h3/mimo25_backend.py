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
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import quote, urlsplit

from openai import OpenAI
from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speech_presentation import SpeechPresentation
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

MIMO25_MODEL = "mimo-v2.5"
MIMO25_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO25_PROMPT_VERSION = "h3_mimo25_unified_av_reconcile_v14"
MIMO25_POLICY_VERSION = "h3_mimo25_av_authority_contract_v9"
MIMO25_SCHEMA_VERSION = "r2v.h3.mimo25_av_annotation.10"
MIMO25_BACKEND_VERSION = "r2v.h3.mimo25_backend.12"
MIMO25_MATERIALIZER_VERSION = "h3_mimo25_materializer_v9"
DEFAULT_BASE64_LIMIT_BYTES = 50 * 1024 * 1024
MimoTransport = Literal["xiaomi", "sglang"]

_GROUP = re.compile(r"g([1-9]\d*)")
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
EvidenceCode = Literal[
    "visible_lip_motion",
    "speaker_visible_mouth_occluded",
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


def _has_onscreen_speaker_evidence(evidence_codes: list[EvidenceCode]) -> bool:
    evidence = set(evidence_codes)
    direct = "visible_lip_motion" in evidence
    occluded = (
        "speaker_visible_mouth_occluded" in evidence
        and bool(evidence & {"av_temporal_alignment", "voice_continuity"})
    )
    return direct or occluded


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


class _MimoSegmentDecisionBase(SchemaModel):
    segment_id: str = Field(min_length=1)
    vocal_composition: VocalComposition
    resolution: Resolution
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")
    binding_status: BindingStatus
    speech_presentation: SpeechPresentation
    entity_id: str | None
    delivery_style: StrictStr | None
    secondary_vocal_activity: MimoSecondaryVocalActivity
    confidence: Literal["high", "medium", "low"]
    evidence_codes: list[EvidenceCode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> _MimoSegmentDecisionBase:
        if self.delivery_style is not None and not self.delivery_style.strip():
            raise ValueError("MiMo segment delivery must be non-empty or null")
        if self.resolution == "resolved" and self.primary_speaker_group is None:
            raise ValueError("resolved segment requires one primary speaker group")
        if self.binding_status == "visible_entity":
            if self.entity_id is None:
                raise ValueError("visible_entity requires supplied entity_id")
            if self.speech_presentation != "onscreen_spoken":
                raise ValueError("visible_entity requires onscreen_spoken presentation")
        elif self.entity_id is not None:
            raise ValueError("only visible_entity may publish entity_id")
        if self.speech_presentation == "onscreen_spoken":
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


class MimoResolvedSegmentDecision(_MimoSegmentDecisionBase):
    resolution: Literal["resolved"]
    primary_speaker_group: str = Field(pattern=r"^g[1-9]\d*$")


class MimoAcousticRefinementSegmentDecision(_MimoSegmentDecisionBase):
    resolution: Literal["needs_acoustic_refinement"]
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")


class MimoUncertainSegmentDecision(_MimoSegmentDecisionBase):
    resolution: Literal["uncertain"]
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")


MimoSegmentDecision = Annotated[
    MimoResolvedSegmentDecision
    | MimoAcousticRefinementSegmentDecision
    | MimoUncertainSegmentDecision,
    Field(discriminator="resolution"),
]


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
        "non_diegetic_music",
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
            or "[[" in self.description
            or "]]" in self.description
        ):
            raise ValueError("MiMo Audio event description contains forbidden syntax")
        return self


class MimoAudioSemantics(SchemaModel):
    temporal_non_speech_events: list[MimoAudioEvent]
    overall_soundscape_status: Literal["present", "absent", "unknown"]
    overall_soundscape: StrictStr | None = None
    non_diegetic_music_status: Literal["present", "absent", "unknown"]
    non_diegetic_music: StrictStr | None = None
    audiovisual_summary: StrictStr

    @model_validator(mode="after")
    def validate_semantics(self) -> MimoAudioSemantics:
        if self.overall_soundscape_status == "present":
            if self.overall_soundscape is None or not self.overall_soundscape.strip():
                raise ValueError("present soundscape requires a concise description")
        elif self.overall_soundscape is not None:
            raise ValueError("absent or unknown soundscape cannot publish a description")
        if self.non_diegetic_music_status == "present":
            if self.non_diegetic_music is None or not self.non_diegetic_music.strip():
                raise ValueError("present music requires a concise description")
        elif self.non_diegetic_music is not None:
            raise ValueError("absent or unknown music cannot publish a description")
        has_non_diegetic_event = any(
            event.category == "non_diegetic_music"
            for event in self.temporal_non_speech_events
        )
        if has_non_diegetic_event and self.non_diegetic_music_status != "present":
            raise ValueError(
                "non-diegetic music event requires present global music semantics"
            )
        has_audible_soundscape_event = any(
            event.category != "non_diegetic_music"
            for event in self.temporal_non_speech_events
        )
        if has_audible_soundscape_event and self.overall_soundscape_status != "present":
            raise ValueError(
                "audible non-speech event requires present global soundscape semantics"
            )
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
        if not self.audiovisual_summary.strip():
            raise ValueError("MiMo audiovisual summary must not be empty")
        if self.overall_soundscape is not None and not self.overall_soundscape.strip():
            raise ValueError("optional MiMo Audio text must be non-empty or null")
        return self


class MimoH3ProsePart(SchemaModel):
    type: Literal["prose"]
    text: StrictStr

    @model_validator(mode="after")
    def validate_prose(self) -> MimoH3ProsePart:
        text_without_allowed_references = _PICTURE_OR_SUBJECT.sub("", self.text)
        if (
            not self.text.strip()
            or "[[" in self.text
            or "]]" in self.text
            or _PIPELINE_OWNED.search(self.text)
            or "<" in text_without_allowed_references
            or ">" in text_without_allowed_references
        ):
            raise ValueError("MiMo H3 prose contains empty or pipeline-owned syntax")
        return self


class MimoH3SpeechPart(SchemaModel):
    type: Literal["speech"]
    segment_id: str = Field(min_length=1)


class MimoH3AudioEventPart(SchemaModel):
    type: Literal["audio_event"]
    event_id: str = Field(pattern=r"^ae[1-9]\d*$")


MimoH3TimelinePart = Annotated[
    MimoH3ProsePart | MimoH3SpeechPart | MimoH3AudioEventPart,
    Field(discriminator="type"),
]


class MimoH3Shot(SchemaModel):
    shot_index: int = Field(gt=0)
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    timeline_parts: list[MimoH3TimelinePart] = Field(min_length=1)


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
            *(
                part.text
                for shot in self.shots
                for part in shot.timeline_parts
                if isinstance(part, MimoH3ProsePart)
            ),
        )
        if any(not value.strip() for value in values):
            raise ValueError("MiMo H3 draft text must not be empty")
        indexes = [shot.shot_index for shot in self.shots]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("MiMo H3 shots must use contiguous indexes")
        if self.shots[0].start_time not in {None, 0}:
            raise ValueError("MiMo H3 Shot 1 must start implicitly or at zero")
        later = [shot.start_time for shot in self.shots[1:]]
        if any(value is None for value in later):
            raise ValueError("later MiMo H3 shots require hard-cut times")
        numeric = [value for value in later if value is not None]
        if any(value <= 0 for value in numeric):
            raise ValueError("later MiMo H3 hard-cut times must be positive")
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


class MimoSpeakerVoiceProfile(SchemaModel):
    speaker_group: str = Field(pattern=r"^g[1-9]\d*$")
    voice_characteristics: StrictStr | None

    @model_validator(mode="after")
    def validate_profile(self) -> MimoSpeakerVoiceProfile:
        if self.voice_characteristics is not None and not self.voice_characteristics.strip():
            raise ValueError("MiMo voice profile must be non-empty or null")
        return self


class MimoAVAnnotationDraft(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_av_annotation.10"] = MIMO25_SCHEMA_VERSION
    segment_decisions: list[MimoSegmentDecision]
    speaker_voice_profiles: list[MimoSpeakerVoiceProfile]
    audio_semantics: MimoAudioSemantics
    h3_draft: MimoH3Draft
    warnings: list[MimoAnnotationWarning] = Field(default_factory=list)


class MimoThinkingContract(SchemaModel):
    type: Literal["disabled"] = "disabled"


class MimoBackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_backend.12"] = MIMO25_BACKEND_VERSION
    backend: Literal[
        "xiaomi_openai_compatible", "sglang_openai_compatible"
    ]
    transport: MimoTransport
    model: Literal["mimo-v2.5"] = MIMO25_MODEL
    base_url: str
    video_fps: Literal[4.0] = 4.0
    media_resolution: Literal["default"] = "default"
    thinking: MimoThinkingContract
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_completion_tokens: int = Field(gt=0)
    response_format: Literal["json_object", "json_schema"]
    stream: Literal[False] = False
    media_mode: Literal["base64", "http"]
    media_root: str
    media_base_url: str | None = None
    prompt_version: Literal["h3_mimo25_unified_av_reconcile_v14"] = (
        MIMO25_PROMPT_VERSION
    )
    policy_version: Literal["h3_mimo25_av_authority_contract_v9"] = (
        MIMO25_POLICY_VERSION
    )
    annotation_schema_version: Literal["r2v.h3.mimo25_av_annotation.10"] = (
        MIMO25_SCHEMA_VERSION
    )
    materializer_version: Literal[
        "h3_mimo25_materializer_v6",
        "h3_mimo25_materializer_v7",
        "h3_mimo25_materializer_v8",
        "h3_mimo25_materializer_v9",
    ] = (
        MIMO25_MATERIALIZER_VERSION
    )
    http_max_attempts: int = Field(ge=1, le=5)
    full_av_recheck_limit: Literal[1] = 1
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> MimoBackendProvenance:
        if not self.base_url.strip() or not self.media_root.strip():
            raise ValueError("MiMo endpoint and media root are required")
        expected_backend = f"{self.transport}_openai_compatible"
        if self.backend != expected_backend:
            raise ValueError("MiMo backend provenance differs from transport")
        expected_response_format = (
            "json_schema" if self.transport == "sglang" else "json_object"
        )
        if self.response_format != expected_response_format:
            raise ValueError("MiMo response format differs from transport")
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
    transport: MimoTransport = "xiaomi"
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
            or self.transport not in {"xiaomi", "sglang"}
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
            raise ValueError("MiMo backend configuration violates the v7 contract")

    def provenance(self) -> MimoBackendProvenance:
        values = {
            "schema_version": MIMO25_BACKEND_VERSION,
            "backend": f"{self.transport}_openai_compatible",
            "transport": self.transport,
            "model": self.model,
            "base_url": self.base_url,
            "video_fps": self.video_fps,
            "media_resolution": self.media_resolution,
            "thinking": {"type": "disabled"},
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": (
                "json_schema" if self.transport == "sglang" else "json_object"
            ),
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

AUTHORITY
- Preserve every supplied DiariZen segment, exact timing, and sample range. Every segment receives one decision even with LR-ASD=0, no binding, or zero direct-anchor support; never split, merge, delete, filter, or invent timing.
- Qwen3-ASR text and language are immutable. Never transcribe, quote, correct, paraphrase, translate, repunctuate, or move dialogue; the schema has no transcript field.
- Frozen Visual entities, Pictures, Subjects, order, and ownership are immutable. LR-ASD, source clusters, and current bindings are fallible proposals. The target video is observation, never <Video N>.
- Sound requires audible evidence. Visual evidence may disambiguate a genuinely audible source but cannot invent sound. Return one compact JSON object matching the supplied contract.

SPEAKER / ENTITY / PRESENTATION
- segment_decisions follow every allowed_segment_id exactly once and in order. Clip-local g1, g2, ... identify speakers by first appearance, not turns. A pause, language, sentence, ASR, or segment boundary must not by itself create a new group. The same resolved visible entity reuses one group; one resolved group never maps to multiple visible entities. Source clusters may split or merge only with AV support.
- Every decision includes entity_id. visible_entity requires one supplied entity and onscreen_spoken. Other binding statuses use null. Visible presence alone is insufficient. Reliable onscreen evidence is visible_lip_motion, or genuine mouth occlusion/back view marked speaker_visible_mouth_occluded together with av_temporal_alignment or voice_continuity. Voice continuity may preserve speaker-group identity but cannot alone bind a visible entity. LR-ASD/direct anchors are neither sufficient nor required.
- Use offscreen_spoken, voice_over, message_voice_over, or device_playback when observed. Silent phone reading/typing must not become visible speech. Uncertain evidence stays uncertain and unbound. Never delete a segment for non-onscreen presentation.
- Multiple vocal sounds inside one segment never make that segment invalid. Same-speaker particles/nonlexical sounds preserve the group; secondary non-speech vocalization preserves primary dialogue ownership. Overlapping or sequential secondary speech preserves the segment but requires acoustic refinement without invented internal timing.

AUDIO SEMANTICS
- Emit only meaningful audible non-speech events. Coalesce repeated micro-events from one action; split distinct sources/times. Use contiguous chronological aeN IDs and concise English descriptions without transcript or H3 syntax.
- Music requires audible musical structure. In-scene music characters can hear is diegetic_music; audience-only score/BGM is non_diegetic_music. Do not infer source from scene plausibility: uncertain source keeps global non-diegetic status unknown. A typed non_diegetic_music event requires present global non-diegetic music and a description.
- overall_soundscape is a required core H3 semantic. Use present with one concise audible description whenever ambience, room tone, an environmental layer, a physical sound, a non-verbal human sound, or another meaningful non-speech sound is audible, including low-level background sound in ordinary dialogue scenes. Do not default a normal audiovisual clip to absent or unknown merely because no salient event was detected. Use absent only for verified complete silence of the soundscape; use unknown only when Audio evidence is genuinely unavailable or uncertain. Do not repeat dialogue in overall_soundscape, and do not use non-diegetic music as a substitute for it.
- overall_soundscape and non_diegetic_music descriptions exist only when status=present. Video may ground or disambiguate a genuinely audible source but never create room tone or another sound from visual context.
- Each transcribed segment requires concise audible delivery_style; non-transcribed segments use null. speaker_voice_profiles contains one row for every resolved speaker group owning transcribed speech, in first-appearance order. Describe only stable audible pitch register, timbre, texture, baseline cadence, articulation, and clearly supported accent/dialect in one concise English sentence, or null. Never copy dialogue or infer age, gender, nationality, role, identity, or personality.
- audiovisual_summary is concise observed AV context without dialogue, plot, relationship, intention, causality, or psychology.

H3 DRAFT
- Write dense generation-quality observed video prose, not a caption. For each real shot cover applicable visual style; shot scale and framing; camera angle; foreground, midground, and background composition; salient subject appearance and spatial relationships; pose; body, arm, hand, and head motion; gaze; facial expression changes; interactions; object states; environment/materials/readable text; lighting/color; camera motion or explicit stability; and temporal progression through early, middle, and late portions. Target roughly 300-450 English words only when evidence supports it; never pad.
- Use observed evidence, not plausible filler. Do not infer unsupported emotion/psychology, intent, causality, relationships, identity, sound, offscreen events, or object detail. Non-onscreen speech must not create visible speaking/lip motion. Add later shots only for real hard cuts.
- Subject definitions are natural official MiniMax H3 Ref2VA prose. Begin each row with its exact Subject and naturally integrate every and only its required Picture labels exactly once. Avoid mechanically repeating one connector such as "as seen in" across all rows; natural forms such as "in", "shown in", "depicted in", "from", or "whose appearance comes from" are all valid. Pictures are content references, not first frames, last frames, or keyframes.
- Retention has exactly one row per Subject and one of fully_preserved, partially_preserved, weak_reference. attribute_transfer is forbidden. Do not emit <Video N>, <Audio N>, (Sx), <d>, shot headers, donor provenance, or invented labels.
- timeline_parts alone place typed prose, speech, and audio events. Speech parts exactly equal transcribed_segment_ids once each in order and contain no dialogue; prose must not prefix a complete speech clause. Audio-event parts exactly match emitted aeN events once each in order and their descriptions are not repeated in prose. The deterministic materializer owns final official H3 syntax."""


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
            or not _has_onscreen_speaker_evidence(decision.evidence_codes)
        ):
            issues.append(
                ValidationIssue(
                    "visible_entity_requires_confirmed_onscreen_speech",
                    decision.segment_id,
                    "visible_entity requires onscreen_spoken, entity_id, and reliable onscreen speaker evidence",
                )
            )
        if decision.speech_presentation == "onscreen_spoken" and not (
            _has_onscreen_speaker_evidence(decision.evidence_codes)
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_speech_requires_reliable_visible_speaker_evidence",
                    decision.segment_id,
                    "onscreen_spoken requires visible articulation or mouth-occluded visible-speaker continuity",
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
    transcribed_segment_set = set(transcribed_segment_ids)
    for decision in decisions:
        if decision.segment_id in transcribed_segment_set:
            if decision.delivery_style is None:
                issues.append(
                    ValidationIssue(
                        "missing_transcribed_segment_delivery",
                        decision.segment_id,
                        "transcribed segment requires audible delivery_style",
                    )
                )
        elif decision.delivery_style is not None:
            issues.append(
                ValidationIssue(
                    "non_transcribed_segment_delivery",
                    decision.segment_id,
                    "non-transcribed segment requires delivery_style=null",
                )
            )
    required_profile_groups: list[str] = []
    for decision in decisions:
        group = decision.primary_speaker_group
        if (
            decision.resolution == "resolved"
            and decision.segment_id in transcribed_segment_set
            and group is not None
            and group not in required_profile_groups
        ):
            required_profile_groups.append(group)
    actual_profile_groups = [
        profile.speaker_group for profile in annotation.speaker_voice_profiles
    ]
    if actual_profile_groups != required_profile_groups or len(
        actual_profile_groups
    ) != len(set(actual_profile_groups)):
        issues.append(
            ValidationIssue(
                "speaker_voice_profile_inventory_mismatch",
                "speaker_voice_profiles",
                "voice profiles must exactly follow resolved transcribed speaker groups",
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
    prose_parts = [
        part.text
        for shot in draft.shots
        for part in shot.timeline_parts
        if isinstance(part, MimoH3ProsePart)
    ]
    speech_parts = [
        part.segment_id
        for shot in draft.shots
        for part in shot.timeline_parts
        if isinstance(part, MimoH3SpeechPart)
    ]
    speech_counts = {item: speech_parts.count(item) for item in set(speech_parts)}
    if (
        speech_parts != transcribed_segment_ids
        or any(count != 1 for count in speech_counts.values())
    ):
        issues.append(
            ValidationIssue(
                "speech_placeholder_inventory_mismatch",
                "h3_draft.shots",
                "typed transcribed speech parts must appear exactly once in order",
            )
        )
    event_parts = [
        part.event_id
        for shot in draft.shots
        for part in shot.timeline_parts
        if isinstance(part, MimoH3AudioEventPart)
    ]
    expected_event_ids = [
        event.event_id
        for event in annotation.audio_semantics.temporal_non_speech_events
    ]
    event_counts = {
        item: event_parts.count(item) for item in set(event_parts)
    }
    unknown_events = sorted(set(event_parts) - set(expected_event_ids))
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
        event_parts != expected_event_ids
    ):
        issues.append(
            ValidationIssue(
                "audio_event_placeholder_order_mismatch",
                "h3_draft.shots",
                "typed Audio-event parts must follow chronological event order",
            )
        )
    non_shot_text = "\n".join(
        (
            *draft.subject_definitions,
            draft.summary,
            *draft.visual_retention_analysis,
        )
    )
    if "[[" in non_shot_text or "]]" in non_shot_text:
        issues.append(
            ValidationIssue(
                "speech_placeholder_outside_shot",
                "h3_draft",
                "free-form internal timeline syntax is forbidden",
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
        for index, part in enumerate(shot.timeline_parts):
            if (
                isinstance(part, MimoH3SpeechPart)
                and index > 0
                and isinstance(shot.timeline_parts[index - 1], MimoH3ProsePart)
                and _SPEECH_LEAD_IN.search(shot.timeline_parts[index - 1].text)
            ):
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
            for part in shot.timeline_parts:
                if not isinstance(part, MimoH3SpeechPart):
                    continue
                interval = segment_intervals.get(part.segment_id)
                if interval is not None and not (
                    interval[0] < shot_end and interval[1] > shot_start
                ):
                    issues.append(
                        ValidationIssue(
                            "speech_placeholder_wrong_shot",
                            "h3_draft.shots",
                            f"{part.segment_id} does not overlap shot {shot.shot_index}",
                        )
                    )
            for part in shot.timeline_parts:
                if not isinstance(part, MimoH3AudioEventPart):
                    continue
                interval = event_by_id.get(part.event_id)
                if interval is not None and not (
                    interval[0] < shot_end and interval[1] > shot_start
                ):
                    issues.append(
                        ValidationIssue(
                            "audio_event_placeholder_wrong_shot",
                            "h3_draft.shots",
                            f"{part.event_id} does not overlap shot {shot.shot_index}",
                        )
                    )
    draft_text = "\n".join((*prose_parts, non_shot_text))
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
    model_owned_text = _normalized_text(
        "\n".join(
            (
                *draft.subject_definitions,
                draft.summary,
                *draft.visual_retention_analysis,
                *prose_parts,
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
        *(
            item.delivery_style
            for item in decisions
            if item.delivery_style is not None
        ),
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
        *(
            item.voice_characteristics
            for item in annotation.speaker_voice_profiles
            if item.voice_characteristics is not None
        ),
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
            audio_url = self.config.media_resolver.resolve(
                Path(job.target_full_audio_path)
            )
            if self.config.transport == "sglang":
                content.append(
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                    }
                )
            else:
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_url},
                    }
                )
        return content

    @staticmethod
    def build_compact_task_contract(job: MimoBackendJob) -> dict[str, object]:
        payload = job.model_dump(mode="json")
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            raise TypeError("MiMo segment inventory is invalid")
        return {
            "clip_uid": job.clip_uid,
            "r2v_instruction": job.r2v_instruction,
            "target_duration_seconds": job.target_duration_seconds,
            "subject_definition_requirements": [
                {
                    "subject_label": subject.subject_label,
                    "required_source_picture_labels": list(
                        subject.source_picture_labels
                    ),
                }
                for subject in job.reference_subjects
            ],
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
        }

    @staticmethod
    def build_mandatory_h3_draft_contract(
        job: MimoBackendJob,
    ) -> dict[str, object]:
        contract = OpenAIMimo25Backend.build_compact_task_contract(job)
        return {
            key: contract[key]
            for key in (
                "subject_definition_requirements",
                "allowed_segment_ids",
                "transcribed_segment_ids",
                "allowed_speaker_bindable_entity_ids",
            )
        }

    @classmethod
    def _mandatory_h3_draft_contract_text(cls, job: MimoBackendJob) -> str:
        return (
            "MANDATORY MACHINE CONTRACT:\n"
            + _compact_json(cls.build_mandatory_h3_draft_contract(job))
            + "\nUse allowed_segment_ids for all decisions and transcribed_segment_ids "
            "for typed speech parts. Preserve exact Subject-to-Picture ownership."
        )

    def _prompt(self, job: MimoBackendJob) -> str:
        schema = (
            "Return one JSON object matching this schema, with no markdown or extra fields:\n"
            + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            + "\n"
            if self.config.transport == "xiaomi"
            else "Return one JSON object constrained by the supplied response_format.\n"
        )
        return (
            schema
            + "R2V INSTRUCTION:\n"
            + job.r2v_instruction
            + "\nThis intent hint never overrides observed target evidence.\n"
            + self._mandatory_h3_draft_contract_text(job)
            + "\nAUTHORITATIVE INPUT:\n"
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
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "stream": False,
        }
        if self.config.transport == "sglang":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "MimoAVAnnotationDraft",
                    "schema": MimoAVAnnotationDraft.model_json_schema(),
                    "strict": True,
                },
            }
            payload.update(
                reasoning_effort="none",
                extra_body={
                    "use_audio_in_video": True,
                    "chat_template_kwargs": {
                        "thinking": False,
                        "enable_thinking": False,
                    },
                },
            )
        else:
            payload["response_format"] = {"type": "json_object"}
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
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
        issue_codes = {item.code for item in issues}
        issue_actions: list[str] = []
        if "subject_definition_contract_mismatch" in issue_codes:
            issue_actions.append(
                "For subject_definition_contract_mismatch, regenerate each affected "
                "definition from the supplied exact Subject-to-Pictures mapping. Do not "
                "guess, omit, add, duplicate, or reassign Picture ownership."
            )
        if "speech_placeholder_inventory_mismatch" in issue_codes:
            issue_actions.append(
                "For speech_placeholder_inventory_mismatch, rebuild all typed speech "
                "timeline parts so their flattened segment_id sequence exactly equals "
                "transcribed_segment_ids. Do not derive eligibility again or emit any "
                "other segment ID."
            )
        if issue_codes & {
            "visible_entity_speaker_group_contradiction",
            "speaker_group_entity_contradiction",
        }:
            issue_actions.append(
                "For a visible entity/speaker-group contradiction, reconsider clip-local "
                "speaker identity from the same audiovisual evidence and regenerate "
                "group assignments consistently. A group represents identity, not a "
                "turn. Do not blindly merge groups unless the audiovisual evidence "
                "supports the same speaker."
            )
        if issue_codes & {
            "visible_entity_requires_confirmed_onscreen_speech",
            "onscreen_speech_requires_reliable_visible_speaker_evidence",
        }:
            issue_actions.append(
                "For unreliable onscreen-speaker evidence, reinspect every affected "
                "segment in the actual audiovisual media. Keep onscreen_spoken and "
                "visible_entity only if (A) synchronized mouth, lip, or jaw motion is "
                "actually visible, then include visible_lip_motion; or (B) the visible "
                "speaker's mouth is genuinely occluded or the speaker is back-facing, "
                "then include speaker_visible_mouth_occluded together with "
                "av_temporal_alignment and/or voice_continuity. If neither A nor B is "
                "supported, (C) do not claim visible_entity: set entity_id=null and "
                "choose the actually supported offscreen with offscreen_spoken, "
                "no_reliable_entity, uncertain, voice_over, message_voice_over, or "
                "device_playback semantics. av_temporal_alignment and voice_continuity "
                "may preserve primary_speaker_group identity continuity, but they do "
                "not establish which visible entity is speaking. Never invent lip "
                "motion or mouth occlusion, never preserve visible_entity merely to "
                "satisfy validation, never bind a person merely because the same voice "
                "continues while that person is visible, and never preserve "
                "visible_entity merely because source_cluster_support or the current "
                "binding proposes it."
            )
        actions = (
            "\nISSUE-SPECIFIC CONTRACT ACTIONS:\n" + "\n".join(issue_actions)
            if issue_actions
            else ""
        )
        schema = (
            "\nSCHEMA: " + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            if self.config.transport == "xiaomi"
            else ""
        )
        return (
            "Reinspect the same full audiovisual evidence and correct the listed issues. "
            "Preserve authoritative DiariZen timing, Qwen3-ASR text/language, and frozen "
            "references. Return one compact JSON object.\n"
            + self._mandatory_h3_draft_contract_text(job)
            + actions
            + "\nAUTHORITATIVE INPUT: "
            + _compact_json(self.build_compact_task_contract(job))
            + schema
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
    "MimoH3AudioEventPart",
    "MimoH3Draft",
    "MimoH3ProsePart",
    "MimoH3Shot",
    "MimoH3SpeechPart",
    "MimoH3TimelinePart",
    "MimoMediaResolver",
    "MimoSegmentDecision",
    "MimoSpeakerVoiceProfile",
    "MimoTransport",
    "OpenAIMimo25Backend",
    "SpeechPresentation",
    "sha256_file",
    "validate_annotation",
]
