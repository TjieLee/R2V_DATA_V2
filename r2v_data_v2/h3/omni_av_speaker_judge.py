from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import (
    CanonicalAudioClip,
    jea_production_paths,
)
from r2v_data_v2.h3.jea_target_audio_caption import (
    AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS,
)
from r2v_data_v2.h3.pilot_schemas import LRASDNativeArtifact
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

OMNI_AV_SPEAKER_MANIFEST_VERSION = "r2v.h3.omni_av_speaker_judge_manifest.1"
OMNI_AV_SPEAKER_RECORD_VERSION = "r2v.h3.omni_av_speaker_judge_record.3"
OMNI_AV_SPEAKER_RAW_VERSION = "r2v.h3.omni_av_speaker_judge_raw.1"
OMNI_AV_SPEAKER_SUMMARY_VERSION = "r2v.h3.omni_av_speaker_judge_summary.3"
OMNI_AV_SPEAKER_POLICY_VERSION = "h3_omni_av_speaker_judge_pilot_v4"
PASS1_PROMPT_VERSION = "h3_omni_av_speaker_blind_identification_v4"
PASS2_PROMPT_VERSION = "h3_omni_av_speaker_blind_verification_v4"
MEDIA_CONSTRUCTION_POLICY = "neutral_faces_with_target_interval_marker_v1"
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CONTEXT_SECONDS = 0.75

Decision = Literal[
    "visible_entity",
    "multiple_speakers",
    "offscreen",
    "other_visible",
    "uncertain",
]
SecondarySpeechStatus = Literal["none", "incidental", "competing"]
DraftStatus = Literal["candidate_mapped", "conflict", "unbound", "ambiguous"]
Comparison = Literal["agree", "disagree", "unresolved", "draft_unresolved"]


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


class OmniAVSpeakerObservation(SchemaModel):
    decision: Decision
    entity_id: str | None
    secondary_speech_status: SecondarySpeechStatus

    @model_validator(mode="after")
    def validate_entity(self) -> OmniAVSpeakerObservation:
        if self.decision == "visible_entity":
            if self.entity_id is None or re.fullmatch(
                r"e[1-9]\d*", self.entity_id
            ) is None:
                raise ValueError("visible_entity requires an eN entity_id")
        elif self.entity_id is not None:
            raise ValueError("non-visible decision requires null entity_id")
        if self.decision == "multiple_speakers" and (
            self.secondary_speech_status != "competing"
        ):
            raise ValueError("multiple_speakers requires competing secondary speech")
        if (
            self.decision == "visible_entity"
            and self.secondary_speech_status == "competing"
        ):
            raise ValueError("visible_entity cannot have competing secondary speech")
        return self


class OmniAVSpeakerHumanLabel(SchemaModel):
    decision: Decision
    entity_id: str | None = None
    secondary_speech_status: SecondarySpeechStatus | None = None

    @model_validator(mode="after")
    def validate_label(self) -> OmniAVSpeakerHumanLabel:
        if self.decision == "visible_entity":
            if self.entity_id is None or re.fullmatch(
                r"e[1-9]\d*", self.entity_id
            ) is None:
                raise ValueError("visible_entity requires an eN entity_id")
        elif self.entity_id is not None:
            raise ValueError("non-visible decision requires null entity_id")
        if self.secondary_speech_status is not None:
            OmniAVSpeakerObservation(
                decision=self.decision,
                entity_id=self.entity_id,
                secondary_speech_status=self.secondary_speech_status,
            )
        return self


class OmniAVSpeakerPilotCase(SchemaModel):
    schema_version: Literal["r2v.h3.omni_av_speaker_judge_manifest.1"] = (
        OMNI_AV_SPEAKER_MANIFEST_VERSION
    )
    clip_uid: str
    segment_id: str
    human_label: OmniAVSpeakerHumanLabel | None = None

    @model_validator(mode="after")
    def validate_ids(self) -> OmniAVSpeakerPilotCase:
        for value in (self.clip_uid, self.segment_id):
            if (
                not value.strip()
                or value in {".", ".."}
                or "/" in value
                or "\\" in value
            ):
                raise ValueError("speaker judge case IDs must be safe path components")
        return self


class OmniAVSpeakerBackendProvenance(SchemaModel):
    backend: Literal["vllm"] = "vllm"
    served_model_name: str
    checkpoint_id: str
    base_url: str
    media_mode: Literal["file", "http"]
    media_root: str
    media_base_url: str | None = None
    input_modality: Literal[
        "target_marked_neutral_video_plus_canonical_trimmed_audio"
    ] = (
        "target_marked_neutral_video_plus_canonical_trimmed_audio"
    )
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    temperature: Literal[0.0] = 0.0
    enable_thinking: Literal[False] = False
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    pass1_prompt_version: Literal[
        "h3_omni_av_speaker_blind_identification_v4"
    ] = PASS1_PROMPT_VERSION
    pass2_prompt_version: Literal[
        "h3_omni_av_speaker_blind_verification_v4"
    ] = PASS2_PROMPT_VERSION
    policy_version: Literal["h3_omni_av_speaker_judge_pilot_v4"] = (
        OMNI_AV_SPEAKER_POLICY_VERSION
    )
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class OmniAVSpeakerJudgeConfig:
    base_url: str
    media_resolver: MediaURLResolver
    api_key: str = "EMPTY"
    served_model_name: str = DEFAULT_MODEL
    checkpoint_id: str = DEFAULT_MODEL
    timeout_seconds: float = 600.0
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.base_url,
                self.api_key,
                self.served_model_name,
                self.checkpoint_id,
            )
        ):
            raise ValueError("Omni AV speaker backend configuration is incomplete")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("Omni AV speaker backend limits must be positive")

    def provenance(self) -> OmniAVSpeakerBackendProvenance:
        values = {
            "backend": "vllm",
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "base_url": self.base_url,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "input_modality": (
                "target_marked_neutral_video_plus_canonical_trimmed_audio"
            ),
            "output_modalities": ["text"],
            "temperature": 0.0,
            "enable_thinking": False,
            "max_tokens": self.max_tokens,
            "repair_retries": 1,
            "pass1_prompt_version": PASS1_PROMPT_VERSION,
            "pass2_prompt_version": PASS2_PROMPT_VERSION,
            "policy_version": OMNI_AV_SPEAKER_POLICY_VERSION,
        }
        return OmniAVSpeakerBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


class OmniAVCompletionDiagnostic(SchemaModel):
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class OmniAVSpeakerJudgeRequest:
    neutral_video_path: Path
    canonical_audio_path: Path
    target_start_in_window: float
    target_end_in_window: float
    visible_candidate_entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class OmniAVSpeakerJudgeResult:
    observation: OmniAVSpeakerObservation
    raw_responses: tuple[str, ...]
    completion_diagnostics: tuple[OmniAVCompletionDiagnostic, ...]
    model_call_count: int
    normalization_applied: bool = False
    normalization_kind: Literal[
        "entity_decision_alias_to_visible_entity"
    ] | None = None

    def __post_init__(self) -> None:
        if self.normalization_applied != (self.normalization_kind is not None):
            raise ValueError("Omni AV speaker normalization provenance is inconsistent")


class OmniAVSpeakerJudgeFailure(ValueError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: Sequence[str] = (),
        completion_diagnostics: Sequence[OmniAVCompletionDiagnostic] = (),
        issues: Sequence[ValidationIssue] = (),
        model_call_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = tuple(raw_responses)
        self.completion_diagnostics = tuple(completion_diagnostics)
        self.issues = tuple(issues)
        self.model_call_count = model_call_count


class OmniAVSpeakerJudgeBackend(Protocol):
    @property
    def provenance(self) -> OmniAVSpeakerBackendProvenance: ...

    def decide(
        self,
        request: OmniAVSpeakerJudgeRequest,
        *,
        verification: bool,
    ) -> OmniAVSpeakerJudgeResult: ...


PASS1_SYSTEM_PROMPT = """Judge speech-turn ownership during one marked target
interval using only synchronized audio-video speaking evidence.

The attached video uses neutral face labels. Labels e1, e2, and so on identify
visible mapped entities; OTHER identifies a visible face without a mapped entity.
The labels do not indicate speaking state.

Frames marked TARGET define the interval whose speech-turn ownership must be
judged. Frames before or after TARGET are context only. Audio remains
synchronized to the full context window. Use context to understand continuity,
but determine the primary speaker and secondary-speech status specifically for
the TARGET interval.

First determine the safest primary speaker attribution. Choose visible_entity
when one supplied entity clearly owns the primary speech turn, offscreen when
the primary speech belongs to no mapped visible subject, other_visible when the
primary speaker is a visible OTHER face, and uncertain when attribution is not
reliable. Use multiple_speakers only when speakers contribute materially and no
reliable single primary owns the turn. Multiple offscreen speakers remain
offscreen when no mapped visible subject owns the speech.

Separately report secondary_speech_status. Use none when there is no meaningful
linguistic speech from another speaker, incidental when another speaker is
audible but one primary still clearly owns the turn, and competing when speech
is material enough that no reliable single primary owns the turn. Breathing,
sighing, laughter, coughing, gasping, grunting, and other non-linguistic
vocalizations alone are none, not secondary speech.

Generic examples: a mapped subject owns the main utterance while another person
briefly interjects -> visible_entity + incidental; materially alternating or
overlapping speakers with no reliable primary -> multiple_speakers + competing;
several offscreen speakers while no mapped subject owns the speech -> offscreen
+ competing; another person only sighs or coughs -> none.

Do not choose a primary merely because someone is louder, longer, visually
central, first, or last. Conversely, do not reject a clear primary merely
because a second voice is briefly audible.

Do not infer identity from gender, age, clothing, dialogue meaning, character
semantics, or visual appearance. Never transcribe, quote, or paraphrase speech.
Return exactly one compact JSON object with decision, entity_id, and
secondary_speech_status only."""

PASS2_SYSTEM_PROMPT = """Independently verify speech-turn ownership during one
marked target interval using only the attached synchronized audio-video speaking
evidence. You have not been given any prior answer.

The video uses neutral labels: e1, e2, and so on are supplied visible entities;
OTHER is an unmatched visible face. Labels never indicate speaking state.

Frames marked TARGET define the interval whose speech-turn ownership must be
judged. Frames before or after TARGET are context only. Audio remains
synchronized to the full context window. Use context to understand continuity,
but determine the primary speaker and secondary-speech status specifically for
the TARGET interval.

First determine the safest primary attribution: visible_entity for one mapped
entity that clearly owns the turn, offscreen when no mapped visible subject owns
the primary speech, other_visible for a visible OTHER primary, and uncertain
when attribution is unreliable. Reserve multiple_speakers for materially
competing speech with no reliable single primary. Multiple offscreen speakers
remain offscreen when no mapped visible subject owns the speech.

Separately classify secondary_speech_status as none, incidental, or competing.
Incidental means another speaker is audible while one primary still clearly
owns the turn. Competing means no reliable primary owns the turn. Non-linguistic
breathing, sighing, laughter, coughing, gasping, or grunting alone count as none.

Generic examples: clear mapped primary plus a brief interjection ->
visible_entity + incidental; materially alternating or overlapping speech with
no reliable primary -> multiple_speakers + competing; several offscreen voices
with no mapped primary -> offscreen + competing; a second person only coughs ->
none. Do not choose by loudness, duration, visual centrality, firstness, or
lastness, and do not discard a clear primary merely because another voice is
briefly audible.

Do not infer identity from gender, age, clothing, dialogue meaning, character
semantics, or visual appearance. Never transcribe, quote, or paraphrase speech.
Return exactly one compact JSON object with decision, entity_id, and
secondary_speech_status only."""


def _user_prompt(request: OmniAVSpeakerJudgeRequest) -> str:
    candidates = ", ".join(request.visible_candidate_entity_ids) or "none"
    return (
        "Target speech interval within this synchronized context window: "
        f"{request.target_start_in_window:.6f} to "
        f"{request.target_end_in_window:.6f} seconds.\n"
        f"Visible mapped entity IDs in the window: {candidates}.\n"
        "Who, if anyone, safely owns the primary speech turn, and what is the "
        "secondary speech status?"
    )


def _repair_prompt(
    request: OmniAVSpeakerJudgeRequest,
    *,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        f"{_user_prompt(request)}\n"
        "Repair only the structured-output errors below. Reinspect the same "
        "synchronized media. Return one JSON object only.\n"
        f"Schema: {_compact_json(OmniAVSpeakerObservation.model_json_schema())}\n"
        f"Issues: {_compact_json([issue.to_dict() for issue in issues])}\n"
        f"Invalid response: {invalid_response}"
    )


def _decision_issues(
    observation: OmniAVSpeakerObservation,
    request: OmniAVSpeakerJudgeRequest,
) -> list[ValidationIssue]:
    if (
        observation.decision == "visible_entity"
        and observation.entity_id not in request.visible_candidate_entity_ids
    ):
        return [
            ValidationIssue(
                code="unknown_visible_entity",
                field="entity_id",
                message="entity_id must be one supplied visible entity ID",
            )
        ]
    return []


def _normalize_entity_decision_alias(
    raw: str,
    request: OmniAVSpeakerJudgeRequest,
) -> OmniAVSpeakerObservation | None:
    try:
        payload = json.loads(normalize_structured_json_envelope(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "decision",
        "entity_id",
        "secondary_speech_status",
    }:
        return None
    decision = payload["decision"]
    entity_id = payload["entity_id"]
    secondary_speech_status = payload["secondary_speech_status"]
    if (
        not isinstance(decision, str)
        or re.fullmatch(r"e[1-9]\d*", decision) is None
        or entity_id != decision
        or decision not in request.visible_candidate_entity_ids
    ):
        return None
    try:
        return OmniAVSpeakerObservation(
            decision="visible_entity",
            entity_id=decision,
            secondary_speech_status=secondary_speech_status,
        )
    except ValueError:
        return None


def _optional_value(value: object, field: str) -> object | None:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def _diagnostic(completion: object, choice: object) -> OmniAVCompletionDiagnostic:
    usage = getattr(completion, "usage", None)

    def token(field: str) -> int | None:
        value = _optional_value(usage, field)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    finish_reason = _optional_value(choice, "finish_reason")
    return OmniAVCompletionDiagnostic(
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        prompt_tokens=token("prompt_tokens"),
        completion_tokens=token("completion_tokens"),
        total_tokens=token("total_tokens"),
    )


class OpenAIOmniAVSpeakerJudge:
    def __init__(
        self,
        config: OmniAVSpeakerJudgeConfig,
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
    def provenance(self) -> OmniAVSpeakerBackendProvenance:
        return self.config.provenance()

    def _request(
        self,
        request: OmniAVSpeakerJudgeRequest,
        *,
        verification: bool,
        prompt: str,
    ) -> tuple[str, OmniAVCompletionDiagnostic]:
        payload = {
            "model": self.config.served_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        PASS2_SYSTEM_PROMPT if verification else PASS1_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": self.config.media_resolver.resolve(
                                    request.neutral_video_path
                                )
                            },
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": self.config.media_resolver.resolve(
                                    request.canonical_audio_path
                                )
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "modalities": ["text"],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        completion = self.client.chat.completions.create(**payload)
        choices = getattr(completion, "choices", None)
        if not choices:
            raise TypeError("Omni AV speaker response has no choices")
        choice = choices[0]
        content = getattr(choice.message, "content", None)
        if not isinstance(content, str):
            raise TypeError("Omni AV speaker response content must be text")
        return content, _diagnostic(completion, choice)

    def decide(
        self,
        request: OmniAVSpeakerJudgeRequest,
        *,
        verification: bool,
    ) -> OmniAVSpeakerJudgeResult:
        raw_responses: list[str] = []
        diagnostics: list[OmniAVCompletionDiagnostic] = []
        issues: list[ValidationIssue] = []
        for attempt in range(2):
            prompt = (
                _user_prompt(request)
                if attempt == 0
                else _repair_prompt(
                    request,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            )
            try:
                raw, diagnostic = self._request(
                    request,
                    verification=verification,
                    prompt=prompt,
                )
            except Exception as exc:
                raise OmniAVSpeakerJudgeFailure(
                    code="omni_av_request_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    raw_responses=raw_responses,
                    completion_diagnostics=diagnostics,
                    issues=issues,
                    model_call_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            diagnostics.append(diagnostic)
            normalized = _normalize_entity_decision_alias(raw, request)
            if normalized is not None:
                return OmniAVSpeakerJudgeResult(
                    observation=normalized,
                    raw_responses=tuple(raw_responses),
                    completion_diagnostics=tuple(diagnostics),
                    model_call_count=attempt + 1,
                    normalization_applied=True,
                    normalization_kind=(
                        "entity_decision_alias_to_visible_entity"
                    ),
                )
            observation, issues = parse_structured_json_issues(
                raw,
                OmniAVSpeakerObservation,
            )
            if observation is not None:
                issues = _decision_issues(observation, request)
            if observation is not None and not issues:
                return OmniAVSpeakerJudgeResult(
                    observation=observation,
                    raw_responses=tuple(raw_responses),
                    completion_diagnostics=tuple(diagnostics),
                    model_call_count=attempt + 1,
                )
        raise OmniAVSpeakerJudgeFailure(
            code="omni_av_structured_output_failed",
            reason="Omni AV speaker output failed after one repair",
            raw_responses=raw_responses,
            completion_diagnostics=diagnostics,
            issues=issues,
            model_call_count=2,
        )


class NeutralFaceSample(SchemaModel):
    face_track_id: str
    label: str
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]


class NeutralFaceTimeline(SchemaModel):
    schema_version: Literal["r2v.h3.omni_av_neutral_face_timeline.1"] = (
        "r2v.h3.omni_av_neutral_face_timeline.1"
    )
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    window_start: float = Field(ge=0)
    window_end: float = Field(gt=0)
    samples: list[NeutralFaceSample]


def build_neutral_face_timeline(
    *,
    native: LRASDNativeArtifact,
    sidecar: AudioBindingSidecar,
    window_start: float,
    window_end: float,
) -> NeutralFaceTimeline:
    if sidecar.evidence is None:
        raise ValueError("neutral face timeline requires Audio binding evidence")
    association_by_track = {
        item.face_track_id: item for item in sidecar.evidence.associations
    }
    samples: list[NeutralFaceSample] = []
    for track in native.tracks:
        association = association_by_track.get(track.face_track_id)
        label = (
            association.entity_id
            if association is not None and association.status == "matched"
            else "OTHER"
        )
        assert label is not None
        for sample in track.samples:
            if window_start <= sample.timestamp_seconds < window_end:
                samples.append(
                    NeutralFaceSample(
                        face_track_id=track.face_track_id,
                        label=label,
                        frame_index=sample.frame_index,
                        timestamp_seconds=sample.timestamp_seconds,
                        bbox_xyxy=sample.bbox_xyxy,
                    )
                )
    samples.sort(key=lambda item: (item.timestamp_seconds, item.face_track_id))
    return NeutralFaceTimeline(
        source_width=native.width,
        source_height=native.height,
        window_start=window_start,
        window_end=window_end,
        samples=samples,
    )


class OmniAVSpeakerMediaBackend(Protocol):
    def render_neutral_video(
        self,
        *,
        source_video_path: Path,
        timeline: NeutralFaceTimeline,
        window_start: float,
        window_end: float,
        target_start: float,
        target_end: float,
        destination_path: Path,
    ) -> None: ...

    def trim_canonical_audio(
        self,
        *,
        source_audio_path: Path,
        window_start: float,
        window_end: float,
        destination_path: Path,
    ) -> None: ...


class SubprocessOmniAVSpeakerMediaBackend:
    def __init__(
        self,
        *,
        python_path: Path,
        ffmpeg: str = "ffmpeg",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.python_path = python_path.expanduser().absolute()
        if not self.python_path.is_file() or not ffmpeg.strip():
            raise ValueError("Omni AV speaker media runtime is unavailable")
        if timeout_seconds <= 0:
            raise ValueError("Omni AV speaker media timeout must be positive")
        self.ffmpeg = ffmpeg
        self.timeout_seconds = timeout_seconds

    def _run(self, command: list[str], destination_path: Path) -> None:
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Omni AV speaker media command failed: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0 or not destination_path.is_file():
            raise RuntimeError(
                "Omni AV speaker media command failed: "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

    def render_neutral_video(
        self,
        *,
        source_video_path: Path,
        timeline: NeutralFaceTimeline,
        window_start: float,
        window_end: float,
        target_start: float,
        target_end: float,
        destination_path: Path,
    ) -> None:
        helper = Path(__file__).resolve().parents[2] / "tools" / (
            "render_h3_omni_av_speaker_media.py"
        )
        timeline_path = destination_path.with_suffix(".timeline.json")
        _write_json(timeline_path, timeline)
        try:
            self._run(
                [
                    str(self.python_path),
                    str(helper),
                    "--source",
                    str(source_video_path),
                    "--timeline",
                    str(timeline_path),
                    "--window-start",
                    f"{window_start:.9f}",
                    "--window-end",
                    f"{window_end:.9f}",
                    "--target-start",
                    f"{target_start:.9f}",
                    "--target-end",
                    f"{target_end:.9f}",
                    "--output",
                    str(destination_path),
                    "--ffmpeg",
                    self.ffmpeg,
                ],
                destination_path,
            )
        finally:
            timeline_path.unlink(missing_ok=True)

    def trim_canonical_audio(
        self,
        *,
        source_audio_path: Path,
        window_start: float,
        window_end: float,
        destination_path: Path,
    ) -> None:
        self._run(
            [
                self.ffmpeg,
                "-y",
                "-ss",
                f"{window_start:.9f}",
                "-i",
                str(source_audio_path),
                "-t",
                f"{window_end - window_start:.9f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination_path),
                "-loglevel",
                "error",
            ],
            destination_path,
        )


class OmniAVSpeakerSourceProvenance(SchemaModel):
    raw_segments_path: str
    raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_segment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audio_binding_path: str
    audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lr_asd_native_path: str
    lr_asd_native_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_audio_path: str
    canonical_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lr_asd_model_video_path: str
    lr_asd_model_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class OmniAVSpeakerMediaProvenance(SchemaModel):
    neutral_video_path: str
    neutral_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trimmed_audio_path: str
    trimmed_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_are_neutral: Literal[True] = True
    lr_asd_speaking_state_exposed: Literal[False] = False
    media_construction_policy: Literal[
        "neutral_faces_with_target_interval_marker_v1"
    ] = MEDIA_CONSTRUCTION_POLICY


class OmniAVSpeakerCaseFailure(SchemaModel):
    code: str
    reason: str
    issues: list[dict[str, str | None]] = Field(default_factory=list)


class OmniAVSpeakerPilotRecord(SchemaModel):
    schema_version: Literal["r2v.h3.omni_av_speaker_judge_record.3"] = (
        OMNI_AV_SPEAKER_RECORD_VERSION
    )
    status: Literal["succeeded", "failed"]
    clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    absolute_segment_start: float = Field(ge=0)
    absolute_segment_end: float = Field(gt=0)
    target_boundary_clipped: bool = False
    target_boundary_clip_seconds: float = Field(default=0, ge=0)
    effective_absolute_segment_end: float | None = Field(default=None, gt=0)
    window_start: float | None = Field(default=None, ge=0)
    window_end: float | None = Field(default=None, gt=0)
    target_start_in_window: float | None = Field(default=None, ge=0)
    target_end_in_window: float | None = Field(default=None, gt=0)
    visible_candidate_entity_ids: list[str]
    draft_binding_status: DraftStatus
    draft_entity_id: str | None = None
    pass1_decision: Decision | None = None
    pass1_entity_id: str | None = None
    pass1_secondary_speech_status: SecondarySpeechStatus | None = None
    pass2_called: bool
    pass2_decision: Decision | None = None
    pass2_entity_id: str | None = None
    pass2_secondary_speech_status: SecondarySpeechStatus | None = None
    primary_observation_stable: bool
    secondary_speech_stable: bool
    confirmed_secondary_speech_status: SecondarySpeechStatus | None = None
    comparison: Comparison | None = None
    proposed_entity_id: str | None = None
    proposed_non_entity_class: Literal["offscreen", "other_visible"] | None = None
    multiple_speakers_confirmed: bool = False
    subject_entity_binding_excluded: bool = False
    identity_specific_voice_products_excluded: bool = False
    source_provenance: OmniAVSpeakerSourceProvenance
    media_provenance: OmniAVSpeakerMediaProvenance | None = None
    backend_provenance: OmniAVSpeakerBackendProvenance
    model_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    failure: OmniAVSpeakerCaseFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> OmniAVSpeakerPilotRecord:
        mapped = self.draft_binding_status == "candidate_mapped"
        if mapped != (self.draft_entity_id is not None):
            raise ValueError("speaker judge draft mapping is inconsistent")
        if (self.pass1_decision is None) != (
            self.pass1_secondary_speech_status is None
        ):
            raise ValueError("speaker judge pass-1 state is inconsistent")
        if (self.pass2_decision is None) != (
            self.pass2_secondary_speech_status is None
        ):
            raise ValueError("speaker judge pass-2 state is inconsistent")
        if self.pass2_decision is not None and not self.pass2_called:
            raise ValueError("speaker judge pass-2 state is inconsistent")
        pass1 = (
            None
            if self.pass1_decision is None
            else OmniAVSpeakerObservation(
                decision=self.pass1_decision,
                entity_id=self.pass1_entity_id,
                secondary_speech_status=self.pass1_secondary_speech_status,
            )
        )
        pass2 = (
            None
            if self.pass2_decision is None
            else OmniAVSpeakerObservation(
                decision=self.pass2_decision,
                entity_id=self.pass2_entity_id,
                secondary_speech_status=self.pass2_secondary_speech_status,
            )
        )
        if self.status == "succeeded":
            if (
                pass1 is None
                or self.media_provenance is None
                or self.failure is not None
                or self.comparison is None
                or self.effective_absolute_segment_end is None
                or self.window_start is None
                or self.window_end is None
                or self.target_start_in_window is None
                or self.target_end_in_window is None
            ):
                raise ValueError("successful speaker judge record is incomplete")
            if self.pass2_called != (pass2 is not None):
                raise ValueError("successful speaker judge pass-2 state is incomplete")
            resolution = _resolve_observations(
                draft_status=self.draft_binding_status,
                draft_entity_id=self.draft_entity_id,
                pass1=pass1,
                pass2=pass2,
            )
            published = (
                self.primary_observation_stable,
                self.secondary_speech_stable,
                self.confirmed_secondary_speech_status,
                self.comparison,
                self.proposed_entity_id,
                self.proposed_non_entity_class,
                self.multiple_speakers_confirmed,
                self.subject_entity_binding_excluded,
                self.identity_specific_voice_products_excluded,
            )
            expected = (
                resolution.primary_observation_stable,
                resolution.secondary_speech_stable,
                resolution.confirmed_secondary_speech_status,
                resolution.comparison,
                resolution.proposed_entity_id,
                resolution.proposed_non_entity_class,
                resolution.multiple_speakers_confirmed,
                resolution.subject_entity_binding_excluded,
                resolution.identity_specific_voice_products_excluded,
            )
            if published != expected:
                raise ValueError("speaker judge derived observation state is inconsistent")
        else:
            if self.failure is None:
                raise ValueError("failed speaker judge record requires failure")
            if any(
                (
                    self.primary_observation_stable,
                    self.secondary_speech_stable,
                    self.confirmed_secondary_speech_status is not None,
                    self.comparison is not None,
                    self.proposed_entity_id is not None,
                    self.proposed_non_entity_class is not None,
                    self.multiple_speakers_confirmed,
                    self.subject_entity_binding_excluded,
                    self.identity_specific_voice_products_excluded,
                )
            ):
                raise ValueError("failed speaker judge record cannot publish conclusions")
        if self.target_boundary_clipped != (self.target_boundary_clip_seconds > 0):
            raise ValueError("speaker judge target boundary provenance is inconsistent")
        if (
            self.effective_absolute_segment_end is not None
            and self.effective_absolute_segment_end > self.absolute_segment_end
        ):
            raise ValueError("effective speaker segment end exceeds source segment")
        if self.subject_entity_binding_excluded and (
            self.proposed_entity_id is not None
        ):
            raise ValueError("excluded subject binding cannot propose an entity")
        return self


class OmniAVSpeakerPilotSummary(SchemaModel):
    schema_version: Literal["r2v.h3.omni_av_speaker_judge_summary.3"] = (
        OMNI_AV_SPEAKER_SUMMARY_VERSION
    )
    source_audio_production_root: str
    output_root: str
    case_count: int = Field(ge=0)
    succeeded_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    pass2_case_count: int = Field(ge=0)
    primary_observation_stable_count: int = Field(ge=0)
    secondary_speech_stable_count: int = Field(ge=0)
    comparison_counts: dict[str, int]
    model_call_count: int = Field(ge=0)
    bindings_modified_count: Literal[0] = 0
    diarization_bindings_modified_count: Literal[0] = 0
    backend_provenance: OmniAVSpeakerBackendProvenance

    @model_validator(mode="after")
    def validate_counts(self) -> OmniAVSpeakerPilotSummary:
        if self.case_count != self.succeeded_case_count + self.failed_case_count:
            raise ValueError("speaker judge case counts do not reconcile")
        if sum(self.comparison_counts.values()) != self.succeeded_case_count:
            raise ValueError("speaker judge comparison counts do not reconcile")
        if max(
            self.primary_observation_stable_count,
            self.secondary_speech_stable_count,
        ) > self.succeeded_case_count:
            raise ValueError("speaker judge stability counts exceed successes")
        return self


@dataclass(frozen=True)
class _SourceCase:
    manifest: OmniAVSpeakerPilotCase
    raw: RawDiarizationSegment
    bound: BoundDiarizationSegment
    cluster: DiarizationClusterBinding
    canonical: CanonicalAudioClip
    sidecar: AudioBindingSidecar
    sidecar_path: Path
    native: LRASDNativeArtifact
    native_path: Path


@dataclass(frozen=True)
class _SynchronizedTargetWindow:
    effective_absolute_segment_end: float
    window_start: float
    window_end: float
    target_start_in_window: float
    target_end_in_window: float
    boundary_clip_seconds: float


def _synchronized_target_window(source: _SourceCase) -> _SynchronizedTargetWindow:
    common_end = min(
        source.canonical.target_duration_seconds,
        source.native.duration_seconds,
    )
    raw = source.raw
    if raw.start_time >= common_end:
        raise OmniAVSpeakerJudgeFailure(
            code="omni_av_synchronized_media_boundary_invalid",
            reason=(
                "selected speaker segment starts at or after synchronized media end: "
                f"start={raw.start_time:.9f}, common_end={common_end:.9f}"
            ),
        )
    if raw.end_time > common_end + AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS:
        raise OmniAVSpeakerJudgeFailure(
            code="omni_av_synchronized_media_boundary_invalid",
            reason=(
                "selected speaker segment exceeds synchronized media tolerance: "
                f"end={raw.end_time:.9f}, common_end={common_end:.9f}, "
                "tolerance="
                f"{AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS:.9f}"
            ),
        )
    effective_end = min(raw.end_time, common_end)
    window_start = max(0.0, raw.start_time - CONTEXT_SECONDS)
    window_end = min(common_end, effective_end + CONTEXT_SECONDS)
    return _SynchronizedTargetWindow(
        effective_absolute_segment_end=effective_end,
        window_start=window_start,
        window_end=window_end,
        target_start_in_window=raw.start_time - window_start,
        target_end_in_window=effective_end - window_start,
        boundary_clip_seconds=max(0.0, raw.end_time - effective_end),
    )


def _load_manifest(path: Path) -> list[OmniAVSpeakerPilotCase]:
    rows = [OmniAVSpeakerPilotCase.model_validate(row) for row in _read_jsonl(path)]
    if not rows:
        raise ValueError("Omni AV speaker pilot manifest must not be empty")
    keys = [(row.clip_uid, row.segment_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Omni AV speaker pilot manifest contains duplicate cases")
    return rows


def _model_rows(path: Path, model: type[SchemaModel]) -> list[SchemaModel]:
    return [model.model_validate(row) for row in _read_jsonl(path)]


def _verify_path(path_value: str, expected_sha256: str, field: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field} path must be absolute")
    path = path.resolve(strict=True)
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"{field} changed or is unavailable")
    return path


def _load_source_cases(
    *,
    production_root: Path,
    cases: Sequence[OmniAVSpeakerPilotCase],
) -> tuple[list[_SourceCase], Path]:
    paths = jea_production_paths(production_root)
    raw_path = paths.diarization / "raw_segments.jsonl"
    bound_path = paths.diarization / "bound_segments.jsonl"
    cluster_path = paths.diarization / "cluster_bindings.jsonl"
    canonical_path = paths.audio / "canonical_clips.jsonl"
    for path in (raw_path, bound_path, cluster_path, canonical_path):
        if not path.is_file():
            raise ValueError(f"Omni AV speaker source artifact is missing: {path}")
    raw_rows = _model_rows(raw_path, RawDiarizationSegment)
    bound_rows = _model_rows(bound_path, BoundDiarizationSegment)
    cluster_rows = _model_rows(cluster_path, DiarizationClusterBinding)
    canonical_rows = _model_rows(canonical_path, CanonicalAudioClip)
    raw_by_key = {(row.target_clip_uid, row.segment_id): row for row in raw_rows}
    bound_by_key = {(row.target_clip_uid, row.segment_id): row for row in bound_rows}
    cluster_by_key = {
        (row.target_clip_uid, row.speaker_cluster_id): row for row in cluster_rows
    }
    canonical_by_clip = {row.clip_uid: row for row in canonical_rows}
    if len(raw_by_key) != len(raw_rows) or len(bound_by_key) != len(bound_rows):
        raise ValueError("Omni AV speaker source segment identities are duplicated")
    if len(cluster_by_key) != len(cluster_rows) or len(canonical_by_clip) != len(
        canonical_rows
    ):
        raise ValueError("Omni AV speaker source clip identities are duplicated")
    output: list[_SourceCase] = []
    for case in cases:
        key = (case.clip_uid, case.segment_id)
        if key not in raw_by_key or key not in bound_by_key:
            raise ValueError(f"selected Omni AV speaker segment is unavailable: {key}")
        raw = raw_by_key[key]
        bound = bound_by_key[key]
        if (
            raw.speaker_cluster_id != bound.speaker_cluster_id
            or raw.source_start_sample != bound.source_start_sample
            or raw.source_end_sample != bound.source_end_sample
        ):
            raise ValueError("Omni AV speaker raw/bound segment evidence differs")
        cluster = cluster_by_key[(case.clip_uid, raw.speaker_cluster_id)]
        if (
            cluster.status != bound.cluster_binding_status
            or cluster.entity_id != bound.entity_id
        ):
            raise ValueError("Omni AV speaker cluster/bound draft evidence differs")
        canonical = canonical_by_clip[case.clip_uid]
        if canonical.target_audio_binding_path is None:
            raise ValueError("Omni AV speaker case has no Audio binding sidecar")
        sidecar_path = Path(canonical.target_audio_binding_path).resolve(strict=True)
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.clip_uid != case.clip_uid or sidecar.status != "ready":
            raise ValueError("Omni AV speaker Audio sidecar is incompatible")
        if canonical.target_audio_binding_sha256 != _sha256_file(sidecar_path):
            raise ValueError("Omni AV speaker Audio sidecar hash differs")
        native_path = paths.audio / "runtime" / case.clip_uid / "lr_asd" / (
            "lr_asd_native.json"
        )
        native = LRASDNativeArtifact.model_validate_json(
            native_path.read_text(encoding="utf-8")
        )
        if native.clip_uid != case.clip_uid:
            raise ValueError("Omni AV speaker LR-ASD clip identity differs")
        output.append(
            _SourceCase(
                manifest=case,
                raw=raw,
                bound=bound,
                cluster=cluster,
                canonical=canonical,
                sidecar=sidecar,
                sidecar_path=sidecar_path,
                native=native,
                native_path=native_path,
            )
        )
    return output, raw_path


def _visible_entities(timeline: NeutralFaceTimeline) -> list[str]:
    return sorted({item.label for item in timeline.samples if item.label != "OTHER"})


def _needs_verification(
    *,
    draft_status: DraftStatus,
    draft_entity_id: str | None,
    pass1: OmniAVSpeakerObservation,
) -> bool:
    if pass1.secondary_speech_status in {"incidental", "competing"}:
        return True
    if draft_status == "candidate_mapped":
        return not (
            pass1.decision == "visible_entity"
            and pass1.entity_id == draft_entity_id
        )
    return pass1.decision in {"visible_entity", "offscreen"}


@dataclass(frozen=True)
class _ObservationResolution:
    primary_observation_stable: bool
    secondary_speech_stable: bool
    confirmed_secondary_speech_status: SecondarySpeechStatus | None
    comparison: Comparison
    proposed_entity_id: str | None
    proposed_non_entity_class: Literal["offscreen", "other_visible"] | None
    multiple_speakers_confirmed: bool
    subject_entity_binding_excluded: bool
    identity_specific_voice_products_excluded: bool


def _resolve_observations(
    *,
    draft_status: DraftStatus,
    draft_entity_id: str | None,
    pass1: OmniAVSpeakerObservation,
    pass2: OmniAVSpeakerObservation | None,
) -> _ObservationResolution:
    pass1_primary = (pass1.decision, pass1.entity_id)
    pass2_primary = (
        None if pass2 is None else (pass2.decision, pass2.entity_id)
    )
    primary_stable = pass2_primary == pass1_primary if pass2 is not None else (
        draft_status == "candidate_mapped"
        and pass1.decision == "visible_entity"
        and pass1.entity_id == draft_entity_id
    )
    secondary_stable = (
        pass2 is not None
        and pass2.secondary_speech_status == pass1.secondary_speech_status
    )
    confirmed_secondary = (
        pass1.secondary_speech_status if secondary_stable else None
    )
    multiple_speakers_confirmed = (
        primary_stable
        and pass2 is not None
        and pass1.decision == "multiple_speakers"
        and pass2.decision == "multiple_speakers"
    )
    if draft_status != "candidate_mapped":
        comparison: Comparison = "draft_unresolved"
    elif not primary_stable or multiple_speakers_confirmed:
        comparison = "unresolved"
    elif pass1.decision == "visible_entity" and pass1.entity_id == draft_entity_id:
        comparison = "agree"
    else:
        comparison = "disagree"
    proposed_entity = (
        pass1.entity_id
        if primary_stable and pass1.decision == "visible_entity"
        else None
    )
    proposed_non_entity = (
        pass1.decision
        if primary_stable and pass1.decision in {"offscreen", "other_visible"}
        else None
    )
    return _ObservationResolution(
        primary_observation_stable=primary_stable,
        secondary_speech_stable=secondary_stable,
        confirmed_secondary_speech_status=confirmed_secondary,
        comparison=comparison,
        proposed_entity_id=proposed_entity,
        proposed_non_entity_class=proposed_non_entity,
        multiple_speakers_confirmed=multiple_speakers_confirmed,
        subject_entity_binding_excluded=multiple_speakers_confirmed,
        identity_specific_voice_products_excluded=(
            confirmed_secondary in {"incidental", "competing"}
        ),
    )


def _source_provenance(
    *,
    source: _SourceCase,
    raw_segments_path: Path,
) -> tuple[OmniAVSpeakerSourceProvenance, Path, Path, Path]:
    canonical_audio = _verify_path(
        source.canonical.target_full_audio_path,
        source.canonical.target_full_audio_sha256,
        "canonical audio",
    )
    target_video = _verify_path(
        source.canonical.target_video_path,
        source.canonical.target_video_sha256,
        "target video",
    )
    model_video = Path(source.native.model_video_path).resolve(strict=True)
    if not model_video.is_file():
        raise ValueError("LR-ASD model video is unavailable")
    return (
        OmniAVSpeakerSourceProvenance(
            raw_segments_path=str(raw_segments_path),
            raw_segments_sha256=_sha256_file(raw_segments_path),
            source_segment_sha256=_sha256_text(
                _compact_json(source.raw.model_dump(mode="json"))
            ),
            audio_binding_path=str(source.sidecar_path),
            audio_binding_sha256=_sha256_file(source.sidecar_path),
            lr_asd_native_path=str(source.native_path),
            lr_asd_native_sha256=_sha256_file(source.native_path),
            canonical_audio_path=str(canonical_audio),
            canonical_audio_sha256=_sha256_file(canonical_audio),
            target_video_path=str(target_video),
            target_video_sha256=_sha256_file(target_video),
            lr_asd_model_video_path=str(model_video),
            lr_asd_model_video_sha256=_sha256_file(model_video),
        ),
        canonical_audio,
        target_video,
        model_video,
    )


def _raw_payload(
    *,
    clip_uid: str,
    segment_id: str,
    pass1: OmniAVSpeakerJudgeResult | None,
    pass2: OmniAVSpeakerJudgeResult | None,
    failure: OmniAVSpeakerJudgeFailure | None,
) -> dict[str, object]:
    def result(value: OmniAVSpeakerJudgeResult | None) -> object:
        if value is None:
            return None
        return {
            "raw_responses": list(value.raw_responses),
            "completion_diagnostics": [
                item.model_dump(mode="json") for item in value.completion_diagnostics
            ],
            "model_call_count": value.model_call_count,
            "normalization_applied": value.normalization_applied,
            "normalization_kind": value.normalization_kind,
        }

    return {
        "schema_version": OMNI_AV_SPEAKER_RAW_VERSION,
        "clip_uid": clip_uid,
        "segment_id": segment_id,
        "pass1": result(pass1),
        "pass2": result(pass2),
        "failure": (
            None
            if failure is None
            else {
                "code": failure.code,
                "reason": failure.reason,
                "raw_responses": list(failure.raw_responses),
                "completion_diagnostics": [
                    item.model_dump(mode="json")
                    for item in failure.completion_diagnostics
                ],
                "issues": [issue.to_dict() for issue in failure.issues],
                "model_call_count": failure.model_call_count,
            }
        ),
    }


def _review_html(
    records: Sequence[OmniAVSpeakerPilotRecord],
    manifest: Sequence[OmniAVSpeakerPilotCase],
) -> str:
    labels = {
        (item.clip_uid, item.segment_id): item.human_label for item in manifest
    }
    sections: list[str] = []
    for record in records:
        media = record.media_provenance
        video = (
            "[unavailable]"
            if media is None
            else f'<video controls preload="metadata" src="{html.escape(media.neutral_video_path)}"></video>'
        )
        audio = (
            ""
            if media is None
            else f'<audio controls preload="metadata" src="{html.escape(media.trimmed_audio_path)}"></audio>'
        )
        human = labels[(record.clip_uid, record.segment_id)]
        sections.append(
            "<section>"
            f"<h2>{html.escape(record.clip_uid)} / {html.escape(record.segment_id)}</h2>"
            f"<p>draft={html.escape(record.draft_binding_status)} "
            f"{html.escape(record.draft_entity_id or '-')} | "
            f"status={html.escape(record.status)} | comparison="
            f"{html.escape(record.comparison or '-')}</p>"
            f"<p>pass1={html.escape(str(record.pass1_decision))}/"
            f"{html.escape(record.pass1_entity_id or '-')}/"
            f"{html.escape(record.pass1_secondary_speech_status or '-')} | pass2="
            f"{html.escape(str(record.pass2_decision))}/"
            f"{html.escape(record.pass2_entity_id or '-')}/"
            f"{html.escape(record.pass2_secondary_speech_status or '-')}</p>"
            f"<p>primary stable={record.primary_observation_stable} | "
            f"secondary stable={record.secondary_speech_stable} | confirmed "
            f"secondary={html.escape(record.confirmed_secondary_speech_status or '-')}"
            f" | subject excluded={record.subject_entity_binding_excluded} | "
            "identity-specific voice excluded="
            f"{record.identity_specific_voice_products_excluded}</p>"
            f"<p>human QA={html.escape(human.model_dump_json() if human else '-')}</p>"
            f"{video}{audio}"
            "</section>"
        )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Omni AV Speaker Judge Pilot</title>
<style>body{font-family:system-ui;margin:24px;background:#f5f5f3;color:#171717}
section{border-top:1px solid #bbb;padding:18px 0}video{width:min(960px,100%);background:#111}
audio{display:block;width:min(960px,100%);margin-top:8px}
h1,h2{letter-spacing:0}code{white-space:pre-wrap}</style></head><body>
<h1>Omni AV Speaker Judge Pilot V1</h1>
<p>Read-only synchronized AV observations. Proposed identities are diagnostic only.</p>
""" + "\n".join(sections) + "\n</body></html>\n"


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"Omni AV speaker output exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_omni_av_speaker_judge_pilot(
    *,
    audio_production_root: Path,
    case_manifest_path: Path,
    output_root: Path | None,
    backend: OmniAVSpeakerJudgeBackend,
    media_backend: OmniAVSpeakerMediaBackend,
    overwrite: bool = False,
) -> OmniAVSpeakerPilotSummary:
    production_root = audio_production_root.expanduser().resolve(strict=True)
    paths = jea_production_paths(production_root)
    destination = (
        production_root / "omni_av_speaker_judge_pilot_v1"
        if output_root is None
        else output_root.expanduser().resolve(strict=False)
    )
    protected = (
        paths.audio,
        paths.primary_voice,
        paths.embedding,
        paths.pairs,
        paths.diarization,
        paths.asr,
        paths.h3,
    )
    if (
        destination == production_root
        or destination in production_root.parents
        or any(
            destination == path or path in destination.parents for path in protected
        )
    ):
        raise ValueError("Omni AV speaker output cannot replace a production stage")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Omni AV speaker output exists: {destination}")
    manifest_path = case_manifest_path.expanduser().resolve(strict=True)
    manifest = _load_manifest(manifest_path)
    sources, raw_segments_path = _load_source_cases(
        production_root=production_root,
        cases=manifest,
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[OmniAVSpeakerPilotRecord] = []
    try:
        temporary.mkdir()
        media_root = temporary / "media"
        raw_root = temporary / "raw"
        media_root.mkdir()
        raw_root.mkdir()
        for index, source in enumerate(sources, start=1):
            stem = f"{index:04d}_{source.manifest.clip_uid}_{source.manifest.segment_id}"
            neutral_video = media_root / f"{stem}.mp4"
            trimmed_audio = media_root / f"{stem}.wav"
            source_provenance, canonical_audio, _, model_video = _source_provenance(
                source=source,
                raw_segments_path=raw_segments_path,
            )
            synchronized_window: _SynchronizedTargetWindow | None = None
            visible_entities: list[str] = []
            pass1: OmniAVSpeakerJudgeResult | None = None
            pass2: OmniAVSpeakerJudgeResult | None = None
            pass2_called = False
            failure: OmniAVSpeakerJudgeFailure | None = None
            media_provenance: OmniAVSpeakerMediaProvenance | None = None
            try:
                synchronized_window = _synchronized_target_window(source)
                timeline = build_neutral_face_timeline(
                    native=source.native,
                    sidecar=source.sidecar,
                    window_start=synchronized_window.window_start,
                    window_end=synchronized_window.window_end,
                )
                visible_entities = _visible_entities(timeline)
                media_backend.render_neutral_video(
                    source_video_path=model_video,
                    timeline=timeline,
                    window_start=synchronized_window.window_start,
                    window_end=synchronized_window.window_end,
                    target_start=source.raw.start_time,
                    target_end=(
                        synchronized_window.effective_absolute_segment_end
                    ),
                    destination_path=neutral_video,
                )
                media_backend.trim_canonical_audio(
                    source_audio_path=canonical_audio,
                    window_start=synchronized_window.window_start,
                    window_end=synchronized_window.window_end,
                    destination_path=trimmed_audio,
                )
                media_provenance = OmniAVSpeakerMediaProvenance(
                    neutral_video_path=f"media/{neutral_video.name}",
                    neutral_video_sha256=_sha256_file(neutral_video),
                    trimmed_audio_path=f"media/{trimmed_audio.name}",
                    trimmed_audio_sha256=_sha256_file(trimmed_audio),
                )
                request = OmniAVSpeakerJudgeRequest(
                    neutral_video_path=neutral_video,
                    canonical_audio_path=trimmed_audio,
                    target_start_in_window=(
                        synchronized_window.target_start_in_window
                    ),
                    target_end_in_window=synchronized_window.target_end_in_window,
                    visible_candidate_entity_ids=tuple(visible_entities),
                )
                pass1 = backend.decide(request, verification=False)
                if _needs_verification(
                    draft_status=source.cluster.status,
                    draft_entity_id=source.cluster.entity_id,
                    pass1=pass1.observation,
                ):
                    pass2_called = True
                    pass2 = backend.decide(request, verification=True)
            except OmniAVSpeakerJudgeFailure as exc:
                failure = exc
            except RuntimeError as exc:
                failure = OmniAVSpeakerJudgeFailure(
                    code="omni_av_media_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            model_calls = sum(
                value.model_call_count for value in (pass1, pass2) if value is not None
            ) + (failure.model_call_count if failure is not None else 0)
            raw_count = sum(
                len(value.raw_responses) for value in (pass1, pass2) if value is not None
            ) + (len(failure.raw_responses) if failure is not None else 0)
            if failure is None:
                assert pass1 is not None
                resolution = _resolve_observations(
                    draft_status=source.cluster.status,
                    draft_entity_id=source.cluster.entity_id,
                    pass1=pass1.observation,
                    pass2=(None if pass2 is None else pass2.observation),
                )
            else:
                resolution = None
            record = OmniAVSpeakerPilotRecord(
                status="failed" if failure is not None else "succeeded",
                clip_uid=source.manifest.clip_uid,
                segment_id=source.manifest.segment_id,
                speaker_cluster_id=source.raw.speaker_cluster_id,
                absolute_segment_start=source.raw.start_time,
                absolute_segment_end=source.raw.end_time,
                target_boundary_clipped=(
                    synchronized_window is not None
                    and synchronized_window.boundary_clip_seconds > 0
                ),
                target_boundary_clip_seconds=(
                    0
                    if synchronized_window is None
                    else synchronized_window.boundary_clip_seconds
                ),
                effective_absolute_segment_end=(
                    None
                    if synchronized_window is None
                    else synchronized_window.effective_absolute_segment_end
                ),
                window_start=(
                    None
                    if synchronized_window is None
                    else synchronized_window.window_start
                ),
                window_end=(
                    None
                    if synchronized_window is None
                    else synchronized_window.window_end
                ),
                target_start_in_window=(
                    None
                    if synchronized_window is None
                    else synchronized_window.target_start_in_window
                ),
                target_end_in_window=(
                    None
                    if synchronized_window is None
                    else synchronized_window.target_end_in_window
                ),
                visible_candidate_entity_ids=visible_entities,
                draft_binding_status=source.cluster.status,
                draft_entity_id=source.cluster.entity_id,
                pass1_decision=(None if pass1 is None else pass1.observation.decision),
                pass1_entity_id=(None if pass1 is None else pass1.observation.entity_id),
                pass1_secondary_speech_status=(
                    None
                    if pass1 is None
                    else pass1.observation.secondary_speech_status
                ),
                pass2_called=pass2_called,
                pass2_decision=(None if pass2 is None else pass2.observation.decision),
                pass2_entity_id=(None if pass2 is None else pass2.observation.entity_id),
                pass2_secondary_speech_status=(
                    None
                    if pass2 is None
                    else pass2.observation.secondary_speech_status
                ),
                primary_observation_stable=(
                    False
                    if resolution is None
                    else resolution.primary_observation_stable
                ),
                secondary_speech_stable=(
                    False if resolution is None else resolution.secondary_speech_stable
                ),
                confirmed_secondary_speech_status=(
                    None
                    if resolution is None
                    else resolution.confirmed_secondary_speech_status
                ),
                comparison=(None if resolution is None else resolution.comparison),
                proposed_entity_id=(
                    None if resolution is None else resolution.proposed_entity_id
                ),
                proposed_non_entity_class=(
                    None
                    if resolution is None
                    else resolution.proposed_non_entity_class
                ),
                multiple_speakers_confirmed=(
                    False
                    if resolution is None
                    else resolution.multiple_speakers_confirmed
                ),
                subject_entity_binding_excluded=(
                    False
                    if resolution is None
                    else resolution.subject_entity_binding_excluded
                ),
                identity_specific_voice_products_excluded=(
                    False
                    if resolution is None
                    else resolution.identity_specific_voice_products_excluded
                ),
                source_provenance=source_provenance,
                media_provenance=media_provenance,
                backend_provenance=backend.provenance,
                model_call_count=model_calls,
                raw_response_count=raw_count,
                failure=(
                    None
                    if failure is None
                    else OmniAVSpeakerCaseFailure(
                        code=failure.code,
                        reason=failure.reason,
                        issues=[issue.to_dict() for issue in failure.issues],
                    )
                ),
            )
            records.append(record)
            _write_json(
                raw_root / f"{stem}.json",
                _raw_payload(
                    clip_uid=source.manifest.clip_uid,
                    segment_id=source.manifest.segment_id,
                    pass1=pass1,
                    pass2=pass2,
                    failure=failure,
                ),
            )
        comparisons = Counter(
            record.comparison for record in records if record.comparison is not None
        )
        summary = OmniAVSpeakerPilotSummary(
            source_audio_production_root=str(production_root),
            output_root=str(destination),
            case_count=len(records),
            succeeded_case_count=sum(record.status == "succeeded" for record in records),
            failed_case_count=sum(record.status == "failed" for record in records),
            pass2_case_count=sum(record.pass2_called for record in records),
            primary_observation_stable_count=sum(
                record.primary_observation_stable for record in records
            ),
            secondary_speech_stable_count=sum(
                record.secondary_speech_stable for record in records
            ),
            comparison_counts=dict(sorted(comparisons.items())),
            model_call_count=sum(record.model_call_count for record in records),
            backend_provenance=backend.provenance,
        )
        _write_jsonl(temporary / "manifest.jsonl", manifest)
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        (temporary / "review.html").write_text(
            _review_html(records, manifest),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary
