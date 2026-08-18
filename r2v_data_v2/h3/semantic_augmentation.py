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
from urllib.parse import quote, urlsplit

from openai import OpenAI
from pydantic import Field, StrictFloat, StrictStr, model_validator

from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    coalesce_audio_bindings,
)
from r2v_data_v2.h3.audio_production import (
    PRODUCTION_PAIR_VERSION,
    H3ProductionInPair,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

SEMANTIC_SCHEMA_VERSION = "r2v.h3.semantic_clip.2"
SEMANTIC_INVENTORY_VERSION = "r2v.h3.semantic_inventory.2"
SEMANTIC_SUMMARY_VERSION = "r2v.h3.semantic_summary.2"
SEMANTIC_PROMPT_VERSION = "h3_dots3_target_semantics_v1"
DEFAULT_DOTS3_MODEL = "dots3-note-prev"
DEFAULT_DOTS3_CHECKPOINT_ID = "dots-studio/dots3-note-prev-fp8"
PILOT_TARGET_COUNT = 20

SYSTEM_PROMPT = """You extract factual audio-video semantics for one target clip.
The supplied speech-turn timestamps and entity bindings are authoritative input.
Do not re-identify speakers, alter entity IDs, alter timestamps, merge turns, split
turns, or invent new turns. Output only turn_id, status, text, and language for each
speech turn; do not emit identity or timing fields. Transcribe only the supplied bound speech turns. If
speech is unclear or inaudible, use null text and the appropriate status instead of
guessing dialogue. Report background and non-speech sounds separately using only
the allowed categories. The audiovisual summary must describe only the target clip,
must be concise and factual, and must not invent names, identities, intentions, or
dialogue. Donor media and pair matching are outside this task. Return exactly one
compact JSON object matching the supplied schema, with no markdown or explanation."""


class SemanticSpeechTurnInput(SchemaModel):
    turn_id: str
    entity_id: str
    entity_occurrence_id: str
    start_time: StrictFloat = Field(ge=0)
    end_time: StrictFloat = Field(gt=0)

    @model_validator(mode="after")
    def validate_turn(self) -> SemanticSpeechTurnInput:
        if not self.turn_id.strip() or not self.entity_id.strip():
            raise ValueError("semantic speech turn identity must not be empty")
        if not self.entity_occurrence_id.endswith(f"/{self.entity_id}"):
            raise ValueError("semantic speech occurrence identity is inconsistent")
        if self.end_time <= self.start_time:
            raise ValueError("semantic speech turn interval must be positive")
        return self


class ModelSpeechTurnTranscript(SchemaModel):
    turn_id: str
    status: Literal["transcribed", "uncertain", "inaudible"]
    text: StrictStr | None = None
    language: StrictStr | None = None

    @model_validator(mode="after")
    def validate_transcript(self) -> ModelSpeechTurnTranscript:
        if not self.turn_id.strip():
            raise ValueError("semantic transcript turn ID must not be empty")
        if self.text is not None and not self.text.strip():
            raise ValueError("semantic transcript text must be non-empty or null")
        if self.language is not None and not self.language.strip():
            raise ValueError("semantic transcript language must be non-empty or null")
        if self.status == "transcribed" and self.text is None:
            raise ValueError("transcribed speech requires text")
        if self.status != "transcribed" and self.text is not None:
            raise ValueError("uncertain or inaudible speech must not guess text")
        return self


class SemanticSpeechTurnTranscript(SemanticSpeechTurnInput):
    status: Literal["transcribed", "uncertain", "inaudible"]
    text: StrictStr | None = None
    language: StrictStr | None = None

    @model_validator(mode="after")
    def validate_transcript(self) -> SemanticSpeechTurnTranscript:
        ModelSpeechTurnTranscript(
            turn_id=self.turn_id,
            status=self.status,
            text=self.text,
            language=self.language,
        )
        return self


class SemanticNonSpeechEvent(SchemaModel):
    start_time: StrictFloat = Field(ge=0)
    end_time: StrictFloat = Field(gt=0)
    category: Literal[
        "music",
        "environmental",
        "human_non_speech",
        "mechanical",
        "impact",
        "animal",
        "other",
    ]
    description: StrictStr

    @model_validator(mode="after")
    def validate_event(self) -> SemanticNonSpeechEvent:
        if self.end_time <= self.start_time or not self.description.strip():
            raise ValueError("non-speech event requires a positive interval and text")
        return self


class Dots3SemanticResponse(SchemaModel):
    speech_turn_transcripts: list[ModelSpeechTurnTranscript]
    non_speech_events: list[SemanticNonSpeechEvent]
    audiovisual_summary: StrictStr
    warnings: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_response(self) -> Dots3SemanticResponse:
        turn_ids = [item.turn_id for item in self.speech_turn_transcripts]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("semantic response turn IDs must be unique")
        if not self.audiovisual_summary.strip():
            raise ValueError("audiovisual summary must not be empty")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("semantic warnings must not be empty")
        events = sorted(
            self.non_speech_events,
            key=lambda item: (item.start_time, item.end_time, item.category),
        )
        if self.non_speech_events != events:
            raise ValueError("non-speech events must be chronologically ordered")
        return self


class SemanticFailure(SchemaModel):
    code: str
    reason: str
    attempt_count: int = Field(ge=0)
    issues: list[dict[str, str | None]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_failure(self) -> SemanticFailure:
        if not self.code.strip() or not self.reason.strip():
            raise ValueError("semantic failure code and reason must not be empty")
        return self


class SemanticBackendProvenance(SchemaModel):
    backend: Literal["vllm"] = "vllm"
    base_url: str
    model_identifier: str
    served_model_name: str
    checkpoint_id: str
    prompt_version: Literal["h3_dots3_target_semantics_v1"] = SEMANTIC_PROMPT_VERSION
    streaming: Literal[False] = False
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    repair_retries: Literal[1] = 1
    max_tokens: int = Field(gt=0)
    media_mode: Literal["file", "http"]
    media_root: str
    media_base_url: str | None = None
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> SemanticBackendProvenance:
        if (
            not self.base_url.strip()
            or not self.model_identifier.strip()
            or not self.served_model_name.strip()
            or not self.checkpoint_id.strip()
            or not self.media_root.strip()
        ):
            raise ValueError("semantic backend endpoint and model are required")
        if self.model_identifier != self.checkpoint_id:
            raise ValueError("semantic model identifier must equal checkpoint ID")
        if (self.media_mode == "http") != (self.media_base_url is not None):
            raise ValueError("semantic HTTP media provenance requires one base URL")
        if self.media_base_url is not None:
            parsed_media_url = urlsplit(self.media_base_url)
            if (
                parsed_media_url.scheme not in {"http", "https"}
                or not parsed_media_url.netloc
            ):
                raise ValueError(
                    "semantic HTTP media provenance has an invalid base URL"
                )
        if self.output_modalities != ["text"]:
            raise ValueError("semantic backend must request text output only")
        expected = _sha256_text(
            _compact_json(
                {
                    "backend": self.backend,
                    "base_url": self.base_url,
                    "model_identifier": self.model_identifier,
                    "served_model_name": self.served_model_name,
                    "checkpoint_id": self.checkpoint_id,
                    "prompt_version": self.prompt_version,
                    "streaming": self.streaming,
                    "output_modalities": self.output_modalities,
                    "repair_retries": self.repair_retries,
                    "max_tokens": self.max_tokens,
                    "media_mode": self.media_mode,
                    "media_root": self.media_root,
                    "media_base_url": self.media_base_url,
                }
            )
        )
        if self.configuration_fingerprint != expected:
            raise ValueError("semantic backend configuration fingerprint is invalid")
        return self


class SemanticClipRecord(SchemaModel):
    schema_version: Literal["r2v.h3.semantic_clip.2"] = SEMANTIC_SCHEMA_VERSION
    target_clip_uid: str
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_identifier: str
    backend_provenance: SemanticBackendProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["ready", "partial", "failed"]
    speech_turn_transcripts: list[SemanticSpeechTurnTranscript]
    non_speech_events: list[SemanticNonSpeechEvent]
    audiovisual_summary: str | None
    warnings: list[str]
    failure: SemanticFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> SemanticClipRecord:
        if (
            not self.target_clip_uid.strip()
            or not self.source_video_path.strip()
            or not self.source_audio_path.strip()
        ):
            raise ValueError("semantic record source identity must not be empty")
        if not self.model_identifier.strip():
            raise ValueError("semantic record model identifier must not be empty")
        if self.backend_provenance.model_identifier != self.model_identifier:
            raise ValueError("semantic record model provenance is inconsistent")
        if self.status == "failed":
            if self.failure is None or self.audiovisual_summary is not None:
                raise ValueError("failed semantics require failure and no summary")
            if self.non_speech_events:
                raise ValueError("failed semantics cannot publish non-speech events")
            if any(
                item.status != "uncertain"
                or item.text is not None
                or item.language is not None
                for item in self.speech_turn_transcripts
            ):
                raise ValueError("failed semantics must leave every transcript unknown")
        elif (
            self.failure is not None
            or self.audiovisual_summary is None
            or not self.audiovisual_summary.strip()
        ):
            raise ValueError("available semantics require a summary and no failure")
        if self.status == "ready" and any(
            item.status != "transcribed" for item in self.speech_turn_transcripts
        ):
            raise ValueError("ready semantics cannot contain uncertain speech")
        if self.status == "partial" and all(
            item.status == "transcribed" for item in self.speech_turn_transcripts
        ):
            raise ValueError("partial semantics require uncertain or inaudible speech")
        return self


class SemanticInventoryItem(SchemaModel):
    target_clip_uid: str
    pair_id: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    subject_count: int = Field(gt=0)
    speech_turns: list[SemanticSpeechTurnInput]

    @model_validator(mode="after")
    def validate_item(self) -> SemanticInventoryItem:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.target_clip_uid) is None:
            raise ValueError("semantic target clip UID is not path-safe")
        if self.pair_id != f"in_pair/{self.target_clip_uid}":
            raise ValueError("semantic inventory must originate from target in-pair")
        turn_ids = [item.turn_id for item in self.speech_turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("semantic inventory speech turns must be unique")
        if any(
            not item.entity_occurrence_id.startswith(f"{self.target_clip_uid}/")
            for item in self.speech_turns
        ):
            raise ValueError("semantic speech turn must belong to target clip")
        return self


class SemanticInventory(SchemaModel):
    schema_version: Literal["r2v.h3.semantic_inventory.2"] = SEMANTIC_INVENTORY_VERSION
    source_pair_schema_version: Literal["r2v.h3.production_pairs.2"] = (
        PRODUCTION_PAIR_VERSION
    )
    source_pairs_path: str
    source_pairs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20", "production"]
    source_target_count: int = Field(ge=0)
    selected_target_count: int = Field(ge=0)
    selection_mode: Literal[
        "multi_subject_first_then_clip_uid_v1",
        "complete_target_inventory_v1",
    ]
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    jobs: list[SemanticInventoryItem]

    @model_validator(mode="after")
    def validate_inventory(self) -> SemanticInventory:
        clip_ids = [item.target_clip_uid for item in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("semantic inventory target clips must be unique")
        if self.selected_target_count != len(self.jobs):
            raise ValueError("semantic selected target count is inconsistent")
        if self.source_target_count < self.selected_target_count:
            raise ValueError("semantic selection cannot exceed source targets")
        if self.mode == "production":
            if (
                self.selection_mode != "complete_target_inventory_v1"
                or self.bounded_selection_applied
                or self.selected_target_count != self.source_target_count
            ):
                raise ValueError("production semantics must cover every target clip")
        elif self.selection_mode != "multi_subject_first_then_clip_uid_v1":
            raise ValueError("pilot semantics must use deterministic pilot selection")
        return self


class SemanticProductionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.semantic_summary.2"] = SEMANTIC_SUMMARY_VERSION
    mode: Literal["pilot20", "production"]
    model_identifier: str
    backend_provenance: SemanticBackendProvenance
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_count: int = Field(ge=0)
    semantic_record_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    initial_call_count: int = Field(ge=0)
    repair_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    valid_pair_inputs_retained_count: int = Field(ge=0)
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    pair_assets_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> SemanticProductionSummary:
        if self.semantic_record_count != (
            self.ready_count + self.partial_count + self.failed_count
        ):
            raise ValueError("semantic record status counts must reconcile")
        if self.backend_provenance.model_identifier != self.model_identifier:
            raise ValueError("semantic summary model provenance is inconsistent")
        if self.initial_call_count > self.semantic_record_count:
            raise ValueError("semantic initial calls cannot exceed selected records")
        if self.raw_response_count > self.initial_call_count + self.repair_call_count:
            raise ValueError("semantic raw responses cannot exceed model calls")
        if self.valid_pair_inputs_retained_count != self.source_target_count:
            raise ValueError("semantic production must retain every source target pair")
        return self


@dataclass(frozen=True)
class SemanticBackendResult:
    response: Dots3SemanticResponse
    raw_responses: tuple[str, ...]


class SemanticAugmentationFailure(ValueError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: Sequence[str] = (),
        issues: Sequence[ValidationIssue] = (),
        attempt_count: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = tuple(raw_responses)
        self.issues = tuple(issues)
        self.attempt_count = (
            len(self.raw_responses) if attempt_count is None else attempt_count
        )


class SemanticAugmentationBackend(Protocol):
    @property
    def model_identifier(self) -> str: ...

    @property
    def provenance(self) -> SemanticBackendProvenance: ...

    def augment(self, job: SemanticInventoryItem) -> SemanticBackendResult: ...


@dataclass(frozen=True)
class MediaURLResolver:
    mode: Literal["file", "http"]
    media_root: Path
    media_base_url: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"file", "http"}:
            raise ValueError("DOTS3 media mode must be file or http")
        root = self.media_root.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("DOTS3 media root must be a directory")
        object.__setattr__(self, "media_root", root)
        if self.mode == "file":
            if self.media_base_url is not None:
                raise ValueError("file media mode cannot define an HTTP base URL")
            return
        if self.media_base_url is None:
            raise ValueError("http media mode requires DOTS3_MEDIA_BASE_URL")
        parsed = urlsplit(self.media_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("DOTS3 HTTP media base URL must be an HTTP(S) root")
        normalized = self.media_base_url.rstrip("/") + "/"
        object.__setattr__(self, "media_base_url", normalized)

    def resolve(self, source_path: Path) -> str:
        source = source_path.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("semantic media source must be a file")
        try:
            relative = source.relative_to(self.media_root)
        except ValueError as exc:
            raise ValueError(
                "semantic media source is outside DOTS3_MEDIA_ROOT"
            ) from exc
        if self.mode == "file":
            return source.as_uri()
        assert self.media_base_url is not None
        return self.media_base_url + quote(relative.as_posix(), safe="/")


@dataclass(frozen=True)
class Dots3VLLMSemanticConfig:
    base_url: str
    media_resolver: MediaURLResolver
    api_key: str = "EMPTY"
    served_model_name: str = DEFAULT_DOTS3_MODEL
    checkpoint_id: str = DEFAULT_DOTS3_CHECKPOINT_ID
    timeout_seconds: float = 600.0
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        if (
            not self.base_url.strip()
            or not self.api_key.strip()
            or not self.served_model_name.strip()
            or not self.checkpoint_id.strip()
        ):
            raise ValueError("dots3/vLLM endpoint, API key, and model IDs are required")
        if self.max_tokens <= 0:
            raise ValueError("dots3/vLLM token limit must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("dots3/vLLM timeout must be positive")

    def provenance(self) -> SemanticBackendProvenance:
        values = {
            "backend": "vllm",
            "base_url": self.base_url,
            "model_identifier": self.checkpoint_id,
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "streaming": False,
            "output_modalities": ["text"],
            "repair_retries": 1,
            "max_tokens": self.max_tokens,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
        }
        return SemanticBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    values = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL line {line_number} must be an object")
        values.append(payload)
    return values


def _canonical_bound_turns(
    *,
    clip_uid: str,
    sidecar: AudioBindingSidecar,
) -> list[SemanticSpeechTurnInput]:
    if sidecar.status != "ready" or sidecar.evidence is None:
        raise ValueError("semantic augmentation requires a ready Audio sidecar")
    sample_rate = sidecar.evidence.audio.sample_rate_hz
    if sample_rate is None:
        raise ValueError("semantic Audio sidecar lacks a sample rate")
    policy = AudioBindingProductionConfig()
    turns = coalesce_audio_bindings(
        sidecar.bindings,
        clip_uid=clip_uid,
        sample_rate_hz=sample_rate,
        maximum_gap_seconds=policy.speech_merge_gap_seconds,
        minimum_voice_reference_duration_seconds=(
            policy.minimum_voice_reference_duration_seconds
        ),
    )
    return [
        SemanticSpeechTurnInput(
            turn_id=turn.turn_id,
            entity_id=turn.entity_id,
            entity_occurrence_id=f"{clip_uid}/{turn.entity_id}",
            start_time=turn.start_time,
            end_time=turn.end_time,
        )
        for turn in turns
        if turn.status == "bound" and turn.entity_id is not None
    ]


def _inventory_fingerprint_payload(
    *,
    source_pairs_sha256: str,
    mode: str,
    jobs: Sequence[SemanticInventoryItem],
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "source_pairs_sha256": source_pairs_sha256,
                "mode": mode,
                "jobs": [item.model_dump(mode="json") for item in jobs],
            }
        )
    )


def build_semantic_inventory(
    *,
    pairs_root: Path,
    mode: Literal["pilot20", "production"],
) -> SemanticInventory:
    root = pairs_root.expanduser().resolve(strict=True)
    source_path = (root / "in_pairs.jsonl").resolve(strict=True)
    source_path.relative_to(root)
    pairs = [
        H3ProductionInPair.model_validate(item) for item in _read_jsonl(source_path)
    ]
    if not pairs:
        raise ValueError("semantic production requires at least one target in-pair")
    pairs.sort(key=lambda item: item.target_clip_uid)
    clip_ids = [item.target_clip_uid for item in pairs]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("production in-pairs contain duplicate target clips")

    all_jobs = []
    for pair in pairs:
        video_path = Path(pair.target_video_path).expanduser().resolve(strict=True)
        audio_path = Path(pair.target_full_audio_path).expanduser().resolve(strict=True)
        sidecar_path = (
            Path(pair.target_audio_binding_path).expanduser().resolve(strict=True)
        )
        if (
            not video_path.is_file()
            or not audio_path.is_file()
            or not sidecar_path.is_file()
        ):
            raise ValueError("semantic target media or Audio sidecar is unavailable")
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.clip_uid != pair.target_clip_uid:
            raise ValueError("semantic Audio sidecar clip identity does not match pair")
        if sidecar.status != "ready" or sidecar.evidence is None:
            raise ValueError("semantic augmentation requires a ready Audio sidecar")
        sidecar_video = (
            Path(sidecar.source_video_path).expanduser().resolve(strict=True)
        )
        if sidecar_video != video_path:
            raise ValueError("semantic Audio sidecar source video does not match pair")
        sidecar_audio_value = sidecar.evidence.audio.full_audio_path
        if sidecar_audio_value is None:
            raise ValueError("semantic Audio sidecar lacks full-audio evidence")
        sidecar_audio = Path(sidecar_audio_value).expanduser().resolve(strict=True)
        if sidecar_audio != audio_path:
            raise ValueError("semantic Audio sidecar full audio does not match pair")
        all_jobs.append(
            SemanticInventoryItem(
                target_clip_uid=pair.target_clip_uid,
                pair_id=pair.pair_id,
                target_video_path=str(video_path),
                target_video_sha256=_sha256_file(video_path),
                target_full_audio_path=str(audio_path),
                target_full_audio_sha256=_sha256_file(audio_path),
                target_audio_binding_path=str(sidecar_path),
                subject_count=len(pair.subjects),
                speech_turns=_canonical_bound_turns(
                    clip_uid=pair.target_clip_uid,
                    sidecar=sidecar,
                ),
            )
        )

    if mode == "pilot20":
        multi_subject = [item for item in all_jobs if item.subject_count > 1]
        if len(multi_subject) > PILOT_TARGET_COUNT:
            raise ValueError("pilot20 cannot include every multi-subject target")
        selected_ids = {item.target_clip_uid for item in multi_subject}
        remaining = [
            item for item in all_jobs if item.target_clip_uid not in selected_ids
        ]
        jobs = multi_subject + remaining[: PILOT_TARGET_COUNT - len(multi_subject)]
        selection_mode = "multi_subject_first_then_clip_uid_v1"
        bounded = len(jobs) < len(all_jobs)
    else:
        jobs = all_jobs
        selection_mode = "complete_target_inventory_v1"
        bounded = False
    source_sha = _sha256_file(source_path)
    return SemanticInventory(
        source_pairs_path=str(source_path),
        source_pairs_sha256=source_sha,
        inventory_fingerprint=_inventory_fingerprint_payload(
            source_pairs_sha256=source_sha,
            mode=mode,
            jobs=jobs,
        ),
        mode=mode,
        source_target_count=len(all_jobs),
        selected_target_count=len(jobs),
        selection_mode=selection_mode,
        bounded_selection_applied=bounded,
        jobs=jobs,
    )


def semantic_request_fingerprint(
    job: SemanticInventoryItem,
    *,
    backend_provenance: SemanticBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "prompt_version": SEMANTIC_PROMPT_VERSION,
                "backend_configuration_fingerprint": (
                    backend_provenance.configuration_fingerprint
                ),
                "target_clip_uid": job.target_clip_uid,
                "source_video_sha256": job.target_video_sha256,
                "source_audio_sha256": job.target_full_audio_sha256,
                "speech_turns": [
                    item.model_dump(mode="json") for item in job.speech_turns
                ],
            }
        )
    )


def _response_validation_issues(
    response: Dots3SemanticResponse,
    job: SemanticInventoryItem,
) -> list[ValidationIssue]:
    expected = job.speech_turns
    actual = response.speech_turn_transcripts
    issues = []
    expected_ids = [item.turn_id for item in expected]
    actual_ids = [item.turn_id for item in actual]
    if actual_ids != expected_ids:
        unknown = sorted(set(actual_ids) - set(expected_ids))
        missing = sorted(set(expected_ids) - set(actual_ids))
        if unknown:
            issues.append(
                ValidationIssue(
                    code="unknown_turn_id",
                    field="speech_turn_transcripts",
                    message=f"unknown turn IDs: {unknown}",
                )
            )
        if missing:
            issues.append(
                ValidationIssue(
                    code="missing_turn_id",
                    field="speech_turn_transcripts",
                    message=f"missing turn IDs: {missing}",
                )
            )
        if not unknown and not missing:
            issues.append(
                ValidationIssue(
                    code="turn_order_changed",
                    field="speech_turn_transcripts",
                    message="speech turns must preserve input order",
                )
            )
    return issues


def _user_prompt(job: SemanticInventoryItem) -> str:
    payload = {
        "target_clip_uid": job.target_clip_uid,
        "authoritative_bound_speech_turns": [
            item.model_dump(mode="json") for item in job.speech_turns
        ],
        "allowed_non_speech_categories": [
            "music",
            "environmental",
            "human_non_speech",
            "mechanical",
            "impact",
            "animal",
            "other",
        ],
    }
    return (
        "Analyze the attached target video and separate target full-audio item. The "
        "input speech-turn identities and timestamps are authoritative. Return only "
        "turn_id, status, text, and language for every supplied turn and no others; "
        "the producer copies identity and timing from input. Use null text rather than "
        "guessing unclear speech.\nInput:\n"
        f"{_compact_json(payload)}\nJSON Schema:\n"
        f"{_compact_json(Dots3SemanticResponse.model_json_schema())}"
    )


def _repair_prompt(
    *,
    job: SemanticInventoryItem,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        "Repair the previous JSON only. Reinspect the same target video when needed. "
        "Do not invent dialogue. Return every authoritative turn ID exactly once; "
        "identity and timing are copied from input and must not be emitted. Return one "
        "compact JSON object only, with no markdown "
        "or explanation.\nOriginal request:\n"
        f"{_user_prompt(job)}\nValidation issues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\n"
        f"Invalid response:\n{invalid_response}"
    )


def _verify_source_hash(*, path: Path, expected: str, kind: str) -> None:
    if _sha256_file(path) != expected:
        raise SemanticAugmentationFailure(
            code=f"source_{kind}_changed",
            reason=f"target {kind} bytes changed after semantic inventory construction",
        )


class OpenAIDots3VLLMBackend:
    def __init__(
        self,
        config: Dots3VLLMSemanticConfig,
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
    def model_identifier(self) -> str:
        return self.config.checkpoint_id

    @property
    def provenance(self) -> SemanticBackendProvenance:
        return self.config.provenance()

    def _request_text(self, *, job: SemanticInventoryItem, prompt: str) -> str:
        video_path = Path(job.target_video_path)
        audio_path = Path(job.target_full_audio_path)
        _verify_source_hash(
            path=video_path,
            expected=job.target_video_sha256,
            kind="video",
        )
        _verify_source_hash(
            path=audio_path,
            expected=job.target_full_audio_sha256,
            kind="audio",
        )
        video_url = self.config.media_resolver.resolve(video_path)
        audio_url = self.config.media_resolver.resolve(audio_path)
        try:
            completion = self.client.chat.completions.create(
                model=self.config.served_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "video_url",
                                "video_url": {"url": video_url},
                            },
                            {
                                "type": "audio_url",
                                "audio_url": {"url": audio_url},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choices = getattr(completion, "choices", None)
            if not choices:
                raise TypeError("dots3/vLLM response has no choices")
            response = getattr(choices[0].message, "content", None)
            if not isinstance(response, str):
                raise TypeError("dots3/vLLM response text must be a string")
        except SemanticAugmentationFailure:
            raise
        except Exception as exc:
            raise SemanticAugmentationFailure(
                code="dots3_vllm_request_failed",
                reason=f"{type(exc).__name__}: {exc}",
                attempt_count=1,
            ) from exc
        if not response.strip():
            raise SemanticAugmentationFailure(
                code="dots3_vllm_empty_response",
                reason="dots3/vLLM returned no text",
                raw_responses=[response],
            )
        return response

    def augment(self, job: SemanticInventoryItem) -> SemanticBackendResult:
        raw_responses = []
        issues: list[ValidationIssue] = []
        for attempt in range(2):
            prompt = _user_prompt(job)
            if attempt:
                prompt = _repair_prompt(
                    job=job,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            try:
                raw = self._request_text(job=job, prompt=prompt)
            except SemanticAugmentationFailure as exc:
                raise SemanticAugmentationFailure(
                    code=exc.code,
                    reason=exc.reason,
                    raw_responses=[*raw_responses, *exc.raw_responses],
                    issues=exc.issues,
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            response, issues = parse_structured_json_issues(raw, Dots3SemanticResponse)
            if response is not None:
                issues = _response_validation_issues(response, job)
            if response is not None and not issues:
                return SemanticBackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                )
        raise SemanticAugmentationFailure(
            code="structured_output_failed",
            reason="dots3 structured output failed closed after one repair",
            raw_responses=raw_responses,
            issues=issues,
        )


def _materialize_transcripts(
    *,
    job: SemanticInventoryItem,
    response: Dots3SemanticResponse,
) -> list[SemanticSpeechTurnTranscript]:
    return [
        SemanticSpeechTurnTranscript(
            **source.model_dump(mode="python"),
            status=decision.status,
            text=decision.text,
            language=decision.language,
        )
        for source, decision in zip(
            job.speech_turns,
            response.speech_turn_transcripts,
            strict=True,
        )
    ]


def _failed_transcripts(
    job: SemanticInventoryItem,
) -> list[SemanticSpeechTurnTranscript]:
    return [
        SemanticSpeechTurnTranscript(
            **item.model_dump(mode="python"),
            status="uncertain",
            text=None,
            language=None,
        )
        for item in job.speech_turns
    ]


def _success_record(
    *,
    job: SemanticInventoryItem,
    result: SemanticBackendResult,
    backend_provenance: SemanticBackendProvenance,
) -> SemanticClipRecord:
    issues = _response_validation_issues(result.response, job)
    if issues:
        raise RuntimeError("semantic backend returned an unvalidated response")
    transcripts = _materialize_transcripts(job=job, response=result.response)
    status: Literal["ready", "partial"] = "ready"
    if any(item.status != "transcribed" for item in transcripts):
        status = "partial"
    return SemanticClipRecord(
        target_clip_uid=job.target_clip_uid,
        source_video_path=job.target_video_path,
        source_video_sha256=job.target_video_sha256,
        source_audio_path=job.target_full_audio_path,
        source_audio_sha256=job.target_full_audio_sha256,
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        request_fingerprint=semantic_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status=status,
        speech_turn_transcripts=transcripts,
        non_speech_events=result.response.non_speech_events,
        audiovisual_summary=result.response.audiovisual_summary,
        warnings=result.response.warnings,
    )


def _failed_record(
    *,
    job: SemanticInventoryItem,
    failure: SemanticAugmentationFailure,
    backend_provenance: SemanticBackendProvenance,
) -> SemanticClipRecord:
    return SemanticClipRecord(
        target_clip_uid=job.target_clip_uid,
        source_video_path=job.target_video_path,
        source_video_sha256=job.target_video_sha256,
        source_audio_path=job.target_full_audio_path,
        source_audio_sha256=job.target_full_audio_sha256,
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        request_fingerprint=semantic_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status="failed",
        speech_turn_transcripts=_failed_transcripts(job),
        non_speech_events=[],
        audiovisual_summary=None,
        warnings=[failure.code],
        failure=SemanticFailure(
            code=failure.code,
            reason=failure.reason,
            attempt_count=failure.attempt_count,
            issues=[item.to_dict() for item in failure.issues],
        ),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(_compact_json(item.model_dump(mode="json")) + "\n" for item in values),
        encoding="utf-8",
    )


def _review_html(
    *,
    inventory: SemanticInventory,
    records: Sequence[SemanticClipRecord],
    video_media_names: dict[str, str],
    audio_media_names: dict[str, str],
) -> str:
    record_by_clip = {item.target_clip_uid: item for item in records}
    cards = []
    for job in inventory.jobs:
        record = record_by_clip[job.target_clip_uid]
        transcripts = {item.turn_id: item for item in record.speech_turn_transcripts}
        rows = []
        for turn in job.speech_turns:
            transcript = transcripts[turn.turn_id]
            rows.append(
                "<tr>"
                f"<td>{html.escape(turn.turn_id)}</td>"
                f"<td>{html.escape(turn.entity_occurrence_id)}</td>"
                f"<td>{turn.start_time:.3f}-{turn.end_time:.3f}</td>"
                f"<td>{html.escape(transcript.status)}</td>"
                f"<td>{html.escape(transcript.text or '[null]')}</td>"
                f"<td>{html.escape(transcript.language or '[null]')}</td>"
                "</tr>"
            )
        events = (
            "".join(
                "<li>"
                f"{event.start_time:.3f}-{event.end_time:.3f} "
                f"[{html.escape(event.category)}] {html.escape(event.description)}"
                "</li>"
                for event in record.non_speech_events
            )
            or "<li>None reported</li>"
        )
        failure = (
            "None"
            if record.failure is None
            else f"{record.failure.code}: {record.failure.reason}"
        )
        cards.append(
            f"<article class='case' data-clip='{html.escape(job.target_clip_uid)}'>"
            f"<h2>{html.escape(job.target_clip_uid)} <span>{record.status}</span></h2>"
            f"<video controls preload='metadata' src='media/{html.escape(video_media_names[job.target_clip_uid])}'></video>"
            f"<audio controls preload='metadata' src='media/{html.escape(audio_media_names[job.target_clip_uid])}'></audio>"
            "<table><thead><tr><th>turn</th><th>entity</th><th>time</th>"
            "<th>status</th><th>transcript</th><th>language</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"<h3>Non-speech events</h3><ul>{events}</ul>"
            f"<h3>Audiovisual summary</h3><p>{html.escape(record.audiovisual_summary or '[unavailable]')}</p>"
            f"<p><b>Warnings:</b> {html.escape(', '.join(record.warnings) or 'None')}</p>"
            f"<p><b>Raw failure status:</b> {html.escape(failure)}</p>"
            "<div class='labels'><b>Human QA:</b> "
            + " ".join(
                f"<label><input type='radio' name='qa-{html.escape(job.target_clip_uid)}' "
                f"value='{label}' onchange='saveLabel(this)'>{label}</label>"
                for label in ("CORRECT", "WRONG", "UNCERTAIN")
            )
            + "</div></article>"
        )
    return (
        """<!doctype html>
<html><head><meta charset="utf-8"><title>H3 Omni semantic review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f7;color:#171717}
header,.case{max-width:1100px;margin:0 auto 24px;background:white;border:1px solid #ccc;padding:18px}
h1,h2,h3{letter-spacing:0}h2 span{font-size:14px;color:#555}video{width:100%;max-height:560px;background:#111}audio{width:100%;margin-top:10px}
table{width:100%;border-collapse:collapse;margin-top:14px}th,td{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}
.labels{padding:12px;background:#eef2f5}.labels label{margin-right:18px}
</style></head><body>
<header><h1>H3 Omni semantic review</h1><p>Human labels are QA only and are not identity truth.</p>
<p>Check hallucinated dialogue, wrong transcription, missing audible events, and hallucinated audio-video facts.</p></header>
"""
        + "".join(cards)
        + """
<script>
function saveLabel(input){localStorage.setItem('h3-semantic-'+input.name,input.value);}
document.querySelectorAll('.labels input').forEach(function(input){
 const saved=localStorage.getItem('h3-semantic-'+input.name); if(saved===input.value) input.checked=true;
});
</script></body></html>
"""
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"semantic output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_semantic_augmentation(
    *,
    inventory: SemanticInventory,
    output_root: Path,
    backend: SemanticAugmentationBackend,
    overwrite: bool = False,
) -> SemanticProductionSummary:
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records = []
    raw_response_count = 0
    initial_call_count = 0
    repair_call_count = 0
    failure_counts: Counter[str] = Counter()
    video_media_names: dict[str, str] = {}
    audio_media_names: dict[str, str] = {}
    try:
        temporary.mkdir()
        (temporary / "raw").mkdir()
        (temporary / "media").mkdir()
        for job in inventory.jobs:
            raw_responses: tuple[str, ...] = ()
            failure: SemanticAugmentationFailure | None = None
            try:
                result = backend.augment(job)
                raw_responses = result.raw_responses
                record = _success_record(
                    job=job,
                    result=result,
                    backend_provenance=backend.provenance,
                )
            except SemanticAugmentationFailure as exc:
                failure = exc
                raw_responses = exc.raw_responses
                failure_counts.update([exc.code])
                record = _failed_record(
                    job=job,
                    failure=exc,
                    backend_provenance=backend.provenance,
                )
            records.append(record)
            attempted_calls = (
                len(raw_responses)
                if failure is None
                else max(len(raw_responses), failure.attempt_count)
            )
            if attempted_calls:
                initial_call_count += 1
                repair_call_count += max(0, attempted_calls - 1)
            raw_response_count += len(raw_responses)
            _write_json(
                temporary / "raw" / f"{job.target_clip_uid}.json",
                {
                    "target_clip_uid": job.target_clip_uid,
                    "model_identifier": backend.model_identifier,
                    "backend_provenance": backend.provenance.model_dump(mode="json"),
                    "request_fingerprint": record.request_fingerprint,
                    "status": record.status,
                    "raw_responses": list(raw_responses),
                    "failure": (
                        None
                        if failure is None
                        else record.failure.model_dump(mode="json")
                    ),
                },
            )
            video_source = Path(job.target_video_path)
            video_suffix = video_source.suffix.lower() or ".video"
            video_name = f"{job.target_clip_uid}{video_suffix}"
            audio_source = Path(job.target_full_audio_path)
            audio_suffix = audio_source.suffix.lower() or ".audio"
            audio_name = f"{job.target_clip_uid}.audio{audio_suffix}"
            if any("/" in name or "\\" in name for name in (video_name, audio_name)):
                raise ValueError("semantic clip UID cannot contain path separators")
            (temporary / "media" / video_name).symlink_to(video_source)
            (temporary / "media" / audio_name).symlink_to(audio_source)
            video_media_names[job.target_clip_uid] = video_name
            audio_media_names[job.target_clip_uid] = audio_name

        status_counts = Counter(item.status for item in records)
        summary = SemanticProductionSummary(
            mode=inventory.mode,
            model_identifier=backend.model_identifier,
            backend_provenance=backend.provenance,
            inventory_fingerprint=inventory.inventory_fingerprint,
            source_target_count=inventory.source_target_count,
            semantic_record_count=len(records),
            ready_count=status_counts["ready"],
            partial_count=status_counts["partial"],
            failed_count=status_counts["failed"],
            initial_call_count=initial_call_count,
            repair_call_count=repair_call_count,
            raw_response_count=raw_response_count,
            failure_reason_counts=dict(sorted(failure_counts.items())),
            valid_pair_inputs_retained_count=inventory.source_target_count,
            bounded_selection_applied=inventory.bounded_selection_applied,
        )
        _write_json(
            temporary / "inventory.json",
            inventory.model_dump(mode="json"),
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "review.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                video_media_names=video_media_names,
                audio_media_names=audio_media_names,
            ),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def semantic_output_root(
    audio_run_root: Path,
    *,
    mode: Literal["pilot20", "production"],
) -> Path:
    root = audio_run_root.expanduser().resolve(strict=False)
    if mode == "pilot20":
        return root / "semantic_pilot20"
    return root / "production" / "semantic"


def semantic_stage_is_complete(path: Path) -> bool:
    return (
        path.is_dir()
        and all(
            (path / name).is_file()
            for name in (
                "inventory.json",
                "records.jsonl",
                "summary.json",
                "review.html",
            )
        )
        and (path / "raw").is_dir()
    )
