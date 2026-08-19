from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field, StrictBool, StrictStr, model_validator

from r2v_data_v2.h3.asr_v2_transcription import (
    ASRV2Inventory,
    ASRV2ProductionInventory,
    ASRV2SegmentJob,
    ASRV2SegmentRecord,
    load_asr_v2_inventory,
    load_asr_v2_summary,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    _inventory_fingerprint as _asr_v2_inventory_fingerprint,
)
from r2v_data_v2.h3.background_audio_scout import (
    BackgroundAudioPilotSelection,
    build_background_audio_scout_inventory,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.text_usability import (
    TextUsabilityInventory,
    TextUsabilitySegment,
    TextUsabilitySummary,
)
from r2v_data_v2.h3.text_usability import (
    _inventory_fingerprint as _text_usability_inventory_fingerprint,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

TARGET_AUDIO_CAPTION_SCHEMA_VERSION = "r2v.h3.target_audio_caption.1"
TARGET_AUDIO_CAPTION_INVENTORY_VERSION = "r2v.h3.target_audio_caption_inventory.1"
TARGET_AUDIO_CAPTION_SUMMARY_VERSION = "r2v.h3.target_audio_caption_summary.1"
TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION = "r2v.h3.target_audio_caption_human_qa.1"
TARGET_AUDIO_CAPTION_PROMPT_VERSION = "h3_dots3_target_audio_caption_v1"
TARGET_AUDIO_CAPTION_OUTPUT_DIRECTORY = "target_audio_caption_pilot20"
TARGET_AUDIO_CAPTION_BACKGROUND_OUTPUT_DIRECTORY = (
    "target_audio_caption_background_pilot"
)
TARGET_AUDIO_INPUT_MODALITY = "native_target_video_with_embedded_audio"
PILOT_TARGET_COUNT = 20
DEFAULT_DOTS3_MODEL = "dots3-note-prev"
DEFAULT_DOTS3_CHECKPOINT_ID = "/mnt/workspace/public/pretrained/dots3-note-prev"

QA_LABELS = ("CORRECT", "WRONG", "UNCERTAIN")
QA_FLAGS = (
    "hallucinated_music",
    "hallucinated_sound_event",
    "wrong_ambient_scene",
    "wrong_speaker_style",
    "dialogue_leakage",
    "other",
)

SYSTEM_PROMPT = """You analyze AUDIO SEMANTICS for one target clip.
Use audible evidence only. The attached native target video includes its original
audio track. Visual content must never be used to guess a sound.

Report only: audible ambient/background environment, background-music presence and
style, non-speech sound events, acoustic atmosphere/style, and delivery/prosody for
the supplied speaker clusters during their exact active time ranges.

Never transcribe, quote, paraphrase, correct, or summarize dialogue. Never identify a
speaker or infer entity identity, subject identity, gender, age, nationality, or
intrinsic voice identity/timbre. Do not emit entity_id. If audible evidence is unclear,
use null or an empty list and do not invent details. Return every supplied
speaker_cluster_id exactly once, in supplied order, and no unknown cluster ID.

Return exactly one compact JSON object matching the supplied schema, with no markdown
or explanation."""


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} in {path} must be an object")
        rows.append(value)
    return rows


class SpeakerTimeRange(SchemaModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> SpeakerTimeRange:
        if self.end_time <= self.start_time:
            raise ValueError("speaker time range must be positive")
        return self


class SpeakerClusterEvidence(SchemaModel):
    speaker_cluster_id: str
    entity_id: str | None = None
    active_time_ranges: list[SpeakerTimeRange]

    @model_validator(mode="after")
    def validate_cluster(self) -> SpeakerClusterEvidence:
        if not self.speaker_cluster_id.strip() or not self.active_time_ranges:
            raise ValueError("speaker cluster evidence must be complete")
        if self.entity_id is not None and not self.entity_id.strip():
            raise ValueError("speaker entity ID must be non-empty or null")
        if self.active_time_ranges != sorted(
            self.active_time_ranges,
            key=lambda item: (item.start_time, item.end_time),
        ):
            raise ValueError("speaker time ranges must be chronological")
        return self


class BackgroundMusic(SchemaModel):
    present: StrictBool | None = None
    style: StrictStr | None = None

    @model_validator(mode="after")
    def validate_music(self) -> BackgroundMusic:
        if self.style is not None and not self.style.strip():
            raise ValueError("background music style must be non-empty or null")
        if self.present is False and self.style is not None:
            raise ValueError("absent background music cannot have a style")
        return self


class ModelSpeakerDelivery(SchemaModel):
    speaker_cluster_id: StrictStr
    delivery_style: list[StrictStr]

    @model_validator(mode="after")
    def validate_delivery(self) -> ModelSpeakerDelivery:
        if not self.speaker_cluster_id.strip():
            raise ValueError("speaker cluster ID must not be empty")
        if any(not item.strip() for item in self.delivery_style):
            raise ValueError("speaker delivery labels must not be empty")
        if len(self.delivery_style) != len(set(self.delivery_style)):
            raise ValueError("speaker delivery labels must be unique")
        return self


class Dots3TargetAudioCaptionResponse(SchemaModel):
    ambient_scene: StrictStr | None = None
    background_music: BackgroundMusic
    sound_events: list[StrictStr]
    acoustic_style: list[StrictStr]
    speaker_delivery: list[ModelSpeakerDelivery]

    @model_validator(mode="after")
    def validate_response(self) -> Dots3TargetAudioCaptionResponse:
        if self.ambient_scene is not None and not self.ambient_scene.strip():
            raise ValueError("ambient scene must be non-empty or null")
        for values, name in (
            (self.sound_events, "sound event"),
            (self.acoustic_style, "acoustic style"),
        ):
            if any(not item.strip() for item in values):
                raise ValueError(f"{name} values must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("speaker delivery cluster IDs must be unique")
        return self


class TargetSpeakerDelivery(ModelSpeakerDelivery):
    entity_id: str | None = None


class TargetAudioCaptionFailure(SchemaModel):
    code: str
    reason: str
    attempt_count: int = Field(ge=0)
    issues: list[dict[str, str | None]] = Field(default_factory=list)


class TargetAudioCaptionBackendProvenance(SchemaModel):
    backend: Literal["vllm"] = "vllm"
    served_model_name: str
    checkpoint_id: str
    base_url: str
    prompt_version: Literal["h3_dots3_target_audio_caption_v1"] = (
        TARGET_AUDIO_CAPTION_PROMPT_VERSION
    )
    input_modality: Literal["native_target_video_with_embedded_audio"] = (
        TARGET_AUDIO_INPUT_MODALITY
    )
    separate_audio_sent: Literal[False] = False
    transcript_supplied: Literal[False] = False
    donor_media_used: Literal[False] = False
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    media_mode: Literal["file", "http"]
    media_root: str
    media_base_url: str | None = None
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> TargetAudioCaptionBackendProvenance:
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("target audio caption backend fingerprint is invalid")
        return self


class TargetAudioCaptionJob(SchemaModel):
    target_clip_uid: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    target_audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_clusters: list[SpeakerClusterEvidence]

    @model_validator(mode="after")
    def validate_job(self) -> TargetAudioCaptionJob:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.target_clip_uid) is None:
            raise ValueError("target clip UID is not path-safe")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_clusters]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("target audio caption clusters must be unique")
        return self


class TargetAudioCaptionInventory(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_inventory.1"] = (
        TARGET_AUDIO_CAPTION_INVENTORY_VERSION
    )
    mode: Literal["pilot20", "background_pilot"] = "pilot20"
    source_pilot_inventory_path: str | None = None
    source_pilot_inventory_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_pilot_inventory_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_selection_path: str | None = None
    source_selection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_scout_inventory_fingerprint: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    source_diarization_root: str
    source_diarization_inventory_path: str
    source_diarization_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_raw_segments_path: str
    source_diarization_raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_bound_segments_path: str
    source_diarization_bound_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_cluster_bindings_path: str
    source_diarization_cluster_bindings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_summary_path: str
    source_diarization_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_root: str
    source_asr_v2_inventory_path: str
    source_asr_v2_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_segments_path: str
    source_asr_v2_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_asr_v2_summary_path: str
    source_asr_v2_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_usability_root: str
    source_text_usability_inventory_path: str
    source_text_usability_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_usability_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_usability_segments_path: str
    source_text_usability_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_usability_summary_path: str
    source_text_usability_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_root: str
    selected_target_count: int = Field(gt=0)
    selection_mode: Literal[
        "exact_asr_v2_pilot20_order_v1", "manual_background_audio_selection_v1"
    ] = "exact_asr_v2_pilot20_order_v1"
    bounded_selection_applied: Literal[True] = True
    parent_quota_applied: Literal[False] = False
    transcript_supplied_to_model: Literal[False] = False
    final_renderer_applied: Literal[False] = False
    jobs: list[TargetAudioCaptionJob]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> TargetAudioCaptionInventory:
        if len(self.jobs) != self.selected_target_count:
            raise ValueError("target audio caption job count is inconsistent")
        pilot_sources = (
            self.source_pilot_inventory_path,
            self.source_pilot_inventory_sha256,
            self.source_pilot_inventory_fingerprint,
        )
        selection_sources = (
            self.source_selection_path,
            self.source_selection_sha256,
            self.source_scout_inventory_fingerprint,
        )
        if self.mode == "pilot20":
            if (
                self.selected_target_count != PILOT_TARGET_COUNT
                or self.selection_mode != "exact_asr_v2_pilot20_order_v1"
                or any(item is None for item in pilot_sources)
                or any(item is not None for item in selection_sources)
            ):
                raise ValueError("target audio caption pilot20 provenance is invalid")
        elif (
            self.selection_mode != "manual_background_audio_selection_v1"
            or any(item is None for item in selection_sources)
            or any(item is not None for item in pilot_sources)
        ):
            raise ValueError("background target audio caption provenance is invalid")
        clip_ids = [item.target_clip_uid for item in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("target audio caption pilot clips must be unique")
        if self.inventory_fingerprint != _inventory_fingerprint(self):
            raise ValueError("target audio caption inventory fingerprint is invalid")
        return self


class TargetAudioCaptionRecord(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption.1"] = (
        TARGET_AUDIO_CAPTION_SCHEMA_VERSION
    )
    target_clip_uid: str
    status: Literal["ready", "failed"]
    ambient_scene: str | None = None
    background_music: BackgroundMusic | None = None
    sound_events: list[str] = Field(default_factory=list)
    acoustic_style: list[str] = Field(default_factory=list)
    speaker_delivery: list[TargetSpeakerDelivery] = Field(default_factory=list)
    audio_prompt_draft: str | None = None
    input_modality: Literal["native_target_video_with_embedded_audio"] = (
        TARGET_AUDIO_INPUT_MODALITY
    )
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    target_audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: TargetAudioCaptionBackendProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure: TargetAudioCaptionFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> TargetAudioCaptionRecord:
        if self.status == "ready":
            if self.failure is not None or self.background_music is None:
                raise ValueError("ready target audio caption requires model output")
            if self.audio_prompt_draft is None or not self.audio_prompt_draft.strip():
                raise ValueError("ready target audio caption requires preview text")
        elif self.failure is None or any(
            (
                self.ambient_scene is not None,
                self.background_music is not None,
                bool(self.sound_events),
                bool(self.acoustic_style),
                bool(self.speaker_delivery),
                self.audio_prompt_draft is not None,
            )
        ):
            raise ValueError("failed target audio caption cannot publish semantics")
        return self


class TargetAudioCaptionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_summary.1"] = (
        TARGET_AUDIO_CAPTION_SUMMARY_VERSION
    )
    mode: Literal["pilot20", "background_pilot"] = "pilot20"
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_identifier: str
    backend_provenance: TargetAudioCaptionBackendProvenance
    target_clip_count: int = Field(gt=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    initial_call_count: int = Field(ge=0)
    repair_call_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    transcript_supplied_to_model: Literal[False] = False
    donor_media_used: Literal[False] = False
    denoising_applied: Literal[False] = False
    enhancement_applied: Literal[False] = False
    final_renderer_modified: Literal[False] = False
    human_qa_pending: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> TargetAudioCaptionSummary:
        if self.ready_count + self.failed_count != self.target_clip_count:
            raise ValueError("target audio caption summary counts must reconcile")
        return self


class TargetAudioCaptionHumanQALabel(SchemaModel):
    target_clip_uid: str
    label: Literal["CORRECT", "WRONG", "UNCERTAIN"]
    failure_flags: list[
        Literal[
            "hallucinated_music",
            "hallucinated_sound_event",
            "wrong_ambient_scene",
            "wrong_speaker_style",
            "dialogue_leakage",
            "other",
        ]
    ] = Field(default_factory=list)


class TargetAudioCaptionHumanQAExport(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_human_qa.1"] = (
        TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20", "background_pilot"] = "pilot20"
    label_count: int = Field(ge=0)
    total_clip_count: int = Field(default=PILOT_TARGET_COUNT, gt=0)
    counts: dict[str, int]
    labels: list[TargetAudioCaptionHumanQALabel]

    @model_validator(mode="after")
    def validate_export(self) -> TargetAudioCaptionHumanQAExport:
        if set(self.counts) != {*QA_LABELS, "UNLABELED"}:
            raise ValueError("target audio caption QA counts are incomplete")
        if self.label_count != len(self.labels):
            raise ValueError("target audio caption QA label count is inconsistent")
        if self.label_count + self.counts["UNLABELED"] != self.total_clip_count:
            raise ValueError("target audio caption QA total is inconsistent")
        if sum(self.counts[label] for label in QA_LABELS) != self.label_count:
            raise ValueError("target audio caption QA decisions do not reconcile")
        return self


@dataclass(frozen=True)
class TargetAudioCaptionBackendResult:
    response: Dots3TargetAudioCaptionResponse
    raw_responses: tuple[str, ...]


class TargetAudioCaptionBackendFailure(RuntimeError):
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


class TargetAudioCaptionBackend(Protocol):
    @property
    def model_identifier(self) -> str: ...

    @property
    def provenance(self) -> TargetAudioCaptionBackendProvenance: ...

    def describe(
        self, job: TargetAudioCaptionJob
    ) -> TargetAudioCaptionBackendResult: ...


@dataclass(frozen=True)
class Dots3TargetAudioCaptionConfig:
    base_url: str
    media_resolver: MediaURLResolver
    api_key: str = "EMPTY"
    served_model_name: str = DEFAULT_DOTS3_MODEL
    checkpoint_id: str = DEFAULT_DOTS3_CHECKPOINT_ID
    timeout_seconds: float = 600.0
    max_tokens: int = 2048

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
            raise ValueError("dots3 target audio caption runtime inputs are required")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("dots3 timeout and token limit must be positive")

    def provenance(self) -> TargetAudioCaptionBackendProvenance:
        values: dict[str, object] = {
            "backend": "vllm",
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "base_url": self.base_url,
            "prompt_version": TARGET_AUDIO_CAPTION_PROMPT_VERSION,
            "input_modality": TARGET_AUDIO_INPUT_MODALITY,
            "separate_audio_sent": False,
            "transcript_supplied": False,
            "donor_media_used": False,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "repair_retries": 1,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
        }
        return TargetAudioCaptionBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


def _canonical_inventory_fingerprint_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _canonical_inventory_fingerprint_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_canonical_inventory_fingerprint_value(item) for item in value]
    return value


def _inventory_fingerprint(
    inventory: TargetAudioCaptionInventory | dict[str, object],
) -> str:
    raw = (
        inventory.model_dump(mode="json")
        if isinstance(inventory, TargetAudioCaptionInventory)
        else dict(inventory)
    )
    raw.pop("inventory_fingerprint", None)
    return _sha256_text(_compact_json(_canonical_inventory_fingerprint_value(raw)))


def _response_issues(
    response: Dots3TargetAudioCaptionResponse,
    job: TargetAudioCaptionJob,
) -> list[ValidationIssue]:
    expected = [item.speaker_cluster_id for item in job.speaker_clusters]
    actual = [item.speaker_cluster_id for item in response.speaker_delivery]
    if actual == expected:
        return []
    if len(actual) != len(set(actual)):
        code = "duplicate_speaker_cluster"
    elif set(actual) - set(expected):
        code = "unknown_speaker_cluster"
    elif set(expected) - set(actual):
        code = "missing_speaker_cluster"
    else:
        code = "speaker_cluster_order_mismatch"
    return [
        ValidationIssue(
            code=code,
            field="speaker_delivery",
            message="speaker delivery must contain every supplied cluster exactly once in order",
        )
    ]


def _model_input(job: TargetAudioCaptionJob) -> dict[str, object]:
    return {
        "target_clip_uid": job.target_clip_uid,
        "speaker_clusters": [
            {
                "speaker_cluster_id": cluster.speaker_cluster_id,
                "active_time_ranges": [
                    item.model_dump(mode="json") for item in cluster.active_time_ranges
                ],
            }
            for cluster in job.speaker_clusters
        ],
    }


def _user_prompt(job: TargetAudioCaptionJob) -> str:
    schema = Dots3TargetAudioCaptionResponse.model_json_schema()
    return (
        "Analyze only sounds actually audible in the attached target video's native "
        "audio track. Do not use visuals to infer sounds. Do not transcribe, quote, "
        "paraphrase, correct, or summarize speech. Do not infer speaker identity, "
        "entity, gender, age, nationality, or timbre. Describe only ambient scene, "
        "background music, non-speech events, acoustic style, and delivery/prosody "
        "for the supplied speaker clusters. Use null or [] when uncertain. The "
        "speaker cluster IDs and time ranges are frozen evidence. Return each cluster "
        "exactly once and no entity_id.\nInput:\n"
        f"{_compact_json(_model_input(job))}\nJSON schema:\n{_compact_json(schema)}"
    )


def _repair_prompt(
    *,
    job: TargetAudioCaptionJob,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        "Repair the previous JSON only. Reinspect the same native target video's "
        "audio when needed. Preserve the audible-only rules and do not emit dialogue "
        "or entity_id. Return every supplied speaker_cluster_id exactly once in order. "
        "Return one compact JSON object only, with no markdown or explanation.\n"
        f"Original request:\n{_user_prompt(job)}\nValidation issues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\nInvalid response:\n"
        f"{invalid_response}"
    )


def _verify_file(path: Path, expected_hash: str, *, code: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise ValueError(f"{code} changed or is unavailable")


class OpenAIDots3TargetAudioCaptionBackend:
    def __init__(
        self,
        config: Dots3TargetAudioCaptionConfig,
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
    def provenance(self) -> TargetAudioCaptionBackendProvenance:
        return self.config.provenance()

    def _request(self, *, job: TargetAudioCaptionJob, prompt: str) -> str:
        video_path = Path(job.target_video_path)
        audio_path = Path(job.target_full_audio_path)
        try:
            _verify_file(video_path, job.target_video_sha256, code="target_video")
            _verify_file(
                audio_path, job.target_full_audio_sha256, code="target_full_audio"
            )
            video_url = self.config.media_resolver.resolve(video_path)
            completion = self.client.chat.completions.create(
                model=self.config.served_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "video_url", "video_url": {"url": video_url}},
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
        except Exception as exc:
            raise TargetAudioCaptionBackendFailure(
                code="dots3_vllm_request_failed",
                reason=f"{type(exc).__name__}: {exc}",
                attempt_count=1,
            ) from exc
        if not response.strip():
            raise TargetAudioCaptionBackendFailure(
                code="dots3_vllm_empty_response",
                reason="dots3/vLLM returned no text",
                raw_responses=[response],
            )
        return response

    def describe(self, job: TargetAudioCaptionJob) -> TargetAudioCaptionBackendResult:
        raw_responses: list[str] = []
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
                raw = self._request(job=job, prompt=prompt)
            except TargetAudioCaptionBackendFailure as exc:
                raise TargetAudioCaptionBackendFailure(
                    code=exc.code,
                    reason=exc.reason,
                    raw_responses=[*raw_responses, *exc.raw_responses],
                    issues=exc.issues,
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            response, issues = parse_structured_json_issues(
                raw, Dots3TargetAudioCaptionResponse
            )
            if response is not None:
                issues = _response_issues(response, job)
            if response is not None and not issues:
                return TargetAudioCaptionBackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                )
        raise TargetAudioCaptionBackendFailure(
            code="structured_output_failed",
            reason="dots3 target audio caption failed closed after one repair",
            raw_responses=raw_responses,
            issues=issues,
        )


def _load_text_usability(
    root: Path,
) -> tuple[
    TextUsabilityInventory,
    list[TextUsabilitySegment],
    TextUsabilitySummary,
    dict[Path, str],
]:
    inventory_path = root / "inventory.json"
    segments_path = root / "segments.jsonl"
    summary_path = root / "summary.json"
    for path in (inventory_path, segments_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen text-usability artifact: {path}")
    inventory = TextUsabilityInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    if inventory.inventory_fingerprint != _text_usability_inventory_fingerprint(
        inventory
    ):
        raise ValueError("text-usability inventory fingerprint is inconsistent")
    records = [
        TextUsabilitySegment.model_validate(row) for row in _read_jsonl(segments_path)
    ]
    summary = TextUsabilitySummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("text-usability summary fingerprint is inconsistent")
    if len(records) != inventory.segment_count or len(records) != summary.segment_count:
        raise ValueError("text-usability segment count is inconsistent")
    return (
        inventory,
        records,
        summary,
        {
            path: _sha256_file(path)
            for path in (inventory_path, segments_path, summary_path)
        },
    )


def _cluster_evidence(jobs: Sequence[ASRV2SegmentJob]) -> list[SpeakerClusterEvidence]:
    grouped: dict[str, list[ASRV2SegmentJob]] = defaultdict(list)
    order: list[str] = []
    for item in jobs:
        if item.speaker_cluster_id not in grouped:
            order.append(item.speaker_cluster_id)
        grouped[item.speaker_cluster_id].append(item)
    clusters = []
    for cluster_id in order:
        rows = grouped[cluster_id]
        entity_ids = {item.entity_id for item in rows}
        if len(entity_ids) != 1:
            raise ValueError("frozen speaker cluster has inconsistent entity binding")
        ranges = [
            SpeakerTimeRange(start_time=item.start_time, end_time=item.end_time)
            for item in rows
        ]
        clusters.append(
            SpeakerClusterEvidence(
                speaker_cluster_id=cluster_id,
                entity_id=next(iter(entity_ids)),
                active_time_ranges=ranges,
            )
        )
    return clusters


def _validate_pilot_selection(
    *, pilot_ids: Sequence[str], production_ids: Sequence[str]
) -> list[str]:
    ordered = list(pilot_ids)
    if len(ordered) != PILOT_TARGET_COUNT or len(ordered) != len(set(ordered)):
        raise ValueError("ASR V2 pilot target set must contain 20 unique clips")
    available = set(production_ids)
    if any(clip_uid not in available for clip_uid in ordered):
        raise ValueError("ASR V2 pilot clip is absent from production ASR V2")
    return ordered


def _validate_background_selection(
    *,
    selection: BackgroundAudioPilotSelection,
    scout_clip_ids: Sequence[str],
    scout_inventory_fingerprint: str,
    diarization_inventory_fingerprint: str,
    audio_evidence_fingerprint: str,
    production_ids: Sequence[str],
) -> list[str]:
    if (
        selection.source_scout_inventory_fingerprint != scout_inventory_fingerprint
        or selection.source_diarization_inventory_fingerprint
        != diarization_inventory_fingerprint
        or selection.source_audio_evidence_fingerprint != audio_evidence_fingerprint
    ):
        raise ValueError("background-audio selection fingerprint is inconsistent")
    ordered = list(selection.selected_clip_ids)
    available = set(production_ids)
    if any(clip_uid not in available for clip_uid in ordered):
        raise ValueError("background-audio selected clip is absent from production")
    selected = set(ordered)
    expected_order = [clip_uid for clip_uid in scout_clip_ids if clip_uid in selected]
    if ordered != expected_order:
        raise ValueError("background-audio selection must preserve scout source order")
    return ordered


def build_target_audio_caption_inventory(
    *, audio_run_root: Path, clip_selection_json: Path | None = None
) -> TargetAudioCaptionInventory:
    root = audio_run_root.expanduser().resolve(strict=True)
    pilot_path = root / "asr_v2_pilot20" / "inventory.json"
    asr_root = root / "production" / "asr_v2"
    text_root = root / "production" / "text_usability"
    audio_root = root / "production" / "audio"
    production = load_asr_v2_inventory(asr_root / "inventory.json")
    if not isinstance(production, ASRV2ProductionInventory):
        raise TypeError("target audio caption requires production ASR V2")
    if production.inventory_fingerprint != _asr_v2_inventory_fingerprint(production):
        raise ValueError("production ASR V2 inventory fingerprint is inconsistent")

    pilot: ASRV2Inventory | None = None
    selection: BackgroundAudioPilotSelection | None = None
    selection_path: Path | None = None
    scout_inventory_fingerprint: str | None = None
    if clip_selection_json is None:
        pilot = load_asr_v2_inventory(pilot_path)
        if not isinstance(pilot, ASRV2Inventory) or pilot.mode != "pilot20":
            raise ValueError(
                "target audio caption requires the ASR V2 pilot20 inventory"
            )
        if pilot.inventory_fingerprint != _asr_v2_inventory_fingerprint(pilot):
            raise ValueError("ASR V2 pilot inventory fingerprint is inconsistent")
    else:
        selection_path = clip_selection_json.expanduser().resolve(strict=True)
        selection = BackgroundAudioPilotSelection.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )

    asr_inventory_path = asr_root / "inventory.json"
    asr_segments_path = asr_root / "segments.jsonl"
    asr_summary_path = asr_root / "summary.json"
    for path in (asr_inventory_path, asr_segments_path, asr_summary_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing frozen production ASR V2 artifact: {path}"
            )
    asr_summary = load_asr_v2_summary(asr_summary_path)
    if asr_summary.inventory_fingerprint != production.inventory_fingerprint:
        raise ValueError("production ASR V2 summary fingerprint is inconsistent")
    asr_records = [
        ASRV2SegmentRecord.model_validate(row) for row in _read_jsonl(asr_segments_path)
    ]
    if len(asr_records) != len(production.jobs):
        raise ValueError("production ASR V2 record count is inconsistent")
    job_fields = tuple(ASRV2SegmentJob.model_fields)
    for job, record in zip(production.jobs, asr_records, strict=True):
        if job != ASRV2SegmentJob.model_validate(
            {field: getattr(record, field) for field in job_fields}
        ):
            raise ValueError("production ASR V2 records differ from inventory")

    text_inventory, text_records, _, text_hashes = _load_text_usability(text_root)
    if (
        text_inventory.source_asr_v2_inventory_fingerprint
        != production.inventory_fingerprint
    ):
        raise ValueError("text-usability source differs from production ASR V2")
    if [
        (item.target_clip_uid, item.segment_id, item.speaker_cluster_id)
        for item in text_records
    ] != [
        (item.target_clip_uid, item.segment_id, item.speaker_cluster_id)
        for item in asr_records
    ]:
        raise ValueError("text-usability identity/order differs from ASR V2")

    diar_root = (root / "production" / "diarization").resolve(strict=True)
    source_diar_root = Path(production.source_diarization_root).resolve(strict=True)
    if source_diar_root != diar_root:
        raise ValueError(
            "production ASR V2 does not reference frozen production diarization"
        )
    diar_inventory_path = Path(production.source_diarization_inventory_path).resolve(
        strict=True
    )
    if diar_inventory_path.parent != diar_root:
        raise ValueError("production diarization inventory path is inconsistent")
    _verify_file(
        diar_inventory_path,
        production.source_diarization_inventory_sha256,
        code="source_diarization_inventory",
    )
    diar_source_paths = {
        "raw_segments": (
            Path(production.source_diarization_raw_segments_path).resolve(strict=True),
            production.source_diarization_raw_segments_sha256,
        ),
        "bound_segments": (
            Path(production.source_diarization_bound_segments_path).resolve(
                strict=True
            ),
            production.source_diarization_bound_segments_sha256,
        ),
        "cluster_bindings": (
            Path(production.source_diarization_cluster_bindings_path).resolve(
                strict=True
            ),
            production.source_diarization_cluster_bindings_sha256,
        ),
        "summary": (
            Path(production.source_diarization_summary_path).resolve(strict=True),
            production.source_diarization_summary_sha256,
        ),
    }
    for name, (path, expected_hash) in diar_source_paths.items():
        if path.parent != diar_root:
            raise ValueError(f"production diarization {name} path is inconsistent")
        _verify_file(path, expected_hash, code=f"source_diarization_{name}")

    targets = {item.target_clip_uid: item for item in production.targets}
    if selection is None:
        assert pilot is not None
        selected_ids = _validate_pilot_selection(
            pilot_ids=[item.target_clip_uid for item in pilot.targets],
            production_ids=list(targets),
        )
        mode = "pilot20"
        selection_mode = "exact_asr_v2_pilot20_order_v1"
    else:
        scout = build_background_audio_scout_inventory(audio_run_root=root)
        scout_inventory_fingerprint = scout.inventory_fingerprint
        selected_ids = _validate_background_selection(
            selection=selection,
            scout_clip_ids=[item.target_clip_uid for item in scout.jobs],
            scout_inventory_fingerprint=scout.inventory_fingerprint,
            diarization_inventory_fingerprint=scout.source_diarization_inventory_fingerprint,
            audio_evidence_fingerprint=scout.source_audio_evidence_fingerprint,
            production_ids=list(targets),
        )
        mode = "background_pilot"
        selection_mode = "manual_background_audio_selection_v1"
    jobs_by_clip: dict[str, list[ASRV2SegmentJob]] = defaultdict(list)
    for item in production.jobs:
        jobs_by_clip[item.target_clip_uid].append(item)
    records_by_clip = Counter(item.target_clip_uid for item in asr_records)
    text_by_clip = Counter(item.target_clip_uid for item in text_records)
    output_jobs: list[TargetAudioCaptionJob] = []
    for clip_uid in selected_ids:
        target = targets.get(clip_uid)
        assert target is not None
        if records_by_clip[clip_uid] != len(jobs_by_clip[clip_uid]) or text_by_clip[
            clip_uid
        ] != len(jobs_by_clip[clip_uid]):
            raise ValueError("frozen per-clip segment evidence is incomplete")
        sidecar_path = (audio_root / "clips" / clip_uid / "audio_binding.json").resolve(
            strict=True
        )
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.status != "ready" or sidecar.evidence is None:
            raise ValueError("target audio binding sidecar must be ready")
        audio = sidecar.evidence.audio
        if audio.status != "ready" or audio.full_audio_path is None:
            raise ValueError("target full audio evidence must be ready")
        if sidecar.clip_uid != clip_uid or Path(sidecar.source_video_path) != Path(
            target.target_video_path
        ):
            raise ValueError("target audio binding identity differs from ASR V2")
        video_path = Path(target.target_video_path).resolve(strict=True)
        audio_path = Path(target.source_audio_path).resolve(strict=True)
        if Path(audio.full_audio_path).resolve(strict=True) != audio_path:
            raise ValueError("target full audio provenance differs from ASR V2")
        audio_hash = _sha256_file(audio_path)
        if audio_hash != target.source_audio_sha256:
            raise ValueError("target full audio hash differs from ASR V2")
        output_jobs.append(
            TargetAudioCaptionJob(
                target_clip_uid=clip_uid,
                target_video_path=str(video_path),
                target_video_sha256=_sha256_file(video_path),
                target_full_audio_path=str(audio_path),
                target_full_audio_sha256=audio_hash,
                target_audio_binding_path=str(sidecar_path),
                target_audio_binding_sha256=_sha256_file(sidecar_path),
                speaker_clusters=_cluster_evidence(jobs_by_clip[clip_uid]),
            )
        )

    values: dict[str, object] = {
        "schema_version": TARGET_AUDIO_CAPTION_INVENTORY_VERSION,
        "mode": mode,
        "source_diarization_root": str(diar_root),
        "source_diarization_inventory_path": str(diar_inventory_path),
        "source_diarization_inventory_sha256": _sha256_file(diar_inventory_path),
        "source_diarization_inventory_fingerprint": production.source_diarization_inventory_fingerprint,
        "source_diarization_raw_segments_path": str(
            diar_source_paths["raw_segments"][0]
        ),
        "source_diarization_raw_segments_sha256": diar_source_paths["raw_segments"][1],
        "source_diarization_bound_segments_path": str(
            diar_source_paths["bound_segments"][0]
        ),
        "source_diarization_bound_segments_sha256": diar_source_paths["bound_segments"][
            1
        ],
        "source_diarization_cluster_bindings_path": str(
            diar_source_paths["cluster_bindings"][0]
        ),
        "source_diarization_cluster_bindings_sha256": diar_source_paths[
            "cluster_bindings"
        ][1],
        "source_diarization_summary_path": str(diar_source_paths["summary"][0]),
        "source_diarization_summary_sha256": diar_source_paths["summary"][1],
        "source_asr_v2_root": str(asr_root.resolve(strict=True)),
        "source_asr_v2_inventory_path": str(asr_inventory_path.resolve(strict=True)),
        "source_asr_v2_inventory_sha256": _sha256_file(asr_inventory_path),
        "source_asr_v2_inventory_fingerprint": production.inventory_fingerprint,
        "source_asr_v2_segments_path": str(asr_segments_path.resolve(strict=True)),
        "source_asr_v2_segments_sha256": _sha256_file(asr_segments_path),
        "source_asr_v2_summary_path": str(asr_summary_path.resolve(strict=True)),
        "source_asr_v2_summary_sha256": _sha256_file(asr_summary_path),
        "source_text_usability_root": str(text_root.resolve(strict=True)),
        "source_text_usability_inventory_path": str(
            (text_root / "inventory.json").resolve(strict=True)
        ),
        "source_text_usability_inventory_sha256": text_hashes[
            text_root / "inventory.json"
        ],
        "source_text_usability_inventory_fingerprint": text_inventory.inventory_fingerprint,
        "source_text_usability_segments_path": str(
            (text_root / "segments.jsonl").resolve(strict=True)
        ),
        "source_text_usability_segments_sha256": text_hashes[
            text_root / "segments.jsonl"
        ],
        "source_text_usability_summary_path": str(
            (text_root / "summary.json").resolve(strict=True)
        ),
        "source_text_usability_summary_sha256": text_hashes[text_root / "summary.json"],
        "source_audio_root": str(audio_root.resolve(strict=True)),
        "selected_target_count": len(selected_ids),
        "selection_mode": selection_mode,
        "bounded_selection_applied": True,
        "parent_quota_applied": False,
        "transcript_supplied_to_model": False,
        "final_renderer_applied": False,
        "jobs": [item.model_dump(mode="json") for item in output_jobs],
    }
    if selection is None:
        assert pilot is not None
        values.update(
            {
                "source_pilot_inventory_path": str(pilot_path.resolve(strict=True)),
                "source_pilot_inventory_sha256": _sha256_file(pilot_path),
                "source_pilot_inventory_fingerprint": pilot.inventory_fingerprint,
            }
        )
    else:
        assert selection_path is not None and scout_inventory_fingerprint is not None
        values.update(
            {
                "source_selection_path": str(selection_path),
                "source_selection_sha256": _sha256_file(selection_path),
                "source_scout_inventory_fingerprint": scout_inventory_fingerprint,
            }
        )
    return TargetAudioCaptionInventory(
        **values,
        inventory_fingerprint=_sha256_text(_compact_json(values)),
    )


def _verify_inventory_sources(inventory: TargetAudioCaptionInventory) -> None:
    path_hash_pairs: list[tuple[str, str]] = [
        (
            inventory.source_diarization_inventory_path,
            inventory.source_diarization_inventory_sha256,
        ),
        (
            inventory.source_diarization_raw_segments_path,
            inventory.source_diarization_raw_segments_sha256,
        ),
        (
            inventory.source_diarization_bound_segments_path,
            inventory.source_diarization_bound_segments_sha256,
        ),
        (
            inventory.source_diarization_cluster_bindings_path,
            inventory.source_diarization_cluster_bindings_sha256,
        ),
        (
            inventory.source_diarization_summary_path,
            inventory.source_diarization_summary_sha256,
        ),
        (
            inventory.source_asr_v2_inventory_path,
            inventory.source_asr_v2_inventory_sha256,
        ),
        (
            inventory.source_asr_v2_segments_path,
            inventory.source_asr_v2_segments_sha256,
        ),
        (inventory.source_asr_v2_summary_path, inventory.source_asr_v2_summary_sha256),
        (
            inventory.source_text_usability_inventory_path,
            inventory.source_text_usability_inventory_sha256,
        ),
        (
            inventory.source_text_usability_segments_path,
            inventory.source_text_usability_segments_sha256,
        ),
        (
            inventory.source_text_usability_summary_path,
            inventory.source_text_usability_summary_sha256,
        ),
    ]
    if inventory.mode == "pilot20":
        assert inventory.source_pilot_inventory_path is not None
        assert inventory.source_pilot_inventory_sha256 is not None
        path_hash_pairs.append(
            (
                inventory.source_pilot_inventory_path,
                inventory.source_pilot_inventory_sha256,
            )
        )
    else:
        assert inventory.source_selection_path is not None
        assert inventory.source_selection_sha256 is not None
        path_hash_pairs.append(
            (inventory.source_selection_path, inventory.source_selection_sha256)
        )
    for path_value, expected in path_hash_pairs:
        _verify_file(Path(path_value), expected, code="frozen_source")
    for job in inventory.jobs:
        _verify_file(
            Path(job.target_video_path), job.target_video_sha256, code="target_video"
        )
        _verify_file(
            Path(job.target_full_audio_path),
            job.target_full_audio_sha256,
            code="target_full_audio",
        )
        _verify_file(
            Path(job.target_audio_binding_path),
            job.target_audio_binding_sha256,
            code="target_audio_binding",
        )


def _sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    return text[0].upper() + text[1:] + ("" if text.endswith((".", "!", "?")) else ".")


def render_audio_prompt_draft(response: Dots3TargetAudioCaptionResponse) -> str:
    sentences: list[str] = []
    if response.ambient_scene is not None:
        sentences.append(_sentence(response.ambient_scene))
    if response.background_music.present is True:
        style = response.background_music.style or "Background music"
        sentences.append(_sentence(f"{style} is audible"))
    elif response.background_music.present is False:
        sentences.append("No background music is audible.")
    if response.sound_events:
        sentences.append(
            _sentence(
                "Audible sound events include " + ", ".join(response.sound_events)
            )
        )
    if response.acoustic_style:
        sentences.append(
            _sentence("The acoustic style is " + ", ".join(response.acoustic_style))
        )
    for delivery in response.speaker_delivery:
        if delivery.delivery_style:
            sentences.append(
                f"{delivery.speaker_cluster_id} speaks in a "
                + ", ".join(delivery.delivery_style)
                + " manner."
            )
    return " ".join(sentences) or "Audio semantics are unknown."


def target_audio_caption_request_fingerprint(
    job: TargetAudioCaptionJob,
    provenance: TargetAudioCaptionBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "prompt_version": TARGET_AUDIO_CAPTION_PROMPT_VERSION,
                "model_input": _model_input(job),
                "target_video_sha256": job.target_video_sha256,
                "target_full_audio_sha256": job.target_full_audio_sha256,
                "target_audio_binding_sha256": job.target_audio_binding_sha256,
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
            }
        )
    )


def _ready_record(
    job: TargetAudioCaptionJob,
    result: TargetAudioCaptionBackendResult,
    provenance: TargetAudioCaptionBackendProvenance,
) -> TargetAudioCaptionRecord:
    issues = _response_issues(result.response, job)
    if issues:
        raise RuntimeError("target audio caption backend returned unvalidated output")
    entity_by_cluster = {
        item.speaker_cluster_id: item.entity_id for item in job.speaker_clusters
    }
    deliveries = [
        TargetSpeakerDelivery(
            speaker_cluster_id=item.speaker_cluster_id,
            entity_id=entity_by_cluster[item.speaker_cluster_id],
            delivery_style=item.delivery_style,
        )
        for item in result.response.speaker_delivery
    ]
    return TargetAudioCaptionRecord(
        target_clip_uid=job.target_clip_uid,
        status="ready",
        ambient_scene=result.response.ambient_scene,
        background_music=result.response.background_music,
        sound_events=result.response.sound_events,
        acoustic_style=result.response.acoustic_style,
        speaker_delivery=deliveries,
        audio_prompt_draft=render_audio_prompt_draft(result.response),
        target_video_path=job.target_video_path,
        target_video_sha256=job.target_video_sha256,
        target_full_audio_path=job.target_full_audio_path,
        target_full_audio_sha256=job.target_full_audio_sha256,
        target_audio_binding_path=job.target_audio_binding_path,
        target_audio_binding_sha256=job.target_audio_binding_sha256,
        backend_provenance=provenance,
        request_fingerprint=target_audio_caption_request_fingerprint(job, provenance),
    )


def _failed_record(
    job: TargetAudioCaptionJob,
    failure: TargetAudioCaptionBackendFailure,
    provenance: TargetAudioCaptionBackendProvenance,
) -> TargetAudioCaptionRecord:
    return TargetAudioCaptionRecord(
        target_clip_uid=job.target_clip_uid,
        status="failed",
        target_video_path=job.target_video_path,
        target_video_sha256=job.target_video_sha256,
        target_full_audio_path=job.target_full_audio_path,
        target_full_audio_sha256=job.target_full_audio_sha256,
        target_audio_binding_path=job.target_audio_binding_path,
        target_audio_binding_sha256=job.target_audio_binding_sha256,
        backend_provenance=provenance,
        request_fingerprint=target_audio_caption_request_fingerprint(job, provenance),
        failure=TargetAudioCaptionFailure(
            code=failure.code,
            reason=failure.reason,
            attempt_count=failure.attempt_count,
            issues=[item.to_dict() for item in failure.issues],
        ),
    )


def _review_html(
    inventory: TargetAudioCaptionInventory,
    records: Sequence[TargetAudioCaptionRecord],
    media_names: dict[str, str],
) -> str:
    by_clip = {item.target_clip_uid: item for item in records}
    cards: list[str] = []
    for job in inventory.jobs:
        record = by_clip[job.target_clip_uid]
        timing = "".join(
            "<li><code>"
            + html.escape(cluster.speaker_cluster_id)
            + "</code>: "
            + ", ".join(
                f"{item.start_time:.3f}-{item.end_time:.3f}s"
                for item in cluster.active_time_ranges
            )
            + "</li>"
            for cluster in job.speaker_clusters
        )
        structured = html.escape(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2)
        )
        radios = " ".join(
            f"<label><input type='radio' name='qa-{html.escape(job.target_clip_uid)}' value='{label}' onchange='saveQA()'>{label}</label>"
            for label in QA_LABELS
        )
        flags = " ".join(
            f"<label><input type='checkbox' data-flag='{flag}' onchange='saveQA()'>{flag}</label>"
            for flag in QA_FLAGS
        )
        cards.append(
            f"<article class='case' data-clip='{html.escape(job.target_clip_uid)}'>"
            f"<h2>{html.escape(job.target_clip_uid)} <span>{record.status}</span></h2>"
            f"<audio controls preload='metadata' src='media/{html.escape(media_names[job.target_clip_uid])}'></audio>"
            f"<h3>Audio prompt draft</h3><p>{html.escape(record.audio_prompt_draft or '[failed]')}</p>"
            f"<h3>Speaker cluster timing</h3><ul>{timing}</ul>"
            f"<details><summary>Structured result</summary><pre>{structured}</pre></details>"
            f"<div class='qa'><b>Decision:</b> {radios}<br><b>Flags:</b> {flags}</div>"
            "</article>"
        )
    order = [item.target_clip_uid for item in inventory.jobs]
    title = (
        "H3 Target Audio Caption Pilot20"
        if inventory.mode == "pilot20"
        else "H3 Target Audio Caption Background Pilot"
    )
    qa_filename = (
        "target_audio_caption_pilot20_human_qa.json"
        if inventory.mode == "pilot20"
        else "target_audio_caption_background_pilot_human_qa.json"
    )
    local_storage_prefix = (
        "h3-target-audio-caption-pilot20-"
        if inventory.mode == "pilot20"
        else "h3-target-audio-caption-background-pilot-"
    )
    target_count = inventory.selected_target_count
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{title} review</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f6;color:#171717}}
header,.case{{max-width:1050px;margin:0 auto 22px;background:white;border:1px solid #bbb;padding:18px}}
h1,h2,h3{{letter-spacing:0}}h2 span{{font-size:14px;color:#555}}audio{{width:100%}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;padding:12px}}
.qa{{margin-top:14px;padding:12px;background:#eef2f5}}.qa label{{margin-right:14px}}
button{{margin-right:10px;padding:8px 12px}}
</style></head><body><header><h1>{title}</h1>
<p>Review audible ambience, music, non-speech events, acoustic style, speaker delivery, and dialogue leakage.</p>
<p id='progress'>Labeled 0 / {target_count}</p>
<button onclick='exportQA()'>Export QA JSON</button><button onclick='clearQA()'>Clear QA labels</button></header>
{"".join(cards)}
<script>
const inventoryFingerprint={json.dumps(inventory.inventory_fingerprint)};
const clipOrder={json.dumps(order)};
const labels=['CORRECT','WRONG','UNCERTAIN'];
const flags={json.dumps(list(QA_FLAGS))};
const keyPrefix={json.dumps(local_storage_prefix)};
function stateFor(clip){{try{{return JSON.parse(localStorage.getItem(keyPrefix+clip)||'null')}}catch(e){{return null}}}}
function restore(){{document.querySelectorAll('.case').forEach(card=>{{const clip=card.dataset.clip;const state=stateFor(clip);if(!state)return;card.querySelectorAll("input[type='radio']").forEach(i=>i.checked=i.value===state.label);card.querySelectorAll("input[type='checkbox']").forEach(i=>i.checked=(state.failure_flags||[]).includes(i.dataset.flag));}});updateCounts();}}
function saveQA(){{document.querySelectorAll('.case').forEach(card=>{{const selected=card.querySelector("input[type='radio']:checked");const selectedFlags=[...card.querySelectorAll("input[type='checkbox']:checked")].map(i=>i.dataset.flag);if(selected){{localStorage.setItem(keyPrefix+card.dataset.clip,JSON.stringify({{label:selected.value,failure_flags:selectedFlags}}));}}else{{localStorage.removeItem(keyPrefix+card.dataset.clip);}}}});updateCounts();}}
function countsAndRows(){{const counts={{CORRECT:0,WRONG:0,UNCERTAIN:0,UNLABELED:0}};const rows=[];clipOrder.forEach(clip=>{{const state=stateFor(clip);if(state&&labels.includes(state.label)){{counts[state.label]++;rows.push({{target_clip_uid:clip,label:state.label,failure_flags:(state.failure_flags||[]).filter(flag=>flags.includes(flag))}});}}else counts.UNLABELED++;}});return {{counts,rows}};}}
function updateCounts(){{const data=countsAndRows();document.getElementById('progress').textContent=`Labeled ${{data.rows.length}} / {target_count} | CORRECT ${{data.counts.CORRECT}} | WRONG ${{data.counts.WRONG}} | UNCERTAIN ${{data.counts.UNCERTAIN}} | UNLABELED ${{data.counts.UNLABELED}}`;}}
function exportQA(){{const data=countsAndRows();const payload={{schema_version:'{TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION}',inventory_fingerprint:inventoryFingerprint,mode:{json.dumps(inventory.mode)},label_count:data.rows.length,total_clip_count:{target_count},counts:data.counts,labels:data.rows}};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={json.dumps(qa_filename)};a.click();URL.revokeObjectURL(a.href);}}
function clearQA(){{if(!confirm('Clear Target Audio Caption QA labels?'))return;clipOrder.forEach(clip=>localStorage.removeItem(keyPrefix+clip));document.querySelectorAll('.qa input').forEach(i=>i.checked=false);updateCounts();}}
restore();
</script></body></html>"""


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"target audio caption output already exists: {destination}"
        )
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def target_audio_caption_output_root(
    audio_run_root: Path,
    *,
    mode: Literal["pilot20", "background_pilot"] = "pilot20",
) -> Path:
    directory = (
        TARGET_AUDIO_CAPTION_OUTPUT_DIRECTORY
        if mode == "pilot20"
        else TARGET_AUDIO_CAPTION_BACKGROUND_OUTPUT_DIRECTORY
    )
    return audio_run_root.expanduser().resolve(strict=False) / directory


def run_target_audio_caption_pilot(
    *,
    inventory: TargetAudioCaptionInventory,
    output_root: Path,
    backend: TargetAudioCaptionBackend,
    overwrite: bool = False,
) -> TargetAudioCaptionSummary:
    if inventory.inventory_fingerprint != _inventory_fingerprint(inventory):
        raise ValueError("target audio caption inventory fingerprint is inconsistent")
    destination = output_root.expanduser().resolve(strict=False)
    source_audio_root = Path(inventory.source_audio_root).resolve(strict=True)
    expected_destination = target_audio_caption_output_root(
        source_audio_root.parents[1], mode=inventory.mode
    )
    if destination != expected_destination:
        raise ValueError("target audio caption pilot output root is fixed")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"target audio caption output already exists: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[TargetAudioCaptionRecord] = []
    failure_counts: Counter[str] = Counter()
    initial_calls = repair_calls = raw_count = 0
    media_names: dict[str, str] = {}
    try:
        temporary.mkdir()
        (temporary / "raw").mkdir()
        (temporary / "media").mkdir()
        for job in inventory.jobs:
            raw_responses: tuple[str, ...] = ()
            failure: TargetAudioCaptionBackendFailure | None = None
            try:
                result = backend.describe(job)
                raw_responses = result.raw_responses
                record = _ready_record(job, result, backend.provenance)
            except TargetAudioCaptionBackendFailure as exc:
                failure = exc
                raw_responses = exc.raw_responses
                failure_counts[exc.code] += 1
                record = _failed_record(job, exc, backend.provenance)
            records.append(record)
            attempts = (
                len(raw_responses)
                if failure is None
                else max(len(raw_responses), failure.attempt_count)
            )
            if attempts:
                initial_calls += 1
                repair_calls += max(0, attempts - 1)
            raw_count += len(raw_responses)
            _write_json(
                temporary / "raw" / f"{job.target_clip_uid}.json",
                {
                    "target_clip_uid": job.target_clip_uid,
                    "status": record.status,
                    "request_fingerprint": record.request_fingerprint,
                    "raw_responses": list(raw_responses),
                    "failure": None
                    if record.failure is None
                    else record.failure.model_dump(mode="json"),
                },
            )
            audio_source = Path(job.target_full_audio_path)
            media_name = (
                f"{job.target_clip_uid}{audio_source.suffix.lower() or '.audio'}"
            )
            (temporary / "media" / media_name).symlink_to(audio_source)
            media_names[job.target_clip_uid] = media_name

        _verify_inventory_sources(inventory)
        status_counts = Counter(item.status for item in records)
        summary = TargetAudioCaptionSummary(
            mode=inventory.mode,
            inventory_fingerprint=inventory.inventory_fingerprint,
            model_identifier=backend.model_identifier,
            backend_provenance=backend.provenance,
            target_clip_count=inventory.selected_target_count,
            ready_count=status_counts["ready"],
            failed_count=status_counts["failed"],
            initial_call_count=initial_calls,
            repair_call_count=repair_calls,
            raw_response_count=raw_count,
            failure_reason_counts=dict(sorted(failure_counts.items())),
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        (temporary / "records.jsonl").write_text(
            "".join(
                _compact_json(item.model_dump(mode="json")) + "\n" for item in records
            ),
            encoding="utf-8",
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "report.html").write_text(
            _review_html(inventory, records, media_names), encoding="utf-8"
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
