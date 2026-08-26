from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import threading
import uuid
import wave
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from openai import OpenAI
from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_audio_production import JEAInPair
from r2v_data_v2.h3.jea_diarization import JEAReadableDiarizationSegment
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption_contract import (
    SpeakerClusterEvidence,
    SpeakerTimeRange,
    TargetAudioCaptionResponse,
    TargetSpeakerDelivery,
)
from r2v_data_v2.structured_output import (
    ValidationIssue,
    parse_structured_json_issues,
)

JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION = "r2v.h3.target_audio_caption.4"
JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION = (
    "r2v.h3.target_audio_caption_inventory.2"
)
JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION = "r2v.h3.target_audio_caption_summary.4"
JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION = (
    "r2v.h3.target_audio_caption_human_qa.2"
)
JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION = "h3_target_audio_caption_v6"
JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION = "h3_target_audio_caption_v5"
JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION = (
    JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION
)
DEFAULT_DOTS3_MODEL = "dots3-note-prev"
DEFAULT_DOTS3_CHECKPOINT_ID = "/mnt/workspace/public/pretrained/dots3-note-prev"
DEFAULT_QWEN3_OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_QWEN3_OMNI_CHECKPOINT_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS = 0.10

BackendFamily = Literal["dots3", "qwen3_omni"]
SemanticSource = Literal[
    "primary_v6",
    "fallback_v5",
    "primary_v6_all_null_confirmed",
    "primary_v6_fallback_failed",
]
InputModality = Literal[
    "native_target_video_with_embedded_audio",
    "target_video_plus_canonical_full_audio",
]
QA_LABELS = ("CORRECT", "WRONG", "UNCERTAIN")
QA_FLAGS = (
    "hallucinated_music",
    "hallucinated_sound_event",
    "wrong_ambient_scene",
    "wrong_speaker_style",
    "dialogue_leakage",
    "other",
)

SYSTEM_PROMPT = """You analyze AUDIO SEMANTICS for one target clip for
audio-generation conditioning.

AUDIBLE EVIDENCE IS THE ONLY SOURCE OF TRUTH. Visual evidence may only help
disambiguate the source or category of a sound that is already genuinely audible.
Never infer that a sound exists merely because an action, object, person, event,
expression, gesture, or scene is visible. Do not describe visual content itself.
If a visually apparent event is silent, omit it.

Analyze the audio in two independent passes before producing JSON.

PASS 1 - COMPLETE NON-DIALOGUE SOUNDSCAPE
Independently inspect the entire clip for every meaningful audible sound other
than spoken linguistic content. Continue even during speech and do not stop after
identifying speaker delivery. background_audio_prompt is one short English
audio-generation description of ALL meaningful audible non-dialogue sound, not
merely distant background ambience.

When genuinely audible, include foreground or background music and musical
accompaniment; ambience and room tone; sound effects and object sounds; traffic,
crowds, footsteps, doors, impacts, machinery, nature, applause, and similar
sounds; and human non-speech vocalizations such as laughter, chuckling, sighing,
gasping, coughing, crying, sobbing, breathing, or cheering. Include a sound when
it overlaps speech, comes from the current speaker, is foreground, is quieter
than dialogue, is partially masked by speech, or occurs only briefly. Do not
transcribe or paraphrase words spoken during a vocalization.

Return background_audio_prompt as null only when no meaningful non-dialogue sound
is reliably audible anywhere. Before returning null, independently verify the
whole clip for music, ambience, sound effects or object sounds, and human
non-speech vocalizations.

PASS 2 - SPEAKER DELIVERY
Separately analyze each supplied speaker cluster for speech delivery and prosody
only, using its exact speaker_cluster_id. Describe concise, generation-useful
audible traits when supported: emotion expressed through voice, pace, energy,
loudness, pitch tendency, rhythm, articulation, hesitation, pauses, whispering,
shouting, questioning, commanding, or similar delivery. Never infer delivery from
facial expression, gesture, pose, scene semantics, or other visual evidence.

Distinct non-speech events such as laughter, coughing, gasping, crying, or sighing
belong in background_audio_prompt when audible, even when produced by the current
speaker. speaker_delivery describes speech prosody and is not the only location
for those events.

Never transcribe, quote, paraphrase, correct, or summarize dialogue. Never infer
speaker, entity, or subject identity, gender, age, nationality, intrinsic voice
identity, or timbre. Do not emit entity_id. Return every supplied
speaker_cluster_id exactly once, in supplied order, and no unknown cluster ID.
Prefer concise generation-useful descriptions without speculative details.

Return exactly one compact JSON object matching the supplied schema, with no
markdown, explanation, reasoning, or extra fields."""

V5_FALLBACK_SYSTEM_PROMPT = """You analyze AUDIO SEMANTICS for one target clip.
Use audible evidence as the source of truth. Visual evidence may help disambiguate
a sound that is genuinely audible, but never invent a sound merely because a
visible action or object could produce it.

Return only:
- background_audio_prompt: one short English description of meaningful non-speech
  audio actually audible, or null when there is none. Actively check for background
  music, ambience, sound effects, traffic, crowds, footsteps, doors, machinery,
  nature, and other non-speech audio. Include faint or partially masked background
  audio when it is genuinely audible.
- speaker_delivery: one entry per supplied speaker cluster, using its exact
  speaker_cluster_id and a concise, generation-useful delivery/prosody condition
  or null. Focus on audible emotion, pace, energy, loudness, pitch tendency,
  rhythm, hesitation, pauses, whispering, shouting, questioning, commanding, and
  similarly useful delivery traits when supported by the audio.

Never transcribe, quote, paraphrase, correct, or summarize dialogue. Never infer
speaker, entity, or subject identity, gender, age, nationality, intrinsic voice
identity, or timbre. Do not emit entity_id. Return every supplied
speaker_cluster_id exactly once, in supplied order, and no unknown cluster ID.

Return exactly one compact JSON object matching the supplied schema, with no
markdown or explanation."""


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: SchemaModel | dict[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            _compact_json(item.model_dump(mode="json")) + "\n" for item in values
        ),
        encoding="utf-8",
    )


class JEATargetAudioCaptionBackendProvenance(SchemaModel):
    backend: Literal["vllm"] = "vllm"
    backend_family: BackendFamily
    served_model_name: str
    checkpoint_id: str
    base_url: str
    input_modality: InputModality
    media_mode: Literal["file", "http"]
    media_root: str
    media_base_url: str | None = None
    output_modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"])
    prompt_version: Literal["h3_target_audio_caption_v6"] = (
        JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION
    )
    temperature: Literal[0.0] = 0.0
    max_tokens: int = Field(gt=0)
    repair_retries: Literal[1] = 1
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> JEATargetAudioCaptionBackendProvenance:
        if any(
            not value.strip()
            for value in (
                self.served_model_name,
                self.checkpoint_id,
                self.base_url,
                self.media_root,
            )
        ):
            raise ValueError("audio caption backend provenance is incomplete")
        if (self.media_mode == "http") != (self.media_base_url is not None):
            raise ValueError("audio caption HTTP media provenance is inconsistent")
        if self.media_base_url is not None and not self.media_base_url.strip():
            raise ValueError("audio caption HTTP media base URL is empty")
        expected_modality = (
            "native_target_video_with_embedded_audio"
            if self.backend_family == "dots3"
            else "target_video_plus_canonical_full_audio"
        )
        if self.input_modality != expected_modality:
            raise ValueError("audio caption backend modality is inconsistent")
        if self.output_modalities != ["text"]:
            raise ValueError("audio caption backend must request text output only")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("audio caption backend fingerprint is invalid")
        return self


class JEATargetAudioCaptionJob(SchemaModel):
    target_clip_uid: str
    clip_display_path: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    target_audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_clusters: list[SpeakerClusterEvidence]

    @model_validator(mode="after")
    def validate_job(self) -> JEATargetAudioCaptionJob:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.target_clip_uid)
            is None
            or not self.clip_display_path.strip()
        ):
            raise ValueError("audio caption target identity must not be empty")
        cluster_ids = [item.speaker_cluster_id for item in self.speaker_clusters]
        if not cluster_ids or len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("audio caption speaker clusters must be non-empty and unique")
        return self


class JEATargetAudioCaptionInventory(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_inventory.2"] = (
        JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION
    )
    source_audio_production_root: str
    source_pairs_path: str
    source_pairs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_readable_segments_path: str
    source_readable_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_qwen3_asr_segments_path: str
    source_qwen3_asr_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_clip_count: int = Field(gt=0)
    readable_segment_count: int = Field(gt=0)
    qwen3_asr_segment_count: int = Field(gt=0)
    transcript_supplied_to_model: Literal[False] = False
    entity_id_supplied_to_model: Literal[False] = False
    donor_media_used: Literal[False] = False
    jobs: list[JEATargetAudioCaptionJob]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> JEATargetAudioCaptionInventory:
        if self.target_clip_count != len(self.jobs):
            raise ValueError("audio caption target count is inconsistent")
        if self.readable_segment_count != self.qwen3_asr_segment_count:
            raise ValueError("audio caption source segment counts differ")
        if self.readable_segment_count != sum(
            len(cluster.active_time_ranges)
            for job in self.jobs
            for cluster in job.speaker_clusters
        ):
            raise ValueError("audio caption readable segment count is inconsistent")
        clip_ids = [item.target_clip_uid for item in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("audio caption target clips must be unique")
        if self.inventory_fingerprint != _inventory_fingerprint(self):
            raise ValueError("audio caption inventory fingerprint is invalid")
        return self


class JEATargetAudioCaptionFailure(SchemaModel):
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    attempt_count: int = Field(ge=0)
    issues: list[dict[str, str | None]] = Field(default_factory=list)


class JEATargetAudioCaptionRecord(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption.4"] = (
        JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION
    )
    target_clip_uid: str
    clip_display_path: str
    status: Literal["ready", "failed"]
    background_audio_prompt: str | None = None
    speaker_delivery: list[TargetSpeakerDelivery] = Field(default_factory=list)
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    target_audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_modality: InputModality
    backend_provenance: JEATargetAudioCaptionBackendProvenance
    repair_count: int = Field(ge=0, le=1)
    semantic_source: SemanticSource | None
    semantic_fallback_attempted: bool
    semantic_fallback_prompt_version: (
        Literal["h3_target_audio_caption_v5"] | None
    )
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    failure: JEATargetAudioCaptionFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> JEATargetAudioCaptionRecord:
        if self.input_modality != self.backend_provenance.input_modality:
            raise ValueError("audio caption record modality is inconsistent")
        if self.status == "ready":
            cluster_ids = [item.speaker_cluster_id for item in self.speaker_delivery]
            if (
                self.failure is not None
                or self.semantic_source is None
                or not cluster_ids
                or len(cluster_ids) != len(set(cluster_ids))
            ):
                raise ValueError("ready audio caption semantics are inconsistent")
        elif self.failure is None or self.background_audio_prompt is not None or bool(
            self.speaker_delivery
        ):
            raise ValueError("failed audio caption cannot publish semantics")
        elif self.semantic_source is not None:
            raise ValueError("failed audio caption cannot publish semantic provenance")
        if self.semantic_fallback_attempted != (
            self.semantic_fallback_prompt_version
            == JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
        ):
            raise ValueError("audio caption semantic fallback provenance is inconsistent")
        fallback_sources = {
            "fallback_v5",
            "primary_v6_all_null_confirmed",
            "primary_v6_fallback_failed",
        }
        if (self.semantic_source in fallback_sources) != self.semantic_fallback_attempted:
            raise ValueError("audio caption semantic source is inconsistent")
        all_semantic_null = self.background_audio_prompt is None and all(
            item.delivery_style is None for item in self.speaker_delivery
        )
        if self.semantic_source == "fallback_v5" and all_semantic_null:
            raise ValueError("recovered V5 semantics cannot be all-null")
        if self.semantic_source in {
            "primary_v6_all_null_confirmed",
            "primary_v6_fallback_failed",
        } and not all_semantic_null:
            raise ValueError("retained V6 fallback semantics must be all-null")
        if (
            self.semantic_source == "primary_v6"
            and self.backend_provenance.backend_family == "qwen3_omni"
            and all_semantic_null
        ):
            raise ValueError("Qwen all-null semantics require fallback provenance")
        return self


class JEATargetAudioCaptionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_summary.4"] = (
        JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: JEATargetAudioCaptionBackendProvenance
    target_clip_count: int = Field(gt=0)
    max_concurrency: int = Field(ge=1)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    initial_call_count: int = Field(ge=0)
    repair_call_count: int = Field(ge=0)
    semantic_fallback_trigger_count: int = Field(ge=0)
    semantic_fallback_initial_call_count: int = Field(ge=0)
    semantic_fallback_repair_call_count: int = Field(ge=0)
    semantic_fallback_recovered_count: int = Field(ge=0)
    semantic_fallback_still_all_null_count: int = Field(ge=0)
    semantic_fallback_failed_count: int = Field(ge=0)
    raw_response_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    transcript_supplied_to_model: Literal[False] = False
    entity_id_supplied_to_model: Literal[False] = False
    donor_media_used: Literal[False] = False
    human_qa_pending: Literal[True] = True

    @model_validator(mode="after")
    def validate_counts(self) -> JEATargetAudioCaptionSummary:
        if self.ready_count + self.failed_count != self.target_clip_count:
            raise ValueError("audio caption summary counts do not reconcile")
        if self.initial_call_count > self.target_clip_count:
            raise ValueError("audio caption initial calls exceed target count")
        if self.repair_call_count > self.initial_call_count:
            raise ValueError("audio caption repair calls exceed initial calls")
        if (
            self.semantic_fallback_trigger_count
            != self.semantic_fallback_initial_call_count
        ):
            raise ValueError("audio caption semantic fallback calls do not reconcile")
        if (
            self.semantic_fallback_trigger_count
            != self.semantic_fallback_recovered_count
            + self.semantic_fallback_still_all_null_count
            + self.semantic_fallback_failed_count
        ):
            raise ValueError("audio caption semantic fallback outcomes do not reconcile")
        if (
            self.semantic_fallback_repair_call_count
            > self.semantic_fallback_initial_call_count
        ):
            raise ValueError("audio caption semantic fallback repairs exceed calls")
        if (
            self.backend_provenance.backend_family == "dots3"
            and self.semantic_fallback_trigger_count != 0
        ):
            raise ValueError("Dots3 cannot use Qwen semantic fallback")
        return self


class JEATargetAudioCaptionHumanQALabel(SchemaModel):
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
    review_note: str = ""


class JEATargetAudioCaptionHumanQAExport(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_human_qa.2"] = (
        JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: JEATargetAudioCaptionBackendProvenance
    label_count: int = Field(ge=0)
    total_clip_count: int = Field(gt=0)
    counts: dict[str, int]
    labels: list[JEATargetAudioCaptionHumanQALabel]

    @model_validator(mode="after")
    def validate_export(self) -> JEATargetAudioCaptionHumanQAExport:
        if set(self.counts) != {*QA_LABELS, "UNLABELED"}:
            raise ValueError("audio caption QA counts are incomplete")
        if self.label_count != len(self.labels):
            raise ValueError("audio caption QA label count is inconsistent")
        if self.label_count + self.counts["UNLABELED"] != self.total_clip_count:
            raise ValueError("audio caption QA total is inconsistent")
        if sum(self.counts[label] for label in QA_LABELS) != self.label_count:
            raise ValueError("audio caption QA decisions do not reconcile")
        return self


def _inventory_fingerprint(
    inventory: JEATargetAudioCaptionInventory | dict[str, object],
) -> str:
    values = (
        inventory.model_dump(mode="json")
        if isinstance(inventory, JEATargetAudioCaptionInventory)
        else dict(inventory)
    )
    values.pop("inventory_fingerprint", None)
    return _sha256_text(_compact_json(values))


def _segment_identity(row: JEAReadableDiarizationSegment | Qwen3ASRSegment) -> tuple[str, str]:
    return row.clip_uid, row.segment_id


def _unique_by_segment(
    rows: Sequence[JEAReadableDiarizationSegment | Qwen3ASRSegment],
    *,
    source_name: str,
) -> dict[tuple[str, str], JEAReadableDiarizationSegment | Qwen3ASRSegment]:
    result: dict[tuple[str, str], JEAReadableDiarizationSegment | Qwen3ASRSegment] = {}
    for row in rows:
        identity = _segment_identity(row)
        if identity in result:
            raise ValueError(f"duplicate {source_name} segment identity")
        result[identity] = row
    return result


def _reconcile_segments(
    readable: Sequence[JEAReadableDiarizationSegment],
    asr_rows: Sequence[Qwen3ASRSegment],
) -> None:
    readable_by_id = _unique_by_segment(readable, source_name="readable DiariZen")
    asr_by_id = _unique_by_segment(asr_rows, source_name="Qwen3 ASR")
    if set(readable_by_id) != set(asr_by_id):
        raise ValueError("Qwen3 ASR and readable DiariZen segment identities differ")
    fields = (
        "clip_uid",
        "clip_display_path",
        "media_collection_relpath",
        "media_collection_name",
        "episode_name",
        "clip_name",
        "shard_id",
        "segment_id",
        "speaker_cluster_id",
        "entity_id",
        "entity_occurrence_id",
        "source_audio_path",
        "source_start_sample",
        "source_end_sample",
        "source_sample_rate_hz",
        "start_time",
        "end_time",
    )
    for identity, readable_row in readable_by_id.items():
        asr_row = asr_by_id[identity]
        if any(
            getattr(readable_row, field) != getattr(asr_row, field)
            for field in fields
        ):
            raise ValueError(
                "Qwen3 ASR and readable DiariZen segment evidence differs: "
                f"{identity[0]}/{identity[1]}"
            )


def _resolved_file(path_value: str, *, field_name: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{field_name} is missing: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file")
    return resolved


@dataclass(frozen=True)
class _AudioTimeline:
    duration_seconds: float
    sample_rate_hz: int | None
    sample_count: int | None


def _audio_timeline(path: Path, *, field_name: str) -> _AudioTimeline:
    try:
        with wave.open(str(path), "rb") as source:
            sample_rate_hz = source.getframerate()
            sample_count = source.getnframes()
        if sample_rate_hz <= 0 or sample_count <= 0:
            raise ValueError(f"{field_name} has an invalid PCM timeline")
        return _AudioTimeline(
            duration_seconds=sample_count / sample_rate_hz,
            sample_rate_hz=sample_rate_hz,
            sample_count=sample_count,
        )
    except (EOFError, OSError, wave.Error):
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError(f"cannot inspect {field_name}: ffprobe is unavailable")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot inspect {field_name} timeline")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        stream = streams[0] if streams else {}
        duration_value = stream.get("duration") or payload.get("format", {}).get(
            "duration"
        )
        duration_seconds = float(duration_value)
        sample_rate_value = stream.get("sample_rate")
        sample_rate_hz = (
            None if sample_rate_value is None else int(sample_rate_value)
        )
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {field_name} timeline") from exc
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0
        or (sample_rate_hz is not None and sample_rate_hz <= 0)
    ):
        raise ValueError(f"{field_name} has an invalid timeline")
    return _AudioTimeline(
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        sample_count=None,
    )


def _validate_audio_timelines(
    *,
    clip_rows: Sequence[JEAReadableDiarizationSegment],
    readable_audio_path: Path,
    canonical_audio_path: Path,
) -> None:
    readable_timeline = _audio_timeline(
        readable_audio_path,
        field_name="readable source audio",
    )
    canonical_timeline = _audio_timeline(
        canonical_audio_path,
        field_name="target full audio",
    )
    if (
        abs(
            readable_timeline.duration_seconds
            - canonical_timeline.duration_seconds
        )
        > AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
    ):
        raise ValueError("readable and canonical full-audio timelines differ")
    if (
        readable_timeline.sample_count is not None
        and canonical_timeline.sample_count is not None
        and readable_timeline.sample_rate_hz == canonical_timeline.sample_rate_hz
    ):
        allowed_samples = math.ceil(
            AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
            * readable_timeline.sample_rate_hz
        )
        if (
            abs(readable_timeline.sample_count - canonical_timeline.sample_count)
            > allowed_samples
        ):
            raise ValueError("readable and canonical full-audio sample counts differ")
    for row in clip_rows:
        if row.source_end_sample <= row.source_start_sample:
            raise ValueError("readable DiariZen segment sample range is invalid")
        if (
            readable_timeline.sample_rate_hz is not None
            and row.source_sample_rate_hz != readable_timeline.sample_rate_hz
        ):
            raise ValueError("readable DiariZen sample rate differs from source audio")
        if (
            readable_timeline.sample_count is not None
            and row.source_end_sample > readable_timeline.sample_count
        ):
            raise ValueError("readable DiariZen segment exceeds source audio samples")
        if row.end_time > (
            readable_timeline.duration_seconds
            + AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError("readable DiariZen segment exceeds source audio timeline")
        if row.end_time > (
            canonical_timeline.duration_seconds
            + AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError("readable DiariZen segment exceeds canonical audio timeline")


def build_jea_target_audio_caption_inventory(
    *,
    audio_production_root: Path,
) -> JEATargetAudioCaptionInventory:
    root = audio_production_root.expanduser().resolve(strict=True)
    pairs_path = (root / "pairs/in_pairs.jsonl").resolve(strict=True)
    readable_path = (root / "diarization/readable_segments.jsonl").resolve(
        strict=True
    )
    asr_path = (root / "asr/segments.jsonl").resolve(strict=True)
    pairs = [JEAInPair.model_validate(row) for row in _read_jsonl(pairs_path)]
    readable = [
        JEAReadableDiarizationSegment.model_validate(row)
        for row in _read_jsonl(readable_path)
    ]
    asr_rows = [Qwen3ASRSegment.model_validate(row) for row in _read_jsonl(asr_path)]
    if not pairs or not readable or not asr_rows:
        raise ValueError("audio caption production inputs must be non-empty")
    _reconcile_segments(readable, asr_rows)

    pair_by_clip: dict[str, JEAInPair] = {}
    for pair in pairs:
        if pair.target_clip_uid in pair_by_clip:
            raise ValueError("duplicate JEA in-pair target clip")
        pair_by_clip[pair.target_clip_uid] = pair
    readable_by_clip: dict[str, list[JEAReadableDiarizationSegment]] = defaultdict(list)
    for row in readable:
        readable_by_clip[row.clip_uid].append(row)
    if set(pair_by_clip) != set(readable_by_clip):
        raise ValueError("JEA in-pairs and readable DiariZen target clips differ")

    jobs: list[JEATargetAudioCaptionJob] = []
    for pair in pairs:
        clip_rows = sorted(
            readable_by_clip[pair.target_clip_uid],
            key=lambda item: (item.source_start_sample, item.segment_id),
        )
        if any(item.clip_display_path != pair.target_clip_display_path for item in clip_rows):
            raise ValueError("JEA pair and readable DiariZen clip paths differ")
        video_path = _resolved_file(pair.target_video_path, field_name="target video")
        audio_path = _resolved_file(
            pair.target_full_audio_path, field_name="target full audio"
        )
        binding_path = _resolved_file(
            pair.target_audio_binding_path, field_name="target audio binding"
        )
        readable_audio_values = {item.source_audio_path for item in clip_rows}
        if len(readable_audio_values) != 1:
            raise ValueError("readable DiariZen source audio paths differ within clip")
        readable_audio_path = _resolved_file(
            next(iter(readable_audio_values)),
            field_name="readable source audio",
        )
        _validate_audio_timelines(
            clip_rows=clip_rows,
            readable_audio_path=readable_audio_path,
            canonical_audio_path=audio_path,
        )

        rows_by_cluster: dict[str, list[JEAReadableDiarizationSegment]] = defaultdict(list)
        cluster_order: list[str] = []
        for row in clip_rows:
            if row.speaker_cluster_id not in rows_by_cluster:
                cluster_order.append(row.speaker_cluster_id)
            rows_by_cluster[row.speaker_cluster_id].append(row)
        clusters: list[SpeakerClusterEvidence] = []
        for cluster_id in cluster_order:
            cluster_rows = rows_by_cluster[cluster_id]
            bound_ids = {item.entity_id for item in cluster_rows if item.entity_id is not None}
            if len(bound_ids) > 1:
                raise ValueError("speaker cluster resolves to conflicting entity IDs")
            clusters.append(
                SpeakerClusterEvidence(
                    speaker_cluster_id=cluster_id,
                    entity_id=next(iter(bound_ids), None),
                    active_time_ranges=[
                        SpeakerTimeRange(
                            start_time=item.start_time,
                            end_time=item.end_time,
                        )
                        for item in cluster_rows
                    ],
                )
            )
        jobs.append(
            JEATargetAudioCaptionJob(
                target_clip_uid=pair.target_clip_uid,
                clip_display_path=pair.target_clip_display_path,
                target_video_path=str(video_path),
                target_video_sha256=_sha256_file(video_path),
                target_full_audio_path=str(audio_path),
                target_full_audio_sha256=_sha256_file(audio_path),
                target_audio_binding_path=str(binding_path),
                target_audio_binding_sha256=_sha256_file(binding_path),
                speaker_clusters=clusters,
            )
        )
    values: dict[str, object] = {
        "schema_version": JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION,
        "source_audio_production_root": str(root),
        "source_pairs_path": str(pairs_path),
        "source_pairs_sha256": _sha256_file(pairs_path),
        "source_readable_segments_path": str(readable_path),
        "source_readable_segments_sha256": _sha256_file(readable_path),
        "source_qwen3_asr_segments_path": str(asr_path),
        "source_qwen3_asr_segments_sha256": _sha256_file(asr_path),
        "target_clip_count": len(jobs),
        "readable_segment_count": len(readable),
        "qwen3_asr_segment_count": len(asr_rows),
        "transcript_supplied_to_model": False,
        "entity_id_supplied_to_model": False,
        "donor_media_used": False,
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    return JEATargetAudioCaptionInventory(
        **values,
        inventory_fingerprint=_inventory_fingerprint(values),
    )


@dataclass(frozen=True)
class JEATargetAudioCaptionConfig:
    backend_family: BackendFamily
    base_url: str
    api_key: str
    served_model_name: str
    checkpoint_id: str
    media_resolver: MediaURLResolver
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
            raise ValueError("audio caption backend endpoint and model are required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("audio caption backend timeout must be positive")
        if self.max_tokens <= 0:
            raise ValueError("audio caption backend max tokens must be positive")

    @property
    def input_modality(self) -> InputModality:
        return (
            "native_target_video_with_embedded_audio"
            if self.backend_family == "dots3"
            else "target_video_plus_canonical_full_audio"
        )

    def provenance(self) -> JEATargetAudioCaptionBackendProvenance:
        values = {
            "backend": "vllm",
            "backend_family": self.backend_family,
            "served_model_name": self.served_model_name,
            "checkpoint_id": self.checkpoint_id,
            "base_url": self.base_url,
            "input_modality": self.input_modality,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "output_modalities": ["text"],
            "prompt_version": JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "repair_retries": 1,
        }
        return JEATargetAudioCaptionBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


@dataclass(frozen=True)
class JEATargetAudioCaptionBackendResult:
    response: TargetAudioCaptionResponse
    raw_responses: tuple[str, ...]


class JEATargetAudioCaptionBackendFailure(RuntimeError):
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


class JEATargetAudioCaptionBackend(Protocol):
    @property
    def provenance(self) -> JEATargetAudioCaptionBackendProvenance: ...

    def describe(
        self, job: JEATargetAudioCaptionJob
    ) -> JEATargetAudioCaptionBackendResult: ...

    def describe_semantic_fallback(
        self, job: JEATargetAudioCaptionJob
    ) -> JEATargetAudioCaptionBackendResult: ...


def _model_input(job: JEATargetAudioCaptionJob) -> dict[str, object]:
    return {
        "speaker_clusters": [
            {
                "speaker_cluster_id": cluster.speaker_cluster_id,
                "active_time_ranges": [
                    item.model_dump(mode="json") for item in cluster.active_time_ranges
                ],
            }
            for cluster in job.speaker_clusters
        ]
    }


def _is_all_semantic_null(response: TargetAudioCaptionResponse) -> bool:
    return response.background_audio_prompt is None and all(
        item.delivery_style is None for item in response.speaker_delivery
    )


def _user_prompt(job: JEATargetAudioCaptionJob) -> str:
    return (
        "Analyze the complete audible soundscape of the attached target media. "
        "First, independently scan the whole clip for all meaningful non-dialogue "
        "audio, including music, ambience, sound effects or object sounds, and "
        "human non-speech vocalizations such as laughter, even when they overlap "
        "speech or come from the current speaker. Second, separately describe "
        "speech delivery and prosody for each supplied speaker cluster. Use audible "
        "evidence only; visual content cannot establish that a sound exists. Do not "
        "transcribe speech or infer identity. Return every speaker_cluster_id "
        "exactly once in supplied order and no entity_id.\nInput:\n"
        f"{_compact_json(_model_input(job))}\nJSON schema:\n"
        f"{_compact_json(TargetAudioCaptionResponse.model_json_schema())}"
    )


def _v5_fallback_user_prompt(job: JEATargetAudioCaptionJob) -> str:
    return (
        "Analyze only sounds actually audible in the attached target media. Do not "
        "transcribe, quote, paraphrase, correct, or summarize speech. Do not infer "
        "speaker, entity, or subject identity. Return one short English "
        "background_audio_prompt for meaningful non-speech audio, or null, and "
        "concise nullable delivery/prosody for each supplied speaker cluster. Return "
        "every cluster exactly once in the supplied order and no entity_id.\nInput:\n"
        f"{_compact_json(_model_input(job))}\nJSON schema:\n"
        f"{_compact_json(TargetAudioCaptionResponse.model_json_schema())}"
    )


def _response_issues(
    response: TargetAudioCaptionResponse,
    job: JEATargetAudioCaptionJob,
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


def _repair_prompt(
    *,
    job: JEATargetAudioCaptionJob,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        "Repair the previous JSON only. Reinspect the same attached audio when "
        "needed. Preserve the original two-pass audible-only analysis. Do not infer "
        "sound from visual content. Do not emit dialogue or entity_id. Return every "
        "speaker_cluster_id exactly once in supplied order. Return one compact JSON "
        "object only.\nOriginal request:\n"
        f"{_user_prompt(job)}\nValidation issues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\nInvalid response:\n"
        f"{invalid_response}"
    )


def _v5_fallback_repair_prompt(
    *,
    job: JEATargetAudioCaptionJob,
    invalid_response: str,
    issues: Sequence[ValidationIssue],
) -> str:
    return (
        "Repair the previous JSON only. Reinspect the same attached audio when "
        "needed. Follow the original audible-only policy. Do not emit dialogue or "
        "entity_id. Return every speaker_cluster_id exactly once in supplied order. "
        "Return one compact JSON object only.\nOriginal request:\n"
        f"{_v5_fallback_user_prompt(job)}\nValidation issues:\n"
        f"{_compact_json([item.to_dict() for item in issues])}\nInvalid response:\n"
        f"{invalid_response}"
    )


def _verify_file(path: Path, expected_hash: str, *, field_name: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected_hash:
        raise ValueError(f"{field_name} changed or is unavailable")


class OpenAIJEATargetAudioCaptionBackend:
    def __init__(
        self,
        config: JEATargetAudioCaptionConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._injected_client = client
        self._thread_local = threading.local()

    @property
    def provenance(self) -> JEATargetAudioCaptionBackendProvenance:
        return self.config.provenance()

    def _client_for_current_thread(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
            self._thread_local.client = client
        return client

    def _request(
        self,
        *,
        job: JEATargetAudioCaptionJob,
        system_prompt: str,
        prompt: str,
    ) -> str:
        video_path = Path(job.target_video_path)
        audio_path = Path(job.target_full_audio_path)
        try:
            _verify_file(video_path, job.target_video_sha256, field_name="target video")
            _verify_file(
                audio_path,
                job.target_full_audio_sha256,
                field_name="target full audio",
            )
            video_item = {
                "type": "video_url",
                "video_url": {
                    "url": self.config.media_resolver.resolve(video_path)
                },
            }
            if self.config.backend_family == "dots3":
                media_items = [video_item]
            else:
                media_items = [
                    video_item,
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": self.config.media_resolver.resolve(audio_path)
                        },
                    },
                ]
            request: dict[str, object] = {
                "model": self.config.served_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *media_items,
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": self.config.max_tokens,
                "stream": False,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            }
            if self.config.backend_family == "qwen3_omni":
                request["modalities"] = ["text"]
            completion = self._client_for_current_thread().chat.completions.create(
                **request
            )
            choices = getattr(completion, "choices", None)
            if not choices:
                raise TypeError("audio caption vLLM response has no choices")
            response = getattr(choices[0].message, "content", None)
            if not isinstance(response, str):
                raise TypeError("audio caption vLLM response text must be a string")
        except Exception as exc:
            raise JEATargetAudioCaptionBackendFailure(
                code=f"{self.config.backend_family}_vllm_request_failed",
                reason=f"{type(exc).__name__}: {exc}",
                attempt_count=1,
            ) from exc
        if not response.strip():
            raise JEATargetAudioCaptionBackendFailure(
                code=f"{self.config.backend_family}_vllm_empty_response",
                reason="audio caption vLLM returned no text",
                raw_responses=[response],
            )
        return response

    def _describe(
        self,
        job: JEATargetAudioCaptionJob,
        *,
        use_v5_fallback: bool,
    ) -> JEATargetAudioCaptionBackendResult:
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        system_prompt = V5_FALLBACK_SYSTEM_PROMPT if use_v5_fallback else SYSTEM_PROMPT
        for attempt in range(2):
            if attempt == 0:
                prompt = (
                    _v5_fallback_user_prompt(job)
                    if use_v5_fallback
                    else _user_prompt(job)
                )
            elif use_v5_fallback:
                prompt = _v5_fallback_repair_prompt(
                    job=job,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            else:
                prompt = _repair_prompt(
                    job=job,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            try:
                raw = self._request(
                    job=job,
                    system_prompt=system_prompt,
                    prompt=prompt,
                )
            except JEATargetAudioCaptionBackendFailure as exc:
                raise JEATargetAudioCaptionBackendFailure(
                    code=exc.code,
                    reason=exc.reason,
                    raw_responses=[*raw_responses, *exc.raw_responses],
                    issues=exc.issues,
                    attempt_count=attempt + 1,
                ) from exc
            raw_responses.append(raw)
            response, issues = parse_structured_json_issues(
                raw, TargetAudioCaptionResponse
            )
            if response is not None:
                issues = _response_issues(response, job)
            if response is not None and not issues:
                return JEATargetAudioCaptionBackendResult(
                    response=response,
                    raw_responses=tuple(raw_responses),
                )
        raise JEATargetAudioCaptionBackendFailure(
            code="structured_output_failed",
            reason="target audio caption failed closed after one repair",
            raw_responses=raw_responses,
            issues=issues,
        )

    def describe(
        self, job: JEATargetAudioCaptionJob
    ) -> JEATargetAudioCaptionBackendResult:
        return self._describe(job, use_v5_fallback=False)

    def describe_semantic_fallback(
        self, job: JEATargetAudioCaptionJob
    ) -> JEATargetAudioCaptionBackendResult:
        if self.config.backend_family != "qwen3_omni":
            raise ValueError("semantic fallback is only available for Qwen3-Omni")
        return self._describe(job, use_v5_fallback=True)


def target_audio_caption_request_fingerprint(
    job: JEATargetAudioCaptionJob,
    provenance: JEATargetAudioCaptionBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "prompt_version": JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION,
                "model_input": _model_input(job),
                "target_video_sha256": job.target_video_sha256,
                "target_full_audio_sha256": job.target_full_audio_sha256,
                "backend_configuration_fingerprint": provenance.configuration_fingerprint,
                "semantic_fallback_prompt_version": (
                    JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
                    if provenance.backend_family == "qwen3_omni"
                    else None
                ),
            }
        )
    )


def _record_from_result(
    *,
    job: JEATargetAudioCaptionJob,
    result: JEATargetAudioCaptionBackendResult,
    provenance: JEATargetAudioCaptionBackendProvenance,
    primary_repair_count: int,
    semantic_source: SemanticSource,
    semantic_fallback_attempted: bool,
) -> JEATargetAudioCaptionRecord:
    issues = _response_issues(result.response, job)
    if issues:
        raise RuntimeError("audio caption backend returned unvalidated output")
    entity_by_cluster = {
        item.speaker_cluster_id: item.entity_id for item in job.speaker_clusters
    }
    return JEATargetAudioCaptionRecord(
        target_clip_uid=job.target_clip_uid,
        clip_display_path=job.clip_display_path,
        status="ready",
        background_audio_prompt=result.response.background_audio_prompt,
        speaker_delivery=[
            TargetSpeakerDelivery(
                speaker_cluster_id=item.speaker_cluster_id,
                entity_id=entity_by_cluster[item.speaker_cluster_id],
                delivery_style=item.delivery_style,
            )
            for item in result.response.speaker_delivery
        ],
        target_video_path=job.target_video_path,
        target_video_sha256=job.target_video_sha256,
        target_full_audio_path=job.target_full_audio_path,
        target_full_audio_sha256=job.target_full_audio_sha256,
        target_audio_binding_path=job.target_audio_binding_path,
        target_audio_binding_sha256=job.target_audio_binding_sha256,
        input_modality=provenance.input_modality,
        backend_provenance=provenance,
        repair_count=primary_repair_count,
        semantic_source=semantic_source,
        semantic_fallback_attempted=semantic_fallback_attempted,
        semantic_fallback_prompt_version=(
            JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
            if semantic_fallback_attempted
            else None
        ),
        request_fingerprint=target_audio_caption_request_fingerprint(job, provenance),
    )


def _record_from_failure(
    *,
    job: JEATargetAudioCaptionJob,
    failure: JEATargetAudioCaptionBackendFailure,
    provenance: JEATargetAudioCaptionBackendProvenance,
) -> JEATargetAudioCaptionRecord:
    return JEATargetAudioCaptionRecord(
        target_clip_uid=job.target_clip_uid,
        clip_display_path=job.clip_display_path,
        status="failed",
        target_video_path=job.target_video_path,
        target_video_sha256=job.target_video_sha256,
        target_full_audio_path=job.target_full_audio_path,
        target_full_audio_sha256=job.target_full_audio_sha256,
        target_audio_binding_path=job.target_audio_binding_path,
        target_audio_binding_sha256=job.target_audio_binding_sha256,
        input_modality=provenance.input_modality,
        backend_provenance=provenance,
        repair_count=max(0, failure.attempt_count - 1),
        semantic_source=None,
        semantic_fallback_attempted=False,
        semantic_fallback_prompt_version=None,
        request_fingerprint=target_audio_caption_request_fingerprint(job, provenance),
        failure=JEATargetAudioCaptionFailure(
            code=failure.code,
            reason=failure.reason,
            attempt_count=failure.attempt_count,
            issues=[item.to_dict() for item in failure.issues],
        ),
    )


@dataclass(frozen=True)
class _ProcessedCaptionJob:
    record: JEATargetAudioCaptionRecord
    primary_raw_responses: tuple[str, ...]
    primary_attempt_count: int
    primary_failure: JEATargetAudioCaptionBackendFailure | None
    primary_validated_response: TargetAudioCaptionResponse | None
    fallback_raw_responses: tuple[str, ...]
    fallback_attempt_count: int
    fallback_failure: JEATargetAudioCaptionBackendFailure | None
    fallback_validated_response: TargetAudioCaptionResponse | None
    fallback_attempted: bool


def _process_caption_job(
    *,
    job: JEATargetAudioCaptionJob,
    backend: JEATargetAudioCaptionBackend,
    provenance: JEATargetAudioCaptionBackendProvenance,
) -> _ProcessedCaptionJob:
    primary_raw_responses: tuple[str, ...] = ()
    primary_attempt_count = 0
    primary_failure: JEATargetAudioCaptionBackendFailure | None = None
    primary_validated_response: TargetAudioCaptionResponse | None = None
    fallback_raw_responses: tuple[str, ...] = ()
    fallback_attempt_count = 0
    fallback_failure: JEATargetAudioCaptionBackendFailure | None = None
    fallback_validated_response: TargetAudioCaptionResponse | None = None
    fallback_attempted = False
    try:
        primary_result = backend.describe(job)
        primary_raw_responses = primary_result.raw_responses
        primary_attempt_count = len(primary_raw_responses)
        primary_validated_response = primary_result.response
        selected_result = primary_result
        semantic_source: SemanticSource = "primary_v6"
        if (
            provenance.backend_family == "qwen3_omni"
            and _is_all_semantic_null(primary_result.response)
        ):
            fallback_attempted = True
            try:
                fallback_result = backend.describe_semantic_fallback(job)
                fallback_raw_responses = fallback_result.raw_responses
                fallback_attempt_count = len(fallback_raw_responses)
                fallback_validated_response = fallback_result.response
                if _is_all_semantic_null(fallback_result.response):
                    semantic_source = "primary_v6_all_null_confirmed"
                else:
                    semantic_source = "fallback_v5"
                    selected_result = fallback_result
            except JEATargetAudioCaptionBackendFailure as exc:
                fallback_failure = exc
                fallback_raw_responses = exc.raw_responses
                fallback_attempt_count = max(
                    len(fallback_raw_responses), exc.attempt_count
                )
                semantic_source = "primary_v6_fallback_failed"
        record = _record_from_result(
            job=job,
            result=selected_result,
            provenance=provenance,
            primary_repair_count=max(0, primary_attempt_count - 1),
            semantic_source=semantic_source,
            semantic_fallback_attempted=fallback_attempted,
        )
    except JEATargetAudioCaptionBackendFailure as exc:
        primary_failure = exc
        primary_raw_responses = exc.raw_responses
        primary_attempt_count = max(len(primary_raw_responses), exc.attempt_count)
        record = _record_from_failure(
            job=job,
            failure=exc,
            provenance=provenance,
        )
    return _ProcessedCaptionJob(
        record=record,
        primary_raw_responses=primary_raw_responses,
        primary_attempt_count=primary_attempt_count,
        primary_failure=primary_failure,
        primary_validated_response=primary_validated_response,
        fallback_raw_responses=fallback_raw_responses,
        fallback_attempt_count=fallback_attempt_count,
        fallback_failure=fallback_failure,
        fallback_validated_response=fallback_validated_response,
        fallback_attempted=fallback_attempted,
    )


def _verify_inventory_sources(inventory: JEATargetAudioCaptionInventory) -> None:
    for path_value, expected, field_name in (
        (inventory.source_pairs_path, inventory.source_pairs_sha256, "source pairs"),
        (
            inventory.source_readable_segments_path,
            inventory.source_readable_segments_sha256,
            "source readable segments",
        ),
        (
            inventory.source_qwen3_asr_segments_path,
            inventory.source_qwen3_asr_segments_sha256,
            "source Qwen3 ASR segments",
        ),
    ):
        _verify_file(Path(path_value), expected, field_name=field_name)
    for job in inventory.jobs:
        _verify_file(
            Path(job.target_video_path),
            job.target_video_sha256,
            field_name="target video",
        )
        _verify_file(
            Path(job.target_full_audio_path),
            job.target_full_audio_sha256,
            field_name="target full audio",
        )
        _verify_file(
            Path(job.target_audio_binding_path),
            job.target_audio_binding_sha256,
            field_name="target audio binding",
        )


def target_audio_caption_output_root(
    audio_production_root: Path,
    *,
    backend_family: BackendFamily,
) -> Path:
    backend_directory = "dots3" if backend_family == "dots3" else "qwen3_omni"
    return (
        audio_production_root.expanduser().resolve(strict=False)
        / "audio_caption"
        / backend_directory
    )


def _media_name(job: JEATargetAudioCaptionJob) -> str:
    suffix = Path(job.target_full_audio_path).suffix.lower() or ".flac"
    return hashlib.sha256(job.target_clip_uid.encode("utf-8")).hexdigest() + suffix


def _review_html(
    *,
    inventory: JEATargetAudioCaptionInventory,
    records: Sequence[JEATargetAudioCaptionRecord],
    provenance: JEATargetAudioCaptionBackendProvenance,
    media_names: dict[str, str],
) -> str:
    records_by_clip = {item.target_clip_uid: item for item in records}
    cards: list[str] = []
    for job in inventory.jobs:
        record = records_by_clip[job.target_clip_uid]
        delivery_by_cluster = {
            item.speaker_cluster_id: item for item in record.speaker_delivery
        }
        delivery_rows = []
        for cluster in job.speaker_clusters:
            delivery = delivery_by_cluster.get(cluster.speaker_cluster_id)
            ranges = ", ".join(
                f"{item.start_time:.3f}-{item.end_time:.3f}s"
                for item in cluster.active_time_ranges
            )
            delivery_rows.append(
                "<li><code>"
                + html.escape(cluster.speaker_cluster_id)
                + "</code> entity_id="
                + html.escape(cluster.entity_id or "UNBOUND")
                + " | "
                + html.escape(ranges)
                + " | delivery_style="
                + html.escape(
                    "[null]" if delivery is None or delivery.delivery_style is None else delivery.delivery_style
                )
                + "</li>"
            )
        flags = " ".join(
            f"<label><input type='checkbox' data-flag='{flag}' onchange='saveQA()'>{flag}</label>"
            for flag in QA_FLAGS
        )
        labels = " ".join(
            f"<label><input type='radio' name='qa-{html.escape(job.target_clip_uid)}' value='{label}' onchange='saveQA()'>{label}</label>"
            for label in QA_LABELS
        )
        cards.append(
            f"<article class='case' data-clip='{html.escape(job.target_clip_uid)}'>"
            f"<h2>{html.escape(job.clip_display_path)}</h2>"
            f"<audio controls preload='metadata' src='media/{html.escape(media_names[job.target_clip_uid])}'></audio>"
            f"<p><b>Status:</b> {html.escape(record.status)}</p>"
            f"<p><b>Semantic source:</b> {html.escape(record.semantic_source or '[none]')}</p>"
            + (
                ""
                if record.failure is None
                else "<p><b>Failure:</b> "
                + html.escape(record.failure.reason)
                + "</p>"
            )
            + (
                ""
                if not record.semantic_fallback_attempted
                else "<details><summary>Semantic fallback diagnostics</summary>"
                "<p>Primary V6 and fallback V5 raw responses are retained in "
                f"<a href='raw/{html.escape(job.target_clip_uid)}.json'>raw diagnostics JSON</a>."
                "</p></details>"
            )
            + "<h3>Background audio</h3><p>"
            + html.escape(record.background_audio_prompt or "[null]")
            + "</p><h3>Speaker delivery</h3><ul>"
            + "".join(delivery_rows)
            + "</ul><div class='qa'><b>Decision:</b> "
            + labels
            + "<br><b>Flags:</b> "
            + flags
            + "<br><label>Note <input type='text' class='review-note' maxlength='500' oninput='saveQA()'></label></div></article>"
        )
    clip_order = [item.target_clip_uid for item in inventory.jobs]
    prefix = (
        f"h3-target-audio-caption-v4-{provenance.backend_family}-"
        f"{inventory.inventory_fingerprint}:"
    )
    provenance_json = json.dumps(
        provenance.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>H3 audio caption {provenance.backend_family} review</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f6;color:#171717}}
header,.case{{max-width:1050px;margin:0 auto 22px;background:white;border:1px solid #bbb;padding:18px}}
audio{{width:100%}}.qa{{margin-top:14px;padding:12px;background:#eef2f5}}.qa label{{margin-right:12px}}
button{{margin-right:10px;padding:8px 12px}}.review-note{{min-width:320px}}
</style></head><body><header><h1>H3 Target Audio Caption: {provenance.backend_family}</h1>
<p>{html.escape(provenance.served_model_name)} | {html.escape(provenance.input_modality)}</p>
<p id='progress'>Labeled 0 / {len(inventory.jobs)}</p>
<button onclick='exportQA()'>Export QA JSON</button><button onclick='clearQA()'>Clear QA labels</button></header>
{"".join(cards)}<script>
const inventoryFingerprint={json.dumps(inventory.inventory_fingerprint)};
const backendProvenance={provenance_json};
const clipOrder={json.dumps(clip_order)};const labels=['CORRECT','WRONG','UNCERTAIN'];
const flags={json.dumps(list(QA_FLAGS))};const keyPrefix={json.dumps(prefix)};
function stateFor(clip){{try{{return JSON.parse(localStorage.getItem(keyPrefix+clip)||'null')}}catch(error){{return null}}}}
function restore(){{document.querySelectorAll('.case').forEach(card=>{{const state=stateFor(card.dataset.clip);if(!state)return;card.querySelectorAll("input[type='radio']").forEach(input=>input.checked=input.value===state.label);card.querySelectorAll("input[type='checkbox']").forEach(input=>input.checked=(state.failure_flags||[]).includes(input.dataset.flag));card.querySelector('.review-note').value=state.review_note||'';}});updateCounts();}}
function saveQA(){{document.querySelectorAll('.case').forEach(card=>{{const selected=card.querySelector("input[type='radio']:checked");const selectedFlags=[...card.querySelectorAll("input[type='checkbox']:checked")].map(input=>input.dataset.flag);const note=card.querySelector('.review-note').value;if(selected||selectedFlags.length||note){{localStorage.setItem(keyPrefix+card.dataset.clip,JSON.stringify({{label:selected?selected.value:null,failure_flags:selectedFlags,review_note:note}}));}}else localStorage.removeItem(keyPrefix+card.dataset.clip);}});updateCounts();}}
function countsAndRows(){{const counts={{CORRECT:0,WRONG:0,UNCERTAIN:0,UNLABELED:0}};const rows=[];clipOrder.forEach(clip=>{{const state=stateFor(clip);if(state&&labels.includes(state.label)){{counts[state.label]++;rows.push({{target_clip_uid:clip,label:state.label,failure_flags:(state.failure_flags||[]).filter(flag=>flags.includes(flag)),review_note:state.review_note||''}});}}else counts.UNLABELED++;}});return {{counts,rows}};}}
function updateCounts(){{const data=countsAndRows();document.getElementById('progress').textContent=`Labeled ${{data.rows.length}} / {len(inventory.jobs)} | CORRECT ${{data.counts.CORRECT}} | WRONG ${{data.counts.WRONG}} | UNCERTAIN ${{data.counts.UNCERTAIN}} | UNLABELED ${{data.counts.UNLABELED}}`;}}
function exportQA(){{saveQA();const data=countsAndRows();const payload={{schema_version:'{JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION}',inventory_fingerprint:inventoryFingerprint,backend_provenance:backendProvenance,label_count:data.rows.length,total_clip_count:{len(inventory.jobs)},counts:data.counts,labels:data.rows}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=`target_audio_caption_${{backendProvenance.backend_family}}_human_qa.json`;link.click();URL.revokeObjectURL(url);}}
function clearQA(){{if(!confirm('Clear Target Audio Caption QA labels?'))return;clipOrder.forEach(clip=>localStorage.removeItem(keyPrefix+clip));document.querySelectorAll('.qa input').forEach(input=>{{if(input.type==='radio'||input.type==='checkbox')input.checked=false;else input.value='';}});updateCounts();}}
restore();</script></body></html>"""


def _publish_directory(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
    backend_family: BackendFamily,
) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"audio caption output already exists: {destination}")
    if _existing_backend_family(destination) != backend_family:
        raise ValueError("one audio caption backend cannot overwrite the other")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def _unknown_output_ownership() -> ValueError:
    return ValueError("cannot establish existing audio caption output ownership")


def _existing_backend_family(destination: Path) -> BackendFamily:
    summary_path = destination / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _unknown_output_ownership() from exc
    if not isinstance(payload, dict):
        raise _unknown_output_ownership()
    provenance = payload.get("backend_provenance")
    if not isinstance(provenance, dict):
        raise _unknown_output_ownership()
    backend_family = provenance.get("backend_family")
    if backend_family == "dots3":
        return "dots3"
    if backend_family == "qwen3_omni":
        return "qwen3_omni"
    raise _unknown_output_ownership()


def _process_caption_jobs(
    *,
    jobs: Sequence[JEATargetAudioCaptionJob],
    backend: JEATargetAudioCaptionBackend,
    provenance: JEATargetAudioCaptionBackendProvenance,
    max_concurrency: int,
) -> tuple[_ProcessedCaptionJob, ...]:
    if max_concurrency == 1:
        return tuple(
            _process_caption_job(
                job=job,
                backend=backend,
                provenance=provenance,
            )
            for job in jobs
        )
    with ThreadPoolExecutor(
        max_workers=max_concurrency,
        thread_name_prefix="h3-audio-caption",
    ) as executor:
        futures = [
            executor.submit(
                _process_caption_job,
                job=job,
                backend=backend,
                provenance=provenance,
            )
            for job in jobs
        ]
        return tuple(future.result() for future in futures)


def run_jea_target_audio_caption(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    backend: JEATargetAudioCaptionBackend,
    overwrite: bool = False,
    max_concurrency: int = 1,
) -> JEATargetAudioCaptionSummary:
    if max_concurrency < 1:
        raise ValueError("audio caption max_concurrency must be at least one")
    if inventory.inventory_fingerprint != _inventory_fingerprint(inventory):
        raise ValueError("audio caption inventory fingerprint is inconsistent")
    _verify_inventory_sources(inventory)
    provenance = backend.provenance
    destination = output_root.expanduser().resolve(strict=False)
    production_root = Path(inventory.source_audio_production_root).resolve(strict=True)
    if destination == production_root or destination in production_root.parents:
        raise ValueError("audio caption output cannot replace production root")
    for protected_name in (
        "audio",
        "primary_voice",
        "embedding",
        "pairs",
        "diarization",
        "asr",
        "h3",
    ):
        protected = (production_root / protected_name).resolve(strict=False)
        if destination == protected or protected in destination.parents:
            raise ValueError("audio caption output cannot replace production stages")
    other_default = target_audio_caption_output_root(
        production_root,
        backend_family=(
            "qwen3_omni" if provenance.backend_family == "dots3" else "dots3"
        ),
    )
    if destination == other_default:
        raise ValueError("one audio caption backend cannot overwrite the other")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"audio caption output already exists: {destination}")
    if destination.exists() and (
        _existing_backend_family(destination) != provenance.backend_family
    ):
        raise ValueError("one audio caption backend cannot overwrite the other")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[JEATargetAudioCaptionRecord] = []
    failures: Counter[str] = Counter()
    initial_calls = repair_calls = raw_count = 0
    fallback_triggers = fallback_initial_calls = fallback_repair_calls = 0
    fallback_recovered = fallback_still_all_null = fallback_failed = 0
    media_names: dict[str, str] = {}
    try:
        temporary.mkdir()
        (temporary / "raw").mkdir()
        (temporary / "media").mkdir()
        _write_json(temporary / "inventory.json", inventory)
        job_results = _process_caption_jobs(
            jobs=inventory.jobs,
            backend=backend,
            provenance=provenance,
            max_concurrency=max_concurrency,
        )
        for job, processed in zip(inventory.jobs, job_results, strict=True):
            record = processed.record
            records.append(record)
            if processed.primary_attempt_count:
                initial_calls += 1
                repair_calls += max(0, processed.primary_attempt_count - 1)
            if record.failure is not None:
                failures[record.failure.code] += 1
            if processed.fallback_attempted:
                fallback_triggers += 1
                fallback_initial_calls += 1
                fallback_repair_calls += max(
                    0, processed.fallback_attempt_count - 1
                )
                if record.semantic_source == "fallback_v5":
                    fallback_recovered += 1
                elif record.semantic_source == "primary_v6_all_null_confirmed":
                    fallback_still_all_null += 1
                elif record.semantic_source == "primary_v6_fallback_failed":
                    fallback_failed += 1
                else:
                    raise RuntimeError("audio caption fallback outcome is inconsistent")
            raw_count += len(processed.primary_raw_responses) + len(
                processed.fallback_raw_responses
            )
            _write_json(
                temporary / "raw" / f"{job.target_clip_uid}.json",
                {
                    "target_clip_uid": job.target_clip_uid,
                    "status": record.status,
                    "request_fingerprint": record.request_fingerprint,
                    "raw_responses": [
                        *processed.primary_raw_responses,
                        *processed.fallback_raw_responses,
                    ],
                    "primary": {
                        "prompt_version": JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION,
                        "raw_responses": list(processed.primary_raw_responses),
                        "validated_response": (
                            None
                            if processed.primary_validated_response is None
                            else processed.primary_validated_response.model_dump(
                                mode="json"
                            )
                        ),
                        "failure": (
                            None
                            if processed.primary_failure is None
                            else {
                                "code": processed.primary_failure.code,
                                "reason": processed.primary_failure.reason,
                                "attempt_count": processed.primary_failure.attempt_count,
                                "issues": [
                                    item.to_dict()
                                    for item in processed.primary_failure.issues
                                ],
                            }
                        ),
                    },
                    "semantic_fallback": {
                        "attempted": processed.fallback_attempted,
                        "prompt_version": (
                            JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION
                            if processed.fallback_attempted
                            else None
                        ),
                        "raw_responses": list(processed.fallback_raw_responses),
                        "validated_response": (
                            None
                            if processed.fallback_validated_response is None
                            else processed.fallback_validated_response.model_dump(
                                mode="json"
                            )
                        ),
                        "failure": (
                            None
                            if processed.fallback_failure is None
                            else {
                                "code": processed.fallback_failure.code,
                                "reason": processed.fallback_failure.reason,
                                "attempt_count": processed.fallback_failure.attempt_count,
                                "issues": [
                                    item.to_dict()
                                    for item in processed.fallback_failure.issues
                                ],
                            }
                        ),
                    },
                    "failure": (
                        None
                        if record.failure is None
                        else record.failure.model_dump(mode="json")
                    ),
                },
            )
            media_name = _media_name(job)
            shutil.copyfile(
                Path(job.target_full_audio_path),
                temporary / "media" / media_name,
            )
            media_names[job.target_clip_uid] = media_name
        summary = JEATargetAudioCaptionSummary(
            inventory_fingerprint=inventory.inventory_fingerprint,
            backend_provenance=provenance,
            target_clip_count=len(inventory.jobs),
            max_concurrency=max_concurrency,
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            initial_call_count=initial_calls,
            repair_call_count=repair_calls,
            semantic_fallback_trigger_count=fallback_triggers,
            semantic_fallback_initial_call_count=fallback_initial_calls,
            semantic_fallback_repair_call_count=fallback_repair_calls,
            semantic_fallback_recovered_count=fallback_recovered,
            semantic_fallback_still_all_null_count=fallback_still_all_null,
            semantic_fallback_failed_count=fallback_failed,
            raw_response_count=raw_count,
            failure_reason_counts=dict(sorted(failures.items())),
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        (temporary / "review.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                provenance=provenance,
                media_names=media_names,
            ),
            encoding="utf-8",
        )
        _publish_directory(
            temporary,
            destination,
            overwrite=overwrite,
            backend_family=provenance.backend_family,
        )
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "DEFAULT_DOTS3_CHECKPOINT_ID",
    "DEFAULT_DOTS3_MODEL",
    "DEFAULT_QWEN3_OMNI_CHECKPOINT_ID",
    "DEFAULT_QWEN3_OMNI_MODEL",
    "JEA_TARGET_AUDIO_CAPTION_FALLBACK_PROMPT_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_PRIMARY_PROMPT_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION",
    "SYSTEM_PROMPT",
    "V5_FALLBACK_SYSTEM_PROMPT",
    "JEATargetAudioCaptionBackendFailure",
    "JEATargetAudioCaptionBackendProvenance",
    "JEATargetAudioCaptionBackendResult",
    "JEATargetAudioCaptionConfig",
    "JEATargetAudioCaptionHumanQAExport",
    "JEATargetAudioCaptionInventory",
    "JEATargetAudioCaptionRecord",
    "JEATargetAudioCaptionSummary",
    "OpenAIJEATargetAudioCaptionBackend",
    "build_jea_target_audio_caption_inventory",
    "run_jea_target_audio_caption",
    "target_audio_caption_output_root",
]
