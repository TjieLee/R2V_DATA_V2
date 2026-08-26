from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import uuid
import wave
from collections import Counter, defaultdict
from collections.abc import Sequence
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

JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION = "r2v.h3.target_audio_caption.3"
JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION = (
    "r2v.h3.target_audio_caption_inventory.2"
)
JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION = "r2v.h3.target_audio_caption_summary.2"
JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION = (
    "r2v.h3.target_audio_caption_human_qa.2"
)
JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION = "h3_target_audio_caption_v5"
DEFAULT_DOTS3_MODEL = "dots3-note-prev"
DEFAULT_DOTS3_CHECKPOINT_ID = "/mnt/workspace/public/pretrained/dots3-note-prev"
DEFAULT_QWEN3_OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_QWEN3_OMNI_CHECKPOINT_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS = 0.05

BackendFamily = Literal["dots3", "qwen3_omni"]
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

SYSTEM_PROMPT = """You analyze AUDIO SEMANTICS for one target clip.
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
    prompt_version: Literal["h3_target_audio_caption_v5"] = (
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
    schema_version: Literal["r2v.h3.target_audio_caption.3"] = (
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
                or not cluster_ids
                or len(cluster_ids) != len(set(cluster_ids))
            ):
                raise ValueError("ready audio caption semantics are inconsistent")
        elif self.failure is None or self.background_audio_prompt is not None or bool(
            self.speaker_delivery
        ):
            raise ValueError("failed audio caption cannot publish semantics")
        return self


class JEATargetAudioCaptionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.target_audio_caption_summary.2"] = (
        JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: JEATargetAudioCaptionBackendProvenance
    target_clip_count: int = Field(gt=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    initial_call_count: int = Field(ge=0)
    repair_call_count: int = Field(ge=0)
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


def _user_prompt(job: JEATargetAudioCaptionJob) -> str:
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
        "needed. Follow the original audible-only policy. Do not emit dialogue or "
        "entity_id. Return every speaker_cluster_id exactly once in supplied order. "
        "Return one compact JSON object only.\nOriginal request:\n"
        f"{_user_prompt(job)}\nValidation issues:\n"
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
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    @property
    def provenance(self) -> JEATargetAudioCaptionBackendProvenance:
        return self.config.provenance()

    def _request(self, *, job: JEATargetAudioCaptionJob, prompt: str) -> str:
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
                    {"role": "system", "content": SYSTEM_PROMPT},
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
            completion = self.client.chat.completions.create(**request)
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

    def describe(
        self, job: JEATargetAudioCaptionJob
    ) -> JEATargetAudioCaptionBackendResult:
        raw_responses: list[str] = []
        issues: list[ValidationIssue] = []
        for attempt in range(2):
            prompt = (
                _user_prompt(job)
                if attempt == 0
                else _repair_prompt(
                    job=job,
                    invalid_response=raw_responses[-1],
                    issues=issues,
                )
            )
            try:
                raw = self._request(job=job, prompt=prompt)
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
            }
        )
    )


def _record_from_result(
    *,
    job: JEATargetAudioCaptionJob,
    result: JEATargetAudioCaptionBackendResult,
    provenance: JEATargetAudioCaptionBackendProvenance,
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
        repair_count=max(0, len(result.raw_responses) - 1),
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
        request_fingerprint=target_audio_caption_request_fingerprint(job, provenance),
        failure=JEATargetAudioCaptionFailure(
            code=failure.code,
            reason=failure.reason,
            attempt_count=failure.attempt_count,
            issues=[item.to_dict() for item in failure.issues],
        ),
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
            + (
                ""
                if record.failure is None
                else "<p><b>Failure:</b> "
                + html.escape(record.failure.reason)
                + "</p>"
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
    JEATargetAudioCaptionInventory.model_validate_json(
        (destination / "inventory.json").read_text(encoding="utf-8")
    )
    existing_summary = JEATargetAudioCaptionSummary.model_validate_json(
        (destination / "summary.json").read_text(encoding="utf-8")
    )
    if existing_summary.backend_provenance.backend_family != backend_family:
        raise ValueError("one audio caption backend cannot overwrite the other")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_jea_target_audio_caption(
    *,
    inventory: JEATargetAudioCaptionInventory,
    output_root: Path,
    backend: JEATargetAudioCaptionBackend,
    overwrite: bool = False,
) -> JEATargetAudioCaptionSummary:
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
    if destination.exists():
        existing_summary = JEATargetAudioCaptionSummary.model_validate_json(
            (destination / "summary.json").read_text(encoding="utf-8")
        )
        if existing_summary.backend_provenance.backend_family != provenance.backend_family:
            raise ValueError("one audio caption backend cannot overwrite the other")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[JEATargetAudioCaptionRecord] = []
    failures: Counter[str] = Counter()
    initial_calls = repair_calls = raw_count = 0
    media_names: dict[str, str] = {}
    try:
        temporary.mkdir()
        (temporary / "raw").mkdir()
        (temporary / "media").mkdir()
        _write_json(temporary / "inventory.json", inventory)
        for job in inventory.jobs:
            raw_responses: tuple[str, ...] = ()
            attempt_count = 0
            try:
                result = backend.describe(job)
                raw_responses = result.raw_responses
                attempt_count = len(raw_responses)
                record = _record_from_result(
                    job=job,
                    result=result,
                    provenance=provenance,
                )
            except JEATargetAudioCaptionBackendFailure as exc:
                raw_responses = exc.raw_responses
                attempt_count = max(len(raw_responses), exc.attempt_count)
                failures[exc.code] += 1
                record = _record_from_failure(
                    job=job,
                    failure=exc,
                    provenance=provenance,
                )
            records.append(record)
            if attempt_count:
                initial_calls += 1
                repair_calls += max(0, attempt_count - 1)
            raw_count += len(raw_responses)
            _write_json(
                temporary / "raw" / f"{job.target_clip_uid}.json",
                {
                    "target_clip_uid": job.target_clip_uid,
                    "status": record.status,
                    "request_fingerprint": record.request_fingerprint,
                    "raw_responses": list(raw_responses),
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
            ready_count=sum(item.status == "ready" for item in records),
            failed_count=sum(item.status == "failed" for item in records),
            initial_call_count=initial_calls,
            repair_call_count=repair_calls,
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
    "JEA_TARGET_AUDIO_CAPTION_HUMAN_QA_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_INVENTORY_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_PROMPT_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_SCHEMA_VERSION",
    "JEA_TARGET_AUDIO_CAPTION_SUMMARY_VERSION",
    "SYSTEM_PROMPT",
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
