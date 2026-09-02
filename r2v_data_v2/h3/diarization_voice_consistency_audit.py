from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import uuid
import wave
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    FFmpegAudioMediaBackend,
    SpeakerEmbeddingBackend,
)
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.schemas import SchemaModel

VOICE_CONSISTENCY_AUDIT_VERSION = "r2v.h3.diarization_voice_consistency.2"
VOICE_CONSISTENCY_SKIP_VERSION = "r2v.h3.diarization_voice_consistency_skip.1"
VOICE_CONSISTENCY_SUMMARY_VERSION = (
    "r2v.h3.diarization_voice_consistency_summary.2"
)
DEFAULT_OUTPUT_DIRECTORY = "diarization_voice_consistency_audit_v1"
SPEAKER_MODEL_IDENTIFIER = "speechbrain/spkrec-ecapa-voxceleb"
VOICE_CONSISTENCY_INPUT_PREPROCESSING = (
    "h3_voice_consistency_ffmpeg_atrim_32k_stereo_to_16k_mono_v1"
)

IdentityScope = Literal["direct_anchor_present", "cluster_propagated_only"]
DurationBucket = Literal["<0.75s", "0.75-1.0s", "1.0-2.0s", ">=2.0s"]
ReviewSelection = Literal[
    "propagated_lowest",
    "propagated_middle",
    "propagated_highest",
    "direct_anchor_lowest_control",
]


class VoiceConsistencyAuditRecord(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_voice_consistency.2"] = (
        VOICE_CONSISTENCY_AUDIT_VERSION
    )
    clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str
    entity_occurrence_id: str
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    direct_anchor_seconds: float = Field(ge=0, allow_inf_nan=False)
    identity_scope: IdentityScope
    duration_bucket: DurationBucket
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: Literal[32000] = 32000
    source_channels: Literal[2] = 2
    model_input_sample_rate_hz: Literal[16000] = 16000
    model_input_channels: Literal[1] = 1
    input_preprocessing: Literal[
        "h3_voice_consistency_ffmpeg_atrim_32k_stereo_to_16k_mono_v1"
    ] = VOICE_CONSISTENCY_INPUT_PREPROCESSING
    segment_audio_path: str
    segment_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_path: str
    primary_voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    speaker_model_identifier: str
    speaker_model_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_record(self) -> VoiceConsistencyAuditRecord:
        if self.end_time <= self.start_time:
            raise ValueError("voice-consistency segment times must be positive")
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("voice-consistency occurrence identity is inconsistent")
        if self.identity_scope == "direct_anchor_present":
            if self.direct_anchor_seconds <= 0:
                raise ValueError("direct-anchor audit record requires anchor evidence")
        elif self.direct_anchor_seconds != 0:
            raise ValueError("propagated-only audit record cannot claim direct anchor")
        return self


class VoiceConsistencySkippedRecord(SchemaModel):
    schema_version: Literal["r2v.h3.diarization_voice_consistency_skip.1"] = (
        VOICE_CONSISTENCY_SKIP_VERSION
    )
    clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str
    entity_occurrence_id: str
    identity_scope: IdentityScope
    reason_code: Literal[
        "missing_target_primary_voice",
        "primary_voice_embedding_failed",
        "segment_audio_preparation_failed",
        "segment_embedding_failed",
    ]
    reason: str


class SimilarityDistribution(SchemaModel):
    count: int = Field(ge=0)
    min: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    max: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    mean: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p05: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p10: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p25: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p50: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p75: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p90: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    p95: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_distribution(self) -> SimilarityDistribution:
        values = (
            self.min,
            self.max,
            self.mean,
            self.p05,
            self.p10,
            self.p25,
            self.p50,
            self.p75,
            self.p90,
            self.p95,
        )
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty similarity distribution cannot publish statistics")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("non-empty similarity distribution is incomplete")
        return self


class VoiceConsistencyReviewCandidate(VoiceConsistencyAuditRecord):
    selection_groups: list[ReviewSelection] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection_groups(self) -> VoiceConsistencyReviewCandidate:
        if len(self.selection_groups) != len(set(self.selection_groups)):
            raise ValueError("review candidate selection groups must be unique")
        return self


class VoiceConsistencyAuditSummary(SchemaModel):
    schema_version: Literal[
        "r2v.h3.diarization_voice_consistency_summary.2"
    ] = VOICE_CONSISTENCY_SUMMARY_VERSION
    source_audio_production_root: str
    source_artifact_sha256: dict[str, str]
    source_audio_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_primary_voice_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    bound_segment_count: int = Field(ge=0)
    mapped_segment_count: int = Field(ge=0)
    audited_segment_count: int = Field(ge=0)
    skipped_segment_count: int = Field(ge=0)
    direct_anchor_present_count: int = Field(ge=0)
    cluster_propagated_only_count: int = Field(ge=0)
    skip_reason_counts: dict[str, int]
    speaker_model_identifier: str
    speaker_model_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    primary_voice_embedding_call_count: int = Field(ge=0)
    segment_embedding_call_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    source_sample_rate_hz: Literal[32000] = 32000
    source_channels: Literal[2] = 2
    model_input_sample_rate_hz: Literal[16000] = 16000
    model_input_channels: Literal[1] = 1
    input_preprocessing: Literal[
        "h3_voice_consistency_ffmpeg_atrim_32k_stereo_to_16k_mono_v1"
    ] = VOICE_CONSISTENCY_INPUT_PREPROCESSING
    similarity_distributions: dict[IdentityScope, SimilarityDistribution]
    duration_bucket_distributions: dict[
        IdentityScope,
        dict[DurationBucket, SimilarityDistribution],
    ]
    review_candidate_count: int = Field(ge=0)
    similarity_threshold_applied: Literal[False] = False
    binding_modified: Literal[False] = False
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> VoiceConsistencyAuditSummary:
        if self.audited_segment_count + self.skipped_segment_count != (
            self.mapped_segment_count
        ):
            raise ValueError("voice-consistency audited/skipped counts do not reconcile")
        if self.direct_anchor_present_count + self.cluster_propagated_only_count != (
            self.audited_segment_count
        ):
            raise ValueError("voice-consistency identity-scope counts do not reconcile")
        if self.model_call_count != (
            self.primary_voice_embedding_call_count
            + self.segment_embedding_call_count
        ):
            raise ValueError("voice-consistency model-call counts do not reconcile")
        return self


@dataclass(frozen=True)
class _TargetVoice:
    clip_uid: str
    entity_id: str
    entity_occurrence_id: str
    path: Path
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _aggregate_file_fingerprint(values: dict[str, str]) -> str:
    return hashlib.sha256(
        _compact_json(sorted(values.items())).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: Path, model: type[SchemaModel]) -> list[SchemaModel]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _write_json(path: Path, value: SchemaModel) -> None:
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def duration_bucket(duration_seconds: float) -> DurationBucket:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration bucket requires a positive finite duration")
    if duration_seconds < 0.75:
        return "<0.75s"
    if duration_seconds < 1.0:
        return "0.75-1.0s"
    if duration_seconds < 2.0:
        return "1.0-2.0s"
    return ">=2.0s"


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("voice-consistency embedding must be finite and non-empty")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("voice-consistency embedding norm must be positive")
    return np.ascontiguousarray(values / norm, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_normalized = _normalize_embedding(left)
    right_normalized = _normalize_embedding(right)
    if left_normalized.shape != right_normalized.shape:
        raise ValueError("voice-consistency embeddings have different dimensions")
    value = float(np.dot(left_normalized, right_normalized))
    return min(1.0, max(-1.0, value))


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError("similarity quantile input is invalid")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def similarity_distribution(values: Sequence[float]) -> SimilarityDistribution:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return SimilarityDistribution(count=0)
    if not all(math.isfinite(value) and -1 <= value <= 1 for value in ordered):
        raise ValueError("similarity distribution contains invalid values")
    return SimilarityDistribution(
        count=len(ordered),
        min=ordered[0],
        max=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p05=_quantile(ordered, 0.05),
        p10=_quantile(ordered, 0.10),
        p25=_quantile(ordered, 0.25),
        p50=_quantile(ordered, 0.50),
        p75=_quantile(ordered, 0.75),
        p90=_quantile(ordered, 0.90),
        p95=_quantile(ordered, 0.95),
    )


def _prepare_voice_consistency_segment(
    raw: RawDiarizationSegment,
    destination: Path,
    *,
    ffmpeg: str,
) -> None:
    if raw.source_sample_rate_hz != 32000 or raw.source_channels != 2:
        raise ValueError("voice-consistency source must be canonical 32 kHz stereo")
    source = Path(raw.source_audio_path).expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-af",
            (
                f"atrim=start_sample={raw.source_start_sample}:"
                f"end_sample={raw.source_end_sample},asetpts=PTS-STARTPTS"
            ),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("voice-consistency ffmpeg segment preparation failed")
    try:
        with wave.open(str(destination), "rb") as prepared:
            if (
                prepared.getframerate() != 16000
                or prepared.getnchannels() != 1
                or prepared.getsampwidth() != 2
                or prepared.getcomptype() != "NONE"
                or prepared.getnframes() <= 0
            ):
                raise ValueError("voice-consistency model input is not 16 kHz mono PCM")
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("voice-consistency model input is unreadable") from exc


def _load_target_voices(
    samples: Sequence[FinalH3SampleV2],
) -> tuple[dict[str, _TargetVoice], dict[str, str]]:
    by_occurrence: dict[str, _TargetVoice] = {}
    file_hashes: dict[str, str] = {}
    target_video_by_clip: dict[str, str] = {}
    for sample in samples:
        previous_video = target_video_by_clip.setdefault(
            sample.clip_uid,
            sample.target_video,
        )
        if previous_video != sample.target_video:
            raise ValueError("H3 samples disagree on target video provenance")
        for voice in sample.subject_voices:
            if voice.voice_source != "target":
                continue
            path = Path(voice.voice_reference_path).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError("target primary voice reference is not a file")
            sha256 = _sha256_file(path)
            candidate = _TargetVoice(
                clip_uid=sample.clip_uid,
                entity_id=voice.entity_id,
                entity_occurrence_id=voice.target_occurrence_id,
                path=path,
                sha256=sha256,
            )
            existing = by_occurrence.setdefault(voice.target_occurrence_id, candidate)
            if existing != candidate:
                raise ValueError("H3 samples disagree on target primary voice")
            file_hashes[str(path)] = sha256
    return by_occurrence, file_hashes


def _validate_segment_inputs(
    raw_segments: Sequence[RawDiarizationSegment],
    bound_segments: Sequence[BoundDiarizationSegment],
) -> dict[tuple[str, str], RawDiarizationSegment]:
    raw_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in raw_segments
    }
    bound_by_key = {
        (item.target_clip_uid, item.segment_id): item for item in bound_segments
    }
    if len(raw_by_key) != len(raw_segments) or len(bound_by_key) != len(bound_segments):
        raise ValueError("diarization voice-consistency inputs contain duplicates")
    if set(raw_by_key) != set(bound_by_key):
        raise ValueError("raw and bound diarization segment inventories differ")
    for key, bound in bound_by_key.items():
        raw = raw_by_key[key]
        if (
            raw.speaker_cluster_id != bound.speaker_cluster_id
            or raw.start_time != bound.start_time
            or raw.end_time != bound.end_time
            or raw.source_start_sample != bound.source_start_sample
            or raw.source_end_sample != bound.source_end_sample
        ):
            raise ValueError("raw and bound diarization segment evidence differs")
    return raw_by_key


def _model_identity(backend: SpeakerEmbeddingBackend) -> tuple[str, str | None]:
    identifier = str(getattr(backend, "model_identifier", type(backend).__name__))
    fingerprint = getattr(backend, "checkpoint_sha256", None)
    return identifier, None if fingerprint is None else str(fingerprint)


def _record_key(record: VoiceConsistencyAuditRecord) -> tuple[str, str]:
    return record.clip_uid, record.segment_id


def build_review_candidates(
    records: Sequence[VoiceConsistencyAuditRecord],
) -> list[VoiceConsistencyReviewCandidate]:
    propagated = sorted(
        (item for item in records if item.identity_scope == "cluster_propagated_only"),
        key=lambda item: (item.cosine_similarity, *_record_key(item)),
    )
    direct = sorted(
        (item for item in records if item.identity_scope == "direct_anchor_present"),
        key=lambda item: (item.cosine_similarity, *_record_key(item)),
    )
    groups: list[tuple[ReviewSelection, list[VoiceConsistencyAuditRecord]]] = []
    groups.append(("propagated_lowest", propagated[:30]))
    if propagated:
        median = similarity_distribution(
            [item.cosine_similarity for item in propagated]
        ).p50
        assert median is not None
        middle = sorted(
            propagated,
            key=lambda item: (
                abs(item.cosine_similarity - median),
                item.cosine_similarity,
                *_record_key(item),
            ),
        )[:10]
    else:
        middle = []
    groups.append(("propagated_middle", middle))
    groups.append(("propagated_highest", list(reversed(propagated[-10:]))))
    groups.append(("direct_anchor_lowest_control", direct[:20]))
    selected: dict[
        tuple[str, str],
        tuple[VoiceConsistencyAuditRecord, list[ReviewSelection]],
    ] = {}
    order: list[tuple[str, str]] = []
    for group, items in groups:
        for item in items:
            key = _record_key(item)
            if key not in selected:
                selected[key] = (item, [])
                order.append(key)
            selected[key][1].append(group)
    return [
        VoiceConsistencyReviewCandidate(
            **selected[key][0].model_dump(mode="python"),
            selection_groups=selected[key][1],
        )
        for key in order
    ]


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(
            f"diarization voice-consistency audit already exists: {destination}"
        )
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_diarization_voice_consistency_audit(
    *,
    audio_production_root: Path,
    speaker_backend: SpeakerEmbeddingBackend,
    output_root: Path | None = None,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> VoiceConsistencyAuditSummary:
    paths = jea_production_paths(audio_production_root)
    production_root = paths.root.resolve(strict=True)
    source_paths = {
        "raw_segments": paths.diarization / "raw_segments.jsonl",
        "bound_segments": paths.diarization / "bound_segments.jsonl",
        "h3_samples": paths.h3 / "samples.jsonl",
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise ValueError("diarization voice-consistency source artifacts are incomplete")
    source_artifact_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    raw_segments = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(source_paths["raw_segments"], RawDiarizationSegment)
    ]
    bound_segments = [
        BoundDiarizationSegment.model_validate(item)
        for item in _read_jsonl(source_paths["bound_segments"], BoundDiarizationSegment)
    ]
    samples = [
        FinalH3SampleV2.model_validate(item)
        for item in _read_jsonl(source_paths["h3_samples"], FinalH3SampleV2)
    ]
    raw_by_key = _validate_segment_inputs(raw_segments, bound_segments)
    target_voices, primary_voice_hashes = _load_target_voices(samples)
    source_audio_hashes: dict[str, str] = {}
    source_audio_frame_counts: dict[str, int] = {}
    media_backend = FFmpegAudioMediaBackend(ffmpeg=ffmpeg, ffprobe=ffprobe)
    for raw in raw_segments:
        source_path = Path(raw.source_audio_path).expanduser().resolve(strict=True)
        source_sha = _sha256_file(source_path)
        if source_sha != raw.source_audio_sha256:
            raise ValueError("raw diarization source audio differs from provenance")
        previous = source_audio_hashes.setdefault(str(source_path), source_sha)
        if previous != source_sha:
            raise ValueError("raw diarization source audio changed during inventory")
        if raw.source_sample_rate_hz != 32000 or raw.source_channels != 2:
            raise ValueError("voice-consistency source must be canonical 32 kHz stereo")
        source_key = str(source_path)
        if source_key not in source_audio_frame_counts:
            probe = media_backend.probe_audio_file(source_path)
            if (
                probe.sample_rate_hz != 32000
                or probe.channels != 2
                or "flac" not in probe.format_name.lower()
            ):
                raise ValueError(
                    "voice-consistency source must be persisted 32 kHz stereo FLAC"
                )
            source_audio_frame_counts[source_key] = probe.frame_count
        if raw.source_end_sample > source_audio_frame_counts[source_key]:
            raise ValueError("voice-consistency segment exceeds source audio")

    destination = (
        output_root.expanduser().resolve(strict=False)
        if output_root is not None
        else production_root / DEFAULT_OUTPUT_DIRECTORY
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    model_identifier, model_fingerprint = _model_identity(speaker_backend)
    primary_cache: dict[str, np.ndarray | str] = {}
    primary_calls = 0
    segment_calls = 0
    records: list[VoiceConsistencyAuditRecord] = []
    skipped: list[VoiceConsistencySkippedRecord] = []
    try:
        temporary.mkdir()
        for bound in sorted(
            bound_segments,
            key=lambda item: (
                item.target_clip_uid,
                item.start_time,
                item.end_time,
                item.segment_id,
            ),
        ):
            if bound.entity_id is None or bound.entity_occurrence_id is None:
                continue
            raw = raw_by_key[(bound.target_clip_uid, bound.segment_id)]
            scope: IdentityScope = bound.identity_scope
            target_voice = target_voices.get(bound.entity_occurrence_id)
            if target_voice is None:
                skipped.append(
                    VoiceConsistencySkippedRecord(
                        clip_uid=bound.target_clip_uid,
                        segment_id=bound.segment_id,
                        speaker_cluster_id=bound.speaker_cluster_id,
                        entity_id=bound.entity_id,
                        entity_occurrence_id=bound.entity_occurrence_id,
                        identity_scope=scope,
                        reason_code="missing_target_primary_voice",
                        reason="no target primary voice reference is published for entity",
                    )
                )
                continue
            primary_value = primary_cache.get(bound.entity_occurrence_id)
            if primary_value is None:
                primary_calls += 1
                try:
                    result = speaker_backend.embed_speaker(
                        entity_occurrence_id=(
                            f"primary/{bound.entity_occurrence_id}"
                        ),
                        audio_path=target_voice.path,
                    )
                    primary_value = _normalize_embedding(result.vector)
                except Exception as exc:  # noqa: BLE001 - isolate one voice reference
                    primary_value = f"{type(exc).__name__}: {exc}"
                primary_cache[bound.entity_occurrence_id] = primary_value
            if isinstance(primary_value, str):
                skipped.append(
                    VoiceConsistencySkippedRecord(
                        clip_uid=bound.target_clip_uid,
                        segment_id=bound.segment_id,
                        speaker_cluster_id=bound.speaker_cluster_id,
                        entity_id=bound.entity_id,
                        entity_occurrence_id=bound.entity_occurrence_id,
                        identity_scope=scope,
                        reason_code="primary_voice_embedding_failed",
                        reason=primary_value,
                    )
                )
                continue
            segment_relative = (
                Path("segment_audio")
                / bound.target_clip_uid
                / f"{bound.segment_id}.wav"
            )
            segment_path = temporary / segment_relative
            try:
                _prepare_voice_consistency_segment(
                    raw,
                    segment_path,
                    ffmpeg=ffmpeg,
                )
            except (OSError, ValueError) as exc:
                skipped.append(
                    VoiceConsistencySkippedRecord(
                        clip_uid=bound.target_clip_uid,
                        segment_id=bound.segment_id,
                        speaker_cluster_id=bound.speaker_cluster_id,
                        entity_id=bound.entity_id,
                        entity_occurrence_id=bound.entity_occurrence_id,
                        identity_scope=scope,
                        reason_code="segment_audio_preparation_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            segment_calls += 1
            try:
                result = speaker_backend.embed_speaker(
                    entity_occurrence_id=(
                        f"segment/{bound.target_clip_uid}/{bound.segment_id}"
                    ),
                    audio_path=segment_path,
                )
                similarity = cosine_similarity(primary_value, result.vector)
            except Exception as exc:  # noqa: BLE001 - isolate one segment inference
                skipped.append(
                    VoiceConsistencySkippedRecord(
                        clip_uid=bound.target_clip_uid,
                        segment_id=bound.segment_id,
                        speaker_cluster_id=bound.speaker_cluster_id,
                        entity_id=bound.entity_id,
                        entity_occurrence_id=bound.entity_occurrence_id,
                        identity_scope=scope,
                        reason_code="segment_embedding_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            duration = (
                raw.source_end_sample - raw.source_start_sample
            ) / raw.source_sample_rate_hz
            records.append(
                VoiceConsistencyAuditRecord(
                    clip_uid=bound.target_clip_uid,
                    segment_id=bound.segment_id,
                    speaker_cluster_id=bound.speaker_cluster_id,
                    entity_id=bound.entity_id,
                    entity_occurrence_id=bound.entity_occurrence_id,
                    start_time=bound.start_time,
                    end_time=bound.end_time,
                    duration_seconds=duration,
                    direct_anchor_seconds=bound.direct_anchor_seconds,
                    identity_scope=scope,
                    duration_bucket=duration_bucket(duration),
                    source_audio_path=str(
                        Path(raw.source_audio_path).expanduser().resolve(strict=True)
                    ),
                    source_audio_sha256=raw.source_audio_sha256,
                    source_start_sample=raw.source_start_sample,
                    source_end_sample=raw.source_end_sample,
                    source_sample_rate_hz=raw.source_sample_rate_hz,
                    source_channels=raw.source_channels,
                    segment_audio_path=str(segment_relative.as_posix()),
                    segment_audio_sha256=_sha256_file(segment_path),
                    primary_voice_reference_path=str(target_voice.path),
                    primary_voice_reference_sha256=target_voice.sha256,
                    speaker_model_identifier=model_identifier,
                    speaker_model_fingerprint=model_fingerprint,
                    cosine_similarity=similarity,
                )
            )

        source_scope_values = {
            scope: [
                item.cosine_similarity
                for item in records
                if item.identity_scope == scope
            ]
            for scope in ("direct_anchor_present", "cluster_propagated_only")
        }
        bucket_values = {
            scope: {
                bucket: [
                    item.cosine_similarity
                    for item in records
                    if item.identity_scope == scope
                    and item.duration_bucket == bucket
                ]
                for bucket in ("<0.75s", "0.75-1.0s", "1.0-2.0s", ">=2.0s")
            }
            for scope in ("direct_anchor_present", "cluster_propagated_only")
        }
        review_candidates = build_review_candidates(records)
        summary = VoiceConsistencyAuditSummary(
            source_audio_production_root=str(production_root),
            source_artifact_sha256=source_artifact_hashes,
            source_audio_set_fingerprint=_aggregate_file_fingerprint(
                source_audio_hashes
            ),
            target_primary_voice_set_fingerprint=_aggregate_file_fingerprint(
                primary_voice_hashes
            ),
            bound_segment_count=len(bound_segments),
            mapped_segment_count=sum(item.entity_id is not None for item in bound_segments),
            audited_segment_count=len(records),
            skipped_segment_count=len(skipped),
            direct_anchor_present_count=sum(
                item.identity_scope == "direct_anchor_present" for item in records
            ),
            cluster_propagated_only_count=sum(
                item.identity_scope == "cluster_propagated_only" for item in records
            ),
            skip_reason_counts=dict(
                sorted(Counter(item.reason_code for item in skipped).items())
            ),
            speaker_model_identifier=model_identifier,
            speaker_model_fingerprint=model_fingerprint,
            primary_voice_embedding_call_count=primary_calls,
            segment_embedding_call_count=segment_calls,
            model_call_count=primary_calls + segment_calls,
            similarity_distributions={
                scope: similarity_distribution(values)
                for scope, values in source_scope_values.items()
            },
            duration_bucket_distributions={
                scope: {
                    bucket: similarity_distribution(values)
                    for bucket, values in buckets.items()
                }
                for scope, buckets in bucket_values.items()
            },
            review_candidate_count=len(review_candidates),
        )
        current_source_hashes = {
            name: _sha256_file(path) for name, path in source_paths.items()
        }
        current_audio_hashes = {
            path: _sha256_file(Path(path)) for path in source_audio_hashes
        }
        current_voice_hashes = {
            path: _sha256_file(Path(path)) for path in primary_voice_hashes
        }
        if (
            current_source_hashes != source_artifact_hashes
            or current_audio_hashes != source_audio_hashes
            or current_voice_hashes != primary_voice_hashes
        ):
            raise ValueError("voice-consistency source artifacts changed during audit")
        _write_jsonl(temporary / "records.jsonl", records)
        _write_jsonl(temporary / "skipped.jsonl", skipped)
        _write_jsonl(
            temporary / "review_candidates.jsonl",
            review_candidates,
        )
        _write_json(temporary / "summary.json", summary)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "SPEAKER_MODEL_IDENTIFIER",
    "SimilarityDistribution",
    "VoiceConsistencyAuditRecord",
    "VoiceConsistencyAuditSummary",
    "VoiceConsistencyReviewCandidate",
    "VoiceConsistencySkippedRecord",
    "build_review_candidates",
    "cosine_similarity",
    "duration_bucket",
    "run_diarization_voice_consistency_audit",
    "similarity_distribution",
]
