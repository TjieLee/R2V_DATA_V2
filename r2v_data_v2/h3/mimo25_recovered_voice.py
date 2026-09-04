from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    AudioFileProbe,
    AudioMediaBackend,
    FFmpegAudioMediaBackend,
)
from r2v_data_v2.h3.jea_audio_production import (
    CANONICAL_AUDIO_CHANNELS,
    CANONICAL_AUDIO_SAMPLE_RATE_HZ,
    CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS,
    VOICE_SAMPLE_MAPPING_POLICY,
)
from r2v_data_v2.h3.jea_final_renderer import FinalSubjectVoice
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob, MimoRecord
from r2v_data_v2.h3.mimo25_backend import MimoSegmentDecision
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.voice_quality import _audio_metrics

RECOVERED_VOICE_POLICY_VERSION = "h3_mimo25_recovered_voice_quality_v1"
RECOVERED_VOICE_RECORD_VERSION = "r2v.h3.mimo25_recovered_voice.1"
_LOCAL_NOISE_CONTEXT_SECONDS = 2.0
_MINIMUM_LOCAL_NOISE_SECONDS = 0.20
_LOCAL_NOISE_WINDOW_SECONDS = 0.020

RecoveredVoiceReasonCode = Literal[
    "unresolved",
    "not_visible_entity",
    "not_onscreen_spoken",
    "non_single_speaker",
    "secondary_vocal_activity",
    "invalid_canonical_sample_range",
    "duration_too_short",
    "rms_too_low",
    "clipping_excessive",
    "local_noise_unavailable",
    "snr_too_low",
    "not_selected_better_candidate",
    "reference_limit",
]

_REASON_ORDER: tuple[RecoveredVoiceReasonCode, ...] = (
    "unresolved",
    "not_visible_entity",
    "not_onscreen_spoken",
    "non_single_speaker",
    "secondary_vocal_activity",
    "invalid_canonical_sample_range",
    "duration_too_short",
    "rms_too_low",
    "clipping_excessive",
    "local_noise_unavailable",
    "snr_too_low",
    "not_selected_better_candidate",
    "reference_limit",
)


@dataclass(frozen=True)
class MimoRecoveredVoiceQualityPolicy:
    version: str = RECOVERED_VOICE_POLICY_VERSION
    minimum_duration_seconds: float = 1.0
    minimum_rms_dbfs: float = -40.0
    maximum_clipping_ratio: float = 0.0001
    require_local_noise_context: bool = True
    minimum_estimated_snr_db: float = 10.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_duration_seconds,
            self.minimum_rms_dbfs,
            self.maximum_clipping_ratio,
            self.minimum_estimated_snr_db,
        )
        if not self.version.strip() or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("MiMo recovered voice policy must be finite and versioned")
        if (
            self.minimum_duration_seconds <= 0
            or not 0 <= self.maximum_clipping_ratio <= 1
        ):
            raise ValueError("MiMo recovered voice policy thresholds are invalid")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class MimoRecoveredVoiceQualityMetrics(SchemaModel):
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    sample_count: int = Field(gt=0)
    rms_amplitude: float = Field(ge=0, allow_inf_nan=False)
    rms_dbfs: float | None = Field(default=None, allow_inf_nan=False)
    peak_amplitude: float = Field(ge=0, allow_inf_nan=False)
    peak_dbfs: float | None = Field(default=None, allow_inf_nan=False)
    clipping_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    local_noise_sample_count: int = Field(ge=0)
    local_noise_duration_seconds: float = Field(ge=0, allow_inf_nan=False)
    local_noise_rms_amplitude: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    local_noise_rms_dbfs: float | None = Field(default=None, allow_inf_nan=False)
    estimated_snr_db: float | None = Field(default=None, allow_inf_nan=False)


class MimoRecoveredVoiceReference(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_recovered_voice.1"] = (
        RECOVERED_VOICE_RECORD_VERSION
    )
    clip_uid: str
    entity_id: str = Field(pattern=r"^e[1-9]\d*$")
    subject_index: int = Field(gt=0)
    speaker_group: str | None = Field(default=None, pattern=r"^g[1-9]\d*$")
    source_segment_id: str
    source_start: float = Field(ge=0, allow_inf_nan=False)
    source_end: float = Field(gt=0, allow_inf_nan=False)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    mimo_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_codes: list[str]
    confidence: Literal["high", "medium", "low"]
    quality_metrics: MimoRecoveredVoiceQualityMetrics | None = None
    quality_policy_version: Literal["h3_mimo25_recovered_voice_quality_v1"] = (
        RECOVERED_VOICE_POLICY_VERSION
    )
    quality_policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["selected", "rejected"]
    reason_codes: list[RecoveredVoiceReasonCode]
    output_path: str | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reference(self) -> MimoRecoveredVoiceReference:
        if self.source_end <= self.source_start or (
            self.source_end_sample <= self.source_start_sample
        ):
            raise ValueError("recovered voice interval is invalid")
        if (self.source_start_sample, self.source_end_sample) != (
            round(self.source_start * self.source_sample_rate_hz),
            round(self.source_end * self.source_sample_rate_hz),
        ):
            raise ValueError("recovered voice samples differ from authoritative times")
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("recovered voice evidence codes must be unique")
        ordered = [code for code in _REASON_ORDER if code in self.reason_codes]
        if ordered != self.reason_codes or len(ordered) != len(set(ordered)):
            raise ValueError("recovered voice reason codes must be unique and ordered")
        if self.status == "selected":
            if (
                self.reason_codes
                or self.speaker_group is None
                or self.quality_metrics is None
                or self.output_path is None
                or self.output_sha256 is None
            ):
                raise ValueError(
                    "selected recovered voice requires one published asset"
                )
            if not Path(self.output_path).is_absolute():
                raise ValueError("selected recovered voice output path must be absolute")
        elif (
            not self.reason_codes
            or self.output_path is not None
            or self.output_sha256 is not None
        ):
            raise ValueError("rejected recovered voice requires only rejection reasons")
        return self


@dataclass(frozen=True)
class CanonicalAudioAnalysis:
    probe: AudioFileProbe
    mono_pcm16: np.ndarray


class RecoveredVoiceAudioAnalyzer(Protocol):
    def load(self, path: Path) -> CanonicalAudioAnalysis: ...


class FFmpegRecoveredVoiceAudioAnalyzer:
    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.media_backend = FFmpegAudioMediaBackend(ffmpeg=ffmpeg, ffprobe=ffprobe)

    def load(self, path: Path) -> CanonicalAudioAnalysis:
        probe = self.media_backend.probe_audio_file(path)
        completed = subprocess.run(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-ar",
                str(CANONICAL_AUDIO_SAMPLE_RATE_HZ),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        samples = np.frombuffer(completed.stdout, dtype="<i2").copy()
        if samples.size != probe.frame_count:
            raise ValueError(
                "recovered voice analysis differs from canonical sample extent"
            )
        return CanonicalAudioAnalysis(probe=probe, mono_pcm16=samples)


def _dbfs(amplitude: float) -> float | None:
    return None if amplitude <= 0 else 20.0 * math.log10(amplitude)


def _noise_metrics(
    samples: np.ndarray,
    *,
    start_sample: int,
    end_sample: int,
    occupied_ranges: Sequence[tuple[int, int]],
) -> dict[str, float | int | None]:
    radius = round(_LOCAL_NOISE_CONTEXT_SECONDS * CANONICAL_AUDIO_SAMPLE_RATE_HZ)
    search_ranges = (
        (max(0, start_sample - radius), start_sample),
        (end_sample, min(samples.size, end_sample + radius)),
    )
    spans: list[tuple[int, int]] = []
    for search_start, search_end in search_ranges:
        cursor = search_start
        for occupied_start, occupied_end in occupied_ranges:
            if occupied_end <= cursor or occupied_start >= search_end:
                continue
            if occupied_start > cursor:
                spans.append((cursor, min(occupied_start, search_end)))
            cursor = max(cursor, occupied_end)
            if cursor >= search_end:
                break
        if cursor < search_end:
            spans.append((cursor, search_end))
    sample_count = sum(end - start for start, end in spans)
    duration = sample_count / CANONICAL_AUDIO_SAMPLE_RATE_HZ
    minimum = math.ceil(_MINIMUM_LOCAL_NOISE_SECONDS * CANONICAL_AUDIO_SAMPLE_RATE_HZ)
    if sample_count < minimum:
        return {
            "local_noise_sample_count": sample_count,
            "local_noise_duration_seconds": duration,
            "local_noise_rms_amplitude": None,
            "local_noise_rms_dbfs": None,
        }
    window = round(_LOCAL_NOISE_WINDOW_SECONDS * CANONICAL_AUDIO_SAMPLE_RATE_HZ)
    window_rms: list[float] = []
    for span_start, span_end in spans:
        for offset in range(span_start, span_end - window + 1, window):
            metrics = _audio_metrics(samples[offset : offset + window].tolist())
            window_rms.append(float(metrics["rms_amplitude"]))
    if not window_rms:
        return {
            "local_noise_sample_count": sample_count,
            "local_noise_duration_seconds": duration,
            "local_noise_rms_amplitude": None,
            "local_noise_rms_dbfs": None,
        }
    noise_rms = float(np.median(np.asarray(window_rms, dtype=np.float64)))
    return {
        "local_noise_sample_count": sample_count,
        "local_noise_duration_seconds": duration,
        "local_noise_rms_amplitude": noise_rms,
        "local_noise_rms_dbfs": _dbfs(noise_rms),
    }


def _quality_metrics(
    analysis: CanonicalAudioAnalysis,
    *,
    start_sample: int,
    end_sample: int,
    occupied_ranges: Sequence[tuple[int, int]],
) -> MimoRecoveredVoiceQualityMetrics:
    audio = _audio_metrics(analysis.mono_pcm16[start_sample:end_sample].tolist())
    noise = _noise_metrics(
        analysis.mono_pcm16,
        start_sample=start_sample,
        end_sample=end_sample,
        occupied_ranges=occupied_ranges,
    )
    rms_dbfs = audio["rms_dbfs"]
    noise_dbfs = noise["local_noise_rms_dbfs"]
    snr = (
        None
        if rms_dbfs is None or noise_dbfs is None
        else float(rms_dbfs) - float(noise_dbfs)
    )
    return MimoRecoveredVoiceQualityMetrics(
        duration_seconds=(end_sample - start_sample) / CANONICAL_AUDIO_SAMPLE_RATE_HZ,
        sample_count=end_sample - start_sample,
        rms_amplitude=float(audio["rms_amplitude"]),
        rms_dbfs=None if rms_dbfs is None else float(rms_dbfs),
        peak_amplitude=float(audio["peak_amplitude"]),
        peak_dbfs=(None if audio["peak_dbfs"] is None else float(audio["peak_dbfs"])),
        clipping_ratio=float(audio["clipping_ratio"]),
        estimated_snr_db=snr,
        **noise,
    )


def _semantic_reasons(decision: MimoSegmentDecision) -> list[RecoveredVoiceReasonCode]:
    reasons: list[RecoveredVoiceReasonCode] = []
    if decision.resolution != "resolved" or decision.primary_speaker_group is None:
        reasons.append("unresolved")
    if decision.binding_status != "visible_entity" or decision.entity_id is None:
        reasons.append("not_visible_entity")
    if decision.speech_presentation != "onscreen_spoken":
        reasons.append("not_onscreen_spoken")
    if decision.vocal_composition != "single_speaker":
        reasons.append("non_single_speaker")
    if decision.secondary_vocal_activity.present:
        reasons.append("secondary_vocal_activity")
    return [code for code in _REASON_ORDER if code in reasons]


def _quality_reasons(
    metrics: MimoRecoveredVoiceQualityMetrics,
    policy: MimoRecoveredVoiceQualityPolicy,
) -> list[RecoveredVoiceReasonCode]:
    reasons: list[RecoveredVoiceReasonCode] = []
    if metrics.duration_seconds < policy.minimum_duration_seconds:
        reasons.append("duration_too_short")
    if metrics.rms_dbfs is None or metrics.rms_dbfs < policy.minimum_rms_dbfs:
        reasons.append("rms_too_low")
    if metrics.clipping_ratio > policy.maximum_clipping_ratio:
        reasons.append("clipping_excessive")
    if metrics.local_noise_rms_dbfs is None:
        if policy.require_local_noise_context:
            reasons.append("local_noise_unavailable")
    elif metrics.estimated_snr_db is None or (
        metrics.estimated_snr_db < policy.minimum_estimated_snr_db
    ):
        reasons.append("snr_too_low")
    return [code for code in _REASON_ORDER if code in reasons]


def _candidate_rank(reference: MimoRecoveredVoiceReference) -> tuple[object, ...]:
    metrics = reference.quality_metrics
    assert metrics is not None
    confidence = {"high": 0, "medium": 1, "low": 2}[reference.confidence]
    return (
        confidence,
        -(
            metrics.estimated_snr_db
            if metrics.estimated_snr_db is not None
            else -math.inf
        ),
        -metrics.duration_seconds,
        -metrics.rms_amplitude,
        metrics.clipping_ratio,
        reference.source_start,
        reference.source_segment_id,
    )


def _replace_reference(
    reference: MimoRecoveredVoiceReference,
    **updates: object,
) -> MimoRecoveredVoiceReference:
    values = reference.model_dump(mode="python")
    values.update(updates)
    return MimoRecoveredVoiceReference.model_validate(values)


def recover_mimo_target_voices(
    *,
    job: MimoClipJob,
    record: MimoRecord,
    subject_index_by_entity: dict[str, int],
    existing_entity_ids: set[str],
    reference_capacity: int,
    temporary_root: Path,
    final_root: Path,
    audio_backend: AudioMediaBackend,
    analyzer: RecoveredVoiceAudioAnalyzer,
    policy: MimoRecoveredVoiceQualityPolicy | None = None,
) -> tuple[
    list[MimoRecoveredVoiceReference], list[FinalSubjectVoice], list[FinalSubjectVoice]
]:
    if record.annotation is None:
        return [], [], []
    active_policy = policy or MimoRecoveredVoiceQualityPolicy()
    source = Path(job.target_full_audio_path).expanduser().resolve(strict=True)
    decisions = {item.segment_id: item for item in record.annotation.segment_decisions}
    candidate_segments = [
        segment
        for segment in job.segments
        if decisions[segment.segment_id].entity_id is not None
        and decisions[segment.segment_id].entity_id not in existing_entity_ids
    ]
    if not candidate_segments:
        return [], [], []
    analysis = analyzer.load(source)
    if (
        analysis.probe.sample_rate_hz != CANONICAL_AUDIO_SAMPLE_RATE_HZ
        or analysis.probe.channels != CANONICAL_AUDIO_CHANNELS
        or "flac" not in analysis.probe.format_name.lower()
        or abs(analysis.probe.duration_seconds - job.target_duration_seconds)
        > CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS
        or analysis.mono_pcm16.ndim != 1
        or analysis.mono_pcm16.dtype != np.dtype("int16")
        or analysis.mono_pcm16.size != analysis.probe.frame_count
    ):
        raise ValueError(
            "MiMo recovered voice source must be canonical 32 kHz stereo FLAC"
        )
    occupied = sorted(
        (item.source_start_sample, item.source_end_sample) for item in job.segments
    )
    provisional: list[MimoRecoveredVoiceReference] = []
    for segment in candidate_segments:
        decision = decisions[segment.segment_id]
        assert decision.entity_id is not None
        reasons = _semantic_reasons(decision)
        sample_valid = (
            segment.source_sample_rate_hz == CANONICAL_AUDIO_SAMPLE_RATE_HZ
            and segment.source_start_sample
            == round(segment.start_time * CANONICAL_AUDIO_SAMPLE_RATE_HZ)
            and segment.source_end_sample
            == round(segment.end_time * CANONICAL_AUDIO_SAMPLE_RATE_HZ)
            and segment.source_end_sample <= analysis.probe.frame_count
        )
        if not sample_valid:
            reasons.append("invalid_canonical_sample_range")
        metrics = None
        if not reasons:
            metrics = _quality_metrics(
                analysis,
                start_sample=segment.source_start_sample,
                end_sample=segment.source_end_sample,
                occupied_ranges=occupied,
            )
            reasons.extend(_quality_reasons(metrics, active_policy))
        reasons = [code for code in _REASON_ORDER if code in reasons]
        provisional.append(
            MimoRecoveredVoiceReference(
                clip_uid=job.clip_uid,
                entity_id=decision.entity_id,
                subject_index=subject_index_by_entity[decision.entity_id],
                speaker_group=decision.primary_speaker_group,
                source_segment_id=segment.segment_id,
                source_start=segment.start_time,
                source_end=segment.end_time,
                source_start_sample=segment.source_start_sample,
                source_end_sample=segment.source_end_sample,
                mimo_record_fingerprint=record.record_fingerprint,
                evidence_codes=list(decision.evidence_codes),
                confidence=decision.confidence,
                quality_metrics=metrics,
                quality_policy_fingerprint=active_policy.fingerprint(),
                status="rejected",
                reason_codes=reasons or ["not_selected_better_candidate"],
            )
        )
    eligible = [
        item
        for item in provisional
        if item.reason_codes == ["not_selected_better_candidate"]
    ]
    selected_by_entity: dict[str, MimoRecoveredVoiceReference] = {}
    for candidate in sorted(eligible, key=_candidate_rank):
        selected_by_entity.setdefault(candidate.entity_id, candidate)
    selected_ids = {
        item.source_segment_id
        for item in sorted(
            selected_by_entity.values(),
            key=lambda item: (item.subject_index, item.entity_id),
        )[: max(0, reference_capacity)]
    }
    capacity_eligible_ids = {
        item.source_segment_id for item in selected_by_entity.values()
    }
    final_records: list[MimoRecoveredVoiceReference] = []
    temporary_voices: list[FinalSubjectVoice] = []
    published_voices: list[FinalSubjectVoice] = []
    for candidate in provisional:
        reasons = list(candidate.reason_codes)
        if candidate.source_segment_id in selected_ids:
            relative = (
                Path("recovered_voice_refs")
                / job.clip_uid
                / f"{candidate.entity_id}.flac"
            )
            temporary_path = temporary_root / relative
            published_path = final_root / relative
            audio_backend.extract_voice_reference(
                clip_uid=job.clip_uid,
                entity_id=candidate.entity_id,
                full_audio_path=source,
                start_time=candidate.source_start,
                end_time=candidate.source_end,
                destination=temporary_path,
                sample_rate_hz=CANONICAL_AUDIO_SAMPLE_RATE_HZ,
                channels=CANONICAL_AUDIO_CHANNELS,
                output_format="flac",
                source_audio_path=source,
                source_start_sample=candidate.source_start_sample,
                source_end_sample=candidate.source_end_sample,
            )
            digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
            selected = _replace_reference(
                candidate,
                status="selected",
                reason_codes=[],
                output_path=str(published_path),
                output_sha256=digest,
            )
            common = {
                "subject_index": selected.subject_index,
                "entity_id": selected.entity_id,
                "target_occurrence_id": f"{job.clip_uid}/{selected.entity_id}",
                "voice_reference_sha256": digest,
                "source_start": selected.source_start,
                "source_end": selected.source_end,
                "source_start_sample": selected.source_start_sample,
                "source_end_sample": selected.source_end_sample,
                "sample_mapping_policy": VOICE_SAMPLE_MAPPING_POLICY,
                "voice_source": "target",
            }
            temporary_voices.append(
                FinalSubjectVoice(voice_reference_path=str(temporary_path), **common)
            )
            published_voices.append(
                FinalSubjectVoice(voice_reference_path=str(published_path), **common)
            )
            final_records.append(selected)
        else:
            if candidate.source_segment_id in capacity_eligible_ids:
                reasons = ["reference_limit"]
            final_records.append(_replace_reference(candidate, reason_codes=reasons))
    final_records.sort(
        key=lambda item: (item.subject_index, item.source_start, item.source_segment_id)
    )
    temporary_voices.sort(key=lambda item: (item.subject_index, item.entity_id))
    published_voices.sort(key=lambda item: (item.subject_index, item.entity_id))
    return final_records, temporary_voices, published_voices


__all__ = [
    "RECOVERED_VOICE_POLICY_VERSION",
    "CanonicalAudioAnalysis",
    "FFmpegRecoveredVoiceAudioAnalyzer",
    "MimoRecoveredVoiceQualityPolicy",
    "MimoRecoveredVoiceReference",
    "RecoveredVoiceAudioAnalyzer",
    "recover_mimo_target_voices",
]
