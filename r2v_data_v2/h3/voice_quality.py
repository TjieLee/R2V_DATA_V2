from __future__ import annotations

import math
import sys
import wave
from array import array
from collections.abc import Sequence
from pathlib import Path

from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    coalesce_audio_bindings,
)
from r2v_data_v2.h3.pilot_schemas import (
    AssociationConfidenceDiagnostics,
    LRASDNativeArtifact,
    LRASDScoreDiagnostics,
    VoiceQualityMetricDistribution,
    VoiceReferenceClipDiagnostics,
    VoiceReferenceQualityPilotReport,
    VoiceReferenceTurnDiagnostics,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("percentile requires values and a quantile in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _read_pcm16_mono(path: Path, *, expected_sample_rate: int) -> array[int]:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != expected_sample_rate
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("voice-quality diagnostics require 16 kHz mono PCM16 WAV")
        frames = source.readframes(source.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _dbfs(amplitude: float) -> float | None:
    return 20 * math.log10(amplitude) if amplitude > 0 else None


def _audio_metrics(samples: Sequence[int]) -> dict[str, float | int | None]:
    if not samples:
        raise ValueError("voice-quality turn has no audio samples")
    sample_count = len(samples)
    rms_amplitude = math.sqrt(
        math.fsum(float(value) * value for value in samples) / sample_count
    ) / 32768.0
    peak_amplitude = max(abs(value) for value in samples) / 32768.0
    clipping_ratio = sum(value in {-32768, 32767} for value in samples) / sample_count
    return {
        "sample_count": sample_count,
        "rms_amplitude": rms_amplitude,
        "rms_dbfs": _dbfs(rms_amplitude),
        "peak_amplitude": peak_amplitude,
        "peak_dbfs": _dbfs(peak_amplitude),
        "clipping_ratio": clipping_ratio,
    }


def build_voice_reference_quality_diagnostics(
    *,
    native: LRASDNativeArtifact,
    sidecar: AudioBindingSidecar,
    source_audio_path: Path,
    published_audio_path: str,
) -> VoiceReferenceClipDiagnostics:
    if sidecar.status != "ready":
        raise ValueError("voice-quality diagnostics require a ready audio sidecar")
    policy = AudioBindingProductionConfig()
    turns = coalesce_audio_bindings(
        sidecar.bindings,
        clip_uid=native.clip_uid,
        sample_rate_hz=native.audio_sample_rate_hz,
        maximum_gap_seconds=policy.speech_merge_gap_seconds,
        minimum_voice_reference_duration_seconds=(
            policy.minimum_voice_reference_duration_seconds
        ),
        frame_rate=native.model_fps,
    )
    candidates = [turn for turn in turns if turn.voice_reference_eligible]
    if not candidates:
        return VoiceReferenceClipDiagnostics(
            clip_uid=native.clip_uid,
            source_audio_path=published_audio_path,
            status="ready",
        )
    audio_samples = _read_pcm16_mono(
        source_audio_path,
        expected_sample_rate=native.audio_sample_rate_hz,
    )
    track_by_id = {track.face_track_id: track for track in native.tracks}
    binding_by_id = {
        f"binding_{index}": binding
        for index, binding in enumerate(sidecar.bindings, start=1)
    }
    diagnostics = []
    for turn in candidates:
        assert turn.entity_id is not None
        assert turn.face_track_id is not None
        if turn.end_sample > len(audio_samples):
            raise ValueError("voice-quality turn exceeds the LR-ASD audio extent")
        track = track_by_id.get(turn.face_track_id)
        if track is None:
            raise ValueError("voice-quality turn references an unknown face track")
        scores = [
            sample.raw_class1_logit
            for sample in track.samples
            if turn.start_time <= sample.timestamp_seconds < turn.end_time
        ]
        if not scores:
            raise ValueError("voice-quality turn has no LR-ASD score samples")
        confidences = [
            binding_by_id[binding_id].evidence.association_confidence
            for binding_id in turn.source_binding_ids
        ]
        if any(value is None for value in confidences):
            raise ValueError("voice-quality turn has no association confidence")
        numeric_confidences = [float(value) for value in confidences if value is not None]
        metrics = _audio_metrics(
            audio_samples[turn.start_sample : turn.end_sample]
        )
        diagnostics.append(
            VoiceReferenceTurnDiagnostics(
                clip_uid=native.clip_uid,
                turn_id=turn.turn_id,
                entity_id=turn.entity_id,
                face_track_id=turn.face_track_id,
                start_time=turn.start_time,
                end_time=turn.end_time,
                duration_seconds=turn.end_time - turn.start_time,
                sample_count=int(metrics["sample_count"]),
                rms_amplitude=float(metrics["rms_amplitude"]),
                rms_dbfs=(
                    None
                    if metrics["rms_dbfs"] is None
                    else float(metrics["rms_dbfs"])
                ),
                peak_amplitude=float(metrics["peak_amplitude"]),
                peak_dbfs=(
                    None
                    if metrics["peak_dbfs"] is None
                    else float(metrics["peak_dbfs"])
                ),
                clipping_ratio=float(metrics["clipping_ratio"]),
                lr_asd_raw_native_score=LRASDScoreDiagnostics(
                    mean=math.fsum(scores) / len(scores),
                    min=min(scores),
                    p10=_percentile(scores, 0.10),
                ),
                association_confidence=AssociationConfidenceDiagnostics(
                    mean=math.fsum(numeric_confidences) / len(numeric_confidences),
                    min=min(numeric_confidences),
                ),
            )
        )
    return VoiceReferenceClipDiagnostics(
        clip_uid=native.clip_uid,
        source_audio_path=published_audio_path,
        status="ready",
        candidate_turns=diagnostics,
    )


_DISTRIBUTION_FIELDS = (
    "duration_seconds",
    "sample_count",
    "rms_amplitude",
    "rms_dbfs",
    "peak_amplitude",
    "peak_dbfs",
    "clipping_ratio",
    "lr_asd_raw_native_score_mean",
    "lr_asd_raw_native_score_min",
    "lr_asd_raw_native_score_p10",
    "association_confidence_mean",
    "association_confidence_min",
)


def _turn_metric(
    turn: VoiceReferenceTurnDiagnostics,
    name: str,
) -> float | None:
    if name.startswith("lr_asd_raw_native_score_"):
        return float(
            getattr(
                turn.lr_asd_raw_native_score,
                name.removeprefix("lr_asd_raw_native_score_"),
            )
        )
    if name.startswith("association_confidence_"):
        return float(
            getattr(turn.association_confidence, name.removeprefix("association_confidence_"))
        )
    value = getattr(turn, name)
    return None if value is None else float(value)


def _distribution(values: Sequence[float]) -> VoiceQualityMetricDistribution:
    if not values:
        return VoiceQualityMetricDistribution(count=0)
    return VoiceQualityMetricDistribution(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=math.fsum(values) / len(values),
        p10=_percentile(values, 0.10),
        p50=_percentile(values, 0.50),
        p90=_percentile(values, 0.90),
    )


def build_voice_reference_quality_report(
    clip_reports: Sequence[VoiceReferenceClipDiagnostics],
) -> VoiceReferenceQualityPilotReport:
    ordered = sorted(clip_reports, key=lambda item: item.clip_uid)
    turns = [
        turn
        for report in ordered
        for turn in report.candidate_turns
    ]
    distributions = {}
    for name in _DISTRIBUTION_FIELDS:
        values = [
            value
            for turn in turns
            if (value := _turn_metric(turn, name)) is not None
        ]
        distributions[name] = _distribution(values)
    return VoiceReferenceQualityPilotReport(
        clip_report_count=len(ordered),
        diagnostics_ready_clip_count=sum(
            report.status == "ready" for report in ordered
        ),
        diagnostics_failed_clip_count=sum(
            report.status == "failed" for report in ordered
        ),
        clips_with_candidate_turns=sum(bool(report.candidate_turns) for report in ordered),
        candidate_turn_count=len(turns),
        metric_distributions=distributions,
    )
