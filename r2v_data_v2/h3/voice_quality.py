from __future__ import annotations

import json
import math
import sys
import uuid
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

_LOCAL_NOISE_CONTEXT_SECONDS = 2.0
_MINIMUM_LOCAL_NOISE_SECONDS = 0.20
_LOCAL_NOISE_WINDOW_SECONDS = 0.020


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


def _local_noise_metrics(
    *,
    audio_samples: Sequence[int],
    sidecar: AudioBindingSidecar,
    turn_start_time: float,
    turn_end_time: float,
    sample_rate_hz: int,
) -> dict[str, bool | float | int | None]:
    search_ranges = (
        (max(0.0, turn_start_time - _LOCAL_NOISE_CONTEXT_SECONDS), turn_start_time),
        (turn_end_time, turn_end_time + _LOCAL_NOISE_CONTEXT_SECONDS),
    )
    sample_spans: list[tuple[int, int]] = []
    for binding in sidecar.bindings:
        if binding.status != "no_speech":
            continue
        for search_start, search_end in search_ranges:
            overlap_start = max(binding.start_time, search_start)
            overlap_end = min(binding.end_time, search_end)
            if overlap_end <= overlap_start:
                continue
            start_sample = max(0, round(overlap_start * sample_rate_hz))
            end_sample = min(
                len(audio_samples),
                round(overlap_end * sample_rate_hz),
            )
            if end_sample > start_sample:
                sample_spans.append((start_sample, end_sample))

    sample_count = sum(end - start for start, end in sample_spans)
    duration_seconds = sample_count / sample_rate_hz
    minimum_sample_count = math.ceil(
        _MINIMUM_LOCAL_NOISE_SECONDS * sample_rate_hz
    )
    if sample_count < minimum_sample_count:
        return {
            "local_noise_context_available": False,
            "local_noise_sample_count": sample_count,
            "local_noise_duration_seconds": duration_seconds,
            "local_noise_rms_amplitude": None,
            "local_noise_rms_dbfs": None,
            "estimated_snr_db": None,
        }

    window_sample_count = max(
        1,
        round(_LOCAL_NOISE_WINDOW_SECONDS * sample_rate_hz),
    )
    window_rms_values = []
    for span_start, span_end in sample_spans:
        for window_start in range(span_start, span_end, window_sample_count):
            window_end = min(span_end, window_start + window_sample_count)
            window_metrics = _audio_metrics(audio_samples[window_start:window_end])
            window_rms_values.append(float(window_metrics["rms_amplitude"]))
    robust_rms = _percentile(window_rms_values, 0.50)
    robust_dbfs = _dbfs(robust_rms)
    return {
        "local_noise_context_available": True,
        "local_noise_sample_count": sample_count,
        "local_noise_duration_seconds": duration_seconds,
        "local_noise_rms_amplitude": robust_rms,
        "local_noise_rms_dbfs": robust_dbfs,
        "estimated_snr_db": None,
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
        local_noise = _local_noise_metrics(
            audio_samples=audio_samples,
            sidecar=sidecar,
            turn_start_time=turn.start_time,
            turn_end_time=turn.end_time,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
        if metrics["rms_dbfs"] is not None and local_noise["local_noise_rms_dbfs"] is not None:
            local_noise["estimated_snr_db"] = float(metrics["rms_dbfs"]) - float(
                local_noise["local_noise_rms_dbfs"]
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
                local_noise_context_available=bool(
                    local_noise["local_noise_context_available"]
                ),
                local_noise_sample_count=int(
                    local_noise["local_noise_sample_count"]
                ),
                local_noise_duration_seconds=float(
                    local_noise["local_noise_duration_seconds"]
                ),
                local_noise_rms_amplitude=(
                    None
                    if local_noise["local_noise_rms_amplitude"] is None
                    else float(local_noise["local_noise_rms_amplitude"])
                ),
                local_noise_rms_dbfs=(
                    None
                    if local_noise["local_noise_rms_dbfs"] is None
                    else float(local_noise["local_noise_rms_dbfs"])
                ),
                estimated_snr_db=(
                    None
                    if local_noise["estimated_snr_db"] is None
                    else float(local_noise["estimated_snr_db"])
                ),
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
    "local_noise_sample_count",
    "local_noise_duration_seconds",
    "local_noise_rms_amplitude",
    "local_noise_rms_dbfs",
    "estimated_snr_db",
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
        noise_context_available_count=sum(
            turn.local_noise_context_available for turn in turns
        ),
        noise_context_unavailable_count=sum(
            not turn.local_noise_context_available for turn in turns
        ),
        noise_context_availability_rate=(
            sum(turn.local_noise_context_available for turn in turns) / len(turns)
            if turns
            else 0.0
        ),
        metric_distributions=distributions,
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_document(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _json_lines(values: Sequence[dict[str, object]]) -> str:
    return "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
        for value in values
    )


def recompute_voice_reference_quality_artifacts(
    *,
    pilot_root: Path,
) -> VoiceReferenceQualityPilotReport:
    root = pilot_root.expanduser().resolve(strict=True)
    if not (root / "summary.json").is_file():
        raise ValueError("voice-quality postprocess requires a pilot summary.json")
    clip_paths = sorted((root / "clips").glob("*/audio_binding.json"))
    if not clip_paths:
        raise ValueError("voice-quality postprocess found no audio binding artifacts")

    reports: list[VoiceReferenceClipDiagnostics] = []
    for sidecar_path in clip_paths:
        clip_uid = sidecar_path.parent.name
        native_path = root / "runtime" / clip_uid / "lr_asd" / "lr_asd_native.json"
        review_dir = root / "review" / clip_uid
        if not native_path.is_file() or not review_dir.is_dir():
            raise ValueError(
                f"voice-quality postprocess artifacts are incomplete for {clip_uid}"
            )
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        native = LRASDNativeArtifact.model_validate_json(
            native_path.read_text(encoding="utf-8")
        )
        if sidecar.clip_uid != clip_uid or native.clip_uid != clip_uid:
            raise ValueError("voice-quality postprocess clip identities do not match")
        source_audio_path = Path(native.audio_path).expanduser()
        if not source_audio_path.is_absolute():
            source_audio_path = root / source_audio_path
        try:
            report = build_voice_reference_quality_diagnostics(
                native=native,
                sidecar=sidecar,
                source_audio_path=source_audio_path.resolve(strict=True),
                published_audio_path=native.audio_path,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics remain non-gating
            report = VoiceReferenceClipDiagnostics(
                clip_uid=clip_uid,
                source_audio_path=native.audio_path,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
            )
        reports.append(report)

    summary = build_voice_reference_quality_report(reports)
    turn_payloads = [
        turn.model_dump(mode="json")
        for report in reports
        for turn in report.candidate_turns
    ]
    for report in reports:
        payload = _json_document(report.model_dump(mode="json"))
        _atomic_write(
            root / "clips" / report.clip_uid / "voice_reference_quality.json",
            payload,
        )
        _atomic_write(
            root / "review" / report.clip_uid / "voice_reference_quality.json",
            payload,
        )
    _atomic_write(
        root / "voice_reference_quality.jsonl",
        _json_lines(turn_payloads),
    )
    _atomic_write(
        root / "voice_reference_quality_summary.json",
        _json_document(summary.model_dump(mode="json")),
    )
    return summary
