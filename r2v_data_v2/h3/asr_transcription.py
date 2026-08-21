from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import uuid
import wave
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import Field, StrictFloat, model_validator

from r2v_data_v2.h3.audio_binding import (
    AudioBindingProductionConfig,
    coalesce_audio_bindings,
)
from r2v_data_v2.h3.audio_production import (
    PRODUCTION_PAIR_VERSION,
    H3ProductionInPair,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel

ASR_TURN_SCHEMA_VERSION = "r2v.h3.asr_turn.1"
ASR_INVENTORY_SCHEMA_VERSION = "r2v.h3.asr_inventory.1"
ASR_SUMMARY_SCHEMA_VERSION = "r2v.h3.asr_summary.1"
ASR_HUMAN_QA_SCHEMA_VERSION = "r2v.h3.asr_human_qa.1"
ASR_REQUEST_CONTRACT_VERSION = "h3_whisper_authoritative_turn_asr_v1"
ASR_PREPROCESSING_VERSION = "pcm16_exact_turn_crop_v1"
CURRENT_BOUNDARY_SOURCE = "frozen_audio_binding_turns_v1"
CURRENT_ENTITY_BINDING_SOURCE = "lr_asd_visual_entity_binding_v1"
DEFAULT_ASR_MODEL = "large-v3"
DEFAULT_ASR_DEVICE = "cuda:0"
DEFAULT_ASR_COMPUTE_TYPE = "float16"
ASR_INPUT_SAMPLE_RATE_HZ = 16000
PILOT_TARGET_COUNT = 20
ASR_HUMAN_QA_LABELS = ("CORRECT", "WRONG", "UNCERTAIN")

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"ASR JSONL line {line_number} must be an object")
        values.append(value)
    return values


class ASRTargetClip(SchemaModel):
    target_clip_uid: str
    target_video_path: str
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    subject_count: int = Field(gt=0)
    turn_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_target(self) -> ASRTargetClip:
        if _SAFE_ID.fullmatch(self.target_clip_uid) is None:
            raise ValueError("ASR target clip UID is not path-safe")
        for value in (
            self.target_video_path,
            self.source_audio_path,
            self.target_audio_binding_path,
        ):
            if not value.strip():
                raise ValueError("ASR target provenance path must not be empty")
        return self


class ASRTurnSegmentationProvenance(SchemaModel):
    boundary_source: str
    source_segment_id: str
    speaker_cluster_id: str | None = None
    entity_binding_source: str

    @model_validator(mode="after")
    def validate_provenance(self) -> ASRTurnSegmentationProvenance:
        required = (
            self.boundary_source,
            self.source_segment_id,
            self.entity_binding_source,
        )
        if any(not value.strip() for value in required):
            raise ValueError("ASR turn segmentation provenance must not be empty")
        if (
            self.speaker_cluster_id is not None
            and not self.speaker_cluster_id.strip()
        ):
            raise ValueError("ASR speaker cluster ID must be non-empty or null")
        return self


class ASRTurnJob(SchemaModel):
    target_clip_uid: str
    turn_id: str
    entity_id: str
    entity_occurrence_id: str
    start_time: StrictFloat = Field(ge=0)
    end_time: StrictFloat = Field(gt=0)
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    segment_provenance: ASRTurnSegmentationProvenance

    @model_validator(mode="after")
    def validate_job(self) -> ASRTurnJob:
        if self.entity_occurrence_id != f"{self.target_clip_uid}/{self.entity_id}":
            raise ValueError("ASR turn entity occurrence identity is inconsistent")
        if not self.turn_id.strip() or not self.entity_id.strip():
            raise ValueError("ASR turn identity must not be empty")
        if self.end_time <= self.start_time:
            raise ValueError("ASR turn interval must be positive")
        if self.source_start_sample != round(
            self.start_time * self.source_sample_rate_hz
        ) or self.source_end_sample != round(
            self.end_time * self.source_sample_rate_hz
        ):
            raise ValueError("ASR source sample range must match authoritative times")
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("ASR source sample range must be positive")
        if not self.source_audio_path.strip():
            raise ValueError("ASR source audio path must not be empty")
        return self


class ASRInventory(SchemaModel):
    schema_version: Literal["r2v.h3.asr_inventory.1"] = (
        ASR_INVENTORY_SCHEMA_VERSION
    )
    source_pair_schema_version: Literal["r2v.h3.production_pairs.2"] = (
        "r2v.h3.production_pairs.2"
    )
    source_pairs_path: str
    source_pairs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20", "production"]
    source_target_count: int = Field(ge=0)
    selected_target_count: int = Field(ge=0)
    selected_turn_count: int = Field(ge=0)
    selection_mode: Literal[
        "multi_subject_first_then_clip_uid_v1",
        "complete_target_inventory_v1",
    ]
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    targets: list[ASRTargetClip]
    jobs: list[ASRTurnJob]

    @model_validator(mode="after")
    def validate_inventory(self) -> ASRInventory:
        clip_ids = [item.target_clip_uid for item in self.targets]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("ASR inventory targets must be unique")
        if self.selected_target_count != len(self.targets):
            raise ValueError("ASR selected target count is inconsistent")
        if self.selected_turn_count != len(self.jobs):
            raise ValueError("ASR selected turn count is inconsistent")
        if self.source_target_count < self.selected_target_count:
            raise ValueError("ASR selection cannot exceed source targets")
        job_ids = [(item.target_clip_uid, item.turn_id) for item in self.jobs]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("ASR inventory turn jobs must be unique")
        target_ids = set(clip_ids)
        if any(item.target_clip_uid not in target_ids for item in self.jobs):
            raise ValueError("ASR turn job references an unknown target")
        turn_counts = Counter(item.target_clip_uid for item in self.jobs)
        if any(
            target.turn_count != turn_counts[target.target_clip_uid]
            for target in self.targets
        ):
            raise ValueError("ASR per-target turn count is inconsistent")
        if self.mode == "production":
            if (
                self.selection_mode != "complete_target_inventory_v1"
                or self.bounded_selection_applied
                or self.selected_target_count != self.source_target_count
            ):
                raise ValueError("production ASR must cover every target clip")
        elif self.selection_mode != "multi_subject_first_then_clip_uid_v1":
            raise ValueError("ASR pilot must use deterministic pilot selection")
        return self


class ASRPreprocessingProvenance(SchemaModel):
    version: Literal["pcm16_exact_turn_crop_v1"] = ASR_PREPROCESSING_VERSION
    source_encoding: Literal["pcm_s16le_wave"] = "pcm_s16le_wave"
    source_sample_rate_hz: int = Field(gt=0)
    source_channels: int = Field(gt=0)
    asr_input_sample_rate_hz: Literal[16000] = ASR_INPUT_SAMPLE_RATE_HZ
    asr_input_channels: Literal[1] = 1
    resampled: bool
    downmixed: bool
    padding_seconds: Literal[0.0] = 0.0
    denoising_applied: Literal[False] = False
    enhancement_applied: Literal[False] = False
    vad_resegmentation_applied: Literal[False] = False


class ASRBackendProvenance(SchemaModel):
    backend: Literal["faster_whisper"] = "faster_whisper"
    model_identifier: str
    checkpoint_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    device: str
    compute_type: str
    task: Literal["transcribe"] = "transcribe"
    condition_on_previous_text: Literal[False] = False
    vad_filter: Literal[False] = False
    word_timestamps: Literal[False] = False
    local_files_only: Literal[True] = True
    request_contract_version: Literal["h3_whisper_authoritative_turn_asr_v1"] = (
        ASR_REQUEST_CONTRACT_VERSION
    )
    preprocessing_version: Literal["pcm16_exact_turn_crop_v1"] = (
        ASR_PREPROCESSING_VERSION
    )
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> ASRBackendProvenance:
        if (
            not self.model_identifier.strip()
            or not self.device.strip()
            or not self.compute_type.strip()
        ):
            raise ValueError("ASR backend identity must not be empty")
        expected = _sha256_text(
            _compact_json(
                {
                    "backend": self.backend,
                    "model_identifier": self.model_identifier,
                    "checkpoint_fingerprint": self.checkpoint_fingerprint,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "task": self.task,
                    "condition_on_previous_text": self.condition_on_previous_text,
                    "vad_filter": self.vad_filter,
                    "word_timestamps": self.word_timestamps,
                    "local_files_only": self.local_files_only,
                    "request_contract_version": self.request_contract_version,
                    "preprocessing_version": self.preprocessing_version,
                }
            )
        )
        if self.configuration_fingerprint != expected:
            raise ValueError("ASR backend configuration fingerprint is invalid")
        return self


@dataclass(frozen=True)
class WhisperASRConfig:
    model_identifier: str = DEFAULT_ASR_MODEL
    checkpoint_fingerprint: str | None = None
    device: str = DEFAULT_ASR_DEVICE
    compute_type: str = DEFAULT_ASR_COMPUTE_TYPE

    def __post_init__(self) -> None:
        if (
            not self.model_identifier.strip()
            or not self.compute_type.strip()
            or re.fullmatch(r"(?:cuda(?::\d+)?|cpu)", self.device) is None
            or (
                self.checkpoint_fingerprint is not None
                and re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_fingerprint) is None
            )
        ):
            raise ValueError("Whisper ASR model, device, or compute type is invalid")

    def provenance(self) -> ASRBackendProvenance:
        values = {
            "backend": "faster_whisper",
            "model_identifier": self.model_identifier,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "device": self.device,
            "compute_type": self.compute_type,
            "task": "transcribe",
            "condition_on_previous_text": False,
            "vad_filter": False,
            "word_timestamps": False,
            "local_files_only": True,
            "request_contract_version": ASR_REQUEST_CONTRACT_VERSION,
            "preprocessing_version": ASR_PREPROCESSING_VERSION,
        }
        return ASRBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


class ASRDecoderDiagnostics(SchemaModel):
    detected_language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    avg_log_probability: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)
    compression_ratio: float | None = Field(default=None, ge=0)
    decoder_segment_count: int = Field(ge=0)
    aggregation: Literal["arithmetic_mean_across_decoder_segments_v1"] = (
        "arithmetic_mean_across_decoder_segments_v1"
    )

    @model_validator(mode="after")
    def validate_diagnostics(self) -> ASRDecoderDiagnostics:
        values = (
            self.language_probability,
            self.avg_log_probability,
            self.no_speech_probability,
            self.compression_ratio,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("ASR decoder diagnostics must be finite")
        if self.detected_language is not None and not self.detected_language.strip():
            raise ValueError("ASR detected language must be non-empty or null")
        return self


class ASRFailure(SchemaModel):
    code: str
    reason: str

    @model_validator(mode="after")
    def validate_failure(self) -> ASRFailure:
        if not self.code.strip() or not self.reason.strip():
            raise ValueError("ASR failure code and reason must not be empty")
        return self


class ASRTurnRecord(ASRTurnJob):
    schema_version: Literal["r2v.h3.asr_turn.1"] = ASR_TURN_SCHEMA_VERSION
    backend: Literal["faster_whisper"] = "faster_whisper"
    model_identifier: str
    task: Literal["transcribe"] = "transcribe"
    backend_provenance: ASRBackendProvenance
    preprocessing: ASRPreprocessingProvenance
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["transcribed", "uncertain", "failed"]
    text: str | None = None
    language: str | None = None
    diagnostics: ASRDecoderDiagnostics | None = None
    warnings: list[str] = Field(default_factory=list)
    failure: ASRFailure | None = None

    @model_validator(mode="after")
    def validate_record(self) -> ASRTurnRecord:
        if self.model_identifier != self.backend_provenance.model_identifier:
            raise ValueError("ASR record model provenance is inconsistent")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("ASR warnings must not be empty")
        if self.status == "transcribed":
            if self.text is None or not self.text.strip() or self.failure is not None:
                raise ValueError("transcribed ASR requires text and no failure")
        elif self.status == "uncertain":
            if self.text is not None or self.failure is not None:
                raise ValueError("uncertain ASR requires null text and no failure")
        elif self.text is not None or self.failure is None:
            raise ValueError("failed ASR requires null text and failure details")
        return self


class ASRSummary(SchemaModel):
    schema_version: Literal["r2v.h3.asr_summary.1"] = ASR_SUMMARY_SCHEMA_VERSION
    mode: Literal["pilot20", "production"]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend_provenance: ASRBackendProvenance
    source_target_count: int = Field(ge=0)
    target_clip_count: int = Field(ge=0)
    turn_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    uncertain_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    backend_call_count: int = Field(ge=0)
    failure_reason_counts: dict[str, int]
    bounded_selection_applied: bool
    parent_quota_applied: Literal[False] = False
    donor_media_used: Literal[False] = False
    pair_assets_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> ASRSummary:
        if self.turn_count != (
            self.transcribed_count + self.uncertain_count + self.failed_count
        ):
            raise ValueError("ASR status counts must reconcile")
        if self.backend_call_count > self.turn_count:
            raise ValueError("ASR backend calls cannot exceed turn count")
        if self.mode == "production" and (
            self.bounded_selection_applied
            or self.target_clip_count != self.source_target_count
        ):
            raise ValueError("production ASR summary must cover every target")
        return self


class ASRHumanQACounts(SchemaModel):
    CORRECT: int = Field(ge=0)
    WRONG: int = Field(ge=0)
    UNCERTAIN: int = Field(ge=0)
    UNLABELED: int = Field(ge=0)


class ASRHumanQALabel(SchemaModel):
    target_clip_uid: str
    turn_id: str
    entity_occurrence_id: str
    label: Literal["CORRECT", "WRONG", "UNCERTAIN"]

    @model_validator(mode="after")
    def validate_label(self) -> ASRHumanQALabel:
        if any(
            not value.strip()
            for value in (
                self.target_clip_uid,
                self.turn_id,
                self.entity_occurrence_id,
            )
        ):
            raise ValueError("ASR human-QA label identity must not be empty")
        return self


class ASRHumanQAExport(SchemaModel):
    schema_version: Literal["r2v.h3.asr_human_qa.1"] = (
        ASR_HUMAN_QA_SCHEMA_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["pilot20", "production"]
    label_count: int = Field(ge=0)
    total_turn_count: int = Field(ge=0)
    counts: ASRHumanQACounts
    labels: list[ASRHumanQALabel]

    @model_validator(mode="after")
    def validate_export(self) -> ASRHumanQAExport:
        label_ids = [
            (item.target_clip_uid, item.turn_id)
            for item in self.labels
        ]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("ASR human-QA labels must be unique")
        actual_counts = Counter(item.label for item in self.labels)
        if any(
            getattr(self.counts, label) != actual_counts[label]
            for label in ASR_HUMAN_QA_LABELS
        ):
            raise ValueError("ASR human-QA per-label counts are inconsistent")
        labeled_count = (
            self.counts.CORRECT
            + self.counts.WRONG
            + self.counts.UNCERTAIN
        )
        if self.label_count != len(self.labels) or self.label_count != labeled_count:
            raise ValueError("ASR human-QA label count is inconsistent")
        if self.total_turn_count != self.label_count + self.counts.UNLABELED:
            raise ValueError("ASR human-QA total count is inconsistent")
        return self


@dataclass(frozen=True)
class ASRBackendResult:
    text: str | None
    language: str | None
    diagnostics: ASRDecoderDiagnostics
    warnings: tuple[str, ...] = ()


class ASRTranscriptionFailure(RuntimeError):
    def __init__(self, *, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class ASRBackend(Protocol):
    @property
    def provenance(self) -> ASRBackendProvenance: ...

    def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
    ) -> ASRBackendResult: ...


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(math.fsum(values) / len(values))


def _device_parts(device: str) -> tuple[str, int]:
    if device == "cpu":
        return "cpu", 0
    return "cuda", int(device.split(":", 1)[1]) if ":" in device else 0


class FasterWhisperASRBackend:
    def __init__(
        self,
        config: WhisperASRConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        if model_factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required in the dedicated ASR environment"
                ) from exc
            model_factory = WhisperModel
        device, device_index = _device_parts(config.device)
        try:
            self.model = model_factory(
                config.model_identifier,
                device=device,
                device_index=device_index,
                compute_type=config.compute_type,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Whisper ASR on explicit device {config.device}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @property
    def provenance(self) -> ASRBackendProvenance:
        return self.config.provenance()

    def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate_hz: int,
    ) -> ASRBackendResult:
        if sample_rate_hz != ASR_INPUT_SAMPLE_RATE_HZ:
            raise ValueError("faster-whisper input must be 16 kHz")
        if audio.ndim != 1 or audio.dtype != np.float32 or not np.isfinite(audio).all():
            raise ValueError("faster-whisper input must be finite mono float32")
        try:
            segments_iter, info = self.model.transcribe(
                audio,
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
            )
            segments = list(segments_iter)
            text = "".join(
                str(getattr(item, "text", "")) for item in segments
            ).strip()
            language = getattr(info, "language", None)
            if not isinstance(language, str) or not language.strip():
                language = None
            language_probability = getattr(info, "language_probability", None)
            diagnostics = ASRDecoderDiagnostics(
                detected_language=language,
                language_probability=(
                    None
                    if language_probability is None
                    else float(language_probability)
                ),
                avg_log_probability=_mean(
                    [float(item.avg_logprob) for item in segments]
                ),
                no_speech_probability=_mean(
                    [float(item.no_speech_prob) for item in segments]
                ),
                compression_ratio=_mean(
                    [float(item.compression_ratio) for item in segments]
                ),
                decoder_segment_count=len(segments),
            )
        except Exception as exc:
            raise ASRTranscriptionFailure(
                code="asr_backend_failed",
                reason=f"{type(exc).__name__}: {exc}",
            ) from exc
        return ASRBackendResult(
            text=text or None,
            language=language,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class PreparedASRAudio:
    waveform: np.ndarray
    preprocessing: ASRPreprocessingProvenance


class ExactASRAudioJob(Protocol):
    source_audio_path: str
    source_audio_sha256: str
    source_sample_rate_hz: int
    source_channels: int
    source_start_sample: int
    source_end_sample: int


@dataclass(frozen=True)
class PreparedExactASRWaveform:
    waveform: np.ndarray
    source_sample_rate_hz: int
    source_channels: int
    resampled: bool
    downmixed: bool


def _verify_source_audio(job: ExactASRAudioJob) -> Path:
    path = Path(job.source_audio_path)
    if not path.is_file():
        raise ASRTranscriptionFailure(
            code="source_audio_changed",
            reason="canonical full audio is unavailable after inventory construction",
        )
    if _sha256_file(path) != job.source_audio_sha256:
        raise ASRTranscriptionFailure(
            code="source_audio_changed",
            reason="canonical full-audio bytes changed after inventory construction",
        )
    return path


def prepare_exact_asr_waveform(
    job: ExactASRAudioJob,
    *,
    unit_label: Literal["turn", "segment"],
) -> PreparedExactASRWaveform:
    source_path = _verify_source_audio(job)
    try:
        with wave.open(str(source_path), "rb") as source:
            sample_rate = source.getframerate()
            channels = source.getnchannels()
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("canonical ASR audio must be uncompressed PCM16 WAV")
            if (
                sample_rate != job.source_sample_rate_hz
                or channels != job.source_channels
            ):
                raise ValueError("canonical ASR audio metadata does not match source")
            if job.source_end_sample > source.getnframes():
                raise ValueError(
                    f"authoritative ASR {unit_label} exceeds canonical audio"
                )
            source.setpos(job.source_start_sample)
            frame_count = job.source_end_sample - job.source_start_sample
            frames = source.readframes(frame_count)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise ASRTranscriptionFailure(
            code="source_audio_invalid",
            reason=f"{type(exc).__name__}: {exc}",
        ) from exc
    expected_bytes = frame_count * channels * 2
    if len(frames) != expected_bytes:
        raise ASRTranscriptionFailure(
            code="source_audio_invalid",
            reason=f"canonical full audio ended before the authoritative {unit_label}",
        )
    pcm = np.frombuffer(frames, dtype="<i2").reshape(frame_count, channels)
    mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
    resampled = sample_rate != ASR_INPUT_SAMPLE_RATE_HZ
    if resampled:
        output_count = round(frame_count * ASR_INPUT_SAMPLE_RATE_HZ / sample_rate)
        if output_count <= 0:
            raise ASRTranscriptionFailure(
                code="source_audio_invalid",
                reason=(
                    f"authoritative ASR {unit_label} is too short after resampling"
                ),
            )
        positions = (
            np.arange(output_count, dtype=np.float64)
            * sample_rate
            / ASR_INPUT_SAMPLE_RATE_HZ
        )
        mono = np.interp(
            positions,
            np.arange(frame_count, dtype=np.float64),
            mono,
        ).astype(np.float32)
    else:
        mono = np.ascontiguousarray(mono, dtype=np.float32)
    return PreparedExactASRWaveform(
        waveform=mono,
        source_sample_rate_hz=sample_rate,
        source_channels=channels,
        resampled=resampled,
        downmixed=channels != 1,
    )


def prepare_asr_audio(job: ASRTurnJob) -> PreparedASRAudio:
    prepared = prepare_exact_asr_waveform(job, unit_label="turn")
    return PreparedASRAudio(
        waveform=prepared.waveform,
        preprocessing=ASRPreprocessingProvenance(
            source_sample_rate_hz=prepared.source_sample_rate_hz,
            source_channels=prepared.source_channels,
            resampled=prepared.resampled,
            downmixed=prepared.downmixed,
        ),
    )


def _write_review_wav(path: Path, waveform: np.ndarray) -> None:
    samples = np.clip(
        np.rint(waveform.astype(np.float64) * 32768.0),
        -32768,
        32767,
    ).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(ASR_INPUT_SAMPLE_RATE_HZ)
        destination.writeframes(samples.tobytes())


def _inventory_fingerprint(
    *,
    source_pairs_sha256: str,
    mode: str,
    targets: Sequence[ASRTargetClip],
    jobs: Sequence[ASRTurnJob],
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "source_pairs_sha256": source_pairs_sha256,
                "mode": mode,
                "targets": [item.model_dump(mode="json") for item in targets],
                "jobs": [item.model_dump(mode="json") for item in jobs],
            }
        )
    )


def build_asr_inventory(
    *,
    pairs_root: Path,
    mode: Literal["pilot20", "production"],
) -> ASRInventory:
    root = pairs_root.expanduser().resolve(strict=True)
    source_path = (root / "in_pairs.jsonl").resolve(strict=True)
    source_path.relative_to(root)
    pairs = [
        H3ProductionInPair.model_validate(item) for item in _read_jsonl(source_path)
    ]
    if not pairs:
        raise ValueError("ASR production requires at least one target in-pair")
    pairs.sort(key=lambda item: item.target_clip_uid)
    clip_ids = [item.target_clip_uid for item in pairs]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("production in-pairs contain duplicate ASR targets")

    all_targets: list[tuple[ASRTargetClip, list[ASRTurnJob]]] = []
    policy = AudioBindingProductionConfig()
    for pair in pairs:
        video_path = Path(pair.target_video_path).expanduser().resolve(strict=True)
        audio_path = Path(pair.target_full_audio_path).expanduser().resolve(strict=True)
        sidecar_path = (
            Path(pair.target_audio_binding_path).expanduser().resolve(strict=True)
        )
        if not video_path.is_file() or not audio_path.is_file() or not sidecar_path.is_file():
            raise ValueError("ASR target media or Audio sidecar is unavailable")
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.clip_uid != pair.target_clip_uid:
            raise ValueError("ASR Audio sidecar clip identity does not match pair")
        if sidecar.status != "ready" or sidecar.evidence is None:
            raise ValueError("ASR requires a ready Audio binding sidecar")
        audio = sidecar.evidence.audio
        if audio.sample_rate_hz is None or audio.channels is None:
            raise ValueError("ASR source audio metadata is incomplete")
        if Path(sidecar.source_video_path).expanduser().resolve(strict=True) != video_path:
            raise ValueError("ASR Audio sidecar source video does not match pair")
        if audio.full_audio_path is None or (
            Path(audio.full_audio_path).expanduser().resolve(strict=True) != audio_path
        ):
            raise ValueError("ASR Audio sidecar full audio does not match pair")
        turns = [
            turn
            for turn in coalesce_audio_bindings(
                sidecar.bindings,
                clip_uid=pair.target_clip_uid,
                sample_rate_hz=audio.sample_rate_hz,
                maximum_gap_seconds=policy.speech_merge_gap_seconds,
                minimum_voice_reference_duration_seconds=(
                    policy.minimum_voice_reference_duration_seconds
                ),
            )
            if turn.status == "bound" and turn.entity_id is not None
        ]
        audio_sha256 = _sha256_file(audio_path)
        target = ASRTargetClip(
            target_clip_uid=pair.target_clip_uid,
            target_video_path=str(video_path),
            source_audio_path=str(audio_path),
            source_audio_sha256=audio_sha256,
            target_audio_binding_path=str(sidecar_path),
            subject_count=len(pair.subjects),
            turn_count=len(turns),
        )
        target_jobs = [
            ASRTurnJob(
                target_clip_uid=pair.target_clip_uid,
                turn_id=turn.turn_id,
                entity_id=str(turn.entity_id),
                entity_occurrence_id=f"{pair.target_clip_uid}/{turn.entity_id}",
                start_time=turn.start_time,
                end_time=turn.end_time,
                source_audio_path=str(audio_path),
                source_audio_sha256=audio_sha256,
                source_sample_rate_hz=audio.sample_rate_hz,
                source_channels=audio.channels,
                source_start_sample=turn.start_sample,
                source_end_sample=turn.end_sample,
                segment_provenance=ASRTurnSegmentationProvenance(
                    boundary_source=CURRENT_BOUNDARY_SOURCE,
                    source_segment_id=turn.turn_id,
                    speaker_cluster_id=None,
                    entity_binding_source=CURRENT_ENTITY_BINDING_SOURCE,
                ),
            )
            for turn in turns
        ]
        all_targets.append((target, target_jobs))

    if mode == "pilot20":
        multi_subject = [item for item in all_targets if item[0].subject_count > 1]
        if len(multi_subject) > PILOT_TARGET_COUNT:
            raise ValueError("ASR pilot20 cannot include every multi-subject target")
        selected_ids = {item[0].target_clip_uid for item in multi_subject}
        remaining = [
            item
            for item in all_targets
            if item[0].target_clip_uid not in selected_ids
        ]
        selected = multi_subject + remaining[: PILOT_TARGET_COUNT - len(multi_subject)]
        selection_mode = "multi_subject_first_then_clip_uid_v1"
        bounded = len(selected) < len(all_targets)
    else:
        selected = all_targets
        selection_mode = "complete_target_inventory_v1"
        bounded = False
    targets = [target for target, _ in selected]
    jobs = [job for _, target_jobs in selected for job in target_jobs]
    source_pairs_sha256 = _sha256_file(source_path)
    return ASRInventory(
        source_pair_schema_version=PRODUCTION_PAIR_VERSION,
        source_pairs_path=str(source_path),
        source_pairs_sha256=source_pairs_sha256,
        inventory_fingerprint=_inventory_fingerprint(
            source_pairs_sha256=source_pairs_sha256,
            mode=mode,
            targets=targets,
            jobs=jobs,
        ),
        mode=mode,
        source_target_count=len(all_targets),
        selected_target_count=len(targets),
        selected_turn_count=len(jobs),
        selection_mode=selection_mode,
        bounded_selection_applied=bounded,
        targets=targets,
        jobs=jobs,
    )


def asr_request_fingerprint(
    job: ASRTurnJob,
    *,
    backend_provenance: ASRBackendProvenance,
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "request_contract_version": ASR_REQUEST_CONTRACT_VERSION,
                "backend_configuration_fingerprint": (
                    backend_provenance.configuration_fingerprint
                ),
                "job": job.model_dump(mode="json"),
            }
        )
    )


def _record(
    *,
    job: ASRTurnJob,
    backend_provenance: ASRBackendProvenance,
    preprocessing: ASRPreprocessingProvenance,
    result: ASRBackendResult,
) -> ASRTurnRecord:
    text = None if result.text is None else result.text.strip() or None
    status: Literal["transcribed", "uncertain"] = (
        "transcribed" if text is not None else "uncertain"
    )
    warnings = list(result.warnings)
    if status == "uncertain":
        warnings.append("empty_transcript")
    return ASRTurnRecord(
        **job.model_dump(mode="python"),
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        preprocessing=preprocessing,
        request_fingerprint=asr_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status=status,
        text=text,
        language=result.language,
        diagnostics=result.diagnostics,
        warnings=warnings,
    )


def _failed_record(
    *,
    job: ASRTurnJob,
    backend_provenance: ASRBackendProvenance,
    failure: ASRTranscriptionFailure,
) -> ASRTurnRecord:
    return ASRTurnRecord(
        **job.model_dump(mode="python"),
        model_identifier=backend_provenance.model_identifier,
        backend_provenance=backend_provenance,
        preprocessing=ASRPreprocessingProvenance(
            source_sample_rate_hz=job.source_sample_rate_hz,
            source_channels=job.source_channels,
            resampled=job.source_sample_rate_hz != ASR_INPUT_SAMPLE_RATE_HZ,
            downmixed=job.source_channels != 1,
        ),
        request_fingerprint=asr_request_fingerprint(
            job,
            backend_provenance=backend_provenance,
        ),
        status="failed",
        text=None,
        language=None,
        diagnostics=None,
        warnings=[failure.code],
        failure=ASRFailure(code=failure.code, reason=failure.reason),
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


def _review_video_name(target: ASRTargetClip) -> str:
    suffix = Path(target.target_video_path).suffix.lower() or ".media"
    return f"{target.target_clip_uid}.video{suffix}"


def _review_audio_name(target: ASRTargetClip) -> str:
    suffix = Path(target.source_audio_path).suffix.lower() or ".media"
    return f"{target.target_clip_uid}.full-audio{suffix}"


def _review_turn_name(job: ASRTurnJob) -> str:
    if _SAFE_ID.fullmatch(job.turn_id) is None:
        raise ValueError("ASR turn ID is not review-path-safe")
    return f"{job.target_clip_uid}/{job.turn_id}.wav"


def _qa_input_name(record: ASRTurnRecord) -> str:
    return f"qa-{record.target_clip_uid}-{record.turn_id}"


def _qa_storage_key(record: ASRTurnRecord) -> str:
    # Preserve the original review-page key so completed browser QA survives refreshes.
    return f"h3-asr-{_qa_input_name(record)}"


def _review_html(
    *,
    inventory: ASRInventory,
    records: Sequence[ASRTurnRecord],
    video_names: dict[str, str],
    audio_names: dict[str, str],
    turn_names: dict[tuple[str, str], str | None],
) -> str:
    export_filename = (
        "asr_pilot20_human_qa.json"
        if inventory.mode == "pilot20"
        else "asr_production_human_qa.json"
    )
    qa_metadata = {
        "schema_version": ASR_HUMAN_QA_SCHEMA_VERSION,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "mode": inventory.mode,
        "total_turn_count": len(records),
        "allowed_labels": list(ASR_HUMAN_QA_LABELS),
        "export_filename": export_filename,
        "turns": [
            {
                "target_clip_uid": record.target_clip_uid,
                "turn_id": record.turn_id,
                "entity_occurrence_id": record.entity_occurrence_id,
                "input_name": _qa_input_name(record),
                "storage_key": _qa_storage_key(record),
            }
            for record in records
        ],
    }
    qa_metadata_json = _compact_json(qa_metadata).replace("<", "\\u003c")
    by_clip: dict[str, list[ASRTurnRecord]] = {}
    for record in records:
        by_clip.setdefault(record.target_clip_uid, []).append(record)
    cards = []
    for target in inventory.targets:
        rows = []
        for record in by_clip.get(target.target_clip_uid, []):
            diagnostics = record.diagnostics
            diagnostic_text = "[unavailable]"
            if diagnostics is not None:
                diagnostic_text = (
                    f"lang_p={diagnostics.language_probability!r}; "
                    f"avg_logprob={diagnostics.avg_log_probability!r}; "
                    f"no_speech={diagnostics.no_speech_probability!r}; "
                    f"compression={diagnostics.compression_ratio!r}; "
                    f"segments={diagnostics.decoder_segment_count}"
                )
            media_name = turn_names[(record.target_clip_uid, record.turn_id)]
            crop_player = "[unavailable]"
            if media_name is not None:
                crop_player = (
                    "<audio controls preload='metadata' "
                    f"src='review_media/{html.escape(media_name)}'></audio>"
                )
            qa_name = _qa_input_name(record)
            qa_key = _qa_storage_key(record)
            rows.append(
                "<tr>"
                f"<td>{html.escape(record.turn_id)}</td>"
                f"<td>{html.escape(record.entity_occurrence_id)}</td>"
                f"<td>{record.start_time:.3f}-{record.end_time:.3f}<br>"
                f"{record.end_time - record.start_time:.3f}s</td>"
                f"<td>{crop_player}</td>"
                f"<td>{html.escape(record.status)}</td>"
                f"<td>{html.escape(record.text or '[null]')}</td>"
                f"<td>{html.escape(record.language or '[null]')}</td>"
                f"<td>{html.escape(diagnostic_text)}</td>"
                "<td>"
                + " ".join(
                    f"<label><input type='radio' name='{html.escape(qa_name)}' "
                    f"data-qa-storage-key='{html.escape(qa_key)}' "
                    f"value='{label}' onchange='saveLabel(this)'>{label}</label>"
                    for label in ASR_HUMAN_QA_LABELS
                )
                + "</td></tr>"
            )
        cards.append(
            f"<article class='case' data-clip='{html.escape(target.target_clip_uid)}'>"
            f"<h2>{html.escape(target.target_clip_uid)}</h2>"
            f"<video controls preload='metadata' src='review_media/{html.escape(video_names[target.target_clip_uid])}'></video>"
            f"<audio controls preload='metadata' src='review_media/{html.escape(audio_names[target.target_clip_uid])}'></audio>"
            "<table><thead><tr><th>turn</th><th>entity occurrence</th><th>time</th>"
            "<th>exact crop</th><th>status</th><th>transcript</th><th>language</th>"
            f"<th>decoder diagnostics</th><th>human QA</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "</article>"
        )
    return (
        """<!doctype html>
<html><head><meta charset="utf-8"><title>H3 Whisper ASR review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f7;color:#171717}
header,.case{max-width:1400px;margin:0 auto 24px;background:white;border:1px solid #ccc;padding:18px}
h1,h2{letter-spacing:0}video{width:100%;max-height:520px;background:#111}audio{width:100%;min-width:180px}
table{width:100%;border-collapse:collapse;margin-top:14px}th,td{border:1px solid #ccc;padding:7px;text-align:left;vertical-align:top}
label{display:block;white-space:nowrap}.qa-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:14px;padding:12px;border:1px solid #bbb;background:#f7f7f7}
.qa-counts{display:flex;flex-wrap:wrap;gap:10px}.qa-actions{display:flex;gap:8px;margin-left:auto}button{padding:7px 11px;cursor:pointer}
</style></head><body>
<header><h1>H3 Whisper-large-v3 ASR review</h1>
<p>Human labels are QA only. Check hallucination, translation, repeated text, language errors, missed speech, and incorrect empty output.</p>
<section class="qa-controls" aria-label="Human QA controls">
<strong id="qa-progress">labeled 0 / 0</strong>
<div class="qa-counts">
<span>CORRECT: <strong id="qa-correct-count">0</strong></span>
<span>WRONG: <strong id="qa-wrong-count">0</strong></span>
<span>UNCERTAIN: <strong id="qa-uncertain-count">0</strong></span>
<span>UNLABELED: <strong id="qa-unlabeled-count">0</strong></span>
</div>
<div class="qa-actions">
<button type="button" onclick="exportQAJSON()">Export QA JSON</button>
<button type="button" onclick="clearQALabels()">Clear QA labels</button>
</div>
</section></header>
"""
        + "".join(cards)
        + f"""
<script>
const qaMetadata = {qa_metadata_json};
const qaAllowedLabels = new Set(qaMetadata.allowed_labels);

function collectQACounts() {{
  const counts = {{CORRECT: 0, WRONG: 0, UNCERTAIN: 0, UNLABELED: 0}};
  qaMetadata.turns.forEach(function(turn) {{
    const label = localStorage.getItem(turn.storage_key);
    if (qaAllowedLabels.has(label)) counts[label] += 1;
    else counts.UNLABELED += 1;
  }});
  return counts;
}}

function updateQACounts() {{
  const counts = collectQACounts();
  const labeled = qaMetadata.total_turn_count - counts.UNLABELED;
  document.getElementById('qa-progress').textContent =
    'labeled ' + labeled + ' / ' + qaMetadata.total_turn_count;
  document.getElementById('qa-correct-count').textContent = counts.CORRECT;
  document.getElementById('qa-wrong-count').textContent = counts.WRONG;
  document.getElementById('qa-uncertain-count').textContent = counts.UNCERTAIN;
  document.getElementById('qa-unlabeled-count').textContent = counts.UNLABELED;
}}

function saveLabel(input) {{
  if (!qaAllowedLabels.has(input.value)) return;
  localStorage.setItem(input.dataset.qaStorageKey, input.value);
  updateQACounts();
}}

function restoreQALabels() {{
  document.querySelectorAll('input[data-qa-storage-key]').forEach(function(input) {{
    const saved = localStorage.getItem(input.dataset.qaStorageKey);
    input.checked = qaAllowedLabels.has(saved) && saved === input.value;
  }});
  updateQACounts();
}}

function exportQAJSON() {{
  const labels = [];
  qaMetadata.turns.forEach(function(turn) {{
    const label = localStorage.getItem(turn.storage_key);
    if (qaAllowedLabels.has(label)) {{
      labels.push({{
        target_clip_uid: turn.target_clip_uid,
        turn_id: turn.turn_id,
        entity_occurrence_id: turn.entity_occurrence_id,
        label: label
      }});
    }}
  }});
  const counts = collectQACounts();
  const payload = {{
    schema_version: qaMetadata.schema_version,
    inventory_fingerprint: qaMetadata.inventory_fingerprint,
    mode: qaMetadata.mode,
    label_count: labels.length,
    total_turn_count: qaMetadata.total_turn_count,
    counts: counts,
    labels: labels
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2) + '\\n'], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = qaMetadata.export_filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

function clearQALabels() {{
  if (!window.confirm('Clear QA labels for this ASR review?')) return;
  qaMetadata.turns.forEach(function(turn) {{
    localStorage.removeItem(turn.storage_key);
  }});
  restoreQALabels();
}}

restoreQALabels();
</script></body></html>
"""
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"ASR output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_asr_transcription(
    *,
    inventory: ASRInventory,
    output_root: Path,
    backend: ASRBackend,
    overwrite: bool = False,
) -> ASRSummary:
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    records: list[ASRTurnRecord] = []
    failure_counts: Counter[str] = Counter()
    backend_call_count = 0
    video_names: dict[str, str] = {}
    audio_names: dict[str, str] = {}
    turn_names: dict[tuple[str, str], str | None] = {}
    try:
        temporary.mkdir()
        review_root = temporary / "review_media"
        review_root.mkdir()
        for target in inventory.targets:
            video = Path(target.target_video_path)
            audio = Path(target.source_audio_path)
            video_name = _review_video_name(target)
            audio_name = _review_audio_name(target)
            (review_root / video_name).symlink_to(video)
            (review_root / audio_name).symlink_to(audio)
            video_names[target.target_clip_uid] = video_name
            audio_names[target.target_clip_uid] = audio_name

        for job in inventory.jobs:
            try:
                prepared = prepare_asr_audio(job)
                turn_name = _review_turn_name(job)
                _write_review_wav(review_root / turn_name, prepared.waveform)
                turn_names[(job.target_clip_uid, job.turn_id)] = turn_name
                backend_call_count += 1
                result = backend.transcribe(
                    audio=prepared.waveform,
                    sample_rate_hz=ASR_INPUT_SAMPLE_RATE_HZ,
                )
                record = _record(
                    job=job,
                    backend_provenance=backend.provenance,
                    preprocessing=prepared.preprocessing,
                    result=result,
                )
            except ASRTranscriptionFailure as exc:
                failure_counts.update([exc.code])
                turn_name = _review_turn_name(job)
                turn_names[(job.target_clip_uid, job.turn_id)] = (
                    turn_name if (review_root / turn_name).is_file() else None
                )
                record = _failed_record(
                    job=job,
                    backend_provenance=backend.provenance,
                    failure=exc,
                )
            records.append(record)

        counts = Counter(item.status for item in records)
        summary = ASRSummary(
            mode=inventory.mode,
            inventory_fingerprint=inventory.inventory_fingerprint,
            backend_provenance=backend.provenance,
            source_target_count=inventory.source_target_count,
            target_clip_count=inventory.selected_target_count,
            turn_count=len(records),
            transcribed_count=counts["transcribed"],
            uncertain_count=counts["uncertain"],
            failed_count=counts["failed"],
            backend_call_count=backend_call_count,
            failure_reason_counts=dict(sorted(failure_counts.items())),
            bounded_selection_applied=inventory.bounded_selection_applied,
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "turns.jsonl", records)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "review.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                video_names=video_names,
                audio_names=audio_names,
                turn_names=turn_names,
            ),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_review_source(
    *,
    inventory: ASRInventory,
    records: Sequence[ASRTurnRecord],
    summary: ASRSummary,
    expected_mode: Literal["pilot20", "production"],
) -> None:
    expected_fingerprint = _inventory_fingerprint(
        source_pairs_sha256=inventory.source_pairs_sha256,
        mode=inventory.mode,
        targets=inventory.targets,
        jobs=inventory.jobs,
    )
    if inventory.inventory_fingerprint != expected_fingerprint:
        raise ValueError("stored ASR inventory fingerprint is inconsistent")
    if inventory.mode != expected_mode or summary.mode != expected_mode:
        raise ValueError("stored ASR review mode does not match the request")
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("stored ASR summary fingerprint does not match inventory")
    if len(records) != len(inventory.jobs) or summary.turn_count != len(records):
        raise ValueError("stored ASR turn count is inconsistent")
    job_fields = tuple(ASRTurnJob.model_fields)
    for job, record in zip(inventory.jobs, records, strict=True):
        record_job = ASRTurnJob.model_validate(
            {field: getattr(record, field) for field in job_fields}
        )
        if record_job != job:
            raise ValueError("stored ASR turn record does not match inventory order")
        if record.backend_provenance != summary.backend_provenance:
            raise ValueError("stored ASR backend provenance is inconsistent")


def _ensure_review_symlink(*, destination: Path, source: Path) -> None:
    source_resolved = source.expanduser().resolve(strict=True)
    if not source_resolved.is_file():
        raise ValueError(f"ASR review source is not a file: {source_resolved}")
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True) == source_resolved:
                return
        except FileNotFoundError:
            pass
    elif destination.exists():
        raise ValueError(f"ASR review media path is not a symlink: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.symlink_to(source_resolved)
    temporary.replace(destination)


def regenerate_asr_review(
    *,
    output_root: Path,
    expected_mode: Literal["pilot20", "production"],
) -> dict[str, object]:
    root = output_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("ASR review output root must be a directory")
    inventory = ASRInventory.model_validate_json(
        (root / "inventory.json").read_text(encoding="utf-8")
    )
    records = [
        ASRTurnRecord.model_validate(item)
        for item in _read_jsonl(root / "turns.jsonl")
    ]
    summary = ASRSummary.model_validate_json(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    _validate_review_source(
        inventory=inventory,
        records=records,
        summary=summary,
        expected_mode=expected_mode,
    )

    review_root = root / "review_media"
    review_root.mkdir(exist_ok=True)
    review_root = review_root.resolve(strict=True)
    review_root.relative_to(root)
    video_names: dict[str, str] = {}
    audio_names: dict[str, str] = {}
    for target in inventory.targets:
        video_name = _review_video_name(target)
        audio_name = _review_audio_name(target)
        _ensure_review_symlink(
            destination=review_root / video_name,
            source=Path(target.target_video_path),
        )
        _ensure_review_symlink(
            destination=review_root / audio_name,
            source=Path(target.source_audio_path),
        )
        video_names[target.target_clip_uid] = video_name
        audio_names[target.target_clip_uid] = audio_name

    regenerated_turn_media_count = 0
    turn_names: dict[tuple[str, str], str | None] = {}
    for job in inventory.jobs:
        turn_name = _review_turn_name(job)
        turn_path = review_root / turn_name
        if not turn_path.is_file():
            if turn_path.exists() or turn_path.is_symlink():
                raise ValueError(f"ASR turn review media is invalid: {turn_path}")
            prepared = prepare_asr_audio(job)
            temporary = turn_path.with_name(
                f".{turn_path.name}.tmp-{uuid.uuid4().hex}"
            )
            _write_review_wav(temporary, prepared.waveform)
            temporary.replace(turn_path)
            regenerated_turn_media_count += 1
        turn_names[(job.target_clip_uid, job.turn_id)] = turn_name

    review_path = root / "review.html"
    temporary_review = review_path.with_name(
        f".{review_path.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        temporary_review.write_text(
            _review_html(
                inventory=inventory,
                records=records,
                video_names=video_names,
                audio_names=audio_names,
                turn_names=turn_names,
            ),
            encoding="utf-8",
        )
        temporary_review.replace(review_path)
    finally:
        if temporary_review.exists():
            temporary_review.unlink()
    return {
        "mode": inventory.mode,
        "review_regenerated": True,
        "model_loaded": False,
        "backend_calls": 0,
        "turn_count": len(records),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "regenerated_turn_media_count": regenerated_turn_media_count,
    }


def asr_output_root(
    audio_run_root: Path,
    *,
    mode: Literal["pilot20", "production"],
) -> Path:
    root = audio_run_root.expanduser().resolve(strict=False)
    if mode == "pilot20":
        return root / "asr_pilot20"
    return root / "production" / "asr"


def asr_stage_is_complete(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).is_file()
        for name in ("inventory.json", "turns.jsonl", "summary.json", "review.html")
    )
