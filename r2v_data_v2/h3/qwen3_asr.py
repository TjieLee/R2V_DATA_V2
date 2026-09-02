from __future__ import annotations

import html
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from r2v_data_v2.h3.schemas import SchemaModel

QWEN3_ASR_MODEL_IDENTIFIER = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_SEGMENT_VERSION = "r2v.h3.qwen3_asr_segment.1"
QWEN3_ASR_INVENTORY_VERSION = "r2v.h3.qwen3_asr_inventory.2"
QWEN3_ASR_SUMMARY_VERSION = "r2v.h3.qwen3_asr_summary.2"


class Qwen3ASRConfiguration(SchemaModel):
    model_identifier: Literal["Qwen/Qwen3-ASR-1.7B"] = QWEN3_ASR_MODEL_IDENTIFIER
    package: Literal["qwen-asr==0.0.6"] = "qwen-asr==0.0.6"
    local_model_path: str
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_inference_batch_size: int = Field(default=1, gt=0, le=16)
    max_new_tokens: Literal[256] = 256
    local_files_only: Literal[True] = True
    context: Literal[""] = ""
    language: None = None
    return_time_stamps: Literal[False] = False

    @model_validator(mode="after")
    def validate_local_configuration(self) -> Qwen3ASRConfiguration:
        if not self.local_model_path.strip() or not self.device.strip():
            raise ValueError("Qwen3 ASR model path and device are required")
        return self

    @classmethod
    def from_environment(cls) -> Qwen3ASRConfiguration:
        model_path = os.environ.get("QWEN3_ASR_MODEL_PATH", "")
        return cls(
            local_model_path=model_path,
            device=os.environ.get("QWEN3_ASR_DEVICE", "cuda:0"),
            dtype=os.environ.get("QWEN3_ASR_DTYPE", "bfloat16"),
            max_inference_batch_size=int(
                os.environ.get("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", "1")
            ),
        )


class Qwen3ASRSegment(SchemaModel):
    schema_version: Literal["r2v.h3.qwen3_asr_segment.1"] = QWEN3_ASR_SEGMENT_VERSION
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    source_audio_path: str
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    status: Literal["transcribed", "empty", "failed"]
    text: str | None = None
    language: str | None = None
    model_identifier: Literal["Qwen/Qwen3-ASR-1.7B"] = QWEN3_ASR_MODEL_IDENTIFIER
    package: Literal["qwen-asr==0.0.6"] = "qwen-asr==0.0.6"
    configuration: Qwen3ASRConfiguration
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Qwen3ASRSegment:
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("Qwen3 ASR segment sample range must be positive")
        if self.status == "transcribed":
            if (
                self.text is None
                or not self.text.strip()
                or self.failure_reason is not None
            ):
                raise ValueError(
                    "transcribed Qwen3 segment requires non-empty raw text"
                )
        elif self.status == "empty":
            if (
                self.text is not None
                or self.language is not None
                or self.failure_reason
            ):
                raise ValueError("empty Qwen3 segment cannot publish text or failure")
        elif (
            not self.failure_reason
            or self.text is not None
            or self.language is not None
        ):
            raise ValueError("failed Qwen3 segment requires only a failure reason")
        return self


class Qwen3ASRInventory(SchemaModel):
    schema_version: Literal[
        "r2v.h3.qwen3_asr_inventory.1",
        "r2v.h3.qwen3_asr_inventory.2",
    ] = (
        QWEN3_ASR_INVENTORY_VERSION
    )
    source_diarization_root: str
    source_visual_production_root: str
    segment_count: int = Field(ge=0)
    clip_count: int = Field(ge=0)
    source_target_clip_count: int = Field(default=0, ge=0)
    clips_with_diarization_segments: int = Field(default=0, ge=0)
    clips_without_diarization_segments: int = Field(default=0, ge=0)
    model_identifier: Literal["Qwen/Qwen3-ASR-1.7B"] = QWEN3_ASR_MODEL_IDENTIFIER
    package: Literal["qwen-asr==0.0.6"] = "qwen-asr==0.0.6"
    configuration: Qwen3ASRConfiguration

    @model_validator(mode="after")
    def validate_target_coverage(self) -> Qwen3ASRInventory:
        if self.schema_version == QWEN3_ASR_INVENTORY_VERSION:
            if self.source_target_clip_count != (
                self.clips_with_diarization_segments
                + self.clips_without_diarization_segments
            ):
                raise ValueError("Qwen3 ASR inventory target coverage must reconcile")
            if self.clip_count != self.clips_with_diarization_segments:
                raise ValueError("Qwen3 ASR inventory clip count differs from coverage")
        return self


class Qwen3ASRSummary(SchemaModel):
    schema_version: Literal[
        "r2v.h3.qwen3_asr_summary.1",
        "r2v.h3.qwen3_asr_summary.2",
    ] = QWEN3_ASR_SUMMARY_VERSION
    segment_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    empty_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    clip_count: int = Field(ge=0)
    source_target_clip_count: int = Field(default=0, ge=0)
    clips_with_diarization_segments: int = Field(default=0, ge=0)
    clips_without_diarization_segments: int = Field(default=0, ge=0)
    model_identifier: Literal["Qwen/Qwen3-ASR-1.7B"] = QWEN3_ASR_MODEL_IDENTIFIER
    package: Literal["qwen-asr==0.0.6"] = "qwen-asr==0.0.6"
    language_counts: dict[str, int]
    confidence_fields_published: Literal[False] = False
    language_probability_gate_applied: Literal[False] = False
    translation_performed: Literal[False] = False
    correction_performed: Literal[False] = False
    timestamp_alignment_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> Qwen3ASRSummary:
        if self.segment_count != (
            self.transcribed_count + self.empty_count + self.failed_count
        ):
            raise ValueError("Qwen3 ASR summary counts must reconcile")
        if self.schema_version == QWEN3_ASR_SUMMARY_VERSION:
            if self.source_target_clip_count != (
                self.clips_with_diarization_segments
                + self.clips_without_diarization_segments
            ):
                raise ValueError("Qwen3 ASR target coverage counts must reconcile")
            if self.clip_count != self.clips_with_diarization_segments:
                raise ValueError("Qwen3 ASR segment clip count differs from coverage")
        return self


class _ReadableDiarizationTarget(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_target.1"] = (
        "r2v.h3.jea_diarization_target.1"
    )
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    source_audio_path: str
    source_sample_rate_hz: int = Field(gt=0)
    target_video_path: str
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class _ReadableDiarizationSegment(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_segment.1"] = (
        "r2v.h3.jea_diarization_segment.1"
    )
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    source_audio_path: str
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    raw_schema_version: Literal["r2v.h3.diarization_segment.2"] = (
        "r2v.h3.diarization_segment.2"
    )
    bound_schema_version: Literal["r2v.h3.diarization_bound_segment.1"] = (
        "r2v.h3.diarization_bound_segment.1"
    )
    mapping_policy_version: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        "h3_diarizen_sparse_anchor_policy_v1"
    )
    segmentation_changed: Literal[False] = False
    numeric_mapping_thresholds_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_segment(self) -> _ReadableDiarizationSegment:
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("readable DiariZen sample range must be positive")
        if self.end_time <= self.start_time:
            raise ValueError("readable DiariZen time range must be positive")
        expected_occurrence = (
            f"{self.clip_uid}/{self.entity_id}"
            if self.entity_id is not None
            else None
        )
        if self.entity_occurrence_id != expected_occurrence:
            raise ValueError("readable DiariZen entity occurrence is inconsistent")
        for value in (
            self.clip_uid,
            self.clip_display_path,
            self.media_collection_relpath,
            self.media_collection_name,
            self.episode_name,
            self.clip_name,
            self.shard_id,
            self.segment_id,
            self.speaker_cluster_id,
            self.source_audio_path,
        ):
            if not value.strip():
                raise ValueError("readable DiariZen metadata must not be empty")
        return self


class _ReadableDiarizationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_summary.1"] = (
        "r2v.h3.jea_diarization_summary.1"
    )
    target_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    media_collection_count: int = Field(ge=0)
    segmentation_changed: Literal[False] = False
    numeric_mapping_thresholds_changed: Literal[False] = False


class _ProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _RawSegmentProvenance(_ProvenanceModel):
    schema_version: Literal["r2v.h3.diarization_segment.2"] = (
        "r2v.h3.diarization_segment.2"
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_audio_path: str
    source_sample_rate_hz: int = Field(gt=0)


class _BoundSegmentProvenance(_ProvenanceModel):
    schema_version: Literal["r2v.h3.diarization_bound_segment.1"] = (
        "r2v.h3.diarization_bound_segment.1"
    )
    target_clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    entity_id: str | None = None
    entity_occurrence_id: str | None = None


class _QwenResult(Protocol):
    text: str
    language: str | None


class _QwenModel(Protocol):
    def transcribe(self, **kwargs: object) -> Sequence[_QwenResult]: ...


class Qwen3ASRBackend:
    def __init__(
        self,
        configuration: Qwen3ASRConfiguration,
        *,
        model_factory: Callable[..., _QwenModel] | None = None,
    ) -> None:
        self.configuration = configuration
        if model_factory is None:
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except (ImportError, OSError) as exc:
                raise RuntimeError(
                    "A usable PyTorch runtime and qwen-asr==0.0.6 are required "
                    "in the isolated QWEN3_ASR_ENV"
                ) from exc
            if (
                configuration.device.partition(":")[0].lower() == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "Qwen3 ASR requested a CUDA device, but "
                    "torch.cuda.is_available() is false"
                )
            dtype = getattr(torch, configuration.dtype)
            model_factory = Qwen3ASRModel.from_pretrained
        else:
            dtype = configuration.dtype
        self._model = model_factory(
            configuration.local_model_path,
            dtype=dtype,
            device_map=configuration.device,
            max_inference_batch_size=configuration.max_inference_batch_size,
            max_new_tokens=configuration.max_new_tokens,
            local_files_only=True,
        )

    def transcribe(
        self,
        *,
        waveform: np.ndarray,
        sample_rate_hz: int,
    ) -> tuple[str, str | None]:
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("Qwen3 ASR input must be finite, non-empty mono audio")
        result = self._model.transcribe(
            audio=(waveform, sample_rate_hz),
            context="",
            language=None,
            return_time_stamps=False,
        )[0]
        return str(result.text), (
            None if result.language is None else str(result.language)
        )


AudioLoader = Callable[[Path], tuple[np.ndarray, int]]


def load_official_diarizen_waveform(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required in the DiariZen/Qwen3 environment"
        ) from exc
    waveform, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise ValueError("DiariZen source audio has no samples or channels")
    return np.ascontiguousarray(waveform[:, 0]), int(sample_rate)


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    return rows


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


def _review_html(rows: Sequence[Qwen3ASRSegment]) -> str:
    body = []
    for row in sorted(
        rows, key=lambda item: (item.clip_display_path, item.source_start_sample)
    ):
        if row.status != "transcribed":
            continue
        body.append(
            "<tr><td>"
            + html.escape(row.clip_display_path)
            + "</td><td>"
            + html.escape(row.segment_id)
            + "</td><td>"
            + html.escape(row.speaker_cluster_id)
            + "</td><td>"
            + html.escape(row.language or "")
            + "</td><td>"
            + html.escape(row.text or "")
            + "</td></tr>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>Qwen3 ASR review</title>"
        "<table><thead><tr><th>clip</th><th>segment</th><th>speaker</th>"
        "<th>language</th><th>raw text</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


@dataclass(frozen=True)
class _Inputs:
    readable_segments: list[_ReadableDiarizationSegment]
    target_clip_ids: list[str]


def _index_by_key(
    rows: Sequence[
        _ReadableDiarizationSegment
        | _RawSegmentProvenance
        | _BoundSegmentProvenance
    ],
    *,
    source_name: str,
) -> dict[tuple[str, str], object]:
    result: dict[tuple[str, str], object] = {}
    for row in rows:
        clip_uid = (
            row.clip_uid
            if isinstance(row, _ReadableDiarizationSegment)
            else row.target_clip_uid
        )
        key = (clip_uid, row.segment_id)
        if key in result:
            raise ValueError(f"duplicate {source_name} DiariZen segment identity")
        result[key] = row
    return result


def _load_inputs(diarization_root: Path) -> _Inputs:
    root = diarization_root.expanduser().resolve(strict=True)
    readable_summary = _ReadableDiarizationSummary.model_validate_json(
        (root / "readable_summary.json").read_text(encoding="utf-8")
    )
    readable = [
        _ReadableDiarizationSegment.model_validate(row)
        for row in _read_rows(root / "readable_segments.jsonl")
    ]
    targets_path = root / "readable_targets.jsonl"
    targets = (
        [
            _ReadableDiarizationTarget.model_validate(row)
            for row in _read_rows(targets_path)
        ]
        if targets_path.is_file()
        else []
    )
    raw = [
        _RawSegmentProvenance.model_validate(row)
        for row in _read_rows(root / "raw_segments.jsonl")
    ]
    bound = [
        _BoundSegmentProvenance.model_validate(row)
        for row in _read_rows(root / "bound_segments.jsonl")
    ]
    if readable_summary.segment_count != len(readable):
        raise ValueError("readable DiariZen segment count differs from summary")
    target_ids = [item.clip_uid for item in targets]
    if targets:
        if (
            readable_summary.target_count != len(targets)
            or len(target_ids) != len(set(target_ids))
        ):
            raise ValueError("readable DiariZen target inventory is inconsistent")
    else:
        target_ids = list(dict.fromkeys(item.clip_uid for item in readable))
    if not {item.clip_uid for item in readable}.issubset(target_ids):
        raise ValueError("readable DiariZen segment references a non-target clip")
    readable_by_key = _index_by_key(readable, source_name="readable")
    raw_by_key = _index_by_key(raw, source_name="raw")
    bound_by_key = _index_by_key(bound, source_name="bound")
    if set(readable_by_key) != set(raw_by_key) or set(readable_by_key) != set(
        bound_by_key
    ):
        raise ValueError("readable, raw, and bound DiariZen inventories differ")
    for key, readable_item in readable_by_key.items():
        if not isinstance(readable_item, _ReadableDiarizationSegment):
            raise TypeError("invalid readable DiariZen provenance row")
        raw_item = raw_by_key[key]
        bound_item = bound_by_key[key]
        if not isinstance(raw_item, _RawSegmentProvenance) or not isinstance(
            bound_item, _BoundSegmentProvenance
        ):
            raise TypeError("invalid raw or bound DiariZen provenance row")
        if (
            readable_item.source_start_sample != raw_item.source_start_sample
            or readable_item.source_end_sample != raw_item.source_end_sample
            or readable_item.source_start_sample != bound_item.source_start_sample
            or readable_item.source_end_sample != bound_item.source_end_sample
            or readable_item.speaker_cluster_id != raw_item.speaker_cluster_id
            or readable_item.speaker_cluster_id != bound_item.speaker_cluster_id
            or readable_item.entity_id != bound_item.entity_id
            or readable_item.entity_occurrence_id != bound_item.entity_occurrence_id
            or readable_item.source_audio_path != raw_item.source_audio_path
            or readable_item.source_sample_rate_hz != raw_item.source_sample_rate_hz
            or readable_item.start_time != raw_item.start_time
            or readable_item.end_time != raw_item.end_time
            or readable_item.start_time != bound_item.start_time
            or readable_item.end_time != bound_item.end_time
        ):
            raise ValueError("readable DiariZen provenance differs from raw or bound")
        if not Path(readable_item.source_audio_path).is_file():
            raise FileNotFoundError(
                f"readable DiariZen source audio is missing: "
                f"{readable_item.source_audio_path}"
            )
    return _Inputs(readable, target_ids)


def run_qwen3_asr(
    *,
    diarization_root: Path,
    source_visual_production_root: str,
    output_root: Path,
    backend: Qwen3ASRBackend,
    audio_loader: AudioLoader = load_official_diarizen_waveform,
    overwrite: bool = False,
) -> Qwen3ASRSummary:
    inputs = _load_inputs(diarization_root)
    if not source_visual_production_root.strip():
        raise ValueError("Qwen3 ASR Visual production provenance is empty")
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qwen3 ASR output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Qwen3ASRSegment] = []
    loaded_audio: dict[str, tuple[np.ndarray, int]] = {}
    try:
        temporary.mkdir()
        for readable in inputs.readable_segments:
            try:
                if readable.clip_uid not in loaded_audio:
                    loaded_audio[readable.clip_uid] = audio_loader(
                        Path(readable.source_audio_path)
                    )
                waveform, sample_rate = loaded_audio[readable.clip_uid]
                if sample_rate != readable.source_sample_rate_hz:
                    raise ValueError("Qwen3 source sample rate differs from DiariZen")
                if readable.source_end_sample > waveform.shape[0]:
                    raise ValueError("Qwen3 segment exceeds DiariZen source waveform")
                crop = waveform[
                    readable.source_start_sample : readable.source_end_sample
                ]
                text, language = backend.transcribe(
                    waveform=crop,
                    sample_rate_hz=sample_rate,
                )
                if text.strip():
                    status = "transcribed"
                    published_text: str | None = text
                    published_language = language
                else:
                    status = "empty"
                    published_text = None
                    published_language = None
                failure_reason = None
            except Exception as exc:  # noqa: BLE001 - isolate individual segments
                status = "failed"
                published_text = None
                published_language = None
                failure_reason = f"{type(exc).__name__}:{exc}"
            rows.append(
                Qwen3ASRSegment(
                    clip_uid=readable.clip_uid,
                    clip_display_path=readable.clip_display_path,
                    media_collection_relpath=readable.media_collection_relpath,
                    media_collection_name=readable.media_collection_name,
                    episode_name=readable.episode_name,
                    clip_name=readable.clip_name,
                    shard_id=readable.shard_id,
                    segment_id=readable.segment_id,
                    speaker_cluster_id=readable.speaker_cluster_id,
                    entity_id=readable.entity_id,
                    entity_occurrence_id=readable.entity_occurrence_id,
                    source_audio_path=readable.source_audio_path,
                    source_start_sample=readable.source_start_sample,
                    source_end_sample=readable.source_end_sample,
                    source_sample_rate_hz=readable.source_sample_rate_hz,
                    start_time=readable.start_time,
                    end_time=readable.end_time,
                    status=status,
                    text=published_text,
                    language=published_language,
                    configuration=backend.configuration,
                    failure_reason=failure_reason,
                )
            )
        rows.sort(
            key=lambda item: (
                item.clip_display_path,
                item.source_start_sample,
                item.segment_id,
            )
        )
        status_counts = Counter(item.status for item in rows)
        if rows and status_counts["failed"] == len(rows):
            raise RuntimeError("Qwen3 ASR failed for every diarization segment")
        language_counts = Counter(
            item.language
            for item in rows
            if item.status == "transcribed" and item.language
        )
        clips_with_segments = len({item.clip_uid for item in inputs.readable_segments})
        source_target_count = len(inputs.target_clip_ids)
        clips_without_segments = source_target_count - clips_with_segments
        inventory = Qwen3ASRInventory(
            source_diarization_root=str(
                Path(diarization_root).expanduser().resolve(strict=True)
            ),
            source_visual_production_root=source_visual_production_root,
            segment_count=len(rows),
            clip_count=clips_with_segments,
            source_target_clip_count=source_target_count,
            clips_with_diarization_segments=clips_with_segments,
            clips_without_diarization_segments=clips_without_segments,
            configuration=backend.configuration,
        )
        summary = Qwen3ASRSummary(
            segment_count=len(rows),
            transcribed_count=status_counts["transcribed"],
            empty_count=status_counts["empty"],
            failed_count=status_counts["failed"],
            clip_count=clips_with_segments,
            source_target_clip_count=source_target_count,
            clips_with_diarization_segments=clips_with_segments,
            clips_without_diarization_segments=clips_without_segments,
            language_counts=dict(sorted(language_counts.items())),
        )
        _write_json(temporary / "inventory.json", inventory)
        _write_jsonl(temporary / "segments.jsonl", rows)
        _write_json(temporary / "summary.json", summary)
        (temporary / "review.html").write_text(_review_html(rows), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
