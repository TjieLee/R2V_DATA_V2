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
from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationInventory,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.visual_production_source import VisualProductionInventory

QWEN3_ASR_MODEL_IDENTIFIER = "Qwen/Qwen3-ASR-1.7B"
QWEN3_ASR_SEGMENT_VERSION = "r2v.h3.qwen3_asr_segment.1"
QWEN3_ASR_INVENTORY_VERSION = "r2v.h3.qwen3_asr_inventory.1"
QWEN3_ASR_SUMMARY_VERSION = "r2v.h3.qwen3_asr_summary.1"


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
    schema_version: Literal["r2v.h3.qwen3_asr_inventory.1"] = (
        QWEN3_ASR_INVENTORY_VERSION
    )
    source_diarization_root: str
    source_visual_production_root: str
    segment_count: int = Field(ge=0)
    clip_count: int = Field(ge=0)
    model_identifier: Literal["Qwen/Qwen3-ASR-1.7B"] = QWEN3_ASR_MODEL_IDENTIFIER
    package: Literal["qwen-asr==0.0.6"] = "qwen-asr==0.0.6"
    configuration: Qwen3ASRConfiguration


class Qwen3ASRSummary(SchemaModel):
    schema_version: Literal["r2v.h3.qwen3_asr_summary.1"] = QWEN3_ASR_SUMMARY_VERSION
    segment_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    empty_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    clip_count: int = Field(ge=0)
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
        return self


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
    diarization: DiarizationInventory
    raw_segments: list[RawDiarizationSegment]
    bound_by_key: dict[tuple[str, str], BoundDiarizationSegment]


def _load_inputs(diarization_root: Path) -> _Inputs:
    root = diarization_root.expanduser().resolve(strict=True)
    inventory = DiarizationInventory.model_validate_json(
        (root / "inventory.json").read_text(encoding="utf-8")
    )
    raw = [
        RawDiarizationSegment.model_validate(row)
        for row in _read_rows(root / "raw_segments.jsonl")
    ]
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read_rows(root / "bound_segments.jsonl")
    ]
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    if len(bound_by_key) != len(bound):
        raise ValueError("duplicate bound DiariZen segment identity")
    if set(bound_by_key) != {(item.target_clip_uid, item.segment_id) for item in raw}:
        raise ValueError("raw and bound DiariZen segment inventories differ")
    return _Inputs(inventory, raw, bound_by_key)


def run_qwen3_asr(
    *,
    visual_inventory: VisualProductionInventory,
    diarization_root: Path,
    output_root: Path,
    backend: Qwen3ASRBackend,
    audio_loader: AudioLoader = load_official_diarizen_waveform,
    overwrite: bool = False,
) -> Qwen3ASRSummary:
    inputs = _load_inputs(diarization_root)
    identity_by_clip = {
        item.identity.clip_uid: item.identity for item in visual_inventory.clips
    }
    target_by_clip = {item.target_clip_uid: item for item in inputs.diarization.targets}
    if set(target_by_clip) - set(identity_by_clip):
        raise ValueError("DiariZen target is absent from canonical Visual Production")
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qwen3 ASR output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[Qwen3ASRSegment] = []
    loaded_audio: dict[str, tuple[np.ndarray, int]] = {}
    try:
        temporary.mkdir()
        for raw in inputs.raw_segments:
            identity = identity_by_clip[raw.target_clip_uid]
            bound = inputs.bound_by_key[(raw.target_clip_uid, raw.segment_id)]
            target = target_by_clip[raw.target_clip_uid]
            try:
                if raw.target_clip_uid not in loaded_audio:
                    loaded_audio[raw.target_clip_uid] = audio_loader(
                        Path(target.source_audio_path)
                    )
                waveform, sample_rate = loaded_audio[raw.target_clip_uid]
                if sample_rate != raw.source_sample_rate_hz:
                    raise ValueError("Qwen3 source sample rate differs from DiariZen")
                if raw.source_end_sample > waveform.shape[0]:
                    raise ValueError("Qwen3 segment exceeds DiariZen source waveform")
                crop = waveform[raw.source_start_sample : raw.source_end_sample]
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
                    **identity.model_dump(mode="python"),
                    segment_id=raw.segment_id,
                    speaker_cluster_id=raw.speaker_cluster_id,
                    entity_id=bound.entity_id,
                    entity_occurrence_id=bound.entity_occurrence_id,
                    source_audio_path=raw.source_audio_path,
                    source_start_sample=raw.source_start_sample,
                    source_end_sample=raw.source_end_sample,
                    source_sample_rate_hz=raw.source_sample_rate_hz,
                    start_time=raw.start_time,
                    end_time=raw.end_time,
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
        inventory = Qwen3ASRInventory(
            source_diarization_root=str(
                Path(diarization_root).expanduser().resolve(strict=True)
            ),
            source_visual_production_root=visual_inventory.visual_production_root,
            segment_count=len(rows),
            clip_count=len({item.clip_uid for item in rows}),
            configuration=backend.configuration,
        )
        summary = Qwen3ASRSummary(
            segment_count=len(rows),
            transcribed_count=status_counts["transcribed"],
            empty_count=status_counts["empty"],
            failed_count=status_counts["failed"],
            clip_count=len({item.clip_uid for item in rows}),
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
