from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import AfterValidator, Field, StrictStr, model_validator

from r2v_data_v2.h3.audio_backends import (
    AudioFileProbe,
    AudioMediaBackend,
    FFmpegAudioMediaBackend,
)
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationBackend,
    DiarizationClipResult,
    DiarizationInventory,
    DiarizationTargetClip,
    RawDiarizationSegment,
    run_diarization_binding_pilot,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.jea_diarization import (
    JEAReadableDiarizationSegment,
    JEAReadableDiarizationSummary,
    JEAReadableDiarizationTarget,
)
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
)
from r2v_data_v2.h3.qwen3_asr import (
    Qwen3ASRBackend,
    Qwen3ASRSegment,
    Qwen3ASRSummary,
    SegmentAudioLoader,
    run_qwen3_asr,
)
from r2v_data_v2.h3.schemas import SchemaModel

SAM_AUDIO_SHADOW_ROOT_NAME = "sam_audio_stem_shadow_v1"
SAM_AUDIO_SEPARATION_STAGE_NAME = "separation"
SAM_AUDIO_STEM_INVENTORY_VERSION = "r2v.h3.sam_audio_stem_inventory.3"
SAM_AUDIO_STEM_RECORD_VERSION = "r2v.h3.sam_audio_stem_record.3"
SAM_AUDIO_STEM_SUMMARY_VERSION = "r2v.h3.sam_audio_stem_summary.2"
STEM_DIARIZATION_SHADOW_VERSION = "r2v.h3.stem_diarization_shadow.4"
STEM_ASR_SHADOW_VERSION = "r2v.h3.stem_asr_shadow.4"
STEM_REFERENCE_VERSION = "r2v.h3.sam_audio_stem_reference.2"
STEM_CLIP_SKIP_VERSION = "r2v.h3.sam_audio_stem_clip_skip.1"
STEM_DIARIZATION_FAILURE_VERSION = "r2v.h3.stem_diarization_clip_failure.1"
STEM_CANONICALIZATION_VERSION = "sam_audio_raw_to_32k_stereo_pcm16_v1"
STEM_REFERENCE_EXTRACTION_VERSION = "sam_audio_stem_exact_sample_crop_v1"
STEM_ALIGNMENT_TOLERANCE_SECONDS = 0.10
STEM_ALIGNMENT_WARNING_SECONDS = 0.05
STEM_TIMELINE_POLICY = "zero_offset_bounded_duration_drift_v1"
STEM_CANONICAL_CHANNEL_POLICY = "dual_mono_from_sam_mono_v1"
STEM_SAMPLE_RATE_HZ = 32000
STEM_CHANNELS = 2

StemType = Literal["music", "speech", "sfx"]
SAMRoute = Literal["music_first", "voice_first"]
VerificationState = Literal["success", "uncertain", "failure", "unverified"]


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    source = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: SchemaModel | dict[str, object]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[SchemaModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_compact_json(row.model_dump(mode="json")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    moved_existing = False
    try:
        if destination.exists():
            destination.replace(backup)
            moved_existing = True
        temporary.replace(destination)
        if moved_existing:
            shutil.rmtree(backup)
    except Exception:
        if moved_existing and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise


def stem_shadow_root(
    audio_production_root: Path, shadow_run_id: str | None = None,
) -> Path:
    root = (
        audio_production_root.expanduser().resolve(strict=False)
        / SAM_AUDIO_SHADOW_ROOT_NAME
    )
    if shadow_run_id is None:
        return root
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", shadow_run_id) is None:
        raise ValueError("invalid SAM Audio shadow run ID")
    selected = root / "runs" / shadow_run_id
    if selected.resolve(strict=False) != root.resolve(strict=False) / "runs" / shadow_run_id:
        raise ValueError("SAM Audio shadow run must not redirect to another root")
    return selected


def stem_separation_root(
    audio_production_root: Path, shadow_run_id: str | None = None,
) -> Path:
    root = stem_shadow_root(audio_production_root, shadow_run_id)
    return require_shadow_output_path(
        shadow_root=root, output_path=root / SAM_AUDIO_SEPARATION_STAGE_NAME,
    )


def require_shadow_output_path(*, shadow_root: Path, output_path: Path) -> Path:
    root = shadow_root.expanduser().resolve(strict=False)
    output = output_path.expanduser().resolve(strict=False)
    if output != root and root not in output.parents:
        raise ValueError("SAM Audio shadow output must remain under its shadow root")
    return output


def _validate_sam_prompt(prompt: str) -> str:
    if not prompt or prompt != prompt.strip() or prompt != prompt.lower():
        raise ValueError("SAM Audio prompt must be non-empty, lowercase, and trimmed")
    return prompt


SAMAudioPrompt = Annotated[StrictStr, Field(min_length=1), AfterValidator(_validate_sam_prompt)]


class SAMAudioModelConfiguration(SchemaModel):
    speech_prompt: SAMAudioPrompt
    music_prompt: SAMAudioPrompt
    implementation_root: str
    implementation_files: dict[str, str]
    model_path: str
    model_name: str
    model_config_path: str
    model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_checkpoint_path: str
    model_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    t5_base_path: str
    t5_dependency_files: dict[str, str]
    device: str
    reranking_candidates: Literal[1] = 1
    predict_spans: Literal[False] = False
    local_files_only: Literal[True] = True
    visual_ranker_disabled: Literal[True] = True
    text_ranker_disabled: Literal[True] = True
    span_predictor_disabled: Literal[True] = True
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_configuration(self) -> SAMAudioModelConfiguration:
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("SAM Audio configuration fingerprint is invalid")
        return self


class _LegacySAMAudioModelConfiguration(SAMAudioModelConfiguration):
    """Read-only v2 configuration: inferred prompts never enter its legacy hash."""

    speech_prompt: Literal["human voices"] = "human voices"
    music_prompt: Literal["music soundtrack"] = "music soundtrack"

    @model_validator(mode="after")
    def validate_configuration(self) -> _LegacySAMAudioModelConfiguration:
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint", "speech_prompt", "music_prompt"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("legacy SAM Audio configuration fingerprint is invalid")
        return self


def _t5_dependency_files(root: Path) -> dict[str, str]:
    required = [root / "config.json"]
    tokenizer = next(
        (path for path in (root / "tokenizer.json", root / "spiece.model") if path.is_file()),
        None,
    )
    if tokenizer is None:
        raise ValueError("local T5-base path lacks tokenizer.json or spiece.model")
    required.append(tokenizer)
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.is_file():
        required.append(tokenizer_config)

    direct_weight = next(
        (
            path
            for path in (root / "model.safetensors", root / "pytorch_model.bin")
            if path.is_file()
        ),
        None,
    )
    if direct_weight is not None:
        required.append(direct_weight)
    else:
        index = next(
            (
                path
                for path in (
                    root / "model.safetensors.index.json",
                    root / "pytorch_model.bin.index.json",
                )
                if path.is_file()
            ),
            None,
        )
        if index is None:
            raise ValueError("local T5-base path lacks model checkpoint files")
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("local T5-base checkpoint index is invalid")
        required.append(index)
        for relative in sorted(set(weight_map.values())):
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("local T5-base checkpoint index contains an invalid path")
            shard = (root / relative).resolve(strict=True)
            if root not in shard.parents or not shard.is_file():
                raise ValueError("local T5-base checkpoint shard is unavailable")
            required.append(shard)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(set(required))
    }


def _implementation_files(root: Path) -> dict[str, str]:
    required = [
        Path("sam_audio/__init__.py"),
        Path("sam_audio/processor.py"),
        Path("sam_audio/model/base.py"),
        Path("sam_audio/model/config.py"),
        Path("sam_audio/model/model.py"),
    ]
    output: dict[str, str] = {}
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"local SAM Audio implementation lacks {relative}")
        output[relative.as_posix()] = sha256_file(path)
    return output


def sam_audio_configuration(
    *,
    implementation_root: Path,
    model_path: Path,
    model_name: str | None,
    t5_base_path: Path,
    device: str,
    reranking_candidates: int,
    speech_prompt: str = "human voices",
    music_prompt: str = "music soundtrack",
) -> SAMAudioModelConfiguration:
    _validate_sam_prompt(speech_prompt)
    _validate_sam_prompt(music_prompt)
    code = implementation_root.expanduser().resolve(strict=True)
    checkpoint = model_path.expanduser().resolve(strict=True)
    t5 = t5_base_path.expanduser().resolve(strict=True)
    if not code.is_dir() or not checkpoint.is_dir() or not t5.is_dir():
        raise ValueError("SAM Audio implementation/model paths must exist locally")
    config_path = checkpoint / "config.json"
    checkpoint_path = checkpoint / "checkpoint.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("local SAM Audio config.json/checkpoint.pt is unavailable")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict) or not isinstance(
        raw_config.get("text_encoder"), dict
    ):
        raise TypeError("local SAM Audio text-encoder configuration is invalid")
    configured_name = raw_config.get("_name_or_path")
    configured_identifier = (
        configured_name.strip()
        if isinstance(configured_name, str) and configured_name.strip()
        else None
    )
    derived_name = configured_identifier or checkpoint.name
    if model_name is not None:
        supplied_name = model_name.strip()
        if not supplied_name or (
            configured_identifier is not None
            and supplied_name != configured_identifier
        ) or (
            configured_identifier is None
            and Path(supplied_name).name != checkpoint.name
        ):
            raise ValueError("SAM Audio model identifier differs from local model path")
        derived_name = supplied_name
    if not device.strip() or reranking_candidates != 1:
        raise ValueError("SAM Audio runtime configuration is incomplete")
    values = {
        "implementation_root": str(code),
        "speech_prompt": speech_prompt,
        "music_prompt": music_prompt,
        "implementation_files": _implementation_files(code),
        "model_path": str(checkpoint),
        "model_name": derived_name,
        "model_config_path": str(config_path),
        "model_config_sha256": sha256_file(config_path),
        "model_checkpoint_path": str(checkpoint_path),
        "model_checkpoint_sha256": sha256_file(checkpoint_path),
        "t5_base_path": str(t5),
        "t5_dependency_files": _t5_dependency_files(t5),
        "device": device,
        "reranking_candidates": reranking_candidates,
        "predict_spans": False,
        "local_files_only": True,
        "visual_ranker_disabled": True,
        "text_ranker_disabled": True,
        "span_predictor_disabled": True,
    }
    return SAMAudioModelConfiguration(
        **values,
        configuration_fingerprint=_sha256_text(_compact_json(values)),
    )


class SAMAudioSeparationResult(SchemaModel):
    target_path: str
    residual_path: str
    verification_state: VerificationState = "unverified"
    candidate_count: int | None = Field(default=None, ge=1)
    selected_candidate_index: int | None = Field(default=None, ge=0)
    judge_scores: dict[str, float] | None = None
    clap_score: float | None = Field(default=None, allow_inf_nan=False)
    runtime_seconds: float = Field(ge=0, allow_inf_nan=False)
    backend_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_paths(self) -> SAMAudioSeparationResult:
        if not self.target_path.strip() or not self.residual_path.strip():
            raise ValueError("SAM Audio result paths must not be blank")
        if self.selected_candidate_index is not None and self.candidate_count is None:
            raise ValueError("selected SAM candidate requires candidate metadata")
        return self


class SAMAudioBackend(Protocol):
    configuration: SAMAudioModelConfiguration

    def separate(
        self,
        *,
        clip_uid: str,
        source_audio_path: Path,
        prompt: str,
        target_path: Path,
        residual_path: Path,
    ) -> SAMAudioSeparationResult: ...


class OfficialSAMAudioBackend:
    """Lazy, local-only adapter for the official SAM Audio Python API."""

    def __init__(self, configuration: SAMAudioModelConfiguration) -> None:
        self.configuration = configuration
        self._model: object | None = None
        self._processor: object | None = None
        self._torch: object | None = None
        self._torchaudio: object | None = None

    @staticmethod
    def _normalized_waveform(waveform: object, *, torch: object, label: str) -> object:
        tensor = waveform.float()  # type: ignore[attr-defined]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ValueError(f"SAM Audio {label} waveform must have shape [1,T]")
        if tensor.shape[1] == 0:
            raise ValueError(f"SAM Audio {label} waveform must not be empty")
        if not bool(torch.isfinite(tensor).all().item()):  # type: ignore[attr-defined]
            raise ValueError(f"SAM Audio {label} waveform must contain finite samples")
        return tensor.cpu()

    def _load(self) -> tuple[object, object, object, object]:
        if self._model is not None:
            assert self._processor is not None
            assert self._torch is not None
            assert self._torchaudio is not None
            return self._model, self._processor, self._torch, self._torchaudio
        if _implementation_files(
            Path(self.configuration.implementation_root)
        ) != self.configuration.implementation_files:
            raise ValueError("local SAM Audio implementation fingerprint changed")
        if (
            sha256_file(Path(self.configuration.model_config_path))
            != self.configuration.model_config_sha256
            or sha256_file(Path(self.configuration.model_checkpoint_path))
            != self.configuration.model_checkpoint_sha256
            or _t5_dependency_files(Path(self.configuration.t5_base_path))
            != self.configuration.t5_dependency_files
        ):
            raise ValueError("local SAM Audio model dependency fingerprint changed")
        code_root = self.configuration.implementation_root
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
            import torchaudio
            from sam_audio import SAMAudio, SAMAudioProcessor
        except ImportError as exc:
            raise RuntimeError(
                "official SAM Audio runtime dependencies are unavailable"
            ) from exc
        raw_config = json.loads(
            Path(self.configuration.model_config_path).read_text(encoding="utf-8")
        )
        text_encoder = dict(raw_config["text_encoder"])
        text_encoder["name"] = self.configuration.t5_base_path
        model = SAMAudio.from_pretrained(
            self.configuration.model_path,
            local_files_only=True,
            text_encoder=text_encoder,
            visual_ranker=None,
            text_ranker=None,
            span_predictor=None,
        )
        processor = SAMAudioProcessor.from_pretrained(self.configuration.model_path)
        model = model.eval().to(self.configuration.device)
        self._model, self._processor = model, processor
        self._torch, self._torchaudio = torch, torchaudio
        return model, processor, torch, torchaudio

    def separate(
        self,
        *,
        clip_uid: str,
        source_audio_path: Path,
        prompt: str,
        target_path: Path,
        residual_path: Path,
    ) -> SAMAudioSeparationResult:
        del clip_uid
        source = source_audio_path.expanduser().resolve(strict=True)
        if not source.is_file() or not prompt.strip() or prompt != prompt.lower():
            raise ValueError("SAM Audio needs a readable source and lowercase text prompt")
        model, processor, torch, torchaudio = self._load()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        batch = processor(audios=[str(source)], descriptions=[prompt]).to(
            self.configuration.device
        )
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=False,
                reranking_candidates=self.configuration.reranking_candidates,
            )
        sample_rate = int(processor.audio_sampling_rate)
        if len(result.target) != 1 or len(result.residual) != 1:
            raise ValueError("SAM Audio batch-one result must contain one target/residual")
        target = self._normalized_waveform(
            result.target[0],
            torch=torch,
            label="target",
        )
        residual = self._normalized_waveform(
            result.residual[0],
            torch=torch,
            label="residual",
        )
        torchaudio.save(str(target_path), target, sample_rate)
        torchaudio.save(str(residual_path), residual, sample_rate)
        return SAMAudioSeparationResult(
            target_path=str(target_path),
            residual_path=str(residual_path),
            verification_state="unverified",
            runtime_seconds=time.monotonic() - started,
            backend_metadata={
                "official_api": "SAMAudio.separate",
                "predict_spans": False,
                "reranking_candidates": self.configuration.reranking_candidates,
                "quality_scores_exposed": False,
            },
        )


class SAMAudioCallProvenance(SchemaModel):
    pass_index: Literal[1, 2]
    prompt: SAMAudioPrompt
    input_path: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_path: str
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    residual_path: str
    residual_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_state: VerificationState
    candidate_count: int | None = Field(default=None, ge=1)
    selected_candidate_index: int | None = Field(default=None, ge=0)
    judge_scores: dict[str, float] | None = None
    clap_score: float | None = Field(default=None, allow_inf_nan=False)
    runtime_seconds: float = Field(ge=0, allow_inf_nan=False)
    backend_metadata: dict[str, object]


class StemArtifact(SchemaModel):
    stem_type: StemType
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start_sample: Literal[0] = 0
    source_end_sample: int = Field(gt=0)
    source_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    raw_stem_path: str
    raw_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sample_rate_hz: int = Field(gt=0)
    raw_channels: Literal[1] = 1
    raw_frame_count: int = Field(gt=0)
    raw_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    canonical_stem_path: str
    canonical_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_sample_rate_hz: Literal[32000] = STEM_SAMPLE_RATE_HZ
    canonical_channels: Literal[2] = STEM_CHANNELS
    canonical_frame_count: int = Field(gt=0)
    canonical_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    raw_duration_delta_seconds: float = Field(allow_inf_nan=False)
    canonical_duration_delta_seconds: float = Field(allow_inf_nan=False)
    canonical_uncovered_tail_seconds: float = Field(ge=0, allow_inf_nan=False)
    alignment_tolerance_seconds: Literal[0.1] = STEM_ALIGNMENT_TOLERANCE_SECONDS
    alignment_state: Literal[
        "normal_bounded_drift",
        "accepted_with_alignment_warning",
    ]
    timeline_policy: Literal["zero_offset_bounded_duration_drift_v1"] = (
        STEM_TIMELINE_POLICY
    )
    canonical_channel_policy: Literal["dual_mono_from_sam_mono_v1"] = (
        STEM_CANONICAL_CHANNEL_POLICY
    )
    conversion_policy_version: Literal[
        "sam_audio_raw_to_32k_stereo_pcm16_v1"
    ] = STEM_CANONICALIZATION_VERSION
    conversion_settings: list[str]
    alignment_offset_seconds: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def validate_alignment(self) -> StemArtifact:
        if self.source_end_sample != round(
            self.source_duration_seconds * STEM_SAMPLE_RATE_HZ
        ):
            raise ValueError("stem source sample extent differs from source duration")
        expected_raw_delta = self.raw_duration_seconds - self.source_duration_seconds
        expected_canonical_delta = (
            self.canonical_duration_seconds - self.source_duration_seconds
        )
        if not math.isclose(
            self.raw_duration_delta_seconds,
            expected_raw_delta,
            rel_tol=0,
            abs_tol=1e-9,
        ) or not math.isclose(
            self.canonical_duration_delta_seconds,
            expected_canonical_delta,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("SAM Audio stem duration diagnostics are inconsistent")
        for delta in (expected_raw_delta, expected_canonical_delta):
            if abs(delta) > STEM_ALIGNMENT_TOLERANCE_SECONDS:
                raise ValueError("SAM Audio stem duration differs from source timeline")
        expected_uncovered = max(0.0, -expected_canonical_delta)
        if not math.isclose(
            self.canonical_uncovered_tail_seconds,
            expected_uncovered,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("canonical SAM Audio uncovered tail is inconsistent")
        expected_state = (
            "accepted_with_alignment_warning"
            if max(abs(expected_raw_delta), abs(expected_canonical_delta))
            > STEM_ALIGNMENT_WARNING_SECONDS
            else "normal_bounded_drift"
        )
        if self.alignment_state != expected_state:
            raise ValueError("SAM Audio stem alignment state is inconsistent")
        if self.raw_duration_seconds != self.raw_frame_count / self.raw_sample_rate_hz:
            raise ValueError("raw SAM Audio duration differs from sample extent")
        if self.canonical_duration_seconds != (
            self.canonical_frame_count / self.canonical_sample_rate_hz
        ):
            raise ValueError("canonical stem duration differs from sample extent")
        return self


class SAMAudioStemJob(SchemaModel):
    clip_uid: str
    clip_display_path: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_path: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: Literal[32000] = STEM_SAMPLE_RATE_HZ
    source_channels: Literal[2] = STEM_CHANNELS
    source_frame_count: int = Field(gt=0)
    source_duration_seconds: float = Field(gt=0, allow_inf_nan=False)


class SAMAudioStemInventory(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_inventory.2", "r2v.h3.sam_audio_stem_inventory.3"] = (
        SAM_AUDIO_STEM_INVENTORY_VERSION
    )
    source_canonical_audio_manifest_path: str
    source_canonical_audio_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_case_manifest_path: str | None = None
    source_case_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    selection_mode: Literal["all_canonical_clips", "explicit_case_manifest"]
    route: SAMRoute
    run_both_routes: bool = False
    model_configuration: SAMAudioModelConfiguration | _LegacySAMAudioModelConfiguration
    job_count: int = Field(ge=1)
    clip_uids: list[str] = Field(min_length=1)
    jobs: list[SAMAudioStemJob] = Field(min_length=1)
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def parse_versioned_configuration(cls, values):
        if isinstance(values, dict) and "model_configuration" in values:
            configuration = values["model_configuration"]
            if isinstance(configuration, SAMAudioModelConfiguration):
                configuration = configuration.model_dump(mode="json")
            model = (_LegacySAMAudioModelConfiguration
                     if values.get("schema_version") == "r2v.h3.sam_audio_stem_inventory.2"
                     else SAMAudioModelConfiguration)
            return {**values, "model_configuration": model.model_validate(configuration)}
        return values

    @model_validator(mode="after")
    def validate_inventory(self) -> SAMAudioStemInventory:
        if self.job_count != len(self.jobs):
            raise ValueError("SAM Audio stem inventory job count differs")
        ids = [item.clip_uid for item in self.jobs]
        if len(ids) != len(set(ids)) or self.clip_uids != ids:
            raise ValueError("SAM Audio stem inventory clip IDs must be unique")
        explicit = self.selection_mode == "explicit_case_manifest"
        if explicit != (self.source_case_manifest_path is not None) or explicit != (
            self.source_case_manifest_sha256 is not None
        ):
            raise ValueError("SAM Audio case-manifest provenance is incomplete")
        values = self.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if self.schema_version == "r2v.h3.sam_audio_stem_inventory.2":
            for name in ("speech_prompt", "music_prompt"):
                values["model_configuration"].pop(name)
        if self.inventory_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("SAM Audio stem inventory fingerprint is invalid")
        return self


class SAMAudioStemRecord(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_record.3"] = (
        SAM_AUDIO_STEM_RECORD_VERSION
    )
    clip_uid: str
    route: SAMRoute
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    separation_state: VerificationState
    model_call_count: int = Field(ge=0, le=2)
    calls: list[SAMAudioCallProvenance]
    stems: list[StemArtifact]
    failure_reason: str | None = None
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record(self) -> SAMAudioStemRecord:
        if self.separation_state == "failure":
            if (
                self.failure_reason is None
                or self.stems
                or len(self.calls) > self.model_call_count
            ):
                raise ValueError("failed SAM Audio record requires only failure evidence")
        elif (
            self.failure_reason is not None
            or len(self.calls) != 2
            or self.model_call_count != 2
        ):
            raise ValueError("usable SAM Audio record requires exactly two calls")
        stem_types = [item.stem_type for item in self.stems]
        if self.separation_state != "failure" and stem_types != [
            "music",
            "speech",
            "sfx",
        ]:
            raise ValueError("SAM Audio record must publish three ordered stems")
        values = self.model_dump(mode="json", exclude={"record_fingerprint"})
        if self.record_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("SAM Audio stem record fingerprint is invalid")
        return self

    def stem(self, stem_type: StemType) -> StemArtifact:
        matches = [item for item in self.stems if item.stem_type == stem_type]
        if len(matches) != 1:
            raise ValueError(f"SAM Audio record lacks one {stem_type} stem")
        return matches[0]


class SAMAudioStemSummary(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_summary.2"] = (
        SAM_AUDIO_STEM_SUMMARY_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_count: int = Field(ge=1)
    record_count: int = Field(ge=1)
    route_counts: dict[str, int]
    verification_state_counts: dict[str, int]
    model_call_count: int = Field(ge=0)
    production_artifacts_modified: Literal[False] = False
    network_download_performed: Literal[False] = False


class StemCanonicalizer(Protocol):
    def canonicalize(
        self,
        *,
        raw_stem_path: Path,
        canonical_stem_path: Path,
    ) -> tuple[AudioFileProbe, list[str]]: ...


class FFmpegStemCanonicalizer:
    def __init__(self, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.media = FFmpegAudioMediaBackend(ffmpeg=ffmpeg, ffprobe=ffprobe)

    def canonicalize(
        self,
        *,
        raw_stem_path: Path,
        canonical_stem_path: Path,
    ) -> tuple[AudioFileProbe, list[str]]:
        source = raw_stem_path.expanduser().resolve(strict=True)
        destination = canonical_stem_path.expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg,
            "-v",
            "error",
            "-i",
            str(source),
            "-ar",
            str(STEM_SAMPLE_RATE_HZ),
            "-ac",
            str(STEM_CHANNELS),
            "-c:a",
            "pcm_s16le",
            "-y",
        ]
        FFmpegAudioMediaBackend._publish_command(command, destination)
        probe = self.media.probe_audio_file(destination)
        if (
            probe.sample_rate_hz != STEM_SAMPLE_RATE_HZ
            or probe.channels != STEM_CHANNELS
            or "wav" not in probe.format_name.lower()
        ):
            raise ValueError("canonical SAM Audio stem must be 32 kHz stereo WAV")
        return probe, command[1:-1]


def build_sam_audio_stem_inventory(
    *,
    canonical_audio_manifest_path: Path,
    model_configuration: SAMAudioModelConfiguration,
    route: SAMRoute = "music_first",
    run_both_routes: bool = False,
    case_manifest_path: Path | None = None,
) -> SAMAudioStemInventory:
    manifest_path = canonical_audio_manifest_path.expanduser().resolve(strict=True)
    canonical = [
        CanonicalAudioClip.model_validate(row) for row in _read_jsonl(manifest_path)
    ]
    by_clip = {item.clip_uid: item for item in canonical}
    if len(by_clip) != len(canonical) or not canonical:
        raise ValueError("canonical Audio manifest must contain unique clips")
    case_path: Path | None = None
    case_hash: str | None = None
    if case_manifest_path is not None:
        case_path = case_manifest_path.expanduser().resolve(strict=True)
        case = MimoCaseManifest.model_validate_json(case_path.read_text(encoding="utf-8"))
        unknown = set(case.clip_uids) - set(by_clip)
        if unknown:
            raise ValueError(f"SAM Audio case manifest contains unknown clips: {sorted(unknown)}")
        selected = [by_clip[item] for item in case.clip_uids]
        case_hash = sha256_file(case_path)
    else:
        selected = canonical
    jobs = [
        SAMAudioStemJob(
            clip_uid=item.clip_uid,
            clip_display_path=item.clip_display_path,
            target_video_path=item.target_video_path,
            target_video_sha256=item.target_video_sha256,
            source_audio_path=item.target_full_audio_path,
            source_audio_sha256=item.target_full_audio_sha256,
            source_frame_count=item.frame_count,
            source_duration_seconds=item.target_duration_seconds,
        )
        for item in selected
    ]
    values = {
        "schema_version": SAM_AUDIO_STEM_INVENTORY_VERSION,
        "source_canonical_audio_manifest_path": str(manifest_path),
        "source_canonical_audio_manifest_sha256": sha256_file(manifest_path),
        "source_case_manifest_path": str(case_path) if case_path else None,
        "source_case_manifest_sha256": case_hash,
        "selection_mode": (
            "explicit_case_manifest" if case_path else "all_canonical_clips"
        ),
        "route": route,
        "run_both_routes": run_both_routes,
        "model_configuration": model_configuration.model_dump(mode="json"),
        "job_count": len(jobs),
        "clip_uids": [item.clip_uid for item in jobs],
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    return SAMAudioStemInventory(
        **values,
        inventory_fingerprint=_sha256_text(_compact_json(values)),
    )


def _call_provenance(
    *,
    pass_index: Literal[1, 2],
    prompt: str,
    input_path: Path,
    result: SAMAudioSeparationResult,
) -> SAMAudioCallProvenance:
    target = Path(result.target_path).expanduser().resolve(strict=True)
    residual = Path(result.residual_path).expanduser().resolve(strict=True)
    return SAMAudioCallProvenance(
        pass_index=pass_index,
        prompt=prompt,
        input_path=str(input_path.expanduser().resolve(strict=True)),
        input_sha256=sha256_file(input_path),
        target_path=str(target),
        target_sha256=sha256_file(target),
        residual_path=str(residual),
        residual_sha256=sha256_file(residual),
        verification_state=result.verification_state,
        candidate_count=result.candidate_count,
        selected_candidate_index=result.selected_candidate_index,
        judge_scores=result.judge_scores,
        clap_score=result.clap_score,
        runtime_seconds=result.runtime_seconds,
        backend_metadata=result.backend_metadata,
    )


def _require_separation_result_paths(
    result: SAMAudioSeparationResult,
    *,
    target_path: Path,
    residual_path: Path,
) -> None:
    if (
        Path(result.target_path).expanduser().resolve(strict=True)
        != target_path.resolve(strict=True)
        or Path(result.residual_path).expanduser().resolve(strict=True)
        != residual_path.resolve(strict=True)
    ):
        raise ValueError("SAM Audio backend returned an unexpected output path")


def _stem_artifact(
    *,
    job: SAMAudioStemJob,
    stem_type: StemType,
    raw_path: Path,
    canonical_path: Path,
    canonicalizer: StemCanonicalizer,
    raw_probe_backend: AudioMediaBackend,
) -> StemArtifact:
    raw_probe = raw_probe_backend.probe_audio_file(raw_path)
    canonical_probe, settings = canonicalizer.canonicalize(
        raw_stem_path=raw_path,
        canonical_stem_path=canonical_path,
    )
    return StemArtifact(
        stem_type=stem_type,
        source_audio_path=job.source_audio_path,
        source_audio_sha256=job.source_audio_sha256,
        source_end_sample=job.source_frame_count,
        source_duration_seconds=job.source_duration_seconds,
        raw_stem_path=str(raw_path.resolve(strict=True)),
        raw_stem_sha256=sha256_file(raw_path),
        raw_sample_rate_hz=raw_probe.sample_rate_hz,
        raw_channels=raw_probe.channels,
        raw_frame_count=raw_probe.frame_count,
        raw_duration_seconds=raw_probe.duration_seconds,
        canonical_stem_path=str(canonical_path.resolve(strict=True)),
        canonical_stem_sha256=sha256_file(canonical_path),
        canonical_frame_count=canonical_probe.frame_count,
        canonical_duration_seconds=canonical_probe.duration_seconds,
        raw_duration_delta_seconds=(
            raw_probe.duration_seconds - job.source_duration_seconds
        ),
        canonical_duration_delta_seconds=(
            canonical_probe.duration_seconds - job.source_duration_seconds
        ),
        canonical_uncovered_tail_seconds=max(
            0.0,
            job.source_duration_seconds - canonical_probe.duration_seconds,
        ),
        alignment_state=(
            "accepted_with_alignment_warning"
            if max(
                abs(raw_probe.duration_seconds - job.source_duration_seconds),
                abs(canonical_probe.duration_seconds - job.source_duration_seconds),
            )
            > STEM_ALIGNMENT_WARNING_SECONDS
            else "normal_bounded_drift"
        ),
        conversion_settings=settings,
    )


class _SAMAudioRouteFailure(Exception):
    def __init__(
        self,
        cause: Exception,
        *,
        calls: Sequence[SAMAudioCallProvenance],
        model_call_count: int,
    ) -> None:
        super().__init__(f"{type(cause).__name__}:{cause}")
        self.calls = list(calls)
        self.model_call_count = model_call_count


def _separate_one_route(
    *,
    inventory: SAMAudioStemInventory,
    job: SAMAudioStemJob,
    route: SAMRoute,
    output_root: Path,
    backend: SAMAudioBackend,
    canonicalizer: StemCanonicalizer,
    raw_probe_backend: AudioMediaBackend,
) -> SAMAudioStemRecord:
    source = Path(job.source_audio_path).expanduser().resolve(strict=True)
    if sha256_file(source) != job.source_audio_sha256:
        raise ValueError("canonical source Audio hash changed before SAM separation")
    source_probe = raw_probe_backend.probe_audio_file(source)
    if (
        source_probe.sample_rate_hz != STEM_SAMPLE_RATE_HZ
        or source_probe.channels != STEM_CHANNELS
        or source_probe.frame_count != job.source_frame_count
    ):
        raise ValueError("canonical source Audio format or sample extent changed")
    raw_root = output_root / "stems" / job.clip_uid / route
    canonical_root = output_root / "canonical_stems" / job.clip_uid / route
    raw_root.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)
    first_prompt = (
        inventory.model_configuration.music_prompt if route == "music_first" else inventory.model_configuration.speech_prompt
    )
    second_prompt = (
        inventory.model_configuration.speech_prompt if route == "music_first" else inventory.model_configuration.music_prompt
    )
    first_target = raw_root / f"{'music' if route == 'music_first' else 'speech'}.raw.wav"
    intermediate = raw_root / "residual_1.raw.wav"
    completed_calls: list[SAMAudioCallProvenance] = []
    model_call_count = 0
    try:
        model_call_count += 1
        first_result = backend.separate(
            clip_uid=job.clip_uid,
            source_audio_path=source,
            prompt=first_prompt,
            target_path=first_target,
            residual_path=intermediate,
        )
        _require_separation_result_paths(
            first_result,
            target_path=first_target,
            residual_path=intermediate,
        )
        completed_calls.append(
            _call_provenance(
                pass_index=1,
                prompt=first_prompt,
                input_path=source,
                result=first_result,
            )
        )
        if first_result.verification_state == "failure":
            raise ValueError("first SAM Audio separation reported failure")

        second_target = (
            raw_root
            / f"{'speech' if route == 'music_first' else 'music'}.raw.wav"
        )
        sfx_raw = raw_root / "sfx.raw.wav"
        model_call_count += 1
        second_result = backend.separate(
            clip_uid=job.clip_uid,
            source_audio_path=intermediate,
            prompt=second_prompt,
            target_path=second_target,
            residual_path=sfx_raw,
        )
        _require_separation_result_paths(
            second_result,
            target_path=second_target,
            residual_path=sfx_raw,
        )
        completed_calls.append(
            _call_provenance(
                pass_index=2,
                prompt=second_prompt,
                input_path=intermediate,
                result=second_result,
            )
        )
        if second_result.verification_state == "failure":
            raise ValueError("second SAM Audio separation reported failure")
        raw_by_type = {
            "music": first_target if route == "music_first" else second_target,
            "speech": second_target if route == "music_first" else first_target,
            "sfx": sfx_raw,
        }
        stems = [
            _stem_artifact(
                job=job,
                stem_type=stem_type,
                raw_path=raw_by_type[stem_type],
                canonical_path=canonical_root / f"{stem_type}.wav",
                canonicalizer=canonicalizer,
                raw_probe_backend=raw_probe_backend,
            )
            for stem_type in ("music", "speech", "sfx")
        ]
    except Exception as exc:
        raise _SAMAudioRouteFailure(
            exc,
            calls=completed_calls,
            model_call_count=model_call_count,
        ) from exc
    states = {item.verification_state for item in completed_calls}
    state: VerificationState = (
        "uncertain"
        if "uncertain" in states
        else "success"
        if states == {"success"}
        else "unverified"
    )
    values = {
        "schema_version": SAM_AUDIO_STEM_RECORD_VERSION,
        "clip_uid": job.clip_uid,
        "route": route,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "model_configuration_fingerprint": (
            inventory.model_configuration.configuration_fingerprint
        ),
        "separation_state": state,
        "model_call_count": model_call_count,
        "calls": [item.model_dump(mode="json") for item in completed_calls],
        "stems": [item.model_dump(mode="json") for item in stems],
        "failure_reason": None,
    }
    record = SAMAudioStemRecord(
        **values,
        record_fingerprint=_sha256_text(_compact_json(values)),
    )
    _write_json(raw_root / "separation.json", record)
    return record


def _published_record_paths(
    record: SAMAudioStemRecord,
    *,
    temporary_root: Path,
    destination_root: Path,
) -> SAMAudioStemRecord:
    old = str(temporary_root)
    new = str(destination_root)
    values = record.model_dump(mode="json", exclude={"record_fingerprint"})
    for call in values["calls"]:
        for key in ("input_path", "target_path", "residual_path"):
            value = call[key]
            if isinstance(value, str) and (value == old or value.startswith(old + "/")):
                call[key] = new + value[len(old) :]
    for stem in values["stems"]:
        for key in ("raw_stem_path", "canonical_stem_path"):
            value = stem[key]
            if isinstance(value, str) and (value == old or value.startswith(old + "/")):
                stem[key] = new + value[len(old) :]
    return SAMAudioStemRecord(
        **values,
        record_fingerprint=_sha256_text(_compact_json(values)),
    )


def run_sam_audio_stem_shadow(
    *,
    inventory: SAMAudioStemInventory,
    output_root: Path,
    backend: SAMAudioBackend,
    canonicalizer: StemCanonicalizer,
    raw_probe_backend: AudioMediaBackend,
    overwrite: bool = False,
) -> SAMAudioStemSummary:
    if inventory.schema_version != SAM_AUDIO_STEM_INVENTORY_VERSION:
        raise ValueError("legacy SAM Audio inventory is read-only; build a current inventory to run")
    if backend.configuration != inventory.model_configuration:
        raise ValueError("SAM Audio backend differs from inventory configuration")
    destination = output_root.expanduser().resolve(strict=False)
    if destination.name != SAM_AUDIO_SEPARATION_STAGE_NAME:
        raise ValueError("SAM Audio separation must own the separation stage root")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    routes: list[SAMRoute] = (
        [inventory.route, "voice_first" if inventory.route == "music_first" else "music_first"]
        if inventory.run_both_routes
        else [inventory.route]
    )
    records: list[SAMAudioStemRecord] = []
    try:
        temporary.mkdir()
        _write_json(temporary / "inventory.json", inventory)
        for job in inventory.jobs:
            for route in routes:
                try:
                    records.append(
                        _separate_one_route(
                            inventory=inventory,
                            job=job,
                            route=route,
                            output_root=temporary,
                            backend=backend,
                            canonicalizer=canonicalizer,
                            raw_probe_backend=raw_probe_backend,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - isolate one route/clip
                    route_failure = (
                        exc if isinstance(exc, _SAMAudioRouteFailure) else None
                    )
                    values = {
                        "schema_version": SAM_AUDIO_STEM_RECORD_VERSION,
                        "clip_uid": job.clip_uid,
                        "route": route,
                        "inventory_fingerprint": inventory.inventory_fingerprint,
                        "model_configuration_fingerprint": (
                            inventory.model_configuration.configuration_fingerprint
                        ),
                        "separation_state": "failure",
                        "model_call_count": (
                            route_failure.model_call_count if route_failure else 0
                        ),
                        "calls": [
                            item.model_dump(mode="json")
                            for item in (route_failure.calls if route_failure else [])
                        ],
                        "stems": [],
                        "failure_reason": str(route_failure or f"{type(exc).__name__}:{exc}"),
                    }
                    records.append(
                        SAMAudioStemRecord(
                            **values,
                            record_fingerprint=_sha256_text(_compact_json(values)),
                        )
                    )
        records = [
            _published_record_paths(
                item,
                temporary_root=temporary,
                destination_root=destination,
            )
            for item in records
        ]
        for item in records:
            _write_json(
                temporary / "stems" / item.clip_uid / item.route / "separation.json",
                item,
            )
        route_counts = Counter(item.route for item in records)
        state_counts = Counter(item.separation_state for item in records)
        summary = SAMAudioStemSummary(
            inventory_fingerprint=inventory.inventory_fingerprint,
            clip_count=inventory.job_count,
            record_count=len(records),
            route_counts=dict(sorted(route_counts.items())),
            verification_state_counts=dict(sorted(state_counts.items())),
            model_call_count=sum(item.model_call_count for item in records),
        )
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_stem_shadow(
    root: Path,
) -> tuple[SAMAudioStemInventory, list[SAMAudioStemRecord], SAMAudioStemSummary]:
    source = root.expanduser().resolve(strict=True)
    if source.name != SAM_AUDIO_SEPARATION_STAGE_NAME:
        raise ValueError("SAM Audio records must be loaded from the separation stage")
    inventory = SAMAudioStemInventory.model_validate_json(
        (source / "inventory.json").read_text(encoding="utf-8")
    )
    records = [
        SAMAudioStemRecord.model_validate(row)
        for row in _read_jsonl(source / "records.jsonl")
    ]
    summary = SAMAudioStemSummary.model_validate_json(
        (source / "summary.json").read_text(encoding="utf-8")
    )
    if summary.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError("SAM Audio summary differs from inventory")
    if summary.record_count != len(records) or any(
        item.inventory_fingerprint != inventory.inventory_fingerprint for item in records
    ):
        raise ValueError("SAM Audio record inventory is inconsistent")
    for record in records:
        for stem in record.stems:
            for path in (stem.raw_stem_path, stem.canonical_stem_path):
                require_shadow_output_path(shadow_root=source, output_path=Path(path))
    return inventory, records, summary


def selected_stem_records(
    records: Sequence[SAMAudioStemRecord],
    *,
    route: SAMRoute,
    allow_unverified: bool = False,
) -> list[SAMAudioStemRecord]:
    selected = [
        item
        for item in records
        if item.route == route and item.separation_state != "failure"
    ]
    ids = [item.clip_uid for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("selected SAM Audio route contains duplicate clips")
    if any(item.separation_state == "unverified" for item in selected) and not (
        allow_unverified
    ):
        raise ValueError(
            "unverified SAM Audio stems require explicit allow_unverified opt-in"
        )
    return selected


class StemShadowClipSkip(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_clip_skip.1"] = (
        STEM_CLIP_SKIP_VERSION
    )
    clip_uid: str
    route: SAMRoute
    source_stage: Literal["separation"] = "separation"
    reason_code: Literal["sam_audio_separation_failed"] = (
        "sam_audio_separation_failed"
    )
    reason: str


class StemDiarizationClipFailure(SchemaModel):
    schema_version: Literal["r2v.h3.stem_diarization_clip_failure.1"] = (
        STEM_DIARIZATION_FAILURE_VERSION
    )
    clip_uid: str
    route: SAMRoute
    source_stage: Literal["diarization"] = "diarization"
    reason_code: Literal["stem_diarization_failed"] = "stem_diarization_failed"
    reason: str
    source_clip_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _diarization_failure(
    result: DiarizationClipResult,
    *,
    route: SAMRoute,
) -> StemDiarizationClipFailure:
    if result.status != "failed" or not result.reason:
        raise ValueError("stem DiariZen failure requires a failed clip result")
    return StemDiarizationClipFailure(
        clip_uid=result.target_clip_uid,
        route=route,
        reason=result.reason,
        source_clip_result_sha256=_sha256_text(
            _compact_json(result.model_dump(mode="json"))
        ),
    )


def separation_skips(
    *,
    inventory: SAMAudioStemInventory,
    records: Sequence[SAMAudioStemRecord],
    route: SAMRoute,
) -> list[StemShadowClipSkip]:
    route_records = [item for item in records if item.route == route]
    by_clip = {item.clip_uid: item for item in route_records}
    if len(by_clip) != len(route_records) or set(by_clip) != set(inventory.clip_uids):
        raise ValueError("SAM Audio route records differ from ordered inventory")
    return [
        StemShadowClipSkip(
            clip_uid=clip_uid,
            route=route,
            reason=by_clip[clip_uid].failure_reason or "SAM Audio separation failed",
        )
        for clip_uid in inventory.clip_uids
        if by_clip[clip_uid].separation_state == "failure"
    ]


class StemReferenceRequest(SchemaModel):
    reference_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    clip_uid: str
    route: SAMRoute
    stem_type: StemType
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    reference_kind: Literal["manual_review", "shadow_primary_voice"] = (
        "manual_review"
    )
    entity_occurrence_id: str | None = None
    source_turn_id: str | None = None
    source_quality_policy_version: str | None = None
    source_quality_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_selection_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_interval(self) -> StemReferenceRequest:
        if self.end_time <= self.start_time:
            raise ValueError("stem reference interval must be positive")
        voice_provenance = (
            self.entity_occurrence_id,
            self.source_turn_id,
            self.source_quality_policy_version,
            self.source_quality_policy_fingerprint,
            self.source_selection_fingerprint,
        )
        if self.reference_kind == "shadow_primary_voice":
            if self.stem_type != "speech" or any(
                value is None or not value.strip() for value in voice_provenance
            ):
                raise ValueError(
                    "shadow primary voice requires speech-stem quality provenance"
                )
        elif any(value is not None for value in voice_provenance):
            raise ValueError("manual stem references cannot claim voice eligibility")
        return self


class StemReferenceManifest(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_reference_manifest.1"] = (
        "r2v.h3.sam_audio_stem_reference_manifest.1"
    )
    requests: list[StemReferenceRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_requests(self) -> StemReferenceManifest:
        ids = [
            (item.clip_uid, item.route, item.reference_id) for item in self.requests
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("stem reference IDs must be unique")
        return self


class StemNativeReference(SchemaModel):
    schema_version: Literal["r2v.h3.sam_audio_stem_reference.2"] = (
        STEM_REFERENCE_VERSION
    )
    reference_id: str
    clip_uid: str
    route: SAMRoute
    stem_type: StemType
    source_stem_path: str
    source_stem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    sample_rate_hz: Literal[32000] = STEM_SAMPLE_RATE_HZ
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    reference_path: str
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_verification_state: VerificationState
    unverified_consumption_authorized: bool
    reference_kind: Literal["manual_review", "shadow_primary_voice"]
    entity_occurrence_id: str | None = None
    source_turn_id: str | None = None
    source_quality_policy_version: str | None = None
    source_quality_policy_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    source_selection_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    extraction_policy_version: Literal[
        "sam_audio_stem_exact_sample_crop_v1"
    ] = STEM_REFERENCE_EXTRACTION_VERSION

    @model_validator(mode="after")
    def validate_reference(self) -> StemNativeReference:
        if (
            self.end_sample <= self.start_sample
            or self.start_time != self.start_sample / self.sample_rate_hz
            or self.end_time != self.end_sample / self.sample_rate_hz
            or not self.reference_path.strip()
        ):
            raise ValueError("stem-native reference extent is inconsistent")
        if (
            self.source_verification_state == "unverified"
            and not self.unverified_consumption_authorized
        ):
            raise ValueError("unverified stem reference lacks explicit authorization")
        voice_provenance = (
            self.entity_occurrence_id,
            self.source_turn_id,
            self.source_quality_policy_version,
            self.source_quality_policy_fingerprint,
            self.source_selection_fingerprint,
        )
        if self.reference_kind == "shadow_primary_voice":
            if self.stem_type != "speech" or any(
                value is None or not value.strip() for value in voice_provenance
            ):
                raise ValueError("shadow primary voice provenance is incomplete")
        elif any(value is not None for value in voice_provenance):
            raise ValueError("manual stem reference contains voice provenance")
        return self


def _resolve_owned_asset(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("primary voice asset path must be safe and relative")
    resolved = (root / relative).resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("primary voice asset escapes its owned root")
    if not resolved.is_file():
        raise FileNotFoundError("primary voice asset is not a file")
    return resolved


def build_shadow_primary_voice_reference_requests(
    *,
    primary_voice_root: Path,
    records: Sequence[SAMAudioStemRecord],
    route: SAMRoute,
    selected_clip_uids: Sequence[str] | None = None,
    allow_unverified: bool = False,
) -> list[StemReferenceRequest]:
    """Project already-selected V1 primary turns onto the speech stem timeline."""

    root = primary_voice_root.expanduser().resolve(strict=True)
    selections = [
        PrimaryVoiceReferenceSelection.model_validate(row)
        for row in _read_jsonl(root / "primary_voice_references.jsonl")
    ]
    summary = PrimaryVoiceReferenceExportSummary.model_validate_json(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    if summary.entity_occurrence_count != len(selections) or (
        summary.occurrences_with_primary_voice_reference
        != sum(item.primary_voice_reference is not None for item in selections)
    ):
        raise ValueError("primary voice selection summary does not reconcile")

    selected_records = selected_stem_records(
        records,
        route=route,
        allow_unverified=allow_unverified,
    )
    records_by_clip = {item.clip_uid: item for item in selected_records}
    if selected_clip_uids is None:
        clip_order = [item.clip_uid for item in selected_records]
    else:
        clip_order = list(selected_clip_uids)
        if len(clip_order) != len(set(clip_order)) or any(
            clip_uid not in records_by_clip for clip_uid in clip_order
        ):
            raise ValueError("voice reference case selection differs from stem records")

    selections_by_clip: dict[str, list[PrimaryVoiceReferenceSelection]] = {}
    for selection in selections:
        selections_by_clip.setdefault(selection.clip_uid, []).append(selection)

    requests: list[StemReferenceRequest] = []
    for clip_uid in clip_order:
        for selection in sorted(
            selections_by_clip.get(clip_uid, []),
            key=lambda item: item.entity_id,
        ):
            artifact = selection.primary_voice_reference
            if artifact is None:
                continue
            metadata = artifact.quality_metadata
            assessment = metadata.get("assessment")
            if not isinstance(assessment, dict) or assessment.get("status") != "accepted":
                raise ValueError("shadow primary voice requires an accepted assessment")
            if (
                artifact.source_turn_id not in selection.accepted_turn_ids
                or assessment.get("turn_id") != artifact.source_turn_id
                or assessment.get("entity_occurrence_id")
                != selection.entity_occurrence_id
                or metadata.get("policy_version") != selection.policy_version
                or metadata.get("policy_fingerprint") != selection.policy_fingerprint
                or metadata.get("thresholds_calibrated") is not True
            ):
                raise ValueError("primary voice quality provenance is inconsistent")
            source_asset = _resolve_owned_asset(root, artifact.asset.path)
            if sha256_file(source_asset) != artifact.asset.sha256:
                raise ValueError("primary voice source asset hash changed")
            selection_fingerprint = _sha256_text(
                _compact_json(selection.model_dump(mode="json"))
            )
            requests.append(
                StemReferenceRequest(
                    reference_id=f"primary_voice_{selection.entity_id}",
                    clip_uid=clip_uid,
                    route=route,
                    stem_type="speech",
                    start_time=artifact.source_start,
                    end_time=artifact.source_end,
                    reference_kind="shadow_primary_voice",
                    entity_occurrence_id=selection.entity_occurrence_id,
                    source_turn_id=artifact.source_turn_id,
                    source_quality_policy_version=selection.policy_version,
                    source_quality_policy_fingerprint=selection.policy_fingerprint,
                    source_selection_fingerprint=selection_fingerprint,
                )
            )
    return requests


def export_stem_native_references(
    *,
    records: Sequence[SAMAudioStemRecord],
    requests: Sequence[StemReferenceRequest],
    output_root: Path,
    media_backend: AudioMediaBackend,
    allow_unverified: bool = False,
) -> list[StemNativeReference]:
    by_key = {(item.clip_uid, item.route): item for item in records}
    if len(by_key) != len(records):
        raise ValueError("SAM Audio stem records are duplicated")
    output: list[StemNativeReference] = []
    for request in requests:
        record = by_key.get((request.clip_uid, request.route))
        if record is None or record.separation_state == "failure":
            raise ValueError("stem reference request lacks a usable stem record")
        if record.separation_state == "unverified" and not allow_unverified:
            raise ValueError(
                "unverified SAM Audio stems require explicit allow_unverified opt-in"
            )
        stem = record.stem(request.stem_type)
        source = Path(stem.canonical_stem_path).resolve(strict=True)
        if sha256_file(source) != stem.canonical_stem_sha256:
            raise ValueError("stem reference source hash changed")
        start = round(request.start_time * STEM_SAMPLE_RATE_HZ)
        end = round(request.end_time * STEM_SAMPLE_RATE_HZ)
        available_end = min(stem.source_end_sample, stem.canonical_frame_count)
        if start < 0 or end <= start or end > available_end:
            raise ValueError("stem reference interval exceeds canonical stem")
        destination = (
            output_root.expanduser().resolve(strict=False)
            / request.clip_uid
            / request.route
            / f"{request.reference_id}.flac"
        )
        media_backend.extract_voice_reference(
            clip_uid=request.clip_uid,
            entity_id=request.reference_id,
            full_audio_path=source,
            start_time=request.start_time,
            end_time=request.end_time,
            destination=destination,
            sample_rate_hz=STEM_SAMPLE_RATE_HZ,
            output_format="flac",
            channels=STEM_CHANNELS,
            source_audio_path=source,
            source_start_sample=start,
            source_end_sample=end,
        )
        output.append(
            StemNativeReference(
                reference_id=request.reference_id,
                clip_uid=request.clip_uid,
                route=request.route,
                stem_type=request.stem_type,
                source_stem_path=str(source),
                source_stem_sha256=stem.canonical_stem_sha256,
                start_sample=start,
                end_sample=end,
                start_time=start / STEM_SAMPLE_RATE_HZ,
                end_time=end / STEM_SAMPLE_RATE_HZ,
                reference_path=str(destination.resolve(strict=True)),
                reference_sha256=sha256_file(destination),
                source_verification_state=record.separation_state,
                unverified_consumption_authorized=(
                    record.separation_state != "unverified" or allow_unverified
                ),
                reference_kind=request.reference_kind,
                entity_occurrence_id=request.entity_occurrence_id,
                source_turn_id=request.source_turn_id,
                source_quality_policy_version=request.source_quality_policy_version,
                source_quality_policy_fingerprint=(
                    request.source_quality_policy_fingerprint
                ),
                source_selection_fingerprint=request.source_selection_fingerprint,
            )
        )
    output.sort(key=lambda item: (item.clip_uid, item.route, item.reference_id))
    _write_jsonl(output_root / "references.jsonl", output)
    return output


class StemDiarizationShadowProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.stem_diarization_shadow.4"] = (
        STEM_DIARIZATION_SHADOW_VERSION
    )
    diarization_source_kind: Literal["sam_audio_speech_stem"] = (
        "sam_audio_speech_stem"
    )
    source_stem_root: str
    source_stem_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bound_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cluster_bindings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_results_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route: SAMRoute
    target_count: int = Field(ge=1)
    clip_uids: list[str] = Field(min_length=1)
    ready_clip_uids: list[str]
    empty_clip_uids: list[str]
    failed_clip_uids: list[str]
    usable_clip_uids: list[str]
    skipped_clips: list[StemShadowClipSkip]
    diarization_failed_clips: list[StemDiarizationClipFailure]
    unverified_clip_uids: list[str]
    unverified_consumption_authorized: bool
    speech_stem_paths_by_clip: dict[str, str]
    speech_stem_hashes_by_clip: dict[str, str]
    original_audio_paths_by_clip: dict[str, str]
    original_audio_hashes_by_clip: dict[str, str]
    production_diarization_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_clip_provenance(self) -> StemDiarizationShadowProvenance:
        skipped = {item.clip_uid for item in self.skipped_clips}
        attempted = [item for item in self.clip_uids if item not in skipped]
        ready = set(self.ready_clip_uids)
        empty = set(self.empty_clip_uids)
        failed = set(self.failed_clip_uids)
        if (
            len(self.clip_uids) != len(set(self.clip_uids))
            or self.target_count != len(attempted)
            or skipped != set(self.clip_uids) - set(attempted)
            or any(
                [item for item in attempted if item in values] != ordered
                for values, ordered in (
                    (ready, self.ready_clip_uids),
                    (empty, self.empty_clip_uids),
                    (failed, self.failed_clip_uids),
                )
            )
            or ready.intersection(empty)
            or ready.intersection(failed)
            or empty.intersection(failed)
            or ready.union(empty).union(failed) != set(attempted)
            or self.usable_clip_uids
            != [item for item in attempted if item in ready.union(empty)]
            or [item.clip_uid for item in self.diarization_failed_clips]
            != self.failed_clip_uids
            or any(
                item.route != self.route for item in self.diarization_failed_clips
            )
        ):
            raise ValueError("stem DiariZen ordered clip provenance is inconsistent")
        usable = set(self.usable_clip_uids)
        if any(
            set(values) != usable
            for values in (
                self.speech_stem_paths_by_clip,
                self.speech_stem_hashes_by_clip,
                self.original_audio_paths_by_clip,
                self.original_audio_hashes_by_clip,
            )
        ):
            raise ValueError("stem DiariZen Audio maps differ from usable clips")
        if set(self.unverified_clip_uids) - usable or (
            self.unverified_clip_uids and not self.unverified_consumption_authorized
        ):
            raise ValueError("stem DiariZen unverified provenance is inconsistent")
        return self


def _load_stem_diarization_clip_results(
    root: Path,
    *,
    expected_clip_uids: Sequence[str],
) -> list[DiarizationClipResult]:
    results = [
        DiarizationClipResult.model_validate(item)
        for item in _read_jsonl(root / "clip_results.jsonl")
    ]
    by_clip = {item.target_clip_uid: item for item in results}
    if len(by_clip) != len(results) or set(by_clip) != set(expected_clip_uids):
        raise ValueError("stem DiariZen clip results differ from attempted clips")
    ordered = [by_clip[item] for item in expected_clip_uids]
    if any(
        item.source_sample_rate_hz != STEM_SAMPLE_RATE_HZ
        or item.source_channels != STEM_CHANNELS
        or (item.status == "ready") != (item.raw_segment_count > 0)
        for item in ordered
    ):
        raise ValueError("stem DiariZen clip result status is inconsistent")
    return ordered


def validate_stem_diarization_lineage(
    root: Path,
    *,
    expected_shadow_root: Path | None = None,
) -> tuple[
    StemDiarizationShadowProvenance,
    SAMAudioStemInventory,
    list[SAMAudioStemRecord],
]:
    diarization = root.expanduser().resolve(strict=True)
    provenance = StemDiarizationShadowProvenance.model_validate_json(
        (diarization / "stem_provenance.json").read_text(encoding="utf-8")
    )
    separation = Path(provenance.source_stem_root).expanduser().resolve(strict=True)
    if expected_shadow_root is not None:
        expected = expected_shadow_root.expanduser().resolve(strict=True)
        require_shadow_output_path(shadow_root=expected, output_path=diarization)
        if separation != expected / SAM_AUDIO_SEPARATION_STAGE_NAME:
            raise ValueError("stem DiariZen source root differs from selected shadow run")
    inventory, records, _ = load_stem_shadow(separation)
    if sha256_file(separation / "records.jsonl") != provenance.source_stem_records_sha256:
        raise ValueError("stem DiariZen source separation records changed")
    if (
        inventory.inventory_fingerprint
        != provenance.source_stem_inventory_fingerprint
        or inventory.clip_uids != provenance.clip_uids
    ):
        raise ValueError("stem DiariZen source inventory changed")

    selected = selected_stem_records(
        records,
        route=provenance.route,
        allow_unverified=True,
    )
    selected_by_clip = {item.clip_uid: item for item in selected}
    attempted = [
        clip_uid for clip_uid in inventory.clip_uids if clip_uid in selected_by_clip
    ]
    clip_results = _load_stem_diarization_clip_results(
        diarization,
        expected_clip_uids=attempted,
    )
    ready = [item.target_clip_uid for item in clip_results if item.status == "ready"]
    empty = [item.target_clip_uid for item in clip_results if item.status == "empty"]
    failed = [item.target_clip_uid for item in clip_results if item.status == "failed"]
    usable = [item for item in attempted if item not in set(failed)]
    failures = [
        _diarization_failure(item, route=provenance.route)
        for item in clip_results
        if item.status == "failed"
    ]
    skips = separation_skips(
        inventory=inventory,
        records=records,
        route=provenance.route,
    )
    unverified = [
        clip_uid
        for clip_uid in usable
        if selected_by_clip[clip_uid].separation_state == "unverified"
    ]
    if (
        provenance.target_count != len(attempted)
        or ready != provenance.ready_clip_uids
        or empty != provenance.empty_clip_uids
        or failed != provenance.failed_clip_uids
        or usable != provenance.usable_clip_uids
        or skips != provenance.skipped_clips
        or failures != provenance.diarization_failed_clips
        or unverified != provenance.unverified_clip_uids
    ):
        raise ValueError("stem DiariZen selection differs from current separation")

    expected_paths = {
        key: selected_by_clip[key].stem("speech").canonical_stem_path
        for key in sorted(usable)
    }
    expected_hashes = {
        key: selected_by_clip[key].stem("speech").canonical_stem_sha256
        for key in sorted(usable)
    }
    expected_original_paths = {
        key: selected_by_clip[key].stem("speech").source_audio_path
        for key in sorted(usable)
    }
    expected_original_hashes = {
        key: selected_by_clip[key].stem("speech").source_audio_sha256
        for key in sorted(usable)
    }
    if (
        provenance.speech_stem_paths_by_clip != expected_paths
        or provenance.speech_stem_hashes_by_clip != expected_hashes
        or provenance.original_audio_paths_by_clip != expected_original_paths
        or provenance.original_audio_hashes_by_clip != expected_original_hashes
    ):
        raise ValueError("stem DiariZen Audio provenance differs from separation")
    for clip_uid in usable:
        speech = selected_by_clip[clip_uid].stem("speech")
        if sha256_file(Path(speech.canonical_stem_path)) != speech.canonical_stem_sha256:
            raise ValueError("stem DiariZen speech stem hash changed")
    raw_segments = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization / "raw_segments.jsonl")
    ]
    bound_segments = [
        BoundDiarizationSegment.model_validate(item)
        for item in _read_jsonl(diarization / "bound_segments.jsonl")
    ]
    if (
        {item.target_clip_uid for item in raw_segments} - set(usable)
        or {item.target_clip_uid for item in bound_segments} - set(usable)
    ):
        raise ValueError("failed stem DiariZen clips published segment evidence")
    failure_rows = [
        StemDiarizationClipFailure.model_validate(item)
        for item in _read_jsonl(diarization / "diarization_failed_clips.jsonl")
    ]
    if failure_rows != provenance.diarization_failed_clips:
        raise ValueError("stem DiariZen failure sidecar differs from provenance")
    for filename, expected_hash in (
        ("raw_segments.jsonl", provenance.raw_segments_sha256),
        ("bound_segments.jsonl", provenance.bound_segments_sha256),
        ("cluster_bindings.jsonl", provenance.cluster_bindings_sha256),
        ("clip_results.jsonl", provenance.clip_results_sha256),
    ):
        if sha256_file(diarization / filename) != expected_hash:
            raise ValueError(f"stem DiariZen {filename} changed")
    return provenance, inventory, records


def _canonical_by_clip(inventory: SAMAudioStemInventory) -> dict[str, CanonicalAudioClip]:
    records = [
        CanonicalAudioClip.model_validate(row)
        for row in _read_jsonl(Path(inventory.source_canonical_audio_manifest_path))
    ]
    result = {item.clip_uid: item for item in records}
    if len(result) != len(records):
        raise ValueError("canonical Audio manifest contains duplicate clips")
    return result


def build_stem_diarization_inventory(
    *,
    stem_inventory: SAMAudioStemInventory,
    stem_records: Sequence[SAMAudioStemRecord],
    production_diarization_inventory: DiarizationInventory,
    route: SAMRoute,
    allow_unverified: bool = False,
) -> DiarizationInventory:
    selected = selected_stem_records(
        stem_records,
        route=route,
        allow_unverified=allow_unverified,
    )
    stem_by_clip = {item.clip_uid: item for item in selected}
    jobs_by_clip = {item.clip_uid: item for item in stem_inventory.jobs}
    source_targets = {
        item.target_clip_uid: item for item in production_diarization_inventory.targets
    }
    ordered_ids = [item.clip_uid for item in stem_inventory.jobs if item.clip_uid in stem_by_clip]
    if set(ordered_ids) != set(stem_by_clip):
        raise ValueError("stem records differ from selected SAM Audio jobs")
    targets: list[DiarizationTargetClip] = []
    for clip_uid in ordered_ids:
        source_target = source_targets.get(clip_uid)
        if source_target is None:
            raise ValueError("speech stem lacks production diarization target provenance")
        speech = stem_by_clip[clip_uid].stem("speech")
        original = Path(source_target.source_audio_path).resolve(strict=True)
        if sha256_file(original) != speech.source_audio_sha256:
            raise ValueError("production DiariZen source differs from SAM source Audio")
        speech_path = Path(speech.canonical_stem_path).resolve(strict=True)
        if sha256_file(speech_path) != speech.canonical_stem_sha256:
            raise ValueError("canonical speech stem hash changed")
        values = source_target.model_dump(mode="python")
        values.update(
            source_audio_path=str(speech_path),
            source_audio_sha256=speech.canonical_stem_sha256,
            source_sample_rate_hz=speech.canonical_sample_rate_hz,
            source_channels=speech.canonical_channels,
            source_frame_count=min(
                speech.canonical_frame_count,
                jobs_by_clip[clip_uid].source_frame_count,
            ),
        )
        targets.append(DiarizationTargetClip.model_validate(values))
    if production_diarization_inventory.source_inventory_kind != "canonical_audio_manifest":
        raise ValueError("stem shadow requires canonical production DiariZen provenance")
    source_visual_hash = production_diarization_inventory.source_visual_inventory_sha256
    source_canonical_hash = (
        production_diarization_inventory.source_canonical_audio_manifest_sha256
    )
    assert source_visual_hash is not None and source_canonical_hash is not None
    fingerprint = _diarization_inventory_fingerprint(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
        source_inventory_kind="canonical_audio_manifest",
        source_visual_inventory_sha256=source_visual_hash,
        source_canonical_audio_manifest_sha256=source_canonical_hash,
    )
    return DiarizationInventory(
        mode="production",
        source_inventory_kind="canonical_audio_manifest",
        source_visual_production_root=(
            production_diarization_inventory.source_visual_production_root
        ),
        source_visual_inventory_path=(
            production_diarization_inventory.source_visual_inventory_path
        ),
        source_visual_inventory_sha256=source_visual_hash,
        source_canonical_audio_manifest_path=(
            production_diarization_inventory.source_canonical_audio_manifest_path
        ),
        source_canonical_audio_manifest_sha256=source_canonical_hash,
        inventory_fingerprint=fingerprint,
        source_target_count=len(targets),
        selected_target_count=len(targets),
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )


def _publish_shadow_readable(
    *,
    diarization_root: Path,
    canonical_by_clip: dict[str, CanonicalAudioClip],
    inventory: DiarizationInventory,
    usable_clip_uids: Sequence[str],
) -> JEAReadableDiarizationSummary:
    raw = [
        RawDiarizationSegment.model_validate(row)
        for row in _read_jsonl(diarization_root / "raw_segments.jsonl")
    ]
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read_jsonl(diarization_root / "bound_segments.jsonl")
    ]
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    if len(bound_by_key) != len(bound) or set(bound_by_key) != {
        (item.target_clip_uid, item.segment_id) for item in raw
    }:
        raise ValueError("stem DiariZen raw and bound inventories differ")
    usable = set(usable_clip_uids)
    if not {item.target_clip_uid for item in raw}.issubset(usable):
        raise ValueError("failed stem DiariZen clip published readable segments")
    targets: list[JEAReadableDiarizationTarget] = []
    for target in inventory.targets:
        if target.target_clip_uid not in usable:
            continue
        source = canonical_by_clip[target.target_clip_uid]
        targets.append(
            JEAReadableDiarizationTarget(
                clip_uid=source.clip_uid,
                clip_display_path=source.clip_display_path,
                media_collection_relpath=source.media_collection_relpath,
                media_collection_name=source.media_collection_name,
                episode_name=source.episode_name,
                clip_name=source.clip_name,
                shard_id=source.shard_id,
                source_audio_path=target.source_audio_path,
                target_video_path=target.target_video_path,
                target_audio_binding_path=target.target_audio_binding_path,
                target_audio_binding_sha256=target.target_audio_binding_sha256,
            )
        )
    segments: list[JEAReadableDiarizationSegment] = []
    for source in raw:
        canonical = canonical_by_clip[source.target_clip_uid]
        mapped = bound_by_key[(source.target_clip_uid, source.segment_id)]
        segments.append(
            JEAReadableDiarizationSegment(
                clip_uid=canonical.clip_uid,
                clip_display_path=canonical.clip_display_path,
                media_collection_relpath=canonical.media_collection_relpath,
                media_collection_name=canonical.media_collection_name,
                episode_name=canonical.episode_name,
                clip_name=canonical.clip_name,
                shard_id=canonical.shard_id,
                segment_id=source.segment_id,
                speaker_cluster_id=source.speaker_cluster_id,
                entity_id=mapped.entity_id,
                entity_occurrence_id=mapped.entity_occurrence_id,
                source_audio_path=source.source_audio_path,
                source_start_sample=source.source_start_sample,
                source_end_sample=source.source_end_sample,
                start_time=source.start_time,
                end_time=source.end_time,
            )
        )
    clip_order = {
        target.target_clip_uid: index for index, target in enumerate(inventory.targets)
    }
    segments.sort(
        key=lambda item: (
            clip_order[item.clip_uid],
            item.source_start_sample,
            item.segment_id,
        )
    )
    _write_jsonl(diarization_root / "readable_targets.jsonl", targets)
    _write_jsonl(diarization_root / "readable_segments.jsonl", segments)
    summary = JEAReadableDiarizationSummary(
        target_count=len(targets),
        segment_count=len(segments),
        media_collection_count=len({item.media_collection_relpath for item in targets}),
    )
    _write_json(diarization_root / "readable_summary.json", summary)
    return summary


def run_stem_diarization_shadow(
    *,
    stem_root: Path,
    production_diarization_root: Path,
    backend: DiarizationBackend,
    route: SAMRoute,
    output_root: Path,
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> StemDiarizationShadowProvenance:
    stem_inventory, stem_records, _ = load_stem_shadow(stem_root)
    production_inventory_path = (
        production_diarization_root.expanduser().resolve(strict=True) / "inventory.json"
    )
    production_inventory = DiarizationInventory.model_validate_json(
        production_inventory_path.read_text(encoding="utf-8")
    )
    inventory = build_stem_diarization_inventory(
        stem_inventory=stem_inventory,
        stem_records=stem_records,
        production_diarization_inventory=production_inventory,
        route=route,
        allow_unverified=allow_unverified,
    )
    destination = output_root.expanduser().resolve(strict=False)
    outer = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    selected = selected_stem_records(
        stem_records,
        route=route,
        allow_unverified=allow_unverified,
    )
    by_clip = {item.clip_uid: item for item in selected}
    skips = separation_skips(
        inventory=stem_inventory,
        records=stem_records,
        route=route,
    )
    attempted_clip_uids = [
        clip_uid for clip_uid in stem_inventory.clip_uids if clip_uid in by_clip
    ]
    records_path = stem_root.expanduser().resolve(strict=True) / "records.jsonl"
    try:
        outer.mkdir(parents=True)
        stage = outer / "stage"
        run_diarization_binding_pilot(
            inventory=inventory,
            output_root=stage,
            backend=backend,
        )
        clip_results = _load_stem_diarization_clip_results(
            stage,
            expected_clip_uids=attempted_clip_uids,
        )
        ready_clip_uids = [
            item.target_clip_uid for item in clip_results if item.status == "ready"
        ]
        empty_clip_uids = [
            item.target_clip_uid for item in clip_results if item.status == "empty"
        ]
        failed_clip_uids = [
            item.target_clip_uid for item in clip_results if item.status == "failed"
        ]
        failed = set(failed_clip_uids)
        usable_clip_uids = [
            item for item in attempted_clip_uids if item not in failed
        ]
        diarization_failures = [
            _diarization_failure(item, route=route)
            for item in clip_results
            if item.status == "failed"
        ]
        unverified_clip_uids = [
            clip_uid
            for clip_uid in usable_clip_uids
            if by_clip[clip_uid].separation_state == "unverified"
        ]
        _publish_shadow_readable(
            diarization_root=stage,
            canonical_by_clip=_canonical_by_clip(stem_inventory),
            inventory=inventory,
            usable_clip_uids=usable_clip_uids,
        )
        provenance = StemDiarizationShadowProvenance(
            source_stem_root=str(stem_root.expanduser().resolve(strict=True)),
            source_stem_inventory_fingerprint=stem_inventory.inventory_fingerprint,
            source_stem_records_sha256=sha256_file(records_path),
            raw_segments_sha256=sha256_file(stage / "raw_segments.jsonl"),
            bound_segments_sha256=sha256_file(stage / "bound_segments.jsonl"),
            cluster_bindings_sha256=sha256_file(stage / "cluster_bindings.jsonl"),
            clip_results_sha256=sha256_file(stage / "clip_results.jsonl"),
            route=route,
            target_count=len(inventory.targets),
            clip_uids=stem_inventory.clip_uids,
            ready_clip_uids=ready_clip_uids,
            empty_clip_uids=empty_clip_uids,
            failed_clip_uids=failed_clip_uids,
            usable_clip_uids=usable_clip_uids,
            skipped_clips=skips,
            diarization_failed_clips=diarization_failures,
            unverified_clip_uids=unverified_clip_uids,
            unverified_consumption_authorized=allow_unverified,
            speech_stem_paths_by_clip={
                key: by_clip[key].stem("speech").canonical_stem_path
                for key in sorted(usable_clip_uids)
            },
            speech_stem_hashes_by_clip={
                key: by_clip[key].stem("speech").canonical_stem_sha256
                for key in sorted(usable_clip_uids)
            },
            original_audio_paths_by_clip={
                key: by_clip[key].stem("speech").source_audio_path
                for key in sorted(usable_clip_uids)
            },
            original_audio_hashes_by_clip={
                key: by_clip[key].stem("speech").source_audio_sha256
                for key in sorted(usable_clip_uids)
            },
        )
        _write_json(stage / "stem_provenance.json", provenance)
        _write_jsonl(stage / "skipped_clips.jsonl", skips)
        _write_jsonl(
            stage / "diarization_failed_clips.jsonl",
            diarization_failures,
        )
        stage.replace(outer / "published")
        published = outer / "published"
        _publish_directory(published, destination, overwrite=overwrite)
        outer.rmdir()
        return provenance
    except Exception:
        if outer.exists():
            shutil.rmtree(outer)
        raise


class StemASRShadowProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.stem_asr_shadow.4"] = STEM_ASR_SHADOW_VERSION
    asr_source_kind: Literal["sam_audio_speech_stem_segments"] = (
        "sam_audio_speech_stem_segments"
    )
    source_stem_diarization_root: str
    source_stem_diarization_provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    route: SAMRoute
    source_clip_uids: list[str] = Field(min_length=1)
    clip_uids: list[str]
    source_diarization_ready_clip_uids: list[str]
    source_diarization_empty_clip_uids: list[str]
    skipped_clips: list[StemShadowClipSkip]
    diarization_failed_clips: list[StemDiarizationClipFailure]
    unverified_clip_uids: list[str]
    unverified_consumption_authorized: bool
    segment_count: int = Field(ge=0)
    segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_asr_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_clip_provenance(self) -> StemASRShadowProvenance:
        excluded = {item.clip_uid for item in self.skipped_clips}.union(
            item.clip_uid for item in self.diarization_failed_clips
        )
        if (
            len(self.source_clip_uids) != len(set(self.source_clip_uids))
            or [
                item
                for item in self.source_clip_uids
                if item in set(self.clip_uids)
            ]
            != self.clip_uids
            or excluded != set(self.source_clip_uids) - set(self.clip_uids)
            or self.clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item
                in set(self.source_diarization_ready_clip_uids).union(
                    self.source_diarization_empty_clip_uids
                )
            ]
            or self.source_diarization_ready_clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item in set(self.source_diarization_ready_clip_uids)
            ]
            or self.source_diarization_empty_clip_uids
            != [
                item
                for item in self.source_clip_uids
                if item in set(self.source_diarization_empty_clip_uids)
            ]
            or set(self.source_diarization_ready_clip_uids).intersection(
                self.source_diarization_empty_clip_uids
            )
            or [item.clip_uid for item in self.diarization_failed_clips]
            != [
                item
                for item in self.source_clip_uids
                if item
                in {failure.clip_uid for failure in self.diarization_failed_clips}
            ]
            or any(item.route != self.route for item in self.diarization_failed_clips)
            or set(self.unverified_clip_uids) - set(self.clip_uids)
            or (
                self.unverified_clip_uids
                and not self.unverified_consumption_authorized
            )
        ):
            raise ValueError("stem ASR ordered clip provenance is inconsistent")
        return self


def validate_stem_asr_lineage(
    root: Path,
    *,
    expected_shadow_root: Path | None = None,
) -> tuple[StemASRShadowProvenance, StemDiarizationShadowProvenance]:
    asr_root = root.expanduser().resolve(strict=True)
    provenance = StemASRShadowProvenance.model_validate_json(
        (asr_root / "stem_provenance.json").read_text(encoding="utf-8")
    )
    diarization = Path(provenance.source_stem_diarization_root).expanduser().resolve(
        strict=True
    )
    if expected_shadow_root is not None:
        expected = expected_shadow_root.expanduser().resolve(strict=True)
        require_shadow_output_path(shadow_root=expected, output_path=asr_root)
        if diarization != expected / "diarization":
            raise ValueError("stem ASR source root differs from selected shadow run")
    source_path = diarization / "stem_provenance.json"
    if sha256_file(source_path) != provenance.source_stem_diarization_provenance_sha256:
        raise ValueError("stem ASR source DiariZen provenance changed")
    source, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=expected_shadow_root,
    )
    if (
        provenance.source_stem_inventory_fingerprint
        != source.source_stem_inventory_fingerprint
        or provenance.route != source.route
        or provenance.source_clip_uids != source.clip_uids
        or provenance.clip_uids != source.usable_clip_uids
        or provenance.source_diarization_ready_clip_uids
        != source.ready_clip_uids
        or provenance.source_diarization_empty_clip_uids
        != source.empty_clip_uids
        or provenance.skipped_clips != source.skipped_clips
        or provenance.diarization_failed_clips
        != source.diarization_failed_clips
        or provenance.unverified_clip_uids != source.unverified_clip_uids
    ):
        raise ValueError("stem ASR provenance differs from current DiariZen stage")
    segments = [
        Qwen3ASRSegment.model_validate(row)
        for row in _read_jsonl(asr_root / "segments.jsonl")
    ]
    if len(segments) != provenance.segment_count or (
        {item.clip_uid for item in segments} - set(provenance.clip_uids)
    ):
        raise ValueError("stem ASR segment inventory differs from provenance")
    if sha256_file(asr_root / "segments.jsonl") != provenance.segments_sha256:
        raise ValueError("stem ASR segments changed")
    return provenance, source


def run_stem_qwen3_asr_shadow(
    *,
    stem_diarization_root: Path,
    source_visual_production_root: str,
    backend: Qwen3ASRBackend,
    output_root: Path,
    case_manifest: MimoCaseManifest | None = None,
    segment_audio_loader: SegmentAudioLoader | None = None,
    ffmpeg: str = "ffmpeg",
    route: SAMRoute | None = None,
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> tuple[Qwen3ASRSummary, StemASRShadowProvenance]:
    diarization = stem_diarization_root.expanduser().resolve(strict=True)
    source_provenance_path = diarization / "stem_provenance.json"
    source_provenance, _, _ = validate_stem_diarization_lineage(diarization)
    if route is not None and source_provenance.route != route:
        raise ValueError("stem ASR route differs from stem diarization")
    target_ids = source_provenance.usable_clip_uids
    if case_manifest is not None and case_manifest.clip_uids != source_provenance.clip_uids:
        raise ValueError("stem ASR case manifest differs from stem diarization order")
    if source_provenance.unverified_clip_uids and not allow_unverified:
        raise ValueError(
            "unverified SAM Audio stems require explicit allow_unverified opt-in"
        )
    destination = output_root.expanduser().resolve(strict=False)
    outer = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        outer.mkdir(parents=True)
        stage = outer / "stage"
        summary = run_qwen3_asr(
            diarization_root=diarization,
            source_visual_production_root=source_visual_production_root,
            output_root=stage,
            backend=backend,
            segment_audio_loader=segment_audio_loader,
            ffmpeg=ffmpeg,
        )
        provenance = StemASRShadowProvenance(
            source_stem_diarization_root=str(diarization),
            source_stem_diarization_provenance_sha256=sha256_file(
                source_provenance_path
            ),
            source_stem_inventory_fingerprint=(
                source_provenance.source_stem_inventory_fingerprint
            ),
            route=source_provenance.route,
            source_clip_uids=source_provenance.clip_uids,
            clip_uids=target_ids,
            source_diarization_ready_clip_uids=(
                source_provenance.ready_clip_uids
            ),
            source_diarization_empty_clip_uids=(
                source_provenance.empty_clip_uids
            ),
            skipped_clips=source_provenance.skipped_clips,
            diarization_failed_clips=(
                source_provenance.diarization_failed_clips
            ),
            unverified_clip_uids=source_provenance.unverified_clip_uids,
            unverified_consumption_authorized=allow_unverified,
            segment_count=summary.segment_count,
            segments_sha256=sha256_file(stage / "segments.jsonl"),
        )
        _write_json(stage / "stem_provenance.json", provenance)
        _write_jsonl(stage / "skipped_clips.jsonl", provenance.skipped_clips)
        _write_jsonl(
            stage / "diarization_failed_clips.jsonl",
            provenance.diarization_failed_clips,
        )
        stage.replace(outer / "published")
        _publish_directory(outer / "published", destination, overwrite=overwrite)
        outer.rmdir()
        return summary, provenance
    except Exception:
        if outer.exists():
            shutil.rmtree(outer)
        raise


__all__ = [
    "STEM_ALIGNMENT_TOLERANCE_SECONDS",
    "STEM_ALIGNMENT_WARNING_SECONDS",
    "STEM_CANONICAL_CHANNEL_POLICY",
    "STEM_TIMELINE_POLICY",
    "FFmpegStemCanonicalizer",
    "OfficialSAMAudioBackend",
    "SAMAudioBackend",
    "SAMAudioModelConfiguration",
    "SAMAudioSeparationResult",
    "SAMAudioStemInventory",
    "SAMAudioStemRecord",
    "SAMAudioStemSummary",
    "SAMRoute",
    "StemArtifact",
    "StemDiarizationClipFailure",
    "StemDiarizationShadowProvenance",
    "StemNativeReference",
    "StemReferenceManifest",
    "StemReferenceRequest",
    "StemShadowClipSkip",
    "StemType",
    "build_sam_audio_stem_inventory",
    "build_shadow_primary_voice_reference_requests",
    "build_stem_diarization_inventory",
    "export_stem_native_references",
    "load_stem_shadow",
    "require_shadow_output_path",
    "run_sam_audio_stem_shadow",
    "run_stem_diarization_shadow",
    "run_stem_qwen3_asr_shadow",
    "sam_audio_configuration",
    "selected_stem_records",
    "separation_skips",
    "stem_separation_root",
    "stem_shadow_root",
    "validate_stem_asr_lineage",
    "validate_stem_diarization_lineage",
]
