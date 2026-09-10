"""Independent AuK candidate stage; never changes the SAM downstream source."""

from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

import numpy as np
import soundfile as sf
from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip, jea_production_paths
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    _publish_directory,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    sha256_file,
    stem_shadow_root,
)
from r2v_data_v2.h3.schemas import SchemaModel

FIXED_INSTRUCTION = (
    "Keep all human voices, including speech and singing, and remove everything else."
)
AUK_STAGE = "auk_speech_v1"
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ClipID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _signed(model, values: dict, field: str):
    payload = {
        name: item.get_default(call_default_factory=True)
        for name, item in model.model_fields.items()
        if not item.is_required()
    }
    payload.update(values)
    payload.pop(field, None)
    return model(**payload, **{field: fingerprint(payload)})


def _check_fingerprint(model: SchemaModel, field: str) -> None:
    if getattr(model, field) != fingerprint(
        model.model_dump(mode="json", exclude={field})
    ):
        raise ValueError(f"AuK {field} differs")


def _file(path: Path) -> Path:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"AuK requires a non-empty local file: {path}")
    return path


class AukConfiguration(SchemaModel):
    backend_version: Literal["r2v.h3.auk_speech_backend.1"] = (
        "r2v.h3.auk_speech_backend.1"
    )
    model_identifier: Literal["tencent/AuK"] = "tencent/AuK"
    python_path: str
    code_root: str
    config_path: str
    checkpoint_path: str
    vae_path: str
    qwen_path: str
    dependency_files: dict[str, Hash]
    worker_sha256: Hash
    instruction: Literal[FIXED_INSTRUCTION] = FIXED_INSTRUCTION
    device: str = Field(min_length=1)
    dtype: Literal["bf16"] = "bf16"
    nfe: Literal[32] = 32
    cfg_strength: Literal[2.0] = 2.0
    sway_sampling_coef: Literal[-1.0] = -1.0
    seed: Literal[0] = 0
    offline: Literal[True] = True
    configuration_fingerprint: Hash

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        _check_fingerprint(self, "configuration_fingerprint")
        for name in (self.config_path, self.checkpoint_path, self.vae_path):
            if name not in self.dependency_files:
                raise ValueError("AuK model file provenance is incomplete")
        return self


def worker_path() -> Path:
    return Path(__file__).resolve().parents[2] / "tools/run_h3_auk_speech_worker.py"


def auk_configuration(
    *,
    python_path: Path,
    code_root: Path,
    checkpoint: Path,
    qwen_path: Path,
    device: str = "cuda:0",
    dtype: str = "bf16",
) -> AukConfiguration:
    # Do not resolve a venv executable symlink into the base interpreter.
    python_path = python_path.expanduser().absolute()
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError("AuK requires an explicit local Python executable")
    code_root = code_root.expanduser().resolve(strict=True)
    qwen_path = qwen_path.expanduser().resolve(strict=True)
    if not code_root.is_dir() or not qwen_path.is_dir():
        raise ValueError("AuK code and Qwen roots must be local directories")
    # Keep the caller's checkpoint directory: sibling config/VAE may accompany
    # a symlink to a local checkpoint cache.
    checkpoint = checkpoint.expanduser().absolute()
    _file(checkpoint)
    if checkpoint.name != "auk_base.safetensors":
        raise ValueError("AuK v1 requires the Base auk_base.safetensors checkpoint")
    config = _file(checkpoint.with_name("config.yaml"))
    vae = _file(checkpoint.with_name("vae.safetensors"))
    _file(code_root / "src/auk/infer/infer_auk.py")
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
        _file(qwen_path / name)
    if not (qwen_path / "tokenizer.json").is_file():
        for name in ("vocab.json", "merges.txt"):
            _file(qwen_path / name)
    index = qwen_path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Qwen weight index is empty")
        for name in set(weight_map.values()):
            shard = _file(qwen_path / name)
            if not shard.is_relative_to(qwen_path):
                raise ValueError("Qwen shard escapes local dependency root")
    else:
        _file(qwen_path / "model.safetensors")
    dependencies = [checkpoint, config, vae, _file(code_root / "pyproject.toml")]
    dependencies += sorted((code_root / "src/auk").rglob("*.py"))
    dependencies += sorted(
        p
        for p in qwen_path.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".txt", ".safetensors", ".model", ".jinja"}
        and ".cache" not in p.relative_to(qwen_path).parts
    )
    return _signed(
        AukConfiguration,
        {
            "python_path": str(python_path),
            "code_root": str(code_root),
            "config_path": str(config),
            "checkpoint_path": str(checkpoint),
            "vae_path": str(vae),
            "qwen_path": str(qwen_path),
            "dependency_files": {
                str(p): sha256_file(_file(p)) for p in sorted(set(dependencies))
            },
            "worker_sha256": sha256_file(worker_path()),
            "device": device,
            "dtype": dtype,
        },
        "configuration_fingerprint",
    )


class AukJob(SchemaModel):
    clip_uid: ClipID
    clip_display_path: str
    target_video_path: str
    target_video_sha256: Hash
    source_audio_path: str
    source_audio_sha256: Hash
    source_sample_rate_hz: Literal[32000] = 32000
    source_channels: Literal[2] = 2
    source_frame_count: int = Field(gt=0)
    source_duration_seconds: PositiveSeconds

    @model_validator(mode="after")
    def validate_duration(self) -> Self:
        if abs(self.source_duration_seconds - self.source_frame_count / 32000) > 1e-12:
            raise ValueError("AuK source duration differs from frame extent")
        return self


class AukInventory(SchemaModel):
    schema_version: Literal["r2v.h3.auk_speech_inventory.1"] = (
        "r2v.h3.auk_speech_inventory.1"
    )
    audio_production_root: str
    shadow_run_id: str
    source_canonical_audio_manifest_path: str
    source_canonical_audio_manifest_sha256: Hash
    source_case_manifest_path: str
    source_case_manifest_sha256: Hash
    model_configuration: AukConfiguration
    clip_uids: list[ClipID] = Field(min_length=1)
    jobs: list[AukJob] = Field(min_length=1)
    inventory_fingerprint: Hash

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.clip_uids != [j.clip_uid for j in self.jobs] or len(
            set(self.clip_uids)
        ) != len(self.clip_uids):
            raise ValueError("AuK ordered inventory must contain unique clips")
        _check_fingerprint(self, "inventory_fingerprint")
        return self


def auk_stage_root(audio_production_root: Path, shadow_run_id: str) -> Path:
    root = stem_shadow_root(audio_production_root, shadow_run_id)
    path = root / AUK_STAGE
    if path.resolve() != path or path.is_symlink():
        raise ValueError("AuK stage must not redirect outside its owned location")
    return path


def check_source(job: AukJob) -> None:
    for path, expected in (
        (job.source_audio_path, job.source_audio_sha256),
        (job.target_video_path, job.target_video_sha256),
    ):
        if sha256_file(_file(Path(path))) != expected:
            raise ValueError("AuK source hash differs")
    info = sf.info(job.source_audio_path)
    if (info.format, info.samplerate, info.channels, info.frames) != (
        "FLAC",
        32000,
        2,
        job.source_frame_count,
    ):
        raise ValueError("AuK canonical source must match its 32k stereo FLAC extent")


def build_auk_inventory(
    *,
    audio_production_root: Path,
    shadow_run_id: str,
    case_manifest_path: Path,
    configuration: AukConfiguration,
) -> AukInventory:
    paths = jea_production_paths(audio_production_root)
    auk_stage_root(paths.root, shadow_run_id)
    manifest = _file(paths.audio / "canonical_clips.jsonl")
    clips = [CanonicalAudioClip.model_validate(row) for row in _read_jsonl(manifest)]
    by_uid = {c.clip_uid: c for c in clips}
    if len(by_uid) != len(clips):
        raise ValueError("canonical Audio contains duplicate clips")
    case_path = _file(case_manifest_path)
    case = MimoCaseManifest.model_validate_json(case_path.read_text())
    if set(case.clip_uids) - by_uid.keys():
        raise ValueError("AuK case manifest contains unknown clips")
    jobs = []
    for uid in case.clip_uids:
        c = by_uid[uid]
        job = AukJob(
            clip_uid=uid,
            clip_display_path=c.clip_display_path,
            target_video_path=c.target_video_path,
            target_video_sha256=c.target_video_sha256,
            source_audio_path=c.target_full_audio_path,
            source_audio_sha256=c.target_full_audio_sha256,
            source_frame_count=c.frame_count,
            source_duration_seconds=c.target_duration_seconds,
        )
        check_source(job)
        jobs.append(job)
    return _signed(
        AukInventory,
        {
            "audio_production_root": str(paths.root),
            "shadow_run_id": shadow_run_id,
            "source_canonical_audio_manifest_path": str(manifest),
            "source_canonical_audio_manifest_sha256": sha256_file(manifest),
            "source_case_manifest_path": str(case_path),
            "source_case_manifest_sha256": sha256_file(case_path),
            "model_configuration": configuration.model_dump(mode="json"),
            "clip_uids": case.clip_uids,
            "jobs": [j.model_dump(mode="json") for j in jobs],
        },
        "inventory_fingerprint",
    )


class AukAudioArtifact(SchemaModel):
    path: str
    sha256: Hash
    sample_rate_hz: int = Field(gt=0)
    channels: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    subtype: str
    format: Literal["WAV"] = "WAV"


def audio_artifact(path: Path, published_path: Path) -> AukAudioArtifact:
    info = sf.info(path)
    return AukAudioArtifact(
        path=str(published_path),
        sha256=sha256_file(path),
        sample_rate_hz=info.samplerate,
        channels=info.channels,
        frame_count=info.frames,
        subtype=info.subtype,
        format=info.format,
    )


class AukRecord(SchemaModel):
    schema_version: Literal["r2v.h3.auk_speech_record.1"] = "r2v.h3.auk_speech_record.1"
    clip_uid: ClipID
    inventory_fingerprint: Hash
    configuration_fingerprint: Hash
    source_audio_sha256: Hash
    instruction: Literal[FIXED_INSTRUCTION] = FIXED_INSTRUCTION
    status: Literal["ready", "failed"]
    failure_reason: str | None = None
    raw: AukAudioArtifact | None = None
    canonical: AukAudioArtifact | None = None
    raw_duration_delta_seconds: float | None = Field(default=None, allow_inf_nan=False)
    canonical_adjustment_samples: int | None = None
    expected_frame_count: int = Field(gt=0)
    alignment_tolerance_seconds: Literal[0.10] = 0.10
    alignment_offset_seconds: Literal[0] = 0
    timeline_policy: Literal["same_extent_bounded_tail_trim_pad_v1"] = (
        "same_extent_bounded_tail_trim_pad_v1"
    )
    channel_policy: Literal["ffmpeg_stereo_v1"] = "ffmpeg_stereo_v1"
    model_runtime_seconds: float = Field(ge=0, allow_inf_nan=False)
    model_call_count: int = Field(ge=0, le=1)
    record_fingerprint: Hash

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _check_fingerprint(self, "record_fingerprint")
        if self.status == "ready":
            if self.failure_reason or self.raw is None or self.canonical is None:
                raise ValueError("ready AuK record requires both waveforms")
            if (
                self.raw_duration_delta_seconds is None
                or abs(self.raw_duration_delta_seconds) > 0.10 + 1e-12
            ):
                raise ValueError("AuK raw duration exceeds tolerance")
            if (
                self.canonical_adjustment_samples is None
                or abs(self.canonical_adjustment_samples) > 3201
            ):
                raise ValueError("AuK canonical tail correction exceeds tolerance")
            if (
                self.canonical.sample_rate_hz,
                self.canonical.channels,
                self.canonical.subtype,
                self.canonical.frame_count,
            ) != (32000, 2, "PCM_16", self.expected_frame_count):
                raise ValueError("AuK canonical waveform contract differs")
        elif not self.failure_reason or self.canonical is not None:
            raise ValueError(
                "failed AuK record requires reason and no canonical output"
            )
        return self


class AukSummary(SchemaModel):
    schema_version: Literal["r2v.h3.auk_speech_summary.1"] = (
        "r2v.h3.auk_speech_summary.1"
    )
    inventory_fingerprint: Hash
    clip_uids: list[ClipID]
    record_count: int = Field(ge=0)
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    production_artifacts_modified: Literal[False] = False


class AukBackend(Protocol):
    configuration: AukConfiguration

    def generate(self, job: AukJob, raw_path: Path) -> dict: ...


class PersistentAukBackend:
    def __init__(
        self, configuration: AukConfiguration, *, timeout_seconds: float = 1800
    ) -> None:
        self.configuration = configuration
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None

    def _read(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        if not select.select([self.process.stdout], [], [], self.timeout_seconds)[0]:
            self.close()
            raise TimeoutError("AuK worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("AuK worker exited without a response")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("AuK worker response must be an object")
        return payload

    def _send(self, payload: dict) -> None:
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

    def __enter__(self) -> Self:
        if sha256_file(worker_path()) != self.configuration.worker_sha256:
            raise ValueError(
                "AuK worker implementation changed after inventory construction"
            )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.process = subprocess.Popen(
            [self.configuration.python_path, str(worker_path())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
        )
        try:
            self._send(self.configuration.model_dump(mode="json"))
            response = self._read()
            if (
                response.get("request_id") != "startup"
                or response.get("status") != "ready"
            ):
                raise RuntimeError(f"AuK worker startup failed: {response}")
        except BaseException:
            self.close()
            raise
        return self

    def generate(self, job: AukJob, raw_path: Path) -> dict:
        self._send(
            {
                "request_id": job.clip_uid,
                "operation": "generate",
                "raw_output_path": str(raw_path),
                "source_audio_path": job.source_audio_path,
                "source_audio_sha256": job.source_audio_sha256,
                "source_duration_seconds": job.source_duration_seconds,
            }
        )
        response = self._read()
        if response.get("request_id") != job.clip_uid or response.get("status") not in {
            "ok",
            "failed",
        }:
            raise ValueError("AuK worker response identity/status differs")
        return response

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def __exit__(self, *args) -> None:
        self.close()


def canonicalize_speech(
    raw: Path, output: Path, expected_frames: int, *, ffmpeg: str
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(raw),
            "-map",
            "0:a:0",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    pcm, rate = sf.read(output, dtype="int16", always_2d=True)
    adjustment = expected_frames - len(pcm)
    if rate != 32000 or pcm.shape[1] != 2 or abs(adjustment) > 3201:
        raise ValueError("AuK resampled extent exceeds bounded tail correction")
    if adjustment > 0:
        pcm = np.pad(pcm, ((0, adjustment), (0, 0)))
    elif adjustment < 0:
        pcm = pcm[:expected_frames]
    sf.write(output, pcm, 32000, format="WAV", subtype="PCM_16")
    return adjustment


def preflight(inventory: AukInventory, *, overwrite: bool) -> Path:
    destination = auk_stage_root(
        Path(inventory.audio_production_root), inventory.shadow_run_id
    )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        previous, _, _ = load_auk_shadow(destination)
        if (previous.audio_production_root, previous.shadow_run_id) != (
            inventory.audio_production_root,
            inventory.shadow_run_id,
        ):
            raise ValueError("AuK destination ownership differs")
    for path, expected in (
        (
            inventory.source_canonical_audio_manifest_path,
            inventory.source_canonical_audio_manifest_sha256,
        ),
        (inventory.source_case_manifest_path, inventory.source_case_manifest_sha256),
    ):
        if sha256_file(_file(Path(path))) != expected:
            raise ValueError("AuK source manifest hash differs")
    source_paths = [Path(j.source_audio_path) for j in inventory.jobs]
    source_paths += [Path(j.target_video_path) for j in inventory.jobs]
    source_paths += [Path(p) for p in inventory.model_configuration.dependency_files]
    source_paths += [
        Path(inventory.source_case_manifest_path),
        Path(inventory.source_canonical_audio_manifest_path),
    ]
    if any(p.resolve().is_relative_to(destination) for p in source_paths):
        raise ValueError("AuK output contains a source dependency")
    for job in inventory.jobs:
        check_source(job)
    return destination


def _summary(inventory: AukInventory, records: list[AukRecord]) -> AukSummary:
    return AukSummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        clip_uids=inventory.clip_uids,
        record_count=len(records),
        ready_count=sum(r.status == "ready" for r in records),
        failed_count=sum(r.status == "failed" for r in records),
        model_call_count=sum(r.model_call_count for r in records),
    )


def run_auk_speech_shadow(
    *,
    inventory: AukInventory,
    backend: AukBackend,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> AukSummary:
    destination = preflight(inventory, overwrite=overwrite)
    if backend.configuration != inventory.model_configuration:
        raise ValueError("AuK backend configuration differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{AUK_STAGE}-", dir=destination.parent))
    try:
        records = []
        for job in inventory.jobs:
            raw_rel = Path("raw") / f"{job.clip_uid}.wav"
            canonical_rel = Path("canonical") / job.clip_uid / "speech.wav"
            raw, canonical = temporary / raw_rel, temporary / canonical_rel
            raw.parent.mkdir(exist_ok=True)
            values = {
                "clip_uid": job.clip_uid,
                "inventory_fingerprint": inventory.inventory_fingerprint,
                "configuration_fingerprint": inventory.model_configuration.configuration_fingerprint,
                "source_audio_sha256": job.source_audio_sha256,
                "status": "failed",
                "expected_frame_count": round(job.source_duration_seconds * 32000),
                "model_runtime_seconds": 0.0,
                "model_call_count": 0,
            }
            started = time.monotonic()
            try:
                check_source(job)
                values["model_call_count"] = 1
                result = backend.generate(job, raw)
                values["model_runtime_seconds"] = result["model_runtime_seconds"]
                if result["status"] != "ok":
                    raise RuntimeError(result["reason"])
                values["raw"] = audio_artifact(raw, destination / raw_rel).model_dump(
                    mode="json"
                )
                info = sf.info(raw)
                delta = info.frames / info.samplerate - job.source_duration_seconds
                values["raw_duration_delta_seconds"] = delta
                if abs(delta) > 0.10 + 1e-12:
                    raise ValueError("AuK raw duration exceeds 0.10 second tolerance")
                values["canonical_adjustment_samples"] = canonicalize_speech(
                    raw,
                    canonical,
                    values["expected_frame_count"],
                    ffmpeg=ffmpeg,
                )
                if sha256_file(raw) != values["raw"]["sha256"]:
                    raise ValueError(
                        "AuK native raw output changed during canonicalization"
                    )
                values["canonical"] = audio_artifact(
                    canonical, destination / canonical_rel
                ).model_dump(mode="json")
                values["status"] = "ready"
                record = _signed(AukRecord, values, "record_fingerprint")
            except Exception as exc:  # noqa: BLE001 - isolate one generated candidate
                values.update(
                    status="failed",
                    canonical=None,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
                if not values["model_runtime_seconds"]:
                    values["model_runtime_seconds"] = time.monotonic() - started
                canonical.unlink(missing_ok=True)
                record = _signed(AukRecord, values, "record_fingerprint")
            records.append(record)
        # Verify the frozen inputs again before publication; never repair source files.
        preflight(inventory, overwrite=overwrite)
        summary = _summary(inventory, records)
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        _write_jsonl(temporary / "records.jsonl", records)
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_auk_shadow(root: Path) -> tuple[AukInventory, list[AukRecord], AukSummary]:
    root = root.expanduser().resolve(strict=True)
    inventory = AukInventory.model_validate_json((root / "inventory.json").read_text())
    if root != auk_stage_root(
        Path(inventory.audio_production_root), inventory.shadow_run_id
    ):
        raise ValueError("AuK stage ownership differs")
    records = [
        AukRecord.model_validate(row) for row in _read_jsonl(root / "records.jsonl")
    ]
    if [r.clip_uid for r in records] != inventory.clip_uids:
        raise ValueError("AuK record order differs")
    for job, record in zip(inventory.jobs, records, strict=True):
        if (
            record.inventory_fingerprint != inventory.inventory_fingerprint
            or record.configuration_fingerprint
            != inventory.model_configuration.configuration_fingerprint
            or record.source_audio_sha256 != job.source_audio_sha256
            or record.expected_frame_count != job.source_frame_count
        ):
            raise ValueError("AuK record lineage differs")
        for artifact in (record.raw, record.canonical):
            if artifact is not None:
                path = Path(artifact.path)
                if (
                    not path.resolve().is_relative_to(root)
                    or audio_artifact(path, path) != artifact
                ):
                    raise ValueError("AuK waveform provenance differs")
    summary = AukSummary.model_validate_json((root / "summary.json").read_text())
    if summary != _summary(inventory, records):
        raise ValueError("AuK summary counts/lineage differ")
    return inventory, records, summary
