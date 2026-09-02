from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    AudioMediaBackend,
    FaceEmbeddingBackend,
    MaterializedMedia,
    SpeakerEmbeddingBackend,
)
from r2v_data_v2.h3.audio_pairing import (
    H3_PAIR_POLICY_FACE_THRESHOLD,
    H3_PAIR_POLICY_VERSION,
    H3_PAIR_POLICY_VOICE_THRESHOLD,
    AudioPairingConfig,
    evaluate_pair_policy_v1,
    select_complete_donor_matching,
)
from r2v_data_v2.h3.audio_schemas import PairEvidence
from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    DiarizationTargetClip,
    DiarizationVisualReference,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.embedding_pilot import _normalize, materialize_face_embedding
from r2v_data_v2.h3.lr_asd import LRASDBackend, SpeechActivityBackend
from r2v_data_v2.h3.pilot import ExplicitPilotClip, run_h3_audio_binding_pilot
from r2v_data_v2.h3.pilot_schemas import H3AudioBindingPilotSummary
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
    VoiceReferenceQualityPolicy,
    export_primary_voice_references,
)
from r2v_data_v2.h3.review import ReviewMediaBackend
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel
from r2v_data_v2.h3.visual_production_source import (
    ReadableClipIdentity,
    VisualProductionClip,
    VisualProductionInventory,
)

JEA_PAIR_SCHEMA_VERSION = "r2v.h3.jea_pairs.2"
CANONICAL_AUDIO_CLIP_VERSION = "r2v.h3.canonical_audio_clip.3"
CANONICAL_AUDIO_CLIP_SUMMARY_VERSION = "r2v.h3.canonical_audio_clip_summary.3"
CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS = 0.10
CANONICAL_AUDIO_SAMPLE_RATE_HZ = 32000
CANONICAL_AUDIO_CHANNELS = 2
ANALYSIS_AUDIO_SAMPLE_RATE_HZ = 16000
ANALYSIS_AUDIO_CHANNELS = 1
VOICE_SAMPLE_MAPPING_POLICY = "round_time_seconds_times_32000_v1"


@dataclass(frozen=True)
class JEAProductionPaths:
    root: Path
    audio: Path
    primary_voice: Path
    embedding: Path
    pairs: Path
    diarization: Path
    asr: Path
    h3: Path


class CanonicalAudioClip(SchemaModel):
    schema_version: Literal["r2v.h3.canonical_audio_clip.3"] = (
        CANONICAL_AUDIO_CLIP_VERSION
    )
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    channels: Literal[2] = CANONICAL_AUDIO_CHANNELS
    audio_source: Literal["original_target_video_audio_stream"] = (
        "original_target_video_audio_stream"
    )
    frame_count: int = Field(gt=0)
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    subject_reference_count: int = Field(ge=0)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_binding(self) -> CanonicalAudioClip:
        expected_duration = self.frame_count / self.sample_rate_hz
        if abs(self.target_duration_seconds - expected_duration) > 1e-12:
            raise ValueError(
                "canonical audio duration must equal its persisted sample extent"
            )
        if (self.target_audio_binding_path is None) != (
            self.target_audio_binding_sha256 is None
        ):
            raise ValueError("canonical audio binding path/hash must be paired")
        if self.subject_reference_count == 0 and self.target_audio_binding_path is not None:
            raise ValueError("no-subject canonical audio cannot publish a binding")
        return self


class CanonicalAudioClipSummary(SchemaModel):
    schema_version: Literal["r2v.h3.canonical_audio_clip_summary.3"] = (
        CANONICAL_AUDIO_CLIP_SUMMARY_VERSION
    )
    canonical_audio_clip_schema_version: Literal[
        "r2v.h3.canonical_audio_clip.3"
    ] = CANONICAL_AUDIO_CLIP_VERSION
    visual_canonical_clip_count: int = Field(gt=0)
    canonical_audio_clip_count: int = Field(gt=0)
    subject_binding_clip_count: int = Field(ge=0)
    sample_rate_hz: Literal[32000] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    channels: Literal[2] = CANONICAL_AUDIO_CHANNELS

    @model_validator(mode="after")
    def validate_counts(self) -> CanonicalAudioClipSummary:
        if self.visual_canonical_clip_count != self.canonical_audio_clip_count:
            raise ValueError("canonical Visual and Audio clip counts differ")
        if self.subject_binding_clip_count > self.canonical_audio_clip_count:
            raise ValueError("canonical subject binding count exceeds clip count")
        return self


class _CanonicalAudioClipV1(SchemaModel):
    schema_version: Literal["r2v.h3.canonical_audio_clip.1"]
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    subject_reference_count: int = Field(ge=0)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class _CanonicalAudioClipV2(SchemaModel):
    schema_version: Literal["r2v.h3.canonical_audio_clip.2"]
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: Literal[32000]
    channels: Literal[2]
    audio_source: Literal["original_target_video_audio_stream"]
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    subject_reference_count: int = Field(ge=0)
    target_audio_binding_path: str | None = None
    target_audio_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


CanonicalAudioClipExisting = (
    CanonicalAudioClip | _CanonicalAudioClipV2 | _CanonicalAudioClipV1
)


def jea_production_paths(audio_production_root: Path) -> JEAProductionPaths:
    root = audio_production_root.expanduser().resolve(strict=False)
    return JEAProductionPaths(
        root=root,
        audio=root / "audio",
        primary_voice=root / "primary_voice",
        embedding=root / "embedding",
        pairs=root / "pairs",
        diarization=root / "diarization",
        asr=root / "asr",
        h3=root / "h3",
    )


def _display_path(identity: ReadableClipIdentity) -> PurePosixPath:
    value = PurePosixPath(identity.clip_display_path)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("clip_display_path must be safe and relative")
    return value


def audio_binding_path(root: Path, identity: ReadableClipIdentity) -> Path:
    return root / "clips" / Path(*_display_path(identity).parts) / "audio_binding.json"


def full_audio_path(root: Path, identity: ReadableClipIdentity) -> Path:
    value = _display_path(identity)
    return (root / "full_audio" / Path(*value.parts)).with_suffix(".flac")


def primary_voice_path(
    root: Path,
    identity: ReadableClipIdentity,
    *,
    entity_id: str,
) -> Path:
    if (
        not entity_id
        or entity_id in {".", ".."}
        or "\\" in entity_id
        or Path(entity_id).name != entity_id
    ):
        raise ValueError("entity_id must be a safe path component")
    return root / Path(*_display_path(identity).parts) / f"{entity_id}.flac"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_json(path: Path, value: SchemaModel) -> None:
    path.write_text(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_binding(
    audio_root: Path,
    item: VisualProductionClip,
) -> tuple[str | None, str | None, float | None]:
    if not item.subject_references:
        return None, None, None
    path = audio_binding_path(audio_root, item.identity)
    if not path.is_file():
        return None, None, None
    sidecar = AudioBindingSidecar.model_validate_json(path.read_text(encoding="utf-8"))
    if sidecar.clip_uid != item.identity.clip_uid:
        raise ValueError("canonical audio binding clip identity differs")
    duration = (
        sidecar.evidence.audio.duration_seconds
        if sidecar.evidence is not None
        else None
    )
    return str(path.resolve(strict=True)), _sha256_file(path), duration


def _publish_canonical_audio_manifest(
    *,
    visual_inventory: VisualProductionInventory,
    audio_root: Path,
    audio_backend: AudioMediaBackend,
    materialized_by_clip: dict[str, MaterializedMedia],
    duration_resolver: Callable[[VisualProductionClip], float] | None = None,
) -> CanonicalAudioClipSummary:
    destination = audio_root.expanduser().resolve(strict=True)
    existing_by_clip: dict[str, CanonicalAudioClipExisting] = {}
    existing_path = destination / "canonical_clips.jsonl"
    if existing_path.is_file():
        for line in existing_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            version = payload.get("schema_version")
            if version == CANONICAL_AUDIO_CLIP_VERSION:
                record: CanonicalAudioClipExisting = CanonicalAudioClip.model_validate(
                    payload
                )
            elif version == "r2v.h3.canonical_audio_clip.2":
                record = _CanonicalAudioClipV2.model_validate(payload)
            elif version == "r2v.h3.canonical_audio_clip.1":
                record = _CanonicalAudioClipV1.model_validate(payload)
            else:
                raise ValueError("canonical Audio manifest schema version is unsupported")
            if record.clip_uid in existing_by_clip:
                raise ValueError("canonical Audio manifest contains duplicate clip IDs")
            existing_by_clip[record.clip_uid] = record
    records: list[CanonicalAudioClip] = []
    for item in visual_inventory.canonical_clips:
        clip_uid = item.identity.clip_uid
        video_path = Path(item.sample.target_video).expanduser().resolve(strict=True)
        audio_path = full_audio_path(destination, item.identity).resolve(strict=True)
        if not audio_path.is_file():
            raise FileNotFoundError(f"canonical full audio is missing: {audio_path}")
        probe = audio_backend.probe_audio_file(audio_path)
        if (
            probe.sample_rate_hz != CANONICAL_AUDIO_SAMPLE_RATE_HZ
            or probe.channels != CANONICAL_AUDIO_CHANNELS
            or "flac" not in probe.format_name.lower()
        ):
            raise ValueError("persisted canonical audio must be 32 kHz stereo FLAC")
        binding_path, binding_hash, binding_duration = _canonical_binding(
            destination,
            item,
        )
        media = materialized_by_clip.get(clip_uid)
        existing = existing_by_clip.get(clip_uid)
        source_duration = (
            media.stream.duration_seconds
            if media is not None
            else (
                binding_duration
                if binding_duration is not None
                else (
                    float(duration_resolver(item))
                    if duration_resolver is not None
                    and not isinstance(existing, CanonicalAudioClip)
                    else None
                )
            )
        )
        if source_duration is not None and (
            abs(source_duration - probe.duration_seconds)
            > CANONICAL_AUDIO_TIMELINE_TOLERANCE_SECONDS
        ):
            raise ValueError("source and canonical full-audio timelines differ")
        audio_hash = _sha256_file(audio_path)
        if media is None and isinstance(existing, CanonicalAudioClip) and (
            existing.target_full_audio_path != str(audio_path)
            or existing.target_full_audio_sha256 != audio_hash
            or existing.frame_count != probe.frame_count
            or abs(existing.target_duration_seconds - probe.duration_seconds) > 1e-12
        ):
            raise ValueError("reused canonical audio differs from its current manifest")
        identity = item.identity
        records.append(
            CanonicalAudioClip(
                clip_uid=clip_uid,
                clip_display_path=identity.clip_display_path,
                media_collection_relpath=identity.media_collection_relpath,
                media_collection_name=identity.media_collection_name,
                episode_name=identity.episode_name,
                clip_name=identity.clip_name,
                shard_id=identity.shard_id,
                target_video_path=str(video_path),
                target_video_sha256=_sha256_file(video_path),
                target_full_audio_path=str(audio_path),
                target_full_audio_sha256=audio_hash,
                frame_count=probe.frame_count,
                target_duration_seconds=probe.duration_seconds,
                subject_reference_count=len(item.subject_references),
                target_audio_binding_path=binding_path,
                target_audio_binding_sha256=binding_hash,
            )
        )
    summary = CanonicalAudioClipSummary(
        visual_canonical_clip_count=visual_inventory.canonical_sample_count,
        canonical_audio_clip_count=len(records),
        subject_binding_clip_count=sum(
            record.target_audio_binding_path is not None for record in records
        ),
    )
    token = uuid.uuid4().hex
    temporary_records = destination / f".canonical_clips.jsonl.tmp-{token}"
    temporary_summary = destination / f".canonical_clips_summary.json.tmp-{token}"
    try:
        _write_jsonl(temporary_records, records)
        _write_json(temporary_summary, summary)
        temporary_records.replace(destination / "canonical_clips.jsonl")
        temporary_summary.replace(destination / "canonical_clips_summary.json")
    finally:
        temporary_records.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
    return summary


def materialize_canonical_audio_clips(
    *,
    visual_inventory: VisualProductionInventory,
    audio_root: Path,
    audio_backend: AudioMediaBackend,
    duration_resolver: Callable[[VisualProductionClip], float] | None = None,
) -> CanonicalAudioClipSummary:
    destination = audio_root.expanduser().resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    existing_current: dict[str, CanonicalAudioClip] = {}
    manifest_path = destination / "canonical_clips.jsonl"
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema_version") == CANONICAL_AUDIO_CLIP_VERSION:
                record = CanonicalAudioClip.model_validate(payload)
                existing_current[record.clip_uid] = record
    materialized: dict[str, MaterializedMedia] = {}
    for item in visual_inventory.canonical_clips:
        source_video = Path(item.sample.target_video).resolve(strict=True)
        expected = full_audio_path(destination, item.identity)
        existing = existing_current.get(item.identity.clip_uid)
        reusable = (
            existing is not None
            and expected.is_file()
            and existing.target_full_audio_path == str(expected.resolve(strict=True))
            and existing.target_full_audio_sha256 == _sha256_file(expected)
            and existing.sample_rate_hz == CANONICAL_AUDIO_SAMPLE_RATE_HZ
            and existing.channels == CANONICAL_AUDIO_CHANNELS
        )
        if not reusable:
            result = audio_backend.materialize_full_audio(
                clip_uid=item.identity.clip_uid,
                source_video_path=source_video,
                destination=expected,
                sample_rate_hz=CANONICAL_AUDIO_SAMPLE_RATE_HZ,
                channels=CANONICAL_AUDIO_CHANNELS,
                output_format="flac",
            )
            if (
                result.sample_rate_hz != CANONICAL_AUDIO_SAMPLE_RATE_HZ
                or result.channels != CANONICAL_AUDIO_CHANNELS
                or result.output_format != "flac"
            ):
                raise ValueError("JEA canonical-audio backend violated media contract")
            if result.path.resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError("JEA canonical-audio backend published an unexpected path")
            materialized[item.identity.clip_uid] = result
    return _publish_canonical_audio_manifest(
        visual_inventory=visual_inventory,
        audio_root=destination,
        audio_backend=audio_backend,
        materialized_by_clip=materialized,
        duration_resolver=duration_resolver,
    )


def run_jea_audio_stage(
    *,
    visual_inventory: VisualProductionInventory,
    output_root: Path,
    lr_asd_backend: LRASDBackend,
    speech_backend: SpeechActivityBackend,
    review_media_backend: ReviewMediaBackend,
    audio_backend: AudioMediaBackend,
    workers: int = 1,
    lr_asd_gpu_ids: list[str] | None = None,
    lr_asd_workers_per_gpu: int = 4,
    review_media_mode: Literal["all", "none"] = "all",
) -> H3AudioBindingPilotSummary:
    """Run frozen Audio binding for exactly the canonical multi-shard clips."""
    destination = output_root.expanduser().resolve(strict=False)
    explicit = []
    for item in visual_inventory.clips:
        clip_path = Path(item.clip_record_path).expanduser().resolve(strict=True)
        shard_root = clip_path.parents[2]
        explicit.append(
            ExplicitPilotClip(
                clip_path=clip_path,
                source_run_root=shard_root,
                artifact_relpath=Path(*_display_path(item.identity).parts),
            )
        )
    try:
        summary = run_h3_audio_binding_pilot(
            run_root=Path(visual_inventory.visual_runs_root),
            output_root=destination,
            lr_asd_backend=lr_asd_backend,
            speech_backend=speech_backend,
            review_media_backend=review_media_backend,
            explicit_clips=explicit,
            workers=workers,
            lr_asd_gpu_ids=lr_asd_gpu_ids,
            lr_asd_workers_per_gpu=lr_asd_workers_per_gpu,
            review_media_mode=review_media_mode,
        )
        materialize_canonical_audio_clips(
            visual_inventory=visual_inventory,
            audio_root=destination,
            audio_backend=audio_backend,
        )
        return summary
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def run_jea_primary_voice_stage(
    *,
    visual_inventory: VisualProductionInventory,
    audio_root: Path,
    output_root: Path,
    audio_backend: AudioMediaBackend,
    overwrite: bool = False,
) -> PrimaryVoiceReferenceExportSummary:
    identity_by_clip = {
        item.identity.clip_uid: item.identity for item in visual_inventory.clips
    }

    def output_path(clip_uid: str, entity_id: str) -> Path:
        identity = identity_by_clip.get(clip_uid)
        if identity is None:
            raise ValueError("primary voice clip is absent from Visual Production")
        return primary_voice_path(Path(), identity, entity_id=entity_id)

    canonical_path = audio_root.expanduser().resolve(strict=True) / "canonical_clips.jsonl"
    canonical = {
        item.clip_uid: item
        for item in (
            CanonicalAudioClip.model_validate_json(line)
            for line in canonical_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    def canonical_source(clip_uid: str) -> Path:
        record = canonical.get(clip_uid)
        if record is None:
            raise ValueError("primary voice clip is absent from canonical Audio")
        path = Path(record.target_full_audio_path).resolve(strict=True)
        if _sha256_file(path) != record.target_full_audio_sha256:
            raise ValueError("canonical target full-audio hash changed")
        return path

    return export_primary_voice_references(
        pilot_root=audio_root,
        output_root=output_root,
        audio_backend=audio_backend,
        policy=VoiceReferenceQualityPolicy(),
        overwrite=overwrite,
        output_path_for_entity=output_path,
        source_audio_for_clip=canonical_source,
        output_sample_rate_hz=CANONICAL_AUDIO_SAMPLE_RATE_HZ,
        output_channels=CANONICAL_AUDIO_CHANNELS,
        sample_mapping_policy=VOICE_SAMPLE_MAPPING_POLICY,
    )


class JEAOccurrenceEmbedding(SchemaModel):
    occurrence_id: str
    clip_uid: str
    entity_id: str
    subject_index: int = Field(gt=0)
    identity: ReadableClipIdentity
    visual_reference_path: str
    primary_voice_reference_path: str
    face_embedding: list[float]
    voice_embedding: list[float]
    speaker_embedding_preprocessing: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_occurrence(self) -> JEAOccurrenceEmbedding:
        if self.occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("JEA occurrence identity is inconsistent")
        if self.identity.clip_uid != self.clip_uid:
            raise ValueError("JEA readable identity differs from occurrence")
        if not self.face_embedding or not self.voice_embedding:
            raise ValueError("JEA pair occurrence requires both embeddings")
        return self


class JEAEmbeddingSummary(SchemaModel):
    input_subject_occurrence_count: int = Field(ge=0)
    primary_voice_available_count: int = Field(ge=0)
    face_embedding_available_count: int = Field(ge=0)
    speaker_embedding_available_count: int = Field(ge=0)
    complete_occurrence_count: int = Field(ge=0)
    parent_quota_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> JEAEmbeddingSummary:
        available = (
            self.primary_voice_available_count,
            self.face_embedding_available_count,
            self.speaker_embedding_available_count,
        )
        if any(value > self.input_subject_occurrence_count for value in available):
            raise ValueError("JEA embedding availability exceeds input occurrences")
        if self.complete_occurrence_count > min(available, default=0):
            raise ValueError("JEA complete embeddings exceed modality availability")
        return self


def _primary_voice_selections(
    root: Path,
) -> dict[str, PrimaryVoiceReferenceSelection]:
    rows = [
        PrimaryVoiceReferenceSelection.model_validate_json(line)
        for line in (root / "primary_voice_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    result = {item.entity_occurrence_id: item for item in rows}
    if len(result) != len(rows):
        raise ValueError("primary voice selections contain duplicate occurrences")
    return result


def run_jea_embedding_stage(
    *,
    visual_inventory: VisualProductionInventory,
    primary_voice_root: Path,
    output_root: Path,
    face_backend: FaceEmbeddingBackend,
    speaker_backend: SpeakerEmbeddingBackend,
    overwrite: bool = False,
) -> JEAEmbeddingSummary:
    voice_root = primary_voice_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    selections = _primary_voice_selections(voice_root)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[JEAOccurrenceEmbedding] = []
    input_count = 0
    voice_count = 0
    face_count = 0
    speaker_count = 0
    try:
        temporary.mkdir()
        for clip in visual_inventory.clips:
            for subject_index, reference in enumerate(
                clip.subject_references, start=1
            ):
                if reference.entity_id is None:
                    raise ValueError("JEA subject reference requires an entity_id")
                input_count += 1
                occurrence_id = f"{clip.identity.clip_uid}/{reference.entity_id}"
                selection = selections.get(occurrence_id)
                if selection is None or selection.primary_voice_reference is None:
                    continue
                voice_asset = selection.primary_voice_reference.asset
                voice_path = (voice_root / voice_asset.path).resolve(strict=True)
                voice_path.relative_to(voice_root)
                if _sha256_file(voice_path) != voice_asset.sha256:
                    raise ValueError("primary voice artifact hash changed")
                voice_count += 1
                reference_path = Path(reference.artifact_path).resolve(strict=True)
                face = materialize_face_embedding(
                    backend=face_backend,
                    entity_occurrence_id=occurrence_id,
                    clip_uid=clip.identity.clip_uid,
                    entity_id=reference.entity_id,
                    canonical_reference_path=reference_path,
                    canonical_reference_sha256=_sha256_file(reference_path),
                    output_root=temporary,
                )
                if face.normalized_vector is None:
                    continue
                face_count += 1
                try:
                    speaker = speaker_backend.embed_speaker(
                        entity_occurrence_id=occurrence_id,
                        audio_path=voice_path,
                    )
                    voice_vector = _normalize(speaker.vector)
                except Exception:  # noqa: BLE001, S112 - isolate one occurrence
                    continue
                speaker_count += 1
                rows.append(
                    JEAOccurrenceEmbedding(
                        occurrence_id=occurrence_id,
                        clip_uid=clip.identity.clip_uid,
                        entity_id=reference.entity_id,
                        subject_index=subject_index,
                        identity=clip.identity,
                        visual_reference_path=str(reference_path),
                        primary_voice_reference_path=str(voice_path),
                        face_embedding=face.normalized_vector.tolist(),
                        voice_embedding=voice_vector.tolist(),
                        speaker_embedding_preprocessing=(
                            speaker.backend_metadata or {}
                        ),
                    )
                )
        rows.sort(
            key=lambda item: (
                item.identity.clip_display_path,
                item.subject_index,
                item.entity_id,
            )
        )
        summary = JEAEmbeddingSummary(
            input_subject_occurrence_count=input_count,
            primary_voice_available_count=voice_count,
            face_embedding_available_count=face_count,
            speaker_embedding_available_count=speaker_count,
            complete_occurrence_count=len(rows),
        )
        _write_jsonl(temporary / "occurrences.jsonl", rows)
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"JEA embedding output already exists: {destination}"
                )
            shutil.rmtree(destination)
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class JEAInPairSubject(SchemaModel):
    subject_index: int = Field(gt=0)
    target_occurrence_id: str
    target_entity_id: str
    target_visual_reference_path: str
    target_primary_voice_reference_path: str


class JEAInPair(SchemaModel):
    schema_version: Literal["r2v.h3.jea_pairs.2"] = JEA_PAIR_SCHEMA_VERSION
    pair_id: str
    target_clip_uid: str
    target_clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    target_video_path: str
    target_full_audio_path: str
    target_audio_binding_path: str
    subjects: list[JEAInPairSubject]


class JEACrossPairMapping(SchemaModel):
    mapping_id: str
    subject_index: int = Field(gt=0)
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_uid: str
    donor_clip_uid: str
    target_clip_display_path: str
    donor_clip_display_path: str
    media_collection_relpath: str
    target_visual_reference_path: str
    target_primary_voice_reference_path: str
    donor_primary_voice_reference_path: str
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD

    @model_validator(mode="after")
    def validate_mapping(self) -> JEACrossPairMapping:
        if self.target_clip_uid == self.donor_clip_uid:
            raise ValueError("JEA cross donor must come from another clip")
        if self.face_similarity < 0.72 or self.voice_similarity < 0.20:
            raise ValueError("JEA cross mapping must pass frozen PairPolicy V1")
        return self


class JEACrossPair(SchemaModel):
    schema_version: Literal["r2v.h3.jea_pairs.2"] = JEA_PAIR_SCHEMA_VERSION
    pair_id: str
    target_clip_uid: str
    target_clip_display_path: str
    media_collection_relpath: str
    target_video_path: str
    target_full_audio_path: str
    target_audio_binding_path: str
    mappings: list[JEACrossPairMapping]

    @model_validator(mode="after")
    def validate_pair(self) -> JEACrossPair:
        if self.pair_id != f"cross_pair/{self.target_clip_uid}/1":
            raise ValueError("JEA allows at most one cross pair per target clip")
        targets = [item.target_occurrence_id for item in self.mappings]
        donors = [item.donor_occurrence_id for item in self.mappings]
        if (
            not self.mappings
            or len(targets) != len(set(targets))
            or len(donors) != len(set(donors))
        ):
            raise ValueError("JEA multi-subject cross mapping must be one-to-one")
        if any(
            item.media_collection_relpath != self.media_collection_relpath
            for item in self.mappings
        ):
            raise ValueError("JEA cross mappings must remain in one full collection")
        return self


class JEAPairEvidence(SchemaModel):
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_display_path: str
    donor_clip_display_path: str
    media_collection_relpath: str
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    eligible: bool
    selected: bool
    rejection_reason_codes: list[str]
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD
    rank_gate_enabled: Literal[False] = False
    margin_gate_enabled: Literal[False] = False
    text_gate_enabled: Literal[False] = False


class JEAPairSummary(SchemaModel):
    schema_version: Literal["r2v.h3.jea_pairs.2"] = JEA_PAIR_SCHEMA_VERSION
    in_pair_count: int = Field(ge=0)
    cross_pair_count: int = Field(ge=0)
    selected_mapping_count: int = Field(ge=0)
    rejection_reason_counts: dict[str, int]
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD
    rank_gate_enabled: Literal[False] = False
    margin_gate_enabled: Literal[False] = False
    text_gate_enabled: Literal[False] = False
    maximum_cross_pairs_per_target: Literal[1] = 1


@dataclass(frozen=True)
class _CanonicalAudioTimeline:
    sample_rate_hz: int
    channels: int
    frame_count: int
    duration_seconds: float


def _probe_canonical_audio(
    path: Path,
    *,
    ffprobe: str,
) -> _CanonicalAudioTimeline:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration,duration_ts,time_base:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("cannot inspect canonical diarization audio")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if len(streams) != 1:
            raise ValueError("canonical audio must expose exactly one selected stream")
        stream = streams[0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        duration_value = stream.get("duration") or payload.get("format", {}).get(
            "duration"
        )
        duration = float(duration_value)
        duration_ts = stream.get("duration_ts")
        time_base = stream.get("time_base")
        if duration_ts not in {None, "N/A"} and time_base not in {None, "N/A"}:
            frames = Fraction(int(duration_ts)) * Fraction(str(time_base)) * sample_rate
            if frames.denominator != 1:
                raise ValueError("canonical audio timestamp extent is not sample-aligned")
            frame_count = int(frames)
        else:
            frame_count = round(duration * sample_rate)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("canonical diarization audio metadata is invalid") from exc
    if (
        sample_rate != CANONICAL_AUDIO_SAMPLE_RATE_HZ
        or channels != CANONICAL_AUDIO_CHANNELS
    ):
        raise ValueError("canonical diarization source must be 32 kHz stereo")
    if frame_count <= 0 or duration <= 0:
        raise ValueError("canonical diarization audio timeline is empty")
    actual_duration = frame_count / sample_rate
    if abs(actual_duration - duration) > (1 / sample_rate + 1e-9):
        raise ValueError("canonical audio duration and sample extent differ")
    return _CanonicalAudioTimeline(
        sample_rate_hz=sample_rate,
        channels=channels,
        frame_count=frame_count,
        duration_seconds=actual_duration,
    )


def build_jea_diarization_inventory(
    *,
    visual_inventory: VisualProductionInventory,
    audio_root: Path,
    ffprobe: str = "ffprobe",
) -> DiarizationInventory:
    canonical_root = audio_root.expanduser().resolve(strict=True)
    manifest_path = (canonical_root / "canonical_clips.jsonl").resolve(strict=True)
    visual_root = Path(visual_inventory.visual_production_root).expanduser().resolve(
        strict=True
    )
    visual_inventory_path = (visual_root / "samples.jsonl").resolve(strict=True)
    canonical = [
        CanonicalAudioClip.model_validate_json(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical_by_clip = {item.clip_uid: item for item in canonical}
    visual_ids = [item.identity.clip_uid for item in visual_inventory.canonical_clips]
    if len(canonical_by_clip) != len(canonical):
        raise ValueError("canonical Audio manifest contains duplicate clip IDs")
    if len(visual_ids) != len(set(visual_ids)):
        raise ValueError("canonical Visual inventory contains duplicate clip IDs")
    if set(canonical_by_clip) != set(visual_ids):
        raise ValueError("canonical Visual and Audio clip inventories differ")
    targets: list[DiarizationTargetClip] = []
    for visual in visual_inventory.canonical_clips:
        clip_uid = visual.identity.clip_uid
        record = canonical_by_clip[clip_uid]
        video_path = Path(visual.sample.target_video).expanduser().resolve(strict=True)
        manifest_video = Path(record.target_video_path).expanduser().resolve(strict=True)
        if (
            manifest_video != video_path
            or _sha256_file(video_path) != record.target_video_sha256
        ):
            raise ValueError("canonical Visual and Audio target video provenance differs")
        source_audio = Path(record.target_full_audio_path).expanduser().resolve(
            strict=True
        )
        if _sha256_file(source_audio) != record.target_full_audio_sha256:
            raise ValueError("canonical full-audio hash differs from manifest")
        timeline = _probe_canonical_audio(source_audio, ffprobe=ffprobe)
        if (
            timeline.frame_count != record.frame_count
            or abs(timeline.duration_seconds - record.target_duration_seconds) > 1e-12
        ):
            raise ValueError("canonical full-audio timeline differs from manifest")
        sidecar_path: Path | None = None
        if record.target_audio_binding_path is not None:
            sidecar_path = Path(record.target_audio_binding_path).expanduser().resolve(
                strict=True
            )
            if _sha256_file(sidecar_path) != record.target_audio_binding_sha256:
                raise ValueError("canonical Audio binding hash differs from manifest")
            sidecar = AudioBindingSidecar.model_validate_json(
                sidecar_path.read_text(encoding="utf-8")
            )
            if sidecar.clip_uid != clip_uid:
                raise ValueError("canonical Audio binding clip identity differs")
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=clip_uid,
                target_video_path=str(video_path),
                source_audio_path=str(source_audio),
                source_audio_sha256=_sha256_file(source_audio),
                source_sample_rate_hz=timeline.sample_rate_hz,
                source_channels=timeline.channels,
                source_frame_count=timeline.frame_count,
                target_audio_binding_path=(
                    None if sidecar_path is None else str(sidecar_path)
                ),
                target_audio_binding_sha256=record.target_audio_binding_sha256,
                visual_references=[
                    DiarizationVisualReference(
                        entity_id=reference.entity_id,
                        image_path=reference.artifact_path,
                    )
                    for reference in visual.subject_references
                    if reference.entity_id is not None
                ],
            )
        )
    manifest_sha256 = _sha256_file(manifest_path)
    visual_sha256 = _sha256_file(visual_inventory_path)
    fingerprint = _diarization_inventory_fingerprint(
        source_pairs_sha256=None,
        source_asr_inventory_fingerprint=None,
        mode="production",
        targets=targets,
        source_inventory_kind="canonical_audio_manifest",
        source_visual_inventory_sha256=visual_sha256,
        source_canonical_audio_manifest_sha256=manifest_sha256,
    )
    return DiarizationInventory(
        mode="production",
        source_inventory_kind="canonical_audio_manifest",
        source_visual_production_root=str(visual_root),
        source_visual_inventory_path=str(visual_inventory_path),
        source_visual_inventory_sha256=visual_sha256,
        source_canonical_audio_manifest_path=str(manifest_path),
        source_canonical_audio_manifest_sha256=manifest_sha256,
        inventory_fingerprint=fingerprint,
        source_target_count=len(targets),
        selected_target_count=len(targets),
        selection_mode="canonical_visual_target_inventory_v1",
        bounded_selection_applied=False,
        targets=targets,
    )


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError("pair embedding must be finite and non-zero")
    return vector / norm


def build_jea_pairs(
    *,
    visual_inventory: VisualProductionInventory,
    occurrences: Sequence[JEAOccurrenceEmbedding],
    audio_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> JEAPairSummary:
    visual_by_clip = {item.identity.clip_uid: item for item in visual_inventory.clips}
    by_clip: dict[str, list[JEAOccurrenceEmbedding]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence.clip_uid not in visual_by_clip:
            raise ValueError(
                "pair occurrence is absent from canonical Visual Production"
            )
        by_clip[occurrence.clip_uid].append(occurrence)
    for values in by_clip.values():
        values.sort(key=lambda item: item.subject_index)

    in_pairs: list[JEAInPair] = []
    for clip_uid, values in sorted(
        by_clip.items(), key=lambda item: item[1][0].identity.clip_display_path
    ):
        visual = visual_by_clip[clip_uid]
        identity = visual.identity
        in_pairs.append(
            JEAInPair(
                pair_id=f"in_pair/{clip_uid}",
                target_clip_display_path=identity.clip_display_path,
                media_collection_relpath=identity.media_collection_relpath,
                media_collection_name=identity.media_collection_name,
                episode_name=identity.episode_name,
                clip_name=identity.clip_name,
                shard_id=identity.shard_id,
                target_clip_uid=clip_uid,
                target_video_path=visual.sample.target_video,
                target_full_audio_path=str(full_audio_path(audio_root, identity)),
                target_audio_binding_path=str(audio_binding_path(audio_root, identity)),
                subjects=[
                    JEAInPairSubject(
                        subject_index=item.subject_index,
                        target_occurrence_id=item.occurrence_id,
                        target_entity_id=item.entity_id,
                        target_visual_reference_path=item.visual_reference_path,
                        target_primary_voice_reference_path=item.primary_voice_reference_path,
                    )
                    for item in values
                ],
            )
        )

    evidence: list[JEAPairEvidence] = []
    eligible: dict[str, list[PairEvidence]] = defaultdict(list)
    donor_by_id = {item.occurrence_id: item for item in occurrences}
    rejection_counts: Counter[str] = Counter()
    face = {item.occurrence_id: _unit(item.face_embedding) for item in occurrences}
    voice = {item.occurrence_id: _unit(item.voice_embedding) for item in occurrences}
    for target in occurrences:
        for donor in occurrences:
            if (
                target.occurrence_id == donor.occurrence_id
                or target.clip_uid == donor.clip_uid
            ):
                continue
            if (
                target.identity.media_collection_relpath
                != donor.identity.media_collection_relpath
            ):
                continue
            face_similarity = float(
                np.dot(face[target.occurrence_id], face[donor.occurrence_id])
            )
            voice_similarity = float(
                np.dot(voice[target.occurrence_id], voice[donor.occurrence_id])
            )
            reasons = []
            if face_similarity < 0.72:
                reasons.append("face_similarity_below_threshold")
            if voice_similarity < 0.20:
                reasons.append("voice_similarity_below_threshold")
            rejection_counts.update(reasons)
            edge = evaluate_pair_policy_v1(
                target_entity_occurrence_id=target.occurrence_id,
                reference_entity_occurrence_id=donor.occurrence_id,
                face_similarity=face_similarity,
                voice_similarity=voice_similarity,
                config=AudioPairingConfig(),
            )
            if not reasons:
                eligible[target.occurrence_id].append(edge)
            evidence.append(
                JEAPairEvidence(
                    target_occurrence_id=target.occurrence_id,
                    donor_occurrence_id=donor.occurrence_id,
                    target_clip_display_path=target.identity.clip_display_path,
                    donor_clip_display_path=donor.identity.clip_display_path,
                    media_collection_relpath=target.identity.media_collection_relpath,
                    face_similarity=face_similarity,
                    voice_similarity=voice_similarity,
                    eligible=not reasons,
                    selected=False,
                    rejection_reason_codes=reasons,
                )
            )
    for values in eligible.values():
        values.sort(
            key=lambda edge: (
                -edge.same_person.face_similarity,
                donor_by_id[edge.reference_entity_occurrence_id]
                .identity.clip_display_path,
                edge.reference_entity_occurrence_id,
            )
        )

    cross_pairs: list[JEACrossPair] = []
    selected_edges: set[tuple[str, str]] = set()
    for pair in in_pairs:
        targets = by_clip[pair.target_clip_uid]
        matching = select_complete_donor_matching(
            [eligible[item.occurrence_id] for item in targets]
        )
        if matching is None:
            continue
        mappings: list[JEACrossPairMapping] = []
        for index, edge in enumerate(matching, start=1):
            target = targets[index - 1]
            donor = donor_by_id[edge.reference_entity_occurrence_id]
            face_similarity = edge.same_person.face_similarity
            voice_similarity = edge.same_voice.voice_similarity
            mapping_id = f"cross_pair/{pair.target_clip_uid}/1/subject_{index}"
            mappings.append(
                JEACrossPairMapping(
                    mapping_id=mapping_id,
                    subject_index=index,
                    target_occurrence_id=target.occurrence_id,
                    donor_occurrence_id=donor.occurrence_id,
                    target_clip_uid=target.clip_uid,
                    donor_clip_uid=donor.clip_uid,
                    target_clip_display_path=target.identity.clip_display_path,
                    donor_clip_display_path=donor.identity.clip_display_path,
                    media_collection_relpath=target.identity.media_collection_relpath,
                    target_visual_reference_path=target.visual_reference_path,
                    target_primary_voice_reference_path=target.primary_voice_reference_path,
                    donor_primary_voice_reference_path=donor.primary_voice_reference_path,
                    face_similarity=face_similarity,
                    voice_similarity=voice_similarity,
                )
            )
            selected_edges.add((target.occurrence_id, donor.occurrence_id))
        cross_pairs.append(
            JEACrossPair(
                pair_id=f"cross_pair/{pair.target_clip_uid}/1",
                target_clip_uid=pair.target_clip_uid,
                target_clip_display_path=pair.target_clip_display_path,
                media_collection_relpath=pair.media_collection_relpath,
                target_video_path=pair.target_video_path,
                target_full_audio_path=pair.target_full_audio_path,
                target_audio_binding_path=pair.target_audio_binding_path,
                mappings=mappings,
            )
        )
    evidence = [
        item.model_copy(
            update={
                "selected": (item.target_occurrence_id, item.donor_occurrence_id)
                in selected_edges
            }
        )
        for item in evidence
    ]
    summary = JEAPairSummary(
        in_pair_count=len(in_pairs),
        cross_pair_count=len(cross_pairs),
        selected_mapping_count=len(selected_edges),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"JEA pair output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        for name, values in (
            ("in_pairs.jsonl", in_pairs),
            ("cross_pairs.jsonl", cross_pairs),
            ("pair_evidence.jsonl", evidence),
        ):
            (temporary / name).write_text(
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
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        rows = "".join(
            f"<tr><td>{html.escape(item.target_clip_display_path)}</td><td>{html.escape(item.donor_clip_display_path)}</td><td>{html.escape(item.media_collection_relpath)}</td><td>{item.face_similarity:.4f}</td><td>{item.voice_similarity:.4f}</td></tr>"
            for item in sorted(
                evidence,
                key=lambda row: (
                    row.target_clip_display_path,
                    row.donor_clip_display_path,
                ),
            )
            if item.eligible
        )
        (temporary / "review.html").write_text(
            "<!doctype html><meta charset='utf-8'><table><tr><th>target</th><th>donor</th><th>collection</th><th>face</th><th>voice</th></tr>"
            + rows
            + "</table>",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
