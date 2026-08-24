from __future__ import annotations

import hashlib
import html
import json
import shutil
import uuid
import wave
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    AudioMediaBackend,
    FaceEmbeddingBackend,
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
    build_complete_diarization_inventory,
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
    VisualProductionInventory,
)

JEA_PAIR_SCHEMA_VERSION = "r2v.h3.jea_pairs.1"


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


def run_jea_audio_stage(
    *,
    visual_inventory: VisualProductionInventory,
    output_root: Path,
    lr_asd_backend: LRASDBackend,
    speech_backend: SpeechActivityBackend,
    review_media_backend: ReviewMediaBackend,
    audio_backend: AudioMediaBackend,
    workers: int = 1,
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
        )
        for item in visual_inventory.clips:
            expected = full_audio_path(destination, item.identity)
            materialized = audio_backend.materialize_full_audio(
                clip_uid=item.identity.clip_uid,
                source_video_path=Path(item.sample.target_video).resolve(strict=True),
                destination=expected,
                sample_rate_hz=16000,
                channels=1,
                output_format="flac",
            )
            if materialized.path.resolve(strict=True) != expected.resolve(strict=True):
                raise ValueError("JEA full-audio backend published an unexpected path")
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

    return export_primary_voice_references(
        pilot_root=audio_root,
        output_root=output_root,
        audio_backend=audio_backend,
        policy=VoiceReferenceQualityPolicy(),
        overwrite=overwrite,
        output_path_for_entity=output_path,
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
    schema_version: Literal["r2v.h3.jea_pairs.1"] = JEA_PAIR_SCHEMA_VERSION
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
    schema_version: Literal["r2v.h3.jea_pairs.1"] = JEA_PAIR_SCHEMA_VERSION
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
    schema_version: Literal["r2v.h3.jea_pairs.1"] = JEA_PAIR_SCHEMA_VERSION
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


def _pcm16_frame_count(path: Path, *, sample_rate: int, channels: int) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getsampwidth() != 2
                or source.getcomptype() != "NONE"
                or source.getframerate() != sample_rate
                or source.getnchannels() != channels
            ):
                raise ValueError("JEA diarization source must be matching PCM16 WAV")
            return source.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("JEA diarization source audio is unreadable") from exc


def build_jea_diarization_inventory(
    *,
    pairs_root: Path,
) -> DiarizationInventory:
    root = pairs_root.expanduser().resolve(strict=True)
    pairs_path = (root / "in_pairs.jsonl").resolve(strict=True)
    pairs = [
        JEAInPair.model_validate_json(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    targets: list[DiarizationTargetClip] = []
    for pair in sorted(pairs, key=lambda item: item.target_clip_uid):
        video_path = Path(pair.target_video_path).expanduser().resolve(strict=True)
        full_audio = Path(pair.target_full_audio_path).expanduser().resolve(strict=True)
        sidecar_path = (
            Path(pair.target_audio_binding_path).expanduser().resolve(strict=True)
        )
        if not full_audio.is_file():
            raise ValueError("JEA in-pair canonical full audio is unavailable")
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if (
            sidecar.clip_uid != pair.target_clip_uid
            or sidecar.status != "ready"
            or sidecar.evidence is None
        ):
            raise ValueError("JEA diarization requires a matching ready Audio sidecar")
        audio = sidecar.evidence.audio
        if (
            audio.full_audio_path is None
            or audio.sample_rate_hz is None
            or audio.channels is None
        ):
            raise ValueError("JEA Audio sidecar provenance is incomplete")
        source_audio = Path(audio.full_audio_path).expanduser().resolve(strict=True)
        targets.append(
            DiarizationTargetClip(
                target_clip_uid=pair.target_clip_uid,
                target_video_path=str(video_path),
                source_audio_path=str(source_audio),
                source_audio_sha256=_sha256_file(source_audio),
                source_sample_rate_hz=audio.sample_rate_hz,
                source_channels=audio.channels,
                source_frame_count=_pcm16_frame_count(
                    source_audio,
                    sample_rate=audio.sample_rate_hz,
                    channels=audio.channels,
                ),
                target_audio_binding_path=str(sidecar_path),
                visual_references=[
                    DiarizationVisualReference(
                        entity_id=subject.target_entity_id,
                        image_path=subject.target_visual_reference_path,
                    )
                    for subject in pair.subjects
                ],
            )
        )
    return build_complete_diarization_inventory(
        source_pairs_path=pairs_path,
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
