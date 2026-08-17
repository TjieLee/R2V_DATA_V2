from __future__ import annotations

import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    FaceEmbeddingBackend,
    SpeakerEmbeddingBackend,
)
from r2v_data_v2.h3.audio_binding import _file_asset, _save_embedding, _sha256
from r2v_data_v2.h3.audio_schemas import EmbeddingAsset, FileAsset
from r2v_data_v2.h3.primary_voice import PrimaryVoiceReferenceSelection
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel

EMBEDDING_PILOT_SCHEMA_VERSION = "r2v.h3.embedding_pilot.1"


@dataclass(frozen=True)
class EmbeddingPilotInput:
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    visual_reference_path: Path
    visual_reference_sha256: str
    primary_voice_reference_path: Path
    primary_voice_reference_sha256: str
    primary_voice_duration_seconds: float
    local_binding_valid: bool


class EmbeddingPilotModalityResult(SchemaModel):
    modality: Literal["face", "speaker"]
    status: Literal["available", "unavailable", "failed"]
    crop_asset: FileAsset | None = None
    embedding_asset: EmbeddingAsset | None = None
    model_identifier: str
    model_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> EmbeddingPilotModalityResult:
        if not self.model_identifier.strip():
            raise ValueError("embedding pilot model identifier must not be empty")
        if self.status == "available":
            if self.embedding_asset is None or self.failure_reason is not None:
                raise ValueError("available embedding requires an asset and no failure")
            if (self.modality == "face") != (self.crop_asset is not None):
                raise ValueError("only available face embeddings require a crop")
        elif (
            self.embedding_asset is not None
            or self.crop_asset is not None
            or self.failure_reason is None
            or not self.failure_reason.strip()
        ):
            raise ValueError("unavailable embedding requires only a failure reason")
        return self


class EmbeddingPilotOccurrence(SchemaModel):
    schema_version: Literal["r2v.h3.embedding_pilot.1"] = (
        EMBEDDING_PILOT_SCHEMA_VERSION
    )
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    visual_reference_path: str
    visual_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_path: str
    primary_voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_binding_valid: Literal[True] = True
    primary_voice_reference_valid: Literal[True] = True
    face: EmbeddingPilotModalityResult
    speaker: EmbeddingPilotModalityResult

    @model_validator(mode="after")
    def validate_occurrence(self) -> EmbeddingPilotOccurrence:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("embedding pilot occurrence identity is inconsistent")
        if not self.visual_reference_path.strip() or not self.primary_voice_reference_path.strip():
            raise ValueError("embedding pilot source paths must not be empty")
        return self


class FaceSimilarityRecord(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    same_clip: bool
    face_similarity: float = Field(ge=-1, le=1)


class VoiceSimilarityRecord(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    same_clip: bool
    voice_similarity: float = Field(ge=-1, le=1)


class JointEmbeddingCandidate(SchemaModel):
    anchor_occurrence_id: str
    candidate_occurrence_id: str
    face_similarity: float = Field(ge=-1, le=1)
    face_anchor_to_candidate_rank: int = Field(gt=0)
    face_candidate_to_anchor_rank: int = Field(gt=0)
    voice_similarity: float = Field(ge=-1, le=1)
    voice_anchor_to_candidate_rank: int = Field(gt=0)
    voice_candidate_to_anchor_rank: int = Field(gt=0)
    same_clip: Literal[False] = False
    both_local_bindings_valid: Literal[True] = True
    both_primary_voice_refs_valid: Literal[True] = True
    face_model_identifier: str
    voice_model_identifier: str


class EmbeddingPilotSummary(SchemaModel):
    schema_version: Literal["r2v.h3.embedding_pilot_summary.1"] = (
        "r2v.h3.embedding_pilot_summary.1"
    )
    input_occurrence_count: int = Field(ge=0)
    face_embedding_available_count: int = Field(ge=0)
    face_embedding_unavailable_count: int = Field(ge=0)
    speaker_embedding_available_count: int = Field(ge=0)
    speaker_embedding_failed_count: int = Field(ge=0)
    both_embeddings_available_count: int = Field(ge=0)
    face_pair_count: int = Field(ge=0)
    voice_pair_count: int = Field(ge=0)
    cross_clip_joint_candidate_count: int = Field(ge=0)
    face_failure_reason_counts: dict[str, int]
    speaker_failure_reason_counts: dict[str, int]
    face_model_identifier: str
    face_model_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    voice_model_identifier: str
    voice_model_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    top_k: int = Field(gt=0)
    thresholds_calibrated: Literal[False] = False


def _resolve_relative_artifact(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("embedding pilot artifact path must be relative")
    resolved = (root / relative).resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError("embedding pilot artifact is not a file")
    return resolved


def load_embedding_pilot_inputs(
    *,
    audio_pilot_root: Path,
    primary_voice_root: Path,
) -> list[EmbeddingPilotInput]:
    pilot = audio_pilot_root.expanduser().resolve(strict=True)
    voice_root = primary_voice_root.expanduser().resolve(strict=True)
    selection_path = voice_root / "primary_voice_references.jsonl"
    selections = [
        PrimaryVoiceReferenceSelection.model_validate_json(line)
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [item for item in selections if item.primary_voice_reference is not None]
    occurrence_ids = [item.entity_occurrence_id for item in selected]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError("primary voice input contains duplicate occurrences")
    inputs: list[EmbeddingPilotInput] = []
    for selection in sorted(selected, key=lambda item: item.entity_occurrence_id):
        assert selection.primary_voice_reference is not None
        sidecar_path = pilot / "clips" / selection.clip_uid / "audio_binding.json"
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.status != "ready" or sidecar.h3_ir is None:
            raise ValueError("embedding pilot requires a ready Audio Binding sidecar")
        if sidecar.clip_uid != selection.clip_uid:
            raise ValueError("embedding pilot sidecar clip identity does not match")
        subjects = [
            item
            for item in sidecar.h3_ir.subjects
            if item.entity_id == selection.entity_id and item.reference_type == "subject"
        ]
        if len(subjects) != 1 or len(subjects[0].source_assets) != 1:
            raise ValueError("embedding pilot requires one canonical subject picture")
        picture_id = subjects[0].source_assets[0]
        pictures = [
            item for item in sidecar.h3_ir.picture_assets if item.picture_id == picture_id
        ]
        if len(pictures) != 1 or pictures[0].entity_id != selection.entity_id:
            raise ValueError("embedding pilot canonical picture binding is inconsistent")
        visual_root = Path(sidecar.source_run_root).expanduser().resolve(strict=True)
        visual_path = _resolve_relative_artifact(visual_root, pictures[0].path)
        voice_asset = selection.primary_voice_reference.asset
        voice_path = _resolve_relative_artifact(voice_root, voice_asset.path)
        if (
            _sha256(voice_path) != voice_asset.sha256
            or voice_path.stat().st_size != voice_asset.byte_size
        ):
            raise ValueError("embedding pilot primary voice asset provenance mismatch")
        local_binding_valid = any(
            binding.status == "bound" and binding.entity_id == selection.entity_id
            for binding in sidecar.bindings
        )
        if not local_binding_valid:
            raise ValueError("embedding pilot primary voice entity has no local binding")
        inputs.append(
            EmbeddingPilotInput(
                entity_occurrence_id=selection.entity_occurrence_id,
                clip_uid=selection.clip_uid,
                entity_id=selection.entity_id,
                visual_reference_path=visual_path,
                visual_reference_sha256=_sha256(visual_path),
                primary_voice_reference_path=voice_path,
                primary_voice_reference_sha256=voice_asset.sha256,
                primary_voice_duration_seconds=(
                    selection.primary_voice_reference.source_end
                    - selection.primary_voice_reference.source_start
                ),
                local_binding_valid=True,
            )
        )
    return inputs


def _backend_identity(backend: object) -> tuple[str, str | None]:
    identifier = str(getattr(backend, "model_identifier", type(backend).__name__))
    fingerprint = getattr(backend, "checkpoint_sha256", None)
    return identifier, None if fingerprint is None else str(fingerprint)


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("embedding pilot vector must be finite and non-empty")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding pilot vector norm must be positive")
    return np.ascontiguousarray(values / norm, dtype=np.float32)


def _face_reason(value: str | None) -> str:
    if value in {"face_not_found", "face_not_found_in_canonical_reference"}:
        return "face_not_found_in_canonical_reference"
    if value in {"multiple_faces", "multiple_faces_in_canonical_reference"}:
        return "multiple_faces_in_canonical_reference"
    return "face_embedding_runtime_failed"


def _jsonl(values: Sequence[SchemaModel]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in values
    )


def _pair_records(
    vectors: dict[str, np.ndarray],
    clip_by_occurrence: dict[str, str],
    *,
    modality: Literal["face", "voice"],
) -> list[FaceSimilarityRecord] | list[VoiceSimilarityRecord]:
    records: list[FaceSimilarityRecord] | list[VoiceSimilarityRecord] = []
    for left, right in combinations(sorted(vectors), 2):
        similarity = float(np.dot(vectors[left], vectors[right]))
        same_clip = clip_by_occurrence[left] == clip_by_occurrence[right]
        if modality == "face":
            records.append(
                FaceSimilarityRecord(
                    left_occurrence_id=left,
                    right_occurrence_id=right,
                    same_clip=same_clip,
                    face_similarity=similarity,
                )
            )
        else:
            records.append(
                VoiceSimilarityRecord(
                    left_occurrence_id=left,
                    right_occurrence_id=right,
                    same_clip=same_clip,
                    voice_similarity=similarity,
                )
            )
    return records


def _directional_ranks(
    vectors: dict[str, np.ndarray],
    clip_by_occurrence: dict[str, str],
) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = {}
    for anchor in sorted(vectors):
        candidates = [
            candidate
            for candidate in vectors
            if candidate != anchor
            and clip_by_occurrence[candidate] != clip_by_occurrence[anchor]
        ]
        candidates.sort(
            key=lambda candidate: (
                -float(np.dot(vectors[anchor], vectors[candidate])),
                candidate,
            )
        )
        ranks[anchor] = {
            candidate: rank for rank, candidate in enumerate(candidates, start=1)
        }
    return ranks


def build_joint_embedding_candidates(
    *,
    occurrences: Sequence[EmbeddingPilotOccurrence],
    face_vectors: dict[str, np.ndarray],
    voice_vectors: dict[str, np.ndarray],
    top_k: int,
) -> list[JointEmbeddingCandidate]:
    if top_k <= 0:
        raise ValueError("embedding pilot top_k must be positive")
    clip_by_occurrence = {
        item.entity_occurrence_id: item.clip_uid for item in occurrences
    }
    by_id = {item.entity_occurrence_id: item for item in occurrences}
    face_ranks = _directional_ranks(face_vectors, clip_by_occurrence)
    voice_ranks = _directional_ranks(voice_vectors, clip_by_occurrence)
    output: list[JointEmbeddingCandidate] = []
    for anchor in sorted(set(face_vectors) & set(voice_vectors)):
        face_top = {
            candidate
            for candidate, rank in face_ranks[anchor].items()
            if rank <= min(top_k, len(face_ranks[anchor]))
        }
        voice_top = {
            candidate
            for candidate, rank in voice_ranks[anchor].items()
            if rank <= min(top_k, len(voice_ranks[anchor]))
        }
        for candidate in sorted(face_top & voice_top):
            if candidate not in face_vectors or candidate not in voice_vectors:
                continue
            candidate_face_rank = face_ranks.get(candidate, {}).get(anchor)
            candidate_voice_rank = voice_ranks.get(candidate, {}).get(anchor)
            if candidate_face_rank is None or candidate_voice_rank is None:
                continue
            anchor_record = by_id[anchor]
            candidate_record = by_id[candidate]
            assert anchor_record.face.embedding_asset is not None
            assert anchor_record.speaker.embedding_asset is not None
            output.append(
                JointEmbeddingCandidate(
                    anchor_occurrence_id=anchor,
                    candidate_occurrence_id=candidate,
                    face_similarity=float(
                        np.dot(face_vectors[anchor], face_vectors[candidate])
                    ),
                    face_anchor_to_candidate_rank=face_ranks[anchor][candidate],
                    face_candidate_to_anchor_rank=candidate_face_rank,
                    voice_similarity=float(
                        np.dot(voice_vectors[anchor], voice_vectors[candidate])
                    ),
                    voice_anchor_to_candidate_rank=voice_ranks[anchor][candidate],
                    voice_candidate_to_anchor_rank=candidate_voice_rank,
                    both_local_bindings_valid=(
                        anchor_record.local_binding_valid
                        and candidate_record.local_binding_valid
                    ),
                    both_primary_voice_refs_valid=(
                        anchor_record.primary_voice_reference_valid
                        and candidate_record.primary_voice_reference_valid
                    ),
                    face_model_identifier=(
                        anchor_record.face.embedding_asset.model_identifier
                    ),
                    voice_model_identifier=(
                        anchor_record.speaker.embedding_asset.model_identifier
                    ),
                )
            )
    output.sort(
        key=lambda item: (
            item.anchor_occurrence_id,
            item.face_anchor_to_candidate_rank,
            item.voice_anchor_to_candidate_rank,
            item.candidate_occurrence_id,
        )
    )
    return output


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"embedding pilot output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    published = False
    try:
        temporary.replace(destination)
        published = True
    finally:
        if published:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)


def run_embedding_pilot(
    *,
    inputs: Sequence[EmbeddingPilotInput],
    output_root: Path,
    face_backend: FaceEmbeddingBackend,
    speaker_backend: SpeakerEmbeddingBackend,
    top_k: int = 5,
    overwrite: bool = False,
) -> EmbeddingPilotSummary:
    if top_k <= 0:
        raise ValueError("embedding pilot top_k must be positive")
    destination = output_root.expanduser().resolve(strict=False)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    ordered_inputs = sorted(inputs, key=lambda item: item.entity_occurrence_id)
    ids = [item.entity_occurrence_id for item in ordered_inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("embedding pilot inputs must use unique occurrence IDs")
    face_identifier, face_fingerprint = _backend_identity(face_backend)
    voice_identifier, voice_fingerprint = _backend_identity(speaker_backend)
    face_results: dict[str, EmbeddingPilotModalityResult] = {}
    speaker_results: dict[str, EmbeddingPilotModalityResult] = {}
    face_vectors: dict[str, np.ndarray] = {}
    voice_vectors: dict[str, np.ndarray] = {}
    try:
        temporary.mkdir()
        for item in ordered_inputs:
            try:
                result = face_backend.embed_face(
                    entity_occurrence_id=item.entity_occurrence_id,
                    image_path=item.visual_reference_path,
                )
                if (
                    result.status != "available"
                    or result.embedding is None
                    or result.face_crop is None
                ):
                    face_results[item.entity_occurrence_id] = (
                        EmbeddingPilotModalityResult(
                            modality="face",
                            status="unavailable",
                            model_identifier=face_identifier,
                            model_fingerprint=face_fingerprint,
                            failure_reason=_face_reason(result.reason),
                        )
                    )
                    continue
                crop_path = (
                    temporary
                    / "face_crops"
                    / item.clip_uid
                    / f"{item.entity_id}.png"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                result.face_crop.convert("RGB").save(crop_path, format="PNG")
                embedding_path = (
                    temporary
                    / "embeddings"
                    / "face"
                    / item.clip_uid
                    / f"{item.entity_id}.npy"
                )
                metadata = {
                    **(result.embedding.backend_metadata or {}),
                    "input_reference_sha256": item.visual_reference_sha256,
                }
                asset = _save_embedding(
                    vector=result.embedding.vector,
                    path=embedding_path,
                    root=temporary,
                    model_identifier=result.embedding.model_identifier,
                    checkpoint_sha256=result.embedding.checkpoint_sha256,
                    backend_metadata=metadata,
                )
                face_vectors[item.entity_occurrence_id] = _normalize(
                    result.embedding.vector
                )
                face_results[item.entity_occurrence_id] = EmbeddingPilotModalityResult(
                    modality="face",
                    status="available",
                    crop_asset=_file_asset(crop_path, temporary, "image/png"),
                    embedding_asset=asset,
                    model_identifier=result.embedding.model_identifier,
                    model_fingerprint=result.embedding.checkpoint_sha256,
                )
            except Exception:  # noqa: BLE001 - one model failure isolates one occurrence
                face_results[item.entity_occurrence_id] = EmbeddingPilotModalityResult(
                    modality="face",
                    status="failed",
                    model_identifier=face_identifier,
                    model_fingerprint=face_fingerprint,
                    failure_reason="face_embedding_runtime_failed",
                )

        for item in ordered_inputs:
            try:
                result = speaker_backend.embed_speaker(
                    entity_occurrence_id=item.entity_occurrence_id,
                    audio_path=item.primary_voice_reference_path,
                )
                embedding_path = (
                    temporary
                    / "embeddings"
                    / "voice"
                    / item.clip_uid
                    / f"{item.entity_id}.npy"
                )
                metadata = {
                    **(result.backend_metadata or {}),
                    "source_flac_sha256": item.primary_voice_reference_sha256,
                    "duration_seconds": item.primary_voice_duration_seconds,
                    "sample_rate_hz": 16000,
                }
                asset = _save_embedding(
                    vector=result.vector,
                    path=embedding_path,
                    root=temporary,
                    model_identifier=result.model_identifier,
                    checkpoint_sha256=result.checkpoint_sha256,
                    backend_metadata=metadata,
                )
                voice_vectors[item.entity_occurrence_id] = _normalize(result.vector)
                speaker_results[item.entity_occurrence_id] = (
                    EmbeddingPilotModalityResult(
                        modality="speaker",
                        status="available",
                        embedding_asset=asset,
                        model_identifier=result.model_identifier,
                        model_fingerprint=result.checkpoint_sha256,
                    )
                )
            except Exception:  # noqa: BLE001 - one model failure isolates one occurrence
                speaker_results[item.entity_occurrence_id] = (
                    EmbeddingPilotModalityResult(
                        modality="speaker",
                        status="failed",
                        model_identifier=voice_identifier,
                        model_fingerprint=voice_fingerprint,
                        failure_reason="speaker_embedding_runtime_failed",
                    )
                )

        occurrences = [
            EmbeddingPilotOccurrence(
                entity_occurrence_id=item.entity_occurrence_id,
                clip_uid=item.clip_uid,
                entity_id=item.entity_id,
                visual_reference_path=str(item.visual_reference_path),
                visual_reference_sha256=item.visual_reference_sha256,
                primary_voice_reference_path=str(item.primary_voice_reference_path),
                primary_voice_reference_sha256=item.primary_voice_reference_sha256,
                local_binding_valid=item.local_binding_valid,
                face=face_results[item.entity_occurrence_id],
                speaker=speaker_results[item.entity_occurrence_id],
            )
            for item in ordered_inputs
        ]
        clip_by_occurrence = {
            item.entity_occurrence_id: item.clip_uid for item in occurrences
        }
        face_pairs = _pair_records(
            face_vectors,
            clip_by_occurrence,
            modality="face",
        )
        voice_pairs = _pair_records(
            voice_vectors,
            clip_by_occurrence,
            modality="voice",
        )
        joint = build_joint_embedding_candidates(
            occurrences=occurrences,
            face_vectors=face_vectors,
            voice_vectors=voice_vectors,
            top_k=top_k,
        )
        face_failures = Counter(
            item.face.failure_reason
            for item in occurrences
            if item.face.failure_reason is not None
        )
        speaker_failures = Counter(
            item.speaker.failure_reason
            for item in occurrences
            if item.speaker.failure_reason is not None
        )
        summary = EmbeddingPilotSummary(
            input_occurrence_count=len(occurrences),
            face_embedding_available_count=sum(
                item.face.status == "available" for item in occurrences
            ),
            face_embedding_unavailable_count=sum(
                item.face.status != "available" for item in occurrences
            ),
            speaker_embedding_available_count=sum(
                item.speaker.status == "available" for item in occurrences
            ),
            speaker_embedding_failed_count=sum(
                item.speaker.status != "available" for item in occurrences
            ),
            both_embeddings_available_count=sum(
                item.face.status == "available"
                and item.speaker.status == "available"
                for item in occurrences
            ),
            face_pair_count=len(face_pairs),
            voice_pair_count=len(voice_pairs),
            cross_clip_joint_candidate_count=len(joint),
            face_failure_reason_counts=dict(sorted(face_failures.items())),
            speaker_failure_reason_counts=dict(sorted(speaker_failures.items())),
            face_model_identifier=face_identifier,
            face_model_fingerprint=face_fingerprint,
            voice_model_identifier=voice_identifier,
            voice_model_fingerprint=voice_fingerprint,
            top_k=top_k,
        )
        (temporary / "occurrences.jsonl").write_text(
            _jsonl(occurrences),
            encoding="utf-8",
        )
        (temporary / "face_similarity.jsonl").write_text(
            _jsonl(face_pairs),
            encoding="utf-8",
        )
        (temporary / "voice_similarity.jsonl").write_text(
            _jsonl(voice_pairs),
            encoding="utf-8",
        )
        (temporary / "joint_candidates.jsonl").write_text(
            _jsonl(joint),
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            summary.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
