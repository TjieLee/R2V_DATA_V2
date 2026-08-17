from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import (
    AudioMediaBackend,
    FaceEmbeddingBackend,
    SpeakerEmbeddingBackend,
)
from r2v_data_v2.h3.audio_binding import _save_embedding
from r2v_data_v2.h3.audio_pairing import (
    H3_PAIR_POLICY_FACE_THRESHOLD,
    H3_PAIR_POLICY_VERSION,
    H3_PAIR_POLICY_VOICE_THRESHOLD,
    AudioPairingConfig,
    evaluate_pair_policy_v1,
)
from r2v_data_v2.h3.audio_schemas import PairEvidence
from r2v_data_v2.h3.embedding_pilot import (
    EmbeddingPilotModalityResult,
    _backend_identity,
    _normalize,
    materialize_face_embedding,
)
from r2v_data_v2.h3.pair_calibration import (
    EligibleVisualFaceOccurrence,
    load_eligible_visual_face_occurrences,
)
from r2v_data_v2.h3.pilot_schemas import H3AudioBindingPilotSummary
from r2v_data_v2.h3.primary_voice import (
    PrimaryVoiceReferenceExportSummary,
    PrimaryVoiceReferenceSelection,
    VoiceReferenceQualityPolicy,
    export_primary_voice_references,
)
from r2v_data_v2.h3.schemas import SchemaModel

PRODUCTION_STAGE_ORDER = ("audio", "primary-voice", "embedding", "pair")
PRODUCTION_INVENTORY_VERSION = "r2v.h3.production_inventory.1"
PRODUCTION_EMBEDDING_VERSION = "r2v.h3.production_embedding.1"
PRODUCTION_PAIR_VERSION = "r2v.h3.production_pairs.1"


class ProductionVisualOccurrence(SchemaModel):
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    parent_video_id: str
    clip_suffix: str
    canonical_reference_path: str
    canonical_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_reference_run_path: str
    visual_integrity_provenance: dict[str, object]

    @model_validator(mode="after")
    def validate_identity(self) -> ProductionVisualOccurrence:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("production occurrence identity is inconsistent")
        if not self.parent_video_id.strip() or not self.clip_suffix.strip():
            raise ValueError("production occurrence source provenance is incomplete")
        return self


class H3ProductionInventory(SchemaModel):
    schema_version: Literal["r2v.h3.production_inventory.1"] = (
        PRODUCTION_INVENTORY_VERSION
    )
    source_run_root: str
    scanned_clip_count: int = Field(ge=0)
    eligible_clip_count: int = Field(ge=0)
    eligible_occurrence_count: int = Field(ge=0)
    eligible_clip_uids: list[str]
    occurrences: list[ProductionVisualOccurrence]
    skip_reason_counts: dict[str, int]
    selection_mode: Literal["complete_visual_human_occurrences_v1"] = (
        "complete_visual_human_occurrences_v1"
    )
    bounded_limit_applied: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    calibration_sampling_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_complete_inventory(self) -> H3ProductionInventory:
        occurrence_ids = [item.entity_occurrence_id for item in self.occurrences]
        if occurrence_ids != sorted(occurrence_ids) or len(occurrence_ids) != len(
            set(occurrence_ids)
        ):
            raise ValueError("production occurrences must be unique and ordered")
        expected_clips = sorted({item.clip_uid for item in self.occurrences})
        if self.eligible_clip_uids != expected_clips:
            raise ValueError("production eligible clip IDs must match occurrences")
        if self.eligible_clip_count != len(expected_clips):
            raise ValueError("production eligible clip count is inconsistent")
        if self.eligible_occurrence_count != len(self.occurrences):
            raise ValueError("production eligible occurrence count is inconsistent")
        if self.scanned_clip_count < self.eligible_clip_count:
            raise ValueError("production inventory cannot exceed scanned clips")
        if self.scanned_clip_count != self.eligible_clip_count + sum(
            self.skip_reason_counts.values()
        ):
            raise ValueError("production inventory must account for every scanned clip")
        return self


class H3ProductionEmbeddingOccurrence(SchemaModel):
    schema_version: Literal["r2v.h3.production_embedding.1"] = (
        PRODUCTION_EMBEDDING_VERSION
    )
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    parent_video_id: str
    clip_suffix: str
    visual_reference_path: str
    visual_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_path: str | None = None
    primary_voice_reference_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    primary_voice_unavailable_reasons: list[str] = Field(default_factory=list)
    face: EmbeddingPilotModalityResult
    speaker: EmbeddingPilotModalityResult

    @model_validator(mode="after")
    def validate_occurrence(self) -> H3ProductionEmbeddingOccurrence:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("production embedding occurrence identity is inconsistent")
        voice_values = (
            self.primary_voice_reference_path,
            self.primary_voice_reference_sha256,
        )
        if any(value is None for value in voice_values) != all(
            value is None for value in voice_values
        ):
            raise ValueError("primary voice path and hash must be set together")
        if self.primary_voice_reference_path is None:
            if not self.primary_voice_unavailable_reasons:
                raise ValueError("missing primary voice requires a reason")
            if self.speaker.status != "unavailable":
                raise ValueError("missing primary voice cannot run speaker embedding")
        elif self.primary_voice_unavailable_reasons:
            raise ValueError("available primary voice cannot have unavailable reasons")
        return self


class H3ProductionEmbeddingSummary(SchemaModel):
    schema_version: Literal["r2v.h3.production_embedding_summary.1"] = (
        "r2v.h3.production_embedding_summary.1"
    )
    input_occurrence_count: int = Field(ge=0)
    primary_voice_available_count: int = Field(ge=0)
    primary_voice_unavailable_count: int = Field(ge=0)
    face_embedding_available_count: int = Field(ge=0)
    face_embedding_unavailable_count: int = Field(ge=0)
    speaker_embedding_available_count: int = Field(ge=0)
    speaker_embedding_unavailable_count: int = Field(ge=0)
    both_embeddings_available_count: int = Field(ge=0)
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
    parent_quota_applied: Literal[False] = False
    calibration_sampling_applied: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> H3ProductionEmbeddingSummary:
        for available, unavailable in (
            (
                self.primary_voice_available_count,
                self.primary_voice_unavailable_count,
            ),
            (
                self.face_embedding_available_count,
                self.face_embedding_unavailable_count,
            ),
            (
                self.speaker_embedding_available_count,
                self.speaker_embedding_unavailable_count,
            ),
        ):
            if available + unavailable != self.input_occurrence_count:
                raise ValueError("production embedding counts must reconcile")
        return self


class H3ProductionInPair(SchemaModel):
    pair_id: str
    target_occurrence_id: str
    clip_uid: str
    entity_id: str
    visual_reference_path: str
    primary_voice_reference_path: str


class H3ProductionCrossPair(SchemaModel):
    pair_id: str
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_uid: str
    donor_clip_uid: str
    target_primary_voice_reference_path: str
    donor_visual_reference_path: str
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD


class H3ProductionPairEvidence(SchemaModel):
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_uid: str
    donor_clip_uid: str
    pair_evidence: PairEvidence
    duplicate_source: bool
    donor_eligible: bool
    selected: bool = False
    rejection_reason_codes: list[str]


class H3ProductionSummary(SchemaModel):
    schema_version: Literal["r2v.h3.production_pairs.1"] = PRODUCTION_PAIR_VERSION
    complete_eligible_occurrence_count: int = Field(ge=0)
    audio_clips_attempted: int = Field(ge=0)
    audio_clips_succeeded: int = Field(ge=0)
    audio_clips_failed: int = Field(ge=0)
    primary_voice_available_count: int = Field(ge=0)
    primary_voice_unavailable_count: int = Field(ge=0)
    face_embedding_available_count: int = Field(ge=0)
    face_embedding_unavailable_count: int = Field(ge=0)
    speaker_embedding_available_count: int = Field(ge=0)
    speaker_embedding_unavailable_count: int = Field(ge=0)
    in_pair_count: int = Field(ge=0)
    cross_pair_eligible_target_count: int = Field(ge=0)
    cross_pair_count: int = Field(ge=0)
    target_without_cross_pair_count: int = Field(ge=0)
    rejection_reason_counts: dict[str, int]
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD
    thresholds_calibrated: Literal[True] = True
    rank_gate_enabled: Literal[False] = False
    margin_gate_enabled: Literal[False] = False
    text_gate_enabled: Literal[False] = False
    exact_candidate_evaluation: Literal[True] = True
    parent_quota_applied: Literal[False] = False
    transitive_clustering_performed: Literal[False] = False
    human_calibration_labels_used_as_identity_truth: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> H3ProductionSummary:
        if self.audio_clips_attempted != (
            self.audio_clips_succeeded + self.audio_clips_failed
        ):
            raise ValueError("production Audio clip counts must reconcile")
        for available, unavailable in (
            (
                self.primary_voice_available_count,
                self.primary_voice_unavailable_count,
            ),
            (
                self.face_embedding_available_count,
                self.face_embedding_unavailable_count,
            ),
            (
                self.speaker_embedding_available_count,
                self.speaker_embedding_unavailable_count,
            ),
        ):
            if available + unavailable != self.complete_eligible_occurrence_count:
                raise ValueError("production occurrence availability must reconcile")
        if self.in_pair_count != self.primary_voice_available_count:
            raise ValueError("every and only primary-voice occurrence must have an in-pair")
        if self.cross_pair_count + self.target_without_cross_pair_count != (
            self.cross_pair_eligible_target_count
        ):
            raise ValueError("production cross-pair target counts must reconcile")
        return self


@dataclass(frozen=True)
class ProductionPaths:
    root: Path
    audio: Path
    primary_voice: Path
    embedding: Path
    pairs: Path
    inventory: Path


def production_paths(audio_run_root: Path) -> ProductionPaths:
    root = audio_run_root.expanduser().resolve(strict=False) / "production"
    return ProductionPaths(
        root=root,
        audio=root / "audio",
        primary_voice=root / "primary_voice",
        embedding=root / "embedding",
        pairs=root / "pairs",
        inventory=root / "source_inventory.json",
    )


def _production_occurrence(
    item: EligibleVisualFaceOccurrence,
) -> ProductionVisualOccurrence:
    return ProductionVisualOccurrence(
        entity_occurrence_id=item.entity_occurrence_id,
        clip_uid=item.clip_uid,
        entity_id=item.entity_id,
        parent_video_id=item.parent_video_id,
        clip_suffix=item.clip_suffix,
        canonical_reference_path=str(item.canonical_reference_path),
        canonical_reference_sha256=item.canonical_reference_sha256,
        canonical_reference_run_path=item.canonical_reference_run_path,
        visual_integrity_provenance=item.visual_integrity_provenance,
    )


def enumerate_h3_production_inventory(run_root: Path) -> H3ProductionInventory:
    source = run_root.expanduser().resolve(strict=True)
    loaded = load_eligible_visual_face_occurrences(source)
    if loaded.run_root != source:
        raise ValueError("production inventory loader returned a different run root")
    clip_paths = sorted((source / "clips").glob("*/clip.json"))
    occurrences = [_production_occurrence(item) for item in loaded.occurrences]
    inventory = H3ProductionInventory(
        source_run_root=str(source),
        scanned_clip_count=len(clip_paths),
        eligible_clip_count=len({item.clip_uid for item in occurrences}),
        eligible_occurrence_count=len(occurrences),
        eligible_clip_uids=sorted({item.clip_uid for item in occurrences}),
        occurrences=occurrences,
        skip_reason_counts=loaded.skip_reason_counts,
    )
    if not inventory.occurrences:
        raise ValueError("production inventory found no eligible human occurrences")
    return inventory


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative_asset(root: Path, value: str, expected_sha256: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("production asset path must be relative")
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("production asset provenance mismatch")
    return path


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"production output already exists: {destination}")
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


def write_production_inventory(
    inventory: H3ProductionInventory,
    path: Path,
) -> None:
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = inventory.model_dump_json(indent=2) + "\n"
    if destination.exists():
        current = H3ProductionInventory.model_validate_json(
            destination.read_text(encoding="utf-8")
        )
        if current != inventory:
            raise ValueError("production source inventory changed after stage publication")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def _primary_voice_selections(
    primary_voice_root: Path,
) -> dict[str, PrimaryVoiceReferenceSelection]:
    rows = [
        PrimaryVoiceReferenceSelection.model_validate_json(line)
        for line in (primary_voice_root / "primary_voice_references.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    output = {item.entity_occurrence_id: item for item in rows}
    if len(output) != len(rows):
        raise ValueError("primary voice selections contain duplicate occurrences")
    return output


def run_production_embedding_stage(
    *,
    inventory: H3ProductionInventory,
    primary_voice_root: Path,
    output_root: Path,
    face_backend: FaceEmbeddingBackend,
    speaker_backend: SpeakerEmbeddingBackend,
    overwrite: bool = False,
) -> H3ProductionEmbeddingSummary:
    voice_root = primary_voice_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    selections = _primary_voice_selections(voice_root)
    inventory_ids = {item.entity_occurrence_id for item in inventory.occurrences}
    if set(selections) - inventory_ids:
        raise ValueError("primary voice output contains a non-production occurrence")
    face_identifier, face_fingerprint = _backend_identity(face_backend)
    voice_identifier, voice_fingerprint = _backend_identity(speaker_backend)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    rows: list[H3ProductionEmbeddingOccurrence] = []
    try:
        temporary.mkdir()
        for item in inventory.occurrences:
            face = materialize_face_embedding(
                backend=face_backend,
                entity_occurrence_id=item.entity_occurrence_id,
                clip_uid=item.clip_uid,
                entity_id=item.entity_id,
                canonical_reference_path=Path(item.canonical_reference_path),
                canonical_reference_sha256=item.canonical_reference_sha256,
                output_root=temporary,
            ).result
            selection = selections.get(item.entity_occurrence_id)
            voice_path: Path | None = None
            voice_sha256: str | None = None
            unavailable_reasons: list[str] = []
            if selection is None:
                unavailable_reasons = ["audio_binding_unavailable"]
            elif selection.primary_voice_reference is None:
                unavailable_reasons = list(selection.reason_codes)
            else:
                asset = selection.primary_voice_reference.asset
                voice_path = _resolve_relative_asset(voice_root, asset.path, asset.sha256)
                voice_sha256 = asset.sha256
            if voice_path is None:
                speaker = EmbeddingPilotModalityResult(
                    modality="speaker",
                    status="unavailable",
                    model_identifier=voice_identifier,
                    model_fingerprint=voice_fingerprint,
                    failure_reason="primary_voice_reference_unavailable",
                )
            else:
                try:
                    result = speaker_backend.embed_speaker(
                        entity_occurrence_id=item.entity_occurrence_id,
                        audio_path=voice_path,
                    )
                    embedding_path = (
                        temporary
                        / "embeddings"
                        / "voice"
                        / item.clip_uid
                        / f"{item.entity_id}.npy"
                    )
                    asset = _save_embedding(
                        vector=result.vector,
                        path=embedding_path,
                        root=temporary,
                        model_identifier=result.model_identifier,
                        checkpoint_sha256=result.checkpoint_sha256,
                        backend_metadata={
                            **(result.backend_metadata or {}),
                            "source_flac_sha256": voice_sha256,
                            "sample_rate_hz": 16000,
                        },
                    )
                    _normalize(result.vector)
                    speaker = EmbeddingPilotModalityResult(
                        modality="speaker",
                        status="available",
                        embedding_asset=asset,
                        model_identifier=result.model_identifier,
                        model_fingerprint=result.checkpoint_sha256,
                    )
                except Exception:  # noqa: BLE001 - isolate one occurrence
                    speaker = EmbeddingPilotModalityResult(
                        modality="speaker",
                        status="failed",
                        model_identifier=voice_identifier,
                        model_fingerprint=voice_fingerprint,
                        failure_reason="speaker_embedding_runtime_failed",
                    )
            rows.append(
                H3ProductionEmbeddingOccurrence(
                    entity_occurrence_id=item.entity_occurrence_id,
                    clip_uid=item.clip_uid,
                    entity_id=item.entity_id,
                    parent_video_id=item.parent_video_id,
                    clip_suffix=item.clip_suffix,
                    visual_reference_path=item.canonical_reference_path,
                    visual_reference_sha256=item.canonical_reference_sha256,
                    primary_voice_reference_path=(
                        str(voice_path) if voice_path is not None else None
                    ),
                    primary_voice_reference_sha256=voice_sha256,
                    primary_voice_unavailable_reasons=unavailable_reasons,
                    face=face,
                    speaker=speaker,
                )
            )
        rows.sort(key=lambda value: value.entity_occurrence_id)
        face_failures = Counter(
            item.face.failure_reason
            for item in rows
            if item.face.failure_reason is not None
        )
        speaker_failures = Counter(
            item.speaker.failure_reason
            for item in rows
            if item.speaker.failure_reason is not None
        )
        summary = H3ProductionEmbeddingSummary(
            input_occurrence_count=len(rows),
            primary_voice_available_count=sum(
                item.primary_voice_reference_path is not None for item in rows
            ),
            primary_voice_unavailable_count=sum(
                item.primary_voice_reference_path is None for item in rows
            ),
            face_embedding_available_count=sum(
                item.face.status == "available" for item in rows
            ),
            face_embedding_unavailable_count=sum(
                item.face.status != "available" for item in rows
            ),
            speaker_embedding_available_count=sum(
                item.speaker.status == "available" for item in rows
            ),
            speaker_embedding_unavailable_count=sum(
                item.speaker.status != "available" for item in rows
            ),
            both_embeddings_available_count=sum(
                item.face.status == "available" and item.speaker.status == "available"
                for item in rows
            ),
            face_failure_reason_counts=dict(sorted(face_failures.items())),
            speaker_failure_reason_counts=dict(sorted(speaker_failures.items())),
            face_model_identifier=face_identifier,
            face_model_fingerprint=face_fingerprint,
            voice_model_identifier=voice_identifier,
            voice_model_fingerprint=voice_fingerprint,
        )
        (temporary / "occurrences.jsonl").write_text(_jsonl(rows), encoding="utf-8")
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


def _load_embedding_rows(root: Path) -> list[H3ProductionEmbeddingOccurrence]:
    rows = [
        H3ProductionEmbeddingOccurrence.model_validate_json(line)
        for line in (root / "occurrences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda item: item.entity_occurrence_id)
    ids = [item.entity_occurrence_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("production embeddings contain duplicate occurrences")
    return rows


def _load_embedding_vector(root: Path, item: EmbeddingPilotModalityResult) -> np.ndarray:
    if item.embedding_asset is None:
        raise ValueError("available production embedding is missing its asset")
    path = _resolve_relative_asset(
        root,
        item.embedding_asset.path,
        item.embedding_asset.sha256,
    )
    vector = np.load(path, allow_pickle=False)
    if vector.dtype != np.float32:
        raise ValueError("production embedding must be float32")
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("production embedding must be finite and non-empty")
    norm = float(np.linalg.norm(values))
    if not math.isclose(norm, 1.0, abs_tol=1e-5):
        raise ValueError("production embedding must be L2-normalized")
    return np.ascontiguousarray(values, dtype=np.float32)


def _source_video_hashes(audio_root: Path, clip_uids: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for clip_uid in sorted(clip_uids):
        sidecar = json.loads(
            (audio_root / "clips" / clip_uid / "audio_binding.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(sidecar, dict) or sidecar.get("clip_uid") != clip_uid:
            raise ValueError("production Audio sidecar identity is invalid")
        if sidecar.get("status") != "ready":
            raise ValueError("production pairing requires ready identity-matched Audio")
        source_video_path = sidecar.get("source_video_path")
        if not isinstance(source_video_path, str) or not source_video_path.strip():
            raise ValueError("production Audio sidecar lacks source video provenance")
        video = Path(source_video_path).expanduser().resolve(strict=True)
        if not video.is_file():
            raise ValueError("production Audio source video is unavailable")
        output[clip_uid] = _sha256(video)
    return output


def _directional_ranks(
    vectors: Mapping[str, np.ndarray],
    clip_by_id: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for target in sorted(vectors):
        donors = [
            donor
            for donor in vectors
            if donor != target and clip_by_id[donor] != clip_by_id[target]
        ]
        donors.sort(key=lambda donor: (-float(np.dot(vectors[target], vectors[donor])), donor))
        output[target] = {
            donor: index for index, donor in enumerate(donors, start=1)
        }
    return output


def build_h3_production_pairs(
    *,
    inventory: H3ProductionInventory,
    audio_root: Path,
    primary_voice_root: Path,
    embedding_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> H3ProductionSummary:
    audio = audio_root.expanduser().resolve(strict=True)
    primary = primary_voice_root.expanduser().resolve(strict=True)
    embedding = embedding_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    audio_summary = H3AudioBindingPilotSummary.model_validate_json(
        (audio / "summary.json").read_text(encoding="utf-8")
    )
    primary_summary = PrimaryVoiceReferenceExportSummary.model_validate_json(
        (primary / "summary.json").read_text(encoding="utf-8")
    )
    embedding_summary = H3ProductionEmbeddingSummary.model_validate_json(
        (embedding / "summary.json").read_text(encoding="utf-8")
    )
    if audio_summary.clips_attempted != inventory.eligible_clip_count:
        raise ValueError("production Audio output does not cover every eligible clip")
    if embedding_summary.input_occurrence_count != inventory.eligible_occurrence_count:
        raise ValueError("production embedding output is not occurrence-complete")
    policy = VoiceReferenceQualityPolicy()
    if (
        primary_summary.policy_version != policy.version
        or primary_summary.policy_fingerprint != policy.fingerprint()
        or not primary_summary.thresholds_calibrated
    ):
        raise ValueError("production primary voice output does not use frozen V1 policy")
    rows = _load_embedding_rows(embedding)
    row_ids = {item.entity_occurrence_id for item in rows}
    if row_ids != {item.entity_occurrence_id for item in inventory.occurrences}:
        raise ValueError("production embedding occurrence set differs from inventory")
    by_id = {item.entity_occurrence_id: item for item in rows}
    voice_rows = [item for item in rows if item.primary_voice_reference_path is not None]
    in_pairs = [
        H3ProductionInPair(
            pair_id=f"in_pair/{item.entity_occurrence_id}",
            target_occurrence_id=item.entity_occurrence_id,
            clip_uid=item.clip_uid,
            entity_id=item.entity_id,
            visual_reference_path=item.visual_reference_path,
            primary_voice_reference_path=str(item.primary_voice_reference_path),
        )
        for item in voice_rows
    ]
    complete = {
        item.entity_occurrence_id: item
        for item in rows
        if item.face.status == "available" and item.speaker.status == "available"
    }
    face_vectors = {
        occurrence_id: _load_embedding_vector(embedding, item.face)
        for occurrence_id, item in complete.items()
    }
    voice_vectors = {
        occurrence_id: _load_embedding_vector(embedding, item.speaker)
        for occurrence_id, item in complete.items()
    }
    clip_by_id = {
        occurrence_id: item.clip_uid for occurrence_id, item in complete.items()
    }
    face_ranks = _directional_ranks(face_vectors, clip_by_id)
    voice_ranks = _directional_ranks(voice_vectors, clip_by_id)
    source_hash = _source_video_hashes(
        audio,
        {item.clip_uid for item in voice_rows},
    )
    config = AudioPairingConfig()
    evidence_rows: list[H3ProductionPairEvidence] = []
    eligible_by_target: dict[str, list[tuple[str, PairEvidence]]] = {
        occurrence_id: [] for occurrence_id in complete
    }
    rejection_counts: Counter[str] = Counter()
    for target_id in sorted(complete):
        for donor_id in sorted(complete):
            if donor_id == target_id or clip_by_id[donor_id] == clip_by_id[target_id]:
                continue
            face_similarity = float(np.dot(face_vectors[target_id], face_vectors[donor_id]))
            voice_similarity = float(
                np.dot(voice_vectors[target_id], voice_vectors[donor_id])
            )
            edge = evaluate_pair_policy_v1(
                target_entity_occurrence_id=target_id,
                reference_entity_occurrence_id=donor_id,
                face_similarity=face_similarity,
                voice_similarity=voice_similarity,
                config=config,
                face_rank_left_to_right=face_ranks[target_id][donor_id],
                face_rank_right_to_left=face_ranks[donor_id][target_id],
                voice_rank_left_to_right=voice_ranks[target_id][donor_id],
                voice_rank_right_to_left=voice_ranks[donor_id][target_id],
            )
            duplicate = source_hash[clip_by_id[target_id]] == source_hash[
                clip_by_id[donor_id]
            ]
            reasons = [
                code
                for part in (edge.same_person, edge.same_voice)
                if part.status != "accepted"
                for code in part.reason_codes
            ]
            if duplicate:
                reasons.append("duplicate_source_video")
            reasons = list(dict.fromkeys(reasons))
            rejection_counts.update(reasons)
            if not reasons:
                eligible_by_target[target_id].append((donor_id, edge))
            evidence_rows.append(
                H3ProductionPairEvidence(
                    target_occurrence_id=target_id,
                    donor_occurrence_id=donor_id,
                    target_clip_uid=clip_by_id[target_id],
                    donor_clip_uid=clip_by_id[donor_id],
                    pair_evidence=edge,
                    duplicate_source=duplicate,
                    donor_eligible=not reasons,
                    rejection_reason_codes=reasons,
                )
            )
    cross_pairs: list[H3ProductionCrossPair] = []
    selected_endpoints: set[tuple[str, str]] = set()
    for target_id in sorted(complete):
        candidates = sorted(
            eligible_by_target[target_id],
            key=lambda item: (
                -item[1].same_person.face_similarity,
                item[0],
            ),
        )
        if not candidates:
            continue
        donor_id, edge = candidates[0]
        target = by_id[target_id]
        donor = by_id[donor_id]
        assert target.primary_voice_reference_path is not None
        cross_pairs.append(
            H3ProductionCrossPair(
                pair_id=f"cross_pair/{target_id}/1",
                target_occurrence_id=target_id,
                donor_occurrence_id=donor_id,
                target_clip_uid=target.clip_uid,
                donor_clip_uid=donor.clip_uid,
                target_primary_voice_reference_path=(
                    target.primary_voice_reference_path
                ),
                donor_visual_reference_path=donor.visual_reference_path,
                face_similarity=edge.same_person.face_similarity,
                voice_similarity=edge.same_voice.voice_similarity,
            )
        )
        selected_endpoints.add((target_id, donor_id))
    evidence_rows = [
        item.model_copy(
            update={
                "selected": (
                    item.target_occurrence_id,
                    item.donor_occurrence_id,
                )
                in selected_endpoints
            }
        )
        for item in evidence_rows
    ]
    summary = H3ProductionSummary(
        complete_eligible_occurrence_count=inventory.eligible_occurrence_count,
        audio_clips_attempted=audio_summary.clips_attempted,
        audio_clips_succeeded=audio_summary.clips_succeeded,
        audio_clips_failed=audio_summary.clips_failed,
        primary_voice_available_count=embedding_summary.primary_voice_available_count,
        primary_voice_unavailable_count=(
            embedding_summary.primary_voice_unavailable_count
        ),
        face_embedding_available_count=(
            embedding_summary.face_embedding_available_count
        ),
        face_embedding_unavailable_count=(
            embedding_summary.face_embedding_unavailable_count
        ),
        speaker_embedding_available_count=(
            embedding_summary.speaker_embedding_available_count
        ),
        speaker_embedding_unavailable_count=(
            embedding_summary.speaker_embedding_unavailable_count
        ),
        in_pair_count=len(in_pairs),
        cross_pair_eligible_target_count=len(complete),
        cross_pair_count=len(cross_pairs),
        target_without_cross_pair_count=len(complete) - len(cross_pairs),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "in_pairs.jsonl").write_text(_jsonl(in_pairs), encoding="utf-8")
        (temporary / "cross_pairs.jsonl").write_text(
            _jsonl(cross_pairs), encoding="utf-8"
        )
        (temporary / "pair_evidence.jsonl").write_text(
            _jsonl(evidence_rows), encoding="utf-8"
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


_STAGE_REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "audio": (
        "summary.json",
        "audio_bindings.jsonl",
        "failures.jsonl",
        "voice_reference_quality_summary.json",
    ),
    "primary-voice": (
        "summary.json",
        "primary_voice_references.jsonl",
        "voice_quality_assessments.jsonl",
    ),
    "embedding": ("summary.json", "occurrences.jsonl"),
    "pair": (
        "summary.json",
        "in_pairs.jsonl",
        "cross_pairs.jsonl",
        "pair_evidence.jsonl",
    ),
}


def stage_is_complete(stage: str, root: Path) -> bool:
    return root.is_dir() and all(
        (root / relative).is_file() for relative in _STAGE_REQUIRED_FILES[stage]
    )


def parse_production_stages(value: str) -> tuple[str, ...]:
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if requested == ["all"]:
        return PRODUCTION_STAGE_ORDER
    if not requested or any(item not in PRODUCTION_STAGE_ORDER for item in requested):
        raise ValueError("production stages must be audio, primary-voice, embedding, pair, or all")
    requested_set = set(requested)
    return tuple(stage for stage in PRODUCTION_STAGE_ORDER if stage in requested_set)


def orchestrate_production_stages(
    *,
    stages: Sequence[str],
    roots: Mapping[str, Path],
    runners: Mapping[str, Callable[[bool], object]],
    overwrite: bool = False,
) -> dict[str, str]:
    requested = tuple(stages)
    if any(stage not in PRODUCTION_STAGE_ORDER for stage in requested):
        raise ValueError("unknown H3 production stage")
    if overwrite:
        requested_set = set(requested)
        for stage in requested:
            stage_index = PRODUCTION_STAGE_ORDER.index(stage)
            stale_downstream = [
                downstream
                for downstream in PRODUCTION_STAGE_ORDER[stage_index + 1 :]
                if downstream not in requested_set
                and stage_is_complete(downstream, roots[downstream])
            ]
            if stale_downstream:
                raise ValueError(
                    f"overwriting production stage {stage} requires also overwriting "
                    f"completed downstream stages: {stale_downstream}"
                )
    status: dict[str, str] = {}
    for stage in PRODUCTION_STAGE_ORDER:
        if stage not in requested:
            continue
        stage_index = PRODUCTION_STAGE_ORDER.index(stage)
        for prerequisite in PRODUCTION_STAGE_ORDER[:stage_index]:
            if not stage_is_complete(prerequisite, roots[prerequisite]):
                raise ValueError(
                    f"production stage {stage} requires completed {prerequisite}"
                )
        if stage_is_complete(stage, roots[stage]) and not overwrite:
            status[stage] = "reused"
            continue
        runners[stage](overwrite)
        if not stage_is_complete(stage, roots[stage]):
            raise RuntimeError(f"production stage {stage} did not publish complete output")
        status[stage] = "completed"
    return status


def run_primary_voice_production_stage(
    *,
    audio_root: Path,
    output_root: Path,
    audio_backend: AudioMediaBackend,
    overwrite: bool = False,
) -> PrimaryVoiceReferenceExportSummary:
    return export_primary_voice_references(
        pilot_root=audio_root,
        output_root=output_root,
        audio_backend=audio_backend,
        policy=VoiceReferenceQualityPolicy(),
        overwrite=overwrite,
    )


def atomic_replace_stage(
    destination: Path,
    *,
    overwrite: bool,
    runner: Callable[[], object],
) -> object:
    if not destination.exists() or not overwrite:
        return runner()
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    succeeded = False
    try:
        result = runner()
        succeeded = True
        return result
    finally:
        if succeeded:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)
