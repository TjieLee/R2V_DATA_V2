from __future__ import annotations

import json
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_backends import FaceEmbeddingBackend
from r2v_data_v2.h3.embedding_pilot import (
    EmbeddingPilotModalityResult,
    materialize_face_embedding,
)
from r2v_data_v2.h3.pair_calibration import (
    EligibleVisualFaceOccurrence,
    load_eligible_visual_face_occurrences,
)
from r2v_data_v2.h3.schemas import SchemaModel

FACE_MINING_SCHEMA_VERSION = "r2v.h3.face_identity_mining.1"
FACE_MINING_CANDIDATE_SCHEMA_VERSION = "r2v.h3.face_identity_candidate.1"


class FaceMiningOccurrence(SchemaModel):
    schema_version: Literal["r2v.h3.face_identity_mining.1"] = (
        FACE_MINING_SCHEMA_VERSION
    )
    entity_occurrence_id: str
    clip_uid: str
    entity_id: str
    parent_video_id: str
    clip_suffix: str
    canonical_reference_path: str
    canonical_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_reference_run_path: str
    existing_v3_cross_pair_provenance: dict[str, str] | None = None
    visual_integrity_provenance: dict[str, object]
    face: EmbeddingPilotModalityResult
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> FaceMiningOccurrence:
        if self.entity_occurrence_id != f"{self.clip_uid}/{self.entity_id}":
            raise ValueError("face mining occurrence identity is inconsistent")
        if not self.parent_video_id.strip() or not self.clip_suffix.strip():
            raise ValueError("face mining occurrence provenance must not be empty")
        return self


class FaceMiningPair(SchemaModel):
    left_occurrence_id: str
    right_occurrence_id: str
    left_parent_video_id: str
    right_parent_video_id: str
    left_clip_suffix: str
    right_clip_suffix: str
    same_parent: bool
    face_similarity: float = Field(ge=-1, le=1)
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_pair(self) -> FaceMiningPair:
        if self.left_occurrence_id >= self.right_occurrence_id:
            raise ValueError("face mining pair endpoints must be canonical")
        if self.same_parent != (
            self.left_parent_video_id == self.right_parent_video_id
        ):
            raise ValueError("face mining same-parent flag is inconsistent")
        return self


class FaceIdentityCandidate(SchemaModel):
    schema_version: Literal["r2v.h3.face_identity_candidate.1"] = (
        FACE_MINING_CANDIDATE_SCHEMA_VERSION
    )
    left_occurrence_id: str
    right_occurrence_id: str
    candidate_pool: Literal["same_parent", "global_different_parent"]
    face_similarity: float = Field(ge=-1, le=1)
    left_to_right_rank: int = Field(gt=0)
    right_to_left_rank: int = Field(gt=0)
    mutual_top_k: bool
    top_k: int = Field(gt=0)
    parent_video_id: str | None = None
    left_parent_video_id: str
    right_parent_video_id: str
    left_clip_suffix: str
    right_clip_suffix: str
    same_person_label: None = None
    thresholds_calibrated: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate(self) -> FaceIdentityCandidate:
        if self.left_occurrence_id >= self.right_occurrence_id:
            raise ValueError("face candidate endpoints must be canonical")
        same_parent = self.left_parent_video_id == self.right_parent_video_id
        if self.candidate_pool == "same_parent":
            if not same_parent or self.parent_video_id != self.left_parent_video_id:
                raise ValueError("same-parent candidate provenance is inconsistent")
        elif same_parent or self.parent_video_id is not None:
            raise ValueError("global diagnostic candidate must cross parents")
        return self


class FaceMiningSummary(SchemaModel):
    schema_version: Literal["r2v.h3.face_identity_mining_summary.1"] = (
        "r2v.h3.face_identity_mining_summary.1"
    )
    source_run_root: str
    occurrence_count: int = Field(ge=0)
    face_embedding_available_count: int = Field(ge=0)
    face_embedding_unavailable_count: int = Field(ge=0)
    face_embedding_failed_count: int = Field(ge=0)
    face_pair_count: int = Field(ge=0)
    same_parent_candidate_count: int = Field(ge=0)
    global_diagnostic_candidate_count: int = Field(ge=0)
    face_failure_reason_counts: dict[str, int]
    source_skip_reason_counts: dict[str, int]
    face_model_identifier: str
    face_model_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    top_k: int = Field(gt=0)
    thresholds_calibrated: Literal[False] = False


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


def _directional_ranks(
    *,
    vectors: dict[str, np.ndarray],
    occurrence_by_id: dict[str, EligibleVisualFaceOccurrence],
    same_parent: bool,
) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = {}
    for anchor in sorted(vectors):
        anchor_parent = occurrence_by_id[anchor].parent_video_id
        candidates = [
            candidate
            for candidate in vectors
            if candidate != anchor
            and (
                occurrence_by_id[candidate].parent_video_id == anchor_parent
            )
            == same_parent
        ]
        candidates.sort(
            key=lambda candidate: (
                -float(np.dot(vectors[anchor], vectors[candidate])),
                candidate,
            )
        )
        ranks[anchor] = {
            candidate: index
            for index, candidate in enumerate(candidates, start=1)
        }
    return ranks


def _candidate_records(
    *,
    vectors: dict[str, np.ndarray],
    occurrence_by_id: dict[str, EligibleVisualFaceOccurrence],
    top_k: int,
    same_parent: bool,
) -> list[FaceIdentityCandidate]:
    ranks = _directional_ranks(
        vectors=vectors,
        occurrence_by_id=occurrence_by_id,
        same_parent=same_parent,
    )
    pairs: set[tuple[str, str]] = set()
    for anchor, directional in ranks.items():
        for candidate, rank in directional.items():
            if rank <= top_k:
                pairs.add(tuple(sorted((anchor, candidate))))
    pool: Literal["same_parent", "global_different_parent"] = (
        "same_parent" if same_parent else "global_different_parent"
    )
    output: list[FaceIdentityCandidate] = []
    for left, right in sorted(pairs):
        left_item = occurrence_by_id[left]
        right_item = occurrence_by_id[right]
        left_rank = ranks[left][right]
        right_rank = ranks[right][left]
        output.append(
            FaceIdentityCandidate(
                left_occurrence_id=left,
                right_occurrence_id=right,
                candidate_pool=pool,
                face_similarity=float(np.dot(vectors[left], vectors[right])),
                left_to_right_rank=left_rank,
                right_to_left_rank=right_rank,
                mutual_top_k=left_rank <= top_k and right_rank <= top_k,
                top_k=top_k,
                parent_video_id=(left_item.parent_video_id if same_parent else None),
                left_parent_video_id=left_item.parent_video_id,
                right_parent_video_id=right_item.parent_video_id,
                left_clip_suffix=left_item.clip_suffix,
                right_clip_suffix=right_item.clip_suffix,
            )
        )
    output.sort(
        key=lambda item: (
            item.candidate_pool != "same_parent",
            not item.mutual_top_k,
            min(item.left_to_right_rank, item.right_to_left_rank),
            max(item.left_to_right_rank, item.right_to_left_rank),
            -item.face_similarity,
            item.left_occurrence_id,
            item.right_occurrence_id,
        )
    )
    return output


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"face mining output already exists: {destination}")
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


def mine_face_identity_candidates(
    *,
    run_root: Path,
    output_root: Path,
    face_backend: FaceEmbeddingBackend,
    top_k: int = 5,
    overwrite: bool = False,
) -> FaceMiningSummary:
    if top_k <= 0:
        raise ValueError("face mining top_k must be positive")
    loaded = load_eligible_visual_face_occurrences(run_root)
    destination = output_root.expanduser().resolve(strict=False)
    source = loaded.run_root
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("face mining output must be separate from source run")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    occurrence_by_id = {
        item.entity_occurrence_id: item for item in loaded.occurrences
    }
    face_vectors: dict[str, np.ndarray] = {}
    occurrence_rows: list[FaceMiningOccurrence] = []
    try:
        temporary.mkdir()
        for item in loaded.occurrences:
            materialized = materialize_face_embedding(
                backend=face_backend,
                entity_occurrence_id=item.entity_occurrence_id,
                clip_uid=item.clip_uid,
                entity_id=item.entity_id,
                canonical_reference_path=item.canonical_reference_path,
                canonical_reference_sha256=item.canonical_reference_sha256,
                output_root=temporary,
            )
            if materialized.normalized_vector is not None:
                face_vectors[item.entity_occurrence_id] = materialized.normalized_vector
            occurrence_rows.append(
                FaceMiningOccurrence(
                    entity_occurrence_id=item.entity_occurrence_id,
                    clip_uid=item.clip_uid,
                    entity_id=item.entity_id,
                    parent_video_id=item.parent_video_id,
                    clip_suffix=item.clip_suffix,
                    canonical_reference_path=str(item.canonical_reference_path),
                    canonical_reference_sha256=item.canonical_reference_sha256,
                    canonical_reference_run_path=item.canonical_reference_run_path,
                    existing_v3_cross_pair_provenance=(
                        item.existing_v3_cross_pair_provenance
                    ),
                    visual_integrity_provenance=item.visual_integrity_provenance,
                    face=materialized.result,
                )
            )

        pair_rows = [
            FaceMiningPair(
                left_occurrence_id=left,
                right_occurrence_id=right,
                left_parent_video_id=occurrence_by_id[left].parent_video_id,
                right_parent_video_id=occurrence_by_id[right].parent_video_id,
                left_clip_suffix=occurrence_by_id[left].clip_suffix,
                right_clip_suffix=occurrence_by_id[right].clip_suffix,
                same_parent=(
                    occurrence_by_id[left].parent_video_id
                    == occurrence_by_id[right].parent_video_id
                ),
                face_similarity=float(np.dot(face_vectors[left], face_vectors[right])),
            )
            for left, right in combinations(sorted(face_vectors), 2)
        ]
        candidates = _candidate_records(
            vectors=face_vectors,
            occurrence_by_id=occurrence_by_id,
            top_k=top_k,
            same_parent=True,
        ) + _candidate_records(
            vectors=face_vectors,
            occurrence_by_id=occurrence_by_id,
            top_k=top_k,
            same_parent=False,
        )
        failures = Counter(
            item.face.failure_reason
            for item in occurrence_rows
            if item.face.failure_reason is not None
        )
        first_result = next(iter(occurrence_rows), None)
        model_identifier = (
            first_result.face.model_identifier
            if first_result is not None
            else str(getattr(face_backend, "model_identifier", type(face_backend).__name__))
        )
        model_fingerprint = (
            first_result.face.model_fingerprint
            if first_result is not None
            else getattr(face_backend, "checkpoint_sha256", None)
        )
        summary = FaceMiningSummary(
            source_run_root=str(source),
            occurrence_count=len(occurrence_rows),
            face_embedding_available_count=sum(
                item.face.status == "available" for item in occurrence_rows
            ),
            face_embedding_unavailable_count=sum(
                item.face.status == "unavailable" for item in occurrence_rows
            ),
            face_embedding_failed_count=sum(
                item.face.status == "failed" for item in occurrence_rows
            ),
            face_pair_count=len(pair_rows),
            same_parent_candidate_count=sum(
                item.candidate_pool == "same_parent" for item in candidates
            ),
            global_diagnostic_candidate_count=sum(
                item.candidate_pool == "global_different_parent"
                for item in candidates
            ),
            face_failure_reason_counts=dict(sorted(failures.items())),
            source_skip_reason_counts=loaded.skip_reason_counts,
            face_model_identifier=model_identifier,
            face_model_fingerprint=(
                None if model_fingerprint is None else str(model_fingerprint)
            ),
            top_k=top_k,
        )
        (temporary / "occurrences.jsonl").write_text(
            _jsonl(occurrence_rows), encoding="utf-8"
        )
        (temporary / "face_pairs.jsonl").write_text(
            _jsonl(pair_rows), encoding="utf-8"
        )
        (temporary / "face_candidates.jsonl").write_text(
            _jsonl(candidates), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(
                summary.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
