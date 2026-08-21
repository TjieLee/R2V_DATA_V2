from __future__ import annotations

import html
import json
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_pairing import (
    H3_PAIR_POLICY_FACE_THRESHOLD,
    H3_PAIR_POLICY_VERSION,
    H3_PAIR_POLICY_VOICE_THRESHOLD,
)
from r2v_data_v2.h3.schemas import SchemaModel
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
    subject_index: int,
) -> Path:
    if subject_index <= 0:
        raise ValueError("subject_index must be positive")
    return root / Path(*_display_path(identity).parts) / f"e{subject_index}.flac"


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


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0:
        raise ValueError("pair embedding must be finite and non-zero")
    return vector / norm


def _complete_matching(
    candidate_sets: Sequence[Sequence[tuple[JEAOccurrenceEmbedding, float, float]]],
) -> list[tuple[JEAOccurrenceEmbedding, float, float]] | None:
    selected: list[tuple[JEAOccurrenceEmbedding, float, float]] = []
    used: set[str] = set()

    def visit(index: int) -> bool:
        if index == len(candidate_sets):
            return True
        for candidate in candidate_sets[index]:
            donor = candidate[0]
            if donor.occurrence_id in used:
                continue
            used.add(donor.occurrence_id)
            selected.append(candidate)
            if visit(index + 1):
                return True
            selected.pop()
            used.remove(donor.occurrence_id)
        return False

    return list(selected) if visit(0) else None


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
    eligible: dict[str, list[tuple[JEAOccurrenceEmbedding, float, float]]] = (
        defaultdict(list)
    )
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
            if not reasons:
                eligible[target.occurrence_id].append(
                    (donor, face_similarity, voice_similarity)
                )
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
            key=lambda item: (
                -item[1],
                item[0].identity.clip_display_path,
                item[0].occurrence_id,
            )
        )

    cross_pairs: list[JEACrossPair] = []
    selected_edges: set[tuple[str, str]] = set()
    for pair in in_pairs:
        targets = by_clip[pair.target_clip_uid]
        matching = _complete_matching(
            [eligible[item.occurrence_id] for item in targets]
        )
        if matching is None:
            continue
        mappings: list[JEACrossPairMapping] = []
        for index, (donor, face_similarity, voice_similarity) in enumerate(
            matching, start=1
        ):
            target = targets[index - 1]
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
