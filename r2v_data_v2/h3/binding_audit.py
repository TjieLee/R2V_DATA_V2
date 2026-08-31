from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_binding import AudioBindingProductionConfig
from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationClusterBinding,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.pilot_schemas import LRASDNativeArtifact
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel

BINDING_AUDIT_VERSION = "r2v.h3.speaker_binding_audit.1"
BINDING_AUDIT_POLICY_VERSION = "h3_speaker_binding_structural_audit_v1"

ReviewPriority = Literal[
    "conflict",
    "exclusive_active_entity_contradiction",
    "mapped_entity_vs_unmatched_face_contradiction",
    "unbound_or_ambiguous",
    "fully_propagated",
    "candidate_mapped_lowest_direct_support_ratio",
]


class ExclusiveActiveFaceOverlap(SchemaModel):
    face_track_id: str
    entity_id: str | None = None
    overlap_seconds: float = Field(gt=0)


class ExclusiveActiveEntityOverlap(SchemaModel):
    entity_id: str
    overlap_seconds: float = Field(gt=0)


class SpeakerBindingSegmentAuditFlags(SchemaModel):
    conflict: bool
    ambiguous: bool
    unbound: bool
    fully_propagated_segment: bool
    multiple_direct_entity_support: bool
    contested_anchor: bool
    multiple_exclusive_active_face_tracks: bool
    exclusive_active_entity_contradiction: bool
    mapped_entity_vs_unmatched_face_contradiction: bool


class SpeakerBindingClusterAuditFlags(SchemaModel):
    conflict: bool
    ambiguous: bool
    unbound: bool
    has_fully_propagated_segments: bool
    multiple_direct_entity_support: bool
    contested_anchor: bool
    multiple_exclusive_active_face_tracks: bool
    exclusive_active_entity_contradiction: bool
    mapped_entity_vs_unmatched_face_contradiction: bool


class SpeakerBindingSegmentAudit(SchemaModel):
    schema_version: Literal["r2v.h3.speaker_binding_audit.1"] = (
        BINDING_AUDIT_VERSION
    )
    clip_uid: str
    segment_id: str
    speaker_cluster_id: str
    current_mapping_status: Literal[
        "candidate_mapped", "unbound", "ambiguous", "conflict"
    ]
    current_entity_id: str | None = None
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    speaker_duration_seconds: float = Field(gt=0)
    direct_anchor_seconds: float = Field(ge=0)
    direct_support_seconds_by_entity: dict[str, float]
    directly_supported_entity_count: int = Field(ge=0)
    contested_anchor_seconds: float = Field(ge=0)
    identity_propagated_seconds: float = Field(ge=0)
    exclusive_active_faces: list[ExclusiveActiveFaceOverlap]
    exclusive_active_entities: list[ExclusiveActiveEntityOverlap]
    unmatched_exclusive_active_face_track_ids: list[str]
    flags: SpeakerBindingSegmentAuditFlags

    @model_validator(mode="after")
    def validate_duration(self) -> SpeakerBindingSegmentAudit:
        if not math.isclose(
            self.speaker_duration_seconds,
            self.end_time - self.start_time,
            abs_tol=1e-9,
        ):
            raise ValueError("segment audit duration is inconsistent")
        if self.current_mapping_status == "candidate_mapped":
            if not math.isclose(
                self.speaker_duration_seconds,
                self.direct_anchor_seconds + self.identity_propagated_seconds,
                abs_tol=1e-9,
            ):
                raise ValueError("mapped segment anchor durations do not reconcile")
        elif self.identity_propagated_seconds != 0:
            raise ValueError("unresolved segment cannot claim identity propagation")
        if self.directly_supported_entity_count != len(
            self.direct_support_seconds_by_entity
        ):
            raise ValueError("segment supported-entity count is inconsistent")
        return self


class SpeakerBindingClusterAudit(SchemaModel):
    schema_version: Literal["r2v.h3.speaker_binding_audit.1"] = (
        BINDING_AUDIT_VERSION
    )
    clip_uid: str
    speaker_cluster_id: str
    current_mapping_status: Literal[
        "candidate_mapped", "unbound", "ambiguous", "conflict"
    ]
    current_entity_id: str | None = None
    cluster_segment_count: int = Field(gt=0)
    cluster_speaker_duration_seconds: float = Field(gt=0)
    direct_anchor_seconds: float = Field(ge=0)
    identity_propagated_seconds: float = Field(ge=0)
    fully_propagated_seconds: float = Field(ge=0)
    contributing_direct_binding_count: int = Field(ge=0)
    direct_support_seconds_by_entity: dict[str, float]
    directly_supported_entity_count: int = Field(ge=0)
    competing_direct_entity_support: dict[str, float]
    direct_support_ratio: float | None = Field(default=None, ge=0, le=1)
    longest_direct_anchor_seconds: float | None = Field(default=None, gt=0)
    shortest_direct_anchor_seconds: float | None = Field(default=None, gt=0)
    contested_anchor_seconds: float = Field(ge=0)
    exclusive_active_faces: list[ExclusiveActiveFaceOverlap]
    exclusive_active_entities: list[ExclusiveActiveEntityOverlap]
    unmatched_exclusive_active_face_track_ids: list[str]
    flags: SpeakerBindingClusterAuditFlags
    review_priority: ReviewPriority
    review_priority_rank: int = Field(ge=1, le=6)
    audit_policy_version: Literal["h3_speaker_binding_structural_audit_v1"] = (
        BINDING_AUDIT_POLICY_VERSION
    )

    @model_validator(mode="after")
    def validate_metrics(self) -> SpeakerBindingClusterAudit:
        if self.directly_supported_entity_count != len(
            self.direct_support_seconds_by_entity
        ):
            raise ValueError("cluster supported-entity count is inconsistent")
        has_anchor_lengths = self.longest_direct_anchor_seconds is not None
        has_shortest_anchor = self.shortest_direct_anchor_seconds is not None
        if has_anchor_lengths != (self.contributing_direct_binding_count > 0) or (
            has_shortest_anchor != has_anchor_lengths
        ):
            raise ValueError("cluster contributing-binding evidence is inconsistent")
        if has_anchor_lengths and (
            self.shortest_direct_anchor_seconds is None
            or self.longest_direct_anchor_seconds < self.shortest_direct_anchor_seconds
        ):
            raise ValueError("cluster direct-anchor length bounds are inconsistent")
        if self.current_mapping_status == "candidate_mapped":
            if not math.isclose(
                self.cluster_speaker_duration_seconds,
                self.direct_anchor_seconds + self.identity_propagated_seconds,
                abs_tol=1e-9,
            ):
                raise ValueError("mapped cluster anchor durations do not reconcile")
        elif self.identity_propagated_seconds != 0:
            raise ValueError("unresolved cluster cannot claim identity propagation")
        return self

class SpeakerBindingReviewCase(SchemaModel):
    schema_version: Literal["r2v.h3.speaker_binding_audit.1"] = (
        BINDING_AUDIT_VERSION
    )
    review_index: int = Field(gt=0)
    review_priority: ReviewPriority
    review_priority_rank: int = Field(ge=1, le=6)
    clip_uid: str
    speaker_cluster_id: str
    current_mapping_status: str
    current_entity_id: str | None = None
    reason_flags: list[str]
    direct_support_ratio: float | None = None


class SpeakerBindingAuditSummary(SchemaModel):
    schema_version: Literal["r2v.h3.speaker_binding_audit.1"] = (
        BINDING_AUDIT_VERSION
    )
    audit_policy_version: Literal["h3_speaker_binding_structural_audit_v1"] = (
        BINDING_AUDIT_POLICY_VERSION
    )
    source_audio_production_root: str
    source_diarization_root: str
    source_artifact_sha256: dict[str, str]
    audio_binding_input_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lr_asd_native_input_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clip_count: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    candidate_mapped_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    ambiguous_count: int = Field(ge=0)
    unbound_count: int = Field(ge=0)
    clusters_with_multiple_exclusive_active_face_tracks: int = Field(ge=0)
    clusters_with_exclusive_active_entity_contradiction: int = Field(ge=0)
    clusters_with_mapped_entity_vs_unmatched_face: int = Field(ge=0)
    clusters_with_multiple_direct_entity_support: int = Field(ge=0)
    clusters_with_fully_propagated_segments: int = Field(ge=0)
    multiple_exclusive_active_face_track_speaker_seconds: float = Field(ge=0)
    exclusive_active_entity_contradiction_speaker_seconds: float = Field(ge=0)
    mapped_entity_vs_unmatched_face_speaker_seconds: float = Field(ge=0)
    fully_propagated_segment_speaker_seconds: float = Field(ge=0)
    identity_propagated_speaker_seconds: float = Field(ge=0)
    review_priority_counts: dict[ReviewPriority, int]
    model_call_count: Literal[0] = 0
    bindings_modified_count: Literal[0] = 0

    @model_validator(mode="after")
    def validate_counts(self) -> SpeakerBindingAuditSummary:
        if self.cluster_count != (
            self.candidate_mapped_count
            + self.conflict_count
            + self.ambiguous_count
            + self.unbound_count
        ):
            raise ValueError("binding audit status counts do not reconcile")
        if sum(self.review_priority_counts.values()) != self.cluster_count:
            raise ValueError("binding audit review-priority counts do not reconcile")
        return self


class _Span:
    __slots__ = ("end", "start")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class _ExclusiveFrame:
    __slots__ = ("end_time", "entity_id", "face_track_id", "start_time")

    def __init__(
        self,
        *,
        face_track_id: str,
        entity_id: str | None,
        start_time: float,
        end_time: float,
    ) -> None:
        self.face_track_id = face_track_id
        self.entity_id = entity_id
        self.start_time = start_time
        self.end_time = end_time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(path: Path, model: type[SchemaModel]) -> list[SchemaModel]:
    rows = [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def _write_json(path: Path, value: SchemaModel) -> None:
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )


def _union_samples(spans: Iterable[_Span]) -> int:
    ordered = sorted((span.start, span.end) for span in spans if span.start < span.end)
    if not ordered:
        return 0
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _overlap_seconds(
    intervals: Sequence[tuple[float, float]],
    frames: Sequence[_ExclusiveFrame],
    *,
    face_track_id: str | None = None,
    entity_id: str | None = None,
) -> float:
    spans: list[tuple[float, float]] = []
    for frame in frames:
        if face_track_id is not None and frame.face_track_id != face_track_id:
            continue
        if entity_id is not None and frame.entity_id != entity_id:
            continue
        for start, end in intervals:
            overlap_start = max(start, frame.start_time)
            overlap_end = min(end, frame.end_time)
            if overlap_start < overlap_end:
                spans.append((overlap_start, overlap_end))
    if not spans:
        return 0.0
    spans.sort()
    start, end = spans[0]
    total = 0.0
    for next_start, next_end in spans[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _exclusive_frames(
    native: LRASDNativeArtifact,
    sidecar: AudioBindingSidecar,
) -> list[_ExclusiveFrame]:
    association_by_track = {
        item.face_track_id: item for item in sidecar.evidence.associations
    }
    active_by_frame: dict[int, list[str]] = defaultdict(list)
    for track in native.tracks:
        for sample in track.samples:
            if sample.backend_native_active:
                active_by_frame[sample.frame_index].append(track.face_track_id)
    output: list[_ExclusiveFrame] = []
    for frame_index, track_ids in sorted(active_by_frame.items()):
        if len(track_ids) != 1:
            continue
        track_id = track_ids[0]
        association = association_by_track.get(track_id)
        entity_id = (
            association.entity_id
            if association is not None and association.status == "matched"
            else None
        )
        start = frame_index / native.model_fps
        end = min((frame_index + 1) / native.model_fps, native.duration_seconds)
        if start < end:
            output.append(
                _ExclusiveFrame(
                    face_track_id=track_id,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                )
            )
    return output


def _exclusive_overlap_evidence(
    intervals: Sequence[tuple[float, float]],
    frames: Sequence[_ExclusiveFrame],
) -> tuple[
    list[ExclusiveActiveFaceOverlap],
    list[ExclusiveActiveEntityOverlap],
    list[str],
]:
    track_ids = sorted(
        {
            frame.face_track_id
            for frame in frames
            if _overlap_seconds(intervals, [frame]) > 0
        }
    )
    association = {
        frame.face_track_id: frame.entity_id
        for frame in frames
        if frame.face_track_id in track_ids
    }
    faces = [
        ExclusiveActiveFaceOverlap(
            face_track_id=track_id,
            entity_id=association[track_id],
            overlap_seconds=_overlap_seconds(
                intervals,
                frames,
                face_track_id=track_id,
            ),
        )
        for track_id in track_ids
    ]
    entity_ids = sorted({item.entity_id for item in faces if item.entity_id is not None})
    entities = [
        ExclusiveActiveEntityOverlap(
            entity_id=entity_id,
            overlap_seconds=_overlap_seconds(
                intervals,
                frames,
                entity_id=entity_id,
            ),
        )
        for entity_id in entity_ids
    ]
    unmatched = [item.face_track_id for item in faces if item.entity_id is None]
    return faces, entities, unmatched


def _segment_flags(
    *,
    status: str,
    current_entity_id: str | None,
    direct_entity_count: int,
    contested_anchor_seconds: float,
    fully_propagated: bool,
    faces: Sequence[ExclusiveActiveFaceOverlap],
    entities: Sequence[ExclusiveActiveEntityOverlap],
    unmatched: Sequence[str],
) -> SpeakerBindingSegmentAuditFlags:
    return SpeakerBindingSegmentAuditFlags(
        conflict=status == "conflict",
        ambiguous=status == "ambiguous",
        unbound=status == "unbound",
        fully_propagated_segment=fully_propagated,
        multiple_direct_entity_support=direct_entity_count > 1,
        contested_anchor=contested_anchor_seconds > 0,
        multiple_exclusive_active_face_tracks=len(faces) > 1,
        exclusive_active_entity_contradiction=len(entities) > 1,
        mapped_entity_vs_unmatched_face_contradiction=(
            status == "candidate_mapped"
            and current_entity_id is not None
            and bool(unmatched)
        ),
    )


def _cluster_flags(
    *,
    status: str,
    current_entity_id: str | None,
    direct_entity_count: int,
    contested_anchor_seconds: float,
    has_fully_propagated: bool,
    faces: Sequence[ExclusiveActiveFaceOverlap],
    entities: Sequence[ExclusiveActiveEntityOverlap],
    unmatched: Sequence[str],
) -> SpeakerBindingClusterAuditFlags:
    return SpeakerBindingClusterAuditFlags(
        conflict=status == "conflict",
        ambiguous=status == "ambiguous",
        unbound=status == "unbound",
        has_fully_propagated_segments=has_fully_propagated,
        multiple_direct_entity_support=direct_entity_count > 1,
        contested_anchor=contested_anchor_seconds > 0,
        multiple_exclusive_active_face_tracks=len(faces) > 1,
        exclusive_active_entity_contradiction=len(entities) > 1,
        mapped_entity_vs_unmatched_face_contradiction=(
            status == "candidate_mapped"
            and current_entity_id is not None
            and bool(unmatched)
        ),
    )


def _review_priority(
    flags: SpeakerBindingClusterAuditFlags,
) -> tuple[ReviewPriority, int]:
    if flags.conflict:
        return "conflict", 1
    if flags.exclusive_active_entity_contradiction:
        return "exclusive_active_entity_contradiction", 2
    if flags.mapped_entity_vs_unmatched_face_contradiction:
        return "mapped_entity_vs_unmatched_face_contradiction", 3
    if flags.unbound or flags.ambiguous:
        return "unbound_or_ambiguous", 4
    if flags.has_fully_propagated_segments:
        return "fully_propagated", 5
    return "candidate_mapped_lowest_direct_support_ratio", 6


def _reason_flags(flags: SpeakerBindingClusterAuditFlags) -> list[str]:
    return [name for name, value in flags.model_dump().items() if value]


@dataclass(frozen=True)
class _AnchorAuditEvidence:
    lengths_by_cluster_entity: dict[tuple[str, str], list[float]]
    direct_spans_by_segment_entity: dict[tuple[str, str, str], list[_Span]]
    contested_spans_by_segment: dict[tuple[str, str], list[_Span]]


def _binding_anchor_evidence(
    *,
    raw_segments: Sequence[RawDiarizationSegment],
    sidecar: AudioBindingSidecar,
) -> _AnchorAuditEvidence:
    if not raw_segments:
        return _AnchorAuditEvidence({}, {}, {})
    sample_rate = raw_segments[0].source_sample_rate_hz
    minimum = AudioBindingProductionConfig().minimum_binding_confidence
    lengths: dict[tuple[str, str], list[float]] = defaultdict(list)
    direct: dict[tuple[str, str, str], list[_Span]] = defaultdict(list)
    contested: dict[tuple[str, str], list[_Span]] = defaultdict(list)
    for binding in sidecar.bindings:
        if (
            binding.status != "bound"
            or binding.entity_id is None
            or binding.confidence < minimum
            or not binding.evidence.synchronization_plausible
        ):
            continue
        anchor_start = round(binding.start_time * sample_rate)
        anchor_end = round(binding.end_time * sample_rate)
        boundaries = {anchor_start, anchor_end}
        for segment in raw_segments:
            start = max(anchor_start, segment.source_start_sample)
            end = min(anchor_end, segment.source_end_sample)
            if start < end:
                boundaries.update((start, end))
        support_by_cluster: dict[str, list[_Span]] = defaultdict(list)
        points = sorted(boundaries)
        for index in range(len(points) - 1):
            start, end = points[index], points[index + 1]
            active_segments = [
                segment
                for segment in raw_segments
                if segment.source_start_sample < end
                and segment.source_end_sample > start
            ]
            active_clusters = {
                segment.speaker_cluster_id for segment in active_segments
            }
            if len(active_clusters) == 1:
                cluster_id = next(iter(active_clusters))
                span = _Span(start, end)
                support_by_cluster[cluster_id].append(span)
                for segment in active_segments:
                    direct[
                        (cluster_id, segment.segment_id, binding.entity_id)
                    ].append(span)
            elif len(active_clusters) > 1:
                span = _Span(start, end)
                for segment in active_segments:
                    contested[(segment.speaker_cluster_id, segment.segment_id)].append(
                        span
                    )
        for cluster_id, spans in support_by_cluster.items():
            samples = _union_samples(spans)
            if samples:
                lengths[(cluster_id, binding.entity_id)].append(samples / sample_rate)
    return _AnchorAuditEvidence(dict(lengths), dict(direct), dict(contested))


def _source_paths(audio_production_root: Path) -> tuple[Path, Path]:
    paths = jea_production_paths(audio_production_root)
    audio_root = paths.audio
    diarization_root = paths.diarization
    if not audio_root.is_dir() or not diarization_root.is_dir():
        raise ValueError("binding audit requires production audio and diarization stages")
    return audio_root, diarization_root


def _load_sidecars(
    audio_root: Path,
) -> dict[str, tuple[AudioBindingSidecar, Path]]:
    output: dict[str, tuple[AudioBindingSidecar, Path]] = {}
    for path in sorted((audio_root / "clips").glob("**/audio_binding.json")):
        sidecar = AudioBindingSidecar.model_validate_json(path.read_text(encoding="utf-8"))
        if sidecar.clip_uid in output:
            raise ValueError("binding audit found duplicate Audio sidecar clip UID")
        if sidecar.status != "ready" or sidecar.evidence is None:
            continue
        output[sidecar.clip_uid] = (sidecar, path)
    return output


def _load_native(audio_root: Path, clip_uid: str) -> tuple[LRASDNativeArtifact, Path]:
    path = audio_root / "runtime" / clip_uid / "lr_asd" / "lr_asd_native.json"
    native = LRASDNativeArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if native.clip_uid != clip_uid:
        raise ValueError("binding audit LR-ASD clip identity is inconsistent")
    return native, path


def _artifact_set_fingerprint(artifacts: dict[str, str]) -> str:
    payload = [
        {"clip_uid": clip_uid, "artifact_sha256": artifact_sha256}
        for clip_uid, artifact_sha256 in sorted(artifacts.items())
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _validate_inputs(
    raw_segments: Sequence[RawDiarizationSegment],
    cluster_bindings: Sequence[DiarizationClusterBinding],
    bound_segments: Sequence[BoundDiarizationSegment],
) -> None:
    raw_keys = [
        (item.target_clip_uid, item.speaker_cluster_id, item.segment_id)
        for item in raw_segments
    ]
    bound_keys = [
        (item.target_clip_uid, item.speaker_cluster_id, item.segment_id)
        for item in bound_segments
    ]
    if len(raw_keys) != len(set(raw_keys)) or set(raw_keys) != set(bound_keys):
        raise ValueError("binding audit raw and bound segment identities differ")
    cluster_keys = [
        (item.target_clip_uid, item.speaker_cluster_id) for item in cluster_bindings
    ]
    if len(cluster_keys) != len(set(cluster_keys)):
        raise ValueError("binding audit found duplicate cluster bindings")
    raw_cluster_keys = {(clip_uid, cluster_id) for clip_uid, cluster_id, _ in raw_keys}
    if set(cluster_keys) != raw_cluster_keys:
        raise ValueError("binding audit cluster and segment inventories differ")


def _build_records(
    *,
    audio_root: Path,
    raw_segments: Sequence[RawDiarizationSegment],
    cluster_bindings: Sequence[DiarizationClusterBinding],
    bound_segments: Sequence[BoundDiarizationSegment],
) -> tuple[
    list[SpeakerBindingClusterAudit],
    list[SpeakerBindingSegmentAudit],
    dict[str, str],
    dict[str, str],
]:
    sidecars = _load_sidecars(audio_root)
    raw_by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    raw_by_cluster: dict[tuple[str, str], list[RawDiarizationSegment]] = defaultdict(list)
    bound_by_key = {
        (item.target_clip_uid, item.speaker_cluster_id, item.segment_id): item
        for item in bound_segments
    }
    for item in raw_segments:
        raw_by_clip[item.target_clip_uid].append(item)
        raw_by_cluster[(item.target_clip_uid, item.speaker_cluster_id)].append(item)
    cluster_by_key = {
        (item.target_clip_uid, item.speaker_cluster_id): item
        for item in cluster_bindings
    }
    cluster_records: list[SpeakerBindingClusterAudit] = []
    segment_records: list[SpeakerBindingSegmentAudit] = []
    sidecar_hashes: dict[str, str] = {}
    native_hashes: dict[str, str] = {}
    for clip_uid in sorted(raw_by_clip):
        loaded_sidecar = sidecars.get(clip_uid)
        if loaded_sidecar is None:
            raise ValueError(f"binding audit is missing ready Audio sidecar for {clip_uid}")
        sidecar, sidecar_path = loaded_sidecar
        native, native_path = _load_native(audio_root, clip_uid)
        sidecar_hashes[clip_uid] = _sha256(sidecar_path)
        native_hashes[clip_uid] = _sha256(native_path)
        frames = _exclusive_frames(native, sidecar)
        anchor_evidence = _binding_anchor_evidence(
            raw_segments=raw_by_clip[clip_uid],
            sidecar=sidecar,
        )
        clip_cluster_ids = sorted(
            {item.speaker_cluster_id for item in raw_by_clip[clip_uid]}
        )
        for cluster_id in clip_cluster_ids:
            key = (clip_uid, cluster_id)
            cluster = cluster_by_key[key]
            segments = sorted(
                raw_by_cluster[key],
                key=lambda item: (
                    item.source_start_sample,
                    item.source_end_sample,
                    item.segment_id,
                ),
            )
            supports = {
                item.entity_id: item.direct_support_seconds
                for item in cluster.entity_supports
            }
            binding_lengths = [
                duration
                for entity_id in supports
                for duration in anchor_evidence.lengths_by_cluster_entity.get(
                    (cluster_id, entity_id), []
                )
            ]
            expected_binding_count = sum(
                item.contributing_binding_count for item in cluster.entity_supports
            )
            if len(binding_lengths) != expected_binding_count:
                raise ValueError("binding audit direct-anchor provenance is inconsistent")
            for entity_id, direct_support_seconds in supports.items():
                reconstructed_support = sum(
                    anchor_evidence.lengths_by_cluster_entity.get(
                        (cluster_id, entity_id), []
                    )
                )
                if not math.isclose(
                    reconstructed_support,
                    direct_support_seconds,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "binding audit direct entity support is inconsistent"
                    )
            intervals = [(item.start_time, item.end_time) for item in segments]
            faces, entities, unmatched = _exclusive_overlap_evidence(intervals, frames)
            fully_propagated_seconds = 0.0
            has_fully_propagated = False
            for segment in segments:
                bound = bound_by_key[(clip_uid, cluster_id, segment.segment_id)]
                segment_intervals = [(segment.start_time, segment.end_time)]
                segment_faces, segment_entities, segment_unmatched = (
                    _exclusive_overlap_evidence(segment_intervals, frames)
                )
                duration = segment.end_time - segment.start_time
                segment_direct_evidence = [
                    (entity_id, spans)
                    for (support_cluster, segment_id, entity_id), spans in (
                        anchor_evidence.direct_spans_by_segment_entity.items()
                    )
                    if support_cluster == cluster_id
                    and segment_id == segment.segment_id
                ]
                segment_supports = {
                    entity_id: samples / segment.source_sample_rate_hz
                    for entity_id, spans in segment_direct_evidence
                    if (samples := _union_samples(spans)) > 0
                }
                reconstructed_direct_samples = _union_samples(
                    span
                    for _, spans in segment_direct_evidence
                    for span in spans
                )
                if reconstructed_direct_samples != bound.direct_anchor_samples:
                    raise ValueError(
                        "binding audit segment direct-anchor evidence is inconsistent"
                    )
                segment_contested_samples = _union_samples(
                    anchor_evidence.contested_spans_by_segment.get(
                        (cluster_id, segment.segment_id), []
                    )
                )
                segment_contested_seconds = (
                    segment_contested_samples / segment.source_sample_rate_hz
                )
                propagated = (
                    max(0.0, duration - bound.direct_anchor_seconds)
                    if bound.cluster_binding_status == "candidate_mapped"
                    else 0.0
                )
                fully_propagated = bound.identity_scope == "cluster_propagated_only"
                if fully_propagated:
                    fully_propagated_seconds += duration
                    has_fully_propagated = True
                segment_flags = _segment_flags(
                    status=bound.cluster_binding_status,
                    current_entity_id=bound.entity_id,
                    direct_entity_count=len(segment_supports),
                    contested_anchor_seconds=segment_contested_seconds,
                    fully_propagated=fully_propagated,
                    faces=segment_faces,
                    entities=segment_entities,
                    unmatched=segment_unmatched,
                )
                segment_records.append(
                    SpeakerBindingSegmentAudit(
                        clip_uid=clip_uid,
                        segment_id=segment.segment_id,
                        speaker_cluster_id=cluster_id,
                        current_mapping_status=bound.cluster_binding_status,
                        current_entity_id=bound.entity_id,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        speaker_duration_seconds=duration,
                        direct_anchor_seconds=bound.direct_anchor_seconds,
                        direct_support_seconds_by_entity=segment_supports,
                        directly_supported_entity_count=len(segment_supports),
                        contested_anchor_seconds=segment_contested_seconds,
                        identity_propagated_seconds=propagated,
                        exclusive_active_faces=segment_faces,
                        exclusive_active_entities=segment_entities,
                        unmatched_exclusive_active_face_track_ids=segment_unmatched,
                        flags=segment_flags,
                    )
                )
            direct_anchor_seconds = cluster.usable_anchor_duration
            identity_propagated_seconds = (
                max(0.0, cluster.cluster_speaker_seconds - direct_anchor_seconds)
                if cluster.status == "candidate_mapped"
                else 0.0
            )
            cluster_flags = _cluster_flags(
                status=cluster.status,
                current_entity_id=cluster.entity_id,
                direct_entity_count=len(supports),
                contested_anchor_seconds=cluster.contested_anchor_duration,
                has_fully_propagated=has_fully_propagated,
                faces=faces,
                entities=entities,
                unmatched=unmatched,
            )
            priority, rank = _review_priority(cluster_flags)
            competing = {
                entity_id: seconds
                for entity_id, seconds in supports.items()
                if entity_id != cluster.entity_id
            }
            cluster_records.append(
                SpeakerBindingClusterAudit(
                    clip_uid=clip_uid,
                    speaker_cluster_id=cluster_id,
                    current_mapping_status=cluster.status,
                    current_entity_id=cluster.entity_id,
                    cluster_segment_count=cluster.cluster_segment_count,
                    cluster_speaker_duration_seconds=cluster.cluster_speaker_seconds,
                    direct_anchor_seconds=direct_anchor_seconds,
                    identity_propagated_seconds=identity_propagated_seconds,
                    fully_propagated_seconds=fully_propagated_seconds,
                    contributing_direct_binding_count=expected_binding_count,
                    direct_support_seconds_by_entity=supports,
                    directly_supported_entity_count=len(supports),
                    competing_direct_entity_support=competing,
                    direct_support_ratio=(
                        direct_anchor_seconds / cluster.cluster_speaker_seconds
                        if cluster.status == "candidate_mapped"
                        else None
                    ),
                    longest_direct_anchor_seconds=(
                        max(binding_lengths) if binding_lengths else None
                    ),
                    shortest_direct_anchor_seconds=(
                        min(binding_lengths) if binding_lengths else None
                    ),
                    contested_anchor_seconds=cluster.contested_anchor_duration,
                    exclusive_active_faces=faces,
                    exclusive_active_entities=entities,
                    unmatched_exclusive_active_face_track_ids=unmatched,
                    flags=cluster_flags,
                    review_priority=priority,
                    review_priority_rank=rank,
                )
            )
    cluster_records.sort(key=lambda item: (item.clip_uid, item.speaker_cluster_id))
    segment_records.sort(
        key=lambda item: (
            item.clip_uid,
            item.start_time,
            item.end_time,
            item.speaker_cluster_id,
            item.segment_id,
        )
    )
    return cluster_records, segment_records, sidecar_hashes, native_hashes


def _review_manifest(
    clusters: Sequence[SpeakerBindingClusterAudit],
) -> list[SpeakerBindingReviewCase]:
    ordered = sorted(
        clusters,
        key=lambda item: (
            item.review_priority_rank,
            item.direct_support_ratio
            if item.direct_support_ratio is not None
            else math.inf,
            item.clip_uid,
            item.speaker_cluster_id,
        ),
    )
    return [
        SpeakerBindingReviewCase(
            review_index=index,
            review_priority=item.review_priority,
            review_priority_rank=item.review_priority_rank,
            clip_uid=item.clip_uid,
            speaker_cluster_id=item.speaker_cluster_id,
            current_mapping_status=item.current_mapping_status,
            current_entity_id=item.current_entity_id,
            reason_flags=_reason_flags(item.flags),
            direct_support_ratio=item.direct_support_ratio,
        )
        for index, item in enumerate(ordered, start=1)
    ]


def _summary(
    *,
    production_root: Path,
    diarization_root: Path,
    source_paths: dict[str, Path],
    audio_binding_hashes: dict[str, str],
    lr_asd_native_hashes: dict[str, str],
    clusters: Sequence[SpeakerBindingClusterAudit],
    segments: Sequence[SpeakerBindingSegmentAudit],
    review: Sequence[SpeakerBindingReviewCase],
) -> SpeakerBindingAuditSummary:
    priority_counts = {name: 0 for name in ReviewPriority.__args__}
    for item in review:
        priority_counts[item.review_priority] += 1
    status_counts = {name: 0 for name in ("candidate_mapped", "conflict", "ambiguous", "unbound")}
    for item in clusters:
        status_counts[item.current_mapping_status] += 1
    return SpeakerBindingAuditSummary(
        source_audio_production_root=str(production_root),
        source_diarization_root=str(diarization_root),
        source_artifact_sha256={name: _sha256(path) for name, path in source_paths.items()},
        audio_binding_input_set_sha256=_artifact_set_fingerprint(
            audio_binding_hashes
        ),
        lr_asd_native_input_set_sha256=_artifact_set_fingerprint(
            lr_asd_native_hashes
        ),
        clip_count=len({item.clip_uid for item in clusters}),
        cluster_count=len(clusters),
        segment_count=len(segments),
        candidate_mapped_count=status_counts["candidate_mapped"],
        conflict_count=status_counts["conflict"],
        ambiguous_count=status_counts["ambiguous"],
        unbound_count=status_counts["unbound"],
        clusters_with_multiple_exclusive_active_face_tracks=sum(
            item.flags.multiple_exclusive_active_face_tracks for item in clusters
        ),
        clusters_with_exclusive_active_entity_contradiction=sum(
            item.flags.exclusive_active_entity_contradiction for item in clusters
        ),
        clusters_with_mapped_entity_vs_unmatched_face=sum(
            item.flags.mapped_entity_vs_unmatched_face_contradiction for item in clusters
        ),
        clusters_with_multiple_direct_entity_support=sum(
            item.flags.multiple_direct_entity_support for item in clusters
        ),
        clusters_with_fully_propagated_segments=sum(
            item.fully_propagated_seconds > 0 for item in clusters
        ),
        multiple_exclusive_active_face_track_speaker_seconds=sum(
            item.cluster_speaker_duration_seconds
            for item in clusters
            if item.flags.multiple_exclusive_active_face_tracks
        ),
        exclusive_active_entity_contradiction_speaker_seconds=sum(
            item.cluster_speaker_duration_seconds
            for item in clusters
            if item.flags.exclusive_active_entity_contradiction
        ),
        mapped_entity_vs_unmatched_face_speaker_seconds=sum(
            item.cluster_speaker_duration_seconds
            for item in clusters
            if item.flags.mapped_entity_vs_unmatched_face_contradiction
        ),
        fully_propagated_segment_speaker_seconds=sum(
            item.fully_propagated_seconds for item in clusters
        ),
        identity_propagated_speaker_seconds=sum(
            item.identity_propagated_seconds for item in clusters
        ),
        review_priority_counts=priority_counts,
    )


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"binding audit output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def run_speaker_binding_audit(
    *,
    audio_production_root: Path,
    overwrite: bool = False,
) -> SpeakerBindingAuditSummary:
    production_root = audio_production_root.expanduser().resolve(strict=True)
    audio_root, diarization_root = _source_paths(production_root)
    source_paths = {
        "raw_segments": diarization_root / "raw_segments.jsonl",
        "cluster_bindings": diarization_root / "cluster_bindings.jsonl",
        "bound_segments": diarization_root / "bound_segments.jsonl",
        "diarization_summary": diarization_root / "summary.json",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise ValueError(f"binding audit source artifact is missing: {path.name}")
    raw_segments = _json_lines(source_paths["raw_segments"], RawDiarizationSegment)
    cluster_bindings = _json_lines(
        source_paths["cluster_bindings"], DiarizationClusterBinding
    )
    bound_segments = _json_lines(
        source_paths["bound_segments"], BoundDiarizationSegment
    )
    _validate_inputs(raw_segments, cluster_bindings, bound_segments)
    clusters, segments, audio_binding_hashes, lr_asd_native_hashes = _build_records(
        audio_root=audio_root,
        raw_segments=raw_segments,
        cluster_bindings=cluster_bindings,
        bound_segments=bound_segments,
    )
    review = _review_manifest(clusters)
    summary = _summary(
        production_root=production_root,
        diarization_root=diarization_root,
        source_paths=source_paths,
        audio_binding_hashes=audio_binding_hashes,
        lr_asd_native_hashes=lr_asd_native_hashes,
        clusters=clusters,
        segments=segments,
        review=review,
    )
    destination = production_root / "binding_audit_v1"
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        _write_json(temporary / "summary.json", summary)
        _write_jsonl(temporary / "clusters.jsonl", clusters)
        _write_jsonl(temporary / "segments.jsonl", segments)
        _write_jsonl(temporary / "review_manifest.jsonl", review)
        _publish_directory(temporary, destination, overwrite=overwrite)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return summary
