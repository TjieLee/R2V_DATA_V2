from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from r2v_data_v2.h3.pilot_schemas import LRASDNativeSample, LRASDNativeTrack
from r2v_data_v2.h3.schemas import (
    EntityAssociationCandidateEvidence,
    EntityFaceAssociation,
)
from r2v_data_v2.v3.mask_codec import decode_binary_mask
from r2v_data_v2.v3.schemas import SampledFramesArtifact, TrackedMasksArtifact


@dataclass(frozen=True)
class FaceEntityAssociationPolicy:
    maximum_timestamp_delta_seconds: float = 0.06
    minimum_face_bbox_coverage: float = 0.50
    minimum_matched_sampled_slots: int = 2
    minimum_temporal_consistency: float = 0.50
    minimum_top1_top2_margin: float = 0.15

    def __post_init__(self) -> None:
        if self.maximum_timestamp_delta_seconds <= 0:
            raise ValueError("association timestamp tolerance must be positive")
        ratios = (
            self.minimum_face_bbox_coverage,
            self.minimum_temporal_consistency,
            self.minimum_top1_top2_margin,
        )
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("association ratio thresholds must be in [0, 1]")
        if not 1 <= self.minimum_matched_sampled_slots <= 10:
            raise ValueError("association matched-slot threshold must be in [1, 10]")


def nearest_track_sample(
    track: LRASDNativeTrack,
    *,
    timestamp_seconds: float,
    maximum_delta_seconds: float,
) -> tuple[LRASDNativeSample, float] | None:
    if maximum_delta_seconds <= 0:
        raise ValueError("nearest-sample tolerance must be positive")
    sample = min(
        track.samples,
        key=lambda item: (
            abs(item.timestamp_seconds - timestamp_seconds),
            item.frame_index,
        ),
    )
    delta = abs(sample.timestamp_seconds - timestamp_seconds)
    if delta > maximum_delta_seconds:
        return None
    return sample, delta


def _face_mask_metrics(
    sample: LRASDNativeSample,
    mask: np.ndarray,
) -> tuple[float, bool]:
    height, width = mask.shape
    x1, y1, x2, y2 = sample.bbox_xyxy
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    if right <= left or bottom <= top:
        return 0.0, False
    face_region = mask[top:bottom, left:right]
    coverage = float(np.count_nonzero(face_region)) / face_region.size
    center_x = max(0, min(width - 1, int((x1 + x2) / 2)))
    center_y = max(0, min(height - 1, int((y1 + y2) / 2)))
    return coverage, bool(mask[center_y, center_x])


def associate_face_tracks_to_entities(
    *,
    frames: SampledFramesArtifact,
    masks: TrackedMasksArtifact,
    tracks: list[LRASDNativeTrack],
    policy: FaceEntityAssociationPolicy | None = None,
) -> list[EntityFaceAssociation]:
    active_policy = policy or FaceEntityAssociationPolicy()
    if frames.clip_uid != masks.clip_uid:
        raise ValueError("frame and mask artifacts must belong to the same clip")
    if (frames.width, frames.height) != (masks.width, masks.height):
        raise ValueError("frame and mask artifact dimensions must match")
    associations: list[EntityFaceAssociation] = []
    for track in tracks:
        aligned = []
        for frame in frames.frames:
            nearest = nearest_track_sample(
                track,
                timestamp_seconds=frame.timestamp_seconds,
                maximum_delta_seconds=active_policy.maximum_timestamp_delta_seconds,
            )
            if nearest is not None:
                sample, delta = nearest
                aligned.append((frame.slot, sample, delta))
        slot_diagnostics: list[dict[str, object]] = []
        candidates: list[EntityAssociationCandidateEvidence] = []
        for entity_id, tracked_entity in masks.entities.items():
            coverages: list[float] = []
            center_count = 0
            matched_slots = 0
            for slot, sample, delta in aligned:
                tracked_frame = tracked_entity.frames[slot]
                if tracked_entity.status == "ready" and tracked_frame.present:
                    mask = decode_binary_mask(tracked_frame.rle).astype(bool, copy=False)
                    coverage, center_inside = _face_mask_metrics(sample, mask)
                else:
                    coverage, center_inside = 0.0, False
                matched = (
                    coverage >= active_policy.minimum_face_bbox_coverage
                    or center_inside
                )
                coverages.append(coverage)
                center_count += int(center_inside)
                matched_slots += int(matched)
                slot_diagnostics.append(
                    {
                        "face_track_id": track.face_track_id,
                        "entity_id": entity_id,
                        "sampled_frame_slot": slot,
                        "lr_asd_frame_index": sample.frame_index,
                        "lr_asd_timestamp_seconds": sample.timestamp_seconds,
                        "timestamp_delta_seconds": delta,
                        "face_bbox_coverage": coverage,
                        "face_center_inside_mask": center_inside,
                        "matched": matched,
                    }
                )
            aligned_count = len(aligned)
            temporal_consistency = (
                matched_slots / aligned_count if aligned_count else 0.0
            )
            mean_coverage = sum(coverages) / aligned_count if aligned_count else 0.0
            center_ratio = center_count / aligned_count if aligned_count else 0.0
            association_score = 0.7 * mean_coverage + 0.3 * center_ratio
            candidates.append(
                EntityAssociationCandidateEvidence(
                    entity_id=entity_id,
                    aligned_sampled_slots=aligned_count,
                    matched_sampled_slots=matched_slots,
                    mean_face_bbox_coverage=mean_coverage,
                    face_center_inside_count=center_count,
                    temporal_consistency=temporal_consistency,
                    association_score=association_score,
                )
            )
        candidates.sort(key=lambda item: (-item.association_score, item.entity_id))
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        margin = (
            max(0.0, top.association_score - second.association_score)
            if top is not None and second is not None
            else None
        )
        evidence = {
            "policy": {
                "maximum_timestamp_delta_seconds": (
                    active_policy.maximum_timestamp_delta_seconds
                ),
                "minimum_face_bbox_coverage": (
                    active_policy.minimum_face_bbox_coverage
                ),
                "minimum_matched_sampled_slots": (
                    active_policy.minimum_matched_sampled_slots
                ),
                "minimum_temporal_consistency": (
                    active_policy.minimum_temporal_consistency
                ),
                "minimum_top1_top2_margin": (
                    active_policy.minimum_top1_top2_margin
                ),
                "validated_for_production": False,
            },
            "aligned_sampled_slots": len(aligned),
            "slot_diagnostics": sorted(
                slot_diagnostics,
                key=lambda item: (
                    int(item["sampled_frame_slot"]),
                    str(item["entity_id"]),
                ),
            ),
        }
        if len(aligned) < active_policy.minimum_matched_sampled_slots:
            associations.append(
                EntityFaceAssociation(
                    face_track_id=track.face_track_id,
                    status="ambiguous",
                    confidence=0.0 if top is None else top.association_score,
                    method="tracked_geometry",
                    candidates=candidates,
                    top1_top2_margin=margin,
                    reason="insufficient_aligned_sampled_slots",
                    evidence=evidence,
                )
            )
            continue
        top_valid = (
            top is not None
            and top.matched_sampled_slots
            >= active_policy.minimum_matched_sampled_slots
            and top.temporal_consistency
            >= active_policy.minimum_temporal_consistency
        )
        second_valid = (
            second is not None
            and second.matched_sampled_slots
            >= active_policy.minimum_matched_sampled_slots
            and second.temporal_consistency
            >= active_policy.minimum_temporal_consistency
        )
        if top_valid and second_valid and margin is not None and (
            margin < active_policy.minimum_top1_top2_margin
        ):
            associations.append(
                EntityFaceAssociation(
                    face_track_id=track.face_track_id,
                    status="ambiguous",
                    confidence=top.association_score,
                    method="tracked_geometry",
                    candidates=candidates,
                    top1_top2_margin=margin,
                    reason="conflicting_entity_masks",
                    evidence=evidence,
                )
            )
        elif top_valid:
            assert top is not None
            associations.append(
                EntityFaceAssociation(
                    face_track_id=track.face_track_id,
                    status="matched",
                    entity_id=top.entity_id,
                    confidence=top.association_score,
                    method="tracked_geometry",
                    candidates=candidates,
                    top1_top2_margin=margin,
                    evidence=evidence,
                )
            )
        else:
            associations.append(
                EntityFaceAssociation(
                    face_track_id=track.face_track_id,
                    status="unmatched",
                    confidence=0.0 if top is None else top.association_score,
                    method="tracked_geometry",
                    candidates=candidates,
                    top1_top2_margin=margin,
                    reason="association_evidence_below_pilot_thresholds",
                    evidence=evidence,
                )
            )
    return associations
