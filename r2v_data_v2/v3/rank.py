from __future__ import annotations

from dataclasses import asdict, dataclass

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    CoverageState,
    EntityVisibilitySummary,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class CoverageStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    passed: int = 0
    rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_mask_semantics(
    artifact: TrackedMasksArtifact,
    entities: list[AnnotationEntity],
) -> None:
    if list(artifact.entities) != [entity.entity_id for entity in entities]:
        raise ValueError("mask artifact entity IDs do not match annotation order")
    for entity in entities:
        tracked = artifact.entities[entity.entity_id]
        if (
            tracked.reference_type != entity.reference_type
            or tracked.grounding_prompt != entity.grounding_prompt
        ):
            raise ValueError(
                "mask artifact entity semantics do not match annotation"
            )


def build_coverage_state(
    *,
    artifact: TrackedMasksArtifact,
    entities: list[AnnotationEntity],
    required_visible_frames: int,
) -> CoverageState:
    if not 1 <= required_visible_frames <= artifact.sampled_frame_count:
        raise ValueError(
            "required_visible_frames must be within sampled frame count"
        )
    _validate_mask_semantics(artifact, entities)
    summaries: dict[str, EntityVisibilitySummary] = {}
    qualifying_entity_ids: list[str] = []
    for entity in entities:
        tracked = artifact.entities[entity.entity_id]
        visible_frames = [
            frame
            for frame in tracked.frames
            if (
                tracked.status == "ready"
                and frame.present
                and frame.track_valid
            )
        ]
        visible_slots = [frame.slot for frame in visible_frames]
        visible_frame_count = len(visible_slots)
        qualifies = visible_frame_count >= required_visible_frames
        if qualifies:
            qualifying_entity_ids.append(entity.entity_id)
        visible_slot_set = set(visible_slots)
        summaries[entity.entity_id] = EntityVisibilitySummary(
            status=tracked.status,
            visible_frame_slots=visible_slots,
            visible_frame_count=visible_frame_count,
            coverage_ratio=(
                visible_frame_count / artifact.sampled_frame_count
            ),
            qualifies=qualifies,
            per_frame_area_ratio=[
                frame.area_ratio if frame.slot in visible_slot_set else 0.0
                for frame in tracked.frames
            ],
            per_frame_confidence=[
                frame.confidence
                if frame.slot in visible_slot_set
                else None
                for frame in tracked.frames
            ],
        )
    return CoverageState(
        passed=bool(qualifying_entity_ids),
        qualifying_entity_ids=qualifying_entity_ids,
        required_visible_frames=required_visible_frames,
        entity_visibility_summary=summaries,
    )


def rank_temporal_coverage(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
) -> CoverageStats:
    processed = skipped_existing = skipped_not_ready = failed = 0
    passed = rejected = 0
    for clip in storage.iter_clips():
        annotation = clip.annotation
        if annotation is None or annotation.status != "ready":
            skipped_not_ready += 1
            continue
        try:
            artifact = storage.read_masks(clip.clip_uid)
            if artifact.clip_uid != clip.clip_uid:
                raise ValueError(
                    "mask artifact clip_uid does not match its clip"
                )
            coverage = build_coverage_state(
                artifact=artifact,
                entities=annotation.entities,
                required_visible_frames=(
                    config.coverage.required_visible_frames
                ),
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-clip mask failures
            storage.clear_coverage(clip.clip_uid)
            storage.append_failure(
                clip_uid=clip.clip_uid,
                stage="rank",
                reason=str(exc),
            )
            failed += 1
            continue

        if clip.coverage == coverage and not overwrite:
            skipped_existing += 1
            continue
        storage.write_coverage(clip.clip_uid, coverage)
        processed += 1
        if coverage.passed:
            passed += 1
        else:
            rejected += 1
    stats = CoverageStats(
        processed=processed,
        skipped_existing=skipped_existing,
        skipped_not_ready=skipped_not_ready,
        failed=failed,
        passed=passed,
        rejected=rejected,
    )
    storage.update_stage_counts("rank", stats.to_dict())
    return stats
