from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguReferenceEditBackend,
    BooguReferenceEditJudge,
    BooguSamReviewer,
    BooguSubprocessBackend,
    BooguWorkerConfig,
    QwenBooguReferenceEditJudge,
    Sam3BooguReferenceReviewer,
    run_boogu_reference_edit,
)
from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    EntityReferenceState,
    PairingState,
    ReferenceCompleteness,
    ReferenceEditEntityState,
    ReferenceEditState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage


class StartableBooguBackend(BooguReferenceEditBackend, Protocol):
    def start(self, *, stderr_log_path: Path) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ReferenceEditStats:
    processed: int = 0
    skipped_existing: int = 0
    skipped_not_ready: int = 0
    failed: int = 0
    entities_eligible: int = 0
    entities_accepted: int = 0
    entities_fallback: int = 0
    entities_rejected: int = 0
    entities_failed: int = 0
    worker_starts: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _route(reference: EntityReferenceState) -> ReferenceCompleteness:
    if reference.completeness is not None:
        return reference.completeness
    return "complete" if reference.reference_scope == "full" else "local_usable"


def _operation(
    config: V3Config,
    reference: EntityReferenceState,
) -> str | None:
    route = _route(reference)
    if route == "repairable":
        return "complete_entity"
    if route == "complete" and config.reference_edit.add_background_to_complete:
        return "add_entity_background"
    return None


def _eligible_references(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool,
) -> int:
    count = 0
    for clip in storage.iter_clips():
        if clip.pairing is None or clip.pairing.status != "ready":
            continue
        if clip.reference_edit is not None and not overwrite:
            continue
        previous = {
            item.entity_id: item
            for item in (
                clip.reference_edit.entities
                if clip.reference_edit is not None
                and clip.reference_edit.status == "ready"
                else []
            )
        }
        retained = set(clip.pairing.retained_entity_ids)
        for current in clip.references.entities:
            if current.entity_id not in retained:
                continue
            reference = (
                previous[current.entity_id].source_reference
                if overwrite and current.entity_id in previous
                else current
            )
            if reference.status == "ready" and _operation(config, reference) is not None:
                count += 1
    return count


def _resolve_artifact(storage: RunStorage, value: str) -> Path:
    root = storage.root.resolve(strict=False)
    path = (root / value).resolve(strict=False)
    if root not in path.parents or not path.is_file():
        raise ValueError("reference edit input must be an existing run artifact")
    return path


def _tokens_for_retained(
    retained: list[str],
    entities: dict[str, AnnotationEntity],
) -> dict[str, str]:
    counters = {"subject": 0, "object": 0, "group": 0}
    tokens: dict[str, str] = {}
    for entity_id in retained:
        reference_type = entities[entity_id].reference_type
        counters[reference_type] += 1
        tokens[entity_id] = f"<ref_{reference_type}_{counters[reference_type]}>"
    return tokens


def _rejected_reference(
    reference: EntityReferenceState,
    reason: str,
) -> EntityReferenceState:
    return EntityReferenceState(
        entity_id=reference.entity_id,
        status="rejected",
        reference_scope="reject",
        visible_region="custom",
        whole_entity_recognizable=False,
        identity_features_visible=False,
        scope_reason=reason,
        image_quality=reference.image_quality,
        completeness=reference.completeness,
    )


def _instruction(operation: str, entity: AnnotationEntity) -> str:
    if operation == "complete_entity":
        return (
            f"Complete the same {entity.phrase} as one coherent full entity. "
            "Preserve its identity, visible attributes, proportions, clothing, "
            "materials, colors, pose continuity, and camera style. Add no second entity."
        )
    return (
        f"Place the same {entity.phrase} on a simple coherent neutral background. "
        "Preserve the entity exactly, do not extend or complete it, and add no other "
        "salient entity."
    )


def _accepted_reference(
    storage: RunStorage,
    reference: EntityReferenceState,
    *,
    clip_uid: str,
    output_path: Path,
    metadata_path: Path,
) -> EntityReferenceState:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_sha256 = metadata.get("source_image_sha256")
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    if (
        not isinstance(source_sha256, str)
        or metadata.get("generated_reference_sha256") != output_sha256
    ):
        raise ValueError("accepted Boogu metadata hashes do not match artifacts")
    return EntityReferenceState(
        entity_id=reference.entity_id,
        status="ready",
        reference_scope="full",
        visible_region="whole",
        whole_entity_recognizable=True,
        identity_features_visible=True,
        scope_reason="accepted_boogu_reference_edit",
        image_path=storage.relative_artifact_path(output_path),
        source_frame_index=reference.source_frame_index,
        source_clip_uid=clip_uid,
        source_entity_id=reference.entity_id,
        image_quality="high",
        completeness="complete",
        synthetic=True,
        generation_metadata_path=storage.relative_artifact_path(metadata_path),
        generation_source_sha256=source_sha256,
        generation_output_sha256=output_sha256,
    )


def reference_edit_clips(
    config: V3Config,
    storage: RunStorage,
    *,
    overwrite: bool = False,
    backend: BooguReferenceEditBackend | None = None,
    judge: BooguReferenceEditJudge | None = None,
    sam_reviewer: BooguSamReviewer | None = None,
) -> ReferenceEditStats:
    config.validate()
    if storage.root != config.resolved_run_root:
        raise ValueError("storage run_root does not match reference edit configuration")
    if not config.reference_edit.enabled:
        raise ValueError("V3 reference_edit stage is disabled")

    counters = {field: 0 for field in ReferenceEditStats.__dataclass_fields__}
    eligible_count = _eligible_references(
        config,
        storage,
        overwrite=overwrite,
    )
    counters["entities_eligible"] = eligible_count
    active_backend = backend
    active_judge = judge
    active_sam = sam_reviewer
    owned_backend: BooguSubprocessBackend | None = None
    owned_judge: QwenBooguReferenceEditJudge | None = None
    owned_segmenter: Sam3SegmentationBackend | None = None
    started_backend: object | None = None

    try:
        if eligible_count:
            if active_backend is None:
                owned_backend = BooguSubprocessBackend(
                    BooguWorkerConfig(
                        python_executable=config.reference_edit.python_executable,
                        code_root=config.reference_edit.code_root,
                        model_path=config.reference_edit.model_path,
                        model_revision=config.reference_edit.model_revision,
                        cuda_visible_devices=(
                            config.reference_edit.cuda_visible_devices
                        ),
                        timeout_seconds=config.reference_edit.timeout_seconds,
                        temporary_root=storage.reference_edit_temporary_dir(),
                    )
                )
                active_backend = owned_backend
            starter = getattr(active_backend, "start", None)
            if callable(starter):
                starter(stderr_log_path=storage.reference_edit_worker_log_path())
                started_backend = active_backend
                counters["worker_starts"] += 1
            if active_judge is None:
                judge_config = config.qwen.reference_edit_judge
                if judge_config is None:
                    raise ValueError("qwen.reference_edit_judge is not configured")
                owned_judge = QwenBooguReferenceEditJudge(judge_config)
                active_judge = owned_judge
            if active_sam is None:
                if config.sam3.model_path is None:
                    raise ValueError(
                        "sam3.model_path is required for production reference_edit"
                    )
                owned_segmenter = Sam3SegmentationBackend(config.sam3)
                active_sam = Sam3BooguReferenceReviewer(
                    owned_segmenter,
                    temporary_root=storage.reference_edit_temporary_dir(),
                    max_area_growth_ratio=(
                        config.reference_edit.sam_max_area_growth_ratio
                    ),
                    max_significant_components=(
                        config.reference_edit.sam_max_significant_components
                    ),
                )
            assert active_backend is not None
            assert active_judge is not None

        for initial_clip in storage.iter_clips():
            clip = storage.read_clip(initial_clip.clip_uid)
            if (
                clip.annotation is None
                or clip.annotation.status != "ready"
                or clip.coverage is None
                or not clip.coverage.passed
                or clip.pairing is None
                or clip.pairing.status != "ready"
            ):
                counters["skipped_not_ready"] += 1
                continue
            if clip.reference_edit is not None and not overwrite:
                counters["skipped_existing"] += 1
                continue
            counters["processed"] += 1
            try:
                if overwrite:
                    storage.cleanup_reference_edit_artifacts(clip.clip_uid)
                entities = {
                    entity.entity_id: entity for entity in clip.annotation.entities
                }
                previous_edits = {
                    item.entity_id: item
                    for item in (
                        clip.reference_edit.entities
                        if clip.reference_edit is not None
                        and clip.reference_edit.status == "ready"
                        else []
                    )
                }
                original_retained = set(clip.pairing.retained_entity_ids)
                final_references: list[EntityReferenceState] = []
                edit_states: list[ReferenceEditEntityState] = []
                for reference in clip.references.entities:
                    previous_edit = previous_edits.get(reference.entity_id)
                    if overwrite and previous_edit is not None:
                        reference = previous_edit.source_reference
                    if (
                        reference.status != "ready"
                        or reference.entity_id not in original_retained
                    ):
                        final_references.append(reference)
                        continue
                    if reference.image_path is None:
                        raise ValueError("ready reference has no image_path")
                    _resolve_artifact(storage, reference.image_path)
                    route = _route(reference)
                    operation = _operation(config, reference)
                    if operation is None:
                        final_references.append(reference)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="not_required",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=reference.image_path,
                            )
                        )
                        continue
                    if active_backend is None or active_judge is None:
                        raise RuntimeError("reference edit runtime was not initialized")
                    entity = entities[reference.entity_id]
                    result = run_boogu_reference_edit(
                        run_root=storage.root,
                        clip_uid=clip.clip_uid,
                        entity_id=reference.entity_id,
                        operation=operation,
                        instruction=_instruction(operation, entity),
                        entity_phrase=entity.phrase,
                        grounding_prompt=entity.grounding_prompt,
                        reference_type=entity.reference_type,
                        backend=active_backend,
                        judge=active_judge,
                        sam_reviewer=active_sam,
                        target_area=config.reference_edit.target_area,
                        alignment=config.reference_edit.alignment,
                        model_revision=config.reference_edit.model_revision,
                        fallback_status=config.reference_edit.fallback_policy,
                        overwrite=overwrite,
                    )
                    if result.status == "accepted":
                        assert result.final_reference_path is not None
                        accepted = _accepted_reference(
                            storage,
                            reference,
                            clip_uid=clip.clip_uid,
                            output_path=result.final_reference_path,
                            metadata_path=result.metadata_path,
                        )
                        final_references.append(accepted)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="accepted",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=accepted.image_path,
                                operation=operation,
                                metadata_path=accepted.generation_metadata_path,
                            )
                        )
                        counters["entities_accepted"] += 1
                        continue
                    rejection_reason = "boogu_candidate_rejected"
                    if result.rejection_path is not None:
                        rejection_payload = json.loads(
                            result.rejection_path.read_text(encoding="utf-8")
                        )
                        raw_reason = rejection_payload.get("reason")
                        if isinstance(raw_reason, str) and raw_reason.strip():
                            rejection_reason = raw_reason.strip()
                    if rejection_reason.startswith("boogu_reference_edit_failed:"):
                        counters["entities_failed"] += 1
                        storage.append_failure(
                            stage="reference_edit",
                            clip_uid=clip.clip_uid,
                            reason=rejection_reason,
                            details={
                                "entity_id": reference.entity_id,
                                "operation": operation,
                            },
                        )
                    if config.reference_edit.fallback_policy == "keep_source":
                        final_references.append(reference)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="fallback",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=reference.image_path,
                                operation=operation,
                                metadata_path=storage.relative_artifact_path(
                                    result.metadata_path
                                ),
                                fallback_policy="keep_source",
                                reason=rejection_reason,
                            )
                        )
                        counters["entities_fallback"] += 1
                    else:
                        final_references.append(
                            _rejected_reference(
                                reference,
                                rejection_reason,
                            )
                        )
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="rejected",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                operation=operation,
                                metadata_path=storage.relative_artifact_path(
                                    result.metadata_path
                                ),
                                fallback_policy="reject_entity",
                                reason=rejection_reason,
                            )
                        )
                        counters["entities_rejected"] += 1

                retained = [
                    reference.entity_id
                    for reference in final_references
                    if reference.status == "ready"
                    and reference.entity_id in original_retained
                ]
                qualifying = set(clip.coverage.qualifying_entity_ids)
                if not set(retained).intersection(qualifying):
                    pairing = PairingState(
                        status="rejected",
                        reason="no_qualifying_reference_after_reference_edit",
                    )
                else:
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(retained, entities),
                        background_token=clip.pairing.background_token,
                    )
                storage.write_reference_edit_result(
                    clip.clip_uid,
                    ReferencesState(
                        entities=final_references,
                        background=clip.references.background,
                    ),
                    pairing,
                    ReferenceEditState(status="ready", entities=edit_states),
                )
            except Exception as exc:  # noqa: BLE001 - isolate clip failures
                reason = str(exc)
                storage.write_reference_edit_failure(clip.clip_uid, reason)
                storage.append_failure(
                    stage="reference_edit",
                    clip_uid=clip.clip_uid,
                    reason=reason,
                    details={"exception_type": type(exc).__name__},
                )
                counters["failed"] += 1
    finally:
        close_error: Exception | None = None
        if started_backend is not None:
            closer = getattr(started_backend, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:  # noqa: BLE001
                    close_error = exc
        if owned_judge is not None:
            owned_judge.close()
        if owned_segmenter is not None:
            owned_segmenter.close()
        if close_error is not None:
            raise close_error

    stats = ReferenceEditStats(**counters)
    storage.update_stage_counts("reference_edit", stats.to_dict())
    return stats
