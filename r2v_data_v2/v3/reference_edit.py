from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguReferenceEditBackend,
    BooguReferenceEditJudge,
    BooguReferenceEditResult,
    BooguSamReviewer,
    BooguSubprocessBackend,
    BooguWorkerConfig,
    QwenBooguReferenceEditJudge,
    Sam3BooguReferenceReviewer,
    publish_boogu_final_reference,
    run_boogu_reference_edit,
)
from r2v_data_v2.v3.reference_geometry import (
    ContentGeometry,
    content_geometry_from_rgba,
    source_geometry_metadata,
    tiny_content_reason,
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


COMPLETION_PROMPT_TEMPLATE = (
    "图片中只有一个实体，实体是“{entity_phrase}”。"
    "补全残缺的部分。不要引入新的实体，风格保持一致。"
    "如果补全不了，则只保留最能表示该实体的部分，"
    "去除零散且不合理的部分。"
)
BACKGROUND_PROMPT = "给图像添加符合风格的背景，不要增加任何实例。"
_ENTITY_COUNTER_FIELDS = (
    "entities_accepted",
    "entities_fallback",
    "entities_rejected",
    "entities_failed",
)


def _route(
    reference: EntityReferenceState,
    *,
    source_touches_boundary: bool = False,
) -> ReferenceCompleteness:
    if reference.completeness is not None:
        return reference.completeness
    if (
        reference.reference_scope == "local"
        and not reference.whole_entity_recognizable
        and source_touches_boundary
    ):
        return "repairable"
    return "complete" if reference.reference_scope == "full" else "local_usable"


def _operations(route: ReferenceCompleteness) -> tuple[str, ...]:
    if route == "repairable":
        return ("complete_entity", "add_entity_background")
    if route in {"complete", "local_usable"}:
        return ("add_entity_background",)
    return ()


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
            if reference.status != "ready" or reference.image_path is None:
                continue
            geometry = _reference_content_geometry(storage, reference)
            if _source_gate_reason(config, geometry) is not None:
                continue
            route = _route(
                reference,
                source_touches_boundary=geometry.touches_canvas_boundary,
            )
            if _operations(route):
                count += 1
    return count


def _resolve_artifact(storage: RunStorage, value: str) -> Path:
    root = storage.root.resolve(strict=False)
    path = (root / value).resolve(strict=False)
    if root not in path.parents or not path.is_file():
        raise ValueError("reference edit input must be an existing run artifact")
    return path


def _reference_content_geometry(
    storage: RunStorage,
    reference: EntityReferenceState,
) -> ContentGeometry:
    if reference.image_path is None:
        raise ValueError("ready reference has no image_path")
    path = _resolve_artifact(storage, reference.image_path)
    with Image.open(path) as opened:
        if opened.format != "PNG":
            raise ValueError("reference edit source must be PNG")
        opened.load()
        source_rgba = opened.convert("RGBA")
    return content_geometry_from_rgba(source_rgba)


def _source_gate_reason(
    config: V3Config,
    geometry: ContentGeometry,
) -> str | None:
    return tiny_content_reason(
        geometry,
        minimum_area_pixels=config.reference_edit.min_source_content_area_pixels,
        minimum_long_side_pixels=(
            config.reference_edit.min_source_content_long_side_pixels
        ),
    )


def _write_source_selection_metadata(
    storage: RunStorage,
    *,
    clip_uid: str,
    reference: EntityReferenceState,
    geometry: ContentGeometry,
    source_gate_reason: str | None,
    reason: str,
    operation_metadata_path: Path | None = None,
) -> Path:
    if reference.image_path is None:
        raise ValueError("source selection requires a reference image")
    source_path = _resolve_artifact(storage, reference.image_path)
    metadata_path = storage.reference_edit_dir(clip_uid) / reference.entity_id
    metadata_path.mkdir(parents=True, exist_ok=True)
    final_metadata_path = metadata_path / "final_metadata.json"
    payload = {
        "schema_version": 1,
        "status": "fallback",
        "clip_uid": clip_uid,
        "entity_id": reference.entity_id,
        "final_selection": "source",
        "final_selection_reason": reason,
        "final_reference_path": reference.image_path,
        "final_reference_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selected_operation_metadata_path": (
            storage.relative_artifact_path(operation_metadata_path)
            if operation_metadata_path is not None
            else None
        ),
        **source_geometry_metadata(
            geometry,
            source_gate_reason=source_gate_reason,
        ),
    }
    write_json_atomic(final_metadata_path, payload)
    return final_metadata_path


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
        return COMPLETION_PROMPT_TEMPLATE.format(entity_phrase=entity.phrase)
    return BACKGROUND_PROMPT


def _accepted_reference(
    storage: RunStorage,
    reference: EntityReferenceState,
    *,
    clip_uid: str,
    output_path: Path,
    metadata_path: Path,
    preserve_local_scope: bool = False,
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
        reference_scope=(reference.reference_scope if preserve_local_scope else "full"),
        visible_region=(reference.visible_region if preserve_local_scope else "whole"),
        whole_entity_recognizable=(
            reference.whole_entity_recognizable if preserve_local_scope else True
        ),
        identity_features_visible=(
            reference.identity_features_visible if preserve_local_scope else True
        ),
        scope_reason="accepted_boogu_reference_edit",
        image_path=storage.relative_artifact_path(output_path),
        source_frame_index=reference.source_frame_index,
        source_clip_uid=clip_uid,
        source_entity_id=reference.entity_id,
        image_quality=(
            reference.image_quality or "acceptable"
            if preserve_local_scope
            else "high"
        ),
        completeness="local_usable" if preserve_local_scope else "complete",
        synthetic=True,
        generation_metadata_path=storage.relative_artifact_path(metadata_path),
        generation_source_sha256=source_sha256,
        generation_output_sha256=output_sha256,
    )


def _rejection_reason(result: BooguReferenceEditResult) -> str:
    reason = "boogu_candidate_rejected"
    if result.rejection_path is None:
        return reason
    payload = json.loads(result.rejection_path.read_text(encoding="utf-8"))
    raw_reason = payload.get("reason")
    if isinstance(raw_reason, str) and raw_reason.strip():
        return raw_reason.strip()
    return reason


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
                    min_candidate_scale_ratio=(
                        config.reference_edit.min_candidate_scale_ratio
                    ),
                    max_candidate_center_shift=(
                        config.reference_edit.max_candidate_center_shift
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
            clip_entity_counters = {field: 0 for field in _ENTITY_COUNTER_FIELDS}
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
                    source_geometry = _reference_content_geometry(storage, reference)
                    source_gate_reason = _source_gate_reason(config, source_geometry)
                    route = _route(
                        reference,
                        source_touches_boundary=(
                            source_geometry.touches_canvas_boundary
                        ),
                    )
                    if source_gate_reason is not None:
                        final_metadata_path = _write_source_selection_metadata(
                            storage,
                            clip_uid=clip.clip_uid,
                            reference=reference,
                            geometry=source_geometry,
                            source_gate_reason=source_gate_reason,
                            reason="tiny_source_entity",
                        )
                        final_references.append(reference)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="fallback",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=reference.image_path,
                                metadata_path=storage.relative_artifact_path(
                                    final_metadata_path
                                ),
                                fallback_policy="keep_source",
                                reason="tiny_source_entity",
                            )
                        )
                        clip_entity_counters["entities_fallback"] += 1
                        continue
                    operations = _operations(route)
                    if not operations:
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
                    results: dict[str, BooguReferenceEditResult] = {}
                    source_image_path: Path | None = None
                    geometry_source_image_path = _resolve_artifact(
                        storage,
                        reference.image_path,
                    )
                    for operation in operations:
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
                            min_source_content_area_pixels=(
                                config.reference_edit.min_source_content_area_pixels
                            ),
                            min_source_content_long_side_pixels=(
                                config.reference_edit.min_source_content_long_side_pixels
                            ),
                            model_revision=config.reference_edit.model_revision,
                            fallback_status=config.reference_edit.fallback_policy,
                            source_image_path=source_image_path,
                            geometry_source_image_path=geometry_source_image_path,
                            publish_final=False,
                            overwrite=overwrite,
                        )
                        results[operation] = result
                        if result.status != "accepted":
                            break
                        if operation == "complete_entity":
                            if result.candidate_path is None:
                                raise RuntimeError(
                                    "accepted completion has no candidate artifact"
                                )
                            source_image_path = result.candidate_path

                    attempted_operations = list(results)
                    completion_result = results.get("complete_entity")
                    background_result = results.get("add_entity_background")
                    if len(results) == len(operations) and all(
                        result.status == "accepted" for result in results.values()
                    ):
                        selected = results[operations[-1]]
                        if selected.candidate_path is None:
                            raise RuntimeError(
                                "accepted reference edit has no candidate artifact"
                            )
                        publication = publish_boogu_final_reference(
                            run_root=storage.root,
                            clip_uid=clip.clip_uid,
                            entity_id=reference.entity_id,
                            candidate_path=selected.candidate_path,
                            selected_metadata_path=selected.metadata_path,
                            completion_metadata_path=(
                                completion_result.metadata_path
                                if completion_result is not None
                                else None
                            ),
                            background_metadata_path=(
                                background_result.metadata_path
                                if background_result is not None
                                else None
                            ),
                            final_selection="background_candidate",
                            final_selection_reason=(
                                "accepted_background_candidate_preferred_over_source"
                            ),
                        )
                        accepted = _accepted_reference(
                            storage,
                            reference,
                            clip_uid=clip.clip_uid,
                            output_path=publication.final_reference_path,
                            metadata_path=publication.final_metadata_path,
                            preserve_local_scope=route == "local_usable",
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
                                operation=operations[-1],
                                metadata_path=accepted.generation_metadata_path,
                                operations=attempted_operations,
                                completion_metadata_path=(
                                    storage.relative_artifact_path(
                                        completion_result.metadata_path
                                    )
                                    if completion_result is not None
                                    else None
                                ),
                                background_metadata_path=(
                                    storage.relative_artifact_path(
                                        background_result.metadata_path
                                    )
                                    if background_result is not None
                                    else None
                                ),
                            )
                        )
                        clip_entity_counters["entities_accepted"] += 1
                        continue

                    rejected_result = results[attempted_operations[-1]]
                    rejection_reason = _rejection_reason(rejected_result)
                    if rejection_reason.startswith("boogu_reference_edit_failed:"):
                        clip_entity_counters["entities_failed"] += 1
                        storage.append_failure(
                            stage="reference_edit",
                            clip_uid=clip.clip_uid,
                            reason=rejection_reason,
                            details={
                                "entity_id": reference.entity_id,
                                "operation": rejected_result.operation,
                            },
                        )
                    if (
                        route == "repairable"
                        and completion_result is not None
                        and completion_result.status == "accepted"
                        and completion_result.candidate_path is not None
                        and background_result is not None
                    ):
                        publication = publish_boogu_final_reference(
                            run_root=storage.root,
                            clip_uid=clip.clip_uid,
                            entity_id=reference.entity_id,
                            candidate_path=completion_result.candidate_path,
                            selected_metadata_path=completion_result.metadata_path,
                            completion_metadata_path=completion_result.metadata_path,
                            background_metadata_path=background_result.metadata_path,
                            background_fallback="completion_candidate",
                            final_selection="completion_candidate",
                            final_selection_reason=(
                                "background_rejected_completion_candidate_preserved"
                            ),
                        )
                        accepted = _accepted_reference(
                            storage,
                            reference,
                            clip_uid=clip.clip_uid,
                            output_path=publication.final_reference_path,
                            metadata_path=publication.final_metadata_path,
                        )
                        final_references.append(accepted)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="fallback",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=accepted.image_path,
                                operation="add_entity_background",
                                metadata_path=accepted.generation_metadata_path,
                                operations=attempted_operations,
                                completion_metadata_path=(
                                    storage.relative_artifact_path(
                                        completion_result.metadata_path
                                    )
                                ),
                                background_metadata_path=(
                                    storage.relative_artifact_path(
                                        background_result.metadata_path
                                    )
                                ),
                                background_fallback="completion_candidate",
                                fallback_policy="completion_candidate",
                                reason=rejection_reason,
                            )
                        )
                        clip_entity_counters["entities_fallback"] += 1
                        continue
                    force_keep_source = (
                        rejected_result.operation == "add_entity_background"
                        or route == "repairable"
                        and completion_result is not None
                        and completion_result.status != "accepted"
                    )
                    if (
                        force_keep_source
                        or config.reference_edit.fallback_policy == "keep_source"
                    ):
                        final_metadata_path = _write_source_selection_metadata(
                            storage,
                            clip_uid=clip.clip_uid,
                            reference=reference,
                            geometry=source_geometry,
                            source_gate_reason=None,
                            reason=rejection_reason,
                            operation_metadata_path=rejected_result.metadata_path,
                        )
                        final_references.append(reference)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=route,
                                status="fallback",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=reference.image_path,
                                operation=rejected_result.operation,
                                metadata_path=storage.relative_artifact_path(
                                    final_metadata_path
                                ),
                                operations=attempted_operations,
                                completion_metadata_path=(
                                    storage.relative_artifact_path(
                                        completion_result.metadata_path
                                    )
                                    if completion_result is not None
                                    else None
                                ),
                                background_metadata_path=(
                                    storage.relative_artifact_path(
                                        background_result.metadata_path
                                    )
                                    if background_result is not None
                                    else None
                                ),
                                fallback_policy="keep_source",
                                reason=rejection_reason,
                            )
                        )
                        clip_entity_counters["entities_fallback"] += 1
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
                                operation=rejected_result.operation,
                                metadata_path=storage.relative_artifact_path(
                                    rejected_result.metadata_path
                                ),
                                operations=attempted_operations,
                                completion_metadata_path=(
                                    storage.relative_artifact_path(
                                        completion_result.metadata_path
                                    )
                                    if completion_result is not None
                                    else None
                                ),
                                background_metadata_path=(
                                    storage.relative_artifact_path(
                                        background_result.metadata_path
                                    )
                                    if background_result is not None
                                    else None
                                ),
                                fallback_policy="reject_entity",
                                reason=rejection_reason,
                            )
                        )
                        clip_entity_counters["entities_rejected"] += 1

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
                for field, value in clip_entity_counters.items():
                    counters[field] += value
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
