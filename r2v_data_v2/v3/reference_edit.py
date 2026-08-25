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
from r2v_data_v2.v3.reference_integrity import (
    _materialize_source_bbox,
    _source_evidence,
)
from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    EntityReferenceState,
    PairingState,
    ReferenceCompleteness,
    ReferenceEditEntityState,
    ReferenceEditState,
    ReferenceVariantState,
    ReferenceVariantsState,
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
    scale_collapse_guard_attempted: int = 0
    scale_collapse_guard_accepted: int = 0
    scale_collapse_guard_rejected: int = 0
    scale_collapse_guard_failed_open: int = 0
    bbox_variants_materialized: int = 0
    bbox_reviews_attempted: int = 0
    bbox_reviews_skipped_background_accepted: int = 0
    background_variants_attempted: int = 0
    background_variants_accepted: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


COMPLETION_PROMPT_TEMPLATE = (
    "图片中只有一个实体，实体是“{entity_phrase}”。"
    "补全残缺的部分。不要引入新的实体，风格保持一致。"
    "如果补全不了，则只保留最能表示该实体的部分，"
    "去除零散且不合理的部分。"
)
_ENTITY_BACKGROUND_DISABLED_REASON = "entity_background_disabled_by_policy"
_ENTITY_COUNTER_FIELDS = (
    "entities_eligible",
    "entities_accepted",
    "entities_fallback",
    "entities_rejected",
    "entities_failed",
)
_SCALE_COLLAPSE_GUARD_COUNTER_FIELDS = (
    "scale_collapse_guard_attempted",
    "scale_collapse_guard_accepted",
    "scale_collapse_guard_rejected",
    "scale_collapse_guard_failed_open",
)
_REFERENCE_VARIANT_COUNTER_FIELDS = (
    "bbox_variants_materialized",
    "bbox_reviews_attempted",
    "bbox_reviews_skipped_background_accepted",
    "background_variants_attempted",
    "background_variants_accepted",
)
_CLIP_COUNTER_FIELDS = (
    *_ENTITY_COUNTER_FIELDS,
    *_SCALE_COLLAPSE_GUARD_COUNTER_FIELDS,
    *_REFERENCE_VARIANT_COUNTER_FIELDS,
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
        return ("complete_entity",)
    return ()


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


def _variant(
    *,
    image_path: str | None,
    status: str,
    reviewed: bool,
    review_status: str,
    reason: str,
    synthetic: bool,
    metadata_path: str | None = None,
    source_frame_index: int | None = None,
) -> ReferenceVariantState:
    return ReferenceVariantState(
        image_path=image_path,
        status=status,
        reviewed=reviewed,
        review_status=review_status,
        reason=reason,
        synthetic=synthetic,
        metadata_path=metadata_path,
        source_frame_index=source_frame_index,
    )


def _variant_state(
    *,
    alpha: ReferenceVariantState,
    bbox: ReferenceVariantState,
    generated_background: ReferenceVariantState,
) -> ReferenceVariantsState:
    return ReferenceVariantsState(
        alpha=alpha,
        bbox=bbox,
        generated_background=generated_background,
    )


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


def _instruction(operation: str, entity: AnnotationEntity) -> str:
    if operation != "complete_entity":
        raise ValueError("fresh production supports completion-only reference editing")
    return COMPLETION_PROMPT_TEMPLATE.format(entity_phrase=entity.phrase)


def _instruction_rewrite_enabled(config: V3Config, operation: str) -> bool:
    if operation == "complete_entity":
        return config.reference_edit.completion_instruction_rewrite_enabled
    raise ValueError(f"unsupported reference-edit operation: {operation}")


def _disabled_entity_background_variant(
    reference: EntityReferenceState,
) -> ReferenceVariantState:
    return _variant(
        image_path=None,
        status="unavailable",
        reviewed=False,
        review_status="not_applicable",
        reason=_ENTITY_BACKGROUND_DISABLED_REASON,
        synthetic=True,
        source_frame_index=reference.source_frame_index,
    )


def _rejected_reference(
    reference: EntityReferenceState,
    reason: str,
) -> EntityReferenceState:
    """Preserve objective source evidence when policy rejects a reference."""
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
        viewpoint=reference.viewpoint,
        independent_reference_value=reference.independent_reference_value,
        requires_substantial_invention=reference.requires_substantial_invention,
        primary_identity_region_visible=reference.primary_identity_region_visible,
        major_structure_visible=reference.major_structure_visible,
        truncation_severity=reference.truncation_severity,
        discrete_foreground_instance=reference.discrete_foreground_instance,
        mask_matches_target=reference.mask_matches_target,
        completion_needed_for_reference_use=(
            reference.completion_needed_for_reference_use
        ),
        detached_target_fragments_present=(
            reference.detached_target_fragments_present
        ),
    )


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
        viewpoint=reference.viewpoint,
        independent_reference_value=reference.independent_reference_value,
        requires_substantial_invention=reference.requires_substantial_invention,
        primary_identity_region_visible=(
            reference.primary_identity_region_visible
        ),
        major_structure_visible=reference.major_structure_visible,
        truncation_severity=reference.truncation_severity,
        discrete_foreground_instance=reference.discrete_foreground_instance,
        mask_matches_target=reference.mask_matches_target,
        completion_needed_for_reference_use=(
            reference.completion_needed_for_reference_use
        ),
        detached_target_fragments_present=(
            reference.detached_target_fragments_present
        ),
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
    bbox_route_judge: object | None = None,
    scale_collapse_judge: object | None = None,
    manage_backend_lifecycle: bool = True,
) -> ReferenceEditStats:
    config.validate()
    if storage.root != config.resolved_run_root:
        raise ValueError("storage run_root does not match reference edit configuration")
    if not config.reference_edit.enabled:
        raise ValueError("V3 reference_edit stage is disabled")

    counters = {field: 0 for field in ReferenceEditStats.__dataclass_fields__}
    active_backend = backend
    active_judge = judge
    active_sam = sam_reviewer
    del bbox_route_judge, scale_collapse_judge
    owned_backend: BooguSubprocessBackend | None = None
    owned_judge: QwenBooguReferenceEditJudge | None = None
    owned_segmenter: Sam3SegmentationBackend | None = None
    started_backend: object | None = None
    runtime_ready = False

    def initialize_runtime() -> tuple[
        BooguReferenceEditBackend,
        BooguReferenceEditJudge,
        BooguSamReviewer | None,
    ]:
        nonlocal active_backend
        nonlocal active_judge
        nonlocal active_sam
        nonlocal owned_backend
        nonlocal owned_judge
        nonlocal owned_segmenter
        nonlocal started_backend
        nonlocal runtime_ready

        if runtime_ready:
            assert active_backend is not None
            assert active_judge is not None
            return active_backend, active_judge, active_sam
        if active_backend is None:
            owned_backend = BooguSubprocessBackend(
                BooguWorkerConfig(
                    python_executable=config.reference_edit.python_executable,
                    code_root=config.reference_edit.code_root,
                    model_path=config.reference_edit.model_path,
                    model_revision=config.reference_edit.model_revision,
                    cuda_visible_devices=config.reference_edit.cuda_visible_devices,
                    timeout_seconds=config.reference_edit.timeout_seconds,
                    temporary_root=storage.reference_edit_temporary_dir(),
                )
            )
            active_backend = owned_backend
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
        starter = getattr(active_backend, "start", None)
        if manage_backend_lifecycle and callable(starter):
            starter(stderr_log_path=storage.reference_edit_worker_log_path())
            started_backend = active_backend
            counters["worker_starts"] += 1
        runtime_ready = True
        return active_backend, active_judge, active_sam

    try:
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
            clip_entity_counters = {field: 0 for field in _CLIP_COUNTER_FIELDS}
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
                    initial_route = _route(reference)
                    if (
                        entities[reference.entity_id].reference_type
                        not in {"subject", "object"}
                        and reference.completeness is not None
                        and not _operations(initial_route)
                    ):
                        final_references.append(reference)
                        edit_states.append(
                            ReferenceEditEntityState(
                                entity_id=reference.entity_id,
                                route=initial_route,
                                status="not_required",
                                source_reference=reference,
                                source_image_path=reference.image_path,
                                output_image_path=reference.image_path,
                            )
                        )
                        continue
                    clip_entity_counters["entities_eligible"] += 1
                    if reference.image_path is None:
                        raise ValueError("ready reference has no image_path")
                    entity = entities[reference.entity_id]
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
                    entity_variant_route = bool(
                        entity.reference_type in {"subject", "object"}
                        and route in {"complete", "local_usable", "repairable"}
                    )
                    bbox_evaluation = None
                    bbox_materialization_error: str | None = None
                    current_reference: Image.Image | None = None
                    if entity_variant_route:
                        try:
                            evidence = _source_evidence(
                                storage,
                                clip_uid=clip.clip_uid,
                                reference=reference,
                            )
                            with Image.open(
                                _resolve_artifact(storage, reference.image_path)
                            ) as opened:
                                opened.load()
                                current_reference = opened.copy()
                            bbox_evaluation = _materialize_source_bbox(
                                storage=storage,
                                clip_uid=clip.clip_uid,
                                entity_id=reference.entity_id,
                                source_evidence=evidence,
                                current_reference=current_reference,
                                current_reference_path=_resolve_artifact(
                                    storage,
                                    reference.image_path,
                                ),
                                reference=reference,
                                original_review=None,
                                diagnostics=None,
                                crop_padding_ratio=config.pair.crop_padding_ratio,
                                trigger="variant_bbox_review",
                            )
                            clip_entity_counters["bbox_variants_materialized"] += 1
                        except Exception as exc:  # noqa: BLE001 - alpha remains valid
                            bbox_materialization_error = f"{type(exc).__name__}:{exc}"
                    alpha_variant = (
                        _variant(
                            image_path=reference.image_path,
                            status="accepted",
                            reviewed=True,
                            review_status="accepted_pair_reference",
                            reason="pair_accepted_source_alpha",
                            synthetic=False,
                            source_frame_index=reference.source_frame_index,
                        )
                        if entity_variant_route
                        else None
                    )
                    bbox_variant = (
                        _variant(
                            image_path=(
                                bbox_evaluation.candidate_relative
                                if bbox_evaluation is not None
                                else None
                            ),
                            status=(
                                "available"
                                if bbox_evaluation is not None
                                else "unavailable"
                            ),
                            reviewed=False,
                            review_status=(
                                "not_reviewed"
                                if bbox_evaluation is not None
                                else "materialization_failed"
                            ),
                            reason=(
                                "source_bbox_materialized"
                                if bbox_evaluation is not None
                                else bbox_materialization_error
                                or "source_bbox_unavailable"
                            ),
                            synthetic=False,
                            metadata_path=(
                                bbox_evaluation.metadata_relative
                                if bbox_evaluation is not None
                                else None
                            ),
                            source_frame_index=reference.source_frame_index,
                        )
                        if entity_variant_route
                        else None
                    )
                    generated_variant = (
                        _disabled_entity_background_variant(reference)
                        if entity_variant_route
                        else None
                    )
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
                                variants=(
                                    _variant_state(
                                        alpha=alpha_variant,
                                        bbox=bbox_variant,
                                        generated_background=generated_variant,
                                    )
                                    if alpha_variant is not None
                                    and bbox_variant is not None
                                    and generated_variant is not None
                                    else None
                                ),
                                default_variant=(
                                    "alpha" if entity_variant_route else None
                                ),
                                default_image_path=(
                                    reference.image_path
                                    if entity_variant_route
                                    else None
                                ),
                                default_reason=(
                                    "source_alpha_preferred_by_policy"
                                    if entity_variant_route
                                    else None
                                ),
                            )
                        )
                        clip_entity_counters["entities_accepted"] += 1
                        continue
                    (
                        operation_backend,
                        operation_judge,
                        operation_sam,
                    ) = initialize_runtime()
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
                            backend=operation_backend,
                            judge=operation_judge,
                            sam_reviewer=operation_sam,
                            instruction_rewrite_enabled=_instruction_rewrite_enabled(
                                config,
                                operation,
                            ),
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
                    rejected_result: BooguReferenceEditResult | None = None
                    rejection_reason: str | None = None
                    last_result = results[attempted_operations[-1]]
                    if last_result.status != "accepted":
                        rejected_result = last_result
                        rejection_reason = _rejection_reason(rejected_result)
                        if rejection_reason.startswith(
                            "boogu_reference_edit_failed:"
                        ):
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
                    if route == "repairable":
                        if (
                            completion_result is not None
                            and completion_result.status == "accepted"
                            and completion_result.candidate_path is not None
                        ):
                            publication = publish_boogu_final_reference(
                                run_root=storage.root,
                                clip_uid=clip.clip_uid,
                                entity_id=reference.entity_id,
                                candidate_path=completion_result.candidate_path,
                                selected_metadata_path=completion_result.metadata_path,
                                completion_metadata_path=completion_result.metadata_path,
                                final_selection="completion_candidate",
                                final_selection_reason=(
                                    "accepted_completion_preferred_over_source_alpha"
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
                                    status="accepted",
                                    source_reference=reference,
                                    source_image_path=reference.image_path,
                                    output_image_path=accepted.image_path,
                                    variants=(
                                        _variant_state(
                                            alpha=alpha_variant,
                                            bbox=bbox_variant,
                                            generated_background=generated_variant,
                                        )
                                        if alpha_variant is not None
                                        and bbox_variant is not None
                                        and generated_variant is not None
                                        else None
                                    ),
                                    default_variant=(
                                        "accepted_base"
                                        if entity_variant_route
                                        else None
                                    ),
                                    default_image_path=(
                                        accepted.image_path
                                        if entity_variant_route
                                        else None
                                    ),
                                    default_reason=(
                                        "completion_review_accepted"
                                        if entity_variant_route
                                        else None
                                    ),
                                    accepted_base_image_path=(
                                        accepted.image_path
                                        if entity_variant_route
                                        else None
                                    ),
                                    operation="complete_entity",
                                    metadata_path=accepted.generation_metadata_path,
                                    operations=attempted_operations,
                                    completion_metadata_path=(
                                        storage.relative_artifact_path(
                                            completion_result.metadata_path
                                        )
                                    ),
                                )
                            )
                            clip_entity_counters["entities_accepted"] += 1
                            continue

                        assert completion_result is not None
                        fallback_reason = (
                            "repairable_completion_rejected:"
                            f"{rejection_reason or 'completion_unavailable'}"
                        )
                        final_metadata_path = _write_source_selection_metadata(
                            storage,
                            clip_uid=clip.clip_uid,
                            reference=reference,
                            geometry=source_geometry,
                            source_gate_reason=None,
                            reason=fallback_reason,
                            operation_metadata_path=completion_result.metadata_path,
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
                                variants=(
                                    _variant_state(
                                        alpha=alpha_variant,
                                        bbox=bbox_variant,
                                        generated_background=generated_variant,
                                    )
                                    if alpha_variant is not None
                                    and bbox_variant is not None
                                    and generated_variant is not None
                                    else None
                                ),
                                default_variant=(
                                    "alpha" if entity_variant_route else None
                                ),
                                default_image_path=(
                                    reference.image_path
                                    if entity_variant_route
                                    else None
                                ),
                                default_reason=(
                                    "completion_rejected_fallback_to_source_alpha"
                                    if entity_variant_route
                                    else None
                                ),
                                accepted_base_image_path=(
                                    reference.image_path
                                    if entity_variant_route
                                    else None
                                ),
                                operation="complete_entity",
                                metadata_path=storage.relative_artifact_path(
                                    final_metadata_path
                                ),
                                operations=attempted_operations,
                                completion_metadata_path=(
                                    storage.relative_artifact_path(
                                        completion_result.metadata_path
                                    )
                                ),
                                fallback_policy="keep_source",
                                reason=fallback_reason,
                            )
                        )
                        clip_entity_counters["entities_fallback"] += 1
                        continue

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
                        reason="no_qualifying_ready_reference",
                    )
                else:
                    pairing = PairingState(
                        status="ready",
                        retained_entity_ids=retained,
                        tokens=_tokens_for_retained(retained, entities),
                        background_token=clip.pairing.background_token,
                    )
                classified_entities = sum(
                    clip_entity_counters[field]
                    for field in (
                        "entities_accepted",
                        "entities_fallback",
                        "entities_rejected",
                    )
                )
                if classified_entities != clip_entity_counters["entities_eligible"]:
                    raise RuntimeError(
                        "reference edit entity outcomes must match eligible count"
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
    if stats.scale_collapse_guard_attempted != (
        stats.scale_collapse_guard_accepted
        + stats.scale_collapse_guard_rejected
        + stats.scale_collapse_guard_failed_open
    ):
        raise RuntimeError("scale-collapse guard outcomes must match attempts")
    if stats.background_variants_accepted > stats.background_variants_attempted:
        raise RuntimeError("accepted Boogu background candidates exceed attempts")
    if (
        stats.bbox_reviews_skipped_background_accepted
        > stats.bbox_variants_materialized
        or stats.bbox_reviews_attempted > stats.bbox_variants_materialized
    ):
        raise RuntimeError("bbox variant review counts exceed materialized variants")
    storage.update_stage_counts("reference_edit", stats.to_dict())
    return stats
