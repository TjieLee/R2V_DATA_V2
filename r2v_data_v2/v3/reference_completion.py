"""Thin optional localized-completion fallback for V3 pairing."""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.frames import _save_jpeg
from r2v_data_v2.v3.reference_completion_benchmark import (
    ReferenceCompletionReview,
)
from r2v_data_v2.v3.reference_completion_policy import (
    ConservativeCompletionPolicy,
    build_conservative_completion_policy,
    restore_protected_source_pixels,
    validate_conservative_completion_candidate,
)
from r2v_data_v2.v3.reference_completion_publish import (
    PublicationConfig,
    _load_source_rgba,
    _mask_from_track_result,
    _save_png,
    build_candidate_reference,
    compute_foreground_metrics,
    evaluate_background_gate,
    evaluate_mask_gate,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
    DEFAULT_QWEN_SEEDS,
    QWEN_LOCALIZED_PROMPT_EN_SHORT,
    QwenImageEdit2511CompletionConfig,
    QwenImageEdit2511ReferenceCompletionBackend,
    QwenLocalizedCompletionJudge,
    QwenLocalizedReferenceCompletionJudge,
    QwenReferenceCompletionBackend,
    _localized_hard_check,
    _localized_prompt,
)
from r2v_data_v2.v3.reference_judge import (
    EntityReferenceJudge,
    validate_entity_reference_decision,
)
from r2v_data_v2.v3.sam3_backend import (
    Sam3SegmentationBackend,
    SegmentationBackend,
)
from r2v_data_v2.v3.schemas import (
    AnnotationEntity,
    EntityReferenceState,
    ReferencesState,
)
from r2v_data_v2.v3.storage import RunStorage

_SCHEMA_VERSION = "r2v.v3.reference_completion.1"
_REJECTION_SCHEMA_VERSION = "r2v.v3.reference_completion.rejection.1"
_REJECTION_STAGES = frozenset(
    {
        "generation",
        "localized_hard_check",
        "segmentation",
        "mask_gate",
        "improvement_gate",
        "background_gate",
        "localized_judge",
        "reference_ranking",
        "publication",
    }
)


@dataclass(frozen=True)
class ReferenceCompletionFallbackStats:
    attempted: int = 0
    ready: int = 0
    rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _RejectionDiagnostics:
    stage: str = "generation"
    source_image_sha256: str | None = None
    details: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, Image.Image] = field(default_factory=dict)

    def set_stage(self, stage: str) -> None:
        if stage not in _REJECTION_STAGES:
            raise ValueError(f"unsupported completion diagnostic stage: {stage}")
        self.stage = stage


class _ExpectedRejection(ValueError):
    """A deterministic reject, rather than an unexpected execution error."""


@dataclass(frozen=True)
class _AcceptedFallback:
    working_directory: Path
    state: EntityReferenceState


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_size_diagnostics(
    backend: QwenReferenceCompletionBackend,
) -> dict[str, object]:
    value = getattr(backend, "last_size_diagnostics", None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("completion backend size diagnostics must be an object")
    return dict(value)


def _artifact(storage: RunStorage, value: str) -> Path:
    root = storage.root.resolve(strict=False)
    path = (root / value).resolve(strict=False)
    if root not in path.parents or not path.is_file():
        raise ValueError("completion artifact must be an existing run artifact")
    return path


def _completion_directory(
    storage: RunStorage,
    clip_uid: str,
    entity_id: str,
) -> Path:
    storage.selected_entity_path(clip_uid, entity_id)
    return storage.clip_dir(clip_uid) / "reference_completion" / entity_id


def _write_diagnostic_artifacts(
    directory: Path,
    diagnostics: _RejectionDiagnostics,
) -> None:
    if not diagnostics.artifacts:
        return
    diagnostic_directory = directory / "diagnostics"
    diagnostic_directory.mkdir(parents=True, exist_ok=False)
    for name, image in sorted(diagnostics.artifacts.items()):
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError("completion diagnostic artifact name is invalid")
        _save_png(diagnostic_directory / name, image)


def _policy_artifact_paths() -> dict[str, str]:
    return {
        "cleaned_source_image": "diagnostics/cleaned_source.png",
        "protected_component_mask": "diagnostics/protected_component_mask.png",
        "recovery_corridor_mask": "diagnostics/recovery_corridor_mask.png",
    }


def _eligible(
    clip_uid: str,
    entity: AnnotationEntity,
    state: EntityReferenceState | None,
) -> bool:
    if (
        state is None
        or state.status != "ready"
        or state.reference_scope != "local"
        or not state.identity_features_visible
        or state.synthetic
    ):
        return False
    return (state.source_clip_uid, state.source_entity_id) in {
        (None, None),
        (clip_uid, entity.entity_id),
    }


def _candidate(
    entity: AnnotationEntity,
    image: Image.Image,
    mask_image: Image.Image,
    *,
    image_path: str,
    source_frame_index: int,
) -> Any:
    from r2v_data_v2.v3.pair import EntityReferenceCandidate

    mask = np.asarray(mask_image, dtype=np.uint8) == 255
    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError("generated fallback mask is empty")
    left, top, right, bottom = (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
    area = int(mask.sum())
    center_distance = math.hypot(
        (left + right) / 2 - image.width / 2,
        (top + bottom) / 2 - image.height / 2,
    ) / math.hypot(image.width, image.height)
    return EntityReferenceCandidate(
        candidate_id="candidate_1",
        entity_id=entity.entity_id,
        frame_slot=0,
        source_frame_index=source_frame_index,
        image_path=image_path,
        mask=mask.copy(),
        bbox_xyxy=(left, top, right, bottom),
        area_pixels=area,
        area_ratio=area / (image.width * image.height),
        bbox_fill_ratio=area / ((right - left) * (bottom - top)),
        border_contact_count=sum(
            (
                bool(mask[0].any()),
                bool(mask[-1].any()),
                bool(mask[:, 0].any()),
                bool(mask[:, -1].any()),
            )
        ),
        normalized_center_distance=center_distance,
    )


def _attempt(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    completion_backend: QwenReferenceCompletionBackend,
    completion_judge: QwenLocalizedCompletionJudge,
    segmenter: SegmentationBackend,
    reference_judge: EntityReferenceJudge,
    gates: PublicationConfig,
    diagnostics: _RejectionDiagnostics,
    policy: ConservativeCompletionPolicy,
) -> _AcceptedFallback:
    if source_state.image_path is None or source_state.source_frame_index is None:
        raise ValueError("local reference provenance is incomplete")
    source_path = _artifact(storage, source_state.image_path)
    source_rgba, source_bytes, source_sha256 = _load_source_rgba(source_path)
    diagnostics.source_image_sha256 = source_sha256
    final_directory = _completion_directory(storage, clip_uid, entity.entity_id)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    working = final_directory.parent / f".tmp-{entity.entity_id}-{uuid.uuid4().hex}"
    working.mkdir(parents=False, exist_ok=False)
    relative = lambda name: storage.relative_artifact_path(
        final_directory / name
    )
    try:
        (working / "source_rgba.png").write_bytes(source_bytes)
        _write_diagnostic_artifacts(working, diagnostics)
        input_rgb = policy.completion_input_rgb.copy()
        input_path = working / "input_rgb.png"
        _save_png(input_path, input_rgb)
        prompt, prompt_language = _localized_prompt(
            entity.reference_type,
            QWEN_LOCALIZED_PROMPT_EN_SHORT,
        )
        seed = DEFAULT_QWEN_SEEDS[0]
        diagnostics.set_stage("generation")
        completion_size_metadata: dict[str, object] = {}
        try:
            generated = completion_backend.complete(
                input_rgb=input_rgb.copy(),
                entity_phrase=entity.phrase,
                seed=seed,
                prompt=prompt,
                negative_prompt=DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
            )
        finally:
            completion_size_metadata = _completion_size_diagnostics(
                completion_backend
            )
            diagnostics.details.update(completion_size_metadata)
        candidate_path = working / "candidate_rgb.png"
        diagnostics.set_stage("localized_hard_check")
        (
            candidate_rgb,
            generated_candidate_sha256,
            hard_check,
        ) = _localized_hard_check(
            generated=generated,
            input_rgb=input_rgb,
            input_path=input_path,
            candidate_path=candidate_path,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        diagnostics.details["localized_hard_check"] = hard_check.to_dict()
        if candidate_rgb is None or not hard_check.passed:
            raise _ExpectedRejection(
                "localized completion hard check failed: "
                + ",".join(hard_check.reasons)
            )
        diagnostics.set_stage("segmentation")
        sam3_frames_directory = working / ".sam3_single_frame"
        sam3_frames_directory.mkdir()
        isolated_frame_path = sam3_frames_directory / "00.jpg"
        try:
            _save_jpeg(candidate_rgb, isolated_frame_path)
            track = segmenter.track(
                frame_paths=[isolated_frame_path],
                entity_id=entity.entity_id,
                reference_type=entity.reference_type,
                grounding_prompt=entity.phrase,
            )
        finally:
            shutil.rmtree(sam3_frames_directory, ignore_errors=True)
        segmentation_details: dict[str, object] = {
            "track_status": getattr(track, "status", None),
            "track_reason": getattr(track, "reason", None),
        }
        diagnostics.details["segmentation"] = segmentation_details
        try:
            mask, mask_ranking = _mask_from_track_result(
                track,
                expected_size=candidate_rgb.size,
            )
        except ValueError as exc:
            raise _ExpectedRejection(str(exc)) from exc
        segmentation_details["ranking"] = mask_ranking
        diagnostics.artifacts["sam3_candidate_mask.png"] = mask.copy()
        _save_png(
            working / "diagnostics" / "sam3_candidate_mask.png",
            mask,
        )
        diagnostics.set_stage("improvement_gate")
        conservative = validate_conservative_completion_candidate(policy, mask)
        diagnostics.details["candidate_new_region_stats"] = conservative.report
        if not conservative.passed:
            raise _ExpectedRejection(conservative.reasons[0])
        mask = conservative.mask
        candidate_rgb = restore_protected_source_pixels(policy, candidate_rgb)
        candidate_sha256 = _save_png(candidate_path, candidate_rgb)
        mask_sha256 = _save_png(working / "candidate_mask.png", mask)
        source_metrics = policy.source_metrics
        candidate_metrics = compute_foreground_metrics(mask)
        diagnostics.set_stage("mask_gate")
        mask_metrics, _, raw_mask_reasons = evaluate_mask_gate(
            mask,
            reference_type=entity.reference_type,
            config=gates,
        )
        component_reasons = {
            "largest_component_ratio_below_min",
            "secondary_component_ratio_above_max",
            "group_largest_component_ratio_below_min",
            "group_secondary_component_ratio_above_max",
            "group_significant_component_isolated",
        }
        mask_reasons = [
            reason
            for reason in raw_mask_reasons
            if reason not in component_reasons
        ]
        mask_ok = not mask_reasons
        diagnostics.details["mask_gate"] = {
            "passed": mask_ok,
            "reasons": list(mask_reasons),
            "suppressed_protected_component_reasons": [
                reason
                for reason in raw_mask_reasons
                if reason in component_reasons
            ],
            "metrics": mask_metrics,
        }
        diagnostics.set_stage("improvement_gate")
        improvement = conservative.report
        improvement_ok = conservative.passed
        improvement_reasons = list(conservative.reasons)
        diagnostics.details["improvement_gate"] = {
            "passed": improvement_ok,
            "reasons": list(improvement_reasons),
            "metrics": improvement,
            "source_metrics": source_metrics,
            "candidate_metrics": candidate_metrics,
        }
        diagnostics.set_stage("background_gate")
        background, background_ok, background_reasons = evaluate_background_gate(
            candidate_rgb,
            mask,
            config=gates,
        )
        diagnostics.details["background_gate"] = {
            "passed": background_ok,
            "reasons": list(background_reasons),
            "metrics": background,
        }
        reasons = [*mask_reasons, *improvement_reasons, *background_reasons]
        if not all((mask_ok, improvement_ok, background_ok)):
            if not mask_ok:
                diagnostics.set_stage("mask_gate")
            elif not improvement_ok:
                diagnostics.set_stage("improvement_gate")
            else:
                diagnostics.set_stage("background_gate")
            raise _ExpectedRejection(
                f"completion production gate failed: {reasons[0]}"
            )
        diagnostics.set_stage("localized_judge")
        review = completion_judge.review(
            source_rgba=source_rgba.copy(),
            candidate_rgb=candidate_rgb.copy(),
            entity_phrase=entity.phrase,
            reference_type=entity.reference_type,
        )
        if not isinstance(review, ReferenceCompletionReview):
            raise TypeError("localized completion judge returned an invalid result")
        diagnostics.details["localized_judge"] = {
            "verdict": review.verdict,
            "reason": review.reason,
        }
        if review.verdict != "accept":
            raise _ExpectedRejection(
                "localized completion judge rejected the candidate"
            )
        diagnostics.set_stage("reference_ranking")
        ranked = _candidate(
            entity,
            candidate_rgb,
            mask,
            image_path=relative("candidate_rgb.png"),
            source_frame_index=source_state.source_frame_index,
        )
        rank_attempt = reference_judge.decide(
            entity=entity,
            candidates=[ranked],
            source_images={ranked.image_path: candidate_rgb.copy()},
        )
        issues = validate_entity_reference_decision(
            rank_attempt.decision,
            candidate_ids={"candidate_1"},
            reference_type=entity.reference_type,
            candidate_by_id={"candidate_1": ranked},
        )
        diagnostics.details["reference_ranking"] = {
            "decision": rank_attempt.decision.model_dump(mode="json"),
            "issues": [issue.to_dict() for issue in issues],
            "repair_attempts": rank_attempt.repair_attempts,
        }
        if issues or rank_attempt.decision.reference_scope != "full":
            raise _ExpectedRejection(
                "generated fallback did not rank as a full reference"
            )
        diagnostics.set_stage("publication")
        output_sha256 = _save_png(
            working / "generated_reference.png",
            build_candidate_reference(candidate_rgb, mask),
        )
        if _sha256(source_path) != source_sha256:
            raise RuntimeError("source reference changed during completion")
        decision = rank_attempt.decision
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "status": "accepted",
            "config_hash": config.fingerprint(),
            "clip_uid": clip_uid,
            "entity_id": entity.entity_id,
            "reference_type": entity.reference_type,
            "source_reference": source_state.model_dump(mode="json"),
            "source_image_path": relative("source_rgba.png"),
            "source_image_sha256": source_sha256,
            "input_image_path": relative("input_rgb.png"),
            "input_image_sha256": _sha256(input_path),
            "candidate_rgb_path": relative("candidate_rgb.png"),
            "candidate_rgb_sha256": candidate_sha256,
            "candidate_mask_path": relative("candidate_mask.png"),
            "candidate_mask_sha256": mask_sha256,
            "completion": {
                "backend": "qwen_image_edit_2511",
                "mode": "localized_raw",
                "model_path": str(config.remove.base_model_path),
                "seed": seed,
                "prompt": prompt,
                "prompt_language": prompt_language,
                "negative_prompt": DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
                "height_width_forced": True,
                "num_inference_steps": config.remove.num_inference_steps,
                "true_cfg_scale": config.remove.true_cfg_scale,
                "guidance_scale": config.remove.guidance_scale,
                **completion_size_metadata,
            },
            "segmentation": {
                "backend": "sam3",
                "track_status": track.status,
                "ranking": mask_ranking,
            },
            "production_gates": {
                "mask_passed": mask_ok,
                "improvement_passed": improvement_ok,
                "background_passed": background_ok,
                "mask_metrics": mask_metrics,
                "source_metrics": source_metrics,
                "candidate_metrics": candidate_metrics,
                "improvement_metrics": improvement,
                "background_metrics": background,
                "thresholds": gates.thresholds_for(entity.reference_type),
            },
            "generated_candidate_sha256": generated_candidate_sha256,
            "localized_completion_review": review.model_dump(mode="json"),
            "reference_ranking": {
                "decision": decision.model_dump(mode="json"),
                "raw_responses": list(rank_attempt.raw_responses),
                "repair_attempts": rank_attempt.repair_attempts,
            },
            "generated_reference_path": source_state.image_path,
            "generated_reference_sha256": output_sha256,
            **policy.diagnostics(),
            "diagnostic_artifacts": _policy_artifact_paths(),
            "candidate_new_region_stats": conservative.report,
            "fallback_used": False,
        }
        write_json_atomic(working / "metadata.json", metadata)
        return _AcceptedFallback(
            working_directory=working,
            state=EntityReferenceState(
                entity_id=entity.entity_id,
                status="ready",
                reference_scope="full",
                visible_region=decision.visible_region,
                whole_entity_recognizable=decision.whole_entity_recognizable,
                identity_features_visible=decision.identity_features_visible,
                viewpoint=decision.viewpoint,
                independent_reference_value=decision.independent_reference_value,
                requires_substantial_invention=(
                    decision.requires_substantial_invention
                ),
                primary_identity_region_visible=(
                    decision.primary_identity_region_visible
                ),
                major_structure_visible=decision.major_structure_visible,
                truncation_severity=decision.truncation_severity,
                discrete_foreground_instance=(
                    decision.discrete_foreground_instance
                ),
                mask_matches_target=decision.mask_matches_target,
                completion_needed_for_reference_use=(
                    decision.completion_needed_for_reference_use
                ),
                detached_target_fragments_present=(
                    decision.detached_target_fragments_present
                ),
                scope_reason=decision.scope_reason,
                image_path=source_state.image_path,
                source_frame_index=source_state.source_frame_index,
                source_clip_uid=clip_uid,
                source_entity_id=entity.entity_id,
                synthetic=True,
                generation_metadata_path=relative("metadata.json"),
                generation_source_sha256=source_sha256,
                generation_output_sha256=output_sha256,
            ),
        )
    except Exception:
        shutil.rmtree(working, ignore_errors=True)
        raise


def _rejection_payload(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
    error: Exception,
) -> dict[str, object]:
    source_sha256 = diagnostics.source_image_sha256
    if source_sha256 is None and source_state.image_path is not None:
        try:
            source_sha256 = _sha256(
                _artifact(storage, source_state.image_path)
            )
        except Exception:  # noqa: BLE001 - diagnostics are best effort
            source_sha256 = None
    payload: dict[str, object] = {
        "schema_version": _REJECTION_SCHEMA_VERSION,
        "status": "rejected",
        "clip_uid": clip_uid,
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "stage": diagnostics.stage,
        "reason": str(error).strip() or type(error).__name__,
        "exception_type": (
            None
            if isinstance(error, _ExpectedRejection)
            else type(error).__name__
        ),
        "source_reference": source_state.model_dump(mode="json"),
        "source_image_sha256": source_sha256,
        "config_hash": config.fingerprint(),
        "git_commit": storage.read_run().git_commit,
        "diagnostic_artifacts": _policy_artifact_paths(),
        "fallback_used": True,
    }
    payload.update(diagnostics.details)
    return payload


def _publish_diagnostic_result(
    *,
    final_directory: Path,
    entity_id: str,
    kind: str,
    filename: str,
    payload: dict[str, object],
    diagnostics: _RejectionDiagnostics,
) -> None:
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    working = final_directory.parent / (
        f".tmp-{kind}-{entity_id}-{uuid.uuid4().hex}"
    )
    backup = final_directory.parent / (
        f".backup-{kind}-{entity_id}-{uuid.uuid4().hex}"
    )
    working.mkdir(parents=False, exist_ok=False)
    backup_created = False
    published = False
    try:
        write_json_atomic(working / filename, payload)
        _write_diagnostic_artifacts(working, diagnostics)
        if final_directory.exists():
            final_directory.replace(backup)
            backup_created = True
        working.replace(final_directory)
        published = True
    except Exception:
        if published and final_directory.exists():
            shutil.rmtree(final_directory)
        if backup_created and backup.exists():
            backup.replace(final_directory)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if working.exists():
            shutil.rmtree(working)


def _publish_rejection(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
    error: Exception,
) -> None:
    _publish_diagnostic_result(
        final_directory=_completion_directory(
            storage,
            clip_uid,
            entity.entity_id,
        ),
        entity_id=entity.entity_id,
        kind="rejection",
        filename="rejection.json",
        payload=_rejection_payload(
            config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            source_state=source_state,
            diagnostics=diagnostics,
            error=error,
        ),
        diagnostics=diagnostics,
    )


def _skip_payload(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": "skipped",
        "clip_uid": clip_uid,
        "entity_id": entity.entity_id,
        "reference_type": entity.reference_type,
        "completion_skipped_reason": reason,
        "source_reference": source_state.model_dump(mode="json"),
        "source_image_sha256": diagnostics.source_image_sha256,
        "config_hash": config.fingerprint(),
        "git_commit": storage.read_run().git_commit,
        "diagnostic_artifacts": _policy_artifact_paths(),
        "fallback_used": True,
        **diagnostics.details,
    }


def _record_skip_safely(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
    reason: str,
) -> None:
    try:
        _publish_diagnostic_result(
            final_directory=_completion_directory(
                storage,
                clip_uid,
                entity.entity_id,
            ),
            entity_id=entity.entity_id,
            kind="skip",
            filename="metadata.json",
            payload=_skip_payload(
                config,
                storage,
                clip_uid=clip_uid,
                entity=entity,
                source_state=source_state,
                diagnostics=diagnostics,
                reason=reason,
            ),
            diagnostics=diagnostics,
        )
    except Exception as diagnostic_error:  # noqa: BLE001 - fallback stays open
        try:
            storage.append_failure(
                stage="pair",
                clip_uid=clip_uid,
                reason="completion skip diagnostic write failed",
                details={
                    "entity_id": entity.entity_id,
                    "completion_skipped_reason": reason,
                    "diagnostic_error_type": type(diagnostic_error).__name__,
                    "diagnostic_error": str(diagnostic_error),
                },
            )
        except Exception:  # noqa: BLE001,S110 - diagnostics stay fail-open
            pass


def _record_rejection_safely(
    config: V3Config,
    storage: RunStorage,
    *,
    clip_uid: str,
    entity: AnnotationEntity,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
    error: Exception,
) -> None:
    try:
        _publish_rejection(
            config,
            storage,
            clip_uid=clip_uid,
            entity=entity,
            source_state=source_state,
            diagnostics=diagnostics,
            error=error,
        )
    except Exception as diagnostic_error:  # noqa: BLE001 - fallback stays open
        try:
            storage.append_failure(
                stage="pair",
                clip_uid=clip_uid,
                reason="completion rejection diagnostic write failed",
                details={
                    "entity_id": entity.entity_id,
                    "completion_stage": diagnostics.stage,
                    "rejection_reason": str(error),
                    "diagnostic_error_type": type(diagnostic_error).__name__,
                    "diagnostic_error": str(diagnostic_error),
                },
            )
        except Exception:  # noqa: BLE001,S110 - diagnostics stay fail-open
            pass


def _publish(
    storage: RunStorage,
    *,
    clip_uid: str,
    states: list[EntityReferenceState],
    accepted: _AcceptedFallback,
) -> None:
    from r2v_data_v2.v3.pair import _pairing_from_references

    entity_id = accepted.state.entity_id
    working = accepted.working_directory
    final_directory = _completion_directory(storage, clip_uid, entity_id)
    directory_backup = final_directory.parent / f".backup-{entity_id}-{uuid.uuid4().hex}"
    final_reference = storage.selected_entity_path(clip_uid, entity_id)
    reference_backup = storage.pair_output_backup_path(clip_uid, entity_id)
    temporary_reference = storage.pair_output_temporary_path(clip_uid, entity_id)
    reference_moved = False
    working_moved = False
    try:
        shutil.copyfile(working / "generated_reference.png", temporary_reference)
        if _sha256(temporary_reference) != accepted.state.generation_output_sha256:
            raise RuntimeError("generated fallback changed before publication")
        if final_directory.exists():
            final_directory.replace(directory_backup)
        final_reference.replace(reference_backup)
        reference_moved = True
        working.replace(final_directory)
        working_moved = True
        temporary_reference.replace(final_reference)
        clip = storage.read_clip(clip_uid)
        storage.write_references_and_pairing(
            clip_uid,
            ReferencesState(entities=states, background=clip.references.background),
            _pairing_from_references(
                clip,
                states,
                bind_ready_background=(
                    storage.config.pair.background_final_guard_mode == "off"
                    or (
                        clip.pairing is not None
                        and clip.pairing.background_token is not None
                    )
                ),
            ),
        )
    except Exception:
        if reference_moved:
            final_reference.unlink(missing_ok=True)
            if reference_backup.exists():
                reference_backup.replace(final_reference)
        if working_moved and final_directory.exists():
            final_directory.replace(working)
        if directory_backup.exists():
            directory_backup.replace(final_directory)
        shutil.rmtree(working, ignore_errors=True)
        raise
    else:
        reference_backup.unlink(missing_ok=True)
        if directory_backup.exists():
            shutil.rmtree(directory_backup)
    finally:
        temporary_reference.unlink(missing_ok=True)
        storage.cleanup_pair_artifacts(clip_uid)


def _prepare_policy(
    storage: RunStorage,
    *,
    source_state: EntityReferenceState,
    diagnostics: _RejectionDiagnostics,
) -> ConservativeCompletionPolicy:
    if source_state.image_path is None:
        raise ValueError("local reference image path is missing")
    source_path = _artifact(storage, source_state.image_path)
    source_rgba, _, source_sha256 = _load_source_rgba(source_path)
    diagnostics.source_image_sha256 = source_sha256
    policy = build_conservative_completion_policy(
        source_rgba,
        whole_entity_recognizable=source_state.whole_entity_recognizable,
    )
    diagnostics.details.update(policy.diagnostics())
    diagnostics.details.update(
        {
            "diagnostic_artifacts": _policy_artifact_paths(),
            "fallback_used": True,
        }
    )
    diagnostics.artifacts.update(
        {
            "cleaned_source.png": policy.cleaned_source_rgba,
            "protected_component_mask.png": policy.protected_component_mask,
            "recovery_corridor_mask.png": policy.recovery_corridor_mask,
        }
    )
    return policy


def _owned_backend(config: V3Config) -> QwenImageEdit2511ReferenceCompletionBackend:
    return QwenImageEdit2511ReferenceCompletionBackend(
        QwenImageEdit2511CompletionConfig(
            model_path=config.remove.base_model_path,
            device=config.remove.device,
            dtype=config.remove.dtype,
            num_inference_steps=config.remove.num_inference_steps,
            true_cfg_scale=config.remove.true_cfg_scale,
            guidance_scale=config.remove.guidance_scale,
            mode="localized_raw",
            force_input_size=True,
        )
    )


def run_reference_completion_fallbacks(
    config: V3Config,
    storage: RunStorage,
    *,
    target_clip_uids: set[str],
    reference_judge: EntityReferenceJudge | None,
    completion_backend: QwenReferenceCompletionBackend | None = None,
    completion_judge: QwenLocalizedCompletionJudge | None = None,
    segmentation_backend: SegmentationBackend | None = None,
) -> ReferenceCompletionFallbackStats:
    """Run after real self/donor selection; every failure keeps the local source."""

    if not config.reference_scope.allow_synthetic_completion:
        return ReferenceCompletionFallbackStats()
    gates = PublicationConfig()
    gates.validate()
    counts = {"attempted": 0, "ready": 0, "rejected": 0}
    backend = completion_backend
    localized_judge = completion_judge
    segmenter = segmentation_backend
    owned_backend: QwenImageEdit2511ReferenceCompletionBackend | None = None
    owned_judge: QwenLocalizedReferenceCompletionJudge | None = None
    owned_segmenter: Sam3SegmentationBackend | None = None
    try:
        for clip_uid in sorted(target_clip_uids):
            clip = storage.read_clip(clip_uid)
            if clip.annotation is None or clip.annotation.status != "ready":
                continue
            states = {state.entity_id: state for state in clip.references.entities}
            for entity in clip.annotation.entities:
                source_state = states.get(entity.entity_id)
                if not _eligible(clip_uid, entity, source_state):
                    continue
                assert source_state is not None
                diagnostics = _RejectionDiagnostics()
                try:
                    policy = _prepare_policy(
                        storage,
                        source_state=source_state,
                        diagnostics=diagnostics,
                    )
                except Exception as exc:  # noqa: BLE001 - fallback fails open
                    counts["attempted"] += 1
                    counts["rejected"] += 1
                    _record_rejection_safely(
                        config,
                        storage,
                        clip_uid=clip_uid,
                        entity=entity,
                        source_state=source_state,
                        diagnostics=diagnostics,
                        error=exc,
                    )
                    continue
                if not policy.eligible:
                    reason = policy.completion_skipped_reason
                    if reason is None:
                        raise RuntimeError("ineligible completion policy has no reason")
                    _record_skip_safely(
                        config,
                        storage,
                        clip_uid=clip_uid,
                        entity=entity,
                        source_state=source_state,
                        diagnostics=diagnostics,
                        reason=reason,
                    )
                    continue
                counts["attempted"] += 1
                try:
                    if backend is None:
                        owned_backend = _owned_backend(config)
                        backend = owned_backend
                    if localized_judge is None:
                        if config.qwen.candidate_judge is None:
                            raise RuntimeError("candidate judge is not configured")
                        owned_judge = QwenLocalizedReferenceCompletionJudge(
                            config.qwen.candidate_judge,
                            repair_retries=config.pair.repair_retries,
                        )
                        localized_judge = owned_judge
                    if segmenter is None:
                        owned_segmenter = Sam3SegmentationBackend(config.sam3)
                        segmenter = owned_segmenter
                    if reference_judge is None:
                        raise RuntimeError("reference judge is unavailable")
                    accepted = _attempt(
                        config,
                        storage,
                        clip_uid=clip_uid,
                        entity=entity,
                        source_state=source_state,
                        completion_backend=backend,
                        completion_judge=localized_judge,
                        segmenter=segmenter,
                        reference_judge=reference_judge,
                        gates=gates,
                        diagnostics=diagnostics,
                        policy=policy,
                    )
                    states[entity.entity_id] = accepted.state
                    ordered = [states[item.entity_id] for item in clip.annotation.entities]
                    diagnostics.set_stage("publication")
                    _publish(
                        storage,
                        clip_uid=clip_uid,
                        states=ordered,
                        accepted=accepted,
                    )
                except Exception as exc:  # noqa: BLE001 - fallback fails open
                    states[entity.entity_id] = source_state
                    counts["rejected"] += 1
                    _record_rejection_safely(
                        config,
                        storage,
                        clip_uid=clip_uid,
                        entity=entity,
                        source_state=source_state,
                        diagnostics=diagnostics,
                        error=exc,
                    )
                else:
                    counts["ready"] += 1
    finally:
        if owned_segmenter is not None:
            owned_segmenter.close()
        if owned_judge is not None:
            owned_judge.close()
        if owned_backend is not None:
            owned_backend.close()
    return ReferenceCompletionFallbackStats(**counts)
