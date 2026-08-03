"""Thin optional localized-completion fallback for V3 pairing."""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.reference_completion_benchmark import (
    ReferenceCompletionReview,
    _white_source,
)
from r2v_data_v2.v3.reference_completion_publish import (
    PublicationConfig,
    _load_source_rgba,
    _mask_from_track_result,
    _save_png,
    build_candidate_reference,
    compute_foreground_metrics,
    evaluate_background_gate,
    evaluate_improvement_gate,
    evaluate_mask_gate,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
    DEFAULT_QWEN_SEEDS,
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


@dataclass(frozen=True)
class ReferenceCompletionFallbackStats:
    attempted: int = 0
    ready: int = 0
    rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


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
) -> _AcceptedFallback:
    if source_state.image_path is None or source_state.source_frame_index is None:
        raise ValueError("local reference provenance is incomplete")
    source_path = _artifact(storage, source_state.image_path)
    source_rgba, source_bytes, source_sha256 = _load_source_rgba(source_path)
    final_directory = _completion_directory(storage, clip_uid, entity.entity_id)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    working = final_directory.parent / f".tmp-{entity.entity_id}-{uuid.uuid4().hex}"
    working.mkdir(parents=False, exist_ok=False)
    relative = lambda name: storage.relative_artifact_path(
        final_directory / name
    )
    try:
        (working / "source_rgba.png").write_bytes(source_bytes)
        input_rgb = _white_source(source_rgba)
        input_path = working / "input_rgb.png"
        _save_png(input_path, input_rgb)
        prompt, prompt_language = _localized_prompt(entity.reference_type, None)
        seed = DEFAULT_QWEN_SEEDS[0]
        generated = completion_backend.complete(
            input_rgb=input_rgb.copy(),
            entity_phrase=entity.phrase,
            seed=seed,
            prompt=prompt,
            negative_prompt=DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
        )
        candidate_path = working / "candidate_rgb.png"
        candidate_rgb, candidate_sha256, hard_check = _localized_hard_check(
            generated=generated,
            input_rgb=input_rgb,
            input_path=input_path,
            candidate_path=candidate_path,
            source_path=source_path,
            source_sha256=source_sha256,
        )
        if candidate_rgb is None or not hard_check.passed:
            raise ValueError(
                "localized completion hard check failed: "
                + ",".join(hard_check.reasons)
            )
        track = segmenter.track(
            frame_paths=[candidate_path],
            entity_id=entity.entity_id,
            reference_type=entity.reference_type,
            grounding_prompt=entity.phrase,
        )
        mask, mask_ranking = _mask_from_track_result(
            track,
            expected_size=candidate_rgb.size,
        )
        mask_sha256 = _save_png(working / "candidate_mask.png", mask)
        source_metrics = compute_foreground_metrics(source_rgba.getchannel("A"))
        candidate_metrics = compute_foreground_metrics(mask)
        mask_metrics, mask_ok, mask_reasons = evaluate_mask_gate(
            mask,
            reference_type=entity.reference_type,
            config=gates,
        )
        improvement, improvement_ok, improvement_reasons = evaluate_improvement_gate(
            source_metrics,
            candidate_metrics,
            reference_type=entity.reference_type,
            config=gates,
        )
        background, background_ok, background_reasons = evaluate_background_gate(
            candidate_rgb,
            mask,
            config=gates,
        )
        reasons = [*mask_reasons, *improvement_reasons, *background_reasons]
        if not all((mask_ok, improvement_ok, background_ok)):
            raise ValueError(f"completion production gate failed: {reasons[0]}")
        review = completion_judge.review(
            source_rgba=source_rgba.copy(),
            candidate_rgb=candidate_rgb.copy(),
            entity_phrase=entity.phrase,
            reference_type=entity.reference_type,
        )
        if not isinstance(review, ReferenceCompletionReview):
            raise TypeError("localized completion judge returned an invalid result")
        if review.verdict != "accept":
            raise ValueError("localized completion judge rejected the candidate")
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
        )
        if issues or rank_attempt.decision.reference_scope != "full":
            raise ValueError("generated fallback did not rank as a full reference")
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
                "num_inference_steps": config.remove.num_inference_steps,
                "true_cfg_scale": config.remove.true_cfg_scale,
                "guidance_scale": config.remove.guidance_scale,
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
            "localized_completion_review": review.model_dump(mode="json"),
            "reference_ranking": {
                "decision": decision.model_dump(mode="json"),
                "raw_responses": list(rank_attempt.raw_responses),
                "repair_attempts": rank_attempt.repair_attempts,
            },
            "generated_reference_path": source_state.image_path,
            "generated_reference_sha256": output_sha256,
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
            _pairing_from_references(clip, states),
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
                    )
                    states[entity.entity_id] = accepted.state
                    ordered = [states[item.entity_id] for item in clip.annotation.entities]
                    _publish(
                        storage,
                        clip_uid=clip_uid,
                        states=ordered,
                        accepted=accepted,
                    )
                except Exception:  # noqa: BLE001 - optional fallback fails open
                    states[entity.entity_id] = source_state
                    counts["rejected"] += 1
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
