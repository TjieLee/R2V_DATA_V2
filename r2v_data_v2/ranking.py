from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw

from prompts.qwen_candidate_judge_prompt import CANDIDATE_JUDGE_PROMPT
from r2v_data_v2.config import PipelineConfig, QwenConfig, RankingConfig
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.mask_utils import bbox_from_mask, decode_mask, save_mask_png
from r2v_data_v2.metrics import (
    CandidateMetrics,
    calculate_candidate_metrics,
    padded_crop_box,
)
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic
from r2v_data_v2.reference_image import build_neutral_subject_crop
from r2v_data_v2.schemas import (
    AnnotationEntity,
    AnnotationResult,
    CandidateJudgeResult,
    CandidateVisualReview,
)
from r2v_data_v2.semantic_alignment import (
    AlignmentMetrics,
    Siglip2Aligner,
    is_siglip_wrong_entity,
)
from r2v_data_v2.structured_output import (
    StructuredOutputFailure,
    ValidationIssue,
    request_structured_output,
)
from r2v_data_v2.visual_embedding import (
    DinoV3Embedder,
    TemporalRepresentationMetrics,
    save_dinov3_embeddings,
    save_selected_dinov3_embedding,
    temporal_representation_metrics,
)


@dataclass(frozen=True)
class RankedCandidate:
    frame_slot: int
    source_frame_index: int
    sam_confidence: float
    metrics: CandidateMetrics
    visual_review: CandidateVisualReview
    dino_metrics: TemporalRepresentationMetrics | None
    alignment_metrics: AlignmentMetrics | None
    best_matching_entity_id: str | None
    hard_rejection_reasons: tuple[str, ...]
    ranking_score: float


@dataclass(frozen=True)
class RankingStats:
    processed: int = 0
    skipped_existing: int = 0
    no_valid_candidate: int = 0
    failed: int = 0
    dino_load_seconds: float = 0.0
    dino_inference_seconds: float = 0.0
    siglip_load_seconds: float = 0.0
    siglip_inference_seconds: float = 0.0
    candidate_count_before_models: int = 0
    candidate_count_after_dino: int = 0
    candidate_count_after_siglip: int = 0


@dataclass
class CandidateWorkItem:
    record: dict[str, object]
    metrics: CandidateMetrics
    neutral_crop: Image.Image
    hard_rejection_reasons: list[str]
    dino_embedding: np.ndarray | None = None
    dino_metrics: TemporalRepresentationMetrics | None = None
    alignment_metrics: AlignmentMetrics | None = None
    best_matching_entity_id: str | None = None

    @property
    def frame_slot(self) -> int:
        return int(self.record["frame_slot"])

    @property
    def sam_confidence(self) -> float:
        return float(self.record["sam_confidence"])


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _isolation_score(metrics: CandidateMetrics) -> float:
    return _clamp_unit(
        1.0
        - max(
            metrics.maximum_other_mask_overlap,
            0.5 * metrics.maximum_bbox_containment,
        )
    )


def _crop_subject_ratio_score(metrics: CandidateMetrics) -> float:
    return _clamp_unit(1.0 - abs(metrics.crop_subject_ratio - 0.55) / 0.55)


def basic_hard_rejection_reasons(
    *,
    metrics: CandidateMetrics,
    sam_confidence: float,
    config: RankingConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if metrics.effective_short_side < config.minimum_effective_short_side:
        reasons.append("effective_short_side")
    if not (
        config.minimum_mask_area_ratio
        <= metrics.mask_area_ratio
        <= config.maximum_mask_area_ratio
    ):
        reasons.append("mask_area_ratio")
    if config.reject_border_touch and metrics.border_touch:
        reasons.append("border_touch")
    if sam_confidence <= 0:
        reasons.append("sam_confidence")
    if metrics.exposure_score < config.minimum_exposure_score:
        reasons.append("extreme_exposure")
    if metrics.crop_subject_ratio < config.minimum_crop_subject_ratio:
        reasons.append("subject_too_small_in_crop")
    if metrics.crop_subject_ratio > config.maximum_crop_subject_ratio:
        reasons.append("crop_too_tight")
    if metrics.maximum_other_mask_overlap > config.maximum_other_mask_overlap:
        reasons.append("other_subject_contamination")
    return tuple(reasons)


def hard_rejection_reasons(
    *,
    metrics: CandidateMetrics,
    sam_confidence: float,
    visual_review: CandidateVisualReview,
    config: RankingConfig,
    dino_temporal_outlier: bool = False,
    siglip_wrong_entity: bool = False,
) -> tuple[str, ...]:
    reasons = list(
        basic_hard_rejection_reasons(
            metrics=metrics,
            sam_confidence=sam_confidence,
            config=config,
        )
    )
    if dino_temporal_outlier and config.dinov3_exclude_cluster_outliers:
        reasons.append("dino_temporal_outlier")
    if siglip_wrong_entity and config.siglip2_hard_reject_wrong_entity:
        reasons.append("siglip_wrong_entity")
    if visual_review.completeness < 0.5:
        reasons.append("incomplete")
    if visual_review.occlusion > 0.5:
        reasons.append("occluded")
    if visual_review.mask_quality < 0.5:
        reasons.append("mask_quality")
    if not visual_review.identity_features_visible:
        reasons.append("identity_features_not_visible")
    reasons.extend(visual_review.rejection_reasons)
    return tuple(dict.fromkeys(reasons))


def rank_candidates(
    candidates: list[tuple[int, int, float, CandidateMetrics, CandidateVisualReview]],
    *,
    config: RankingConfig,
    dino_metrics_by_slot: dict[int, TemporalRepresentationMetrics] | None = None,
    dino_outlier_slots: set[int] | None = None,
    alignment_metrics_by_slot: dict[int, AlignmentMetrics] | None = None,
    normalized_siglip_scores_by_slot: dict[int, float] | None = None,
    best_matching_entity_ids: dict[int, str] | None = None,
    siglip_wrong_entity_slots: set[int] | None = None,
) -> list[RankedCandidate]:
    if not candidates:
        return []
    dino_metrics_by_slot = dino_metrics_by_slot or {}
    dino_outlier_slots = dino_outlier_slots or set()
    alignment_metrics_by_slot = alignment_metrics_by_slot or {}
    normalized_siglip_scores_by_slot = normalized_siglip_scores_by_slot or {}
    best_matching_entity_ids = best_matching_entity_ids or {}
    siglip_wrong_entity_slots = siglip_wrong_entity_slots or set()
    normalized_sharpness_by_slot = dict(
        zip(
            [item[0] for item in candidates],
            _min_max([item[3].tenengrad_sharpness for item in candidates]),
        )
    )
    siglip_raw_scores = []
    for frame_slot, _, _, _, _ in candidates:
        alignment = alignment_metrics_by_slot.get(frame_slot)
        if alignment is None:
            siglip_raw_scores.append(0.5)
        elif alignment.alignment_margin is not None:
            siglip_raw_scores.append(alignment.alignment_margin)
        else:
            siglip_raw_scores.append(alignment.target_similarity)
    normalized_siglip_by_slot = {
        **dict(
            zip(
                [item[0] for item in candidates],
                _min_max(siglip_raw_scores),
            )
        ),
        **normalized_siglip_scores_by_slot,
    }
    ranked: list[RankedCandidate] = []
    for frame_slot, source_index, sam_confidence, metrics, visual_review in candidates:
        dino_metrics = dino_metrics_by_slot.get(frame_slot)
        alignment_metrics = alignment_metrics_by_slot.get(frame_slot)
        reasons = hard_rejection_reasons(
            metrics=metrics,
            sam_confidence=sam_confidence,
            visual_review=visual_review,
            config=config,
            dino_temporal_outlier=frame_slot in dino_outlier_slots,
            siglip_wrong_entity=frame_slot in siglip_wrong_entity_slots,
        )
        component_scores = {
            "dino": (
                _clamp_unit(dino_metrics.dino_representativeness)
                if dino_metrics is not None
                else 0.5
            ),
            "completeness": visual_review.completeness,
            "recognizability": visual_review.recognizability,
            "siglip": normalized_siglip_by_slot[frame_slot],
            "mask_quality": visual_review.mask_quality,
            "mask_area_continuity": metrics.mask_area_continuity,
            "quality": (
                0.6 * normalized_sharpness_by_slot[frame_slot]
                + 0.4 * metrics.exposure_score
            ),
            "visual_quality": visual_review.visual_quality,
            "occlusion": 1.0 - visual_review.occlusion,
            "isolation": _isolation_score(metrics),
            "crop": _crop_subject_ratio_score(metrics),
            "sam": sam_confidence,
        }
        weights = {
            "dino": 0.23,
            "completeness": 0.17,
            "recognizability": 0.14,
            "siglip": 0.10,
            "mask_quality": 0.08,
            "mask_area_continuity": 0.05,
            "quality": 0.06,
            "visual_quality": 0.04,
            "occlusion": 0.04,
            "isolation": 0.04,
            "crop": 0.03,
            "sam": 0.02,
        }
        if not config.dinov3_enabled:
            weights.pop("dino")
        if not config.siglip2_enabled:
            weights.pop("siglip")
        total_weight = sum(weights.values())
        score = (
            sum(weights[name] * _clamp_unit(component_scores[name]) for name in weights)
            / total_weight
        )
        ranked.append(
            RankedCandidate(
                frame_slot=frame_slot,
                source_frame_index=source_index,
                sam_confidence=sam_confidence,
                metrics=metrics,
                visual_review=visual_review,
                dino_metrics=dino_metrics,
                alignment_metrics=alignment_metrics,
                best_matching_entity_id=best_matching_entity_ids.get(frame_slot),
                hard_rejection_reasons=reasons,
                ranking_score=float(score),
            )
        )
    return sorted(
        ranked,
        key=lambda item: (
            bool(item.hard_rejection_reasons),
            -item.ranking_score,
            item.frame_slot,
        ),
    )


class QwenCandidateJudge:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(self, *, prompt: str, encoded_image: str) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "max_tokens": min(1024, self.config.max_tokens),
        }
        try:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "candidate_judge",
                        "strict": True,
                        "schema": CandidateJudgeResult.model_json_schema(),
                    },
                },
            )
        except BadRequestError:
            response = self.client.chat.completions.create(
                **parameters,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Qwen returned an empty candidate review")
        return content

    def review(
        self,
        *,
        entity_id: str,
        entity_phrase: str,
        contact_sheet: Path,
        frame_slots: list[int],
    ) -> CandidateJudgeResult:
        encoded = base64.b64encode(contact_sheet.read_bytes()).decode()
        prompt = (
            CANDIDATE_JUDGE_PROMPT.format(entity_phrase=entity_phrase)
            + f"\nentity_id={entity_id}\nframe_slots={frame_slots}"
        )

        def validate(result: CandidateJudgeResult) -> list[ValidationIssue]:
            issues: list[ValidationIssue] = []
            returned_slot_list = [
                candidate.frame_slot for candidate in result.candidates
            ]
            returned_slots = set(returned_slot_list)
            if (
                result.entity_id != entity_id
                or returned_slots != set(frame_slots)
                or len(returned_slot_list) != len(frame_slots)
            ):
                issues.append(
                    ValidationIssue(
                        "candidate_slots_mismatch",
                        "candidates",
                        "candidate review must match the requested entity and slots",
                    )
                )
            if result.best_frame_slot not in returned_slots:
                issues.append(
                    ValidationIssue(
                        "invalid_best_frame_slot",
                        "best_frame_slot",
                        "best_frame_slot must be one of the reviewed slots",
                    )
                )
            return issues

        return request_structured_output(
            request=lambda request_text: self._request(
                prompt=request_text,
                encoded_image=encoded,
            ),
            original_request=prompt,
            model=CandidateJudgeResult,
            validate=validate,
        )


def _fit_candidate_panel(
    image: Image.Image,
    *,
    width: int,
    height: int,
) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), color=(24, 24, 24))
    panel.paste(
        fitted,
        ((width - fitted.width) // 2, (height - fitted.height) // 2),
    )
    return panel


def _candidate_sheet_label(
    record: dict[str, object],
    metrics: CandidateMetrics,
) -> str:
    return (
        f"frame_slot={int(record['frame_slot'])}  "
        f"SAM={float(record['sam_confidence']):.3f}  "
        f"effective_short_side={metrics.effective_short_side}  "
        f"mask_area_ratio={metrics.mask_area_ratio:.4f}  "
        f"Tenengrad={metrics.tenengrad_sharpness:.1f}  "
        f"border_touch={str(metrics.border_touch).lower()}"
    )


def _top_candidate_sheet(
    *,
    frame_paths: dict[int, Path],
    masks: dict[int, np.ndarray],
    candidates: list[tuple[dict[str, object], CandidateMetrics]],
    destination: Path,
) -> None:
    panel_width = 480
    panel_height = 320
    label_height = 42
    rows: list[Image.Image] = []
    for record, metrics in candidates:
        slot = int(record["frame_slot"])
        frame_path = frame_paths[slot]
        image = Image.open(frame_path).convert("RGB")
        overlay = np.asarray(image).copy()
        overlay[masks[slot]] = (
            0.5 * overlay[masks[slot]] + 0.5 * np.array([255, 60, 60])
        ).astype(np.uint8)
        mask = masks[slot]
        y_indices, x_indices = np.nonzero(mask)
        bbox = (
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()) + 1,
            int(y_indices.max()) + 1,
        )
        crop_box = padded_crop_box(
            bbox,
            image_width=image.width,
            image_height=image.height,
        )
        panels = [
            _fit_candidate_panel(image, width=panel_width, height=panel_height),
            _fit_candidate_panel(
                Image.fromarray(overlay),
                width=panel_width,
                height=panel_height,
            ),
            _fit_candidate_panel(
                image.crop(crop_box),
                width=panel_width,
                height=panel_height,
            ),
        ]
        row = Image.new(
            "RGB",
            (panel_width * 3, label_height + panel_height),
            color=(16, 16, 16),
        )
        draw = ImageDraw.Draw(row)
        draw.text(
            (10, 8),
            _candidate_sheet_label(record, metrics),
            fill=(255, 255, 0),
        )
        for panel_index, (panel, panel_name) in enumerate(
            zip(
                panels,
                ("ORIGINAL FRAME", "MASK OVERLAY", "PADDED SUBJECT CROP"),
            )
        ):
            ImageDraw.Draw(panel).text((8, 8), panel_name, fill=(255, 255, 0))
            row.paste(panel, (panel_index * panel_width, label_height))
        rows.append(row)
    sheet = Image.new(
        "RGB",
        (panel_width * 3, (label_height + panel_height) * len(rows)),
        color=(16, 16, 16),
    )
    for row_index, row in enumerate(rows):
        sheet.paste(row, (0, row_index * (label_height + panel_height)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def _load_entity_candidates(
    candidate_dir: Path,
) -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    records = list(iter_source_records(candidate_dir / "candidates.jsonl"))
    encoded = json.loads(
        (candidate_dir / "top_masks.rle.json").read_text(encoding="utf-8")
    )
    masks: dict[int, np.ndarray] = {}
    for record in records:
        key = record.get("mask_rle_key")
        if isinstance(key, str) and key in encoded:
            masks[int(record["frame_slot"])] = decode_mask(encoded[key])
    return records, masks


def _load_other_entity_masks(
    *,
    output_root: Path,
    clip_uid: str,
    current_entity_id: str,
    annotation: AnnotationResult,
) -> dict[int, list[np.ndarray]]:
    by_slot: dict[int, list[np.ndarray]] = {}
    for entity in annotation.entities:
        if entity.entity_id == current_entity_id or not entity.reference_worthy:
            continue
        candidate_dir = output_root / "candidates" / clip_uid / entity.entity_id
        if not (candidate_dir / "top_masks.rle.json").is_file():
            continue
        _, masks = _load_entity_candidates(candidate_dir)
        for slot, mask in masks.items():
            by_slot.setdefault(slot, []).append(mask)
    return by_slot


def _default_review(record: dict[str, object]) -> CandidateVisualReview:
    return CandidateVisualReview(
        frame_slot=int(record["frame_slot"]),
        completeness=0.5,
        recognizability=0.5,
        occlusion=0.5,
        mask_quality=0.5,
        visual_quality=0.5,
        identity_features_visible=True,
        rejection_reasons=[],
    )


def _preliminary_top_candidates(
    candidates: list[CandidateWorkItem],
    *,
    config: RankingConfig,
) -> list[CandidateWorkItem]:
    if not candidates:
        return []
    normalized_sharpness = _min_max(
        [candidate.metrics.tenengrad_sharpness for candidate in candidates]
    )
    scored: list[tuple[float, CandidateWorkItem]] = []
    for index, candidate in enumerate(candidates):
        components = {
            "dino": (
                _clamp_unit(candidate.dino_metrics.dino_representativeness)
                if candidate.dino_metrics is not None
                else 0.5
            ),
            "sharpness": normalized_sharpness[index],
            "exposure": candidate.metrics.exposure_score,
            "isolation": _isolation_score(candidate.metrics),
            "sam": candidate.sam_confidence,
        }
        weights = {
            "dino": 0.45,
            "sharpness": 0.20,
            "exposure": 0.15,
            "isolation": 0.10,
            "sam": 0.10,
        }
        if not config.dinov3_enabled:
            weights.pop("dino")
        total_weight = sum(weights.values())
        score = (
            sum(weights[name] * _clamp_unit(components[name]) for name in weights)
            / total_weight
        )
        scored.append((score, candidate))
    return [
        candidate
        for _, candidate in sorted(
            scored,
            key=lambda item: (-item[0], item[1].frame_slot),
        )[: config.top_k_for_vlm_judge]
    ]


def _normalized_siglip_scores(
    candidates: list[CandidateWorkItem],
) -> dict[int, float]:
    aligned = [
        candidate for candidate in candidates if candidate.alignment_metrics is not None
    ]
    raw_scores = [
        (
            candidate.alignment_metrics.alignment_margin
            if candidate.alignment_metrics.alignment_margin is not None
            else candidate.alignment_metrics.target_similarity
        )
        for candidate in aligned
    ]
    return dict(
        zip(
            [candidate.frame_slot for candidate in aligned],
            _min_max([float(value) for value in raw_scores]),
        )
    )


def _normalized_entity_text(value: str) -> str:
    return " ".join(value.casefold().split())


def siglip_distractor_entities(
    annotation: AnnotationResult,
    target: AnnotationEntity,
) -> list[AnnotationEntity]:
    target_label = _normalized_entity_text(target.canonical_label)
    target_prompt = _normalized_entity_text(target.grounding_prompt)
    return [
        entity
        for entity in annotation.entities
        if entity.entity_id != target.entity_id
        and entity.separability != "attached_accessory"
        and _normalized_entity_text(entity.canonical_label) != target_label
        and _normalized_entity_text(entity.grounding_prompt) != target_prompt
        and not (not entity.reference_worthy and entity.salience == "incidental")
    ]


def _candidate_ranking_metadata(
    items: list[CandidateWorkItem],
    *,
    qwen_reviews: dict[int, CandidateVisualReview] | None = None,
    dino_medoid_slot: int | None = None,
    qwen_suggested_best_frame_slot: int | None = None,
) -> dict[str, object]:
    qwen_reviews = qwen_reviews or {}
    candidates = []
    for item in items:
        alignment = item.alignment_metrics
        candidates.append(
            {
                "frame_slot": item.frame_slot,
                "source_frame_index": int(item.record["source_frame_index"]),
                "sam_confidence": item.sam_confidence,
                "metrics": item.metrics.to_dict(),
                "hard_rejection_reasons": item.hard_rejection_reasons,
                "dino": (
                    asdict(item.dino_metrics) if item.dino_metrics is not None else None
                ),
                "siglip": (
                    {
                        "target_similarity": alignment.target_similarity,
                        "alignment_margin": alignment.alignment_margin,
                        "best_matching_entity_id": item.best_matching_entity_id,
                    }
                    if alignment is not None
                    else None
                ),
                "visual_review": (
                    qwen_reviews[item.frame_slot].model_dump(mode="json")
                    if item.frame_slot in qwen_reviews
                    else None
                ),
            }
        )
    return {
        "dino_medoid_slot": dino_medoid_slot,
        "qwen_suggested_best_frame_slot": qwen_suggested_best_frame_slot,
        "candidates": candidates,
    }


def _save_reference(
    *,
    output_root: Path,
    clip_uid: str,
    entity_id: str,
    frame_path: Path,
    mask: np.ndarray,
    selected: RankedCandidate,
    dino_embedding: np.ndarray | None,
    dino_medoid_slot: int | None,
    qwen_suggested_best_frame_slot: int,
) -> dict[str, object]:
    frame_bgr = cv2.imread(str(frame_path))
    if frame_bgr is None:
        raise RuntimeError(f"selected frame is unreadable: {frame_path}")
    bbox = tuple(
        int(value)
        for value in (
            np.nonzero(mask)[1].min(),
            np.nonzero(mask)[0].min(),
            np.nonzero(mask)[1].max() + 1,
            np.nonzero(mask)[0].max() + 1,
        )
    )
    crop_box = padded_crop_box(
        bbox,
        image_width=frame_bgr.shape[1],
        image_height=frame_bgr.shape[0],
    )
    x1, y1, x2, y2 = crop_box
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_mask = mask[y1:y2, x1:x2]
    destination = output_root / "references" / clip_uid / entity_id
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop_rgb).save(destination / "canonical.jpg", quality=95)
    save_mask_png(destination / "mask.png", crop_mask)
    rgba = np.dstack((crop_rgb, crop_mask.astype(np.uint8) * 255))
    Image.fromarray(rgba).save(destination / "foreground_rgba.png")
    neutral = np.full_like(crop_rgb, 204)
    neutral[crop_mask] = crop_rgb[crop_mask]
    Image.fromarray(neutral).save(destination / "neutral_background.jpg", quality=95)
    dino_embedding_path = None
    if dino_embedding is not None:
        dino_embedding_path = destination / "dinov3_embedding.npy"
        save_selected_dinov3_embedding(dino_embedding_path, dino_embedding)
    metadata = {
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "frame_slot": selected.frame_slot,
        "source_frame_index": selected.source_frame_index,
        "ranking_score": selected.ranking_score,
        "sam_confidence": selected.sam_confidence,
        "metrics": selected.metrics.to_dict(),
        "visual_review": selected.visual_review.model_dump(mode="json"),
        "dino": (
            asdict(selected.dino_metrics) if selected.dino_metrics is not None else None
        ),
        "dino_medoid_slot": dino_medoid_slot,
        "qwen_suggested_best_frame_slot": qwen_suggested_best_frame_slot,
        "siglip": (
            {
                "target_similarity": selected.alignment_metrics.target_similarity,
                "alignment_margin": selected.alignment_metrics.alignment_margin,
                "best_matching_entity_id": selected.best_matching_entity_id,
            }
            if selected.alignment_metrics is not None
            else None
        ),
        "crop_xyxy": list(crop_box),
        "canonical_path": str(destination / "canonical.jpg"),
        "mask_path": str(destination / "mask.png"),
        "foreground_rgba_path": str(destination / "foreground_rgba.png"),
        "neutral_background_path": str(destination / "neutral_background.jpg"),
        "dinov3_embedding_path": (
            str(dino_embedding_path) if dino_embedding_path is not None else None
        ),
    }
    return metadata


def _reference_record(
    metadata: dict[str, object],
    entity: AnnotationEntity,
) -> dict[str, object]:
    return {
        **metadata,
        "phrase": entity.phrase,
        "canonical_label": entity.canonical_label,
        "category": entity.category,
        "ref_token": entity.ref_token,
        "genericity": entity.genericity,
        "name_evidence": entity.name_evidence,
        "separability": entity.separability,
    }


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def rank_manifest_references(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    judge: QwenCandidateJudge | None = None,
    dino_embedder: DinoV3Embedder | None = None,
    siglip_aligner: Siglip2Aligner | None = None,
) -> RankingStats:
    output_root = config.ensure_output_root()
    annotation_manifest = output_root / "manifests" / "annotations.jsonl"
    reference_manifest = output_root / "manifests" / "references.jsonl"
    if not annotation_manifest.is_file():
        raise FileNotFoundError("run Stage 02 and Stage 03 before ranking")
    if overwrite:
        reference_manifest.unlink(missing_ok=True)
        for artifact in (output_root / "references").glob("*/*/metadata.json"):
            artifact.unlink()
    qwen = judge or QwenCandidateJudge(config.qwen)
    dino = dino_embedder
    siglip = siglip_aligner
    owns_dino = False
    owns_siglip = False
    dino_load_seconds = 0.0
    siglip_load_seconds = 0.0
    try:
        if config.ranking.dinov3_enabled and dino is None:
            started = time.perf_counter()
            dino = DinoV3Embedder(config.ranking)
            dino_load_seconds = time.perf_counter() - started
            owns_dino = True
        if config.ranking.siglip2_enabled and siglip is None:
            started = time.perf_counter()
            siglip = Siglip2Aligner(
                config.ranking.siglip2_model_path,
                config.ranking.siglip2_batch_size,
            )
            siglip_load_seconds = time.perf_counter() - started
            owns_siglip = True
    except Exception:
        if owns_dino and dino is not None:
            dino.close()
        raise

    processed = skipped = no_valid = failed = 0
    dino_inference_seconds = 0.0
    siglip_inference_seconds = 0.0
    before_models = after_dino = after_siglip = 0
    try:
        for payload in iter_source_records(annotation_manifest):
            clip = str(payload["clip_uid"])
            annotation = AnnotationResult.model_validate(
                {
                    key: payload.get(key)
                    for key in (
                        "caption",
                        "prompt_with_refs",
                        "entities",
                        "relations",
                        "background",
                    )
                }
            )
            for entity in annotation.entities:
                if not entity.reference_worthy:
                    continue
                reference_dir = output_root / "references" / clip / entity.entity_id
                if (reference_dir / "metadata.json").is_file() and not overwrite:
                    try:
                        existing_metadata = json.loads(
                            (reference_dir / "metadata.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        if not isinstance(existing_metadata, dict):
                            raise TypeError("reference metadata must be a JSON object")
                        write_json_atomic(
                            reference_dir / "metadata.json",
                            _reference_record(existing_metadata, entity),
                        )
                        skipped += 1
                    except Exception as exc:  # noqa: BLE001
                        _append_jsonl(
                            output_root / "logs" / "ranking_failed.jsonl",
                            {
                                "clip_uid": clip,
                                "entity_id": entity.entity_id,
                                "error": str(exc),
                            },
                        )
                        failed += 1
                    continue
                candidate_dir = output_root / "candidates" / clip / entity.entity_id
                try:
                    records, masks = _load_entity_candidates(candidate_dir)
                    if not masks:
                        no_valid += 1
                        continue
                    other_masks = _load_other_entity_masks(
                        output_root=output_root,
                        clip_uid=clip,
                        current_entity_id=entity.entity_id,
                        annotation=annotation,
                    )
                    areas = [float(mask.mean()) for mask in masks.values()]
                    area_median = float(np.median(areas))
                    items: list[CandidateWorkItem] = []
                    for record in records:
                        slot = int(record["frame_slot"])
                        if slot not in masks:
                            continue
                        frame = cv2.imread(
                            str(output_root / "frames" / clip / f"frame_{slot:02d}.jpg")
                        )
                        if frame is None:
                            raise RuntimeError("candidate frame is unreadable")
                        metrics = calculate_candidate_metrics(
                            frame=frame,
                            mask=masks[slot],
                            area_median=area_median,
                            other_masks=other_masks.get(slot, []),
                        )
                        crop_box = padded_crop_box(
                            bbox_from_mask(masks[slot]),
                            image_width=frame.shape[1],
                            image_height=frame.shape[0],
                        )
                        items.append(
                            CandidateWorkItem(
                                record=record,
                                metrics=metrics,
                                neutral_crop=build_neutral_subject_crop(
                                    frame,
                                    masks[slot],
                                    crop_box,
                                ),
                                hard_rejection_reasons=list(
                                    basic_hard_rejection_reasons(
                                        metrics=metrics,
                                        sam_confidence=float(record["sam_confidence"]),
                                        config=config.ranking,
                                    )
                                ),
                            )
                        )

                    model_candidates = [
                        item for item in items if not item.hard_rejection_reasons
                    ]
                    dino_medoid_slot = None
                    before_models += len(model_candidates)
                    if not model_candidates:
                        write_json_atomic(
                            candidate_dir / "ranking_metadata.json",
                            _candidate_ranking_metadata(
                                items,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    if config.ranking.dinov3_enabled:
                        if dino is None:
                            raise RuntimeError("DINOv3 embedder was not initialized")
                        started = time.perf_counter()
                        embeddings = dino.encode(
                            [item.neutral_crop for item in model_candidates]
                        )
                        dino_inference_seconds += time.perf_counter() - started
                        if len(embeddings) != len(model_candidates):
                            raise RuntimeError(
                                "DINOv3 returned an unexpected embedding count"
                            )
                        save_dinov3_embeddings(
                            candidate_dir / "dinov3_embeddings.npz",
                            frame_slots=[item.frame_slot for item in model_candidates],
                            embeddings=embeddings,
                            model_name=config.ranking.dinov3_model_name,
                        )
                        normalized_sharpness = _min_max(
                            [
                                item.metrics.tenengrad_sharpness
                                for item in model_candidates
                            ]
                        )
                        (
                            temporal_metrics,
                            dino_medoid_slot,
                            warning,
                        ) = temporal_representation_metrics(
                            frame_slots=[item.frame_slot for item in model_candidates],
                            embeddings=embeddings,
                            sam_confidences=[
                                item.sam_confidence for item in model_candidates
                            ],
                            base_quality_scores=[
                                0.6 * normalized_sharpness[index]
                                + 0.4 * item.metrics.exposure_score
                                for index, item in enumerate(model_candidates)
                            ],
                            threshold=(
                                config.ranking.dinov3_cluster_similarity_threshold
                            ),
                        )
                        temporal_by_slot = {
                            metric.frame_slot: metric for metric in temporal_metrics
                        }
                        for index, item in enumerate(model_candidates):
                            item.dino_embedding = embeddings[index]
                            item.dino_metrics = temporal_by_slot[item.frame_slot]
                            if (
                                warning is None
                                and config.ranking.dinov3_exclude_cluster_outliers
                                and not item.dino_metrics.dino_in_stable_cluster
                            ):
                                item.hard_rejection_reasons.append(
                                    "dino_temporal_outlier"
                                )
                        if warning is not None:
                            _append_jsonl(
                                output_root / "logs" / "ranking_warnings.jsonl",
                                {
                                    "clip_uid": clip,
                                    "entity_id": entity.entity_id,
                                    "warning": warning,
                                },
                            )
                    model_candidates = [
                        item
                        for item in model_candidates
                        if not item.hard_rejection_reasons
                    ]
                    after_dino += len(model_candidates)
                    if not model_candidates:
                        write_json_atomic(
                            candidate_dir / "ranking_metadata.json",
                            _candidate_ranking_metadata(
                                items,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    if config.ranking.siglip2_enabled:
                        if siglip is None:
                            raise RuntimeError("SigLIP 2 aligner was not initialized")
                        distractors = siglip_distractor_entities(
                            annotation,
                            entity,
                        )
                        distractor_texts = [
                            other.grounding_prompt for other in distractors
                        ]
                        entity_id_by_text = {
                            other.grounding_prompt: other.entity_id
                            for other in distractors
                        }
                        started = time.perf_counter()
                        alignment_results = siglip.score(
                            [item.neutral_crop for item in model_candidates],
                            entity.grounding_prompt,
                            distractor_texts,
                        )
                        siglip_inference_seconds += time.perf_counter() - started
                        if len(alignment_results) != len(model_candidates):
                            raise RuntimeError(
                                "SigLIP 2 returned an unexpected score count"
                            )
                        for item, alignment in zip(
                            model_candidates,
                            alignment_results,
                        ):
                            item.alignment_metrics = alignment
                            item.best_matching_entity_id = (
                                entity.entity_id
                                if alignment.best_matching_text
                                == entity.grounding_prompt
                                else entity_id_by_text.get(alignment.best_matching_text)
                            )
                            if (
                                config.ranking.siglip2_hard_reject_wrong_entity
                                and is_siglip_wrong_entity(
                                    alignment,
                                    target_text=entity.grounding_prompt,
                                )
                            ):
                                item.hard_rejection_reasons.append(
                                    "siglip_wrong_entity"
                                )
                    model_candidates = [
                        item
                        for item in model_candidates
                        if not item.hard_rejection_reasons
                    ]
                    after_siglip += len(model_candidates)
                    if not model_candidates:
                        write_json_atomic(
                            candidate_dir / "ranking_metadata.json",
                            _candidate_ranking_metadata(
                                items,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    normalized_siglip_scores = _normalized_siglip_scores(
                        model_candidates
                    )
                    preliminary = _preliminary_top_candidates(
                        model_candidates,
                        config=config.ranking,
                    )
                    slots = [item.frame_slot for item in preliminary]
                    selected_dir = candidate_dir / "selected"
                    sheet = selected_dir / "top_candidates.jpg"
                    _top_candidate_sheet(
                        frame_paths={
                            slot: (
                                output_root / "frames" / clip / f"frame_{slot:02d}.jpg"
                            )
                            for slot in slots
                        },
                        masks={slot: masks[slot] for slot in slots},
                        candidates=[
                            (item.record, item.metrics) for item in preliminary
                        ],
                        destination=sheet,
                    )
                    review_result = qwen.review(
                        entity_id=entity.entity_id,
                        entity_phrase=entity.phrase,
                        contact_sheet=sheet,
                        frame_slots=slots,
                    )
                    reviews = {
                        review.frame_slot: review for review in review_result.candidates
                    }
                    ranked = rank_candidates(
                        [
                            (
                                item.frame_slot,
                                int(item.record["source_frame_index"]),
                                item.sam_confidence,
                                item.metrics,
                                reviews.get(
                                    item.frame_slot,
                                    _default_review(item.record),
                                ),
                            )
                            for item in preliminary
                        ],
                        config=config.ranking,
                        dino_metrics_by_slot={
                            item.frame_slot: item.dino_metrics
                            for item in preliminary
                            if item.dino_metrics is not None
                        },
                        alignment_metrics_by_slot={
                            item.frame_slot: item.alignment_metrics
                            for item in preliminary
                            if item.alignment_metrics is not None
                        },
                        normalized_siglip_scores_by_slot={
                            item.frame_slot: normalized_siglip_scores[item.frame_slot]
                            for item in preliminary
                            if item.frame_slot in normalized_siglip_scores
                        },
                        best_matching_entity_ids={
                            item.frame_slot: item.best_matching_entity_id
                            for item in preliminary
                            if item.best_matching_entity_id is not None
                        },
                    )
                    ranked_by_slot = {
                        candidate.frame_slot: candidate for candidate in ranked
                    }
                    for item in preliminary:
                        item.hard_rejection_reasons = list(
                            ranked_by_slot[item.frame_slot].hard_rejection_reasons
                        )
                    write_json_atomic(
                        candidate_dir / "ranking_metadata.json",
                        _candidate_ranking_metadata(
                            items,
                            qwen_reviews=reviews,
                            dino_medoid_slot=dino_medoid_slot,
                            qwen_suggested_best_frame_slot=(
                                review_result.best_frame_slot
                            ),
                        ),
                    )
                    valid = [item for item in ranked if not item.hard_rejection_reasons]
                    if not valid:
                        no_valid += 1
                        continue
                    selected = valid[0]
                    selected_work_item = next(
                        item
                        for item in preliminary
                        if item.frame_slot == selected.frame_slot
                    )
                    metadata = _save_reference(
                        output_root=output_root,
                        clip_uid=clip,
                        entity_id=entity.entity_id,
                        frame_path=(
                            output_root
                            / "frames"
                            / clip
                            / f"frame_{selected.frame_slot:02d}.jpg"
                        ),
                        mask=masks[selected.frame_slot],
                        selected=selected,
                        dino_embedding=selected_work_item.dino_embedding,
                        dino_medoid_slot=dino_medoid_slot,
                        qwen_suggested_best_frame_slot=(review_result.best_frame_slot),
                    )
                    reference_record = _reference_record(metadata, entity)
                    write_json_atomic(
                        reference_dir / "metadata.json",
                        reference_record,
                    )
                    selected_dir.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(
                        selected_dir / "selection.json",
                        reference_record,
                    )
                    processed += 1
                except Exception as exc:  # noqa: BLE001
                    failure: dict[str, object] = {
                        "clip_uid": clip,
                        "entity_id": entity.entity_id,
                        "error": str(exc),
                    }
                    if isinstance(exc, StructuredOutputFailure):
                        failure["structured_output"] = exc.to_dict()
                    _append_jsonl(
                        output_root / "logs" / "ranking_failed.jsonl",
                        failure,
                    )
                    failed += 1
    finally:
        if owns_siglip and siglip is not None:
            siglip.close()
        if owns_dino and dino is not None:
            dino.close()
    reconcile_references(output_root)
    return RankingStats(
        processed=processed,
        skipped_existing=skipped,
        no_valid_candidate=no_valid,
        failed=failed,
        dino_load_seconds=dino_load_seconds,
        dino_inference_seconds=dino_inference_seconds,
        siglip_load_seconds=siglip_load_seconds,
        siglip_inference_seconds=siglip_inference_seconds,
        candidate_count_before_models=before_models,
        candidate_count_after_dino=after_dino,
        candidate_count_after_siglip=after_siglip,
    )


def stats_dict(stats: RankingStats) -> dict[str, int | float]:
    return asdict(stats)
