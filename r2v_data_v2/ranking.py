from __future__ import annotations

import inspect
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw

from prompts.qwen_candidate_judge_prompt import CANDIDATE_JUDGE_PROMPT
from r2v_data_v2.config import (
    PipelineConfig,
    QwenImageConfig,
    RankingConfig,
    _qwen_services,
)
from r2v_data_v2.image_utils import image_data_uri
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.mask_utils import bbox_from_mask, decode_mask, save_mask_png
from r2v_data_v2.metrics import (
    CandidateMetrics,
    calculate_candidate_metrics,
    padded_crop_box,
)
from r2v_data_v2.normalization import normalize_metric_values
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
    raw_scores: dict[str, float]
    normalized_scores: dict[str, float]
    effective_final_weights: dict[str, float]


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
    raw_scores: dict[str, float] = field(default_factory=dict)
    normalized_scores: dict[str, float] = field(default_factory=dict)
    preselection_score: float | None = None
    final_score: float | None = None

    @property
    def frame_slot(self) -> int:
        return int(self.record["frame_slot"])

    @property
    def sam_confidence(self) -> float:
        return float(self.record["sam_confidence"])


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _normalized_by_slot(
    *,
    frame_slots: list[int],
    raw_values: list[float],
    metric_name: str,
    config: RankingConfig,
) -> dict[int, float]:
    if len(frame_slots) != len(raw_values):
        raise ValueError("normalization slots and values must have equal lengths")
    try:
        policy = config.normalization[metric_name]
    except KeyError as exc:
        raise ValueError(
            f"missing normalization policy for metric: {metric_name}"
        ) from exc
    return dict(
        zip(
            frame_slots,
            normalize_metric_values(raw_values, policy),
        )
    )


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


def _border_completeness_score(metrics: CandidateMetrics) -> float:
    return 0.75 if metrics.border_touch else 1.0


def is_preferred_canonical_candidate(review: CandidateVisualReview) -> bool:
    return (
        review.occlusion <= 0.20
        and review.completeness >= 0.80
        and review.recognizability >= 0.75
        and review.canonical_view_score >= 0.70
    )


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
    if dino_temporal_outlier and config.evaluators.dinov3.hard_reject_outlier:
        reasons.append("dino_temporal_outlier")
    if (
        siglip_wrong_entity
        and config.evaluators.siglip2.hard_reject_wrong_entity
    ):
        reasons.append("siglip_wrong_entity")
    if config.evaluators.qwen_visual.enabled:
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


def _effective_weights(
    configured: dict[str, float],
    available_metrics: set[str],
) -> dict[str, float]:
    retained = {
        name: float(weight)
        for name, weight in configured.items()
        if name in available_metrics and weight > 0
    }
    total = sum(retained.values())
    if total <= 0:
        raise ValueError("at least one available metric must have a positive weight")
    return {name: weight / total for name, weight in retained.items()}


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
    frame_slots = [item[0] for item in candidates]
    normalized_by_metric = {
        "sharpness": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[3].tenengrad_sharpness for item in candidates],
            metric_name="sharpness",
            config=config,
        ),
        "exposure": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[3].exposure_score for item in candidates],
            metric_name="exposure",
            config=config,
        ),
        "isolation": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[_isolation_score(item[3]) for item in candidates],
            metric_name="isolation",
            config=config,
        ),
        "crop_subject_ratio": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[_crop_subject_ratio_score(item[3]) for item in candidates],
            metric_name="crop_subject_ratio",
            config=config,
        ),
        "sam_confidence": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[2] for item in candidates],
            metric_name="sam_confidence",
            config=config,
        ),
        "mask_area_continuity": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[3].mask_area_continuity for item in candidates],
            metric_name="mask_area_continuity",
            config=config,
        ),
        "qwen_completeness": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[4].completeness for item in candidates],
            metric_name="qwen_completeness",
            config=config,
        ),
        "qwen_recognizability": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[4].recognizability for item in candidates],
            metric_name="qwen_recognizability",
            config=config,
        ),
        "qwen_mask_quality": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[4].mask_quality for item in candidates],
            metric_name="qwen_mask_quality",
            config=config,
        ),
        "qwen_visual_quality": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[4].visual_quality for item in candidates],
            metric_name="qwen_visual_quality",
            config=config,
        ),
        "inverse_qwen_occlusion": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[1.0 - item[4].occlusion for item in candidates],
            metric_name="inverse_qwen_occlusion",
            config=config,
        ),
        "qwen_canonical_view": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[item[4].canonical_view_score for item in candidates],
            metric_name="qwen_canonical_view",
            config=config,
        ),
        "border_completeness": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[_border_completeness_score(item[3]) for item in candidates],
            metric_name="border_completeness",
            config=config,
        ),
        "dino_representativeness": _normalized_by_slot(
            frame_slots=frame_slots,
            raw_values=[
                (
                    dino_metrics_by_slot[slot].dino_representativeness
                    if slot in dino_metrics_by_slot
                    else 0.0
                )
                for slot in frame_slots
            ],
            metric_name="dino_representativeness",
            config=config,
        ),
    }
    aligned_slots = [
        frame_slot
        for frame_slot, _, _, _, _ in candidates
        if frame_slot in alignment_metrics_by_slot
    ]
    normalized_siglip_by_slot = _normalized_by_slot(
        frame_slots=aligned_slots,
        raw_values=[
            float(
                alignment_metrics_by_slot[slot].alignment_margin
                if alignment_metrics_by_slot[slot].alignment_margin is not None
                else alignment_metrics_by_slot[slot].target_similarity
            )
            for slot in aligned_slots
        ],
        metric_name="siglip_alignment",
        config=config,
    )
    normalized_siglip_by_slot.update(normalized_siglip_scores_by_slot)
    available_metrics = {
        "mask_area_continuity",
        "sharpness_exposure",
        "isolation",
        "border_completeness",
        "crop_subject_ratio",
        "sam_confidence",
    }
    if (
        config.evaluators.qwen_visual.enabled
        and config.evaluators.qwen_visual.use_for_final_score
    ):
        available_metrics.update(
            {
                "qwen_completeness",
                "qwen_recognizability",
                "qwen_mask_quality",
                "qwen_visual_quality",
                "inverse_qwen_occlusion",
                "qwen_canonical_view",
            }
        )
    if (
        config.evaluators.dinov3.enabled
        and config.evaluators.dinov3.use_for_final_score
        and all(item[0] in dino_metrics_by_slot for item in candidates)
    ):
        available_metrics.add("dino_representativeness")
    if (
        config.evaluators.siglip2.enabled
        and config.evaluators.siglip2.use_for_final_score
        and all(item[0] in normalized_siglip_by_slot for item in candidates)
    ):
        available_metrics.add("siglip_alignment")
    weights = _effective_weights(config.final_weights, available_metrics)
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
            "dino_representativeness": normalized_by_metric[
                "dino_representativeness"
            ][frame_slot],
            "qwen_completeness": normalized_by_metric["qwen_completeness"][
                frame_slot
            ],
            "qwen_recognizability": normalized_by_metric[
                "qwen_recognizability"
            ][frame_slot],
            "siglip_alignment": normalized_siglip_by_slot.get(frame_slot, 0.0),
            "qwen_mask_quality": normalized_by_metric["qwen_mask_quality"][
                frame_slot
            ],
            "mask_area_continuity": normalized_by_metric[
                "mask_area_continuity"
            ][frame_slot],
            "sharpness_exposure": (
                0.6 * normalized_by_metric["sharpness"][frame_slot]
                + 0.4 * normalized_by_metric["exposure"][frame_slot]
            ),
            "qwen_visual_quality": normalized_by_metric["qwen_visual_quality"][
                frame_slot
            ],
            "inverse_qwen_occlusion": normalized_by_metric[
                "inverse_qwen_occlusion"
            ][frame_slot],
            "qwen_canonical_view": normalized_by_metric[
                "qwen_canonical_view"
            ][frame_slot],
            "isolation": normalized_by_metric["isolation"][frame_slot],
            "border_completeness": normalized_by_metric[
                "border_completeness"
            ][frame_slot],
            "crop_subject_ratio": normalized_by_metric["crop_subject_ratio"][
                frame_slot
            ],
            "sam_confidence": normalized_by_metric["sam_confidence"][frame_slot],
        }
        raw_scores = {
            "sharpness": metrics.tenengrad_sharpness,
            "exposure": metrics.exposure_score,
            "sharpness_exposure": (
                0.6 * metrics.tenengrad_sharpness
                + 0.4 * metrics.exposure_score
            ),
            "mask_area_continuity": metrics.mask_area_continuity,
            "isolation": _isolation_score(metrics),
            "border_completeness": _border_completeness_score(metrics),
            "crop_subject_ratio": _crop_subject_ratio_score(metrics),
            "sam_confidence": sam_confidence,
        }
        normalized_scores = {
            "sharpness": normalized_by_metric["sharpness"][frame_slot],
            "exposure": normalized_by_metric["exposure"][frame_slot],
            "sharpness_exposure": component_scores["sharpness_exposure"],
            "mask_area_continuity": component_scores["mask_area_continuity"],
            "isolation": component_scores["isolation"],
            "border_completeness": component_scores["border_completeness"],
            "crop_subject_ratio": component_scores["crop_subject_ratio"],
            "sam_confidence": component_scores["sam_confidence"],
        }
        if config.evaluators.qwen_visual.enabled:
            raw_scores.update(
                {
                    "qwen_completeness": visual_review.completeness,
                    "qwen_recognizability": visual_review.recognizability,
                    "qwen_mask_quality": visual_review.mask_quality,
                    "qwen_visual_quality": visual_review.visual_quality,
                    "inverse_qwen_occlusion": 1.0 - visual_review.occlusion,
                    "qwen_canonical_view": visual_review.canonical_view_score,
                }
            )
            normalized_scores.update(
                {
                    name: component_scores[name]
                    for name in (
                        "qwen_completeness",
                        "qwen_recognizability",
                        "qwen_mask_quality",
                        "qwen_visual_quality",
                        "inverse_qwen_occlusion",
                        "qwen_canonical_view",
                    )
                }
            )
        if dino_metrics is not None:
            raw_scores["dino_representativeness"] = (
                dino_metrics.dino_representativeness
            )
            normalized_scores["dino_representativeness"] = component_scores[
                "dino_representativeness"
            ]
        if alignment_metrics is not None:
            raw_scores["siglip_alignment"] = float(
                alignment_metrics.alignment_margin
                if alignment_metrics.alignment_margin is not None
                else alignment_metrics.target_similarity
            )
            normalized_scores["siglip_alignment"] = component_scores[
                "siglip_alignment"
            ]
        score = sum(
            weights[name] * _clamp_unit(component_scores[name]) for name in weights
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
                raw_scores=raw_scores,
                normalized_scores=normalized_scores,
                effective_final_weights=dict(weights),
            )
        )
    has_preferred_valid_candidate = any(
        not item.hard_rejection_reasons
        and is_preferred_canonical_candidate(item.visual_review)
        for item in ranked
    )
    return sorted(
        ranked,
        key=lambda item: (
            bool(item.hard_rejection_reasons),
            (
                has_preferred_valid_candidate
                and not is_preferred_canonical_candidate(item.visual_review)
            ),
            -item.ranking_score,
            item.frame_slot,
        ),
    )


class QwenCandidateJudge:
    def __init__(self, config: QwenImageConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _request(self, *, prompt: str, image_url: str) -> str:
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
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
        category: str = "object",
        canonical_label: str = "",
    ) -> CandidateJudgeResult:
        prompt = (
            CANDIDATE_JUDGE_PROMPT.format(
                entity_phrase=entity_phrase,
                category=category,
                canonical_label=canonical_label,
            )
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
                image_url=image_data_uri(contact_sheet),
            ),
            original_request=prompt,
            model=CandidateJudgeResult,
            validate=validate,
        )


def _review_candidates(
    judge: object,
    *,
    entity_id: str,
    entity_phrase: str,
    category: str,
    canonical_label: str,
    contact_sheet: Path,
    frame_slots: list[int],
) -> CandidateJudgeResult:
    review = judge.review
    kwargs: dict[str, object] = {
        "entity_id": entity_id,
        "entity_phrase": entity_phrase,
        "contact_sheet": contact_sheet,
        "frame_slots": frame_slots,
    }
    try:
        parameters = inspect.signature(review).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    parameter_names = {parameter.name for parameter in parameters}
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_keywords or "category" in parameter_names:
        kwargs["category"] = category
    if accepts_keywords or "canonical_label" in parameter_names:
        kwargs["canonical_label"] = canonical_label
    return review(**kwargs)


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


def _preselection_available_metrics(
    candidates: list[CandidateWorkItem],
    config: RankingConfig,
) -> set[str]:
    available_metrics = {
        "sharpness",
        "exposure",
        "isolation",
        "sam_confidence",
    }
    if (
        config.evaluators.dinov3.enabled
        and config.evaluators.dinov3.use_for_preselection
        and all(candidate.dino_metrics is not None for candidate in candidates)
    ):
        available_metrics.add("dino_representativeness")
    if (
        config.evaluators.siglip2.enabled
        and config.evaluators.siglip2.use_for_preselection
        and all(
            candidate.alignment_metrics is not None for candidate in candidates
        )
    ):
        available_metrics.add("siglip_alignment")
    return available_metrics


def _preliminary_top_candidates(
    candidates: list[CandidateWorkItem],
    *,
    config: RankingConfig,
) -> list[CandidateWorkItem]:
    if not candidates:
        return []
    frame_slots = [candidate.frame_slot for candidate in candidates]
    normalized_sharpness = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[
            candidate.metrics.tenengrad_sharpness for candidate in candidates
        ],
        metric_name="sharpness",
        config=config,
    )
    normalized_dino = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[
            (
                candidate.dino_metrics.dino_representativeness
                if candidate.dino_metrics is not None
                else 0.0
            )
            for candidate in candidates
        ],
        metric_name="dino_representativeness",
        config=config,
    )
    normalized_exposure = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[candidate.metrics.exposure_score for candidate in candidates],
        metric_name="exposure",
        config=config,
    )
    normalized_isolation = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[_isolation_score(candidate.metrics) for candidate in candidates],
        metric_name="isolation",
        config=config,
    )
    normalized_sam = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[candidate.sam_confidence for candidate in candidates],
        metric_name="sam_confidence",
        config=config,
    )
    normalized_continuity = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[
            candidate.metrics.mask_area_continuity for candidate in candidates
        ],
        metric_name="mask_area_continuity",
        config=config,
    )
    normalized_crop = _normalized_by_slot(
        frame_slots=frame_slots,
        raw_values=[
            _crop_subject_ratio_score(candidate.metrics) for candidate in candidates
        ],
        metric_name="crop_subject_ratio",
        config=config,
    )
    normalized_siglip = _normalized_siglip_scores(
        candidates,
        config=config,
    )
    weights = _effective_weights(
        config.preselection_weights,
        _preselection_available_metrics(candidates, config),
    )
    scored: list[tuple[float, CandidateWorkItem]] = []
    for candidate in candidates:
        components = {
            "dino_representativeness": normalized_dino[candidate.frame_slot],
            "siglip_alignment": normalized_siglip.get(candidate.frame_slot, 0.0),
            "sharpness": normalized_sharpness[candidate.frame_slot],
            "exposure": normalized_exposure[candidate.frame_slot],
            "isolation": normalized_isolation[candidate.frame_slot],
            "sam_confidence": normalized_sam[candidate.frame_slot],
        }
        candidate.raw_scores.update(
            {
                "sharpness": candidate.metrics.tenengrad_sharpness,
                "exposure": candidate.metrics.exposure_score,
                "sharpness_exposure": (
                    0.6 * candidate.metrics.tenengrad_sharpness
                    + 0.4 * candidate.metrics.exposure_score
                ),
                "mask_area_continuity": candidate.metrics.mask_area_continuity,
                "isolation": _isolation_score(candidate.metrics),
                "crop_subject_ratio": _crop_subject_ratio_score(candidate.metrics),
                "sam_confidence": candidate.sam_confidence,
            }
        )
        candidate.normalized_scores.update(
            {
                name: components[name]
                for name in ("sharpness", "exposure", "isolation", "sam_confidence")
            }
            | {
                "sharpness_exposure": (
                    0.6 * components["sharpness"] + 0.4 * components["exposure"]
                ),
                "mask_area_continuity": normalized_continuity[
                    candidate.frame_slot
                ],
                "crop_subject_ratio": normalized_crop[candidate.frame_slot],
            }
        )
        if candidate.dino_metrics is not None:
            candidate.raw_scores["dino_representativeness"] = (
                candidate.dino_metrics.dino_representativeness
            )
            candidate.normalized_scores["dino_representativeness"] = components[
                "dino_representativeness"
            ]
        if candidate.alignment_metrics is not None:
            candidate.raw_scores["siglip_alignment"] = float(
                candidate.alignment_metrics.alignment_margin
                if candidate.alignment_metrics.alignment_margin is not None
                else candidate.alignment_metrics.target_similarity
            )
            candidate.normalized_scores["siglip_alignment"] = components[
                "siglip_alignment"
            ]
        score = sum(
            weights[name] * _clamp_unit(components[name]) for name in weights
        )
        candidate.preselection_score = float(score)
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
    *,
    config: RankingConfig,
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
    return _normalized_by_slot(
        frame_slots=[candidate.frame_slot for candidate in aligned],
        raw_values=[float(value) for value in raw_scores],
        metric_name="siglip_alignment",
        config=config,
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
    config: RankingConfig,
    qwen_reviews: dict[int, CandidateVisualReview] | None = None,
    dino_medoid_slot: int | None = None,
    qwen_suggested_best_frame_slot: int | None = None,
    preselection_effective_weights: dict[str, float] | None = None,
    final_effective_weights: dict[str, float] | None = None,
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
                "raw_scores": item.raw_scores,
                "normalized_scores": item.normalized_scores,
                "preselection_score": item.preselection_score,
                "final_score": item.final_score,
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
        "preselection_effective_weights": preselection_effective_weights or {},
        "final_effective_weights": final_effective_weights or {},
        "normalization": {
            name: asdict(policy) for name, policy in config.normalization.items()
        },
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
    qwen_suggested_best_frame_slot: int | None,
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
    raw_canonical_path = destination / "canonical_raw.jpg"
    canonical_path = destination / "canonical.jpg"
    Image.fromarray(crop_rgb).save(raw_canonical_path, quality=95)
    shutil.copyfile(raw_canonical_path, canonical_path)
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
        "raw_scores": selected.raw_scores,
        "normalized_scores": selected.normalized_scores,
        "effective_final_weights": selected.effective_final_weights,
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
        "raw_canonical_path": str(raw_canonical_path),
        "canonical_path": str(canonical_path),
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
        "raw_canonical_path": metadata.get(
            "raw_canonical_path",
            metadata.get("canonical_path"),
        ),
        "reference_id": entity.entity_id,
        "reference_type": "entity",
        "phrase": entity.phrase,
        "canonical_label": entity.canonical_label,
        "category": entity.category,
        "ref_token": entity.ref_token,
        "genericity": entity.genericity,
        "name_evidence": entity.name_evidence,
        "separability": entity.separability,
        "inpainted": bool(metadata.get("inpainted", False)),
        "status": str(metadata.get("status", "ready")),
        "rejected": bool(metadata.get("rejected", False)),
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
    qwen = (
        judge
        or QwenCandidateJudge(_qwen_services(config.qwen).candidate_judge)
        if config.ranking.evaluators.qwen_visual.enabled
        else None
    )
    dino = dino_embedder
    siglip = siglip_aligner
    owns_dino = False
    owns_siglip = False
    dino_load_seconds = 0.0
    siglip_load_seconds = 0.0
    try:
        if config.ranking.evaluators.dinov3.enabled and dino is None:
            started = time.perf_counter()
            dino = DinoV3Embedder(config.ranking)
            dino_load_seconds = time.perf_counter() - started
            owns_dino = True
        if config.ranking.evaluators.siglip2.enabled and siglip is None:
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
                if not (
                    (candidate_dir / "candidates.jsonl").is_file()
                    and (candidate_dir / "top_masks.rle.json").is_file()
                ):
                    no_valid += 1
                    continue
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
                                config=config.ranking,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    if config.ranking.evaluators.dinov3.enabled:
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
                        frame_slots = [
                            item.frame_slot for item in model_candidates
                        ]
                        normalized_sharpness = _normalized_by_slot(
                            frame_slots=frame_slots,
                            raw_values=[
                                item.metrics.tenengrad_sharpness
                                for item in model_candidates
                            ],
                            metric_name="sharpness",
                            config=config.ranking,
                        )
                        normalized_exposure = _normalized_by_slot(
                            frame_slots=frame_slots,
                            raw_values=[
                                item.metrics.exposure_score
                                for item in model_candidates
                            ],
                            metric_name="exposure",
                            config=config.ranking,
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
                                0.6 * normalized_sharpness[item.frame_slot]
                                + 0.4 * normalized_exposure[item.frame_slot]
                                for item in model_candidates
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
                                and config.ranking.evaluators.dinov3.hard_reject_outlier
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
                                config=config.ranking,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    if config.ranking.evaluators.siglip2.enabled:
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
                                config.ranking.evaluators.siglip2.hard_reject_wrong_entity
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
                                config=config.ranking,
                                dino_medoid_slot=dino_medoid_slot,
                            ),
                        )
                        no_valid += 1
                        continue

                    normalized_siglip_scores = _normalized_siglip_scores(
                        model_candidates,
                        config=config.ranking,
                    )
                    preselection_effective_weights = _effective_weights(
                        config.ranking.preselection_weights,
                        _preselection_available_metrics(
                            model_candidates,
                            config.ranking,
                        ),
                    )
                    preselected = _preliminary_top_candidates(
                        model_candidates,
                        config=config.ranking,
                    )
                    qwen_suggested_best_frame_slot = None
                    reviews: dict[int, CandidateVisualReview] = {}
                    selected_dir = candidate_dir / "selected"
                    if config.ranking.evaluators.qwen_visual.enabled:
                        preliminary = preselected
                        slots = [item.frame_slot for item in preliminary]
                        sheet = selected_dir / "top_candidates.jpg"
                        _top_candidate_sheet(
                            frame_paths={
                                slot: (
                                    output_root
                                    / "frames"
                                    / clip
                                    / f"frame_{slot:02d}.jpg"
                                )
                                for slot in slots
                            },
                            masks={slot: masks[slot] for slot in slots},
                            candidates=[
                                (item.record, item.metrics) for item in preliminary
                            ],
                            destination=sheet,
                        )
                        if qwen is None:
                            raise RuntimeError(
                                "Qwen candidate judge was not initialized"
                            )
                        review_result = _review_candidates(
                            qwen,
                            entity_id=entity.entity_id,
                            entity_phrase=entity.phrase,
                            category=entity.category,
                            canonical_label=entity.canonical_label,
                            contact_sheet=sheet,
                            frame_slots=slots,
                        )
                        reviews = {
                            review.frame_slot: review
                            for review in review_result.candidates
                        }
                        qwen_suggested_best_frame_slot = (
                            review_result.best_frame_slot
                        )
                    else:
                        preliminary = model_candidates
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
                        ranked_candidate = ranked_by_slot[item.frame_slot]
                        item.hard_rejection_reasons = list(
                            ranked_candidate.hard_rejection_reasons
                        )
                        item.raw_scores = ranked_candidate.raw_scores
                        item.normalized_scores = ranked_candidate.normalized_scores
                        item.final_score = ranked_candidate.ranking_score
                    final_effective_weights = (
                        ranked[0].effective_final_weights if ranked else {}
                    )
                    write_json_atomic(
                        candidate_dir / "ranking_metadata.json",
                        _candidate_ranking_metadata(
                            items,
                            config=config.ranking,
                            qwen_reviews=reviews,
                            dino_medoid_slot=dino_medoid_slot,
                            qwen_suggested_best_frame_slot=(
                                qwen_suggested_best_frame_slot
                            ),
                            preselection_effective_weights=(
                                preselection_effective_weights
                            ),
                            final_effective_weights=final_effective_weights,
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
                        qwen_suggested_best_frame_slot=(
                            qwen_suggested_best_frame_slot
                        ),
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
                except FileNotFoundError:
                    no_valid += 1
                    continue
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
