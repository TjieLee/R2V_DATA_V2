from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from r2v_data_v2.v3.reference_quality import cheap_foreground_technical_metrics

NEAR_SILHOUETTE_RULE = "subject_near_silhouette_v1"
RELATIVE_BLUR_V2_RULE = "subject_relative_blur_v2"


class EntityLike(Protocol):
    reference_type: str


class CandidateLike(Protocol):
    candidate_id: str
    image_path: str
    mask: np.ndarray


@dataclass(frozen=True)
class ReferencePrefilterDecision:
    candidate_id: str
    flagged: bool
    flagged_by: tuple[str, ...]
    technical_metrics: dict[str, object] | None
    laplacian_ratio: float | None
    tenengrad_ratio: float | None
    relative_blur_v2_applicable: bool
    relative_blur_v2_inapplicable_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReferencePrefilterResult[CandidateT: CandidateLike]:
    original_candidates: tuple[CandidateT, ...]
    retained_candidates: tuple[CandidateT, ...]
    decisions: tuple[ReferencePrefilterDecision, ...]

    @property
    def filtered_count(self) -> int:
        return len(self.original_candidates) - len(self.retained_candidates)


def _finite_metric(metrics: Mapping[str, object], field: str) -> float:
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"technical metric {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"technical metric {field} must be finite")
    return result


def _safe_ratio(value: float, maximum: float) -> float | None:
    if maximum <= 0:
        return None
    ratio = value / maximum
    if not math.isfinite(ratio):
        raise ValueError("relative blur ratio must be finite")
    return ratio


def _subject_near_silhouette(metrics: Mapping[str, object]) -> bool:
    return (
        _finite_metric(metrics, "luma_mean") <= 15
        and _finite_metric(metrics, "dark_fraction_32") >= 0.95
        and _finite_metric(metrics, "laplacian_variance") <= 5
        and _finite_metric(metrics, "tenengrad_mean") <= 100
    )


def _subject_relative_blur_v2(
    metrics: Mapping[str, object],
    *,
    laplacian_ratio: float | None,
    tenengrad_ratio: float | None,
) -> bool:
    return bool(
        laplacian_ratio is not None
        and tenengrad_ratio is not None
        and laplacian_ratio <= 0.35
        and tenengrad_ratio <= 0.50
        and _finite_metric(metrics, "laplacian_variance") <= 50
        and _finite_metric(metrics, "tenengrad_mean") <= 1500
    )


def prefilter_entity_reference_candidates[CandidateT: CandidateLike](
    entity: EntityLike,
    candidates: Sequence[CandidateT],
    source_images: Mapping[str, Image.Image],
) -> ReferencePrefilterResult[CandidateT]:
    original = tuple(candidates)
    if not original:
        raise ValueError("reference prefilter requires candidates")
    candidate_ids = [candidate.candidate_id for candidate in original]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("reference prefilter candidate IDs must be unique")

    if entity.reference_type not in {"subject", "object"}:
        decisions = tuple(
            ReferencePrefilterDecision(
                candidate_id=candidate.candidate_id,
                flagged=False,
                flagged_by=(),
                technical_metrics=None,
                laplacian_ratio=None,
                tenengrad_ratio=None,
                relative_blur_v2_applicable=False,
                relative_blur_v2_inapplicable_reason="subject_or_object_only",
            )
            for candidate in original
        )
        return ReferencePrefilterResult(
            original_candidates=original,
            retained_candidates=original,
            decisions=decisions,
        )

    technical_metrics: list[dict[str, object]] = []
    laplacians: list[float] = []
    tenengrads: list[float] = []
    for candidate in original:
        source_image = source_images.get(candidate.image_path)
        if source_image is None:
            raise ValueError("reference prefilter source image is missing")
        source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        metrics = cheap_foreground_technical_metrics(source_rgb, candidate.mask)
        technical_metrics.append(metrics)
        laplacians.append(_finite_metric(metrics, "laplacian_variance"))
        tenengrads.append(_finite_metric(metrics, "tenengrad_mean"))

    relative_blur_applicable = len(original) == 3
    inapplicable_reason = (
        None if relative_blur_applicable else "requires_three_candidates"
    )
    max_laplacian = max(laplacians)
    max_tenengrad = max(tenengrads)
    decisions_list: list[ReferencePrefilterDecision] = []
    retained: list[CandidateT] = []
    for candidate, metrics, laplacian, tenengrad in zip(
        original,
        technical_metrics,
        laplacians,
        tenengrads,
        strict=True,
    ):
        laplacian_ratio = _safe_ratio(laplacian, max_laplacian)
        tenengrad_ratio = _safe_ratio(tenengrad, max_tenengrad)
        flagged_by: list[str] = []
        if entity.reference_type == "subject" and _subject_near_silhouette(metrics):
            flagged_by.append(NEAR_SILHOUETTE_RULE)
        if relative_blur_applicable and _subject_relative_blur_v2(
            metrics,
            laplacian_ratio=laplacian_ratio,
            tenengrad_ratio=tenengrad_ratio,
        ):
            flagged_by.append(RELATIVE_BLUR_V2_RULE)
        flagged = bool(flagged_by)
        if not flagged:
            retained.append(candidate)
        decisions_list.append(
            ReferencePrefilterDecision(
                candidate_id=candidate.candidate_id,
                flagged=flagged,
                flagged_by=tuple(flagged_by),
                technical_metrics=metrics,
                laplacian_ratio=laplacian_ratio,
                tenengrad_ratio=tenengrad_ratio,
                relative_blur_v2_applicable=relative_blur_applicable,
                relative_blur_v2_inapplicable_reason=inapplicable_reason,
            )
        )
    return ReferencePrefilterResult(
        original_candidates=original,
        retained_candidates=tuple(retained),
        decisions=tuple(decisions_list),
    )
