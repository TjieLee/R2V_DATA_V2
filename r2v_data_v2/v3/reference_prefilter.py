from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from r2v_data_v2.v3.reference_quality import cheap_foreground_technical_metrics

NEAR_SILHOUETTE_RULE = "subject_near_silhouette_v1"
# Legacy stats/debug key kept for compatibility. Production no longer applies
# relative-blur filtering.
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


def _subject_near_silhouette(metrics: Mapping[str, object]) -> bool:
    return (
        _finite_metric(metrics, "luma_mean") <= 15
        and _finite_metric(metrics, "dark_fraction_32") >= 0.95
        and _finite_metric(metrics, "laplacian_variance") <= 5
        and _finite_metric(metrics, "tenengrad_mean") <= 100
    )


def prefilter_entity_reference_candidates[CandidateT: CandidateLike](
    entity: EntityLike,
    candidates: Sequence[CandidateT],
    source_images: Mapping[str, Image.Image],
) -> ReferencePrefilterResult[CandidateT]:
    """Apply only the frozen near-silhouette subject prefilter.

    Relative sharpness/blur comparisons and the later extreme-blur filter are
    intentionally disabled. Objects and groups bypass technical measurement.
    """
    original = tuple(candidates)
    if not original:
        raise ValueError("reference prefilter requires candidates")
    candidate_ids = [candidate.candidate_id for candidate in original]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("reference prefilter candidate IDs must be unique")

    if entity.reference_type != "subject":
        decisions = tuple(
            ReferencePrefilterDecision(
                candidate_id=candidate.candidate_id,
                flagged=False,
                flagged_by=(),
                technical_metrics=None,
                laplacian_ratio=None,
                tenengrad_ratio=None,
                relative_blur_v2_applicable=False,
                relative_blur_v2_inapplicable_reason="subject_only",
            )
            for candidate in original
        )
        return ReferencePrefilterResult(
            original_candidates=original,
            retained_candidates=original,
            decisions=decisions,
        )

    decisions: list[ReferencePrefilterDecision] = []
    retained: list[CandidateT] = []
    for candidate in original:
        source_image = source_images.get(candidate.image_path)
        if source_image is None:
            raise ValueError("reference prefilter source image is missing")
        source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        metrics = cheap_foreground_technical_metrics(source_rgb, candidate.mask)
        flagged_by = (
            (NEAR_SILHOUETTE_RULE,) if _subject_near_silhouette(metrics) else ()
        )
        if not flagged_by:
            retained.append(candidate)
        decisions.append(
            ReferencePrefilterDecision(
                candidate_id=candidate.candidate_id,
                flagged=bool(flagged_by),
                flagged_by=flagged_by,
                technical_metrics=metrics,
                laplacian_ratio=None,
                tenengrad_ratio=None,
                relative_blur_v2_applicable=False,
                relative_blur_v2_inapplicable_reason="relative_blur_disabled",
            )
        )

    return ReferencePrefilterResult(
        original_candidates=original,
        retained_candidates=tuple(retained),
        decisions=tuple(decisions),
    )
