from __future__ import annotations

import base64
import json
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
from r2v_data_v2.mask_utils import decode_mask, save_mask_png
from r2v_data_v2.metrics import (
    CandidateMetrics,
    calculate_candidate_metrics,
    padded_crop_box,
)
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic
from r2v_data_v2.schemas import (
    AnnotationEntity,
    AnnotationResult,
    CandidateJudgeResult,
    CandidateVisualReview,
)


@dataclass(frozen=True)
class RankedCandidate:
    frame_slot: int
    source_frame_index: int
    sam_confidence: float
    metrics: CandidateMetrics
    visual_review: CandidateVisualReview
    hard_rejection_reasons: tuple[str, ...]
    ranking_score: float


@dataclass(frozen=True)
class RankingStats:
    processed: int = 0
    skipped_existing: int = 0
    no_valid_candidate: int = 0
    failed: int = 0


def hard_rejection_reasons(
    *,
    metrics: CandidateMetrics,
    sam_confidence: float,
    visual_review: CandidateVisualReview,
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
) -> list[RankedCandidate]:
    if not candidates:
        return []
    sharpness_values = [item[3].tenengrad_sharpness for item in candidates]
    low, high = min(sharpness_values), max(sharpness_values)
    ranked: list[RankedCandidate] = []
    for frame_slot, source_index, sam_confidence, metrics, visual_review in candidates:
        reasons = hard_rejection_reasons(
            metrics=metrics,
            sam_confidence=sam_confidence,
            visual_review=visual_review,
            config=config,
        )
        normalized_sharpness = (
            1.0 if high == low else (metrics.tenengrad_sharpness - low) / (high - low)
        )
        isolation = max(0.0, 1.0 - metrics.maximum_other_mask_overlap)
        score = (
            0.25 * visual_review.completeness
            + 0.20 * visual_review.recognizability
            + 0.15 * normalized_sharpness
            + 0.15 * visual_review.mask_quality
            + 0.10 * sam_confidence
            + 0.10 * isolation
            + 0.05 * visual_review.visual_quality
        )
        ranked.append(
            RankedCandidate(
                frame_slot=frame_slot,
                source_frame_index=source_index,
                sam_confidence=sam_confidence,
                metrics=metrics,
                visual_review=visual_review,
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
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
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
        result = CandidateJudgeResult.model_validate_json(content)
        returned_slots = {candidate.frame_slot for candidate in result.candidates}
        if result.entity_id != entity_id or returned_slots != set(frame_slots):
            raise ValueError(
                "candidate review does not match the requested entity and slots"
            )
        if result.best_frame_slot not in returned_slots:
            raise ValueError("candidate review best_frame_slot is invalid")
        return result


def _top_candidate_sheet(
    *,
    frame_paths: dict[int, Path],
    masks: dict[int, np.ndarray],
    destination: Path,
) -> None:
    tiles: list[Image.Image] = []
    for slot, frame_path in frame_paths.items():
        image = Image.open(frame_path).convert("RGB")
        overlay = np.asarray(image).copy()
        overlay[masks[slot]] = (
            0.5 * overlay[masks[slot]] + 0.5 * np.array([255, 60, 60])
        ).astype(np.uint8)
        tile = Image.fromarray(overlay)
        ImageDraw.Draw(tile).text((10, 10), f"frame_slot={slot}", fill=(255, 255, 0))
        tiles.append(tile)
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width * len(tiles), height), color=(20, 20, 20))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (index * width, 0))
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


def _save_reference(
    *,
    output_root: Path,
    clip_uid: str,
    entity_id: str,
    frame_path: Path,
    mask: np.ndarray,
    selected: RankedCandidate,
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
    Image.fromarray(rgba, mode="RGBA").save(destination / "foreground_rgba.png")
    neutral = np.full_like(crop_rgb, 204)
    neutral[crop_mask] = crop_rgb[crop_mask]
    Image.fromarray(neutral).save(destination / "neutral_background.jpg", quality=95)
    metadata = {
        "clip_uid": clip_uid,
        "entity_id": entity_id,
        "frame_slot": selected.frame_slot,
        "source_frame_index": selected.source_frame_index,
        "ranking_score": selected.ranking_score,
        "sam_confidence": selected.sam_confidence,
        "metrics": selected.metrics.to_dict(),
        "visual_review": selected.visual_review.model_dump(mode="json"),
        "crop_xyxy": list(crop_box),
        "canonical_path": str(destination / "canonical.jpg"),
        "mask_path": str(destination / "mask.png"),
        "foreground_rgba_path": str(destination / "foreground_rgba.png"),
        "neutral_background_path": str(destination / "neutral_background.jpg"),
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
    processed = skipped = no_valid = failed = 0
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
                        (reference_dir / "metadata.json").read_text(encoding="utf-8")
                    )
                    if not isinstance(existing_metadata, dict):
                        raise TypeError("reference metadata must be a JSON object")
                    write_json_atomic(
                        reference_dir / "metadata.json",
                        _reference_record(existing_metadata, entity),
                    )
                    skipped += 1
                except Exception as exc:  # noqa: BLE001 - continue with other entities
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
                metric_records: list[tuple[dict[str, object], CandidateMetrics]] = []
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
                    metric_records.append((record, metrics))
                preliminary = sorted(
                    metric_records,
                    key=lambda item: (
                        item[1].border_touch,
                        -item[1].tenengrad_sharpness,
                        -float(item[0]["sam_confidence"]),
                    ),
                )[: config.ranking.top_k_for_vlm_judge]
                slots = [int(item[0]["frame_slot"]) for item in preliminary]
                selected_dir = candidate_dir / "selected"
                sheet = selected_dir / "top_candidates.jpg"
                _top_candidate_sheet(
                    frame_paths={
                        slot: output_root / "frames" / clip / f"frame_{slot:02d}.jpg"
                        for slot in slots
                    },
                    masks={slot: masks[slot] for slot in slots},
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
                            int(record["frame_slot"]),
                            int(record["source_frame_index"]),
                            float(record["sam_confidence"]),
                            metrics,
                            reviews.get(
                                int(record["frame_slot"]),
                                _default_review(record),
                            ),
                        )
                        for record, metrics in preliminary
                    ],
                    config=config.ranking,
                )
                valid = [item for item in ranked if not item.hard_rejection_reasons]
                if not valid:
                    no_valid += 1
                    continue
                selected = valid[0]
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
            except Exception as exc:  # noqa: BLE001 - continue with other entities
                _append_jsonl(
                    output_root / "logs" / "ranking_failed.jsonl",
                    {
                        "clip_uid": clip,
                        "entity_id": entity.entity_id,
                        "error": str(exc),
                    },
                )
                failed += 1
    reconcile_references(output_root)
    return RankingStats(processed, skipped, no_valid, failed)


def stats_dict(stats: RankingStats) -> dict[str, int]:
    return asdict(stats)
