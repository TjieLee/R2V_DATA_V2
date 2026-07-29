from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import BadRequestError, OpenAI
from PIL import Image, ImageDraw

from prompts.qwen_background_judge_prompt import BACKGROUND_JUDGE_PROMPT
from r2v_data_v2.config import (
    BackgroundConfig,
    PipelineConfig,
    QwenImageConfig,
    _qwen_services,
)
from r2v_data_v2.image_utils import image_data_uri
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.mask_utils import decode_mask, save_mask_png
from r2v_data_v2.reconciliation import reconcile_references, write_json_atomic
from r2v_data_v2.schemas import (
    AnnotationResult,
    BackgroundJudgeResult,
    BackgroundVisualReview,
)
from r2v_data_v2.semantic_alignment import Siglip2Aligner
from r2v_data_v2.structured_output import (
    ValidationIssue,
    request_structured_output,
)
from r2v_data_v2.visual_embedding import (
    DinoV3Embedder,
    temporal_representation_metrics,
)


@dataclass(frozen=True)
class BackgroundReferenceStats:
    processed: int = 0
    skipped_existing: int = 0
    no_valid_candidate: int = 0
    failed: int = 0
    raw_background_count: int = 0
    needs_inpainting_count: int = 0


@dataclass(frozen=True)
class BackgroundCandidate:
    frame_slot: int
    source_frame_index: int
    frame_path: Path
    foreground_mask: np.ndarray
    foreground_area_ratio: float
    hole_area_ratio: float
    sharpness: float
    exposure: float
    ranking_score: float
    rejection_reasons: tuple[str, ...]
    visual_review: BackgroundVisualReview | None = None
    dino_representativeness: float | None = None
    siglip_alignment: float | None = None


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _annotation_from_payload(payload: dict[str, Any]) -> AnnotationResult:
    return AnnotationResult.model_validate(
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


def _load_foreground_masks(
    *,
    output_root: Path,
    clip_uid: str,
    annotation: AnnotationResult,
) -> tuple[dict[int, list[np.ndarray]], tuple[str, ...]]:
    masks_by_slot: dict[int, list[np.ndarray]] = {}
    incomplete_entities: list[str] = []
    for entity in annotation.entities:
        if not entity.reference_worthy:
            continue
        mask_path = (
            output_root
            / "candidates"
            / clip_uid
            / entity.entity_id
            / "top_masks.rle.json"
        )
        if not mask_path.is_file():
            incomplete_entities.append(entity.entity_id)
            continue
        encoded = json.loads(mask_path.read_text(encoding="utf-8"))
        if not isinstance(encoded, dict):
            raise TypeError(f"foreground mask artifact must be an object: {mask_path}")
        for key, value in encoded.items():
            if not key.startswith("frame_"):
                continue
            slot = int(key.removeprefix("frame_"))
            masks_by_slot.setdefault(slot, []).append(decode_mask(value))
        if not encoded:
            incomplete_entities.append(entity.entity_id)
    return masks_by_slot, tuple(dict.fromkeys(incomplete_entities))


def _union_mask(
    masks: list[np.ndarray],
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        binary = np.asarray(mask, dtype=bool)
        if binary.shape != shape:
            raise ValueError("foreground mask shape does not match sampled frame")
        union |= binary
    return union


def _frame_quality(frame_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sharpness = float(np.mean(gx * gx + gy * gy))
    clipped = np.mean((gray <= 8) | (gray >= 247))
    return sharpness, float(max(0.0, 1.0 - clipped))


def _normalize_sharpness(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.5] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def _basic_rejection_reasons(
    *,
    foreground_area_ratio: float,
    hole_area_ratio: float,
    sharpness: float,
    exposure: float,
    config: BackgroundConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if foreground_area_ratio > config.maximum_foreground_area_ratio:
        reasons.append("foreground_area_ratio")
    if hole_area_ratio > config.maximum_hole_area_ratio:
        reasons.append("hole_area_ratio")
    if exposure < config.minimum_exposure_score:
        reasons.append("extreme_exposure")
    if sharpness < config.minimum_sharpness:
        reasons.append("blurred")
    return tuple(reasons)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _contact_sheet(
    candidates: list[BackgroundCandidate],
    destination: Path,
) -> None:
    panel_width = 480
    panel_height = 320
    label_height = 38
    rows: list[Image.Image] = []
    for candidate in candidates:
        image = Image.open(candidate.frame_path).convert("RGB")
        image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (panel_width, panel_height), (24, 24, 24))
        panel.paste(
            image,
            ((panel_width - image.width) // 2, (panel_height - image.height) // 2),
        )
        row = Image.new(
            "RGB",
            (panel_width, panel_height + label_height),
            (16, 16, 16),
        )
        ImageDraw.Draw(row).text(
            (8, 8),
            (
                f"frame_slot={candidate.frame_slot} "
                f"foreground={candidate.foreground_area_ratio:.4f} "
                f"exposure={candidate.exposure:.3f}"
            ),
            fill=(255, 255, 0),
        )
        row.paste(panel, (0, label_height))
        rows.append(row)
    sheet = Image.new(
        "RGB",
        (panel_width, len(rows) * (panel_height + label_height)),
        (16, 16, 16),
    )
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * (panel_height + label_height)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


class QwenBackgroundJudge:
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
                        "name": "background_judge",
                        "strict": True,
                        "schema": BackgroundJudgeResult.model_json_schema(),
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
            raise RuntimeError("Qwen returned an empty background review")
        return content

    def review(
        self,
        *,
        background_phrase: str,
        contact_sheet: Path,
        frame_slots: list[int],
    ) -> BackgroundJudgeResult:
        prompt = (
            BACKGROUND_JUDGE_PROMPT.format(
                background_phrase=background_phrase
            )
            + f"\nframe_slots={frame_slots}"
        )
        def validate(result: BackgroundJudgeResult) -> list[ValidationIssue]:
            returned = [candidate.frame_slot for candidate in result.candidates]
            issues: list[ValidationIssue] = []
            if len(returned) != len(frame_slots) or set(returned) != set(frame_slots):
                issues.append(
                    ValidationIssue(
                        "candidate_slots_mismatch",
                        "candidates",
                        "background review must match the requested slots",
                    )
                )
            if result.best_frame_slot not in returned:
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
            model=BackgroundJudgeResult,
            validate=validate,
        )


def _apply_visual_reviews(
    candidates: list[BackgroundCandidate],
    reviews: dict[int, BackgroundVisualReview],
) -> list[BackgroundCandidate]:
    reviewed: list[BackgroundCandidate] = []
    for candidate in candidates:
        review = reviews.get(candidate.frame_slot)
        if review is None:
            reviewed.append(candidate)
            continue
        reasons = list(candidate.rejection_reasons)
        if review.scene_completeness < 0.5:
            reasons.append("scene_incomplete")
        if review.scene_recognizability < 0.5:
            reasons.append("scene_unrecognizable")
        if not review.reusable_as_background:
            reasons.append("not_reusable_as_background")
        reasons.extend(review.rejection_reasons)
        visual_score = (
            0.30 * review.scene_completeness
            + 0.25 * review.scene_recognizability
            + 0.25 * review.visual_quality
            + 0.20 * (1.0 - review.foreground_distraction)
        )
        reviewed.append(
            BackgroundCandidate(
                **{
                    **asdict(candidate),
                    "ranking_score": (
                        0.65 * candidate.ranking_score + 0.35 * visual_score
                    ),
                    "rejection_reasons": tuple(dict.fromkeys(reasons)),
                    "visual_review": review,
                }
            )
        )
    return reviewed


def _save_reference(
    *,
    output_root: Path,
    clip_uid: str,
    phrase: str,
    candidate: BackgroundCandidate,
    raw_threshold: float,
) -> dict[str, object]:
    destination = output_root / "references" / clip_uid / "bg1"
    raw_path = destination / "canonical_raw.jpg"
    canonical_path = destination / "canonical.jpg"
    mask_path = destination / "foreground_mask.png"
    _copy_atomic(candidate.frame_path, raw_path)
    _copy_atomic(raw_path, canonical_path)
    save_mask_png(mask_path, candidate.foreground_mask)
    record: dict[str, object] = {
        "clip_uid": clip_uid,
        "reference_id": "bg1",
        "reference_type": "background",
        "entity_id": None,
        "phrase": phrase,
        "canonical_label": "background",
        "category": "background",
        "ref_token": "<ref_bg_1>",
        "source_frame_index": candidate.source_frame_index,
        "frame_slot": candidate.frame_slot,
        "raw_canonical_path": str(raw_path),
        "canonical_path": str(canonical_path),
        "mask_path": str(mask_path),
        "foreground_area_ratio": candidate.foreground_area_ratio,
        "hole_area_ratio": candidate.hole_area_ratio,
        "ranking_score": candidate.ranking_score,
        "visual_review": (
            candidate.visual_review.model_dump(mode="json")
            if candidate.visual_review is not None
            else None
        ),
        "needs_inpainting": (
            candidate.foreground_area_ratio > raw_threshold
        ),
        "inpainted": False,
        "status": (
            "pending_inpainting"
            if candidate.foreground_area_ratio > raw_threshold
            else "ready"
        ),
        "rejected": False,
    }
    write_json_atomic(destination / "reference_metadata.json", record)
    return record


def build_background_references(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    judge: QwenBackgroundJudge | None = None,
    dino_embedder: DinoV3Embedder | None = None,
    siglip_aligner: Siglip2Aligner | None = None,
) -> BackgroundReferenceStats:
    if not config.background.enabled:
        return BackgroundReferenceStats()
    output_root = config.ensure_output_root()
    annotation_manifest = output_root / "manifests" / "annotations.jsonl"
    if not annotation_manifest.is_file():
        raise FileNotFoundError("run annotation and frame stages before background")
    qwen = judge
    if config.background.qwen_judge_enabled and qwen is None:
        qwen = QwenBackgroundJudge(
            _qwen_services(config.qwen).background_judge
        )
    dino = dino_embedder
    siglip = siglip_aligner
    owns_dino = False
    owns_siglip = False

    processed = skipped = no_valid = failed = raw_count = inpaint_count = 0
    for payload in iter_source_records(annotation_manifest):
        clip_uid = str(payload["clip_uid"])
        try:
            annotation = _annotation_from_payload(payload)
            background = annotation.background
            deferred = "background_reference_deferred" in payload.get(
                "warnings", []
            )
            if background is None or not (
                background.reference_worthy or deferred
            ):
                continue
            reference_dir = output_root / "references" / clip_uid / "bg1"
            metadata_path = reference_dir / "reference_metadata.json"
            if metadata_path.is_file() and not overwrite:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    raise TypeError(
                        f"background reference metadata must be an object: "
                        f"{metadata_path}"
                    )
                if existing.get("rejected") is True:
                    existing["status"] = "rejected"
                elif (
                    existing.get("needs_inpainting") is True
                    and existing.get("inpainted") is not True
                ):
                    existing["status"] = "pending_inpainting"
                else:
                    existing["status"] = "ready"
                write_json_atomic(metadata_path, existing)
                skipped += 1
                continue
            if overwrite:
                for filename in (
                    "canonical_raw.jpg",
                    "canonical.jpg",
                    "foreground_mask.png",
                    "reference_metadata.json",
                ):
                    (reference_dir / filename).unlink(missing_ok=True)

            frame_dir = output_root / "frames" / clip_uid
            frame_metadata = json.loads(
                (frame_dir / "frames.json").read_text(encoding="utf-8")
            )
            source_indices = frame_metadata["sampled_indices"]
            masks_by_slot, incomplete_mask_entities = _load_foreground_masks(
                output_root=output_root,
                clip_uid=clip_uid,
                annotation=annotation,
            )
            raw_candidates: list[dict[str, object]] = []
            candidate_dir = output_root / "background_candidates" / clip_uid
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for slot in range(config.frames.count):
                frame_path = frame_dir / f"frame_{slot:02d}.jpg"
                stored_path = candidate_dir / f"frame_{slot:02d}.jpg"
                _copy_atomic(frame_path, stored_path)
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"sampled frame is unreadable: {frame_path}")
                foreground_mask = _union_mask(
                    masks_by_slot.get(slot, []),
                    shape=frame.shape[:2],
                )
                foreground_ratio = float(foreground_mask.mean())
                sharpness, exposure = _frame_quality(frame)
                raw_candidates.append(
                    {
                        "frame_slot": slot,
                        "source_frame_index": int(source_indices[slot]),
                        "frame_path": stored_path,
                        "foreground_mask": foreground_mask,
                        "foreground_area_ratio": foreground_ratio,
                        "hole_area_ratio": foreground_ratio,
                        "sharpness": sharpness,
                        "exposure": exposure,
                        "rejection_reasons": _basic_rejection_reasons(
                            foreground_area_ratio=foreground_ratio,
                            hole_area_ratio=foreground_ratio,
                            sharpness=sharpness,
                            exposure=exposure,
                            config=config.background,
                        )
                        + (
                            ("incomplete_foreground_masks",)
                            if incomplete_mask_entities
                            else ()
                        ),
                    }
                )
            normalized_sharpness = _normalize_sharpness(
                [float(item["sharpness"]) for item in raw_candidates]
            )
            base_scores = [
                (
                    0.40 * normalized_sharpness[index]
                    + 0.35 * float(item["exposure"])
                    + 0.25 * (1.0 - float(item["foreground_area_ratio"]))
                )
                for index, item in enumerate(raw_candidates)
            ]
            if not incomplete_mask_entities:
                if (
                    config.ranking.evaluators.dinov3.enabled
                    and dino is None
                ):
                    dino = DinoV3Embedder(config.ranking)
                    owns_dino = True
                if (
                    config.ranking.evaluators.siglip2.enabled
                    and siglip is None
                ):
                    siglip = Siglip2Aligner(
                        config.ranking.siglip2_model_path,
                        config.ranking.siglip2_batch_size,
                    )
                    owns_siglip = True
            dino_scores: dict[int, float] = {}
            if dino is not None and not incomplete_mask_entities:
                images = [
                    Image.open(Path(item["frame_path"])).convert("RGB")
                    for item in raw_candidates
                ]
                embeddings = dino.encode(images)
                metrics, _, _ = temporal_representation_metrics(
                    frame_slots=[
                        int(item["frame_slot"]) for item in raw_candidates
                    ],
                    embeddings=embeddings,
                    sam_confidences=[1.0] * len(raw_candidates),
                    base_quality_scores=base_scores,
                    threshold=config.ranking.dinov3_cluster_similarity_threshold,
                )
                dino_scores = {
                    metric.frame_slot: metric.dino_representativeness
                    for metric in metrics
                }
            siglip_scores: dict[int, float] = {}
            if siglip is not None and not incomplete_mask_entities:
                images = [
                    Image.open(Path(item["frame_path"])).convert("RGB")
                    for item in raw_candidates
                ]
                alignment = siglip.score(images, background.phrase, [])
                if len(alignment) != len(raw_candidates):
                    raise RuntimeError(
                        "SigLIP returned an unexpected background score count"
                    )
                siglip_scores = {
                    int(item["frame_slot"]): result.target_similarity
                    for item, result in zip(raw_candidates, alignment)
                }
            candidates = [
                BackgroundCandidate(
                    **item,
                    ranking_score=(
                        (
                            0.50 * base_scores[index]
                            + 0.30
                            * dino_scores.get(int(item["frame_slot"]), 0.0)
                            + 0.20
                            * siglip_scores.get(int(item["frame_slot"]), 0.0)
                        )
                        / (
                            0.50
                            + (0.30 if dino_scores else 0.0)
                            + (0.20 if siglip_scores else 0.0)
                        )
                    ),
                    dino_representativeness=dino_scores.get(
                        int(item["frame_slot"])
                    ),
                    siglip_alignment=siglip_scores.get(int(item["frame_slot"])),
                )
                for index, item in enumerate(raw_candidates)
            ]
            valid = [
                candidate
                for candidate in candidates
                if not candidate.rejection_reasons
            ]
            if valid:
                top = sorted(
                    valid,
                    key=lambda candidate: (
                        -candidate.ranking_score,
                        candidate.frame_slot,
                    ),
                )[: config.background.top_k_for_vlm_judge]
                sheet_path = candidate_dir / "contact_sheet.jpg"
                _contact_sheet(top, sheet_path)
            if config.background.qwen_judge_enabled and valid:
                if qwen is None:
                    raise RuntimeError("Qwen background judge was not initialized")
                result = qwen.review(
                    background_phrase=background.phrase,
                    contact_sheet=sheet_path,
                    frame_slots=[candidate.frame_slot for candidate in top],
                )
                reviews = {
                    review.frame_slot: review for review in result.candidates
                }
                candidates = _apply_visual_reviews(candidates, reviews)
                valid = [
                    candidate
                    for candidate in candidates
                    if candidate.frame_slot in reviews
                    and not candidate.rejection_reasons
                ]

            selected = min(
                valid,
                key=lambda candidate: (
                    -candidate.ranking_score,
                    candidate.frame_slot,
                ),
                default=None,
            )
            write_json_atomic(
                candidate_dir / "ranking_metadata.json",
                {
                    "clip_uid": clip_uid,
                    "background_phrase": background.phrase,
                    "incomplete_mask_entities": list(incomplete_mask_entities),
                    "selected_frame_slot": (
                        selected.frame_slot if selected is not None else None
                    ),
                    "candidates": [
                        {
                            **asdict(candidate),
                            "frame_path": str(candidate.frame_path),
                            "foreground_mask": None,
                            "visual_review": (
                                candidate.visual_review.model_dump(mode="json")
                                if candidate.visual_review is not None
                                else None
                            ),
                        }
                        for candidate in candidates
                    ],
                },
            )
            if selected is None:
                _append_jsonl(
                    output_root / "logs" / "background_rejected.jsonl",
                    {
                        "clip_uid": clip_uid,
                        "reason": (
                            "incomplete foreground masks"
                            if incomplete_mask_entities
                            else "no valid background candidate"
                        ),
                        "incomplete_mask_entities": list(
                            incomplete_mask_entities
                        ),
                    },
                )
                no_valid += 1
                continue
            record = _save_reference(
                output_root=output_root,
                clip_uid=clip_uid,
                phrase=background.phrase,
                candidate=selected,
                raw_threshold=config.background.raw_foreground_area_ratio,
            )
            if record["needs_inpainting"]:
                inpaint_count += 1
            else:
                raw_count += 1
            processed += 1
        except Exception as exc:  # noqa: BLE001
            _append_jsonl(
                output_root / "logs" / "background_failed.jsonl",
                {
                    "clip_uid": clip_uid,
                    "error": str(exc),
                },
            )
            failed += 1
    reconcile_references(output_root)
    if owns_dino and dino is not None:
        dino.close()
    if owns_siglip and siglip is not None:
        siglip.close()
    return BackgroundReferenceStats(
        processed=processed,
        skipped_existing=skipped,
        no_valid_candidate=no_valid,
        failed=failed,
        raw_background_count=raw_count,
        needs_inpainting_count=inpaint_count,
    )
