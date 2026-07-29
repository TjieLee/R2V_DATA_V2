from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from r2v_data_v2.config import PipelineConfig, Sam3Config
from r2v_data_v2.entity_policy import requires_foreground_mask
from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.mask_utils import (
    bbox_from_mask,
    encode_mask,
    fill_small_enclosed_holes,
    save_mask_contact_sheet,
    touches_border,
)
from r2v_data_v2.schemas import AnnotationResult


@dataclass(frozen=True)
class SamObservation:
    frame_slot: int
    mask: np.ndarray
    confidence: float
    object_id: int


@dataclass(frozen=True)
class AnchorResult:
    frame_slot: int
    confidence: float
    object_id: int
    mask_area_ratio: float


@dataclass(frozen=True)
class SamExtractionStats:
    processed: int = 0
    sam_failed: int = 0
    no_valid_candidate: int = 0


class Sam3Backend:
    def __init__(
        self,
        config: Sam3Config,
        *,
        predictor: object | None = None,
        frame_count: int = 10,
    ) -> None:
        self.config = config
        self.frame_count = frame_count
        if predictor is not None:
            self.predictor = predictor
            return
        code_root = config.code_root.expanduser().resolve()
        checkpoint = (
            config.checkpoint.expanduser().resolve() if config.checkpoint else None
        )
        if not code_root.is_dir():
            raise FileNotFoundError(f"SAM3 code_root does not exist: {code_root}")
        if checkpoint is None:
            raise ValueError("sam3.checkpoint must be configured explicitly")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint}")
        sys.path.insert(0, str(code_root))
        from sam3.model_builder import build_sam3_video_predictor

        self.predictor = build_sam3_video_predictor(checkpoint_path=str(checkpoint))

    @staticmethod
    def _observations(frame_slot: int, outputs: dict[str, Any]) -> list[SamObservation]:
        masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
        if masks.size == 0:
            return []
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError("SAM3 masks must have N-H-W or N-1-H-W shape")
        probabilities = np.asarray(
            outputs.get("out_probs", np.ones(len(masks))),
            dtype=float,
        ).reshape(-1)
        object_ids = np.asarray(
            outputs.get("out_obj_ids", np.arange(len(masks))),
            dtype=int,
        ).reshape(-1)
        result: list[SamObservation] = []
        for index, mask in enumerate(masks):
            if not mask.any():
                continue
            result.append(
                SamObservation(
                    frame_slot=frame_slot,
                    mask=mask,
                    confidence=(
                        float(probabilities[index])
                        if index < len(probabilities)
                        else 1.0
                    ),
                    object_id=(
                        int(object_ids[index]) if index < len(object_ids) else index
                    ),
                )
            )
        return result

    def _start_session(self, frames_dir: Path) -> str:
        start = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(frames_dir)}
        )
        return str(start["session_id"])

    def _close_session(self, session_id: str) -> None:
        self.predictor.handle_request(
            {"type": "close_session", "session_id": session_id}
        )

    def _best_anchor_observation(
        self,
        frame_slot: int,
        outputs: dict[str, Any],
    ) -> SamObservation | None:
        candidates = [
            observation
            for observation in self._observations(frame_slot, outputs)
            if observation.confidence >= self.config.minimum_confidence
            and 0.001 <= float(observation.mask.mean()) <= 0.90
        ]
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        if not candidates:
            return None
        if (
            len(candidates) > 1
            and candidates[0].confidence - candidates[1].confidence < 0.10
        ):
            return None
        return candidates[0]

    def find_best_anchor(
        self,
        frames_dir: Path,
        grounding_prompt: str,
    ) -> AnchorResult | None:
        candidates: list[AnchorResult] = []
        for frame_slot in range(self.frame_count):
            session_id = self._start_session(frames_dir)
            try:
                prompted = self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_slot,
                        "text": grounding_prompt,
                    }
                )
                observation = self._best_anchor_observation(
                    int(prompted["frame_index"]),
                    prompted["outputs"],
                )
                if observation is not None:
                    candidates.append(
                        AnchorResult(
                            frame_slot=observation.frame_slot,
                            confidence=observation.confidence,
                            object_id=observation.object_id,
                            mask_area_ratio=float(observation.mask.mean()),
                        )
                    )
            finally:
                self._close_session(session_id)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item.confidence, item.mask_area_ratio, -item.frame_slot),
        )

    def track(self, *, frames_dir: Path, grounding_prompt: str) -> list[SamObservation]:
        anchor = self.find_best_anchor(frames_dir, grounding_prompt)
        if anchor is None:
            return []
        session_id = self._start_session(frames_dir)
        observations: list[SamObservation] = []
        try:
            prompted = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor.frame_slot,
                    "text": grounding_prompt,
                }
            )
            anchored = self._best_anchor_observation(
                int(prompted["frame_index"]),
                prompted["outputs"],
            )
            if anchored is None:
                return []
            tracked_object_id = anchored.object_id
            observations.append(anchored)
            for response in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                }
            ):
                current = self._observations(
                    int(response["frame_index"]),
                    response["outputs"],
                )
                if not current:
                    continue
                matching = [
                    item for item in current if item.object_id == tracked_object_id
                ]
                if len(matching) > 1:
                    raise ValueError(
                        "SAM3 returned duplicate observations for the tracked object"
                    )
                if not matching:
                    raise ValueError(
                        "SAM3 object identity switched across sampled frames"
                    )
                observations.append(matching[0])
        finally:
            self._close_session(session_id)
        object_ids = {item.object_id for item in observations}
        if len(object_ids) > 1:
            raise ValueError("SAM3 object identity switched across sampled frames")
        unique_by_slot = {item.frame_slot: item for item in observations}
        if len(unique_by_slot) < self.config.minimum_visible_frames:
            return []
        return [unique_by_slot[slot] for slot in sorted(unique_by_slot)]

    def close(self) -> None:
        shutdown = getattr(self.predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _annotation_from_payload(payload: dict[str, Any]) -> AnnotationResult:
    return AnnotationResult.model_validate(
        {
            "caption": payload["caption"],
            "prompt_with_refs": payload["prompt_with_refs"],
            "entities": payload["entities"],
            "relations": payload.get("relations", []),
            "background": payload.get("background"),
        }
    )


def _candidate_from_observation(
    *,
    observation: SamObservation,
    frame: np.ndarray,
    source_frame_index: int,
    config: PipelineConfig,
) -> tuple[dict[str, object], np.ndarray] | None:
    mask = np.asarray(observation.mask, dtype=bool)
    if mask.shape != frame.shape[:2] or not mask.any():
        return None
    mask = fill_small_enclosed_holes(mask, frame=frame)
    bbox = bbox_from_mask(mask)
    x1, y1, x2, y2 = bbox
    area_ratio = float(mask.mean())
    effective_short_side = min(x2 - x1, y2 - y1)
    if observation.confidence < config.sam3.minimum_confidence:
        return None
    if effective_short_side < config.ranking.minimum_effective_short_side:
        return None
    if not (
        config.ranking.minimum_mask_area_ratio
        <= area_ratio
        <= config.ranking.maximum_mask_area_ratio
    ):
        return None
    return (
        {
            "frame_slot": observation.frame_slot,
            "source_frame_index": source_frame_index,
            "bbox_xyxy": list(bbox),
            "mask_area_ratio": area_ratio,
            "sam_confidence": observation.confidence,
            "touches_border": touches_border(mask),
            "visible": True,
            "effective_short_side": effective_short_side,
            "mask_rle_key": None,
        },
        mask,
    )


def _tracked_mask_from_observation(
    *,
    observation: SamObservation,
    frame: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    mask = np.asarray(observation.mask, dtype=bool)
    if mask.shape != frame.shape[:2]:
        return None, ("mask_shape_mismatch",)
    if not mask.any():
        return None, ("empty_mask",)
    if observation.confidence < config.sam3.minimum_confidence:
        return None, ("sam_confidence",)
    return fill_small_enclosed_holes(mask, frame=frame), ()


def _candidate_filter_reasons(
    mask: np.ndarray,
    *,
    config: PipelineConfig,
) -> tuple[str, ...]:
    bbox = bbox_from_mask(mask)
    x1, y1, x2, y2 = bbox
    reasons: list[str] = []
    if min(x2 - x1, y2 - y1) < config.ranking.minimum_effective_short_side:
        reasons.append("effective_short_side")
    area_ratio = float(mask.mean())
    if not (
        config.ranking.minimum_mask_area_ratio
        <= area_ratio
        <= config.ranking.maximum_mask_area_ratio
    ):
        reasons.append("mask_area_ratio")
    return tuple(reasons)


def _write_candidates(
    *,
    output_root: Path,
    clip_uid: str,
    entity_id: str,
    candidates: list[dict[str, object]],
    masks: dict[int, np.ndarray],
    frame_paths: list[Path],
    save_top_k: int,
    tracked_masks: dict[int, np.ndarray] | None = None,
    mask_coverage: dict[int, dict[str, object]] | None = None,
) -> None:
    destination = output_root / "candidates" / clip_uid / entity_id
    destination.mkdir(parents=True, exist_ok=True)
    ranked_slots = [
        int(candidate["frame_slot"])
        for candidate in sorted(
            candidates,
            key=lambda item: (
                float(item["sam_confidence"]),
                int(item["effective_short_side"]),
            ),
            reverse=True,
        )[:save_top_k]
    ]
    encoded: dict[str, object] = {}
    for slot in ranked_slots:
        key = f"frame_{slot:02d}"
        encoded[key] = encode_mask(masks[slot])
        for candidate in candidates:
            if int(candidate["frame_slot"]) == slot:
                candidate["mask_rle_key"] = key
                break
    with (destination / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in sorted(candidates, key=lambda item: int(item["frame_slot"])):
            record = {"clip_uid": clip_uid, "entity_id": entity_id, **candidate}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (destination / "top_masks.rle.json").write_text(
        json.dumps(encoded),
        encoding="utf-8",
    )
    raw_masks = tracked_masks if tracked_masks is not None else masks
    (destination / "tracked_masks.rle.json").write_text(
        json.dumps(
            {
                f"frame_{slot:02d}": encode_mask(mask)
                for slot, mask in sorted(raw_masks.items())
            }
        ),
        encoding="utf-8",
    )
    coverage = mask_coverage or {
        slot: {
            "tracked": slot in raw_masks,
            "mask_available": slot in raw_masks,
            "candidate_accepted": slot in masks,
            "filtered_reasons": [],
        }
        for slot in range(len(frame_paths))
    }
    (destination / "mask_coverage.json").write_text(
        json.dumps(
            {
                "clip_uid": clip_uid,
                "entity_id": entity_id,
                "slots": {
                    f"frame_{slot:02d}": value
                    for slot, value in sorted(coverage.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if candidates:
        save_mask_contact_sheet(
            frame_paths=frame_paths,
            candidates=candidates,
            masks=masks,
            destination=destination / "contact_sheet.jpg",
        )


def _append_failure(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def extract_manifest_candidates(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    backend: Sam3Backend | None = None,
) -> SamExtractionStats:
    output_root = config.ensure_output_root()
    annotation_manifest = output_root / "manifests" / "annotations.jsonl"
    if not annotation_manifest.is_file():
        raise FileNotFoundError("run Stage 02 before SAM3 extraction")
    sam = backend or Sam3Backend(config.sam3, frame_count=config.frames.count)
    processed = failed = no_valid = 0
    try:
        for payload in iter_source_records(annotation_manifest):
            clip = str(payload["clip_uid"])
            annotation = _annotation_from_payload(payload)
            frame_dir = output_root / "frames" / clip
            frame_metadata = json.loads(
                (frame_dir / "frames.json").read_text(encoding="utf-8")
            )
            frame_paths = [
                frame_dir / f"frame_{slot:02d}.jpg"
                for slot in range(config.frames.count)
            ]
            for entity in annotation.entities:
                if not requires_foreground_mask(entity):
                    continue
                destination = (
                    output_root
                    / "candidates"
                    / clip
                    / entity.entity_id
                    / "candidates.jsonl"
                )
                if destination.is_file() and not overwrite:
                    continue
                try:
                    observations = sam.track(
                        frames_dir=frame_dir,
                        grounding_prompt=entity.grounding_prompt,
                    )
                    candidates: list[dict[str, object]] = []
                    masks: dict[int, np.ndarray] = {}
                    tracked_masks: dict[int, np.ndarray] = {}
                    mask_coverage: dict[int, dict[str, object]] = {
                        slot: {
                            "tracked": False,
                            "mask_available": False,
                            "candidate_accepted": False,
                            "filtered_reasons": ["no_tracked_mask"],
                        }
                        for slot in range(config.frames.count)
                    }
                    for observation in observations:
                        frame = cv2.imread(str(frame_paths[observation.frame_slot]))
                        if frame is None:
                            raise RuntimeError("sampled frame is unreadable")
                        tracked_mask, tracking_reasons = (
                            _tracked_mask_from_observation(
                                observation=observation,
                                frame=frame,
                                config=config,
                            )
                        )
                        coverage = {
                            "tracked": True,
                            "mask_available": tracked_mask is not None,
                            "candidate_accepted": False,
                            "filtered_reasons": list(tracking_reasons),
                        }
                        mask_coverage[observation.frame_slot] = coverage
                        if tracked_mask is None:
                            continue
                        tracked_masks[observation.frame_slot] = tracked_mask
                        item = _candidate_from_observation(
                            observation=observation,
                            frame=frame,
                            source_frame_index=int(
                                frame_metadata["sampled_indices"][
                                    observation.frame_slot
                                ]
                            ),
                            config=config,
                        )
                        filter_reasons = _candidate_filter_reasons(
                            tracked_mask,
                            config=config,
                        )
                        coverage["filtered_reasons"] = list(filter_reasons)
                        if item is not None:
                            candidate, mask = item
                            coverage["candidate_accepted"] = True
                            candidates.append(candidate)
                            masks[observation.frame_slot] = mask
                    _write_candidates(
                        output_root=output_root,
                        clip_uid=clip,
                        entity_id=entity.entity_id,
                        candidates=candidates,
                        masks=masks,
                        frame_paths=frame_paths,
                        save_top_k=config.ranking.save_top_k_mask_rle,
                        tracked_masks=tracked_masks,
                        mask_coverage=mask_coverage,
                    )
                    if len(candidates) < config.sam3.minimum_visible_frames:
                        no_valid += 1
                        _append_failure(
                            output_root / "logs" / "sam_failed.jsonl",
                            {
                                "clip_uid": clip,
                                "entity_id": entity.entity_id,
                                "error": "entity is visible in too few sampled frames",
                            },
                        )
                        continue
                    processed += 1
                except Exception as exc:  # noqa: BLE001 - continue with other entities
                    _append_failure(
                        output_root / "logs" / "sam_failed.jsonl",
                        {
                            "clip_uid": clip,
                            "entity_id": entity.entity_id,
                            "error": str(exc),
                        },
                    )
                    failed += 1
    finally:
        if backend is None:
            sam.close()
    return SamExtractionStats(processed, failed, no_valid)


def stats_dict(stats: SamExtractionStats) -> dict[str, int]:
    return asdict(stats)
