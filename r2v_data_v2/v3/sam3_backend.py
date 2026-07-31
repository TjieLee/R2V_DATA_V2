from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from r2v_data_v2.v3.config import Sam3Config

TrackStatus = Literal["ready", "not_found", "failed"]
_ANCHOR_PROBE_ORDER = (5, 2, 7, 0, 9)


@dataclass(frozen=True)
class BackendMaskObservation:
    slot: int
    mask: np.ndarray
    confidence: float
    object_id: str
    valid: bool = True


@dataclass(frozen=True)
class EntityTrackResult:
    status: TrackStatus
    observations: tuple[BackendMaskObservation, ...] = ()
    reason: str | None = None
    group_tracks_verified: bool = False

    def __post_init__(self) -> None:
        if self.status == "ready" and not self.observations:
            raise ValueError("ready entity track requires observations")
        if self.status != "ready" and self.observations:
            raise ValueError("non-ready entity track cannot publish observations")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed entity track requires a reason")


class SegmentationBackend(Protocol):
    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
    ) -> EntityTrackResult: ...


class Sam3SegmentationBackend:
    """Lazy adapter for the predictor API already exercised by the V2 pipeline."""

    def __init__(
        self,
        config: Sam3Config,
        *,
        predictor: object | None = None,
        builder: Callable[..., object] | None = None,
    ) -> None:
        self.config = config
        self._predictor = predictor
        self._builder = builder

    def _load_predictor(self) -> object:
        if self._predictor is not None:
            return self._predictor
        if self.config.model_path is None:
            raise ValueError(
                "sam3.model_path must be configured before segment runs"
            )
        model_path = self.config.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"SAM3 model checkpoint does not exist: {model_path}"
            )
        builder = self._builder
        if builder is None:
            from sam3.model_builder import build_sam3_video_predictor

            builder = build_sam3_video_predictor
        parameters = inspect.signature(builder).parameters
        arguments: dict[str, object] = {
            "checkpoint_path": str(model_path),
        }
        if "device" in parameters:
            arguments["device"] = self.config.device
        elif self.config.device != "cuda":
            raise ValueError(
                "installed SAM3 builder does not expose a device parameter; "
                "only its verified default cuda device can be requested"
            )
        self._predictor = builder(**arguments)
        return self._predictor

    @staticmethod
    def _observations(
        slot: int,
        outputs: dict[str, Any],
    ) -> list[BackendMaskObservation]:
        raw_masks = np.asarray(outputs.get("out_binary_masks", []))
        if raw_masks.size == 0:
            return []
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None, ...]
        if raw_masks.ndim == 4 and raw_masks.shape[1] == 1:
            raw_masks = raw_masks[:, 0]
        if raw_masks.ndim != 3:
            raise ValueError("SAM3 masks must have N-H-W or N-1-H-W shape")
        if not np.isin(raw_masks, (0, 1)).all():
            raise ValueError("SAM3 returned a non-binary mask")

        probabilities = np.asarray(outputs.get("out_probs", [])).reshape(-1)
        object_ids = np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)
        if len(probabilities) != len(raw_masks):
            raise ValueError("SAM3 confidence count does not match mask count")
        if len(object_ids) != len(raw_masks):
            raise ValueError("SAM3 object ID count does not match mask count")

        observations: list[BackendMaskObservation] = []
        for index, raw_mask in enumerate(raw_masks):
            mask = np.asarray(raw_mask, dtype=bool)
            if not mask.any():
                continue
            confidence = float(probabilities[index])
            if not math.isfinite(confidence):
                raise ValueError("SAM3 returned a non-finite confidence")
            observations.append(
                BackendMaskObservation(
                    slot=slot,
                    mask=mask.copy(),
                    confidence=confidence,
                    object_id=str(object_ids[index]),
                )
            )
        return observations

    @staticmethod
    def _frames_dir(frame_paths: list[Path]) -> Path:
        if not frame_paths:
            raise ValueError("SAM3 requires sampled frame paths")
        resolved = [path.expanduser().resolve() for path in frame_paths]
        parent = resolved[0].parent
        if any(path.parent != parent for path in resolved):
            raise ValueError("SAM3 frame paths must share one directory")
        if any(not path.is_file() for path in resolved):
            raise FileNotFoundError("one or more sampled SAM3 frames are missing")
        return parent

    @staticmethod
    def _start_session(predictor: object, frames_dir: Path) -> str:
        start = predictor.handle_request(  # type: ignore[attr-defined]
            {"type": "start_session", "resource_path": str(frames_dir)}
        )
        return str(start["session_id"])

    @staticmethod
    def _close_session(predictor: object, session_id: str) -> None:
        predictor.handle_request(  # type: ignore[attr-defined]
            {"type": "close_session", "session_id": session_id}
        )

    def _prompt_frame(
        self,
        predictor: object,
        *,
        frames_dir: Path,
        slot: int,
        grounding_prompt: str,
    ) -> list[BackendMaskObservation]:
        session_id = self._start_session(predictor, frames_dir)
        try:
            prompted = predictor.handle_request(  # type: ignore[attr-defined]
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": slot,
                    "text": grounding_prompt,
                }
            )
            prompted_slot = int(prompted["frame_index"])
            return self._observations(prompted_slot, prompted["outputs"])
        finally:
            self._close_session(predictor, session_id)

    def _find_anchor(
        self,
        predictor: object,
        *,
        frames_dir: Path,
        frame_count: int,
        reference_type: str,
        grounding_prompt: str,
    ) -> tuple[int | None, str | None]:
        ambiguous_instance = False
        for slot in _ANCHOR_PROBE_ORDER:
            if slot >= frame_count:
                continue
            observations = self._prompt_frame(
                predictor,
                frames_dir=frames_dir,
                slot=slot,
                grounding_prompt=grounding_prompt,
            )
            if len(observations) > 1:
                if reference_type == "group":
                    return None, "unverified_multi_object_group"
                ambiguous_instance = True
                continue
            if not observations:
                continue
            return slot, None
        if ambiguous_instance:
            return None, "ambiguous_multi_object_instance"
        return None, None

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
    ) -> EntityTrackResult:
        del entity_id
        if reference_type not in {"subject", "object", "group"}:
            return EntityTrackResult(
                status="failed",
                reason=f"unsupported reference type: {reference_type}",
            )
        if not grounding_prompt.strip():
            return EntityTrackResult(
                status="failed",
                reason="grounding prompt is empty",
            )
        frames_dir = self._frames_dir(frame_paths)
        predictor = self._load_predictor()
        anchor_slot, anchor_failure = self._find_anchor(
            predictor,
            frames_dir=frames_dir,
            frame_count=len(frame_paths),
            reference_type=reference_type,
            grounding_prompt=grounding_prompt,
        )
        if anchor_slot is None:
            if anchor_failure == "unverified_multi_object_group":
                return EntityTrackResult(
                    status="failed",
                    reason=anchor_failure,
                )
            if anchor_failure == "ambiguous_multi_object_instance":
                return EntityTrackResult(
                    status="failed",
                    reason=(
                        "SAM3 returned multiple ambiguous instances for a "
                        "single-entity prompt"
                    ),
                )
            return EntityTrackResult(
                status="not_found",
                reason="SAM3 did not find the prompted entity",
            )

        session_id = self._start_session(predictor, frames_dir)
        try:
            prompted = predictor.handle_request(  # type: ignore[attr-defined]
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor_slot,
                    "text": grounding_prompt,
                }
            )
            anchored = self._observations(
                int(prompted["frame_index"]),
                prompted["outputs"],
            )
            if reference_type == "group" and len(anchored) > 1:
                return EntityTrackResult(
                    status="failed",
                    reason="unverified_multi_object_group",
                )
            if len(anchored) != 1:
                return EntityTrackResult(
                    status="failed",
                    reason=(
                        "SAM3 did not resolve one stable tracked identity for "
                        "the entity"
                    ),
                )
            if not anchored:
                return EntityTrackResult(
                    status="not_found",
                    reason="SAM3 anchor disappeared before propagation",
                )
            tracked_ids = {item.object_id for item in anchored}
            observations: dict[
                tuple[int, str],
                BackendMaskObservation,
            ] = {
                (item.slot, item.object_id): item for item in anchored
            }
            for direction in ("forward", "backward"):
                for response in predictor.handle_stream_request(  # type: ignore[attr-defined]
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                    }
                ):
                    slot = int(response["frame_index"])
                    if not 0 <= slot < len(frame_paths):
                        raise ValueError(
                            "SAM3 returned a frame slot outside sampled frames"
                        )
                    current = self._observations(slot, response["outputs"])
                    unexpected_ids = {
                        item.object_id for item in current
                    } - tracked_ids
                    if unexpected_ids:
                        raise ValueError(
                            "SAM3 object identity changed during propagation"
                        )
                    for item in current:
                        observations.setdefault(
                            (item.slot, item.object_id),
                            item,
                        )
        finally:
            self._close_session(predictor, session_id)

        ordered = tuple(
            observations[key]
            for key in sorted(observations, key=lambda item: (item[0], item[1]))
        )
        return EntityTrackResult(
            status="ready",
            observations=ordered,
            group_tracks_verified=False,
        )

    def close(self) -> None:
        if self._predictor is None:
            return
        shutdown = getattr(self._predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()
