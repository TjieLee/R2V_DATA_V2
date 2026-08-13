from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from r2v_data_v2.v3.config import Sam3Config

TrackStatus = Literal["ready", "not_found", "failed"]
PropagationDirection = Literal["forward", "backward"]
_ANCHOR_FAST_PROBE_ORDER = (5, 2, 7, 0, 9)
_ANCHOR_FALLBACK_PROBE_ORDER = (4, 6, 3, 8, 1)
_MINIMUM_MATCHING_MASK_IOU = 0.95


@dataclass(frozen=True)
class BackendMaskObservation:
    """A SAM3 mask plus its propagated object score diagnostic.

    ``confidence`` retains the existing internal and artifact-facing name for
    compatibility. This field stores the SAM3 object score propagated with the
    track. It is not an independently estimated per-frame tracking confidence.
    """

    slot: int
    mask: np.ndarray
    confidence: float
    object_id: str
    valid: bool = True

    @property
    def object_score(self) -> float:
        return self.confidence


@dataclass(frozen=True)
class DirectionalTrackResult:
    direction: PropagationDirection
    anchor: BackendMaskObservation
    observations: tuple[BackendMaskObservation, ...]
    non_owned_observations: tuple[BackendMaskObservation, ...] = ()
    identity_switch_detected: bool = False


@dataclass(frozen=True)
class MultiInstanceAnchorDecision:
    verdict: Literal["select", "reject", "uncertain"]
    candidate_id: int | None
    reason: str


class MultiInstanceAnchorSelector(Protocol):
    def select(
        self,
        *,
        frame_path: Path,
        candidates: tuple[BackendMaskObservation, ...],
        entity_phrase: str,
        grounding_prompt: str,
        reference_type: str,
    ) -> MultiInstanceAnchorDecision: ...


@dataclass(frozen=True)
class _AnchorProbeSelection:
    slot: int
    observation: BackendMaskObservation
    multi_instance_rescued: bool = False


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


class _TrackValidationError(ValueError):
    pass


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("mask IoU requires two-dimensional masks")
    if first.shape != second.shape:
        raise ValueError("mask IoU requires equal mask shapes")
    union = np.logical_or(first, second)
    union_area = int(union.sum())
    if union_area == 0:
        return 0.0
    value = float(np.logical_and(first, second).sum() / union_area)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("mask IoU must be finite and between zero and one")
    return value


def _masks_match(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    try:
        return mask_iou(mask_a, mask_b) >= _MINIMUM_MATCHING_MASK_IOU
    except ValueError:
        return False


def _validate_anchor_consistency(
    forward: DirectionalTrackResult,
    backward: DirectionalTrackResult,
) -> None:
    if (
        forward.anchor.slot != backward.anchor.slot
        or not forward.anchor.mask.any()
        or not backward.anchor.mask.any()
        or not _masks_match(forward.anchor.mask, backward.anchor.mask)
    ):
        raise _TrackValidationError("anchor_identity_mismatch_between_directions")


def _remap_direction_object_id(
    observations: tuple[BackendMaskObservation, ...],
    canonical_object_id: str,
) -> tuple[BackendMaskObservation, ...]:
    return tuple(
        replace(observation, object_id=canonical_object_id)
        for observation in observations
    )


def _merge_directional_tracks(
    forward: DirectionalTrackResult,
    backward: DirectionalTrackResult,
) -> tuple[BackendMaskObservation, ...]:
    canonical_object_id = forward.anchor.object_id
    propagation_observations = (
        *forward.observations,
        *forward.non_owned_observations,
        *backward.observations,
        *backward.non_owned_observations,
    )
    propagation_by_key: dict[tuple[int, str], BackendMaskObservation] = {}
    for observation in _remap_direction_object_id(
        propagation_observations,
        canonical_object_id,
    ):
        key = (observation.slot, observation.object_id)
        existing = propagation_by_key.get(key)
        if existing is None:
            propagation_by_key[key] = observation
            continue
        if not _masks_match(existing.mask, observation.mask):
            raise _TrackValidationError("conflicting_bidirectional_mask")

    merged: dict[tuple[int, str], BackendMaskObservation] = {}
    sources = (
        (replace(forward.anchor, object_id=canonical_object_id),),
        _remap_direction_object_id(
            forward.observations,
            canonical_object_id,
        ),
        _remap_direction_object_id(
            backward.observations,
            canonical_object_id,
        ),
    )
    for source in sources:
        for observation in source:
            key = (observation.slot, observation.object_id)
            existing = merged.get(key)
            if existing is None:
                merged[key] = observation
                continue
            if not _masks_match(existing.mask, observation.mask):
                raise _TrackValidationError("conflicting_bidirectional_mask")
    return tuple(merged[key] for key in sorted(merged))


class SegmentationBackend(Protocol):
    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
        entity_phrase: str | None = None,
    ) -> EntityTrackResult: ...


class Sam3SegmentationBackend:
    """Lazy adapter for the predictor API already exercised by the V2 pipeline."""

    def __init__(
        self,
        config: Sam3Config,
        *,
        predictor: object | None = None,
        builder: Callable[..., object] | None = None,
        anchor_selector: MultiInstanceAnchorSelector | None = None,
    ) -> None:
        self.config = config
        self._predictor = predictor
        self._builder = builder
        self._anchor_selector = anchor_selector
        self._anchor_counters = {
            "anchor_fast_path_hits": 0,
            "anchor_fallback_attempted": 0,
            "anchor_fallback_hits": 0,
            "anchor_all_frames_not_found": 0,
            "anchor_probe_calls": 0,
        }
        self._recall_rescue_counters = {
            "multi_instance_rescue_attempted": 0,
            "multi_instance_rescue_selected": 0,
            "multi_instance_rescue_rejected": 0,
            "propagation_identity_switch_detected": 0,
            "partial_track_salvage_attempted": 0,
            "partial_track_salvage_ready": 0,
            "partial_track_salvage_insufficient": 0,
        }

    def anchor_search_counters(self) -> dict[str, int]:
        return dict(self._anchor_counters)

    def recall_rescue_counters(self) -> dict[str, int]:
        return dict(self._recall_rescue_counters)

    def _load_predictor(self) -> object:
        if self._predictor is not None:
            return self._predictor
        if self.config.model_path is None:
            raise ValueError("sam3.model_path must be configured before segment runs")
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

        object_scores = np.asarray(outputs.get("out_probs", [])).reshape(-1)
        object_ids = np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)
        if len(object_scores) != len(raw_masks):
            raise ValueError("SAM3 object score count does not match mask count")
        if len(object_ids) != len(raw_masks):
            raise ValueError("SAM3 object ID count does not match mask count")

        observations: list[BackendMaskObservation] = []
        for index, raw_mask in enumerate(raw_masks):
            mask = np.asarray(raw_mask, dtype=bool)
            if not mask.any():
                continue
            object_score = float(object_scores[index])
            if not math.isfinite(object_score):
                raise ValueError("SAM3 returned a non-finite object score")
            observations.append(
                BackendMaskObservation(
                    slot=slot,
                    mask=mask.copy(),
                    confidence=object_score,
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
        frame_paths: list[Path],
        reference_type: str,
        entity_phrase: str,
        grounding_prompt: str,
    ) -> tuple[_AnchorProbeSelection | None, str | None]:
        ambiguous_instance = False

        def probe(order: tuple[int, ...]) -> _AnchorProbeSelection | None:
            nonlocal ambiguous_instance
            for slot in order:
                if slot >= len(frame_paths):
                    continue
                self._anchor_counters["anchor_probe_calls"] += 1
                observations = self._prompt_frame(
                    predictor,
                    frames_dir=frames_dir,
                    slot=slot,
                    grounding_prompt=grounding_prompt,
                )
                if len(observations) > 1:
                    if reference_type == "group":
                        raise _TrackValidationError("unverified_multi_object_group")
                    ambiguous_instance = True
                    if (
                        self.config.multi_instance_rescue_mode
                        != "qwen_anchor_select_v1"
                    ):
                        continue
                    self._recall_rescue_counters[
                        "multi_instance_rescue_attempted"
                    ] += 1
                    if self._anchor_selector is None:
                        self._recall_rescue_counters[
                            "multi_instance_rescue_rejected"
                        ] += 1
                        continue
                    try:
                        decision = self._anchor_selector.select(
                            frame_path=frame_paths[slot],
                            candidates=tuple(observations),
                            entity_phrase=entity_phrase,
                            grounding_prompt=grounding_prompt,
                            reference_type=reference_type,
                        )
                    except Exception:  # noqa: BLE001 - judge failure skips this probe
                        self._recall_rescue_counters[
                            "multi_instance_rescue_rejected"
                        ] += 1
                        continue
                    candidate_id = decision.candidate_id
                    if decision.verdict != "select":
                        self._recall_rescue_counters[
                            "multi_instance_rescue_rejected"
                        ] += 1
                        continue
                    if (
                        candidate_id is None
                        or candidate_id < 1
                        or candidate_id > len(observations)
                    ):
                        self._recall_rescue_counters[
                            "multi_instance_rescue_rejected"
                        ] += 1
                        continue
                    self._recall_rescue_counters[
                        "multi_instance_rescue_selected"
                    ] += 1
                    return _AnchorProbeSelection(
                        slot=slot,
                        observation=observations[candidate_id - 1],
                        multi_instance_rescued=True,
                    )
                if observations:
                    return _AnchorProbeSelection(
                        slot=slot,
                        observation=observations[0],
                    )
            return None

        try:
            anchor = probe(_ANCHOR_FAST_PROBE_ORDER)
        except _TrackValidationError as exc:
            return None, str(exc)
        if anchor is not None:
            self._anchor_counters["anchor_fast_path_hits"] += 1
            return anchor, None
        if self.config.anchor_search_mode == "progressive_v1":
            self._anchor_counters["anchor_fallback_attempted"] += 1
            try:
                anchor = probe(_ANCHOR_FALLBACK_PROBE_ORDER)
            except _TrackValidationError as exc:
                return None, str(exc)
            if anchor is not None:
                self._anchor_counters["anchor_fallback_hits"] += 1
                return anchor, None
        if ambiguous_instance:
            return None, (
                "unverified_multi_object_group"
                if reference_type == "group"
                else "ambiguous_multi_object_instance"
            )
        if self.config.anchor_search_mode == "progressive_v1":
            self._anchor_counters["anchor_all_frames_not_found"] += 1
        return None, None

    def _run_direction(
        self,
        predictor: object,
        *,
        frames_dir: Path,
        frame_count: int,
        anchor_selection: _AnchorProbeSelection,
        reference_type: str,
        grounding_prompt: str,
        direction: PropagationDirection,
    ) -> DirectionalTrackResult:
        anchor_slot = anchor_selection.slot
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
                raise _TrackValidationError("unverified_multi_object_group")
            ignored_object_ids: set[str] = set()
            if anchor_selection.multi_instance_rescued:
                matching = [
                    observation
                    for observation in anchored
                    if _masks_match(
                        observation.mask,
                        anchor_selection.observation.mask,
                    )
                ]
                if len(matching) != 1:
                    raise _TrackValidationError(
                        "selected_anchor_identity_not_reidentified"
                    )
                anchor = matching[0]
                ignored_object_ids = {
                    observation.object_id
                    for observation in anchored
                    if observation.object_id != anchor.object_id
                }
            else:
                if len(anchored) != 1:
                    raise _TrackValidationError(
                        "SAM3 did not resolve one stable tracked identity for the entity"
                    )
                anchor = anchored[0]
            observations: list[BackendMaskObservation] = []
            non_owned_observations: list[BackendMaskObservation] = []
            identity_switch_detected = False
            for response in predictor.handle_stream_request(  # type: ignore[attr-defined]
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": direction,
                }
            ):
                slot = int(response["frame_index"])
                if not 0 <= slot < frame_count:
                    raise ValueError(
                        "SAM3 returned a frame slot outside sampled frames"
                    )
                if slot == anchor_slot:
                    continue
                current = self._observations(slot, response["outputs"])
                unknown = [
                    observation
                    for observation in current
                    if observation.object_id != anchor.object_id
                    and observation.object_id not in ignored_object_ids
                ]
                if unknown:
                    identity_switch_detected = True
                    invalid_slots = (
                        range(slot, frame_count)
                        if direction == "forward"
                        else range(slot + 1)
                    )
                    observations.extend(
                        BackendMaskObservation(
                            slot=invalid_slot,
                            mask=np.zeros_like(anchor.mask, dtype=bool),
                            confidence=0.0,
                            object_id=anchor.object_id,
                            valid=False,
                        )
                        for invalid_slot in invalid_slots
                    )
                    break
                current = [
                    observation
                    for observation in current
                    if observation.object_id == anchor.object_id
                ]
                owns_slot = (direction == "forward" and slot > anchor_slot) or (
                    direction == "backward" and slot < anchor_slot
                )
                # Retain anomalous cross-direction responses only to validate
                # duplicate masks; they are never published by the track.
                destination = observations if owns_slot else non_owned_observations
                destination.extend(current)
            return DirectionalTrackResult(
                direction=direction,
                anchor=anchor,
                observations=tuple(observations),
                non_owned_observations=tuple(non_owned_observations),
                identity_switch_detected=identity_switch_detected,
            )
        finally:
            self._close_session(predictor, session_id)

    def track(
        self,
        *,
        frame_paths: list[Path],
        entity_id: str,
        reference_type: str,
        grounding_prompt: str,
        entity_phrase: str | None = None,
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
        anchor_selection, anchor_failure = self._find_anchor(
            predictor,
            frames_dir=frames_dir,
            frame_paths=frame_paths,
            reference_type=reference_type,
            entity_phrase=entity_phrase or grounding_prompt,
            grounding_prompt=grounding_prompt,
        )
        if anchor_selection is None:
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
            if anchor_failure is not None:
                return EntityTrackResult(status="failed", reason=anchor_failure)
            return EntityTrackResult(
                status="not_found",
                reason="SAM3 did not find the prompted entity",
            )

        switch_count = 0
        try:
            forward = self._run_direction(
                predictor,
                frames_dir=frames_dir,
                frame_count=len(frame_paths),
                anchor_selection=anchor_selection,
                reference_type=reference_type,
                grounding_prompt=grounding_prompt,
                direction="forward",
            )
            backward = self._run_direction(
                predictor,
                frames_dir=frames_dir,
                frame_count=len(frame_paths),
                anchor_selection=anchor_selection,
                reference_type=reference_type,
                grounding_prompt=grounding_prompt,
                direction="backward",
            )
            switch_count = sum(
                (
                    forward.identity_switch_detected,
                    backward.identity_switch_detected,
                )
            )
            if switch_count:
                self._recall_rescue_counters[
                    "propagation_identity_switch_detected"
                ] += switch_count
                self._recall_rescue_counters[
                    "partial_track_salvage_attempted"
                ] += 1
            _validate_anchor_consistency(forward, backward)
            # SAM3 object IDs are session-local. The backward session ID is
            # remapped to the forward anchor ID after anchor-mask validation.
            observations = _merge_directional_tracks(forward, backward)
        except _TrackValidationError as exc:
            if switch_count:
                self._recall_rescue_counters[
                    "partial_track_salvage_insufficient"
                ] += 1
            return EntityTrackResult(status="failed", reason=str(exc))
        valid_observations = [
            observation
            for observation in observations
            if observation.valid and np.asarray(observation.mask, dtype=bool).any()
        ]
        if switch_count and len(valid_observations) <= 1:
            self._recall_rescue_counters[
                "partial_track_salvage_insufficient"
            ] += 1
            changed_directions = [
                result.direction
                for result in (forward, backward)
                if result.identity_switch_detected
            ]
            changed_label = (
                changed_directions[0]
                if len(changed_directions) == 1
                else "bidirectional"
            )
            return EntityTrackResult(
                status="failed",
                reason=(
                    "sam3_object_identity_changed_during_"
                    f"{changed_label}_propagation"
                ),
            )
        return EntityTrackResult(
            status="ready",
            observations=observations,
            group_tracks_verified=False,
        )

    def close(self) -> None:
        selector_close = getattr(self._anchor_selector, "close", None)
        if callable(selector_close):
            selector_close()
        if self._predictor is None:
            return
        shutdown = getattr(self._predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()
