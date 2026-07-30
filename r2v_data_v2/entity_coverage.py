from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def temporal_visibility_metrics(
    slots: Mapping[object, object],
    *,
    sampled_frame_count: int,
    minimum_entity_visible_ratio: float,
) -> dict[str, object]:
    if sampled_frame_count < 1:
        raise ValueError("sampled_frame_count must be positive")
    if not 0.0 < minimum_entity_visible_ratio <= 1.0:
        raise ValueError(
            "minimum_entity_visible_ratio must be greater than 0 and at most 1"
        )
    visible_frame_count = sum(
        isinstance(value, Mapping)
        and value.get("mask_available") is True
        for value in slots.values()
    )
    required_visible_frames = math.ceil(
        sampled_frame_count * minimum_entity_visible_ratio
    )
    return {
        "sampled_frame_count": sampled_frame_count,
        "visible_frame_count": visible_frame_count,
        "visible_frame_ratio": visible_frame_count / sampled_frame_count,
        "required_visible_frames": required_visible_frames,
        "minimum_entity_visible_ratio": minimum_entity_visible_ratio,
        "temporal_coverage_passed": (
            visible_frame_count >= required_visible_frames
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _entity_value(entity: object, field: str, default: object = None) -> object:
    if isinstance(entity, Mapping):
        return entity.get(field, default)
    return getattr(entity, field, default)


def _strict_candidate_count(candidate_dir: Path) -> int:
    status = _read_json_object(candidate_dir / "candidate_status.json")
    if status is not None:
        value = status.get("candidate_count")
        if isinstance(value, int) and value >= 0:
            return value
    path = candidate_dir / "candidates.jsonl"
    if not path.is_file():
        return 0
    try:
        return sum(
            bool(line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return 0


def _has_ready_canonical_reference(
    output_root: Path,
    *,
    clip_uid: str,
    entity_id: str,
) -> bool:
    metadata = _read_json_object(
        output_root / "references" / clip_uid / entity_id / "metadata.json"
    )
    if metadata is None:
        return False
    if metadata.get("status", "ready") != "ready":
        return False
    if metadata.get("rejected") is True:
        return False
    canonical_path = metadata.get("canonical_path")
    return bool(canonical_path) and Path(str(canonical_path)).is_file()


def build_clip_entity_coverage(
    *,
    output_root: Path,
    clip_uid: str,
    entities: Sequence[object],
    sampled_frame_count: int,
    minimum_entity_visible_ratio: float,
) -> dict[str, object]:
    summaries: dict[str, dict[str, object]] = {}
    qualifying_entity_ids: list[str] = []
    required_visible_frames = math.ceil(
        sampled_frame_count * minimum_entity_visible_ratio
    )
    for entity in entities:
        entity_id = str(_entity_value(entity, "entity_id") or "")
        if not entity_id:
            continue
        candidate_dir = output_root / "candidates" / clip_uid / entity_id
        coverage = _read_json_object(candidate_dir / "mask_coverage.json")
        slots = coverage.get("slots", {}) if coverage is not None else {}
        if not isinstance(slots, dict):
            slots = {}
        visibility = temporal_visibility_metrics(
            slots,
            sampled_frame_count=sampled_frame_count,
            minimum_entity_visible_ratio=minimum_entity_visible_ratio,
        )
        strict_candidate_count = _strict_candidate_count(candidate_dir)
        ready_canonical = _has_ready_canonical_reference(
            output_root,
            clip_uid=clip_uid,
            entity_id=entity_id,
        )
        reference_worthy = bool(
            _entity_value(entity, "reference_worthy", False)
        )
        qualifies = (
            reference_worthy
            and visibility["temporal_coverage_passed"] is True
            and strict_candidate_count >= 1
            and ready_canonical
        )
        summary = {
            **visibility,
            "reference_worthy": reference_worthy,
            "strict_candidate_count": strict_candidate_count,
            "has_ready_canonical_reference": ready_canonical,
            "qualifies": qualifies,
        }
        summaries[entity_id] = summary
        if qualifies:
            qualifying_entity_ids.append(entity_id)
    return {
        "clip_uid": clip_uid,
        "qualifying_entity_ids": qualifying_entity_ids,
        "required_visible_frames": required_visible_frames,
        "entity_visibility_summary": summaries,
        "entity_coverage_passed": bool(qualifying_entity_ids),
    }


def read_clip_entity_coverage(
    output_root: Path,
    clip_uid: str,
) -> bool | None:
    path = output_root / "candidates" / clip_uid / "entity_coverage.json"
    if not path.exists():
        return None
    payload = _read_json_object(path)
    if payload is None:
        return False
    return payload.get("entity_coverage_passed") is True


def read_clip_qualifying_entity_ids(
    output_root: Path,
    clip_uid: str,
) -> set[str] | None:
    path = output_root / "candidates" / clip_uid / "entity_coverage.json"
    if not path.exists():
        return None
    payload = _read_json_object(path)
    if payload is None or payload.get("entity_coverage_passed") is not True:
        return set()
    values = payload.get("qualifying_entity_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        return set()
    return set(values)
