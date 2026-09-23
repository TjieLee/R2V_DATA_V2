"""Elastic group claiming for formal Post-Mask Resource Epoch production."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.post_mask_epoch_groups import (
    ResourceEpochGroup,
    group_ownership,
    group_root,
    record_group_outcome,
    write_group_descriptor,
)
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger, atomic_write_json


class IncompleteGroupError(RuntimeError):
    """A retryable group attempt stopped before its semantic/export barrier."""


def _marker_path(root: Path, group: ResourceEpochGroup) -> Path:
    return group_root(root, group) / "completed.json"


def production_shard_completed(paths: Any, clip_uids: Sequence[str]) -> int | None:
    """Return the published sample count without reading exports or receipts."""
    marker = Path(paths.state_root) / "completed.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid production shard marker: {marker}") from exc
    count = len(clip_uids)
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("shard_id") != Path(paths.state_root).name
        or value.get("eligible_rows") != count
        or value.get("terminal_rows") != count
        or value.get("export_completed") is not True
        or type(value.get("sample_count")) is not int
        or value["sample_count"] < 0
    ):
        raise ValueError(f"production shard marker differs from shard: {marker}")
    return value["sample_count"]


def write_production_shard_complete(
    paths: Any, clip_uids: Sequence[str], sample_count: int
) -> None:
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError("invalid production sample count")
    count = len(clip_uids)
    atomic_write_json(
        Path(paths.state_root) / "completed.json",
        {
            "status": "completed",
            "shard_id": Path(paths.state_root).name,
            "eligible_rows": count,
            "terminal_rows": count,
            "export_completed": True,
            "sample_count": sample_count,
        },
    )


def cleanup_export_backups(paths: Any) -> None:
    """Remove only this locked shard's exporter crash backups after republish."""
    destination = Path(paths.export_root)
    pattern = re.compile(re.escape(f".{destination.name}.backup-") + r"[0-9a-f]{32}")
    for candidate in destination.parent.glob(f".{destination.name}.backup-*"):
        if not pattern.fullmatch(candidate.name):
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"invalid Post-Mask export backup: {candidate}")
        shutil.rmtree(candidate)


def production_group_completed(root: Path, group: ResourceEpochGroup) -> bool:
    """Check only the small formal terminal marker, never receipts or media."""
    marker = _marker_path(root, group)
    if not marker.is_file():
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid production group marker: {marker}") from exc
    if (
        not isinstance(value, dict)
        or value.get("status") != "completed"
        or value.get("group_id") != group.group_id
        or value.get("canonical_shards") != list(group.canonical_shards)
        or value.get("export_completed") is not True
        or type(value.get("sample_count")) is not int
        or value["sample_count"] < 0
    ):
        raise ValueError(f"production group marker differs from group: {marker}")
    return True


def write_production_group_complete(
    root: Path, group: ResourceEpochGroup, outcome: dict[str, Any]
) -> Path:
    """Publish only after the existing semantic, attribute and export barriers."""
    stats = outcome.get("export_stats")
    if not (
        outcome.get("completed") is True
        and outcome.get("subject_attributes_completed") is True
        and outcome.get("export_completed") is True
        and isinstance(stats, dict)
        and set(stats) == set(group.canonical_shards)
    ):
        raise ValueError("production group has not completed every stage and export")
    counts = [stats[shard].get("sample_count") for shard in group.canonical_shards]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("production export counts are invalid")
    marker = _marker_path(root, group)
    if marker.exists():
        production_group_completed(root, group)
        return marker
    atomic_write_json(
        marker,
        {
            "status": "completed",
            "group_id": group.group_id,
            "canonical_shards": list(group.canonical_shards),
            "export_completed": True,
            "sample_count": sum(counts),
        },
    )
    return marker


def run_elastic_groups(
    root: Path,
    groups: Sequence[ResourceEpochGroup],
    runner: Callable[[ResourceEpochGroup, GroupLedger, Callable[..., None]], dict],
    *,
    rank: int,
    world_size: int,
    emit: Callable[..., None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    backoff_seconds: float = 2.0,
) -> dict[str, int]:
    """Every node scans all groups; rank only rotates scan order."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be nonnegative")
    root = Path(root)
    ordered = tuple(groups)
    if not ordered:
        raise ValueError("production has no groups")
    start = (rank * max(1, len(ordered) // world_size)) % len(ordered)
    ordered = ordered[start:] + ordered[:start]
    report = emit or (lambda *_args, **_kwargs: None)
    claimed = 0
    attempted_incomplete: dict[str, str] = {}
    while True:
        for group in ordered:
            if production_group_completed(root, group):
                continue
            if group.group_id in attempted_incomplete:
                continue
            with group_ownership(root, group) as held:
                if not held:
                    continue
                if production_group_completed(root, group):
                    continue
                write_group_descriptor(root, group)
                claimed += 1
                report("post_mask_production_group_started", group_id=group.group_id)
                outcome = runner(group, GroupLedger(group_root(root, group)), report)
                if not outcome or not outcome.get("completed"):
                    reason = str((outcome or {}).get("reason", "incomplete"))
                    record_group_outcome(
                        root, group, outcome="incomplete", reason=reason
                    )
                    attempted_incomplete[group.group_id] = reason
                    report(
                        "post_mask_production_group_incomplete",
                        group_id=group.group_id,
                        reason=reason,
                    )
                    continue
                write_production_group_complete(root, group, outcome)
                record_group_outcome(root, group, outcome="completed")
                report("post_mask_production_group_completed", group_id=group.group_id)
        pending = [
            group for group in ordered if not production_group_completed(root, group)
        ]
        if not pending:
            return {"group_count": len(ordered), "groups_claimed": claimed}
        if all(group.group_id in attempted_incomplete for group in pending):
            details = ", ".join(
                f"{group.group_id}: {attempted_incomplete[group.group_id]}"
                for group in pending
            )
            raise IncompleteGroupError(details)
        sleep(backoff_seconds)
