"""Read-only execution hints; never checkpoint or completion authority."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.post_mask_epoch_groups import ResourceEpochGroup, group_root

PRIORITY_STAGES = (
    "subject_attributes_completed",
    "subject_attributes_started",
    "instruct_started",
    "reference_integrity_started",
    "reference_edit_started",
    "pair_started",
)


def load_priority_hints(path: Path | None) -> dict[str, float]:
    """Invalid or unavailable advisory hints fall back to marker ordering."""
    if path is None:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    ratios = {}
    for group_id, counts in payload.items():
        if not isinstance(counts, dict):
            continue
        terminal, eligible = counts.get("terminal"), counts.get("eligible")
        if (
            type(terminal) is int
            and type(eligible) is int
            and 0 <= terminal <= eligible
            and eligible > 0
        ):
            ratios[group_id] = terminal / eligible
    return ratios


def group_priority_key(
    root: Path, group: ResourceEpochGroup, hints: dict[str, float]
) -> tuple[int, float]:
    """Only stat a bounded number of markers; authority stays in the runner."""
    directory = group_root(root, group)
    if (directory / "completed.json").is_file():
        return -1, 0.0  # The scheduler retains its formal marker validation/skip.
    for priority, stage in enumerate(PRIORITY_STAGES):
        if (directory / "composition" / f"{stage}.json").is_file():
            return priority, -hints.get(group.group_id, -1.0) if priority == 1 else 0.0
    return (
        len(PRIORITY_STAGES)
        if (directory / "group.json").is_file()
        else len(PRIORITY_STAGES) + 1
    ), 0.0


def create_priority_hints(state_root: Path) -> dict[str, dict[str, int]]:
    """One-shot direct outcome-directory counts, without opening outcomes.

    Invoke once while all nodes are stopped. Only the SA-started marker's
    explicit eligible shard/UID scope is scanned. Results are advisory counts.
    """
    epochs = Path(state_root) / "resource_epochs"
    result = {}
    if not epochs.is_dir():
        return result
    with os.scandir(epochs) as entries:
        directories = sorted(
            (
                Path(entry.path)
                for entry in entries
                if re.fullmatch(r"group-\d{6}", entry.name)
                and entry.is_dir(follow_symlinks=False)
            ),
            key=lambda path: path.name,
        )
    for directory in directories:
        if (directory / "completed.json").is_file():
            continue
        marker = directory / "composition/subject_attributes_started.json"
        if not marker.is_file():
            continue
        payload: Any = json.loads(marker.read_text(encoding="utf-8"))
        scope = (
            payload.get("eligible_clip_uids_by_shard")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(scope, dict):
            raise TypeError(f"missing eligible scope: {marker}")
        terminal = eligible = 0
        for shard, uids in scope.items():
            if (
                not isinstance(shard, str)
                or Path(shard).name != shard
                or shard in {".", ".."}
                or not isinstance(uids, list)
                or any(not isinstance(uid, str) or not uid for uid in uids)
                or len(uids) != len(set(uids))
            ):
                raise ValueError(f"invalid eligible scope: {marker}")
            eligible += len(uids)
            allowed = {f"{uid}.json" for uid in uids}
            outcomes = directory / "semantic/subject_attributes/outcomes" / shard
            if outcomes.is_symlink():
                raise ValueError(f"symlink outcome directory: {outcomes}")
            try:
                with os.scandir(outcomes) as entries:
                    terminal += sum(
                        entry.name in allowed and entry.is_file(follow_symlinks=False)
                        for entry in entries
                    )
            except FileNotFoundError:
                pass
        result[directory.name] = {"terminal": terminal, "eligible": eligible}
    return result
