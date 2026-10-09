#!/usr/bin/env python3
"""One-shot, read-only SA group-priority snapshot for elastic Post-Mask restart.

Run ONCE on one node before restarting workers. Only named SA outcome leaves
are listed; there is no recursive walk, image access, receipt scan or hash.
Each node reads the same small snapshot; nothing here changes durable state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_STATE = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/"
    "in_pair_reference/state/resource_epochs"
)


def _json_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    with os.scandir(directory) as entries:
        return sum(entry.name.endswith(".json") for entry in entries)


def build_sa_group_priority(state_root: Path) -> dict:
    records = []
    # Only 48 group directory entries, not a filesystem-wide scan.
    for group in sorted(state_root.glob("group-*")):
        if not group.is_dir() or (group / "completed.json").is_file():
            continue
        composition = group / "composition"
        done_marker = composition / "subject_attributes_completed.json"
        started_marker = composition / "subject_attributes_started.json"
        if done_marker.is_file():
            status = "sa_done"
            completed, total = 1, 1  # an authoritative completed handoff
            priority = 2
        elif started_marker.is_file():
            handoff = json.loads(started_marker.read_text(encoding="utf-8"))
            eligible = handoff["eligible_clip_uids_by_shard"]
            total = sum(len(uids) for uids in eligible.values())
            completed = sum(
                _json_count(group / "semantic" / "subject_attributes" /
                            "outcomes" / shard)
                for shard in eligible
            )
            priority = 1
            status = "sa_started"
        else:
            completed, total, priority = 0, 0, 0
            status = "not_in_sa"
        fraction = min(completed, total) / total if total else (
            1.0 if priority else 0.0
        )
        records.append({
            "group_id": group.name,
            "status": status,
            "terminal_clips": completed,
            "eligible_clips": total,
            "completion_fraction": fraction,
            "_priority": priority,
        })
    records.sort(key=lambda item: (
        -item["_priority"], -item["completion_fraction"],
        -item["terminal_clips"], item["group_id"],
    ))
    for item in records:
        del item["_priority"]
    return {
        "schema": "post_mask_sa_priority_snapshot_v1",
        "ordered_group_ids": [item["group_id"] for item in records],
        "groups": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_sa_group_priority(args.state_root)
    if not payload["ordered_group_ids"]:
        raise RuntimeError("No pending groups found: check --state-root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(f"Priority snapshot: {args.output}")
    for item in payload["groups"]:
        print(
            f"{item['group_id']} {item['status']} "
            f"{item['terminal_clips']}/{item['eligible_clips']} "
            f"({item['completion_fraction']:.1%})"
        )


if __name__ == "__main__":
    main()
