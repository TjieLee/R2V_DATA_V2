#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect H3 embedding pilot rankings")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def _print_pairs(
    title: str,
    rows: list[dict[str, Any]],
    score_key: str,
    top: int,
) -> None:
    print(f"\n{title}")
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[score_key]),
            str(row["left_occurrence_id"]),
            str(row["right_occurrence_id"]),
        ),
    )
    for row in ordered[:top]:
        print(
            "\t".join(
                (
                    str(row["left_occurrence_id"]),
                    str(row["right_occurrence_id"]),
                    f"{float(row[score_key]):.6f}",
                    f"same_clip={str(bool(row['same_clip'])).lower()}",
                )
            )
        )


def main() -> int:
    args = _parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")
    root = args.embedding_root.expanduser().resolve(strict=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    _print_pairs(
        "FACE TOP PAIRS",
        _load_jsonl(root / "face_similarity.jsonl"),
        "face_similarity",
        args.top,
    )
    _print_pairs(
        "VOICE TOP PAIRS",
        _load_jsonl(root / "voice_similarity.jsonl"),
        "voice_similarity",
        args.top,
    )
    print("\nJOINT CROSS-CLIP CANDIDATES")
    joint = _load_jsonl(root / "joint_candidates.jsonl")
    joint.sort(
        key=lambda row: (
            int(row["face_anchor_to_candidate_rank"])
            + int(row["voice_anchor_to_candidate_rank"]),
            str(row["anchor_occurrence_id"]),
            str(row["candidate_occurrence_id"]),
        )
    )
    for row in joint[: args.top]:
        print(
            "\t".join(
                (
                    str(row["anchor_occurrence_id"]),
                    str(row["candidate_occurrence_id"]),
                    f"face={float(row['face_similarity']):.6f}",
                    (
                        "face_ranks="
                        f"{row['face_anchor_to_candidate_rank']}/"
                        f"{row['face_candidate_to_anchor_rank']}"
                    ),
                    f"voice={float(row['voice_similarity']):.6f}",
                    (
                        "voice_ranks="
                        f"{row['voice_anchor_to_candidate_rank']}/"
                        f"{row['voice_candidate_to_anchor_rank']}"
                    ),
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
