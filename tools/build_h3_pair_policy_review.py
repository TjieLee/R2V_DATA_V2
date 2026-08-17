#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.pair_policy_calibration import build_pair_policy_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a HUMAN hard-negative PairPolicy calibration review"
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--confirmed-face-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--face-mining-root", type=Path)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_pair_policy_review(
        embedding_root=args.embedding_root,
        confirmed_face_pairs=args.confirmed_face_pairs,
        output_root=args.output_root,
        top=args.top,
        face_mining_root=args.face_mining_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
