#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.face_label_planning import plan_audio_from_face_labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan expensive H3 Audio runs from HUMAN SAME face labels"
    )
    parser.add_argument("--face-mining-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-clips", type=int)
    args = parser.parse_args()
    plan = plan_audio_from_face_labels(
        face_mining_root=args.face_mining_root,
        labels_path=args.labels,
        output_root=args.output_root,
        max_clips=args.max_clips,
    )
    print(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
