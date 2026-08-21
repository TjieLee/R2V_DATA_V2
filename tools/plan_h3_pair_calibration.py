#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.pair_calibration import plan_h3_pair_calibration


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a bounded read-only H3 pair-calibration Visual clip set"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-audio-pilot-root", type=Path)
    parser.add_argument("--max-clips", type=_positive_integer, default=60)
    parser.add_argument("--max-clips-per-parent", type=_positive_integer, default=4)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    plan = plan_h3_pair_calibration(
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        seed_audio_pilot_root=arguments.seed_audio_pilot_root,
        max_clips=arguments.max_clips,
        max_clips_per_parent=arguments.max_clips_per_parent,
    )
    result = plan.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
