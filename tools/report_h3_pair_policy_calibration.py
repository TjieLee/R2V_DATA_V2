#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.pair_policy_calibration import (
    ThresholdSimulationPolicy,
    report_pair_policy_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report HUMAN-labeled H3 PairPolicy calibration evidence"
    )
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--confirmed-face-pairs", type=Path, required=True)
    parser.add_argument("--hard-negative-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--face-mining-root", type=Path)
    parser.add_argument("--simulate-minimum-face-cosine", type=float)
    parser.add_argument("--simulate-minimum-voice-cosine", type=float)
    parser.add_argument("--simulate-maximum-face-mutual-rank", type=int)
    parser.add_argument("--simulate-maximum-voice-mutual-rank", type=int)
    parser.add_argument("--simulate-minimum-face-margin", type=float)
    parser.add_argument("--simulate-minimum-voice-margin", type=float)
    args = parser.parse_args()
    simulation_values = (
        args.simulate_minimum_voice_cosine,
        args.simulate_maximum_face_mutual_rank,
        args.simulate_maximum_voice_mutual_rank,
        args.simulate_minimum_face_margin,
        args.simulate_minimum_voice_margin,
    )
    if args.simulate_minimum_face_cosine is None and any(
        value is not None for value in simulation_values
    ):
        parser.error(
            "--simulate-minimum-face-cosine is required for threshold simulation"
        )
    policy = (
        ThresholdSimulationPolicy(
            minimum_face_cosine=args.simulate_minimum_face_cosine,
            minimum_voice_cosine=args.simulate_minimum_voice_cosine,
            maximum_face_mutual_rank=args.simulate_maximum_face_mutual_rank,
            maximum_voice_mutual_rank=args.simulate_maximum_voice_mutual_rank,
            minimum_face_margin=args.simulate_minimum_face_margin,
            minimum_voice_margin=args.simulate_minimum_voice_margin,
        )
        if args.simulate_minimum_face_cosine is not None
        else None
    )
    report = report_pair_policy_calibration(
        embedding_root=args.embedding_root,
        confirmed_face_pairs=args.confirmed_face_pairs,
        hard_negative_labels=args.hard_negative_labels,
        output_root=args.output_root,
        simulation_policy=policy,
        face_mining_root=args.face_mining_root,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
