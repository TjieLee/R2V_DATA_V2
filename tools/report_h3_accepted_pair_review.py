#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.pairing_pilot import report_accepted_pair_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report HUMAN review of accepted H3 PairPolicy V1 donors"
    )
    parser.add_argument("--pairing-pilot-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_accepted_pair_review(
        pairing_pilot_root=args.pairing_pilot_root,
        labels_path=args.labels,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
