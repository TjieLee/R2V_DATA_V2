#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.pairing_pilot import build_pairing_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only H3 PairPolicy V1 in/cross-pair pilot"
    )
    parser.add_argument("--audio-pilot-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_pairing_pilot(
        audio_pilot_root=args.audio_pilot_root,
        embedding_root=args.embedding_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
