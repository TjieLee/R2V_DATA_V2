#!/usr/bin/env python3
"""Build a new CPU-only task bundle from frozen single-call reconcile."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.ra2va_training_bundle import materialize_ra2va_training_bundle


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconcile-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-uid", dest="clip_uids", action="append")
    parser.add_argument("--source-h3-root", type=Path, help="Matching canonical H3 metadata; default: Audio root/h3")
    parser.add_argument("--allow-unverified", action="store_true", help="Explicit pilot opt-in for unverified resolved stems")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    result = materialize_ra2va_training_bundle(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
