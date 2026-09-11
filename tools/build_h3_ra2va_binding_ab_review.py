#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2v_data_v2.h3.ra2va_binding_ab_review import build_review


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build read-only RA2VA speaker-binding A/B review")
    for name in ("visual-production-root", "visual-runs-root", "case-manifest", "legacy-mimo-root", "no-lrasd-mimo-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--legacy-source-contract", type=Path, help="Exact frozen OLD source contract or job inventory; never resampled")
    result = build_review(**vars(parser.parse_args(argv)))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
