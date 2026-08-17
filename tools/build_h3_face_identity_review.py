#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from r2v_data_v2.h3.face_identity_review import build_face_identity_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a local HUMAN face-identity review page"
    )
    parser.add_argument("--face-mining-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_face_identity_review(args.face_mining_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
