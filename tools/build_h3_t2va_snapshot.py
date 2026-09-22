"""Publish an immutable snapshot of currently ready target-side training rows."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.t2va_production import build_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args(argv)
    return build_snapshot(args.production_root, args.snapshot_id)


if __name__ == "__main__":
    print(main())
