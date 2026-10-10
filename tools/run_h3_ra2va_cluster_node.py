"""One local RA2VA node; Workers discover the designated Coordinator."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2v_data_v2.h3.ra2va_group_launch import launch_node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake", action="store_true", help="CPU-only test; no models or training products")
    args = parser.parse_args()
    try:
        return launch_node(ROOT, os.environ, dry_run=args.dry_run, fake=args.fake)
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
