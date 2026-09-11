from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.ta2va_qa import build_ta2va_qa


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build static target-only TA2VA QA")
    parser.add_argument("--ta2va-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--media-base-url")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return build_ta2va_qa(
        args.ta2va_root,
        media_root=args.media_root,
        media_base_url=args.media_base_url,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    print(main())
