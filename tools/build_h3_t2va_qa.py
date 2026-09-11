from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.t2va_qa import build_t2va_qa


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Build independent static T2VA QA (no model calls)"
    )
    parser.add_argument("--t2va-root", required=True, type=Path)
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Shared served root containing original video and Audio lineage, e.g. /mnt/workspace",
    )
    parser.add_argument(
        "--media-base-url",
        help="HTTP origin serving media-root, e.g. http://127.0.0.1:8765",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return build_t2va_qa(
        args.t2va_root,
        media_root=args.media_root,
        media_base_url=args.media_base_url,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    print(main())
