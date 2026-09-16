from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.training_manifest_export import export_training_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Flatten frozen H3 products into dataloader-ready task JSONL files."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ra2va-shadow-root", type=Path)
    parser.add_argument("--t2va-root", type=Path)
    parser.add_argument("--ta2va-root", type=Path)
    args = parser.parse_args(argv)
    return export_training_manifests(
        output_root=args.output_root,
        ra2va_shadow_root=args.ra2va_shadow_root,
        t2va_root=args.t2va_root,
        ta2va_root=args.ta2va_root,
    )


if __name__ == "__main__":
    main()
