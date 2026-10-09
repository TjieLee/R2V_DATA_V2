"""Local CPU fixtures through real Group media/H3 consumers. No model requests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.ra2va_group_fake import pilot_run_root
from r2v_data_v2.h3.ra2va_group_local import run_cpu_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu-fixtures", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    result = run_cpu_pipeline(source_root=args.source_root,
        run_root=pilot_run_root(args.output_root, args.run_id), fixtures_path=args.cpu_fixtures,
        workers=args.workers, limit=args.limit, max_groups=args.max_groups, ffmpeg=args.ffmpeg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
