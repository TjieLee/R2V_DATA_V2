"""Bounded RA2VA group CPU scheduling pilot. No model or real export runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.ra2va_group_fake import (
    OUTPUT_ROOT,
    SOURCE_ROOT,
    pilot_run_root,
    pilot_summary,
    run_fake_pilot,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
        command.add_argument("--run-id", required=True)
        if name == "run":
            command.add_argument("--fake", action="store_true", required=True)
            command.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
            command.add_argument("--limit", type=int, required=True)
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--max-groups", type=int, default=1)
            command.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()
    root = pilot_run_root(args.output_root, args.run_id)
    if args.command == "status":
        result = pilot_summary(root)
    else:
        result = run_fake_pilot(source_root=args.source_root, run_root=root,
                                workers=args.workers, limit=args.limit,
                                max_groups=args.max_groups, poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
