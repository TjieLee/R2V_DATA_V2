#!/usr/bin/env python3
"""Publish an immutable ready-only view of durable RA2VA MiMo annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.ra2va_production import publish_ready_reconcile_view


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", required=True)
    args = parser.parse_args(argv)
    shard_ids = [int(value) for value in args.shards.split(",")]
    destination = publish_ready_reconcile_view(args.production_root, args.output_root, shard_ids)
    with (destination / "records.jsonl").open(encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    result = {"output_root": str(destination), "ready_count": count, "model_called": False}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
