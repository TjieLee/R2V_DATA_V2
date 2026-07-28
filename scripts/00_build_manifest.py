from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import load_config
from r2v_data_v2.manifest import build_manifest, stats_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the lightweight source manifest"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = build_manifest(
        load_config(args.config),
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats_dict(stats), indent=2))


if __name__ == "__main__":
    main()
