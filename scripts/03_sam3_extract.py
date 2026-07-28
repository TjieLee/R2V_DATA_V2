from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import load_config
from r2v_data_v2.sam3_backend import extract_manifest_candidates, stats_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract entity candidates from the eight sampled frames"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sam3-checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sam3_checkpoint:
        from dataclasses import replace

        config = replace(
            config,
            sam3=replace(config.sam3, checkpoint=Path(args.sam3_checkpoint)),
        )
    stats = extract_manifest_candidates(config, overwrite=args.overwrite)
    print(json.dumps(stats_dict(stats), indent=2))


if __name__ == "__main__":
    main()
