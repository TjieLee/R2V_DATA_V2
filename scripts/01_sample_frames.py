from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import load_config
from r2v_data_v2.video_io import sample_manifest_frames, stats_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample exactly eight video frames")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = sample_manifest_frames(load_config(args.config), overwrite=args.overwrite)
    print(json.dumps(stats_dict(stats), indent=2))


if __name__ == "__main__":
    main()
