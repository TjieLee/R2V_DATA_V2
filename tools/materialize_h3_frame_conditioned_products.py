#!/usr/bin/env python3
"""Project finalized named-run Audio/H3 products onto exact first/last video frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.frame_conditioning_projection import (
    materialize_frame_conditioned_products,
)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output-root", type=Path)
    result = materialize_frame_conditioned_products(**vars(parser.parse_args(argv))).model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
