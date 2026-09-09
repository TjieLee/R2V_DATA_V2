#!/usr/bin/env python3
"""Materialize Audio reuse products from prepared frozen roots; never infer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_reuse_materializer import materialize_audio_reuse_products


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mimo-root", "source-h3-root", "separation-root", "reuse-root", "audio-production-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--enable-full-audio-reuse", action="store_true")
    args = parser.parse_args(argv)
    summary = materialize_audio_reuse_products(**vars(args))
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
