#!/usr/bin/env python3
"""Finalize an existing named SAM/MiMo run without model calls or overwrites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_reuse_finalizer import finalize_audio_reuse_shadow


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--sam-route", choices=("music_first", "voice_first"), default="music_first")
    parser.add_argument("--allow-unverified", action="store_true")
    result = finalize_audio_reuse_shadow(**vars(parser.parse_args(argv))).model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
