#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.auk_speech_qa import build_auk_speech_qa


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Build independent static AuK / SAM speech QA"
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    args = parser.parse_args(argv)
    result = build_auk_speech_qa(
        audio_production_root=args.audio_production_root,
        shadow_run_id=args.shadow_run_id,
    )
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
