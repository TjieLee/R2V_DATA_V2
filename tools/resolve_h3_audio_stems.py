#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.resolved_audio_stems import resolve_audio_stems


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve fixed AuK speech + SAM music_first music/SFX"
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = resolve_audio_stems(
        audio_production_root=args.audio_production_root,
        shadow_run_id=args.shadow_run_id,
        overwrite=args.overwrite,
    )
    print(summary.model_dump_json(indent=2))
    return summary


if __name__ == "__main__":
    main()
