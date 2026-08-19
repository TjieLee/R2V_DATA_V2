#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.background_audio_scout import (
    background_audio_scout_output_root,
    build_background_audio_scout_inventory,
    run_background_audio_scout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the model-free H3 background-audio scouting report",
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    audio_run_root = arguments.audio_run_root.expanduser().resolve(strict=True)
    inventory = build_background_audio_scout_inventory(audio_run_root=audio_run_root)
    output_root = background_audio_scout_output_root(audio_run_root)
    summary = run_background_audio_scout(
        inventory=inventory,
        output_root=output_root,
        overwrite=arguments.overwrite,
    )
    result = {
        "output_root": str(output_root),
        "target_clip_count": inventory.target_clip_count,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "model_calls": 0,
        "automatic_background_selection_applied": False,
        "summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
