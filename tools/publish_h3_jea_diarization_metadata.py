#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_diarization import publish_readable_diarization_metadata
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Attach readable JEA metadata to unchanged DiariZen segments",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    visual = load_visual_production_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
    )
    summary = publish_readable_diarization_metadata(
        visual_inventory=visual,
        diarization_root=arguments.audio_production_root / "diarization",
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
