#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import render_jea_final_samples
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render canonical JEA H3 samples from DiariZen and Qwen3 ASR",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    visual = load_visual_production_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
    )
    paths = jea_production_paths(arguments.audio_production_root)
    summary = render_jea_final_samples(
        visual_inventory=visual,
        pairs_root=paths.pairs,
        diarization_root=paths.diarization,
        binding_audit_root=paths.root / "binding_audit_v1",
        qwen3_asr_root=paths.asr,
        output_root=paths.h3,
        overwrite=arguments.overwrite,
    )
    result = {
        "audio_production_root": str(paths.root),
        "h3_output_root": str(paths.h3),
        "summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
