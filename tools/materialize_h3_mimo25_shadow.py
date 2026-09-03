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
from r2v_data_v2.h3.mimo25_h3_materializer import materialize_mimo25_h3_shadow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize MiMo H3 shadow prompts")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    output = paths.root / "mimo25_h3_shadow_v5"
    summary = materialize_mimo25_h3_shadow(
        mimo_root=paths.root / "mimo25_av_reconcile_v5",
        source_h3_root=paths.h3,
        output_root=output,
        overwrite=arguments.overwrite,
    )
    result = {"output_root": str(output), "summary": summary.model_dump(mode="json")}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
