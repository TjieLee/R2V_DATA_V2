from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.voice_quality import (
    recompute_voice_reference_quality_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute calibration-only H3 voice-quality diagnostics from an "
            "existing LR-ASD pilot without rerunning any backend"
        ),
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = recompute_voice_reference_quality_artifacts(
        pilot_root=arguments.pilot_root,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
