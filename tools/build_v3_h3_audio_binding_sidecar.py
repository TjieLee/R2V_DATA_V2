from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.backends import PrecomputedEvidenceBackend
from r2v_data_v2.h3.sidecar import build_audio_binding_sidecar_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only H3 audio-binding sidecar from precomputed evidence"
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = build_audio_binding_sidecar_run(
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        backend=PrecomputedEvidenceBackend.from_path(arguments.evidence_json),
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
