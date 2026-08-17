from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.subject_attributes import run_subject_attribute_enrichment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich accepted Visual V3 human subjects with bounded owner-bound "
            "attribute references without changing the source run"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-owners",
        type=int,
        default=50,
        help="maximum eligible human owners to benchmark (default: 50)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = run_subject_attribute_enrichment(
        load_config(arguments.config),
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        max_owners=arguments.max_owners,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
