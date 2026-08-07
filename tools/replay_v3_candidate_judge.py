from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.candidate_judge_replay import (
    run_candidate_judge_replay,
)
from r2v_data_v2.v3.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the V3 candidate judge against an existing run",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--evidence-mode",
        choices=("baseline", "paired_card"),
        default="baseline",
    )
    parser.add_argument(
        "--card-panel-max-side",
        type=int,
        choices=(384, 512),
        default=512,
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = run_candidate_judge_replay(
        load_config(arguments.config),
        run_root=arguments.run_root,
        base_url=arguments.base_url,
        model=arguments.model,
        output_path=arguments.output,
        api_key=arguments.api_key,
        save_raw=arguments.save_raw,
        fail_fast=arguments.fail_fast,
        evidence_mode=arguments.evidence_mode,
        card_panel_max_side=arguments.card_panel_max_side,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
