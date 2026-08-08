from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.background_final_guard_replay import (
    run_background_final_guard_replay,
)
from r2v_data_v2.v3.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the V3 final background guard against an existing run",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = run_background_final_guard_replay(
        load_config(arguments.config),
        run_root=arguments.run_root,
        output_path=arguments.output,
        base_url=arguments.base_url,
        model=arguments.model,
        api_key=arguments.api_key,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
