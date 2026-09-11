"""Build target-only TA2VA from a frozen, summary-enabled T2VA run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoConfig
from r2v_data_v2.h3.ta2va_shadow import (
    TA2VAProfileBackend,
    load_source,
    run_ta2va_shadow,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2va-root", type=Path, required=True)
    parser.add_argument("--ta2va-run-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--transport", choices=["sglang", "xiaomi"], default="sglang")
    parser.add_argument("--model", choices=["mimo-v2.5"], default="mimo-v2.5")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        _, clips, _, _ = load_source(args.t2va_root)
        return {"clip_uids": [j.clip_uid for j, *_ in clips], "model_call_count": 0}
    config = T2VAMimoConfig(
        media_resolver=MimoMediaResolver(mode="base64", media_root=args.media_root),
        base_url=args.base_url,
        transport=args.transport,
        model=args.model,
        api_key=os.environ.get("MIMO_API_KEY", "local-no-key"),
        max_completion_tokens=args.max_completion_tokens,
    )
    output = run_ta2va_shadow(
        args.t2va_root,
        args.ta2va_run_id,
        TA2VAProfileBackend(config),
        allow_unverified=args.allow_unverified,
    )
    return {
        "output_root": str(output),
        **json.loads((output / "summary.json").read_text()),
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
