#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from r2v_data_v2.h3.final_renderer import (
    plan_final_h3_renderer,
    publish_final_h3_renderer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble frozen H3 pair, speech, and Visual artifacts."
    )
    parser.add_argument("--audio-run-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        result = plan_final_h3_renderer(audio_run_root=args.audio_run_root)
    else:
        summary = publish_final_h3_renderer(
            audio_run_root=args.audio_run_root,
            overwrite=args.overwrite,
        )
        result = {
            "output_root": str(args.audio_run_root / "production" / "h3"),
            "published": True,
            "inventory_fingerprint": summary.inventory_fingerprint,
            "in_pair_sample_count": summary.in_pair_sample_count,
            "cross_pair_sample_count": summary.cross_pair_sample_count,
            "total_sample_count": summary.total_sample_count,
            "failed_sample_count": summary.failed_sample_count,
            "model_calls": summary.model_calls,
            "mllm_calls": summary.mllm_calls,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
