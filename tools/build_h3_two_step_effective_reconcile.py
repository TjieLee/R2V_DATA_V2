#!/usr/bin/env python3
"""Publish Random20 effective records without models or source-media reads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.two_step_effective_reconcile import build_effective_reconcile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--override-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    summary = build_effective_reconcile(**vars(parser.parse_args(argv)))
    print(summary.model_dump_json(indent=2))
    if (summary.clip_count, summary.ready_count, summary.failed_count) != (20, 17, 3):
        print("Effective reconcile differs from expected 20/17/3; see effective_sources.json for actual failures.",
              file=sys.stderr)
        raise SystemExit(2)
    return summary


if __name__ == "__main__":
    main()
