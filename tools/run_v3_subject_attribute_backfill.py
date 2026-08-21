#!/usr/bin/env python3
"""Backfill subject attributes for an already completed Visual V3 run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.config import V3Config, load_config
from r2v_data_v2.v3.subject_attributes import run_subject_attribute_enrichment


@contextmanager
def _visible_gpu(physical_gpu: str) -> Iterator[None]:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def run_backfill(
    config: V3Config,
    *,
    max_owners: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    physical_gpu = config.runtime.gpu_workers.subject_attributes_segment
    if physical_gpu is None:
        raise ValueError(
            "runtime.gpu_workers.subject_attributes_segment is required for backfill"
        )
    with _visible_gpu(physical_gpu):
        return run_subject_attribute_enrichment(
            config,
            run_root=config.resolved_run_root,
            output_root=config.resolved_run_root / "subject_attributes",
            max_owners=max_owners,
            overwrite=overwrite,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-owners", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    summary = run_backfill(
        load_config(args.config),
        max_owners=args.max_owners,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
