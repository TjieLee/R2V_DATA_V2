#!/usr/bin/env python3
"""GPU-isolated entrypoint. Only stdlib imports before checking placement."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--worker-slot", type=int, required=True)
    early.add_argument("--sam-gpu", required=True)
    early.add_argument("--boogu-gpu", required=True)
    placement, remaining = early.parse_known_args(argv)
    if (
        not placement.sam_gpu.isdecimal()
        or not placement.boogu_gpu.isdecimal()
        or placement.sam_gpu == placement.boogu_gpu
        or os.environ.get("CUDA_VISIBLE_DEVICES") != placement.sam_gpu
    ):
        early.error("worker must start isolated to its physical SAM GPU")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from tools.run_v3_post_mask_visual import (
        _interrupt,
        emit,
        load_launch,
        parser,
        run_worker,
    )

    signal.signal(signal.SIGTERM, _interrupt)
    try:
        args = parser().parse_args(remaining)
        config, campaign, groups = load_launch(args)
        if not 0 <= placement.worker_slot < len(groups) or groups[
            placement.worker_slot
        ] != (placement.sam_gpu, placement.boogu_gpu):
            raise ValueError("worker GPU pair does not match local group slot")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        run_worker(
            campaign,
            config,
            runtime=config.runtime,
            rank=args.rank,
            world_size=args.world_size,
            slot=placement.worker_slot,
            groups=groups,
            git_commit=commit,
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - CLI failure boundary
        emit(
            {
                "event": "post_mask_worker_failed",
                "slot": placement.worker_slot,
                "reason": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
