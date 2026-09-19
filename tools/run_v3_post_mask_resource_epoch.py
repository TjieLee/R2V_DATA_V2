#!/usr/bin/env python3
"""Opt-in resource-epoch launcher for Post-Mask Visual production.

This entrypoint is deliberately separate from
``scripts/run_v3_post_mask_visual_cluster.sh``. The existing production launcher
and all of ``legacy_serial`` / ``wavefront_v2`` / ``parallel_review_v21`` keep
their current behaviour; nothing here is wired into them.

What it does today:

* enumerate the canonical shards of the campaign;
* build deterministic 8-shard resource-epoch groups whose identity excludes
  hostname, RANK, WORLD_SIZE, GPU, PID and concurrency;
* take each statically assigned group's shared lock
  (``group_index % world_size == rank``), never stealing work;
* publish/validate the immutable group descriptor and durable plan;
* run the group through the resource-epoch scheduler when a job runner is
  supplied, otherwise report the plan and exit.

What it does NOT do: it contains no Visual/Post-Mask semantic logic. The
per-phase model-job semantics (background removal, reference edit, subject
attributes) must be supplied as an external runner, because those call graphs
live in the frozen semantic modules and must be reused, never duplicated.

    --job-runner package.module:callable

The callable receives ``(group, ledger, emit)`` and returns a mapping with at
least ``{"completed": bool}``. It owns executors, job seeding and CPU
finalizers, and therefore owns every prompt, seed, threshold and accept/reject
rule. Until that runner exists for the real phases, run with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ENTITY_MASK_ROOT = (
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_mask"
)
DEFAULT_TAG = "post-mask-v1"


def _emit(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, sort_keys=True), flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--entity-mask-root", default=DEFAULT_ENTITY_MASK_ROOT)
    parser.add_argument("--post-mask-root", default=None)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--job-runner",
        default=None,
        help="package.module:callable owning real phase semantics",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and report groups without executing model jobs",
    )
    return parser


def _load_runner(spec: str):
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError("--job-runner must be package.module:callable")
    from importlib import import_module

    module = import_module(module_name)
    runner = getattr(module, attribute)
    if not callable(runner):
        raise TypeError(f"--job-runner {spec} is not callable")
    return runner


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from r2v_data_v2.v3 import config as config_module
    from r2v_data_v2.v3.config import load_config
    from r2v_data_v2.v3.post_mask_epoch_groups import (
        GROUP_COMPLETED,
        GROUP_INCOMPLETE,
        assigned_groups,
        build_groups,
        group_ownership,
        record_group_outcome,
        resource_epoch_root,
        write_group_descriptor,
    )
    from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger
    from r2v_data_v2.v3.post_mask_production import enumerate_shards

    try:
        config = load_config(Path(args.base_config))
        entity_mask_root = Path(args.entity_mask_root)
        shards = enumerate_shards(entity_mask_root)
        if not shards:
            raise ValueError("campaign has no canonical shards")

        state_root = (
            Path(args.post_mask_root)
            if args.post_mask_root
            else config_module.ALLOWED_WRITABLE_ROOT
            / "r2v_v3_post_mask"
            / "jea_motion_v1"
            / args.tag
        )
        campaign = {
            "config_hash": config.fingerprint(),
            "dataset_json": str(getattr(config, "dataset_json", "")),
            "entity_mask_root": str(entity_mask_root),
            "canonical_shard_count": len(shards),
        }
        groups = build_groups(
            [shard.stem for shard in shards],
            campaign=campaign,
            group_size=args.group_size,
        )
        owned = assigned_groups(groups, rank=args.rank, world_size=args.world_size)
        runner = _load_runner(args.job_runner) if args.job_runner else None
        if runner is None and not args.dry_run:
            raise ValueError(
                "resource_epoch_v3 needs --job-runner for real phase semantics; "
                "use --dry-run to plan only"
            )

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = "unknown"

        _emit(
            "post_mask_resource_epoch_started",
            rank=args.rank,
            world_size=args.world_size,
            group_count=len(groups),
            assigned_groups=len(owned),
            group_size=args.group_size,
            git_commit=git_commit,
            dry_run=bool(args.dry_run or runner is None),
        )

        planned, completed, incomplete, skipped = 0, 0, 0, 0
        for group in owned:
            with group_ownership(state_root, group) as held:
                if not held:
                    skipped += 1
                    _emit(
                        "post_mask_resource_epoch_group_locked_elsewhere",
                        group_id=group.group_id,
                    )
                    continue
                write_group_descriptor(state_root, group)
                planned += 1
                _emit(
                    "post_mask_resource_epoch_group_started",
                    group_id=group.group_id,
                    group_identity=group.identity(),
                    shards=list(group.canonical_shards),
                )
                if runner is None:
                    _emit(
                        "post_mask_resource_epoch_group_planned",
                        group_id=group.group_id,
                    )
                    continue
                ledger = GroupLedger(
                    resource_epoch_root(state_root) / group.group_id
                )
                outcome = runner(group, ledger, _emit) or {}
                if outcome.get("completed"):
                    completed += 1
                    record_group_outcome(state_root, group, outcome=GROUP_COMPLETED)
                    _emit(
                        "post_mask_resource_epoch_group_completed",
                        group_id=group.group_id,
                    )
                else:
                    incomplete += 1
                    record_group_outcome(
                        state_root,
                        group,
                        outcome=GROUP_INCOMPLETE,
                        reason=str(outcome.get("reason", "")),
                    )
                    _emit(
                        "post_mask_resource_epoch_group_incomplete",
                        group_id=group.group_id,
                        reason=str(outcome.get("reason", "")),
                    )

        summary = {
            "rank": args.rank,
            "world_size": args.world_size,
            "group_count": len(groups),
            "groups_attempted": planned,
            "group_completed": completed,
            "group_incomplete": incomplete,
            "groups_locked_elsewhere": skipped,
        }
        state_root.mkdir(parents=True, exist_ok=True)
        (resource_epoch_root(state_root) / f"summary-rank{args.rank}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        _emit("post_mask_resource_epoch_summary", **summary)
        return 0 if incomplete == 0 else 1
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - CLI boundary
        _emit("post_mask_resource_epoch_failed", reason=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
