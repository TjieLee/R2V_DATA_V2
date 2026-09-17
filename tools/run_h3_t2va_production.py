"""Run fixed target-side production shards without changing frozen semantics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3 import t2va_production as production


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-manifest", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--source-videos-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--audio-shadow-run-id", required=True)
    parser.add_argument("--production-root", type=Path, default=production.DEFAULT_ROOT)
    parser.add_argument("--shards")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument("--shard-seed", type=int, default=20260917)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--request-workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-mode", choices=["http", "base64"], default="base64")
    parser.add_argument("--media-base-url")
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.production_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise PermissionError(f"production output is not writable: {root}")
    if args.request_workers < 1:
        raise ValueError("request workers must be positive")
    for source in (
        args.shot_manifest,
        args.clips_root,
        args.source_videos_root,
        args.audio_production_root,
    ):
        if source.resolve().is_relative_to(root):
            raise ValueError("production output must not contain source inputs")
    index = production.build_source_index(args.shot_manifest, root)
    order = production.shard_order(
        explicit=[int(x) for x in args.shards.split(",")] if args.shards else None,
        start=args.shard_start,
        end=args.shard_end if args.shard_end is not None else len(index["shards"]) - 1,
        seed=args.shard_seed,
        rank=args.rank,
        world_size=args.world_size,
    )
    if any(i >= len(index["shards"]) for i in order):
        raise ValueError("shard outside source population")
    if args.dry_run:
        return {
            "dry_run": True,
            "shards": order,
            "model_call_count": 0,
            "source_record_count": index["source_record_count"],
            "production_root": str(root),
        }
    from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
    from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoBackend, T2VAMimoConfig
    from r2v_data_v2.h3.ta2va_shadow import TA2VAProfileBackend

    config = T2VAMimoConfig(
        media_resolver=MimoMediaResolver(
            mode=args.media_mode,
            media_root=args.media_root,
            media_base_url=args.media_base_url,
        ),
        base_url=args.base_url,
        api_key=os.environ.get("MIMO_API_KEY", "local-no-key"),
        transport="sglang",
        max_completion_tokens=args.max_completion_tokens,
    )
    backend, profiles = T2VAMimoBackend(config), TA2VAProfileBackend(config)
    results = {}
    for shard_id in order:
        # A separate preparation lock covers manifest publication as well; the
        # process_shard lock protects the journal and per-stage directories.
        try:
            shard = root / "shards" / production.shard_name(shard_id)
            with production.file_lock(shard / "invocation.lock"):
                rows, contexts = production.prepare_shard(
                    root,
                    index,
                    shard_id,
                    audio_production_root=args.audio_production_root,
                    audio_shadow_run_id=args.audio_shadow_run_id,
                    clips_root=args.clips_root,
                    source_videos_root=args.source_videos_root,
                    backend=config.provenance(),
                )
                processor = production.FrozenProductionProcessor(
                    contexts, backend, profiles, allow_unverified=args.allow_unverified
                )
                states = production.process_shard(
                    root,
                    shard_id,
                    rows,
                    processor,
                    request_workers=args.request_workers,
                    progress_every=args.progress_every,
                )
                results[shard_id] = {
                    f"{stage}_{status}": sum(
                        r[f"{stage}_status"] == status for r in states.values()
                    )
                    for stage in ("t2va", "ta2va")
                    for status in ("ready", "failed", "skipped")
                }
        except production.ShardLockedError:
            results[shard_id] = {"locked": True}
            print(f"shard={shard_id} locked; skipped", flush=True)
    return {"shards": results, "production_root": str(root)}


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
