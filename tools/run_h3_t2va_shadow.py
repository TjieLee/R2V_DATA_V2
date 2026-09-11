"""Build an independent no-reference T2VA run from finalized speech facts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoBackend, T2VAMimoConfig
from r2v_data_v2.h3.t2va_shadow import build_t2va_inventory, run_t2va_shadow, t2va_root


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-manifest", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path)
    parser.add_argument("--source-videos-root", type=Path)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--audio-shadow-run-id", required=True)
    parser.add_argument("--t2va-run-id", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-manifest", type=Path)
    selection.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument(
        "--shot-index-root",
        type=Path,
        help="Writable JSONL offset cache (outside source directory)",
    )
    parser.add_argument("--model", default="mimo-v2.5", choices=["mimo-v2.5"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--transport", choices=["sglang", "xiaomi"], default="sglang")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-mode", choices=["base64", "http"], default="base64")
    parser.add_argument("--media-base-url")
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    config = T2VAMimoConfig(
        media_resolver=MimoMediaResolver(
            mode=args.media_mode,
            media_root=args.media_root,
            media_base_url=args.media_base_url,
        ),
        base_url=args.base_url,
        api_key=os.environ.get("MIMO_API_KEY", "local-no-key"),
        transport=args.transport,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
    )
    inventory = build_t2va_inventory(
        shot_manifest=args.shot_manifest,
        clips_root=args.clips_root,
        source_videos_root=args.source_videos_root,
        audio_production_root=args.audio_production_root,
        audio_shadow_run_id=args.audio_shadow_run_id,
        t2va_run_id=args.t2va_run_id,
        backend=config.provenance(),
        case_manifest=args.case_manifest,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        shot_index_root=args.shot_index_root,
    )
    if args.dry_run:
        return {
            "dry_run": True,
            "inventory_fingerprint": inventory.inventory_fingerprint,
            "clip_uids": inventory.clip_uids,
            "eligible_clip_count": sum(
                j.upstream_failure is None for j in inventory.jobs
            ),
            "model_call_count": 0,
            "output_root": str(t2va_root(args.audio_production_root, args.t2va_run_id)),
        }
    return run_t2va_shadow(
        inventory, T2VAMimoBackend(config), overwrite=args.overwrite
    ).model_dump(mode="json")


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
