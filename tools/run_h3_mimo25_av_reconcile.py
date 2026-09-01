#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.mimo25_av_reconcile import (
    build_mimo25_inventory,
    known_case_manifest,
    run_mimo25_av_reconcile,
)
from r2v_data_v2.h3.mimo25_backend import (
    MIMO25_DEFAULT_BASE_URL,
    MIMO25_MODEL,
    MimoBackendConfig,
    MimoMediaResolver,
    OpenAIMimo25Backend,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiMo-V2.5 unified AV shadow")
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-manifest", type=Path)
    selection.add_argument("--known-case-pilot", action="store_true")
    parser.add_argument("--model", default=MIMO25_MODEL)
    parser.add_argument("--base-url", default=MIMO25_DEFAULT_BASE_URL)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--media-resolution", default="default")
    parser.add_argument("--media-mode", choices=("base64", "http"), default="base64")
    parser.add_argument("--media-root", type=Path, default=Path("/mnt/workspace"))
    parser.add_argument("--media-base-url")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--http-max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    inventory = build_mimo25_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
        case_manifest=(known_case_manifest() if arguments.known_case_pilot else None),
    )
    output_root = paths.root / "mimo25_av_reconcile_v1"
    if arguments.dry_run:
        result = {
            "dry_run": True,
            "model_called": False,
            "output_root": str(output_root),
            "clip_count": inventory.clip_count,
            "clip_uids": [item.clip_uid for item in inventory.jobs],
            "inventory_fingerprint": inventory.inventory_fingerprint,
            "inventory_scope": inventory.inventory_scope,
            "canonical_wide_coverage": inventory.canonical_wide_coverage,
        }
    else:
        api_key = os.environ.get("MIMO_API_KEY", "")
        resolver = MimoMediaResolver(
            mode=arguments.media_mode,
            media_root=arguments.media_root,
            media_base_url=arguments.media_base_url,
        )
        backend = OpenAIMimo25Backend(
            MimoBackendConfig(
                media_resolver=resolver,
                api_key=api_key,
                base_url=arguments.base_url,
                model=arguments.model,
                video_fps=arguments.fps,
                media_resolution=arguments.media_resolution,
                temperature=arguments.temperature,
                max_completion_tokens=arguments.max_completion_tokens,
                timeout_seconds=arguments.timeout_seconds,
                http_max_attempts=arguments.http_max_attempts,
            )
        )
        summary = run_mimo25_av_reconcile(
            inventory=inventory,
            backend=backend,
            output_root=output_root,
            overwrite=arguments.overwrite,
        )
        result = {
            "dry_run": False,
            "output_root": str(output_root),
            "summary": summary.model_dump(mode="json"),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
