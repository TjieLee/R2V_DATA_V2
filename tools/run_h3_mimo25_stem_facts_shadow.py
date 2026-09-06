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
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_facts_runtime import (
    AuthoritativeOpenAIStemFactsBackend,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    FFmpegStemViewBackend,
    run_mimo25_stem_facts_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    load_stem_shadow,
    require_shadow_output_path,
    selected_stem_records,
    stem_separation_root,
    stem_shadow_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate SAM Audio stems with MiMo")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id")
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default="music_first",
    )
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--media-mode", choices=("base64", "http"), default="base64")
    parser.add_argument("--media-root", type=Path, default=Path("/mnt/workspace"))
    parser.add_argument("--media-base-url")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    shadow = stem_shadow_root(paths.root, arguments.shadow_run_id)
    separation = stem_separation_root(paths.root, arguments.shadow_run_id)
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / "mimo_stem_facts",
    )
    inventory, records, _ = load_stem_shadow(separation)
    if arguments.case_manifest is not None:
        manifest = MimoCaseManifest.model_validate_json(
            arguments.case_manifest.read_text(encoding="utf-8")
        )
        if manifest.clip_uids != inventory.clip_uids:
            raise ValueError("case manifest differs from SAM Audio stem inventory")
    selected = selected_stem_records(
        records,
        route=arguments.sam_route,
        allow_unverified=arguments.allow_unverified,
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "output_root": str(output),
        "clip_count": len(selected),
        "clip_uids": [item.clip_uid for item in selected],
        "inventory_fingerprint": inventory.inventory_fingerprint,
    }
    if not arguments.dry_run:
        backend = AuthoritativeOpenAIStemFactsBackend(
            model=arguments.model,
            base_url=arguments.base_url,
            api_key=os.environ.get("MIMO_API_KEY", ""),
            media_resolver=MimoMediaResolver(
                mode=arguments.media_mode,
                media_root=arguments.media_root,
                media_base_url=arguments.media_base_url,
            ),
            temperature=arguments.temperature,
        )
        summary = run_mimo25_stem_facts_shadow(
            stem_root=separation,
            route=arguments.sam_route,
            backend=backend,
            view_backend=FFmpegStemViewBackend(ffmpeg=arguments.ffmpeg),
            output_root=output,
            allow_unverified=arguments.allow_unverified,
            overwrite=arguments.overwrite,
        )
        result.update(model_called=True, summary=summary.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
