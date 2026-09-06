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
from r2v_data_v2.h3.mimo25_av_reconcile import build_mimo25_inventory
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_shadow import (
    StemAwareOpenAIMimo25Backend,
    build_stem_reconcile_jobs,
    load_stem_fact_records,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    load_stem_shadow,
    require_shadow_output_path,
    separation_skips,
    stem_separation_root,
    stem_shadow_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile original AV with SAM Audio stem facts",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default="music_first",
    )
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--media-mode", choices=("base64", "http"), default="base64")
    parser.add_argument("--media-root", type=Path, default=Path("/mnt/workspace"))
    parser.add_argument("--media-base-url")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    shadow = stem_shadow_root(paths.root)
    separation = stem_separation_root(paths.root)
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / "mimo_reconcile",
    )
    base = build_mimo25_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=base,
        stem_diarization_root=shadow / "diarization",
        stem_asr_root=shadow / "asr",
    )
    facts = load_stem_fact_records(shadow / "mimo_stem_facts")
    stem_inventory, stem_records, _ = load_stem_shadow(separation)
    skipped = separation_skips(
        inventory=stem_inventory,
        records=stem_records,
        route=arguments.sam_route,
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "output_root": str(output),
        "clip_count": len(jobs),
        "clip_uids": [item.clip_uid for item in jobs],
        "original_target_av_is_highest_authority": True,
    }
    if not arguments.dry_run:
        facts_by_clip = {item.clip_uid: item for item in facts}
        backend = StemAwareOpenAIMimo25Backend(
            MimoBackendConfig(
                media_resolver=MimoMediaResolver(
                    mode=arguments.media_mode,
                    media_root=arguments.media_root,
                    media_base_url=arguments.media_base_url,
                ),
                api_key=os.environ.get("MIMO_API_KEY", ""),
                transport="sglang",
                base_url=arguments.base_url,
                model=arguments.model,
                max_completion_tokens=arguments.max_completion_tokens,
            ),
            stem_facts_by_clip=facts_by_clip,
        )
        summary = run_mimo25_stem_reconcile_shadow(
            jobs=jobs,
            stem_facts=facts,
            backend=backend,
            output_root=output,
            source_clip_uids=stem_inventory.clip_uids,
            skipped_clips=skipped,
            allow_unverified=arguments.allow_unverified,
            overwrite=arguments.overwrite,
        )
        result.update(model_called=True, summary=summary.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
