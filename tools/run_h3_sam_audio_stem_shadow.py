#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    FFmpegStemCanonicalizer,
    OfficialSAMAudioBackend,
    build_sam_audio_stem_inventory,
    require_shadow_output_path,
    run_sam_audio_stem_shadow,
    sam_audio_configuration,
    stem_separation_root,
    stem_shadow_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run opt-in SAM Audio stem shadow")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id")
    parser.add_argument("--sam-audio-code-root", type=Path, required=True)
    parser.add_argument("--sam-audio-model-path", type=Path, required=True)
    parser.add_argument("--sam-audio-model-name")
    parser.add_argument("--sam-audio-t5-base-path", type=Path, required=True)
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default="music_first",
    )
    parser.add_argument(
        "--sam-reranking-candidates",
        type=int,
        choices=(1,),
        default=1,
    )
    parser.add_argument("--run-both-routes", action="store_true")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    shadow = stem_shadow_root(paths.root, arguments.shadow_run_id)
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or stem_separation_root(
            paths.root, arguments.shadow_run_id,
        ),
    )
    configuration = sam_audio_configuration(
        implementation_root=arguments.sam_audio_code_root,
        model_path=arguments.sam_audio_model_path,
        model_name=arguments.sam_audio_model_name,
        t5_base_path=arguments.sam_audio_t5_base_path,
        device=arguments.sam_device,
        reranking_candidates=arguments.sam_reranking_candidates,
    )
    inventory = build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=paths.audio / "canonical_clips.jsonl",
        model_configuration=configuration,
        route=arguments.sam_route,
        run_both_routes=arguments.run_both_routes,
        case_manifest_path=arguments.case_manifest,
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "output_root": str(output),
        "clip_count": inventory.job_count,
        "clip_uids": [item.clip_uid for item in inventory.jobs],
        "route": inventory.route,
        "run_both_routes": inventory.run_both_routes,
        "inventory_fingerprint": inventory.inventory_fingerprint,
    }
    if not arguments.dry_run:
        summary = run_sam_audio_stem_shadow(
            inventory=inventory,
            output_root=output,
            backend=OfficialSAMAudioBackend(configuration),
            canonicalizer=FFmpegStemCanonicalizer(
                ffmpeg=arguments.ffmpeg,
                ffprobe=arguments.ffprobe,
            ),
            raw_probe_backend=FFmpegAudioMediaBackend(
                ffmpeg=arguments.ffmpeg,
                ffprobe=arguments.ffprobe,
            ),
            overwrite=arguments.overwrite,
        )
        result.update(model_called=True, summary=summary.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
