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
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    StemReferenceManifest,
    build_shadow_primary_voice_reference_requests,
    export_stem_native_references,
    load_stem_shadow,
    require_shadow_output_path,
    stem_separation_root,
    stem_shadow_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export exact stem-native references")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-manifest", type=Path)
    source.add_argument("--primary-voice-root", type=Path)
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default="music_first",
    )
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    shadow = stem_shadow_root(paths.root)
    inventory, records, _ = load_stem_shadow(stem_separation_root(paths.root))
    selected_clip_uids: list[str] | None = None
    if arguments.case_manifest is not None:
        cases = MimoCaseManifest.model_validate_json(
            arguments.case_manifest.read_text(encoding="utf-8")
        )
        if cases.clip_uids != inventory.clip_uids:
            raise ValueError("case manifest differs from SAM Audio stem inventory")
        selected_clip_uids = cases.clip_uids
    if arguments.reference_manifest is not None:
        manifest = StemReferenceManifest.model_validate_json(
            arguments.reference_manifest.read_text(encoding="utf-8")
        )
        if selected_clip_uids is not None and any(
            item.clip_uid not in set(selected_clip_uids) for item in manifest.requests
        ):
            raise ValueError("reference manifest contains a clip outside case selection")
        requests = manifest.requests
        selection_source = "explicit_reference_manifest"
    else:
        if arguments.primary_voice_root is None:
            raise AssertionError("argparse requires a reference source")
        requests = build_shadow_primary_voice_reference_requests(
            primary_voice_root=arguments.primary_voice_root,
            records=records,
            route=arguments.sam_route,
            selected_clip_uids=selected_clip_uids,
            allow_unverified=arguments.allow_unverified,
        )
        selection_source = "accepted_primary_voice_v1"
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / "references",
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "output_root": str(output),
        "reference_count": len(requests),
        "reference_ids": [item.reference_id for item in requests],
        "selection_source": selection_source,
    }
    if not arguments.dry_run:
        references = export_stem_native_references(
            records=records,
            requests=requests,
            output_root=output,
            media_backend=FFmpegAudioMediaBackend(
                ffmpeg=arguments.ffmpeg,
                ffprobe=arguments.ffprobe,
            ),
            allow_unverified=arguments.allow_unverified,
        )
        result["references"] = [item.model_dump(mode="json") for item in references]
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
