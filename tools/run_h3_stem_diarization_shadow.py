#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    build_stem_diarization_inventory,
    load_stem_shadow,
    require_shadow_output_path,
    run_stem_diarization_shadow,
    stem_separation_root,
    stem_shadow_root,
)
from tools.run_h3_diarization_binding import _runtime_backend


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DiariZen on SAM speech stems")
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id")
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default="music_first",
    )
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
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
        output_path=arguments.output_root or shadow / "diarization",
    )
    stem_inventory, records, _ = load_stem_shadow(separation)
    if arguments.case_manifest is not None:
        manifest = MimoCaseManifest.model_validate_json(
            arguments.case_manifest.read_text(encoding="utf-8")
        )
        if manifest.clip_uids != stem_inventory.clip_uids:
            raise ValueError("case manifest differs from SAM Audio stem inventory")
    from r2v_data_v2.h3.diarization_binding import DiarizationInventory

    production_inventory = DiarizationInventory.model_validate_json(
        (paths.diarization / "inventory.json").read_text(encoding="utf-8")
    )
    inventory = build_stem_diarization_inventory(
        stem_inventory=stem_inventory,
        stem_records=records,
        production_diarization_inventory=production_inventory,
        route=arguments.sam_route,
        allow_unverified=arguments.allow_unverified,
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "output_root": str(output),
        "target_count": len(inventory.targets),
        "target_clip_uids": [item.target_clip_uid for item in inventory.targets],
        "diarization_source_kind": "sam_audio_speech_stem",
    }
    if not arguments.dry_run:
        backend, diagnostics = _runtime_backend(
            output_root=output,
            input_profile="canonical_32k_stereo",
        )
        try:
            with backend:
                provenance = run_stem_diarization_shadow(
                    stem_root=separation,
                    production_diarization_root=paths.diarization,
                    backend=backend,
                    route=arguments.sam_route,
                    output_root=output,
                    allow_unverified=arguments.allow_unverified,
                    overwrite=arguments.overwrite,
                )
        finally:
            if diagnostics.exists():
                shutil.rmtree(diagnostics)
        result.update(model_called=True, provenance=provenance.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
