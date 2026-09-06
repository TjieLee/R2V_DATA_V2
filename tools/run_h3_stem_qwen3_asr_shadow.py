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
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRBackend, Qwen3ASRConfiguration
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    StemDiarizationShadowProvenance,
    require_shadow_output_path,
    run_stem_qwen3_asr_shadow,
    stem_shadow_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR on SAM speech stems")
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    shadow = stem_shadow_root(paths.root)
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / "asr",
    )
    manifest = (
        MimoCaseManifest.model_validate_json(
            arguments.case_manifest.read_text(encoding="utf-8")
        )
        if arguments.case_manifest
        else None
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "diarization_root": str(shadow / "diarization"),
        "output_root": str(output),
    }
    source_provenance = StemDiarizationShadowProvenance.model_validate_json(
        (shadow / "diarization" / "stem_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest is not None and manifest.clip_uids != source_provenance.clip_uids:
        raise ValueError("stem ASR case manifest differs from stem diarization order")
    result["clip_uids"] = source_provenance.usable_clip_uids
    if not arguments.dry_run:
        if not os.environ.get("QWEN3_ASR_ENV"):
            raise ValueError("QWEN3_ASR_ENV must identify the isolated qwen-asr env")
        summary, provenance = run_stem_qwen3_asr_shadow(
            stem_diarization_root=shadow / "diarization",
            source_visual_production_root=str(
                arguments.visual_production_root.expanduser().resolve(strict=True)
            ),
            backend=Qwen3ASRBackend(Qwen3ASRConfiguration.from_environment()),
            output_root=output,
            case_manifest=manifest,
            ffmpeg=arguments.ffmpeg,
            allow_unverified=arguments.allow_unverified,
            overwrite=arguments.overwrite,
        )
        result.update(
            model_called=True,
            summary=summary.model_dump(mode="json"),
            provenance=provenance.model_dump(mode="json"),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
