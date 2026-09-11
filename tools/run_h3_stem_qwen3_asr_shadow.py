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
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration
from r2v_data_v2.h3.qwen3_asr_subprocess import PersistentQwen3ASRBackend
from r2v_data_v2.h3.resolved_audio_stems import (
    downstream_stem_route,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    require_shadow_output_path,
    run_stem_qwen3_asr_shadow,
    stem_shadow_root,
    validate_stem_diarization_lineage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR on SAM speech stems")
    parser.add_argument("--visual-production-root", type=Path)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id")
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default=None,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--diarization-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _isolated_backend() -> PersistentQwen3ASRBackend:
    environment_root = os.environ.get("QWEN3_ASR_ENV", "").strip()
    if not environment_root:
        raise ValueError("QWEN3_ASR_ENV must identify the isolated qwen-asr env")
    python = Path(environment_root).expanduser() / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"QWEN3_ASR_ENV Python is missing: {python}")
    try:
        timeout_seconds = float(os.environ.get("QWEN3_ASR_TIMEOUT_SECONDS", "300"))
    except ValueError as exc:
        raise ValueError("QWEN3_ASR_TIMEOUT_SECONDS must be numeric") from exc
    return PersistentQwen3ASRBackend(
        Qwen3ASRConfiguration.from_environment(),
        python_path=python,
        worker_path=REPOSITORY_ROOT / "tools" / "qwen3_asr_worker.py",
        timeout_seconds=timeout_seconds,
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    route = downstream_stem_route(arguments.shadow_run_id, arguments.sam_route)
    shadow = stem_shadow_root(paths.root, arguments.shadow_run_id)
    diarization = require_shadow_output_path(
        shadow_root=shadow, output_path=arguments.diarization_root or shadow / "diarization",
    )
    source_provenance, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=shadow,
    )
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / (
            "asr_no_lrasd_v1" if source_provenance.binding_evidence_mode == "none" else "asr"
        ),
    )
    if source_provenance.binding_evidence_mode == "none" and output == shadow / "asr":
        raise ValueError("no-LR-ASD pilot requires an isolated ASR output root")
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
        "diarization_root": str(diarization),
        "binding_evidence_mode": source_provenance.binding_evidence_mode,
        "output_root": str(output),
    }
    if source_provenance.route != route:
        raise ValueError("stem ASR route differs from selected SAM route")
    if manifest is not None and manifest.clip_uids != source_provenance.clip_uids:
        raise ValueError("stem ASR case manifest differs from stem diarization order")
    result["clip_uids"] = source_provenance.usable_clip_uids
    if not arguments.dry_run:
        if arguments.visual_production_root is None:
            from r2v_data_v2.h3.diarization_binding import DiarizationInventory

            source = DiarizationInventory.model_validate_json(
                (diarization / "inventory.json").read_text()
            )
            if source.source_inventory_kind != "jea_shot_manifest":
                raise ValueError("--visual-production-root is required for legacy Visual-rooted ASR")
        backend = _isolated_backend()
        with backend:
            summary, provenance = run_stem_qwen3_asr_shadow(
                stem_diarization_root=diarization,
                source_visual_production_root=str(
                    arguments.visual_production_root.expanduser().resolve(strict=True)
                ) if arguments.visual_production_root is not None else None,
                backend=backend,
                output_root=output,
                case_manifest=manifest,
                ffmpeg=arguments.ffmpeg,
                route=route,
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
