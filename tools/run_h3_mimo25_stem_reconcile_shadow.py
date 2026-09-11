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
    MimoCaseManifest,
    build_mimo25_inventory,
    build_mimo25_reference_inventory,
)
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    StemAwareOpenAIMimo25Backend,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.resolved_audio_stems import (
    StemInventory,
    downstream_stem_root,
    downstream_stem_route,
    load_stem_source,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    StemDiarizationShadowProvenance,
    require_shadow_output_path,
    selected_stem_records,
    separation_skips,
    stem_shadow_root,
    validate_stem_diarization_lineage,
)


def _validate_stage_closure(
    *,
    case_manifest: MimoCaseManifest,
    stem_inventory: StemInventory,
    separation_root: Path,
    diarization_provenance: StemDiarizationShadowProvenance,
) -> None:
    separation = separation_root.expanduser().resolve(strict=True)
    selected_ids = set(case_manifest.clip_uids)
    projected = [uid for uid in stem_inventory.clip_uids if uid in selected_ids]
    if projected != case_manifest.clip_uids:
        raise ValueError("case manifest is not an ordered subset of current SAM Audio inventory")
    if (
        Path(diarization_provenance.source_stem_root)
        .expanduser()
        .resolve(strict=True)
        != separation
    ):
        raise ValueError("stem DiariZen source root differs from current separation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two audio-only stem candidates, then one final original AV request",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--shadow-run-id")
    parser.add_argument("--binding-evidence-mode", choices=("legacy_lr_asd", "none"), default="legacy_lr_asd")
    parser.add_argument("--stem-diarization-root", type=Path)
    parser.add_argument("--stem-asr-root", type=Path)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument(
        "--sam-route",
        choices=("music_first", "voice_first"),
        default=None,
    )
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--media-mode", choices=("base64", "http"), default="base64")
    parser.add_argument("--media-root", type=Path, default=Path("/mnt/workspace"))
    parser.add_argument("--media-base-url")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--icl", choices=("none", "official_ref2va_v1"), default="official_ref2va_v1")
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    route = downstream_stem_route(arguments.shadow_run_id, arguments.sam_route)
    shadow = stem_shadow_root(paths.root, arguments.shadow_run_id)
    separation = downstream_stem_root(paths.root, arguments.shadow_run_id)
    diarization = require_shadow_output_path(
        shadow_root=shadow, output_path=arguments.stem_diarization_root or shadow / "diarization",
    )
    asr = require_shadow_output_path(
        shadow_root=shadow, output_path=arguments.stem_asr_root or shadow / "asr",
    )
    output = require_shadow_output_path(
        shadow_root=shadow,
        output_path=arguments.output_root or shadow / (
            "mimo_reconcile_no_lrasd_v1" if arguments.binding_evidence_mode == "none" else MIMO25_STEM_RECONCILE_STAGE
        ),
    )
    if arguments.binding_evidence_mode == "none" and output == shadow / MIMO25_STEM_RECONCILE_STAGE:
        raise ValueError("no-LR-ASD pilot requires an isolated MiMo output root")
    case_manifest = MimoCaseManifest.model_validate_json(
        arguments.case_manifest.read_text(encoding="utf-8")
    )
    stem_inventory, stem_records, _ = load_stem_source(separation)
    diarization_provenance, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=shadow,
    )
    if diarization_provenance.binding_evidence_mode != arguments.binding_evidence_mode:
        raise ValueError("selected binding mode differs from DiariZen source")
    _validate_stage_closure(
        case_manifest=case_manifest,
        stem_inventory=stem_inventory,
        separation_root=separation,
        diarization_provenance=diarization_provenance,
    )
    builder = build_mimo25_reference_inventory if arguments.binding_evidence_mode == "none" else build_mimo25_inventory
    base = builder(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base, diarization_provenance),
        stem_diarization_root=diarization,
        stem_asr_root=asr,
        route=route,
        binding_evidence_mode=arguments.binding_evidence_mode,
    )
    selected_ids = set(case_manifest.clip_uids)
    skipped = [
        item for item in separation_skips(
            inventory=stem_inventory,
            records=stem_records,
            route=route,
        )
        if item.clip_uid in selected_ids
    ]
    diarization_failed = [
        item for item in diarization_provenance.diarization_failed_clips
        if item.clip_uid in selected_ids
    ]
    selected = selected_stem_records(
        [record for record in stem_records if record.clip_uid in {job.clip_uid for job in jobs}],
        route=route, allow_unverified=arguments.allow_unverified,
    )
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "model_called": False,
        "output_root": str(output),
        "clip_count": len(jobs),
        "binding_evidence_mode": arguments.binding_evidence_mode,
        "clip_uids": [item.clip_uid for item in jobs],
        "temperature": arguments.temperature,
        "thinking": arguments.thinking,
        "icl": arguments.icl,
        "original_target_av_is_highest_authority": True,
    }
    if not arguments.dry_run:
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
                temperature=arguments.temperature,
                thinking=arguments.thinking,
                icl=arguments.icl,
                max_completion_tokens=arguments.max_completion_tokens,
            ),
            stem_records_by_clip={record.clip_uid: record for record in selected},
        )
        summary = run_mimo25_stem_reconcile_shadow(
            jobs=jobs,
            stem_records=stem_records,
            backend=backend,
            output_root=output,
            source_clip_uids=case_manifest.clip_uids,
            binding_evidence_mode=arguments.binding_evidence_mode,
            skipped_clips=skipped,
            diarization_failed_clips=diarization_failed,
            route=route,
            allow_unverified=arguments.allow_unverified,
            overwrite=arguments.overwrite,
        )
        result.update(model_called=True, summary=summary.model_dump(mode="json"))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
