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

from r2v_data_v2.h3.jea_audio_production import (
    JEAOccurrenceEmbedding,
    build_jea_pairs,
    jea_production_paths,
)
from r2v_data_v2.h3.jea_final_renderer import render_jea_final_samples
from r2v_data_v2.h3.qwen3_asr import (
    QWEN3_ASR_MODEL_IDENTIFIER,
    Qwen3ASRBackend,
    Qwen3ASRConfiguration,
    run_qwen3_asr,
)
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory

_STAGES = ("pair", "qwen3-asr", "h3")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finish readable JEA Audio Production with Qwen3 ASR",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(_STAGES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_stages(value: str) -> tuple[str, ...]:
    stages = tuple(
        dict.fromkeys(part.strip() for part in value.split(",") if part.strip())
    )
    unknown = set(stages) - set(_STAGES)
    if not stages or unknown:
        raise ValueError(f"invalid JEA production stages: {sorted(unknown)}")
    return tuple(stage for stage in _STAGES if stage in stages)


def _read_occurrences(path: Path) -> list[JEAOccurrenceEmbedding]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            JEAOccurrenceEmbedding.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    stages = _parse_stages(arguments.stages)
    visual = load_visual_production_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
    )
    paths = jea_production_paths(arguments.audio_production_root)
    plan: dict[str, object] = {
        "visual_production_root": visual.visual_production_root,
        "visual_runs_root": visual.visual_runs_root,
        "audio_production_root": str(paths.root),
        "canonical_sample_count": visual.canonical_sample_count,
        "eligible_subject_occurrence_count": visual.eligible_subject_occurrence_count,
        "media_collection_count": visual.media_collection_count,
        "media_collection_clip_counts": visual.media_collection_clip_counts,
        "shard_count": visual.shard_count,
        "selected_asr_model": QWEN3_ASR_MODEL_IDENTIFIER,
        "requested_stages": list(stages),
        "bounded_limit_applied": False,
        "quota_applied": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan

    stage_results: dict[str, object] = {}
    if "pair" in stages:
        occurrences_path = (paths.embedding / "occurrences.jsonl").resolve(strict=True)
        stage_results["pair"] = build_jea_pairs(
            visual_inventory=visual,
            occurrences=_read_occurrences(occurrences_path),
            audio_root=paths.audio,
            output_root=paths.pairs,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "qwen3-asr" in stages:
        if not os.environ.get("QWEN3_ASR_ENV"):
            raise ValueError("QWEN3_ASR_ENV is required for the active Qwen3 stage")
        configuration = Qwen3ASRConfiguration.from_environment()
        stage_results["qwen3-asr"] = run_qwen3_asr(
            visual_inventory=visual,
            diarization_root=paths.diarization,
            output_root=paths.asr,
            backend=Qwen3ASRBackend(configuration),
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    if "h3" in stages:
        stage_results["h3"] = render_jea_final_samples(
            visual_inventory=visual,
            pairs_root=paths.pairs,
            diarization_root=paths.diarization,
            qwen3_asr_root=paths.asr,
            output_root=paths.h3,
            overwrite=arguments.overwrite,
        ).model_dump(mode="json")
    result = {**plan, "stage_results": stage_results}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
