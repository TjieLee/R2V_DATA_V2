from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from r2v_data_v2.augmentation import augment_references
from r2v_data_v2.background_reference import build_background_references
from r2v_data_v2.config import load_config
from r2v_data_v2.inpainting import run_inpainting
from r2v_data_v2.manifest import build_manifest
from r2v_data_v2.pairing import build_pairs
from r2v_data_v2.qwen_client import annotate_manifest
from r2v_data_v2.ranking import rank_manifest_references
from r2v_data_v2.sam3_backend import extract_manifest_candidates
from r2v_data_v2.video_io import sample_manifest_frames

_STAGE_ORDER = (
    "manifest",
    "qwen",
    "frames",
    "sam",
    "rank",
    "background",
    "inpaint",
    "pair",
    "augment",
)


def run_pipeline(
    *,
    config_path: str,
    stages: tuple[str, ...],
    limit: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    unknown = sorted(set(stages) - set(_STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown pipeline stages: {unknown}")
    config = load_config(config_path)
    results: dict[str, object] = {}
    for stage in _STAGE_ORDER:
        if stage not in stages:
            continue
        if stage == "manifest":
            value = build_manifest(
                config,
                limit=limit,
                start_index=start_index,
                overwrite=overwrite,
            )
        elif stage == "frames":
            value = sample_manifest_frames(config, overwrite=overwrite)
        elif stage == "qwen":
            value = annotate_manifest(config, overwrite=overwrite)
        elif stage == "sam":
            value = extract_manifest_candidates(config, overwrite=overwrite)
        elif stage == "rank":
            value = rank_manifest_references(config, overwrite=overwrite)
        elif stage == "background":
            value = build_background_references(config, overwrite=overwrite)
        elif stage == "inpaint":
            value = run_inpainting(config, overwrite=overwrite)
        elif stage == "pair":
            value = build_pairs(config, overwrite=overwrite)
        else:
            value = augment_references(config, overwrite=overwrite)
        results[stage] = asdict(value)

    annotation = results.get("qwen", {})
    sam = results.get("sam", {})
    ranking = results.get("rank", {})
    background = results.get("background", {})
    inpainting = results.get("inpaint", {})
    pairing = results.get("pair", {})
    processed = 0
    for stage in reversed(_STAGE_ORDER):
        stage_result = results.get(stage)
        if isinstance(stage_result, dict) and "processed" in stage_result:
            processed = stage_result["processed"]
            break
    results["summary"] = {
        "processed": processed,
        "qwen_failed": annotation.get("qwen_failed", 0),
        "no_reference_entity": annotation.get("no_reference_entity", 0),
        "generic_entity_labels": annotation.get("generic_entity_labels", 0),
        "sam_failed": sam.get("sam_failed", 0),
        "no_valid_candidate": ranking.get(
            "no_valid_candidate",
            sam.get("no_valid_candidate", 0),
        ),
        "background_failed": background.get("failed", 0),
        "raw_background_count": background.get("raw_background_count", 0),
        "needs_inpainting_count": background.get(
            "needs_inpainting_count",
            0,
        ),
        "inpainting_repaired": inpainting.get("repaired", 0),
        "inpainting_fallback_to_raw": inpainting.get(
            "fallback_to_raw",
            0,
        ),
        "in_pair_count": pairing.get("in_pair_count", 0),
        "cross_pair_count": pairing.get("cross_pair_count", 0),
        "fallback_count": pairing.get("fallback_count", 0),
        "skipped_no_bindable_reference": pairing.get(
            "skipped_no_bindable_reference",
            0,
        ),
        "skipped_background_only_reference": pairing.get(
            "skipped_background_only_reference",
            0,
        ),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightweight R2V data construction stages in order"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--stages",
        default="manifest,qwen,frames,sam,rank,background,pair",
        help=(
            "comma-separated subset of "
            "manifest,qwen,frames,sam,rank,background,inpaint,pair,augment"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    result = run_pipeline(
        config_path=args.config,
        stages=stages,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
