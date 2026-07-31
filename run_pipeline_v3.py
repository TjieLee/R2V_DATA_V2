from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from r2v_data_v2.v3.annotation import AnnotationClient, annotate_clips
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.frames import FrameDecoder, sample_frames
from r2v_data_v2.v3.instruction import InstructionClient, instruct_clips
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.rank import rank_temporal_coverage
from r2v_data_v2.v3.sam3_backend import SegmentationBackend
from r2v_data_v2.v3.segment import segment_clips
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage

STAGE_ORDER = (
    "manifest",
    "annotate",
    "frames",
    "segment",
    "rank",
    "background",
    "remove",
    "pair",
    "instruct",
    "export",
)
_IMPLEMENTED_STAGES = frozenset(
    {
        "manifest",
        "annotate",
        "frames",
        "segment",
        "rank",
        "background",
        "instruct",
        "export",
    }
)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def run_pipeline_v3(
    *,
    config_path: str | Path,
    stages: tuple[str, ...] = (),
    overwrite: bool = False,
    git_commit: str | None = None,
    annotation_client: AnnotationClient | None = None,
    frame_decoder: FrameDecoder | None = None,
    segmentation_backend: SegmentationBackend | None = None,
    instruction_client: InstructionClient | None = None,
) -> dict[str, object]:
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown V3 pipeline stages: {unknown}")
    unavailable = [stage for stage in stages if stage not in _IMPLEMENTED_STAGES]
    if unavailable:
        raise NotImplementedError(
            "this V3 implementation currently provides manifest, annotate, "
            "frames, segment, rank, background, instruct, and export only; "
            f"unimplemented stages requested: {unavailable}"
        )
    requested = set(stages)
    ordered_stages = tuple(
        stage for stage in STAGE_ORDER if stage in requested
    )
    config = load_config(config_path)
    storage = RunStorage(config)
    run = storage.initialize(git_commit=git_commit or _git_commit())
    results: dict[str, object] = {
        "run": {
            "run_id": run.run_id,
            "run_root": str(storage.root),
            "config_hash": run.config_hash,
        }
    }
    for stage in ordered_stages:
        if stage == "manifest":
            results[stage] = build_manifest(config, storage).to_dict()
        elif stage == "annotate":
            results[stage] = annotate_clips(
                config,
                storage,
                overwrite=overwrite,
                client=annotation_client,
            ).to_dict()
        elif stage == "frames":
            results[stage] = sample_frames(
                config,
                storage,
                overwrite=overwrite,
                decoder=frame_decoder,
            ).to_dict()
        elif stage == "segment":
            results[stage] = segment_clips(
                config,
                storage,
                overwrite=overwrite,
                backend=segmentation_backend,
            ).to_dict()
        elif stage == "rank":
            results[stage] = rank_temporal_coverage(
                config,
                storage,
                overwrite=overwrite,
            ).to_dict()
        elif stage == "background":
            results[stage] = build_background_candidates(
                config,
                storage,
                overwrite=overwrite,
            ).to_dict()
        elif stage == "instruct":
            results[stage] = instruct_clips(
                config,
                storage,
                overwrite=overwrite,
                client=instruction_client,
            ).to_dict()
        else:
            dataset = DatasetExporter(config, storage).export(
                overwrite=overwrite
            )
            results[stage] = dataset.model_dump(mode="json")
    results["completed_stages"] = list(ordered_stages)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize V3 storage or run implemented V3 stages"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        default="",
        help=(
            "comma-separated V3 stages; manifest, annotate, frames, segment, "
            "rank, background, instruct, and export are currently implemented"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing final dataset during export",
    )
    args = parser.parse_args()
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    result = run_pipeline_v3(
        config_path=args.config,
        stages=stages,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
