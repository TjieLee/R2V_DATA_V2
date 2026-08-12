from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

from r2v_data_v2.v3.annotation import AnnotationClient, annotate_clips
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudge
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.cross_pair_judge import CrossPairJudge
from r2v_data_v2.v3.frames import FrameDecoder, sample_frames
from r2v_data_v2.v3.instruction import InstructionClient, instruct_clips
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.pair import pair_clips
from r2v_data_v2.v3.profiling import V3Profiler, active_profiler, profile_stage
from r2v_data_v2.v3.qwen_image_edit_backend import BackgroundRemovalBackend
from r2v_data_v2.v3.rank import rank_temporal_coverage
from r2v_data_v2.v3.reference_completion_qwen import (
    QwenLocalizedCompletionJudge,
    QwenReferenceCompletionBackend,
)
from r2v_data_v2.v3.reference_edit import reference_edit_clips
from r2v_data_v2.v3.reference_edit_boogu import (
    BooguReferenceEditBackend,
    BooguReferenceEditJudge,
    BooguSamReviewer,
)
from r2v_data_v2.v3.reference_integrity import (
    ReferenceIntegrityJudge,
    reference_integrity_clips,
)
from r2v_data_v2.v3.reference_judge import EntityReferenceJudge
from r2v_data_v2.v3.removal_judge import BackgroundRemovalJudge
from r2v_data_v2.v3.remove import remove_backgrounds
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
    "reference_edit",
    "reference_integrity",
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
        "pair",
        "reference_edit",
        "reference_integrity",
        "instruct",
        "remove",
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
    background_removal_backend: BackgroundRemovalBackend | None = None,
    background_removal_judge: BackgroundRemovalJudge | None = None,
    entity_reference_judge: EntityReferenceJudge | None = None,
    cross_pair_judge: CrossPairJudge | None = None,
    background_final_judge: FinalBackgroundJudge | None = None,
    reference_completion_backend: QwenReferenceCompletionBackend | None = None,
    reference_completion_judge: QwenLocalizedCompletionJudge | None = None,
    reference_edit_backend: BooguReferenceEditBackend | None = None,
    reference_edit_judge: BooguReferenceEditJudge | None = None,
    reference_edit_sam_reviewer: BooguSamReviewer | None = None,
    reference_integrity_judge: ReferenceIntegrityJudge | None = None,
    profile: bool = False,
) -> dict[str, object]:
    unknown = sorted(set(stages) - set(STAGE_ORDER))
    if unknown:
        raise ValueError(f"unknown V3 pipeline stages: {unknown}")
    unavailable = [stage for stage in stages if stage not in _IMPLEMENTED_STAGES]
    if unavailable:
        raise NotImplementedError(
            "this V3 implementation currently provides manifest, annotate, "
            "frames, segment, rank, background, remove, pair, reference_edit, "
            "instruct, and export only; "
            f"unimplemented stages requested: {unavailable}"
        )
    requested = set(stages)
    ordered_stages = tuple(stage for stage in STAGE_ORDER if stage in requested)
    config = load_config(config_path)
    storage = RunStorage(config)
    run = storage.initialize(git_commit=git_commit or _git_commit())
    profiler = V3Profiler(storage.root, git_commit=run.git_commit) if profile else None
    results: dict[str, object] = {
        "run": {
            "run_id": run.run_id,
            "run_root": str(storage.root),
            "config_hash": run.config_hash,
        }
    }
    pipeline_error: BaseException | None = None
    profiler_context = active_profiler(profiler) if profiler else nullcontext()
    try:
        with profiler_context:
            for stage in ordered_stages:
                with profile_stage(stage) as stage_profile:
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
                    elif stage == "remove":
                        results[stage] = remove_backgrounds(
                            config,
                            storage,
                            overwrite=overwrite,
                            backend=background_removal_backend,
                            judge=background_removal_judge,
                        ).to_dict()
                    elif stage == "pair":
                        results[stage] = pair_clips(
                            config,
                            storage,
                            overwrite=overwrite,
                            judge=entity_reference_judge,
                            cross_pair_judge=cross_pair_judge,
                            completion_backend=reference_completion_backend,
                            completion_judge=reference_completion_judge,
                            completion_segmentation_backend=segmentation_backend,
                            background_final_judge=background_final_judge,
                        ).to_dict()
                    elif stage == "reference_edit":
                        results[stage] = reference_edit_clips(
                            config,
                            storage,
                            overwrite=overwrite,
                            backend=reference_edit_backend,
                            judge=reference_edit_judge,
                            sam_reviewer=reference_edit_sam_reviewer,
                        ).to_dict()
                    elif stage == "reference_integrity":
                        results[stage] = reference_integrity_clips(
                            config,
                            storage,
                            overwrite=overwrite,
                            judge=reference_integrity_judge,
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
                    stage_profile.set_counters(results[stage])
            results["completed_stages"] = list(ordered_stages)
            return results
    except BaseException as exc:
        pipeline_error = exc
        raise
    finally:
        if profiler is not None:
            try:
                profiler.write_summary()
            except Exception as exc:
                if pipeline_error is None:
                    raise
                pipeline_error.add_note(f"profiling summary write failed: {exc}")


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
            "rank, background, remove, pair, reference_edit, instruct, and "
            "reference_integrity, "
            "export are currently implemented"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing final dataset during export",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="record observational stage and model-call profiling artifacts",
    )
    args = parser.parse_args()
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    result = run_pipeline_v3(
        config_path=args.config,
        stages=stages,
        overwrite=args.overwrite,
        profile=args.profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
