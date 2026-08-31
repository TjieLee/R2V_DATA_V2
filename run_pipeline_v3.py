from __future__ import annotations

import argparse
import json
import subprocess
import threading
from contextlib import ExitStack, nullcontext
from pathlib import Path

from r2v_data_v2.v3.annotation import AnnotationClient, annotate_clips
from r2v_data_v2.v3.background import build_background_candidates
from r2v_data_v2.v3.background_final_guard import FinalBackgroundJudge
from r2v_data_v2.v3.config import V3Config, load_config
from r2v_data_v2.v3.cross_pair_judge import CrossPairJudge
from r2v_data_v2.v3.frames import FrameDecoder, sample_frames
from r2v_data_v2.v3.instruction import InstructionClient, instruct_clips
from r2v_data_v2.v3.manifest import build_manifest
from r2v_data_v2.v3.pair import pair_clips
from r2v_data_v2.v3.profiling import (
    QwenConcurrencyGate,
    V3Profiler,
    active_profiler,
    profile_stage,
    qwen_concurrency_gate,
)
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
from r2v_data_v2.v3.runtime import (
    ClipScopedStorage,
    PersistentStageProcess,
    PersistentStageProcessPool,
    StreamingDAGScheduler,
    StreamingStage,
    runtime_worker_config,
    runtime_segment_pool_worker_configs,
    streaming_model_stage_enabled,
)
from r2v_data_v2.v3.sam3_backend import SegmentationBackend
from r2v_data_v2.v3.segment import segment_clips
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage, evaluate_export_state
from r2v_data_v2.v3.subject_attributes import (
    AttributeCompletionBackend,
    AttributeCompletionJudge,
    AttributeFrameSegmentationBackend,
    AttributeGmeScreener,
    PersistentWorkerAttributeFrameSegmenter,
    QwenSubjectAttributeCompletionJudge,
    QwenSubjectAttributeClient,
    SubjectAttributeDiscoveryClient,
    SubjectAttributeReviewClient,
    process_subject_attribute_clip,
    reconcile_subject_attribute_outputs,
)

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
    "subject_attributes",
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
        "subject_attributes",
        "remove",
        "export",
    }
)

_STREAMING_GPU_STAGES = frozenset({"segment", "remove", "reference_edit"})
_SUBJECT_ATTRIBUTE_SEGMENT_WORKER = "subject_attributes_segment"


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


def _streaming_stage_handler(
    *,
    stage: str,
    config: V3Config,
    storage: RunStorage,
    overwrite: bool,
    shared_write_lock: threading.Lock,
    process: PersistentStageProcess | PersistentStageProcessPool | None,
    annotation_client: AnnotationClient | None,
    frame_decoder: FrameDecoder | None,
    segmentation_backend: SegmentationBackend | None,
    instruction_client: InstructionClient | None,
    background_removal_backend: BackgroundRemovalBackend | None,
    background_removal_judge: BackgroundRemovalJudge | None,
    entity_reference_judge: EntityReferenceJudge | None,
    cross_pair_judge: CrossPairJudge | None,
    background_final_judge: FinalBackgroundJudge | None,
    reference_completion_backend: QwenReferenceCompletionBackend | None,
    reference_completion_judge: QwenLocalizedCompletionJudge | None,
    reference_edit_backend: BooguReferenceEditBackend | None,
    reference_edit_judge: BooguReferenceEditJudge | None,
    reference_edit_sam_reviewer: BooguSamReviewer | None,
    reference_integrity_judge: ReferenceIntegrityJudge | None,
    subject_attribute_discovery_client: SubjectAttributeDiscoveryClient | None,
    subject_attribute_review_client: SubjectAttributeReviewClient | None,
    attribute_segmentation_backend: AttributeFrameSegmentationBackend | None,
    attribute_gme_screener: AttributeGmeScreener | None,
    attribute_completion_backend: AttributeCompletionBackend | None,
    attribute_completion_judge: AttributeCompletionJudge | None,
) -> object:
    if process is not None:
        return process.request

    def run(clip_uid: str) -> dict[str, object]:
        scoped = ClipScopedStorage(
            storage,
            clip_uid,
            shared_write_lock=shared_write_lock,
        )
        if stage == "annotate":
            stats = annotate_clips(
                config,
                scoped,
                overwrite=overwrite,
                client=annotation_client,
            )
        elif stage == "frames":
            stats = sample_frames(
                config,
                scoped,
                overwrite=overwrite,
                decoder=frame_decoder,
            )
        elif stage == "segment":
            stats = segment_clips(
                config,
                scoped,
                overwrite=overwrite,
                backend=segmentation_backend,
            )
        elif stage == "rank":
            stats = rank_temporal_coverage(
                config,
                scoped,
                overwrite=overwrite,
            )
        elif stage == "background":
            stats = build_background_candidates(
                config,
                scoped,
                overwrite=overwrite,
            )
        elif stage == "remove":
            stats = remove_backgrounds(
                config,
                scoped,
                overwrite=overwrite,
                backend=background_removal_backend,
                judge=background_removal_judge,
            )
        elif stage == "pair":
            stats = pair_clips(
                config,
                scoped,
                overwrite=overwrite,
                judge=entity_reference_judge,
                cross_pair_judge=cross_pair_judge,
                completion_backend=reference_completion_backend,
                completion_judge=reference_completion_judge,
                completion_segmentation_backend=segmentation_backend,
                background_final_judge=background_final_judge,
            )
        elif stage == "reference_edit":
            stats = reference_edit_clips(
                config,
                scoped,
                overwrite=overwrite,
                backend=reference_edit_backend,
                judge=reference_edit_judge,
                sam_reviewer=reference_edit_sam_reviewer,
                manage_backend_lifecycle=False,
            )
        elif stage == "reference_integrity":
            stats = reference_integrity_clips(
                config,
                scoped,
                overwrite=overwrite,
                judge=reference_integrity_judge,
            )
        elif stage == "subject_attributes":
            if (
                subject_attribute_discovery_client is None
                or subject_attribute_review_client is None
                or attribute_segmentation_backend is None
            ):
                raise RuntimeError(
                    "subject attribute streaming clients are unavailable"
                )
            stored_clip = storage.read_clip(clip_uid)
            visual_export = evaluate_export_state(
                stored_clip,
                require_reference_edit=config.reference_edit.enabled,
                require_reference_integrity=config.reference_integrity.enabled,
            )
            accepted_clip = stored_clip.model_copy(update={"export": visual_export})
            return process_subject_attribute_clip(
                config,
                storage=storage,
                output_root=storage.root / "subject_attributes",
                clip=accepted_clip,
                discovery_client=subject_attribute_discovery_client,
                review_client=subject_attribute_review_client,
                segmentation_backend=attribute_segmentation_backend,
                gme_screener=attribute_gme_screener,
                completion_backend=attribute_completion_backend,
                completion_judge=attribute_completion_judge,
                overwrite=overwrite,
            ).to_counts()
        else:
            stats = instruct_clips(
                config,
                scoped,
                overwrite=overwrite,
                client=instruction_client,
            )
        return stats.to_dict()

    return run


def _run_streaming_pipeline(
    *,
    config_path: str | Path,
    config: V3Config,
    storage: RunStorage,
    ordered_stages: tuple[str, ...],
    overwrite: bool,
    profiler: V3Profiler | None,
    results: dict[str, object],
    annotation_client: AnnotationClient | None,
    frame_decoder: FrameDecoder | None,
    segmentation_backend: SegmentationBackend | None,
    instruction_client: InstructionClient | None,
    background_removal_backend: BackgroundRemovalBackend | None,
    background_removal_judge: BackgroundRemovalJudge | None,
    entity_reference_judge: EntityReferenceJudge | None,
    cross_pair_judge: CrossPairJudge | None,
    background_final_judge: FinalBackgroundJudge | None,
    reference_completion_backend: QwenReferenceCompletionBackend | None,
    reference_completion_judge: QwenLocalizedCompletionJudge | None,
    reference_edit_backend: BooguReferenceEditBackend | None,
    reference_edit_judge: BooguReferenceEditJudge | None,
    reference_edit_sam_reviewer: BooguSamReviewer | None,
    reference_integrity_judge: ReferenceIntegrityJudge | None,
    subject_attribute_discovery_client: SubjectAttributeDiscoveryClient | None,
    subject_attribute_review_client: SubjectAttributeReviewClient | None,
    attribute_segmentation_backend: AttributeFrameSegmentationBackend | None,
    attribute_gme_screener: AttributeGmeScreener | None,
    attribute_completion_backend: AttributeCompletionBackend | None = None,
    attribute_completion_judge: AttributeCompletionJudge | None = None,
    profile: bool = False,
) -> dict[str, object]:
    if "manifest" in ordered_stages:
        with profile_stage("manifest") as stage_profile:
            results["manifest"] = build_manifest(config, storage).to_dict()
            stage_profile.set_counters(results["manifest"])
    deferred_subject_attributes = (
        config.runtime.subject_attributes_deferred
        and "subject_attributes" in ordered_stages
    )
    if deferred_subject_attributes:
        results["subject_attributes"] = {"deferred": True}
        results["deferred_stages"] = ["subject_attributes"]
    clip_stages = [
        stage
        for stage in ordered_stages
        if stage not in {"manifest", "export"}
        and not (stage == "subject_attributes" and deferred_subject_attributes)
    ]
    clip_uids = [clip.clip_uid for clip in storage.iter_clips()]
    if clip_stages and not clip_uids:
        raise FileNotFoundError(
            "streaming stages require manifest to create clip.json records first"
        )
    shared_write_lock = threading.Lock()
    processes: dict[
        str,
        PersistentStageProcess | PersistentStageProcessPool,
    ] = {}
    segment_pool: PersistentStageProcessPool | None = None
    with ExitStack() as stack:
        if "reference_edit" in clip_stages and reference_edit_backend is not None:
            starter = getattr(reference_edit_backend, "start", None)
            started = getattr(reference_edit_backend, "started", False)
            if callable(starter) and not started:
                starter(stderr_log_path=storage.reference_edit_worker_log_path())
                closer = getattr(reference_edit_backend, "close", None)
                if callable(closer):
                    stack.callback(closer)
        worker_stages = [
            stage for stage in clip_stages if stage in _STREAMING_GPU_STAGES
        ]
        if (
            "subject_attributes" in clip_stages
            and config.subject_attributes.completion.enabled
            and attribute_completion_backend is None
            and (
                config.runtime.gpu_workers.reference_edit is not None
                or config.runtime.gpu_workers.subject_attributes_completion
                is not None
            )
            and "reference_edit" not in worker_stages
        ):
            worker_stages.append("reference_edit")
        dedicated_attribute_segment = (
            "subject_attributes" in clip_stages
            and attribute_segmentation_backend is None
            and config.runtime.gpu_workers.subject_attributes_segment is not None
            and not config.runtime.gpu_workers.segment_pool
        )
        if (
            "subject_attributes" in clip_stages
            and attribute_segmentation_backend is None
            and not dedicated_attribute_segment
            and segmentation_backend is None
            and "segment" not in worker_stages
        ):
            worker_stages.append("segment")
        for stage in worker_stages:
            injected = {
                "segment": segmentation_backend,
                "remove": background_removal_backend,
                "reference_edit": reference_edit_backend,
            }.get(stage)
            if (
                stage not in _STREAMING_GPU_STAGES
                or injected is not None
                or (
                    not streaming_model_stage_enabled(config, stage)
                    and not (
                        stage == "reference_edit"
                        and "subject_attributes" in clip_stages
                    )
                )
            ):
                continue
            if stage == "segment" and config.runtime.gpu_workers.segment_pool:
                workers = [
                    PersistentStageProcess(worker_config)
                    for worker_config in runtime_segment_pool_worker_configs(
                        config,
                        config_path=config_path,
                        overwrite=overwrite,
                        profile=profile,
                    )
                ]
                segment_pool = PersistentStageProcessPool(workers)
                segment_pool.start()
                processes[stage] = segment_pool
                stack.callback(segment_pool.close)
                continue
            worker = PersistentStageProcess(
                runtime_worker_config(
                    config,
                    config_path=config_path,
                    stage=stage,
                    gpu_assignment=(
                        "subject_attributes_completion"
                        if (
                            stage == "reference_edit"
                            and not config.reference_edit.enabled
                            and config.runtime.gpu_workers.subject_attributes_completion
                            is not None
                        )
                        else None
                    ),
                    overwrite=overwrite,
                    profile=profile,
                )
            )
            worker.start()
            process_key = (
                "attribute_completion"
                if stage == "reference_edit" and not config.reference_edit.enabled
                else stage
            )
            processes[process_key] = worker
            stack.callback(worker.close)
        if dedicated_attribute_segment:
            worker = PersistentStageProcess(
                runtime_worker_config(
                    config,
                    config_path=config_path,
                    stage="segment",
                    gpu_assignment=_SUBJECT_ATTRIBUTE_SEGMENT_WORKER,
                    overwrite=overwrite,
                    profile=profile,
                )
            )
            worker.start()
            processes[_SUBJECT_ATTRIBUTE_SEGMENT_WORKER] = worker
            stack.callback(worker.close)
        attribute_inference_lock: threading.Lock | None = None
        owned_attribute_qwen: QwenSubjectAttributeClient | None = None
        owned_completion_judge: QwenSubjectAttributeCompletionJudge | None = None
        if "subject_attributes" in clip_stages:
            if (
                subject_attribute_discovery_client is None
                or subject_attribute_review_client is None
            ):
                service = config.qwen.candidate_judge
                if service is None:
                    raise ValueError(
                        "qwen.candidate_judge is required for subject attributes"
                    )
                owned_attribute_qwen = QwenSubjectAttributeClient(service)
                stack.callback(owned_attribute_qwen.close)
                subject_attribute_discovery_client = (
                    subject_attribute_discovery_client or owned_attribute_qwen
                )
                subject_attribute_review_client = (
                    subject_attribute_review_client or owned_attribute_qwen
                )
            if attribute_segmentation_backend is None:
                segment_process = processes.get(
                    _SUBJECT_ATTRIBUTE_SEGMENT_WORKER
                ) or processes.get("segment")
                if segment_process is None:
                    raise ValueError(
                        "subject attributes require the persistent segment worker "
                        "or an injected attribute segmentation backend"
                    )
                attribute_segmentation_backend = (
                    PersistentWorkerAttributeFrameSegmenter(
                        storage,
                        segment_process,
                        inference_lock=attribute_inference_lock,
                    )
                )
            if (
                config.subject_attributes.completion.enabled
                and attribute_completion_backend is None
            ):
                completion_process = processes.get(
                    "attribute_completion"
                ) or processes.get("reference_edit")
                if isinstance(completion_process, PersistentStageProcessPool):
                    raise ValueError(
                        "attribute completion requires persistent Boogu worker"
                    )
                if completion_process is not None:
                    attribute_completion_backend = completion_process
                elif config.subject_attributes.completion.enabled:
                    raise ValueError(
                        "attribute completion requires persistent Boogu worker"
                    )
            if config.subject_attributes.completion.enabled:
                if attribute_completion_judge is None:
                    service = config.qwen.candidate_judge
                    assert service is not None
                    owned_completion_judge = QwenSubjectAttributeCompletionJudge(
                        service,
                        completion_component="qwen_attribute_completion_review",
                    )
                    stack.callback(owned_completion_judge.close)
                    attribute_completion_judge = owned_completion_judge
        stages: list[StreamingStage] = []
        for stage in clip_stages:
            workers = getattr(config.runtime.stage_workers, stage)
            stages.append(
                StreamingStage(
                    name=stage,
                    workers=workers,
                    resource=("gpu" if stage in _STREAMING_GPU_STAGES else "cpu"),
                    handler=_streaming_stage_handler(
                        stage=stage,
                        config=config,
                        storage=storage,
                        overwrite=overwrite,
                        shared_write_lock=shared_write_lock,
                        process=processes.get(stage),
                        annotation_client=annotation_client,
                        frame_decoder=frame_decoder,
                        segmentation_backend=segmentation_backend,
                        instruction_client=instruction_client,
                        background_removal_backend=background_removal_backend,
                        background_removal_judge=background_removal_judge,
                        entity_reference_judge=entity_reference_judge,
                        cross_pair_judge=cross_pair_judge,
                        background_final_judge=background_final_judge,
                        reference_completion_backend=reference_completion_backend,
                        reference_completion_judge=reference_completion_judge,
                        reference_edit_backend=reference_edit_backend,
                        reference_edit_judge=reference_edit_judge,
                        reference_edit_sam_reviewer=reference_edit_sam_reviewer,
                        reference_integrity_judge=reference_integrity_judge,
                        subject_attribute_discovery_client=(
                            subject_attribute_discovery_client
                        ),
                        subject_attribute_review_client=(
                            subject_attribute_review_client
                        ),
                        attribute_segmentation_backend=(
                            attribute_segmentation_backend
                        ),
                        attribute_gme_screener=attribute_gme_screener,
                        attribute_completion_backend=attribute_completion_backend,
                        attribute_completion_judge=attribute_completion_judge,
                    ),
                )
            )
        if stages:
            runtime_result = StreamingDAGScheduler(
                stages,
                cpu_workers=config.runtime.cpu_workers,
                profiler=profiler,
            ).run(clip_uids)
            if segment_pool is not None and "segment" in runtime_result.stage_counts:
                runtime_result.stage_counts["segment"].update(
                    segment_pool.performance_counters()
                )
            for stage in clip_stages:
                counts = runtime_result.stage_counts[stage]
                integer_counts = {
                    name: value
                    for name, value in counts.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
                storage.update_stage_counts(stage, integer_counts)
                results[stage] = counts
            for failure in runtime_result.failed_tasks:
                storage.append_failure(
                    stage=f"runtime_{failure['stage']}",
                    clip_uid=failure["clip_uid"],
                    reason=failure["reason"],
                    details={"exception_type": failure["error_type"]},
                )
            results["runtime"] = runtime_result.to_dict()
            if "subject_attributes" in clip_stages:
                subject_counts = runtime_result.stage_counts["subject_attributes"]
                results["subject_attributes_summary"] = (
                    reconcile_subject_attribute_outputs(
                        storage=storage,
                        output_root=storage.root / "subject_attributes",
                        owner_limit=None,
                        invocation_wall_time_seconds=0.0,
                        skipped_existing_owners=int(
                            subject_counts.get("skipped_existing_owners", 0)
                        ),
                    )
                )
    if "export" in ordered_stages:
        with profile_stage("export") as stage_profile:
            dataset = DatasetExporter(config, storage).export(overwrite=overwrite)
            results["export"] = dataset.model_dump(mode="json")
            stage_profile.set_counters(results["export"])
    results["completed_stages"] = [
        stage
        for stage in ordered_stages
        if not (stage == "subject_attributes" and deferred_subject_attributes)
    ]
    return results


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
    subject_attribute_discovery_client: SubjectAttributeDiscoveryClient | None = None,
    subject_attribute_review_client: SubjectAttributeReviewClient | None = None,
    attribute_segmentation_backend: AttributeFrameSegmentationBackend | None = None,
    attribute_gme_screener: AttributeGmeScreener | None = None,
    attribute_completion_backend: AttributeCompletionBackend | None = None,
    attribute_completion_judge: AttributeCompletionJudge | None = None,
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
            "instruct, subject_attributes, and export only; "
            f"unimplemented stages requested: {unavailable}"
        )
    requested = set(stages)
    ordered_stages = tuple(stage for stage in STAGE_ORDER if stage in requested)
    config = load_config(config_path)
    if (
        "subject_attributes" in ordered_stages
        and config.runtime.mode != "streaming_v1"
    ):
        raise ValueError("subject_attributes is available only in streaming_v1")
    storage = RunStorage(config)
    run = storage.initialize(git_commit=git_commit or _git_commit())
    profiler = (
        V3Profiler(
            storage.root,
            git_commit=run.git_commit,
            qwen_max_inflight=getattr(
                getattr(config, "runtime", None),
                "qwen_max_inflight",
                None,
            ),
        )
        if profile
        else None
    )
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
            runtime_mode = getattr(
                getattr(config, "runtime", None),
                "mode",
                "staged_legacy",
            )
            if runtime_mode == "streaming_v1":
                gate = QwenConcurrencyGate(
                    config.runtime.qwen_max_inflight,
                    lock_directory=storage.root / "profiling" / "qwen_slots",
                )
                with qwen_concurrency_gate(gate):
                    return _run_streaming_pipeline(
                        config_path=config_path,
                        config=config,
                        storage=storage,
                        ordered_stages=ordered_stages,
                        overwrite=overwrite,
                        profiler=profiler,
                        results=results,
                        annotation_client=annotation_client,
                        frame_decoder=frame_decoder,
                        segmentation_backend=segmentation_backend,
                        instruction_client=instruction_client,
                        background_removal_backend=background_removal_backend,
                        background_removal_judge=background_removal_judge,
                        entity_reference_judge=entity_reference_judge,
                        cross_pair_judge=cross_pair_judge,
                        background_final_judge=background_final_judge,
                        reference_completion_backend=reference_completion_backend,
                        reference_completion_judge=reference_completion_judge,
                        reference_edit_backend=reference_edit_backend,
                        reference_edit_judge=reference_edit_judge,
                        reference_edit_sam_reviewer=reference_edit_sam_reviewer,
                        reference_integrity_judge=reference_integrity_judge,
                        subject_attribute_discovery_client=(
                            subject_attribute_discovery_client
                        ),
                        subject_attribute_review_client=(
                            subject_attribute_review_client
                        ),
                        attribute_segmentation_backend=(
                            attribute_segmentation_backend
                        ),
                        attribute_gme_screener=attribute_gme_screener,
                        attribute_completion_backend=attribute_completion_backend,
                        attribute_completion_judge=attribute_completion_judge,
                        profile=profile,
                    )
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
            "reference_integrity, subject_attributes, "
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
