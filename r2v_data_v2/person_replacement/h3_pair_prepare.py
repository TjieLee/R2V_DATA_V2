"""One model instance per partition; per-case durable preparation commits.

The text V22 path uses one resident Qwen3.5-27B writer for two calls per case.
The frame0 experiment keeps the older 8B + Boogu path.
"""

from dataclasses import asdict
from pathlib import Path

from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter

from .h3_pair_diversity import diversity_cues
from .h3_pair_state import (
    atomic_bytes,
    begin_attempt,
    eligible,
    fail_attempt,
    partition,
    publish_prepared,
    publish_qwen,
    publish_skipped,
    qwen_prepared,
)
from .h3_ref2va import match_aspect_ratio, output_size, plan_h3_timeline
from .h3_reference_audio import ensure_h3_reference
from .h3_two_person import build_prompt
from .qwen3vl import LocalQwen
from .qwen38_h3_prompt_writer import (
    PROMPT_MAX_NEW_TOKENS,
    REPLACEMENT_MAX_NEW_TOKENS,
    VIDEO_FPS,
    LocalQwen38H3PromptWriter,
    OpenAIMimoH3PromptWriter,
    OpenAIQwen38H3PromptWriter,
    validate_h3_prompt_writer_output,
)
from .timeline import inspect_video_timeline

# H3 pads any shorter clip up to 124 frames @24fps, so the tail would carry no
# real source motion. Anything below this is a durable input skip, not a failure.
MINIMUM_SOURCE_DURATION_SECONDS = 5.0
SOURCE_SUBJECT_FIELDS = ("source_subject_1","source_subject_2")
PERFORMANCE_FIELDS = ("source_performance_1","source_performance_2")
QWEN_FIELDS = (*SOURCE_SUBJECT_FIELDS,*PERFORMANCE_FIELDS,"shot_description")
PROMPT_WRITER_CONTRACT = "slim_two_call_h3_prompt_v18"
PROMPT_SOURCE = "qwen35_full_video_fps8_slim"


def skip_payload(case, config, video, reason, *, detail="", timeline=None):
    return {"case_id":case["case_id"],"row_sha256":case["row_sha256"],"identity":config["identity"],
            "source":str(video),"variant":config.get("variant","text"),"reason":reason,
            "minimum_source_duration_seconds":MINIMUM_SOURCE_DURATION_SECONDS,
            "source_timeline":asdict(timeline) if timeline is not None else None,"detail":detail}


def skip_or_plan(case, config, video):
    """Durable ineligibility commit, or the authoritative H3 timeline plan."""
    try:
        source = inspect_video_timeline(video)
    except ValueError as exc:  # deterministic media incompatibility, never retryable
        publish_skipped(case,skip_payload(case,config,video,"h3_timeline_ineligible",detail=str(exc)))
        return None
    if source.duration_seconds < MINIMUM_SOURCE_DURATION_SECONDS:
        publish_skipped(case,skip_payload(case,config,video,"source_duration_below_5s",timeline=source))
        return None
    try:
        return plan_h3_timeline(source)  # still the only H3 timeline authority
    except ValueError as exc:
        publish_skipped(case,skip_payload(case,config,video,"h3_timeline_ineligible",
                                          detail=str(exc),timeline=source))
        return None


def prepare_partition(config, worker, qwen_factory=None):
    cases = partition(config["cases"],worker,config.get("group_size",2))
    limits = config["limits"]
    # frame0: a durable Qwen marker means Qwen already succeeded, so resume
    # skips it and only the Boogu frame0 phase still has to run.
    if config.get("variant") == "frame0":
        cases = [case for case in cases if qwen_prepared(case) is None]
    if not any(eligible(case,"prepare",limits[case["case_id"]]["prepare"]) for case in cases):
        return
    factory = qwen_factory or (lambda path:LocalQwen(path,max_new_tokens=1024))
    qwen = None  # loaded lazily, only once a case survives eligibility screening
    root = Path(config["clips_root"])
    resolver = JeaVideoMotionAdapter(clips_root=root,source_videos_root=root)
    try:
        for case in cases:
            while eligible(case,"prepare",limits[case["case_id"]]["prepare"]):
                attempt = begin_attempt(case,"prepare",{"pair_id":config["pair_id"],"qwen_worker":worker})
                try:
                    video, _ = resolver.resolve_clip_path(case["row"])
                    if video.suffix.lower() != ".mp4":
                        raise ValueError("Processed clip must be MP4")
                    plan = skip_or_plan(case,config,video)  # durable skip or H3 plan
                    if plan is None:
                        break
                    source = plan.source
                    aspect = match_aspect_ratio(source)
                    reference, audio = ensure_h3_reference(video,Path(case["directory"])/"preparation")
                    if qwen is None:
                        qwen = factory(config["qwen_model"])
                        qwen._load()  # infrastructure failure is fatal, unlike bad inputs/parses
                    subject1, subject2, shot = qwen.describe_two(video)
                    cue1, cue2 = diversity_cues(config["seed"],case["row_sha256"])
                    replacement1, replacement2 = qwen.invent_two_with_diversity(
                        subject1,subject2,cue1,cue2)
                    texts = {"source_subject_1":subject1,"source_subject_2":subject2,
                             "shot_description":shot,
                             "replacement_subject_1":replacement1,"replacement_subject_2":replacement2}
                    original = Path(case["directory"])/"original.mp4"
                    if original.is_symlink() or original.exists():
                        if not original.is_symlink() or original.resolve() != video.resolve():
                            raise ValueError("Existing original symlink conflicts with input")
                    else:
                        original.symlink_to(video)
                    width,height = output_size(aspect)
                    payload = {"case_id":case["case_id"],"row_sha256":case["row_sha256"],
                        "identity":config["identity"],"source":str(video),"seed":config["seed"],
                        "h3_reference_source":str(reference),
                        "replacement_diversity_1":cue1,"replacement_diversity_2":cue2,
                        "frames":plan.native_frame_count,"width":width,"height":height,
                        "timeline":asdict(plan),"aspect_ratio":aspect,**audio,**texts}
                    if config.get("variant") == "frame0":
                        # Qwen only: the Boogu frame0 phase owns the final prompt
                        # and the prepared.json commit marker. The marker does not
                        # advance the global phase, so stop this case here.
                        publish_qwen(case,payload,texts)
                        break
                    else:
                        prompt = build_prompt(subject1,subject2,shot,replacement1,replacement2)
                        publish_prepared(case,{**payload,"prompt":prompt,"variant":"text_two_person"},
                                         {**texts,"h3_prompt":prompt})
                except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                    fail_attempt(case,attempt,exc)
    finally:
        if qwen is not None:
            qwen.close()


MAX_WRITER_FRAMES = 120


def writer_frames(duration_seconds):
    """Bounded 8fps sampling budget; the processor does the actual sampling."""
    return max(1,min(MAX_WRITER_FRAMES,round(float(duration_seconds)*VIDEO_FPS)))


def prepare_text_partition_v19(config, worker, writer_factory=None):
    """Text V22: one resident Qwen3.5-27B writer, Call 1 then Call 2 per case."""
    cases = partition(config["cases"],worker,config.get("group_size",2))
    limits = config["limits"]
    if not any(eligible(case,"prepare",limits[case["case_id"]]["prepare"]) for case in cases):
        return
    if writer_factory is not None:
        factory = writer_factory
    elif config.get("prompt_writer_backend","local") == "mimo":
        factory = lambda path:OpenAIMimoH3PromptWriter(
            path,
            base_url=config["prompt_writer_base_url"],
            served_model=config["prompt_writer_served_model"],
        )
    elif config.get("prompt_writer_backend","local") == "sglang":
        factory = lambda path:OpenAIQwen38H3PromptWriter(
            path,
            base_url=config["prompt_writer_base_url"],
            served_model=config["prompt_writer_served_model"],
        )
    else:
        factory = lambda path:LocalQwen38H3PromptWriter(path)
    writer = None  # loaded lazily and exactly once per worker partition
    root = Path(config["clips_root"])
    resolver = JeaVideoMotionAdapter(clips_root=root,source_videos_root=root)
    try:
        for case in cases:
            while eligible(case,"prepare",limits[case["case_id"]]["prepare"]):
                attempt = begin_attempt(case,"prepare",{"pair_id":config["pair_id"],"text_worker":worker})
                try:
                    video, _ = resolver.resolve_clip_path(case["row"])
                    if video.suffix.lower() != ".mp4":
                        raise ValueError("Processed clip must be MP4")
                    plan = skip_or_plan(case,config,video)
                    if plan is None:
                        break
                    source = plan.source
                    aspect = match_aspect_ratio(source)
                    reference, audio = ensure_h3_reference(video,Path(case["directory"])/"preparation")
                    width,height = output_size(aspect)
                    base = {"case_id":case["case_id"],"row_sha256":case["row_sha256"],
                            "identity":config["identity"],"source":str(video),"seed":config["seed"],
                            "h3_reference_source":str(reference),
                            "frames":plan.native_frame_count,"width":width,"height":height,
                            "timeline":asdict(plan),"aspect_ratio":aspect,**audio}
                    # Bounded 8fps sampling for both calls of this case.
                    num_frames = writer_frames(plan.source.duration_seconds)
                    marker = qwen_prepared(case)
                    if marker is None:
                        if writer is None:
                            writer = factory(config["prompt_writer_model"])
                            writer._load()
                        cue1, cue2 = diversity_cues(config["seed"],case["row_sha256"])
                        (performer1, performer2,
                         replacement1, replacement2) = writer.invent_replacements(
                            video,cue1,cue2,num_frames=num_frames)
                        texts = {"source_performer_1":performer1,"source_performer_2":performer2,
                                 "replacement_subject_1":replacement1,
                                 "replacement_subject_2":replacement2}
                        publish_qwen(case,{**base,"replacement_diversity_1":cue1,
                                           "replacement_diversity_2":cue2,**texts},texts)
                        marker = qwen_prepared(case)
                    if writer is None:
                        writer = factory(config["prompt_writer_model"])
                        writer._load()
                    prompt = writer.write_h3_prompt(
                        video,marker["source_performer_1"],marker["source_performer_2"],
                        marker["replacement_subject_1"],marker["replacement_subject_2"],
                        num_frames=num_frames)
                    validate_h3_prompt_writer_output(prompt)  # fail closed, no fallback
                    directory = Path(case["directory"])/"preparation"
                    atomic_bytes(directory/"prompt_writer_request.txt",
                                 _writer_request(marker).encode())
                    publish_prepared(case,{**marker,"variant":"text_two_person","prompt":prompt,
                        "prompt_writer_model":str(config["prompt_writer_model"]),
                        "prompt_writer_backend":config.get("prompt_writer_backend","local"),
                        "prompt_writer_base_url":config.get("prompt_writer_base_url"),
                        "prompt_writer_served_model":config.get("prompt_writer_served_model"),
                        "prompt_writer_contract":PROMPT_WRITER_CONTRACT,
                        "prompt_writer_video_fps":(
                            getattr(writer,"video_fps",VIDEO_FPS)
                            if (getattr(writer,"uses_local_frame_sampling",True)
                                or config.get("prompt_writer_backend") == "mimo")
                            else None),
                        "prompt_writer_replacement_max_new_tokens":REPLACEMENT_MAX_NEW_TOKENS,
                        "prompt_writer_prompt_max_new_tokens":PROMPT_MAX_NEW_TOKENS,
                        "prompt_writer_thinking":writer.thinking_disabled,
                        "prompt_writer_num_frames":(
                            num_frames if getattr(writer,"uses_local_frame_sampling",True) else None),
                        "prompt_source":(
                            "mimo25_sglang_full_video_fps8_slim"
                            if config.get("prompt_writer_backend") == "mimo"
                            else ("qwen38_sglang_full_video_slim"
                                  if config.get("prompt_writer_backend") == "sglang"
                                  else PROMPT_SOURCE))},
                        {"h3_prompt":prompt})
                except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                    fail_attempt(case,attempt,exc)
    finally:
        if writer is not None:
            writer.close()


def _writer_request(marker):
    from .qwen38_h3_prompt_writer import H3_PROMPT_USER_TEMPLATE

    return H3_PROMPT_USER_TEMPLATE.format(
        source_performer_1=marker["source_performer_1"],
        replacement_subject_1=marker["replacement_subject_1"],
        source_performer_2=marker["source_performer_2"],
        replacement_subject_2=marker["replacement_subject_2"])
