"""One Qwen instance per partition; per-case durable preparation commits."""

from dataclasses import asdict
from pathlib import Path

from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter

from .h3_pair_state import (
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
from .timeline import inspect_video_timeline

# H3 pads any shorter clip up to 124 frames @24fps, so the tail would carry no
# real source motion. Anything below this is a durable input skip, not a failure.
MINIMUM_SOURCE_DURATION_SECONDS = 5.0
SOURCE_SUBJECT_FIELDS = ("source_subject_1","source_subject_2")
PERFORMANCE_FIELDS = ("source_performance_1","source_performance_2")
QWEN_FIELDS = (*SOURCE_SUBJECT_FIELDS,*PERFORMANCE_FIELDS,"shot_description")


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
                    (subject1, subject2, performance1, performance2,
                     shot) = qwen.describe_two_with_performance(video)
                    replacement1, replacement2 = qwen.invent_two(subject1,subject2)
                    texts = {"source_subject_1":subject1,"source_subject_2":subject2,
                             "source_performance_1":performance1,"source_performance_2":performance2,
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
                        "frames":plan.native_frame_count,"width":width,"height":height,
                        "timeline":asdict(plan),"aspect_ratio":aspect,**audio,**texts}
                    if config.get("variant") == "frame0":
                        # Qwen only: the Boogu frame0 phase owns the final prompt
                        # and the prepared.json commit marker. The marker does not
                        # advance the global phase, so stop this case here.
                        publish_qwen(case,payload,texts)
                        break
                    else:
                        prompt = build_prompt(subject1,subject2,shot,replacement1,replacement2,
                                              performance1=performance1,performance2=performance2)
                        publish_prepared(case,{**payload,"prompt":prompt,"variant":"text_two_person"},
                                         {**texts,"h3_prompt":prompt})
                except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                    fail_attempt(case,attempt,exc)
    finally:
        if qwen is not None:
            qwen.close()
