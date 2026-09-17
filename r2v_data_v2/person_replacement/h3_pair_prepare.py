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
)
from .h3_ref2va import match_aspect_ratio, output_size, plan_h3_timeline
from .h3_two_person import build_prompt
from .qwen3vl import LocalQwen
from .timeline import inspect_video_timeline


def prepare_partition(config, worker, qwen_factory=None):
    cases = partition(config["cases"],worker)
    limits = config["limits"]
    if not any(eligible(case,"prepare",limits[case["case_id"]]["prepare"]) for case in cases):
        return
    factory = qwen_factory or (lambda path:LocalQwen(path,max_new_tokens=1024))
    qwen = factory(config["qwen_model"])
    root = Path(config["clips_root"])
    resolver = JeaVideoMotionAdapter(clips_root=root,source_videos_root=root)
    try:
        qwen._load()  # infrastructure failure is fatal, unlike individual bad inputs/parses
        for case in cases:
            while eligible(case,"prepare",limits[case["case_id"]]["prepare"]):
                attempt = begin_attempt(case,"prepare",{"pair_id":config["pair_id"],"qwen_worker":worker})
                try:
                    video, _ = resolver.resolve_clip_path(case["row"])
                    if video.suffix.lower() != ".mp4":
                        raise ValueError("Processed clip must be MP4")
                    source = inspect_video_timeline(video)
                    plan = plan_h3_timeline(source)
                    aspect = match_aspect_ratio(source)
                    subject1, subject2, shot = qwen.describe_two(video)
                    replacement1, replacement2 = qwen.invent_two(subject1,subject2)
                    prompt = build_prompt(subject1,subject2,shot,replacement1,replacement2)
                    texts = {"source_subject_1":subject1,"source_subject_2":subject2,"shot_description":shot,
                             "replacement_subject_1":replacement1,"replacement_subject_2":replacement2,
                             "h3_prompt":prompt}
                    original = Path(case["directory"])/"original.mp4"
                    if original.is_symlink() or original.exists():
                        if not original.is_symlink() or original.resolve() != video.resolve():
                            raise ValueError("Existing original symlink conflicts with input")
                    else:
                        original.symlink_to(video)
                    width,height = output_size(aspect)
                    publish_prepared(case,{"case_id":case["case_id"],"row_sha256":case["row_sha256"],
                        "identity":config["identity"],"source":str(video),"prompt":prompt,"seed":config["seed"],
                        "frames":plan.native_frame_count,"width":width,"height":height,
                        "timeline":asdict(plan),"aspect_ratio":aspect,**texts},texts)
                except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                    fail_attempt(case,attempt,exc)
    finally:
        qwen.close()
