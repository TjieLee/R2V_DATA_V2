"""Small sequential paired pilot: shared descriptions, optional single Boogu frame0 edit."""

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageOps

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.frames import OpenCvFrameDecoder
from r2v_data_v2.v3.reference_edit_boogu import (
    BOOGU_MODEL_NAME,
    BooguSubprocessBackend,
)

from .h3_ref2va import (
    match_aspect_ratio,
    output_size,
    plan_h3_timeline,
    runtime_environment,
)
from .h3_two_person import boogu_prompt, build_prompt
from .pipeline import validate_output_root
from .timeline import inspect_video_timeline

VARIANTS = {"text":"text_two_person", "frame0":"frame0_two_person"}


@contextmanager
def _runtime_environment(directory):
    """Sequential pilot only: restore caller env after local/offline model stage."""
    previous = os.environ.copy()
    environment = runtime_environment(directory)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _edit_frame_zero(video, directory, aspect, descriptions, replacements, config):
    started = time.monotonic()
    width, height = output_size(aspect)
    # Exact decoded index 0, no search or visibility/quality gate. Only the
    # still reference is cover-resized/cropped; H3 receives the original video.
    frame = OpenCvFrameDecoder().decode_indices(video, [0])[0].image
    source = ImageOps.fit(frame.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS,
                          centering=(0.5, 0.5))
    source.save(directory/"source_frame0.png", format="PNG")
    prompt = boogu_prompt(*descriptions[:2], *replacements)
    (directory/"boogu_prompt.txt").write_text(prompt+"\n", encoding="utf-8")
    config = replace(config, temporary_root=directory/"boogu_tmp")
    job = {"model_name":BOOGU_MODEL_NAME, "model_revision":config.model_revision,
           "python":str(config.python_executable), "code_root":str(config.code_root),
           "model_path":str(config.model_path), "cuda_visible_devices":config.cuda_visible_devices,
           "source_frame_index":0, "source_video":str(video),
           "source_image":str(directory/"source_frame0.png"), "width":width, "height":height,
           "instruction":prompt, "seed":42, "thinking_enabled":False,
           "instruction_rewrite_enabled":False, "num_inference_steps":4,
           "text_guidance_scale":1.0, "image_guidance_scale":1.0,
           "use_dmd_student_inference":True, "frame0_transform":"cover_resize_center_crop"}
    write_json_atomic(directory/"boogu_job.json", job)
    cache = directory/"boogu_runtime"
    cache.mkdir()
    with _runtime_environment(cache):
        backend = BooguSubprocessBackend(config)
        try:
            backend.start(stderr_log_path=directory/"boogu.log")
            result = backend.edit(source_rgb=source, instruction=prompt, width=width, height=height,
                                  thinking_enabled=False, instruction_rewrite_enabled=False, seed=42)
        finally:
            backend.close()
    output = directory/"repainted_frame0.png"
    output.write_bytes(result.png_bytes)  # Existing backend verifies exact-size native RGB PNG.
    job.update(worker_metadata=result.worker_metadata, wall_seconds=time.monotonic()-started)
    return output, job


def run_cases(cases, *, qwen, backend, boogu_config, seed=42, variant="both"):
    names = list(VARIANTS) if variant == "both" else [variant]
    if any(name not in VARIANTS for name in names):
        raise ValueError("variant must be text, frame0 or both")
    prepared = []
    report = {"manifests":[], "failures":[]}
    try:
        backend.validate()
        for case in cases:
            directory = validate_output_root(Path(case["output_dir"]))
            if directory.exists():
                raise FileExistsError(f"Pilot never overwrites an existing case: {directory}")
            if "frame0" in names:
                replace(boogu_config, temporary_root=directory/"frame0"/"boogu_tmp").validate()
        for case in cases:
            directory = Path(case["output_dir"])
            video = Path(case["target_video_path"])
            source = inspect_video_timeline(video)
            plan, aspect = plan_h3_timeline(source), match_aspect_ratio(source)
            directory.mkdir(parents=True, exist_ok=False)
            (directory/"original.mp4").symlink_to(video)
            try:
                with _runtime_environment(directory):
                    descriptions = qwen.describe_two(video)
                    replacements = qwen.invent_two(*descriptions[:2])
                for name, value in zip(("source_subject_1", "source_subject_2", "shot_description",
                                        "replacement_subject_1", "replacement_subject_2"),
                                       (*descriptions, *replacements), strict=True):
                    (directory/f"{name}.txt").write_text(value+"\n", encoding="utf-8")
                prepared.append((case, directory, video, plan, aspect, descriptions, replacements))
            except Exception as exc:  # noqa: BLE001 -- candidate pilot records errors, never falls back
                failure = {"case_id":case["case_id"], "stage":"qwen", "error_type":type(exc).__name__,
                           "error":str(exc)}
                write_json_atomic(directory/"failure.json", failure)
                report["failures"].append(failure)
    finally:
        qwen.close()  # No resident Qwen while Boogu/PDD use GPU memory.
    for case, directory, video, plan, aspect, descriptions, replacements in prepared:
        for name in names:
            target = directory/name
            target.mkdir()
            try:
                started = time.monotonic()
                reference, boogu = None, None
                if name == "frame0":
                    reference, boogu = _edit_frame_zero(video, target, aspect, descriptions, replacements, boogu_config)
                prompt = build_prompt(*descriptions, *replacements, frame0=name == "frame0")
                (target/"h3_prompt.txt").write_text(prompt+"\n", encoding="utf-8")
                metadata = backend.run(video, prompt, plan, aspect, target, seed, reference_image=reference)
                output = inspect_video_timeline(target/"raw.mp4")
                if (output.frame_count != plan.native_frame_count or output.rate != 24
                        or abs(output.duration_seconds-plan.native_frame_count/24) > 1/24
                        or (output.width,output.height) != output_size(aspect)):
                    raise ValueError("PDD output violates planned native timeline/canvas")
                references = [{"type":"video", "label":"Video 1", "path":str(video)}]
                if reference is not None:
                    references.append({"type":"image", "label":"Picture 1", "path":str(reference)})
                manifest = {**metadata, "case_id":case["case_id"], "variant":VARIANTS[name],
                            "source_video":str(video), "source_subject_1":descriptions[0],
                            "source_subject_2":descriptions[1], "shot_description":descriptions[2],
                            "replacement_subject_1":replacements[0], "replacement_subject_2":replacements[1],
                            "h3_prompt":prompt, "references":references,
                            "reference_image":str(reference) if reference else None, "boogu":boogu,
                            "qwen_model_path":str(qwen.model_path), "h3_model_path":str(backend.model_root),
                            "pdd_checkpoint":str(backend.lora), "seed":seed,
                            "timeline":asdict(plan), "output_timeline":asdict(output),
                            "aspect_ratio":aspect, "output_size":output_size(aspect),
                            "source_aspect_ratio":plan.source.width/plan.source.height,
                            "variant_wall_seconds":time.monotonic()-started,
                            "production_timeline_compatible":False}
                path = target/"manifest.json"
                write_json_atomic(path, manifest)
                report["manifests"].append(str(path))
            except Exception as exc:  # noqa: BLE001 -- no quality rejection, retry or reference fallback
                failure = {"case_id":case["case_id"], "variant":VARIANTS[name],
                           "error_type":type(exc).__name__, "error":str(exc)}
                write_json_atomic(target/"failure.json", failure)
                report["failures"].append(failure)
    return report
