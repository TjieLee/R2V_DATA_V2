"""Boogu first-frame preparation phase for the parallel pair executor.

Runs only after every Qwen worker has exited, one worker per GPU, and is the
only stage that publishes preparation/prepared.json for the frame0 variant.
Heavy imports stay inside functions so the parent process stays lightweight.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from .h3_pair_state import (
    atomic_json,
    begin_attempt,
    eligible,
    fail_attempt,
    partition,
    publish_prepared,
    qwen_prepared,
)
from .h3_ref2va import runtime_environment
from .h3_two_person import boogu_frame0_prompt, build_prompt

VARIANT = "frame0_two_person"
REFERENCE_MODE = "frame0_boogu"
SOURCE_FRAME = "source_frame0.png"
REPAINTED_FRAME = "repainted_frame0.png"


@contextmanager
def boogu_runtime(directory):
    """Boogu owns its own cache namespace; restore the pair worker env afterwards."""
    previous = os.environ.copy()
    environment = runtime_environment(directory)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def boogu_config(config, directory):
    from r2v_data_v2.v3.reference_edit_boogu import BooguWorkerConfig

    # CUDA_VISIBLE_DEVICES already names this worker's single physical GPU.
    # Overwriting it with "0" would remap every worker onto physical GPU 0.
    return BooguWorkerConfig(
        python_executable=Path(config["boogu_python"]),
        code_root=Path(config["boogu_code_root"]),
        model_path=Path(config["boogu_model_root"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES") or "0",
        temporary_root=directory/"boogu_tmp",
    )


def _decode_frame_zero(video, directory, width, height):
    from PIL import Image, ImageOps

    from r2v_data_v2.v3.frames import OpenCvFrameDecoder

    # Exact decoded index 0. No search, visibility gate or source modification.
    frame = OpenCvFrameDecoder().decode_indices(video,[0])[0].image
    image = ImageOps.fit(frame.convert("RGB"),(width,height),method=Image.Resampling.LANCZOS,
                         centering=(0.5,0.5))
    path = directory/SOURCE_FRAME
    image.save(path,format="PNG")
    return image, path


def edit_frame_zero(video, directory, width, height, texts, config, backend_factory=None):
    """Boogu replaces only the two visible people; pose/layout/props stay exact."""
    from r2v_data_v2.v3.reference_edit_boogu import BooguSubprocessBackend

    started = time.monotonic()
    image, source_path = _decode_frame_zero(video,directory,width,height)
    instruction = boogu_frame0_prompt(texts["source_subject_1"],texts["source_subject_2"],
                                      texts["replacement_subject_1"],texts["replacement_subject_2"])
    (directory/"boogu_prompt.txt").write_text(instruction+"\n",encoding="utf-8")
    settings = boogu_config(config,directory)
    job = {"model_name":"Boogu-Image-0.1-Edit-Turbo","python":str(settings.python_executable),
           "code_root":str(settings.code_root),"model_path":str(settings.model_path),
           "cuda_visible_devices":settings.cuda_visible_devices,"source_frame_index":0,
           "source_video":str(video),"width":width,"height":height,"seed":42,
           "thinking_enabled":False,"instruction_rewrite_enabled":False,"num_inference_steps":4,
           "text_guidance_scale":1.0,"image_guidance_scale":1.0,"use_dmd_student_inference":True,
           "frame0_transform":"cover_resize_center_crop"}
    output = directory/REPAINTED_FRAME
    with boogu_runtime(directory/"boogu_runtime"):
        backend = backend_factory(settings) if backend_factory else BooguSubprocessBackend(settings)
        try:
            backend.start(stderr_log_path=directory/"boogu.log")
            result = backend.edit(source_rgb=image,instruction=instruction,width=width,height=height,
                                  thinking_enabled=False,instruction_rewrite_enabled=False,seed=42)
        finally:
            backend.close()
        output.write_bytes(result.png_bytes)
    job.update(source_frame=str(source_path),repainted_frame=str(output),
               worker_metadata=getattr(result,"worker_metadata",None),
               wall_seconds=time.monotonic()-started)
    atomic_json(directory/"boogu_job.json",job)
    return output, job


def prepare_case(case, config, backend_factory=None):
    marker = qwen_prepared(case)
    if marker is None:
        raise ValueError("frame0 preparation requires the durable Qwen marker")
    directory = Path(case["directory"])/"preparation"
    video = Path(marker["source"])
    width, height = int(marker["width"]), int(marker["height"])
    texts = {name:marker[name] for name in ("source_subject_1","source_subject_2","shot_description",
                                            "replacement_subject_1","replacement_subject_2")}
    for name, value in texts.items():
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Qwen marker missing field: {name}")
    reference, job = edit_frame_zero(video,directory,width,height,texts,config,backend_factory)
    if not reference.is_file() or reference.stat().st_size == 0:
        raise ValueError("Boogu did not produce a usable repainted first frame")
    prompt = build_prompt(texts["source_subject_1"],texts["source_subject_2"],texts["shot_description"],
                          texts["replacement_subject_1"],texts["replacement_subject_2"],frame0=True)
    publish_prepared(case,{**marker,"prompt":prompt,"variant":VARIANT,"reference_mode":REFERENCE_MODE,
                           "reference_image":str(reference),"boogu":job},
                     {**texts,"h3_prompt":prompt})
    return reference


def prepare_frame0_partition(config, worker, backend_factory=None):
    cases = partition(config["cases"],worker,config.get("group_size",2))
    limits = config["limits"]
    # Only Qwen-marked cases that are not yet committed; Qwen has already exited.
    cases = [case for case in cases if qwen_prepared(case) is not None
             and eligible(case,"prepare",limits[case["case_id"]]["prepare"])]
    if not cases:
        return
    for case in cases:
        while eligible(case,"prepare",limits[case["case_id"]]["prepare"]):
            attempt = begin_attempt(case,"prepare",{"pair_id":config["pair_id"],"frame0_worker":worker})
            try:
                prepare_case(case,config,backend_factory)
            except Exception as exc:  # noqa: BLE001 -- failure is case-local and durable
                fail_attempt(case,attempt,exc)
