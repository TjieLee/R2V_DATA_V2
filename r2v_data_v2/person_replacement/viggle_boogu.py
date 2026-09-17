"""Candidate-only Boogu repaint + unmodified official Viggle inference."""

import hashlib
import io
import json
import math
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.frames import OpenCvFrameDecoder
from r2v_data_v2.v3.reference_edit_boogu import (
    BOOGU_MODEL_NAME,
    BOOGU_MODEL_REVISION,
    DEFAULT_BOOGU_MODEL_PATH,
    BooguSubprocessBackend,
    BooguWorkerConfig,
)

from .pipeline import validate_output_root
from .timeline import VideoTimeline, inspect_video_timeline

VIGGLE_REVISION = "16e05b96cf715035544f0330d1d29b123340a2ba"
FIXED_PROMPT_SHA256 = "97cd0ed3626509c0395e80f418b8f3cbbe874fced16afa9ba10f4f54ba65f13f"
SAMPLE_SHA256 = "9f731e5857ff5f312602819d97a163d1ff338bfcb1a7c581edb36e725d14615b"
LADDER = ((416,960),(448,896),(480,832),(544,736),(640,640),
          (736,544),(832,480),(896,448),(960,416))


@dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    source_aspect_ratio: float
    canvas_aspect_ratio: float
    aspect_ratio_relative_error: float
    crop_fraction_estimate: float
    spatial_transform: str = "scale_to_cover_center_crop"


def select_viggle_canvas(width, height):
    if min(width, height) <= 0:
        raise ValueError("Source dimensions must be positive")
    ratio = width / height
    w, h = min(LADDER, key=lambda b: abs(b[0]/b[1]-ratio))
    target = w/h
    return Canvas(w, h, ratio, target, abs(ratio/target-1), 1-min(ratio/target,target/ratio))


@dataclass(frozen=True)
class ViggleTimeline:
    keep_frames: int
    render_frames: int
    pad_frames: int
    fps: int = 24


def plan_viggle_timeline(source: VideoTimeline):
    duration = source.duration_seconds
    if not math.isfinite(duration) or not 4 <= duration <= 243/24:
        raise ValueError("Viggle pilot requires full source duration in [4, 243/24] seconds; no truncation")
    # Nearest 24-Hz endpoint, like FFmpeg fps's default near rounding. Validate
    # the decoded result before any padding; never hide a short conversion.
    keep = math.floor(duration*24+0.5)
    render = max(124, ((keep-5+16)//17)*17+5)
    if render > 243:
        raise ValueError("Viggle render exceeds 243 frames")
    return ViggleTimeline(keep, render, render-keep)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def has_audio(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    return bool(json.loads(result.stdout)["streams"])


def ffmpeg_video(source, output, filters, count, *, keep_audio=False):
    if output.exists() or output.is_symlink() or source.resolve() == output.resolve():
        raise FileExistsError(output)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source), "-map", "0:v:0"]
    command += (["-map", "0:a?", "-c:a", "aac", "-t", str(count/24)] if keep_audio else ["-an"])
    command += ["-vf", filters, "-frames:v", str(count), "-fps_mode", "cfr", "-r", "24",
                "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", str(output)]
    with (output.parent / "timeline.log").open("a") as log:
        log.write(json.dumps(command)+"\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def validate_prepared(path, canvas, count):
    actual = inspect_video_timeline(path)
    if (actual.frame_count != count or actual.rate != 24
            or (actual.width, actual.height) != (canvas.width, canvas.height)):
        raise ValueError(f"Invalid prepared/generated geometry or timeline: {path}: {actual}")
    return actual


def prepare_driving(source, directory, canvas, plan):
    unpadded = directory / "driving_unpadded.mp4"
    w, h = canvas.width, canvas.height
    ffmpeg_video(source, unpadded,
                 f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,fps=24",
                 plan.keep_frames)
    validate_prepared(unpadded, canvas, plan.keep_frames)
    ffmpeg_video(unpadded, directory / "driving.mp4",
                 f"tpad=stop_mode=clone:stop={plan.pad_frames}" if plan.pad_frames else "null",
                 plan.render_frames)
    validate_prepared(directory / "driving.mp4", canvas, plan.render_frames)


def extract_frame_zero(source, destination):
    frame = OpenCvFrameDecoder().decode_indices(source, [0])[0]
    frame.image.convert("RGB").save(destination, format="PNG")


def build_boogu_repaint_prompt(source, replacement):
    return (
        "Edit the first image, which is one frame of a video. Change exactly one thing: who the main person is.\n"
        f"Replace: {source}\nwith: {replacement}\n"
        "The body does not move. Keep every limb, both hands, hand-body contact, hand-object contact, "
        "head angle, facial expression, gaze direction, mouth state, body position, body scale and occlusion exactly. "
        "Do not re-pose, straighten, re-center or move the person. Do not change camera angle, framing, crop, "
        "background, lighting, shadows or color grade. Do not add, remove or move objects, alter held or touched "
        "props, or add text, captions or watermarks."
    )


def read_descriptions(path, source):
    record = json.loads(Path(path).read_text())
    if Path(record["target_video_path"]).expanduser().resolve() != source.expanduser().resolve():
        raise ValueError("Descriptions source does not match original processed clip")
    values = (record["source_person_description"], record["replacement_person_description"])
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Descriptions must be non-empty strings")
    return values


def cache_environment(root, gpu):
    return {"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(root / "huggingface"),
            "HF_HUB_CACHE": str(root / "huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": str(root / "huggingface/hub"),
            "TRANSFORMERS_CACHE": str(root / "transformers"),
            "TORCH_HOME": str(root / "torch"), "XDG_CACHE_HOME": str(root / "cache"),
            "TORCH_EXTENSIONS_DIR": str(root / "torch_extensions"),
            "TRITON_CACHE_DIR": str(root / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
            "TMPDIR": str(root)}


@contextmanager
def boogu_environment(root, gpu):
    # Existing backend inherits os.environ. This pilot is strictly sequential.
    updates = cache_environment(root, gpu)
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_boogu(config: BooguWorkerConfig):
    config.validate()
    for path in (config.python_executable, config.worker_script, config.model_path / "model_index.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not config.code_root.is_dir() or not config.model_path.is_dir():
        raise ValueError("Existing local Boogu code/model directory required")
    if config.model_revision != BOOGU_MODEL_REVISION:
        raise ValueError("Require existing Boogu hotfix-1k-20260708")


def repaint(directory, backend, prompt, canvas, *, model_root=DEFAULT_BOOGU_MODEL_PATH):
    with Image.open(directory / "driving_frame0.png") as opened:
        opened.load()
        if opened.mode != "RGB" or opened.size != (canvas.width, canvas.height):
            raise ValueError("Prepared frame0 must be RGB at exact canvas size")
        source = opened.copy()
    job = {"model_root": str(model_root), "model_revision": BOOGU_MODEL_REVISION,
           "input_image": str(directory / "driving_frame0.png"), "prompt": prompt,
           "output_image": str(directory / "repainted_frame.png"), "width": canvas.width,
           "height": canvas.height, "seed": 42, "thinking_enabled": False,
           "instruction_rewrite_enabled": False, "num_inference_steps": 4,
           "text_guidance_scale": 1.0, "image_guidance_scale": 1.0}
    write_json_atomic(directory / "boogu_job.json", job)
    result = backend.edit(source_rgb=source, instruction=prompt, width=canvas.width,
                          height=canvas.height, thinking_enabled=False,
                          instruction_rewrite_enabled=False, seed=42)
    with Image.open(io.BytesIO(result.png_bytes)) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (canvas.width, canvas.height):
            raise ValueError("Boogu output must be an exact-size RGB PNG; no resizing")
    (directory / "repainted_frame.png").write_bytes(result.png_bytes)
    return result.worker_metadata


class ViggleAdapter:
    def __init__(self, python, root, h3_root, gpu="1"):
        # Resolving a venv's python symlink would bypass that environment.
        self.python = Path(python).expanduser().absolute()
        self.root = Path(root).expanduser().resolve()
        self.h3_root = Path(h3_root).expanduser().resolve()
        self.gpu = gpu

    def validate(self):
        for path in (self.python, self.root / "assets/fixed_prompt.txt",
                     self.root / "assets/fixed_embed_fwd_anyframe.pt", self.root / "inference/sample.py",
                     self.h3_root / "modular_model_index.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in (self.root / "transformer", self.root / "lora", *(self.h3_root / name
                     for name in ("vae", "audio_vae", "scheduler", "audio_scheduler"))):
            if not path.is_dir():
                raise FileNotFoundError(path)
        if (sha256(self.root / "assets/fixed_prompt.txt") != FIXED_PROMPT_SHA256
                or sha256(self.root / "inference/sample.py") != SAMPLE_SHA256):
            raise ValueError("Viggle pinned prompt/sample hash mismatch; vendor edits are forbidden")
        if not (self.root / "assets/fixed_embed_fwd_anyframe.pt").stat().st_size:
            raise ValueError("Viggle frozen embedding must be non-empty")

    def run(self, directory, *, render_frames, source, source_has_audio):
        self.validate()
        command = [str(self.python), str(self.root / "inference/sample.py"),
                   "--model-dir", str(self.h3_root), "--cond", str(directory / "driving.mp4"),
                   "--ref", str(directory / "repainted_frame.png"), "--out", str(directory / "viggle_raw.mp4"),
                   "--num-frames", str(render_frames), "--steps", "4", "--flow-shift", "3", "--seed", "42"]
        if source_has_audio:
            command += ["--audio", str(source)]
        cache = directory / "runtime/viggle"
        cache.mkdir(parents=True, exist_ok=True)
        with (directory / "viggle.log").open("w") as log:
            log.write(json.dumps(command)+"\n")
            log.flush()
            subprocess.run(command, cwd=self.root, env={**os.environ, **cache_environment(cache, self.gpu)},
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def run_cases(cases, *, sources, audio, qwen, descriptions, boogu_config, viggle):
    for case in cases:
        output = validate_output_root(Path(case["output_dir"]))
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    values = []
    try:
        for case in cases:
            source = Path(case["target_video_path"])
            if descriptions is not None:
                values.append(read_descriptions(descriptions, source))
            else:
                description = qwen.describe(source)
                values.append((description, qwen.invent(description)))
    finally:
        if qwen is not None:
            qwen.close()
    manifests = []
    for case, source_timeline, source_audio, (description, replacement) in zip(cases, sources, audio, values, strict=True):
        source = Path(case["target_video_path"]).resolve()
        directory = Path(case["output_dir"]).resolve()
        canvas = select_viggle_canvas(source_timeline.width, source_timeline.height)
        plan = plan_viggle_timeline(source_timeline)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "original.mp4").symlink_to(source)
        prepare_driving(source, directory, canvas, plan)
        extract_frame_zero(directory / "driving_unpadded.mp4", directory / "driving_frame0.png")
        prompt = build_boogu_repaint_prompt(description, replacement)
        for name, text in (("source_person",description), ("replacement_person",replacement), ("boogu_prompt",prompt)):
            (directory / f"{name}.txt").write_text(text+"\n")
        runtime = directory / "runtime/boogu"
        runtime.mkdir(parents=True)
        with boogu_environment(runtime, boogu_config.cuda_visible_devices):
            backend = BooguSubprocessBackend(replace(boogu_config, temporary_root=runtime, seed=42))
            try:
                backend.start(stderr_log_path=directory / "boogu.log")
                boogu_metadata = repaint(directory, backend, prompt, canvas, model_root=boogu_config.model_path)
            finally:
                backend.close()
        viggle.run(directory, render_frames=plan.render_frames, source=source, source_has_audio=source_audio)
        raw = directory / "viggle_raw.mp4"
        validate_prepared(raw, canvas, plan.render_frames)
        ffmpeg_video(raw, directory / "replacement.mp4", f"trim=end_frame={plan.keep_frames},setpts=PTS-STARTPTS",
                     plan.keep_frames, keep_audio=True)
        validate_prepared(directory / "replacement.mp4", canvas, plan.keep_frames)
        record = {**case, "source_person_description":description, "replacement_person_description":replacement,
                  "boogu_prompt":prompt, "boogu_model_path":str(boogu_config.model_path), "boogu_model_name":BOOGU_MODEL_NAME,
                  "boogu_revision":BOOGU_MODEL_REVISION, "boogu_steps":4, "boogu_seed":42, "boogu_metadata":boogu_metadata,
                  "viggle_model_path":str(viggle.root), "viggle_revision":VIGGLE_REVISION, "h3_model_path":str(viggle.h3_root),
                  "viggle_fixed_prompt_path":str(viggle.root / "assets/fixed_prompt.txt"),
                  "viggle_fixed_prompt_sha256":FIXED_PROMPT_SHA256, "viggle_prompt_mode":"official_frozen_embedding",
                  "custom_h3_prompt":False, "viggle_steps":4, "viggle_flow_shift":3, "viggle_seed":42,
                  "source_has_audio":source_audio, "viggle_audio_conditioning":source_audio,
                  **{f"source_{k}":v for k,v in asdict(source_timeline).items()}, **asdict(plan),
                  "prepared_fps":24, "canvas_width":canvas.width, "canvas_height":canvas.height,
                  **{k:v for k,v in asdict(canvas).items() if k not in {"width","height"}},
                  "production_geometry_compatible":False, "production_timeline_compatible":False,
                  "candidate_only_reason":"official cover-cropped canvas and native 24fps qualitative evaluation"}
        path = directory / "manifest.json"
        write_json_atomic(path, record)
        manifests.append(path)
    return manifests
