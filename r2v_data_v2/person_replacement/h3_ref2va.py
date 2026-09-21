"""Lightweight H3 candidate contracts; no heavy model imports or source conversion."""

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .timeline import VideoTimeline, validate_matching_timeline

TURBO_REVISION = "02e26d591f7a04d5d1a074c9566d5dd4f22f6225"
PDD_REVISION = "335001fb9e5455d68a0caa18ec2e319072150328"
LIGHTX2V_CHECKPOINT = "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"
PDD_CHECKPOINT = "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"
CONTRACTS = {
    "lightx2v": {"inference_nfe": 8, "video_shift": 6, "audio_shift": 3,
                 "lora_alpha": 8, "lora_scale": 1, "reference_resize_mode": "match",
                 "checkpoint": LIGHTX2V_CHECKPOINT, "code_revision": TURBO_REVISION},
    "pdd": {"inference_nfe": 8, "strength": 1.0, "loader": "apply_pdd_lora",
            "schedule": "checkpoint plan with base scheduler shifts (12/3)",
            "checkpoint": PDD_CHECKPOINT, "code_revision": PDD_REVISION},
}


@dataclass(frozen=True)
class H3TimelinePlan:
    source: VideoTimeline
    requested_duration: float
    native_fps: int
    native_frame_count: int
    duration_drift_seconds: float


def plan_h3_timeline(source: VideoTimeline) -> H3TimelinePlan:
    validate_matching_timeline(source, source)
    duration = source.duration_seconds
    if not 2 <= duration <= 15:
        raise ValueError("H3 requires one whole source reference between 2 and 15 seconds; no truncation")
    count = max(5, round(duration * 24))
    count += (5 - count % 17) % 17
    # ModelTC's formula does not clamp to runtime min_duration=5.0. The first
    # legal >=5s length is 124, so pass this explicitly, recording the drift.
    count = max(124, count)
    if count / 24 > 15:
        raise ValueError("Aligned H3 output exceeds 15-second runtime limit; source will not be truncated")
    return H3TimelinePlan(source, duration, 24, count, count/24-duration)


def match_aspect_ratio(source: VideoTimeline) -> str:
    supported = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21")
    ratio = source.width / source.height
    def distance(label):
        w, h = map(int, label.split(":"))
        return abs(ratio / (w/h) - 1)
    selected = min(supported, key=distance)
    return selected


def output_size(aspect: str) -> tuple[int, int]:
    # ModelTC resolution_util.py at TURBO_REVISION, 1.0 MP ladder entry.
    if aspect == "16:9":
        return 1376, 768
    if aspect == "9:16":
        return 768, 1376
    w, h = map(int, aspect.split(":"))
    pixels = 1376 * 768
    return (round(math.sqrt(pixels*w/h)/32)*32, round(math.sqrt(pixels*h/w)/32)*32)


def require_local(value, name, *, directory=False) -> Path:
    if not value:
        raise ValueError(f"Provide {name}; only local paths are supported, no downloads")
    path = Path(value).expanduser().absolute()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{name} must be an existing local {'directory' if directory else 'file'}: {path}")
    return path


def validate_checkpoint(path: Path, expected: str):
    if path.name != expected:
        raise ValueError(f"Require exact Diffusers checkpoint {expected}; no alternative/fallback LoRA")
    with path.open("rb") as handle:
        if handle.read(40).startswith(b"version https://git-lfs"):
            raise ValueError("Checkpoint is an LFS pointer, not provisioned weights")


def validate_code(root: Path, revision: str, files: tuple[str, ...]):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, env=env).strip()
    if head != revision:
        raise ValueError(f"Upstream revision mismatch: require {revision}, got {head}")
    for name in files:
        require_local(root / name, name)
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", *files],
                   check=True, env=env, stdout=subprocess.DEVNULL)


def runtime_environment(directory: Path) -> dict:
    cache = directory / "runtime"
    cache.mkdir()
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update({"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                "HF_HUB_DISABLE_TELEMETRY":"1", "PYTHONDONTWRITEBYTECODE":"1"})
    for key, folder in {
        "HF_HOME":"hf", "HF_HUB_CACHE":"hf/hub", "HUGGINGFACE_HUB_CACHE":"hf/hub",
        "TRANSFORMERS_CACHE":"hf/transformers", "TORCH_HOME":"torch",
        "TORCHINDUCTOR_CACHE_DIR":"inductor", "TRITON_CACHE_DIR":"triton",
        "XDG_CACHE_HOME":"cache", "TMPDIR":"tmp",
    }.items():
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env
