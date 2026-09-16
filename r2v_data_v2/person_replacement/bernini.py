"""Thin adapter to the operator-supplied official Bernini v2v launcher."""

import os
import re
import subprocess
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from .timeline import (
    VideoTimeline,
    inspect_video_timeline,
    transcode_timeline,
    validate_matching_timeline,
)

UPSTREAM_COMMIT = "e6c2cf11843470b08491745947abcba9c3c6b079"


@dataclass(frozen=True)
class BerniniTimelinePlan:
    source: VideoTimeline
    internal_frame_count: int
    internal_fps: int
    pad_frames: int

    @property
    def requires_input_conversion(self):
        return self.pad_frames > 0 or self.source.rate != self.internal_fps


def plan_bernini_timeline(source: VideoTimeline) -> BerniniTimelinePlan:
    validate_matching_timeline(source, source)
    # Pinned smart_video_nframes(frame_factor=4, add_one=True) enforces min 5.
    # Giving it equal input/CLI FPS and 4n+1 frames makes VAE indices exactly 0..N-1.
    count = max(5, ((source.frame_count - 1 + 3) // 4) * 4 + 1)
    fps = max(1, int(source.rate + Fraction(1, 2)))
    return BerniniTimelinePlan(source, count, fps, count - source.frame_count)


def _internal_timeline(plan):
    return replace(plan.source, frame_count=plan.internal_frame_count, fps=float(plan.internal_fps),
                   fps_num=plan.internal_fps, fps_den=1,
                   duration_seconds=plan.internal_frame_count / plan.internal_fps)


def prepare_generation_input(video: Path, case_dir: Path, plan: BerniniTimelinePlan) -> Path:
    if not plan.requires_input_conversion:
        return video
    output = case_dir / "bernini_input.mp4"
    # Fractional source FPS also needs retiming, even with pad=0: otherwise the
    # upstream duration/FPS-based linspace can skip/repeat real VAE input frames.
    prefix = f"tpad=stop_mode=clone:stop={plan.pad_frames}" if plan.pad_frames else ""
    transcode_timeline(video, output, rate=Fraction(plan.internal_fps), prefix_filter=prefix, lossless=True)
    validate_matching_timeline(_internal_timeline(plan), inspect_video_timeline(output))
    return output


def normalize_output(raw: Path, final: Path, plan: BerniniTimelinePlan):
    before = inspect_video_timeline(raw)
    validate_matching_timeline(_internal_timeline(plan), before)
    if final.exists():
        raise FileExistsError(final)
    if not plan.requires_input_conversion:
        raw.rename(final)
    else:
        transcode_timeline(raw, final, rate=plan.source.rate,
                           prefix_filter=f"trim=end_frame={plan.source.frame_count}")
    after = inspect_video_timeline(final)
    validate_matching_timeline(plan.source, after)
    return before, after

def serialize_case(video: Path, output: Path, prompt: str) -> dict:
    # Official assets/testcases/v2v format; generation seed belongs to the CLI.
    return {"task_type": "v2v", "prompt": prompt,
            "video": str(video.resolve()), "output": str(output.resolve())}


def prepare_launcher(script: str, *, seed: int, num_frames: int, fps: int) -> str:
    """Adapt the verified demo loop, seed and timeline; preserve quality options.

    The upstream 2026-09-16 script ignores CASE_PATH by looping over three demos.
    Fail on unfamiliar control flow rather than silently launching those demos.
    The result lives in the NEW case directory; official source stays unchanged.
    """
    demo = (
        "for CASE_PATH in assets/testcases/v2v/v2v_case1.json "
        "assets/testcases/v2v/v2v_case2.json assets/testcases/v2v/v2v_case3.json;\ndo\n"
    )
    if script.count(demo) == 1 and script.rstrip().endswith("\ndone"):
        script = script.replace(demo, "", 1).rstrip()[:-4]
    if re.search(r"(?m)^\s*(for|while|until|if|elif|else|case|select|function|source|\.)\s", script):
        raise ValueError("Unsupported Bernini launcher control flow; inspect supplied official code")
    if re.search(r"(?m)^\s*(?:[a-zA-Z_]\w*\s*\(\s*\)|[{}()])", script):
        raise ValueError("Unsupported Bernini launcher function/group control flow")
    for variable in ("CASE_PATH", "BERNINI_CONFIG"):
        assignments = re.findall(rf"(?m)^\s*(?:export\s+)?{variable}=(.*)$", script)
        if len(assignments) != 1 or not re.fullmatch(
            rf"\$\{{{variable}:-[^\n]+\}}", assignments[0]
        ):
            raise ValueError(f"Unsupported Bernini {variable} override; must honor supplied environment")
    if (len(re.findall(r"(?m)^torchrun\s", script)) != 1
            or '--case "$CASE_PATH"' not in script
            or '--config "$BERNINI_CONFIG"' not in script
            or "--use_pe" in script):
        raise ValueError("Unsupported Bernini launcher: require one case/config and no prompt enhancer")
    for option, value in (("seed", seed), ("num_frames", num_frames), ("fps", fps)):
        if type(value) is not int or value < (0 if option == "seed" else 1):
            raise ValueError(f"Invalid Bernini {option}: {value}")
        if len(re.findall(rf"--{option}\b", script)) != 1:
            raise ValueError(f"Require exactly one --{option}")
        script, count = re.subn(rf"--{option}\s+\d+(?=\s|$)", f"--{option} {value}", script)
        if count != 1:
            raise ValueError(f"Unsupported Bernini {option} option; require literal integer")
    return script + "\n"


class BerniniAdapter:
    def __init__(self, code_root, config):
        self.code_root = Path(code_root).expanduser() if code_root else None
        self.config = Path(config).expanduser()

    def validate(self):
        if self.code_root is None:
            raise ValueError("Provide --bernini-code-root or BERNINI_CODE_ROOT; checkpoint is not code")
        self.code_root = self.code_root.resolve(strict=True)
        self.config = self.config.resolve(strict=True)
        if not self.config.is_dir():
            raise ValueError("Bernini config must be a local checkpoint directory")
        self.launcher = self.code_root / "scripts/bernini/run_v2v.sh"
        if not self.launcher.is_file() or not (self.code_root / "infer_multi_gpu.py").is_file():
            raise ValueError("Bernini code root needs scripts/bernini/run_v2v.sh and infer_multi_gpu.py")
        actual = subprocess.check_output(["git", "-C", str(self.code_root), "rev-parse", "HEAD"],
                                         text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f"Bernini commit mismatch: {actual}; require {UPSTREAM_COMMIT}")
        prepare_launcher(self.launcher.read_text(), seed=42, num_frames=81, fps=16)

    def run(self, case_path: Path, seed: int, *, plan: BerniniTimelinePlan):
        self.validate()
        effective = case_path.parent / "bernini_launcher.sh"
        effective.write_text(prepare_launcher(self.launcher.read_text(), seed=seed,
                                             num_frames=plan.internal_frame_count, fps=plan.internal_fps))
        cache = case_path.parent / "runtime"
        cache.mkdir()
        env = dict(os.environ)
        env.update({
            "CASE_PATH": str(case_path.resolve()), "BERNINI_CONFIG": str(self.config),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"), "XDG_CACHE_HOME": str(cache / "cache"),
            "TRITON_CACHE_DIR": str(cache / "triton"), "TMPDIR": str(cache),
        })
        with (case_path.parent / "bernini.log").open("w") as log:
            subprocess.run(
                ["bash", str(effective)], cwd=self.code_root, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
