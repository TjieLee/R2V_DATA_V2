"""Isolated JoyAI pilot orchestration; never imports the external model runtime."""

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from r2v_data_v2.reconciliation import write_json_atomic

from .pipeline import extract_frame_zero, validate_output_root
from .timeline import inspect_video_timeline, validate_matching_timeline

UPSTREAM_COMMIT = "69fdaea89e2fd7708435790d73eb30989e722a5f"
REQUIRED_CODE_FILES = (
    "deploy/xvideo/serving/joyomni_streaming.py", "deploy/xvideo/models/pipeline.py",
    "deploy/xvideo/models/models.py", "deploy/xvideo/config.py",
)


def build_joyai_prompt(source: str, replacement: str) -> str:
    return (
        f"Replace {source} with {replacement}.\n\n"
        "The replacement person follows the source person's original pose and motion "
        "throughout the video, including the same body orientation, limb movement, "
        "hand placement, head motion, spatial position, timing, and interactions with "
        "objects or surfaces.\n\n"
        "Keep the entire original background unchanged, including all objects, camera "
        "framing and motion, lighting, shadows, reflections, and scene layout. "
        "Only the person's identity and visible appearance are changed. "
        "Maintain natural temporal consistency and realistic motion across the whole video."
    )


def runtime_environment(root: Path) -> dict[str, str]:
    root = validate_output_root(root)
    return {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf/hub"), "HUGGINGFACE_HUB_CACHE": str(root / "hf/hub"),
        "HF_MODULES_CACHE": str(root / "hf/modules"), "TRANSFORMERS_CACHE": str(root / "hf/transformers"),
        "TORCH_HOME": str(root / "torch"), "TORCH_EXTENSIONS_DIR": str(root / "torch/extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"), "TRITON_CACHE_DIR": str(root / "triton"),
        "CUDA_CACHE_PATH": str(root / "cuda"), "XDG_CACHE_HOME": str(root / "cache"),
        "TMPDIR": str(root),
    }


@contextmanager
def local_runtime_environment(root: Path):
    values = runtime_environment(root)
    root.mkdir(parents=True, exist_ok=False)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_file(path: Path):
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f"Required local checkpoint/code file missing or empty: {path}")


class JoyAIAdapter:
    def __init__(self, code_root, python, model_root, text_encoder_ckpt):
        self.code_root = Path(code_root).expanduser().resolve() if code_root else None
        # Preserve venv executable symlinks: resolving them can select the wrong environment.
        self.python = Path(python).expanduser().absolute() if python else None
        self.model_root = Path(model_root).expanduser().resolve()
        self.text_encoder_ckpt = Path(text_encoder_ckpt).expanduser().resolve()
        self.dit_ckpt = self.model_root / "dit/joyai_video_edit_dit_0811.pth"
        self.vae_ckpt = self.model_root / "vae"

    def validate(self):
        if self.code_root is None or not self.code_root.is_dir():
            raise ValueError("Provide --joyai-code-root / JOYAI_CODE_ROOT with the pinned official checkout")
        git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        commit = subprocess.check_output(
            ["git", "-C", str(self.code_root), "rev-parse", "HEAD"], text=True, env=git_env,
        ).strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"JoyAI upstream commit mismatch: {commit}; require {UPSTREAM_COMMIT}")
        if subprocess.check_output(
            ["git", "-C", str(self.code_root), "status", "--porcelain", "--", "deploy/xvideo"],
            text=True, env=git_env,
        ).strip():
            raise ValueError("JoyAI runtime source must be clean at the pinned commit")
        for name in REQUIRED_CODE_FILES:
            _require_file(self.code_root / name)
        if self.python is None or not self.python.is_file() or not os.access(self.python, os.X_OK):
            raise ValueError("Provide --joyai-python / JOYAI_PYTHON: executable in the isolated JoyAI env")
        for path in (self.dit_ckpt, self.vae_ckpt / "config.json",
                     self.vae_ckpt / "diffusion_pytorch_model.safetensors"):
            _require_file(path)
        encoder = self.text_encoder_ckpt
        for name in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "preprocessor_config.json"):
            _require_file(encoder / name)
        index = encoder / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                raise ValueError("Empty MiMo weight map")
            for shard in shards:
                path = (encoder / shard).resolve()
                if not path.is_relative_to(encoder):
                    raise ValueError("MiMo weight index escapes checkpoint directory")
                _require_file(path)
        else:
            _require_file(encoder / "model.safetensors")

    def provenance(self):
        return {"backend": "joyai_video_edit", "joyai_code_root": str(self.code_root),
                "joyai_upstream_commit": UPSTREAM_COMMIT, "joyai_model_root": str(self.model_root),
                "joyai_dit_ckpt": str(self.dit_ckpt), "joyai_vae_ckpt": str(self.vae_ckpt),
                "joyai_text_encoder_ckpt": str(self.text_encoder_ckpt)}

    def run(self, source, output, prompt_path, timeline, seed):
        self.validate()
        output = validate_output_root(Path(output))
        if output.exists():
            raise FileExistsError(output)
        for root in (self.code_root, self.model_root, self.text_encoder_ckpt, Path(source).parent):
            if output.is_relative_to(root):
                raise ValueError("JoyAI output must not overlap inputs/code/checkpoints")
        request = {**self.provenance(), "source": str(Path(source).resolve(strict=True)),
                   "output": str(output), "prompt_path": str(Path(prompt_path).resolve(strict=True)),
                   "timeline": asdict(timeline), "seed": seed}
        request_path = output.parent / "joyai_request.json"
        write_json_atomic(request_path, request)
        runtime = output.parent / "joyai_runtime"
        runtime.mkdir(exist_ok=False)
        env = {**os.environ, **runtime_environment(runtime)}
        # Don't inherit the parent Qwen environment's module overlays.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        worker = Path(__file__).resolve().parents[2] / "tools/person_replacement/joyai_offline_worker.py"
        with (output.parent / "joyai.log").open("w") as log:
            subprocess.run([str(self.python), str(worker), "--request", str(request_path)],
                           cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_joyai_cases(cases, *, qwen, adapter, seed: int):
    adapter.validate()
    for case in cases:
        output = validate_output_root(Path(case["output_dir"]))
        if output.exists():
            raise FileExistsError(f"Pilot does not overwrite case: {output}")
    prepared = []
    try:
        for case in cases:
            output = Path(case["output_dir"])
            output.mkdir(parents=True, exist_ok=False)
            source = Path(case["target_video_path"])
            timeline = inspect_video_timeline(source)
            reference = output / "reference_frame0.jpg"
            extract_frame_zero(source, reference)
            with local_runtime_environment(output / "qwen_runtime"):
                description = qwen.describe(source)
                replacement = qwen.invent(description)
            prompt = build_joyai_prompt(description, replacement)
            for name, text in (("source_person", description), ("replacement_person", replacement),
                               ("joyai_prompt", prompt)):
                (output / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            prepared.append((case, timeline, description, replacement, prompt, reference))
    finally:
        qwen.close()
    manifests = []
    for case, timeline, description, replacement, prompt, reference in prepared:
        output = Path(case["output_dir"])
        generated = output / "replacement.mp4"
        adapter.run(Path(case["target_video_path"]), generated, output / "joyai_prompt.txt", timeline, seed)
        edited = inspect_video_timeline(generated)
        validate_matching_timeline(timeline, edited)
        manifest = {**case, **adapter.provenance(), "seed": seed,
                    "qwen_model_path": str(qwen.model_path),
                    "reference_image_path": str(reference), "input_video_path": str(generated),
                    "source_person_description": description,
                    "replacement_person_description": replacement, "joyai_prompt": prompt,
                    **{f"source_{k}": v for k, v in asdict(timeline).items()},
                    **{f"output_{k}": v for k, v in asdict(edited).items()}}
        path = output / "manifest.json"
        write_json_atomic(path, manifest)
        manifests.append(path)
    return manifests
