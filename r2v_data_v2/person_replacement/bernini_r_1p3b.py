"""Renderer-only variant of the pinned Bernini pilot; no planner or prompt changes."""

import json
import os
import subprocess
import sys
from pathlib import Path

from .bernini import UPSTREAM_COMMIT, BerniniTimelinePlan


class BerniniR1p3BAdapter:
    def __init__(self, code_root, config):
        self.code_root = Path(code_root).expanduser() if code_root else None
        self.config = Path(config).expanduser() if config else None

    def validate(self):
        if self.code_root is None or self.config is None:
            raise ValueError("Provide local --bernini-code-root and --bernini-r-config")
        self.code_root = self.code_root.resolve(strict=True)
        self.config = self.config.resolve(strict=True)
        if not self.code_root.is_dir() or not (self.code_root / "infer_multi_gpu.py").is_file():
            raise ValueError("Bernini code root must contain infer_multi_gpu.py")
        actual = subprocess.check_output(
            ["git", "-C", str(self.code_root), "rev-parse", "HEAD"], text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f"Bernini commit mismatch: {actual}; require {UPSTREAM_COMMIT}")
        if not self.config.is_dir():
            raise ValueError("Bernini-R config must be a local checkpoint directory")
        config = json.loads((self.config / "config.json").read_text())
        if config.get("model_type") != "bernini_renderer" or config.get("skip_transformer_2") is not True:
            raise ValueError("Require bernini_renderer with skip_transformer_2=true")
        transformer = self.config / "transformer/config.json"
        if transformer.exists():
            facts = json.loads(transformer.read_text())
            expected = {"num_attention_heads": 12, "attention_head_dim": 128, "num_layers": 30}
            if any(facts.get(key) != value for key, value in expected.items()):
                raise ValueError("Transformer config is not the official Wan2.1-1.3B variant")

    def run(self, case_path: Path, seed: int, *, plan: BerniniTimelinePlan):
        self.validate()
        processes = int(os.environ.get("NPROC_PER_NODE", "4"))
        ulysses = int(os.environ.get("ULYSSES", "4"))
        if processes < 1 or ulysses < 1:
            raise ValueError("NPROC_PER_NODE and ULYSSES must be positive")
        # Pinned scripts/bernini_r/run_v2v.sh uses CLI defaults: 40 steps,
        # flow_shift=5.0 (overrides config shift=3.0), max_image_size=848.
        # Renderer V2V takes geometry from preprocessed video, not height/width.
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node", str(processes), str(self.code_root / "infer_multi_gpu.py"),
            "--config", str(self.config), "--ulysses", str(ulysses),
            "--case", str(case_path.resolve()), "--guidance_mode", "v2v_apg",
            "--num_frames", str(plan.internal_frame_count), "--fps", str(plan.internal_fps),
            "--seed", str(seed), "--num_inference_steps", "40", "--height", "0", "--width", "0",
            "--flow_shift", "5.0", "--max_image_size", "848", "--use_unipc", "--use_src_tgt_id",
        ]
        cache = case_path.resolve().parent / "runtime"
        cache.mkdir()
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "HF_HOME": str(cache / "huggingface"),
               "TORCH_HOME": str(cache / "torch"), "XDG_CACHE_HOME": str(cache / "cache"),
               "TRITON_CACHE_DIR": str(cache / "triton"), "TMPDIR": str(cache)}
        with (case_path.parent / "bernini.log").open("w") as log:
            subprocess.run(command, cwd=self.code_root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)


class ManifestDescriptionsQwen:
    """Reuse one Full Bernini case's exact descriptions without loading Qwen."""

    def __init__(self, manifest_path, video):
        path = Path(manifest_path).expanduser().resolve(strict=True)
        manifest = json.loads(path.read_text())
        self.source = Path(manifest["target_video_path"]).expanduser().resolve()
        if self.source != Path(video).expanduser().resolve():
            raise ValueError("Descriptions manifest source does not match processed source video")
        self.description = manifest["source_person_description"]
        self.replacement = manifest["replacement_person_description"]
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.description, self.replacement)):
            raise ValueError("Descriptions manifest requires two non-empty descriptions")
        self.model_path = manifest.get("qwen_model_path") or path

    def describe(self, video):
        if Path(video).expanduser().resolve() != self.source:
            raise ValueError("Descriptions source mismatch")
        return self.description

    def invent(self, description):
        if description != self.description:
            raise ValueError("Source description mismatch")
        return self.replacement

    def close(self):
        pass
