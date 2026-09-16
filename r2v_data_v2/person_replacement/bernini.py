"""Thin adapter to the operator-supplied official Bernini v2v launcher."""

import os
import re
import subprocess
from pathlib import Path


def serialize_case(video: Path, output: Path, prompt: str) -> dict:
    # Official assets/testcases/v2v format; generation seed belongs to the CLI.
    return {"task_type": "v2v", "prompt": prompt,
            "video": str(video.resolve()), "output": str(output.resolve())}


def prepare_launcher(script: str, seed: int) -> str:
    """Adapt only the verified demo loop and fixed seed; never rewrite inference.

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
    if re.search(r"(?m)^\s*(for|while|until|source|\.)\s", script):
        raise ValueError("Unsupported Bernini launcher control flow; inspect supplied official code")
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
    script, count = re.subn(r"--seed\s+\d+\b", f"--seed {int(seed)}", script)
    if count != 1:
        raise ValueError("Unsupported Bernini seed option; inspect supplied official launcher")
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
        prepare_launcher(self.launcher.read_text(), 42)

    def run(self, case_path: Path, seed: int):
        self.validate()
        effective = case_path.parent / "bernini_launcher.sh"
        effective.write_text(prepare_launcher(self.launcher.read_text(), seed))
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
