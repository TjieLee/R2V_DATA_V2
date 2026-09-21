"""Thin subprocess adapter to the pinned ModelTC Ref2VA CLI."""

import json
import subprocess
from pathlib import Path

from .h3_ref2va import (
    CONTRACTS,
    LIGHTX2V_CHECKPOINT,
    TURBO_REVISION,
    require_local,
    runtime_environment,
    validate_checkpoint,
    validate_code,
)


class LightX2VBackend:
    name = "lightx2v"

    def __init__(self, python, model_root, code_root, lora, *, fsdp2=False, nproc=1):
        self.python, self.model_root, self.code_root, self.lora = python, model_root, code_root, lora
        self.fsdp2, self.nproc = fsdp2, nproc

    def validate(self):
        self.python = require_local(self.python, "--h3-python / H3_PYTHON")
        self.model_root = require_local(self.model_root, "--h3-model-root / H3_MODEL_ROOT", directory=True)
        require_local(self.model_root / "transformer_ref", "Ref2VA base transformer_ref", directory=True)
        self.code_root = require_local(self.code_root, "--lightx2v-code-root / H3_TURBO_CODE_ROOT", directory=True)
        self.lora = require_local(self.lora, "--lightx2v-lora / H3_LIGHTX2V_LORA")
        validate_checkpoint(self.lora, LIGHTX2V_CHECKPOINT)
        validate_code(self.code_root, TURBO_REVISION,
                      ("inference_minimax_h3.py", "resolution_util.py", "minimax_h3_ref2va_pipeline.py"))
        if self.nproc < 1 or (not self.fsdp2 and self.nproc != 1):
            raise ValueError("H3_NPROC_PER_NODE >1 requires H3_FSDP2=1 for LightX2V")

    def run(self, video, prompt, plan, aspect, directory, seed):
        job = directory / "job.json"
        job.write_text(json.dumps({"examples":[{
            "task":"ref2va", "prompt":prompt, "duration":plan.requested_duration,
            "megapixels":1.0, "aspect_ratio":aspect,
            "references":[{"type":"video", "path":str(video.absolute())}],
        }]}, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [str(self.python)]
        if self.fsdp2:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={self.nproc}"]
        command += [str(Path(self.code_root)/"inference_minimax_h3.py"),
            "--jobs-json",str(job), "--model-id",str(self.model_root), "--lora-path",str(self.lora),
            "--inference-steps","8", "--video-shift","6", "--audio-shift","3",
            "--lora-alpha","8", "--lora-scale","1", "--reference-resize-mode","match",
            "--megapixels","1.0", "--num-frames",str(plan.native_frame_count),
            "--seed",str(seed), "--output-dir",str(directory/"lightx2v_raw")]
        if self.fsdp2:
            command.append("--fsdp2")
        with (directory/"log.txt").open("w") as log:
            log.write(json.dumps(command)+"\n")
            log.flush()
            subprocess.run(command, cwd=self.code_root, env=runtime_environment(directory),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        suffix = "_rank0000" if self.fsdp2 else ""
        generated = directory/"lightx2v_raw"/f"0000_ref2va_seed{seed}{suffix}.mp4"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise ValueError("LightX2V did not produce the expected original-job output")
        generated.rename(directory/"raw.mp4")  # No transcode / normalization.
        return {**CONTRACTS[self.name], "execution_mode":"fsdp2" if self.fsdp2 else "cpu_offload",
                "nproc_per_node":self.nproc}
