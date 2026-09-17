"""Isolated PDD worker adapter; never loads PDD with a generic LoRA loader."""

import json
import subprocess
from pathlib import Path

from .h3_ref2va import (
    PDD_CHECKPOINT,
    PDD_REVISION,
    output_size,
    require_local,
    runtime_environment,
    validate_checkpoint,
    validate_code,
)


class PDDBackend:
    name = "pdd"

    def __init__(self, python, model_root, code_root, lora):
        self.python, self.model_root, self.code_root, self.lora = python, model_root, code_root, lora

    def validate(self):
        self.python = require_local(self.python, "--h3-python / H3_PYTHON")
        self.model_root = require_local(self.model_root, "--h3-model-root / H3_MODEL_ROOT", directory=True)
        require_local(self.model_root/"transformer_ref", "Ref2VA base transformer_ref", directory=True)
        self.code_root = require_local(self.code_root, "--pdd-code-root / H3_PDD_CODE_ROOT", directory=True)
        self.lora = require_local(self.lora, "--pdd-lora / H3_PDD_LORA")
        validate_checkpoint(self.lora, PDD_CHECKPOINT)
        validate_code(self.code_root, PDD_REVISION, ("minimax_h3_pdd.py", "predict_ref2v.py"))

    def command(self, job):
        worker = Path(__file__).resolve().parents[2]/"tools/person_replacement/h3_pdd_worker.py"
        return [str(self.python), str(worker), "--job",str(job), "--pdd-code-root",str(self.code_root)]

    def run(self, video, prompt, plan, aspect, directory, seed, *, reference_image=None):
        width, height = output_size(aspect)
        job = directory/"job.json"
        payload = {"source":str(video), "prompt":prompt, "frames":plan.native_frame_count,
            "width":width, "height":height, "seed":seed, "model_root":str(self.model_root),
            "checkpoint":str(self.lora), "output":str(directory/"raw.mp4")}
        if reference_image is not None:
            payload["reference_image"] = str(require_local(reference_image, "reference image"))
        job.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        command = self.command(job)
        with (directory/"log.txt").open("w") as log:
            log.write(json.dumps(command)+"\n")
            log.flush()
            subprocess.run(command, cwd=directory, env=runtime_environment(directory),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = json.loads((directory/"pdd_metadata.json").read_text())
        if metadata["inference_nfe"] != 8 or metadata["code_revision"] != PDD_REVISION:
            raise ValueError("PDD worker contract mismatch")
        return metadata
