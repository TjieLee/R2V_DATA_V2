"""Run only inside the operator's isolated H3 environment; no generic LoRA loading."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.h3_ref2va import (
    PDD_CHECKPOINT,
    PDD_REVISION,
    require_local,
    validate_checkpoint,
    validate_code,
)


def validate_pdd_weights(checkpoint):
    # Official loader is strict=False for the base backbone. Reject a renamed
    # ordinary LoRA before it can leave the repeated PDD heads uninitialized.
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as header:
        keys = set(header.keys())
        if not {"proj_out.weight", "audio_proj_out.weight"} <= keys:
            raise ValueError("Checkpoint must include PDD heads, not an ordinary LoRA")
        for name in ("proj_out.weight", "audio_proj_out.weight"):
            if len(header.get_slice(name).get_shape()) != 3:
                raise ValueError("PDD heads must contain the step-specific output head bank")
        if not any(key.endswith(".lora_down") for key in keys) or not any(key.endswith(".lora_up") for key in keys):
            raise ValueError("PDD checkpoint lacks its native backbone updates")


def validate_loaded_pdd_weights(checkpoint, transformer):
    """Official strict=False loading must not leave any adapter/head parameter unloaded."""
    from safetensors import safe_open

    expected = {name:tuple(parameter.shape) for name,parameter in transformer.named_parameters()
                if name.endswith((".lora_down", ".lora_up"))
                or name.startswith(("proj_out.", "audio_proj_out."))}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as header:
        keys = set(header.keys())
        if not expected or not expected.keys() <= keys:
            raise ValueError(f"Incomplete/incompatible PDD state: missing {sorted(expected.keys()-keys)[:5]}")
        for name, shape in expected.items():
            if tuple(header.get_slice(name).get_shape()) != shape:
                raise ValueError(f"Incomplete/incompatible PDD tensor shape: {name}")


def generate(*, model_root, checkpoint, source, prompt, frames, width, height, output, seed):
    # Same construction / schedule / head injection as pinned predict_ref2v.py.
    # Only reference modality, requested geometry/timeline and output are supplied by our job.
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
    from diffusers.utils.export_utils import encode_video
    from minimax_h3_pdd import apply_pdd_lora

    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
    pipeline = ModularPipeline.from_pretrained(str(model_root), workflow="ref2va", components_manager=manager)
    pipeline.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(model_root))
    video_shift, audio_shift = float(pipeline.scheduler.shift), float(pipeline.audio_scheduler.shift)
    if (video_shift, audio_shift) != (12., 3.):
        raise ValueError("PDD base scheduler must retain official 12/3 shifts")
    nfe = apply_pdd_lora(pipeline.transformer_ref, str(checkpoint), video_shift, audio_shift)
    validate_loaded_pdd_weights(checkpoint, pipeline.transformer_ref)
    arm = pipeline.transformer_ref._pdd_step_arm
    steps = pipeline.transformer_ref.proj_out.num_steps
    if (nfe != 8 or arm.nfe != nfe or steps != nfe*arm.block_size
            or pipeline.transformer_ref.audio_proj_out.num_steps != steps):
        raise ValueError("PDD checkpoint/head plan is not the requested 8-NFE contract")
    result = pipeline(prompt=prompt, references=[MiniMaxH3VideoReference.from_file(str(source))],
        height=height, width=width, num_frames=frames, num_inference_steps=nfe+1,
        generator=torch.Generator().manual_seed(seed), output_type="np", output=["videos","audio","sampling_rate"])
    encode_video(result["videos"][0], fps=24, output_path=str(output),
                 audio=result["audio"][0], audio_sample_rate=int(result["sampling_rate"]))
    return {"code_revision":PDD_REVISION, "pdd_checkpoint":str(checkpoint),
            "inference_nfe":nfe, "pdd_num_steps":steps, "pdd_block_size":arm.block_size,
            "video_shift":video_shift, "audio_shift":audio_shift, "strength":1.0,
            "scheduler_grid_points":nfe+1, "execution_mode":"single_process_cpu_offload"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--pdd-code-root", type=Path, required=True)
    args = parser.parse_args(argv)
    validate_code(args.pdd_code_root, PDD_REVISION, ("minimax_h3_pdd.py", "predict_ref2v.py"))
    sys.path.insert(0, str(args.pdd_code_root.resolve()))
    job = json.loads(args.job.read_text())
    validate_checkpoint(require_local(job["checkpoint"], "PDD weights"), PDD_CHECKPOINT)
    validate_pdd_weights(job["checkpoint"])
    require_local(job["model_root"], "H3 base", directory=True)
    require_local(job["source"], "original processed source")
    output = Path(job["output"])
    if output.resolve().parent != args.job.resolve().parent or output.exists():
        raise ValueError("PDD output must be a new case-local file")
    metadata = generate(**job)
    (args.job.parent/"pdd_metadata.json").write_text(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
