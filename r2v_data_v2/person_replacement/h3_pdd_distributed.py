"""Isolated FSDP2 PDD construction; no model imports in the parent/Qwen process."""

import os
from datetime import timedelta

from tools.person_replacement.h3_pdd_worker import (
    validate_loaded_pdd_weights,
    validate_pdd_weights,
)

from .h3_ref2va import PDD_REVISION


def shard_transformer(transformer, mesh, fully_shard, lora_type):
    options = {"mesh":mesh,"reshard_after_forward":True}
    # Official PDD LoRALinear owns FP32 updates and a BF16 base. Separate
    # groups preserve both original dtypes without generic LoRA conversion.
    for module in tuple(transformer.modules()):
        if isinstance(module,lora_type):
            fully_shard(module.base,**options)
            fully_shard(module,**options)
    for block in transformer.token_refiner.refiner_blocks:
        fully_shard(block,**options)
    fully_shard(transformer.token_refiner,**options)
    for block in transformer.transformer_blocks:
        fully_shard(block,**options)
    for name in ("proj_in","audio_proj_in","context_embedder","time_embedder",
                 "norm_out","proj_out","audio_proj_out"):
        fully_shard(getattr(transformer,name),**options)
    fully_shard(transformer,**options)


def shard_models(pipe, mesh, device):
    import torch
    from minimax_h3_pdd import LoRALinear
    from torch.distributed.fsdp import fully_shard

    transformer = pipe.transformer_ref
    text = pipe.text_encoder.model  # H3 calls this inner model directly, not the outer LM head.
    text.requires_grad_(False).eval()
    for layer in text.language_model.layers:
        fully_shard(layer,mesh=mesh,reshard_after_forward=True)
    fully_shard(text,mesh=mesh,reshard_after_forward=True)
    # Release unsharded conditioner root before denoising. No manager hook owns it.
    def reshard_root(module, inputs, output):
        module.reshard()
        return output
    text.register_forward_hook(reshard_root)
    shard_transformer(transformer,mesh,fully_shard,LoRALinear)
    for component in pipe.components.values():
        if isinstance(component,torch.nn.Module) and component not in (transformer,pipe.text_encoder):
            component.requires_grad_(False).eval().to(device)
    if pipe._execution_device != device:
        raise RuntimeError("FSDP pipeline execution device mismatch; do not silently move sharded modules")


def load_pipeline(model_root, checkpoint, mesh, device):
    import diffusers
    import torch
    from diffusers import ModularPipeline
    from minimax_h3_pdd import apply_pdd_lora

    if diffusers.__version__ != "0.40.0":
        raise RuntimeError("Pair executor requires inspected Diffusers 0.40.0; no automatic dependency changes")
    validate_pdd_weights(checkpoint)
    pipe = ModularPipeline.from_pretrained(str(model_root),workflow="ref2va")
    pipe.load_components(dtype=torch.bfloat16,pretrained_model_name_or_path=str(model_root))
    shifts = float(pipe.scheduler.shift),float(pipe.audio_scheduler.shift)
    if shifts != (12.,3.):
        raise ValueError("PDD scheduler must retain 12/3 shifts")
    nfe = apply_pdd_lora(pipe.transformer_ref,str(checkpoint),*shifts)
    validate_loaded_pdd_weights(checkpoint,pipe.transformer_ref)
    arm = pipe.transformer_ref._pdd_step_arm
    if (nfe != 8 or arm.nfe != 8 or pipe.transformer_ref.proj_out.num_steps != 8*arm.block_size
            or pipe.transformer_ref.audio_proj_out.num_steps != 8*arm.block_size):
        raise ValueError("PDD head plan must be the official 8-NFE plan")
    shard_models(pipe,mesh,device)  # ONLY after adapter loading and weight validation
    return pipe, {"model_load_count":1,"pdd_apply_count":1,"code_revision":PDD_REVISION,
                  "pdd_checkpoint":str(checkpoint),"model_root":str(model_root),"inference_nfe":8,
                  "scheduler_grid_points":9,"video_shift":12,"audio_shift":3,
                  "fsdp_strategy":"fsdp2_dtype_uniform_pdd_v1","components_manager_cpu_offload":False,
                  "residency":"transformer_ref and text_encoder.model sharded; VAE/audio VAE replicated on local GPU",
                  "distributed_init_count":1}


class DistributedChannel:
    def __init__(self, timeout_seconds=600):
        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import (
            fully_shard,  # noqa: F401 -- preflight before loading weights
        )

        if int(os.environ["WORLD_SIZE"]) != 2:
            raise ValueError("Pair executor requires exactly two torchrun ranks")
        self.rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("Exactly two visible CUDA GPUs required")
        torch.cuda.set_device(local)
        self.device = torch.device("cuda",local)
        dist.init_process_group("nccl",device_id=self.device,timeout=timedelta(seconds=timeout_seconds))
        self.mesh = init_device_mesh("cuda",(2,),mesh_dim_names=("fsdp",))

    def broadcast(self, payload):
        import torch.distributed as dist
        objects = [payload]
        dist.broadcast_object_list(objects,src=0,device=self.device)
        return objects[0]

    def gather(self, payload):
        import torch.distributed as dist
        objects = [None,None]
        dist.all_gather_object(objects,payload)
        return objects

    def close(self):
        import torch.distributed as dist
        dist.destroy_process_group()


class PersistentPDD:
    def __init__(self, config, channel):
        self.device = channel.device
        self.pipeline,self.metadata = load_pipeline(config["h3_model_root"],config["pdd_lora"],
                                                   channel.mesh,channel.device)

    def prepare(self, job):
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference
        return MiniMaxH3VideoReference.from_file(job["source"])

    def infer(self, job, reference):
        import torch

        torch.cuda.reset_peak_memory_stats(self.device)
        # Official forward hook auto-cycles after 8 evaluations. Explicit reset
        # protects cross-job state; failed forward sessions are never reused.
        arm = self.pipeline.transformer_ref._pdd_step_arm
        arm.index = 0
        arm.arm(0)
        with torch.no_grad():
            return self.pipeline(prompt=job["prompt"],
                references=[reference],
                num_frames=job["frames"],width=job["width"],height=job["height"],
                num_inference_steps=9,generator=torch.Generator().manual_seed(job["seed"]),
                output_type="np",output=["videos","audio","sampling_rate"])

    def encode(self, result, path):
        from diffusers.utils.export_utils import encode_video
        encode_video(result["videos"][0],fps=24,output_path=str(path),
                     audio=result["audio"][0],audio_sample_rate=int(result["sampling_rate"]))

    def memory(self):
        import torch
        return {"peak_allocated_bytes":torch.cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes":torch.cuda.max_memory_reserved(self.device)}
