import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("size,cp", [(2,1),(4,1),(8,1),(4,2),(4,4),(8,4),(8,8)])
def test_channel_group_mesh_validation_and_gather(size, cp, monkeypatch):
    from r2v_data_v2.person_replacement.h3_pdd_distributed import DistributedChannel

    events = []
    dist = SimpleNamespace(init_process_group=lambda *a,**k:events.append("init"))
    def gather(result, payload):
        assert len(result) == size
        result[:] = [payload] * size
    dist.all_gather_object = gather
    cuda = SimpleNamespace(is_available=lambda:True,device_count=lambda:size,set_device=lambda _:None)
    torch = SimpleNamespace(cuda=cuda,device=lambda *a:a,distributed=dist)
    def mesh(kind, shape, **kwargs):
        expected = (size,) if cp == 1 else (1,cp,size//cp)
        names = ("fsdp",) if cp == 1 else ("ring","ulysses","fsdp")
        assert kind == "cuda" and shape == expected and kwargs == {"mesh_dim_names":names}
        return shape
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setitem(sys.modules,"torch.distributed",dist)
    monkeypatch.setitem(sys.modules,"torch.distributed.device_mesh",SimpleNamespace(init_device_mesh=mesh))
    monkeypatch.setitem(sys.modules,"torch.distributed.fsdp",SimpleNamespace(fully_shard=None))
    monkeypatch.setenv("WORLD_SIZE",str(size))
    monkeypatch.setenv("RANK","0")
    monkeypatch.setenv("LOCAL_RANK","0")
    channel = DistributedChannel(group_size=size,ulysses_degree=cp)
    assert channel.mesh == ((size,) if cp == 1 else (1,cp,size//cp))
    assert channel.gather("ready") == ["ready"] * size
    assert events == ["init"]
    monkeypatch.setenv("WORLD_SIZE","1")
    with pytest.raises(ValueError,match="torchrun ranks"):
        DistributedChannel(group_size=size)
    monkeypatch.setenv("WORLD_SIZE",str(size))
    cuda.device_count = lambda:1
    with pytest.raises(RuntimeError,match="visible CUDA"):
        DistributedChannel(group_size=size)
    assert events == ["init"]


@pytest.mark.parametrize("size,cp", [(4,3),(4,8),(3,2)])
def test_invalid_parallel_layout(size, cp):
    from r2v_data_v2.person_replacement.h3_pdd_distributed import parallel_layout

    with pytest.raises(ValueError):
        parallel_layout(size,cp)


def test_worker_imports_do_not_require_parent_manifest_dependencies():
    script = '''
import importlib.abc
import sys
class BlockParentDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "r2v_data_v2.manifest" or fullname.split(".")[0] == "ijson":
            raise ModuleNotFoundError(f"Parent-only dependency imported: {fullname}")
sys.meta_path.insert(0, BlockParentDependencies())
from r2v_data_v2.person_replacement.h3_pair_generation import generate_loop
from r2v_data_v2.person_replacement.h3_pdd_distributed import *
import tools.person_replacement.h3_pdd_fsdp_worker
assert "r2v_data_v2.manifest" not in sys.modules
assert "ijson" not in sys.modules
print("H3 worker imports OK")
'''
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "H3 worker imports OK" in result.stdout


@pytest.mark.parametrize("cp", [1,2])
def test_pdd_is_applied_and_validated_before_sharding_without_offload(tmp_path, monkeypatch, cp):
    from r2v_data_v2.person_replacement import h3_pdd_distributed as module

    events = []
    transformer = SimpleNamespace(_pdd_step_arm=SimpleNamespace(nfe=8,block_size=4),
                                  proj_out=SimpleNamespace(num_steps=32),audio_proj_out=SimpleNamespace(num_steps=32))
    arm = transformer._pdd_step_arm
    def enable(**kwargs):
        assert set(kwargs) == {"config"}  # built-in cp_plan, no custom plan/attention rewrite
        assert vars(kwargs["config"]) == {"ring_degree":1,"ulysses_degree":2,"ulysses_anything":True,"mesh":mesh}
        assert events[-1] == "validated" and transformer._pdd_step_arm is arm
        events.append("cp")
    transformer.enable_parallelism = enable
    mesh = {"fsdp":"fsdp_submesh"}
    class Pipe:
        transformer_ref = transformer
        scheduler = SimpleNamespace(shift=12.)
        audio_scheduler = SimpleNamespace(shift=3.)
        def load_components(self, **kwargs):
            assert "workflow" not in kwargs
            assert kwargs["dtype"] == "bf16"
            events.append("load")
    def construct(path, **kwargs):
        assert kwargs == {"workflow":"ref2va"}
        events.append("construct")
        return Pipe()
    def apply(*args):
        assert args == (transformer,str(tmp_path/"pdd"),12.,3.)
        events.append("pdd")
        return 8
    monkeypatch.setitem(sys.modules,"diffusers",SimpleNamespace(__version__="0.40.0",ContextParallelConfig=SimpleNamespace,
        ModularPipeline=SimpleNamespace(from_pretrained=construct)))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(bfloat16="bf16",cuda=SimpleNamespace(synchronize=lambda _:None)))
    monkeypatch.setitem(sys.modules,"minimax_h3_pdd",SimpleNamespace(apply_pdd_lora=apply))
    monkeypatch.setattr(module,"validate_pdd_weights",lambda _:events.append("header"))
    monkeypatch.setattr(module,"validate_loaded_pdd_weights",lambda *a:events.append("validated"))
    def shard(pipe, received_mesh, device):
        assert received_mesh == (mesh if cp == 1 else "fsdp_submesh")
        assert pipe.transformer_ref._pdd_step_arm is arm
        events.append("shard")
    monkeypatch.setattr(module,"shard_models",shard)
    _, metadata = module.load_pipeline(tmp_path,tmp_path/"pdd",mesh,"device",group_size=4,ulysses_degree=cp)
    assert events == ["header","construct","load","pdd","validated"] + (["cp"] if cp > 1 else []) + ["shard"]
    assert metadata["model_load_count"] == metadata["pdd_apply_count"] == 1
    assert metadata["parallel_setup_count"] == 1
    assert metadata["fsdp_degree"] == 4//cp
    for key in ("model_load_wall_seconds","pdd_apply_wall_seconds","parallel_setup_wall_seconds"):
        assert metadata[key] >= 0
    assert metadata["components_manager_cpu_offload"] is False


@pytest.mark.parametrize("cp", [2,4])
def test_hybrid_conditioner_submesh_and_pure_cp_no_fsdp(cp, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pdd_distributed as module

    events = []
    class Node:
        def __init__(self,name):
            self.name = name
        def requires_grad_(self, value):
            return self
        def eval(self):
            return self
        def to(self, device):
            events.append(("move",self.name,device))
            return self
        def register_forward_hook(self, hook):
            self.hook = hook
    transformer, text, layer, vae = [Node(n) for n in ("transformer","conditioner","layer","vae")]
    transformer.enable_parallelism = lambda **kwargs:events.append(("cp",kwargs["config"].mesh))
    text.language_model = SimpleNamespace(layers=[layer])
    outer = Node("outer")
    outer.model = text
    pipe = SimpleNamespace(transformer_ref=transformer,text_encoder=outer,
                           components={"transformer":transformer,"text":outer,"vae":vae},_execution_device="device")
    mesh = {"fsdp":"submesh"}
    monkeypatch.setitem(sys.modules,"diffusers",SimpleNamespace(ContextParallelConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(nn=SimpleNamespace(Module=Node)))
    monkeypatch.setitem(sys.modules,"minimax_h3_pdd",SimpleNamespace(LoRALinear=Node))
    def shard(node, **kwargs):
        assert kwargs["mesh"] == "submesh"
        events.append(("fsdp",node.name))
    monkeypatch.setitem(sys.modules,"torch.distributed.fsdp",SimpleNamespace(fully_shard=shard))
    def transformer_shard(node, received_mesh, *args):
        assert node is transformer and received_mesh == "submesh"
        events.append(("fsdp","transformer"))
    monkeypatch.setattr(module,"shard_transformer",transformer_shard)
    module.setup_parallelism(pipe,mesh,"device",module.parallel_layout(4,cp))
    assert events[0] == ("cp",mesh)
    if cp == 2:
        assert events[1:] == [("fsdp","layer"),("fsdp","conditioner"),("fsdp","transformer"),("move","vae","device")]
    else:
        assert events[1:] == [("move","transformer","device"),("move","conditioner","device"),("move","vae","device")]


def test_shard_pdd_adapter_base_and_fp32_parameters_separately():
    from r2v_data_v2.person_replacement.h3_pdd_distributed import shard_transformer

    class Node:
        def __init__(self,name):
            self.name = name
    class LoRA(Node):
        def __init__(self):
            super().__init__("fp32_adapter")
            self.base = Node("bf16_base")
    adapter = LoRA()
    refiner = Node("refiner")
    refiner.refiner_blocks = [Node("refiner_block")]
    transformer = Node("root")
    transformer.modules = lambda: [transformer,adapter,adapter.base]
    transformer.token_refiner = refiner
    transformer.transformer_blocks = [Node("transformer_block")]
    roots = ("proj_in","audio_proj_in","context_embedder","time_embedder","norm_out","proj_out","audio_proj_out")
    for name in roots:
        setattr(transformer,name,Node(name))
    calls = []
    def shard(model, **kwargs):
        assert kwargs == {"mesh":"mesh","reshard_after_forward":True}
        calls.append(model.name)
    shard_transformer(transformer,"mesh",shard,LoRA)
    assert calls[:2] == ["bf16_base","fp32_adapter"]
    assert calls[-1] == "root"
    assert set(roots) <= set(calls)
