import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


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


def test_pdd_is_applied_and_validated_before_sharding_without_offload(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pdd_distributed as module

    events = []
    transformer = SimpleNamespace(_pdd_step_arm=SimpleNamespace(nfe=8,block_size=4),
                                  proj_out=SimpleNamespace(num_steps=32),audio_proj_out=SimpleNamespace(num_steps=32))
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
    monkeypatch.setitem(sys.modules,"diffusers",SimpleNamespace(__version__="0.40.0",
        ModularPipeline=SimpleNamespace(from_pretrained=construct)))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules,"minimax_h3_pdd",SimpleNamespace(apply_pdd_lora=apply))
    monkeypatch.setattr(module,"validate_pdd_weights",lambda _:events.append("header"))
    monkeypatch.setattr(module,"validate_loaded_pdd_weights",lambda *a:events.append("validated"))
    monkeypatch.setattr(module,"shard_models",lambda *a:events.append("shard"))
    _, metadata = module.load_pipeline(tmp_path,tmp_path/"pdd","mesh","device")
    assert events == ["header","construct","load","pdd","validated","shard"]
    assert metadata["model_load_count"] == metadata["pdd_apply_count"] == 1
    assert metadata["components_manager_cpu_offload"] is False


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
