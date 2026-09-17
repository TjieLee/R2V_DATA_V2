import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.person_replacement import h3_lightx2v, h3_pdd
from r2v_data_v2.person_replacement.h3_ref2va import (
    LIGHTX2V_CHECKPOINT,
    PDD_CHECKPOINT,
    plan_h3_timeline,
    validate_checkpoint,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline


@pytest.mark.parametrize("name", ["fl2va.safetensors", "ref2v_4step.safetensors", "ref2v_comfyui.safetensors"])
@pytest.mark.parametrize("expected", [LIGHTX2V_CHECKPOINT, PDD_CHECKPOINT])
def test_wrong_checkpoint_rejected(tmp_path, name, expected):
    path = tmp_path / name
    path.write_bytes(b"weights")
    with pytest.raises(ValueError, match="exact"):
        validate_checkpoint(path, expected)


@pytest.mark.parametrize("fsdp", [False, True])
def test_lightx2v_original_video_job_and_exact_command(tmp_path, monkeypatch, fsdp):
    backend = h3_lightx2v.LightX2VBackend(sys.executable, tmp_path, tmp_path, tmp_path / LIGHTX2V_CHECKPOINT,
                                       fsdp2=fsdp, nproc=4)
    target = tmp_path / "case"
    target.mkdir()
    video = tmp_path / "source.mp4"
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raw = target / "lightx2v_raw"
        raw.mkdir()
        suffix = "_rank0000" if fsdp else ""
        (raw / f"0000_ref2va_seed42{suffix}.mp4").write_bytes(b"native bytes")
    monkeypatch.setattr(h3_lightx2v.subprocess, "run", run)
    plan = plan_h3_timeline(VideoTimeline(138,25,25,1,1920,1080,5.52))
    metadata = backend.run(video, "same prompt", plan, "16:9", target, 42)
    job = json.loads((target / "job.json").read_text())["examples"][0]
    assert job["references"] == [{"type":"video", "path":str(video)}]
    assert job["task"] == "ref2va" and job["prompt"] == "same prompt"
    cmd, kwargs = calls[0]
    for option, value in [("--inference-steps","8"), ("--video-shift","6"), ("--audio-shift","3"),
                          ("--lora-alpha","8"), ("--lora-scale","1"), ("--reference-resize-mode","match"),
                          ("--num-frames","141")]:
        assert cmd[cmd.index(option)+1] == value
    assert ("--fsdp2" in cmd) == fsdp
    if fsdp:
        assert cmd[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert Path(kwargs["env"]["HF_HOME"]).is_relative_to(target)
    assert (target / "raw.mp4").read_bytes() == b"native bytes"
    assert metadata["inference_nfe"] == 8


def test_pdd_official_head_loader_and_video_reference(tmp_path, monkeypatch):
    from tools.person_replacement.h3_pdd_worker import generate

    calls = []
    transformer = SimpleNamespace(proj_out=SimpleNamespace(num_steps=32),
        audio_proj_out=SimpleNamespace(num_steps=32))
    shapes = {"proj_out.weight":(32,4,4), "audio_proj_out.weight":(32,4,4),
              "layer.lora_down":(64,4), "layer.lora_up":(4,64)}
    transformer.named_parameters = lambda: [(name,SimpleNamespace(shape=shape)) for name,shape in shapes.items()]
    monkeypatch.setitem(sys.modules,"safetensors", SimpleNamespace(safe_open=lambda *a, **k:
        nullcontext(SimpleNamespace(keys=lambda:shapes,
            get_slice=lambda name:SimpleNamespace(get_shape=lambda:shapes[name])))))
    class Pipe:
        transformer_ref = transformer
        scheduler = SimpleNamespace(shift=12.)
        audio_scheduler = SimpleNamespace(shift=3.)
        def load_components(self, **kwargs):
            assert "workflow" not in kwargs
            assert kwargs["dtype"] == "bf16"
            assert kwargs["pretrained_model_name_or_path"] == str(tmp_path)
        def __call__(self, **kwargs):
            assert kwargs["references"] == ["whole video"]
            assert kwargs["num_inference_steps"] == 9
            assert kwargs["num_frames"] == 141
            return {"videos":["video"], "audio":["audio"], "sampling_rate":32000}
    class Manager:
        def enable_auto_cpu_offload(self, **kwargs):
            assert kwargs["device"] == "cuda"
    def from_pretrained(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs["workflow"] == "ref2va"
        assert isinstance(kwargs["components_manager"], Manager)
        return Pipe()
    def apply(model, path, video_shift, audio_shift):
        calls.append((model,path,video_shift,audio_shift))
        model._pdd_step_arm = SimpleNamespace(nfe=8, block_size=4)
        return 8
    monkeypatch.setitem(sys.modules,"torch", SimpleNamespace(bfloat16="bf16", Generator=lambda:
        SimpleNamespace(manual_seed=lambda seed: "generator")))
    monkeypatch.setitem(sys.modules,"diffusers", SimpleNamespace(ComponentsManager=Manager,
        ModularPipeline=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setitem(sys.modules,"diffusers.modular_pipelines.minimax_h3", SimpleNamespace(
        MiniMaxH3VideoReference=SimpleNamespace(from_file=lambda p: "whole video")))
    monkeypatch.setitem(sys.modules,"diffusers.utils.export_utils", SimpleNamespace(
        encode_video=lambda *a, **k: Path(k["output_path"]).write_bytes(b"raw")))
    monkeypatch.setitem(sys.modules,"minimax_h3_pdd", SimpleNamespace(apply_pdd_lora=apply))
    metadata = generate(model_root=tmp_path, checkpoint=tmp_path/PDD_CHECKPOINT,
        source=tmp_path/"original.mp4", prompt="shared", frames=141, width=1376, height=768,
        output=tmp_path/"raw.mp4", seed=42)
    assert calls == [(transformer,str(tmp_path/PDD_CHECKPOINT),12.,3.)]
    assert metadata["pdd_num_steps"] == 32 and metadata["pdd_block_size"] == 4
    assert metadata["inference_nfe"] == 8


def test_pdd_adapter_uses_own_worker_not_generic_lora(tmp_path):
    backend = h3_pdd.PDDBackend(sys.executable,tmp_path,tmp_path,tmp_path/PDD_CHECKPOINT)
    cmd = backend.command(tmp_path/"job.json")
    assert "h3_pdd_worker.py" in cmd[1]
    assert "--pdd-code-root" in cmd


def test_pdd_worker_rejects_ordinary_lora_even_with_correct_filename(tmp_path, monkeypatch):
    from tools.person_replacement.h3_pdd_worker import validate_pdd_weights

    class Header:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def keys(self):
            return ["layer.lora_A.weight", "layer.lora_B.weight"]
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Header()))
    with pytest.raises(ValueError, match="PDD heads"):
        validate_pdd_weights(tmp_path/PDD_CHECKPOINT)


@pytest.mark.parametrize("missing,wrong_shape", [(True,False), (False,True), (False,False)])
def test_complete_pdd_adapter_state_required(tmp_path, monkeypatch, missing, wrong_shape):
    from tools.person_replacement.h3_pdd_worker import validate_loaded_pdd_weights

    shapes = {"proj_out.weight":(32,4,4), "audio_proj_out.weight":(32,4,4),
              "layer.lora_down":(64,4), "layer.lora_up":(4,64)}
    model = SimpleNamespace(named_parameters=lambda: [(name,SimpleNamespace(shape=shape)) for name,shape in shapes.items()])
    stored = dict(shapes)
    if missing:
        del stored["layer.lora_up"]
    if wrong_shape:
        stored["proj_out.weight"] = (8,4,4)
    class Header:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def keys(self):
            return stored
        def get_slice(self, key):
            return SimpleNamespace(get_shape=lambda: stored[key])
    monkeypatch.setitem(sys.modules,"safetensors",SimpleNamespace(safe_open=lambda *a, **k: Header()))
    if missing or wrong_shape:
        with pytest.raises(ValueError, match="Incomplete/incompatible PDD"):
            validate_loaded_pdd_weights(tmp_path/PDD_CHECKPOINT, model)
    else:
        validate_loaded_pdd_weights(tmp_path/PDD_CHECKPOINT, model)
