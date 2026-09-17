import json
import sys
from pathlib import Path

import pytest

from r2v_data_v2.person_replacement import bernini_r_1p3b as renderer
from r2v_data_v2.person_replacement.bernini import plan_bernini_timeline, serialize_case
from r2v_data_v2.person_replacement.timeline import VideoTimeline
from tools.person_replacement import run_bernini_r_1p3b_pilot as cli


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    (code / "infer_multi_gpu.py").write_text("")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "bernini_renderer", "skip_transformer_2": True,
    }))
    (model / "transformer").mkdir()
    (model / "transformer/config.json").write_text(json.dumps({
        "num_attention_heads": 12, "attention_head_dim": 128, "num_layers": 30,
    }))
    monkeypatch.setattr(renderer.subprocess, "check_output", lambda *a, **k: renderer.UPSTREAM_COMMIT)
    return renderer.BerniniR1p3BAdapter(code, model)


def test_validation_accepts_renderer(adapter):
    adapter.validate()


@pytest.mark.parametrize("patch", [
    {"model_type": "bernini"}, {"skip_transformer_2": False}, {"skip_transformer_2": "true"},
])
def test_validation_rejects_other_models(adapter, patch):
    path = adapter.config / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **patch}))
    with pytest.raises(ValueError):
        adapter.validate()


def test_validation_rejects_14b_transformer(adapter):
    (adapter.config / "transformer/config.json").write_text(json.dumps({
        "num_attention_heads": 40, "attention_head_dim": 128, "num_layers": 40,
    }))
    with pytest.raises(ValueError, match="1.3B"):
        adapter.validate()


def test_wrong_upstream_revision(adapter, monkeypatch):
    monkeypatch.setattr(renderer.subprocess, "check_output", lambda *a, **k: "wrong")
    with pytest.raises(ValueError, match="commit"):
        adapter.validate()


@pytest.mark.parametrize("override", [False, True])
def test_command_and_unchanged_case(adapter, tmp_path, monkeypatch, override):
    monkeypatch.delenv("NPROC_PER_NODE", raising=False)
    monkeypatch.delenv("ULYSSES", raising=False)
    if override:
        monkeypatch.setenv("NPROC_PER_NODE", "2")
        monkeypatch.setenv("ULYSSES", "2")
    plan = plan_bernini_timeline(VideoTimeline(138, 25., 25, 1, 640, 480, 138/25))
    case = tmp_path / "case.json"
    payload = serialize_case(tmp_path / "input.mp4", tmp_path / "raw.mp4", "exact prompt")
    case.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    adapter.run(case, 42, plan=plan)
    (command,), kwargs = calls[0]
    assert command[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    for option, expected in {
        "--nproc-per-node": "2" if override else "4", "--ulysses": "2" if override else "4",
        "--config": str(adapter.config), "--case": str(case), "--guidance_mode": "v2v_apg",
        "--num_frames": "141", "--fps": "25", "--seed": "42", "--num_inference_steps": "40",
        "--height": "0", "--width": "0", "--flow_shift": "5.0", "--max_image_size": "848",
    }.items():
        assert command[command.index(option) + 1] == expected
    assert "--use_unipc" in command and "--use_src_tgt_id" in command
    for forbidden in ("--use_pe", "vae_txt_vit_wapg", "--vit_denoising_step", "--vit_txt_cfg",
                      "--vit_img_cfg", "--omega_tgt", "--planning_step"):
        assert forbidden not in command
    assert kwargs["check"] is True and kwargs["cwd"] == adapter.code_root
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert Path(kwargs["env"]["HF_HOME"]).is_relative_to(tmp_path / "runtime")
    assert json.loads(case.read_text()) == payload
    assert payload["task_type"] == "v2v"


def descriptions(tmp_path, video):
    path = tmp_path / "full_manifest.json"
    path.write_text(json.dumps({"target_video_path": str(video),
        "source_person_description": " exact source ", "replacement_person_description": "exact replacement"}))
    return path


def test_descriptions_exact_and_source_checked(tmp_path):
    video = tmp_path / "source.mp4"
    video.touch()
    manifest = descriptions(tmp_path, video)
    qwen = renderer.ManifestDescriptionsQwen(manifest, video)
    assert qwen.model_path == manifest.resolve()
    assert qwen.describe(video) == " exact source "
    assert qwen.invent(" exact source ") == "exact replacement"
    qwen.close()
    with pytest.raises(ValueError, match="source"):
        qwen.describe(tmp_path / "wrong.mp4")
    with pytest.raises(ValueError):
        qwen.invent("different")
    with pytest.raises(ValueError, match="source"):
        renderer.ManifestDescriptionsQwen(manifest, tmp_path / "wrong.mp4")


@pytest.mark.parametrize("count,fps,expected,pad", [(81, 16, 81, 0), (138, 25, 141, 3)])
def test_dry_run_reuses_timeline_without_loading_or_writing(tmp_path, monkeypatch, capsys, count, fps, expected, pad):
    video = tmp_path / "source.mp4"
    video.touch()
    manifest = descriptions(tmp_path, video)
    output = tmp_path / "out"
    monkeypatch.setattr(cli, "select_cases", lambda *a, **k: [
        {"target_video_path": str(video), "output_dir": str(output / "case")}
    ])
    monkeypatch.setattr(cli, "inspect_video_timeline", lambda p: VideoTimeline(count, fps, fps, 1, 640, 480, count/fps))
    monkeypatch.setattr(cli, "LocalQwen", lambda *a: pytest.fail("Qwen loaded"))
    monkeypatch.setattr(cli, "run_cases", lambda *a, **k: pytest.fail("generation launched"))
    assert cli.main(["--input-jsonl", str(tmp_path / "input.jsonl"), "--clips-root", str(tmp_path),
        "--output-root", str(output), "--bernini-r-config", str(tmp_path / "model"),
        "--descriptions-from", str(manifest), "--limit", "1", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["variant"] == "bernini-r-1.3b"
    assert result["cases"][0]["bernini_timeline"] == {
        "internal_frame_count": expected, "internal_fps": fps, "pad_frames": pad}
    assert not output.exists()


def test_reuse_rejects_multiple_cases_before_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "select_cases", lambda *a, **k: pytest.fail("selected cases"))
    with pytest.raises(SystemExit):
        cli.main(["--input-jsonl", "input.jsonl", "--clips-root", str(tmp_path),
                  "--output-root", str(tmp_path / "out"), "--bernini-r-config", "model",
                  "--descriptions-from", "manifest.json", "--limit", "2"])


def test_config_required(tmp_path, monkeypatch):
    monkeypatch.delenv("BERNINI_R_1P3B_CONFIG", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--input-jsonl", "input.jsonl", "--clips-root", str(tmp_path),
                  "--output-root", str(tmp_path / "out"), "--dry-run"])


@pytest.mark.parametrize("bad_output", [False, True])
def test_runner_uses_existing_pipeline_and_manifest_gate(adapter, tmp_path, monkeypatch, bad_output):
    from r2v_data_v2.person_replacement import bernini, pipeline

    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "source.mp4"
    video.write_bytes(b"original")
    manifest = descriptions(tmp_path, video)
    source_jsonl = tmp_path / "input.jsonl"
    source_jsonl.write_text(json.dumps({"video_path": video.name}) + "\n")
    output = tmp_path / "out"
    timeline = VideoTimeline(81, 16., 16, 1, 640, 480, 81/16)
    monkeypatch.setattr(pipeline, "inspect_video_timeline", lambda p: timeline)
    monkeypatch.setattr(bernini, "inspect_video_timeline", lambda p:
        VideoTimeline(80, 16., 16, 1, 640, 480, 5.) if bad_output else timeline)
    monkeypatch.setattr(pipeline, "extract_frame_zero", lambda v, p: p.write_bytes(b"frame0"))
    monkeypatch.setattr(cli, "LocalQwen", lambda *a: pytest.fail("must not load Qwen"))
    def generate(command, **kwargs):
        case = json.loads(Path(command[command.index("--case") + 1]).read_text())
        assert case["task_type"] == "v2v"
        assert case["video"] == str(video)
        assert case["prompt"] == pipeline.build_prompt(" exact source ", "exact replacement")
        Path(case["output"]).write_bytes(b"generated")
    monkeypatch.setattr(renderer.subprocess, "run", generate)
    args = ["--input-jsonl", str(source_jsonl), "--clips-root", str(clips),
            "--output-root", str(output), "--bernini-code-root", str(adapter.code_root),
            "--bernini-r-config", str(adapter.config), "--descriptions-from", str(manifest), "--limit", "1"]
    if bad_output:
        with pytest.raises(ValueError, match="Frame count mismatch"):
            cli.main(args)
        assert not list(output.rglob("manifest.json"))
    else:
        assert cli.main(args) == 0
        published = json.loads(next(output.rglob("manifest.json")).read_text())
        assert published["source_frame_count"] == published["output_frame_count"] == 81
        assert published["source_fps"] == published["output_fps"] == 16
        assert published["bernini_model_path"] == str(adapter.config)
        assert published["source_person_description"] == " exact source "
        assert published["replacement_person_description"] == "exact replacement"
        assert Path(published["input_video_path"]).read_bytes() == b"generated"
    assert video.read_bytes() == b"original"
