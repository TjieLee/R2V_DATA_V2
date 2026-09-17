import io
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.person_replacement import viggle_boogu as pilot
from r2v_data_v2.person_replacement.timeline import VideoTimeline
from tools.person_replacement import run_viggle_boogu_pilot as cli


@pytest.mark.parametrize("w,h,expected", [(1920,1080,(832,480)), (1080,1920,(480,832)),
    (800,800,(640,640)), (1500,1000,(736,544)), (5000,500,(960,416))])
def test_official_canvas(w, h, expected):
    canvas = pilot.select_viggle_canvas(w, h)
    assert (canvas.width, canvas.height) == expected
    assert canvas.crop_fraction_estimate == pytest.approx(1-min((w/h)/(expected[0]/expected[1]),
                                                               (expected[0]/expected[1])/(w/h)))
    assert canvas.spatial_transform == "scale_to_cover_center_crop"


@pytest.mark.parametrize("count,render", [(96,124),(124,124),(125,141),(243,243)])
def test_timeline(count, render):
    plan = pilot.plan_viggle_timeline(VideoTimeline(count,24,24,1,832,480,count/24))
    assert (plan.keep_frames, plan.render_frames, plan.pad_frames) == (count,render,render-count)
    assert (render-5) % 17 == 0


@pytest.mark.parametrize("duration", [3.99, 243/24+0.01])
def test_duration_reject(duration):
    with pytest.raises(ValueError):
        pilot.plan_viggle_timeline(VideoTimeline(96,24,24,1,832,480,duration))


@pytest.fixture
def viggle(tmp_path, monkeypatch):
    root = tmp_path / "viggle"
    for directory in ("assets", "inference", "transformer", "lora"):
        (root / directory).mkdir(parents=True)
    (root / "assets/fixed_prompt.txt").write_bytes(b"frozen prompt")
    (root / "assets/fixed_embed_fwd_anyframe.pt").write_bytes(b"frozen embedding")
    (root / "inference/sample.py").write_bytes(b"official sample")
    monkeypatch.setattr(pilot, "FIXED_PROMPT_SHA256", pilot.sha256(root / "assets/fixed_prompt.txt"))
    monkeypatch.setattr(pilot, "SAMPLE_SHA256", pilot.sha256(root / "inference/sample.py"))
    h3 = tmp_path / "h3"
    h3.mkdir()
    (h3 / "modular_model_index.json").write_text("{}")
    for name in ("vae", "audio_vae", "scheduler", "audio_scheduler"):
        (h3 / name).mkdir()
    python = tmp_path / "python"
    python.touch()
    return pilot.ViggleAdapter(python, root, h3, "1")


def test_frozen_assets_fail_closed(viggle):
    viggle.validate()
    prompt = viggle.root / "assets/fixed_prompt.txt"
    prompt.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        viggle.validate()


def test_missing_embedding(viggle):
    (viggle.root / "assets/fixed_embed_fwd_anyframe.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        viggle.validate()


def test_python_symlink_preserves_virtualenv_entrypoint(viggle, tmp_path):
    entrypoint = tmp_path / "venv/bin/python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(viggle.python)
    adapter = pilot.ViggleAdapter(entrypoint, viggle.root, viggle.h3_root)
    assert adapter.python == entrypoint


@pytest.mark.parametrize("audio", [False, True])
def test_official_command(viggle, tmp_path, monkeypatch, audio):
    calls = []
    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **k: calls.append((a,k)))
    viggle.run(tmp_path, render_frames=141, source=tmp_path / "original.mp4", source_has_audio=audio)
    (cmd,), kwargs = calls[0]
    for option,value in {"--num-frames":"141", "--steps":"4", "--flow-shift":"3", "--seed":"42",
                         "--cond":str(tmp_path / "driving.mp4"), "--ref":str(tmp_path / "repainted_frame.png")}.items():
        assert cmd[cmd.index(option)+1] == value
    assert cmd[1] == str(viggle.root / "inference/sample.py")
    assert ("--audio" in cmd) == audio
    if audio:
        assert cmd[cmd.index("--audio")+1] == str(tmp_path / "original.mp4")
    assert not {"--prompt", "--height", "--width", "--short-edge", "--lora", "--transformer", "--embed"} & set(cmd)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"


def test_cover_and_repeat_tail_commands(tmp_path, monkeypatch):
    calls = []
    def ffmpeg(source, output, filters, count, **kwargs):
        calls.append((source,output,filters,count))
        output.write_bytes(b"video")
    monkeypatch.setattr(pilot, "ffmpeg_video", ffmpeg)
    canvas = pilot.select_viggle_canvas(1920,1080)
    plan = pilot.plan_viggle_timeline(VideoTimeline(125,24,24,1,1920,1080,125/24))
    monkeypatch.setattr(pilot, "validate_prepared", lambda *a: None)
    pilot.prepare_driving(tmp_path / "source.mp4", tmp_path, canvas, plan)
    assert "force_original_aspect_ratio=increase" in calls[0][2]
    assert "crop=832:480" in calls[0][2] and "fps=24" in calls[0][2]
    assert calls[0][3] == 125
    assert calls[1][2] == "tpad=stop_mode=clone:stop=16"
    assert calls[1][3] == 141


@pytest.mark.parametrize("bad_size", [False,True])
def test_repaint_reuses_existing_backend_exact_canvas(tmp_path, bad_size):
    Image.new("RGB", (832,480)).save(tmp_path / "driving_frame0.png")
    calls = []
    class Backend:
        def edit(self, **kwargs):
            calls.append(kwargs)
            data = io.BytesIO()
            Image.new("RGB", (640,640) if bad_size else (832,480)).save(data, format="PNG")
            return SimpleNamespace(png_bytes=data.getvalue(), worker_metadata={})
    canvas = pilot.select_viggle_canvas(1920,1080)
    if bad_size:
        with pytest.raises(ValueError, match="RGB PNG"):
            pilot.repaint(tmp_path, Backend(), "edit", canvas)
        assert not (tmp_path / "repainted_frame.png").exists()
    else:
        pilot.repaint(tmp_path, Backend(), "edit", canvas)
    assert calls[0]["width"] == 832 and calls[0]["height"] == 480
    assert calls[0]["thinking_enabled"] is False
    assert calls[0]["instruction_rewrite_enabled"] is False
    assert calls[0]["seed"] == 42


def test_description_mismatch(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"target_video_path":str(tmp_path / "other.mp4"),
        "source_person_description":"source", "replacement_person_description":"replacement"}))
    with pytest.raises(ValueError, match="source"):
        pilot.read_descriptions(manifest, tmp_path / "source.mp4")


@pytest.mark.parametrize("invalid_output", [False, True])
def test_dry_run(viggle, tmp_path, monkeypatch, capsys, invalid_output):
    output = tmp_path / "out"
    monkeypatch.setattr(cli, "select_cases", lambda *a, **k: [{"target_video_path":str(tmp_path / "source.mp4"),
                                                            "output_dir":str(output / "case")}])
    monkeypatch.setattr(cli, "validate_boogu", lambda config:
        config.validate() if invalid_output and config.temporary_root is not None else None)
    monkeypatch.setattr(cli, "inspect_video_timeline", lambda p: VideoTimeline(138,25,25,1,1920,1080,5.52))
    monkeypatch.setattr(cli, "has_audio", lambda p: False)
    monkeypatch.setattr(cli, "run_cases", lambda *a, **k: pytest.fail("models launched"))
    monkeypatch.setattr(cli, "LocalQwen", lambda *a: pytest.fail("Qwen loaded"))
    args = ["--input-jsonl","input.jsonl", "--clips-root",str(tmp_path), "--output-root",str(output),
        "--viggle-python",str(viggle.python), "--viggle-root",str(viggle.root),
        "--h3-model-root",str(viggle.h3_root), "--dry-run"]
    if invalid_output:
        with pytest.raises(ValueError, match="temporary_root must remain inside allowed_server_root"):
            cli.main(args)
        assert not output.exists()
        return
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"][0]["timeline"]["render_frames"] == 141
    assert result["custom_h3_prompt"] is False
    assert not output.exists()


@pytest.mark.parametrize("bad_repaint", [False, True])
def test_fake_pipeline_manifest_and_original_symlink(tmp_path, viggle, monkeypatch, bad_repaint):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    output = tmp_path / "out"
    events = []
    config = replace(pilot.BooguWorkerConfig(), allowed_server_root=tmp_path)
    class Backend:
        def __init__(self, config):
            pass
        def start(self, **kwargs):
            events.append("boogu_start")
        def close(self):
            events.append("boogu_close")
        def edit(self, **kwargs):
            data = io.BytesIO()
            image = Image.new("RGB", (16,16)) if bad_repaint else kwargs["source_rgb"]
            image.save(data, format="PNG")
            return SimpleNamespace(png_bytes=data.getvalue(), worker_metadata={})
    monkeypatch.setattr(pilot, "BooguSubprocessBackend", Backend)
    def prepare(src, directory, canvas, plan):
        for name in ("driving_unpadded.mp4", "driving.mp4"):
            (directory / name).write_bytes(b"driving")
    monkeypatch.setattr(pilot, "prepare_driving", prepare)
    monkeypatch.setattr(pilot, "extract_frame_zero", lambda src,dest:
        Image.new("RGB",(832,480)).save(dest))
    def generate(directory, **kwargs):
        assert not bad_repaint
        assert events[-1] == "boogu_close"
        (directory / "viggle_raw.mp4").write_bytes(b"raw")
    monkeypatch.setattr(viggle, "run", generate)
    monkeypatch.setattr(pilot, "validate_prepared", lambda *a: None)
    monkeypatch.setattr(pilot, "ffmpeg_video", lambda src,out,*a,**k: out.write_bytes(b"final"))
    timeline = VideoTimeline(138,25,25,1,1920,1080,5.52)
    qwen = SimpleNamespace(describe=lambda v:"source", invent=lambda s:"replacement",
                           close=lambda:events.append("qwen_close"))
    case = {"case_id":"00209", "target_video_path":str(source), "output_dir":str(output)}
    kwargs = {"sources":[timeline], "audio":[False], "qwen":qwen,
              "descriptions":None, "boogu_config":config, "viggle":viggle}
    if bad_repaint:
        with pytest.raises(ValueError, match="RGB PNG"):
            pilot.run_cases([case], **kwargs)
        assert events == ["qwen_close","boogu_start","boogu_close"]
        assert not (output / "manifest.json").exists()
        return
    paths = pilot.run_cases([case], **kwargs)
    record = json.loads(paths[0].read_text())
    assert record["custom_h3_prompt"] is False
    assert record["production_geometry_compatible"] is False
    assert record["production_timeline_compatible"] is False
    assert record["viggle_prompt_mode"] == "official_frozen_embedding"
    assert (output / "original.mp4").is_symlink()
    assert (output / "original.mp4").readlink() == source
    assert source.read_bytes() == b"original"
    assert events == ["qwen_close","boogu_start","boogu_close"]


def test_existing_boogu_worker_exact_bucket_and_dmd_args(tmp_path):
    from tools.run_v3_boogu_reference_edit_worker import _run_loaded_request

    source = tmp_path / "frame.png"
    Image.new("RGB", (832,480)).save(source)
    calls = []
    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB",(kwargs["width"],kwargs["height"]))])
    torch = SimpleNamespace(Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed))
    response = _run_loaded_request({"schema_version":1,"type":"edit","request_id":"pilot",
        "input_image_path":str(source), "output_image_path":str(tmp_path / "out.png"),
        "instruction":"Replace person", "thinking_enabled":False,"instruction_rewrite_enabled":False,
        "seed":42,"width":832,"height":480}, pipeline=pipe, torch_module=torch,
        device="cuda:0", seed=0, model_name=pilot.BOOGU_MODEL_NAME, model_revision=pilot.BOOGU_MODEL_REVISION)
    call = calls[0]
    assert call["num_inference_steps"] == 4
    assert call["text_guidance_scale"] == call["image_guidance_scale"] == 1.0
    assert call["use_dmd_student_inference"] is True
    assert call["use_rewrite_text_instruction"] is False
    assert call["generator"] == 42
    assert call["align_res"] is False
    assert response["returned_size"] == [832,480]


@pytest.mark.parametrize("frames,fps,w", [(140,24,832),(141,25,832),(141,24,800)])
def test_raw_output_validation_fails_closed(tmp_path, monkeypatch, frames, fps, w):
    monkeypatch.setattr(pilot, "inspect_video_timeline", lambda p: VideoTimeline(frames,fps,fps,1,w,480,frames/fps))
    with pytest.raises(ValueError, match="geometry or timeline"):
        pilot.validate_prepared(tmp_path / "raw.mp4", pilot.select_viggle_canvas(1920,1080), 141)


@pytest.mark.parametrize("present", [False,True])
def test_audio_probe(tmp_path, monkeypatch, present):
    def probe(command, **kwargs):
        assert command[command.index("-select_streams")+1] == "a:0"
        return SimpleNamespace(stdout=json.dumps({"streams":[{"codec_type":"audio"}] if present else []}))
    monkeypatch.setattr(pilot.subprocess, "run", probe)
    assert pilot.has_audio(tmp_path / "source.mp4") == present


def test_real_ffmpeg_prepare_and_trim_when_available(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools unavailable")
    source = tmp_path / "original.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=300x200:rate=25", "-frames:v", "138", "-c:v", "libx264", str(source)], check=True)
    original = source.read_bytes()
    timeline = pilot.inspect_video_timeline(source)
    canvas = pilot.select_viggle_canvas(300,200)
    plan = pilot.plan_viggle_timeline(timeline)
    assert (plan.keep_frames, plan.render_frames) == (132,141)
    pilot.prepare_driving(source, tmp_path, canvas, plan)
    pilot.extract_frame_zero(tmp_path / "driving_unpadded.mp4", tmp_path / "frame0.png")
    with Image.open(tmp_path / "frame0.png") as image:
        assert image.size == (736,544) and image.mode == "RGB"
    pilot.ffmpeg_video(tmp_path / "driving.mp4", tmp_path / "final.mp4", "trim=end_frame=132",132,keep_audio=True)
    pilot.validate_prepared(tmp_path / "final.mp4",canvas,132)
    assert source.read_bytes() == original
