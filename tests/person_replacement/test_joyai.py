import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.person_replacement import joyai
from r2v_data_v2.person_replacement.timeline import (
    VideoTimeline,
    validate_matching_timeline,
)
from tools.person_replacement import joyai_offline_worker as worker


@pytest.mark.parametrize("fps", [25.0, 24000 / 1001])
def test_long_timeline_requires_all_137_frames_and_source_fps(fps):
    source = VideoTimeline(137, fps, 1920, 1080, 137 / fps)
    validate_matching_timeline(source, replace(source, width=1248, height=720))
    with pytest.raises(ValueError, match="frame count"):
        validate_matching_timeline(source, replace(source, frame_count=81))
    with pytest.raises(ValueError, match="fps"):
        validate_matching_timeline(source, replace(source, fps=16))
    with pytest.raises(ValueError, match="duration"):
        validate_matching_timeline(source, replace(source, duration_seconds=source.duration_seconds + 1))


def test_joyai_prompt_preserves_performance_not_original_appearance():
    prompt = joyai.build_joyai_prompt(
        "the man near the center with both hands in his trouser pockets",
        "a middle-aged woman with shoulder-length brown hair wearing a cream jacket",
    )
    assert prompt.startswith("Replace the man")
    assert "cream jacket" in prompt
    assert "original pose and motion" in prompt and "hand placement" in prompt
    assert "entire original background unchanged" in prompt
    assert "preserve original clothing" not in prompt.lower()


@pytest.fixture
def assets(tmp_path, monkeypatch):
    code = tmp_path / "official"
    code.mkdir()
    for name in joyai.REQUIRED_CODE_FILES:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    subprocess.run(["git", "init", "-q", str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "add", "."], check=True)
    subprocess.run(["git", "-C", str(code), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(joyai, "UPSTREAM_COMMIT", commit)
    model, mimo = tmp_path / "models", tmp_path / "MiMo-VL-7B-RL-2508"
    for root, names in ((model, ["dit/joyai_video_edit_dit_0811.pth", "vae/config.json",
                                 "vae/diffusion_pytorch_model.safetensors"]),
                        (mimo, ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                                "preprocessor_config.json", "model.safetensors"])):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    return joyai.JoyAIAdapter(code, Path(sys.executable), model, mimo)


def test_validate_resolves_0811_vae_and_mimo_without_fallback(assets):
    assets.validate()
    assert assets.dit_ckpt.name == "joyai_video_edit_dit_0811.pth"
    assert assets.vae_ckpt == assets.model_root / "vae"
    assets.dit_ckpt.rename(assets.dit_ckpt.with_name("joyai_video_edit_dit_0804.pth"))
    with pytest.raises(FileNotFoundError, match="0811"):
        assets.validate()


def test_validate_rejects_missing_encoder_shard_and_wrong_code_commit(assets, monkeypatch):
    (assets.text_encoder_ckpt / "model.safetensors").unlink()
    (assets.text_encoder_ckpt / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"weight": "model-00001-of-00002.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="00001"):
        assets.validate()
    monkeypatch.setattr(joyai, "UPSTREAM_COMMIT", "0" * 40)
    with pytest.raises(ValueError, match="commit"):
        assets.validate()


def test_upstream_git_validation_cannot_refresh_read_only_index(assets, monkeypatch):
    check_output = subprocess.check_output

    def read_only_git(*args, **kwargs):
        assert kwargs.get("env", {}).get("GIT_OPTIONAL_LOCKS") == "0"
        return check_output(*args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", read_only_git)
    assets.validate()


@pytest.mark.parametrize("count", [137, 138])
def test_stream_all_frames_flush_padding_and_keep_order(count):
    events, written = [], []

    class Session:
        def __init__(self):
            self.pending = []
            self.queue = []

        def push_frame(self, frame, frame_meta):
            self.pending.append((frame, frame_meta))
            events.append(frame_meta["seq"])
            if len(events) == 1 or len(self.pending) == 8:
                self.emit()
            return []

        def emit(self, valid=None):
            pairs, self.pending = self.pending, []
            self.queue.append(SimpleNamespace(jpegs=[p[0] for p in pairs],
                                               source_metas=[p[1] for p in pairs], valid_count=valid))

        def inflight_chunks(self):
            return len(self.queue)

        def wait_async_result(self, timeout):
            return self.queue.pop(0)

        def flush_pending(self):
            events.append("flush")
            if self.pending:
                valid = len(self.pending)
                self.pending += [self.pending[-1]] * (8 - valid)
                self.emit(valid)

        def close(self):
            events.append("close")

    class Runtime:
        def create_v2v_session(self, prompt, *, settings, ref_image):
            assert ref_image is None
            assert settings.num_inference_steps == 2
            return Session()

    settings = SimpleNamespace(num_inference_steps=2)
    frames = (np.full((2, 2, 3), i, dtype=np.uint8) for i in range(count))
    worker.stream_frames(Runtime(), settings, "replace", frames, count, 25, written.append)
    assert events == list(range(1, count + 1)) + ["flush", "close"]
    assert [int(f[0, 0, 0]) for f in written] == list(range(count))


def make_video(path, count=137, fps=25):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((24, 32, 3), i % 240, dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize("fps", [25.0, 24000 / 1001])
def test_real_media_writer_retains_137_frames_source_fps(tmp_path, fps):
    from r2v_data_v2.person_replacement.timeline import inspect_video_timeline
    source = tmp_path / "source.mp4"
    make_video(source, fps=fps)
    timeline = inspect_video_timeline(source)
    assert timeline.frame_count == 137
    assert timeline.duration_seconds > 5
    output = tmp_path / "out.mp4"
    with worker.video_writer(output, timeline.fps, 32, 24) as write:
        for frame in worker.decode_frames(source):
            write(frame)
    validate_matching_timeline(timeline, inspect_video_timeline(output))


@pytest.mark.parametrize("valid", [True, False])
def test_manifest_after_timeline_validation_and_qwen_release(tmp_path, monkeypatch, valid):
    source = tmp_path / "source.mp4"
    make_video(source)
    output = tmp_path / "case"
    events = []
    qwen = SimpleNamespace(model_path=tmp_path,
                           describe=lambda v: events.append("describe") or "source",
                           invent=lambda d: events.append("invent") or "appearance",
                           close=lambda: events.append("release qwen"))

    class Adapter:
        def validate(self):
            pass

        def provenance(self):
            return {"backend": "joyai_video_edit"}

        def run(self, source, output, prompt_path, timeline, seed):
            events.append("joyai")
            assert timeline.frame_count == 137
            make_video(output, count=137 if valid else 81)

    cases = [{"case_id": "one", "target_video_path": str(source), "output_dir": str(output)}]
    if valid:
        joyai.run_joyai_cases(cases, qwen=qwen, adapter=Adapter(), seed=42)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["source_frame_count"] == manifest["output_frame_count"] == 137
        assert manifest["target_video_path"] == str(source)
    else:
        with pytest.raises(ValueError, match="frame count"):
            joyai.run_joyai_cases(cases, qwen=qwen, adapter=Adapter(), seed=42)
        assert not (output / "manifest.json").exists()
    assert events == ["describe", "invent", "release qwen", "joyai"]


def test_joyai_dry_run_is_read_only_without_model_paths(tmp_path, monkeypatch, capsys):
    from tools.person_replacement import run_joyai_pilot as cli
    clips = tmp_path / "clips"
    clips.mkdir()
    make_video(clips / "clip.mp4")
    source = tmp_path / "collected_face_1.jsonl"
    source.write_text('{"video_path":"clip.mp4","source_video_path":"/not/input/full.mp4"}\n')
    monkeypatch.setattr(joyai.JoyAIAdapter, "validate", lambda self: pytest.fail("model validation"))
    assert cli.main(["--input-jsonl", str(source), "--clips-root", str(clips),
                     "--output-root", str(tmp_path / "out"), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cases"][0]["timeline"]["frame_count"] == 137
    assert not (tmp_path / "out").exists()


def test_cli_checkpoint_defaults_and_environment_overrides(monkeypatch):
    from tools.person_replacement.run_joyai_pilot import parse_args
    for name in ("JOYAI_MODEL_ROOT", "JOYAI_TEXT_ENCODER_CKPT"):
        monkeypatch.delenv(name, raising=False)
    args = parse_args(["--clips-root", "/clips", "--dry-run"])
    assert str(args.joyai_model_root) == "/mnt/workspace/public/pretrained/jdopensource/JoyAI-Video-Edit"
    assert str(args.joyai_text_encoder_ckpt) == "/mnt/workspace/public/pretrained/MiMo/MiMo-VL-7B-RL-2508"
    monkeypatch.setenv("JOYAI_MODEL_ROOT", "/explicit/joyai")
    monkeypatch.setenv("JOYAI_TEXT_ENCODER_CKPT", "/explicit/mimo")
    args = parse_args(["--clips-root", "/clips", "--dry-run"])
    assert str(args.joyai_model_root) == "/explicit/joyai"
    assert str(args.joyai_text_encoder_ckpt) == "/explicit/mimo"


def test_adapter_external_python_single_request_and_local_caches(assets, tmp_path):
    source_dir = tmp_path / "clips"
    source_dir.mkdir()
    source = source_dir / "source.mp4"
    source.write_bytes(b"read-only source fixture")
    output_dir = tmp_path / "case"
    output_dir.mkdir()
    prompt = output_dir / "joyai_prompt.txt"
    prompt.write_text("replace person")
    fake_python = tmp_path / "external-python"
    fake_python.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
assert sys.argv[1].endswith('joyai_offline_worker.py')
assert sys.argv[2] == '--request'
r = json.loads(Path(sys.argv[3]).read_text())
assert r['timeline']['frame_count'] == 137
assert r['timeline']['fps'] == 25
assert Path(r['prompt_path']).read_text() == 'replace person'
for key in ('HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH'):
    assert Path(os.environ[key]).is_relative_to(Path(r['output']).parent)
assert os.environ['HF_HUB_OFFLINE'] == '1'
assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
Path(r['output']).write_bytes(b'fake generated')
print('single external case')
''')
    fake_python.chmod(0o755)
    assets.python = fake_python
    assets.run(source, output_dir / "replacement.mp4", prompt, VideoTimeline(137, 25, 32, 24, 5.48), 42)
    assert (output_dir / "replacement.mp4").read_bytes() == b"fake generated"
    assert (output_dir / "joyai.log").read_text().strip() == "single external case"
    assert source.read_bytes() == b"read-only source fixture"


def test_official_bucket_choice_keeps_orientation_and_nearest_ratio():
    buckets = [(1, 1, 1, 960, 960), (1, 1, 1, 704, 1248), (1, 1, 1, 1248, 704)]
    assert worker.choose_bucket(1920, 1080, buckets, 32) == (704, 1248)
    assert worker.choose_bucket(1080, 1920, buckets, 32) == (1248, 704)


def test_worker_entrypoint_uses_official_loading_and_preserves_timeline(tmp_path, monkeypatch):
    from dataclasses import asdict

    from r2v_data_v2.person_replacement.timeline import inspect_video_timeline

    source, output = tmp_path / "source.mp4", tmp_path / "replacement.mp4"
    make_video(source)
    source_bytes = source.read_bytes()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Replace the person.")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "source": str(source), "output": str(output), "prompt_path": str(prompt),
        "timeline": asdict(inspect_video_timeline(source)), "seed": 19,
        "joyai_code_root": str(tmp_path / "code"), "joyai_dit_ckpt": "dit0811",
        "joyai_vae_ckpt": "vae", "joyai_text_encoder_ckpt": "mimo-rl",
    }))
    events = []

    class Session:
        def push_frame(self, frame, frame_meta):
            return [SimpleNamespace(jpegs=[np.asarray(frame)], source_metas=[frame_meta], valid_count=None)]

        def inflight_chunks(self):
            return 0

        def flush_pending(self):
            events.append("flush")

        def close(self):
            events.append("close")

    class Runtime:
        pipeline = SimpleNamespace(vae=SimpleNamespace(stem=SimpleNamespace(stride=1)))

        @classmethod
        def load(cls, dit, **kwargs):
            assert dit == "dit0811"
            assert kwargs == {"vae_ckpt": "vae", "text_encoder_ckpt": "mimo-rl", "seed": 19}
            return cls()

        def create_v2v_session(self, prompt, *, settings, ref_image):
            assert self.lossless_output is True
            assert prompt == "Replace the person." and ref_image is None
            assert (settings.num_inference_steps, settings.seed) == (2, 19)
            return Session()

    monkeypatch.setitem(sys.modules, "xvideo.config", SimpleNamespace(
        generate_video_image_bucket=lambda **kwargs: [(1, 1, 1, 24, 32)]))
    monkeypatch.setitem(sys.modules, "xvideo.serving.joyomni_streaming", SimpleNamespace(
        JoyOmniRuntime=Runtime, StreamingSettings=SimpleNamespace))
    assert worker.main(["--request", str(request)]) == 0
    assert inspect_video_timeline(output).frame_count == 137
    assert source.read_bytes() == source_bytes
    assert events == ["flush", "close"]
